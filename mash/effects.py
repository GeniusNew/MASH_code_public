from collections import Counter

from .hooks import HookError, HookRuntime


def model_result_kind(result):
    if isinstance(result, dict):
        content = result.get("content")
        additional = result.get("additional_kwargs")
        tool_calls = result.get("tool_calls") or (
            additional.get("tool_calls") if isinstance(additional, dict) else None)
    else:
        content, tool_calls = result, None
    if tool_calls:
        return "tool_calls"
    return ("empty_or_placeholder" if str(content or "").strip() in
            ("", "[Model returned empty response]") else "nonempty_text")


def text_values(value):
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from text_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from text_values(item)


def output_transition(before, after):
    if before == after:
        return "unchanged"
    if not before:
        return "added"
    return "modified" if after else "removed"


def hook_effects(pair, candidate, parent):
    if candidate.parent_id != parent.harness_id:
        raise ValueError("hook comparison parent differs from candidate parent")
    child = parent.apply(candidate.patches)
    try:
        runtimes = HookRuntime(parent), HookRuntime(child)
    except HookError as error:
        return {"replay_available": False, "replay_error": str(error)}
    targets = {patch.target: Counter() for patch in candidate.patches}
    changed_tasks = {target: set() for target in targets}
    cohorts = {name: [] for name in ("changed", "unchanged", "unverified")}
    examples = []
    for before, trajectory in zip(pair.parent, pair.candidate, strict=True):
        if (before.task.task_id != trajectory.task.task_id or before.harness_id != parent.harness_id
                or trajectory.harness_id != child.harness_id):
            raise ValueError("hook comparison trajectories differ from the paired harnesses")
        events = trajectory.execution.events
        by_id = {event.event_id: event for event in events}
        children = {}
        for event in events:
            for predecessor in event.parents:
                children.setdefault(predecessor, []).append(event)
        changed = unknown = False
        for event in events:
            if event.target not in targets:
                continue
            counts = targets[event.target]
            counts["calls"] += 1
            try:
                old, new = (runtime.call(event.target, event.inputs) for runtime in runtimes)
            except HookError:
                counts["replay_errors"] += 1
                unknown = True
                continue
            if new != event.outputs:
                counts["recorded_output_mismatches"] += 1
                unknown = True
                continue
            if new == old:
                counts["unchanged_outputs"] += 1
                continue
            changed = True
            counts["changed_outputs"] += 1
            transition = output_transition(old, new)
            counts[transition + "_outputs"] += 1
            changed_tasks[event.target].add(trajectory.task.task_id)
            text_fields = ("guidance", "retry_instruction", "regeneration_guidance", "findings",
                           "evidence_items", "missing_information", "synthesis_guidance",
                           "finance_snapshot", "tool_context", "subagent_context", "subagent_round_guidance")
            added = set(text_values({key: new[key] for key in text_fields if key in new})) - set(text_values(old))
            descendants = set()
            pending = [event.event_id]
            while pending:
                current = pending.pop()
                for item in children.get(current, ()):
                    if item.event_id not in descendants:
                        descendants.add(item.event_id)
                        pending.append(item.event_id)
            deliveries = []
            for event_id in sorted(descendants):
                consumer = by_id[event_id]
                if consumer.target != "model_input":
                    continue
                messages = tuple(text_values(consumer.inputs.get("messages", ())))
                present = [text for text in added if any(text in message for message in messages)]
                if not present:
                    continue
                responses = [item for item in children.get(event_id, ())
                             if item.target in ("after_worker_model", "after_lead_model")
                             and "result" in item.outputs]
                deliveries.append({"event_id": event_id,
                                   "phase": consumer.inputs.get("phase"),
                                   "matched_text_count": len(present),
                                   "immediate_results": [model_result_kind(item.outputs["result"])
                                                         for item in responses]})
            counts["changed_with_observed_text_delivery"] += bool(deliveries)
            counts["changed_without_observed_text_delivery"] += not bool(deliveries)
            counts["changed_without_added_text"] += not bool(added)
            if counts["changed_outputs"] <= 4:
                examples.append({"run_id": trajectory.run_id, "event_id": event.event_id,
                                 "task_id": trajectory.task.task_id, "target": event.target,
                                 "output_transition": transition,
                                 "phase": event.inputs.get("phase"), "model_inputs": deliveries})
        cohort = "changed" if changed else "unverified" if unknown else "unchanged"
        cohorts[cohort].append(trajectory.score - before.score)
    return {
        "replay_available": True,
        "targets": {target: {**dict(counts), "changed_task_count": len(changed_tasks[target])}
                    for target, counts in sorted(targets.items())},
        "paired_outcomes_by_activation": {
            name: {"pairs": len(values), "mean_delta": sum(values) / len(values) if values else None,
                   "improved": sum(value > 0 for value in values),
                   "tied": sum(value == 0 for value in values),
                   "regressed": sum(value < 0 for value in values)}
            for name, values in cohorts.items()},
        "changed_event_examples": examples,
        "delivery_measure": "Exact added text observed in recorded prepared model inputs; presence is not causal attribution. Missing delivery evidence is not proof of non-delivery.",
    }
