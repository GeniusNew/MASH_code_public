
from __future__ import annotations

from copy import deepcopy
from typing import Protocol

from .diagnosis import organize
from .evaluation import Evaluator
from .feedback import repair_feedback
from .effects import hook_effects
from .hooks import (HookError, HookRuntime, TARGETS, check_source,
                    candidate_boundary_errors, cross_group_case_errors, stage1)
from .models import (Candidate, DiagnosisBatch, Event, FailureGroup, FreeformBatch, FrozenHarness, Harness,
                     PairResult, SearchConfig, TaskSplit, Trajectory)


def task_outcomes(pair: PairResult, strata: dict[str, str | None] | None = None) -> list[dict]:

    return [{"task_id": p.task.task_id,
             "stratum": ((strata or {}).get(p.task.task_id) or next((p.task.payload[key] for key in
                          ("question_type", "category", "task_type", "domain")
                          if key in p.task.payload), None)),
             "parent_run": p.run_id, "candidate_run": c.run_id,
             "parent_score": p.score, "candidate_score": c.score,
             "delta": c.score - p.score}
            for p, c in zip(pair.parent, pair.candidate)]


def filtering_feedback(pair: PairResult, candidate: Candidate,
                       strata: dict[str, str | None] | None = None, *, parent: Harness | None = None) -> dict:
    outcomes = task_outcomes(pair, strata)
    edited = {patch.target for patch in candidate.patches}
    activity = {target: {"calls": 0, "nonempty_outputs": 0, "reached_terminal": 0,
                         "nonempty_reached_terminal": 0,
                         "output_nonempty": {}, "output_items": {}}
                for target in sorted(edited)}
    for trajectory in pair.candidate:
        events = trajectory.execution.events
        by_id = {event.event_id: event for event in events}
        ancestors = set()
        pending = [parent for event in events if event.target == "lead_answer" for parent in event.parents]
        while pending:
            event_id = pending.pop()
            if event_id in ancestors:
                continue
            ancestors.add(event_id)
            if event_id in by_id:
                pending.extend(by_id[event_id].parents)
        for event in events:
            if event.target not in activity:
                continue
            row = activity[event.target]
            row["calls"] += 1
            row["nonempty_outputs"] += bool(event.outputs)
            row["reached_terminal"] += event.event_id in ancestors
            row["nonempty_reached_terminal"] += bool(event.outputs) and event.event_id in ancestors
            for name, value in event.outputs.items():
                if not value:
                    continue
                row["output_nonempty"][name] = row["output_nonempty"].get(name, 0) + 1
                items = len(value) if isinstance(value, (list, tuple, dict)) else 1
                row["output_items"][name] = row["output_items"].get(name, 0) + items
    for row in activity.values():
        row["output_nonempty"] = dict(sorted(row["output_nonempty"].items()))
        row["output_items"] = dict(sorted(row["output_items"].items()))
    by_stratum = {}
    for outcome in outcomes:
        if outcome["stratum"] is not None:
            by_stratum.setdefault(str(outcome["stratum"]), []).append(outcome["delta"])
    effect_fields = ("plan_revision_calls", "handoff_regeneration_calls",
                     "model_result_repair_requests", "model_result_retry_requests",
                     "model_result_retry_instruction_chars",
                     "tool_transition_retry_requests", "tool_transition_switch_requests",
                     "tool_transition_stop_requests", "working_memory_model_calls",
                     "evidence_memory_items")
    native_effects = {field: 0 for field in effect_fields}
    revision_paths = {}
    orchestration_actions = {}
    for trajectory in pair.candidate:
        for event in trajectory.execution.events:
            if event.target != "native_execution_control":
                continue
            for field in effect_fields:
                native_effects[field] += int(event.outputs.get(field, 0) or 0)
            path = str(event.outputs.get("plan_revision_path", "not_requested"))
            revision_paths[path] = revision_paths.get(path, 0) + 1
            for decision in event.outputs.get("orchestration_decisions", []):
                action = str(decision.get("action", "native"))
                orchestration_actions[action] = orchestration_actions.get(action, 0) + 1
    return {
        **({"hook_effects": hook_effects(pair, candidate, parent)} if parent is not None else {}),
        "improved_count": sum(row["delta"] > 0 for row in outcomes),
        "tied_count": sum(row["delta"] == 0 for row in outcomes),
        "regressed_count": sum(row["delta"] < 0 for row in outcomes),
        "mean_delta_by_stratum": {key: sum(values) / len(values)
                                  for key, values in sorted(by_stratum.items())},
        "target_activity": activity,
        "native_effects": native_effects,
        "plan_revision_path_counts": dict(sorted(revision_paths.items())),
        "orchestration_action_counts": dict(sorted(orchestration_actions.items())),
        "parent_error_count": sum(trajectory.error is not None for trajectory in pair.parent),
        "candidate_error_count": sum(trajectory.error is not None for trajectory in pair.candidate),
        "empty_candidate_answer_count": sum(not trajectory.execution.answer.strip()
                                            for trajectory in pair.candidate),
    }


def behaviorally_equivalent(parent: Harness, left: Candidate, right: Candidate,
                            evidence: dict[tuple[str, str], Event]) -> bool:
    if left.parent_id != right.parent_id or left.parent_id != parent.harness_id:
        return False
    left_targets = {patch.target for patch in left.patches}
    right_targets = {patch.target for patch in right.patches}
    if left_targets != right_targets:
        return False
    events = {target: [(key, event) for key, event in evidence.items() if event.target == target]
              for target in left_targets}
    if not events or any(not values for values in events.values()):
        return False
    try:
        left_runtime = HookRuntime(parent.apply(left.patches))
        right_runtime = HookRuntime(parent.apply(right.patches))

        def effect(runtime, target, key, event):
            if target == "review_tool_result":
                result = runtime.call(target, event.inputs)
                if "decide_tool_transition" in left_targets:
                    return None
                return result.get("observation", event.inputs.get("observation"))
            if target != "decide_tool_transition" or "review_tool_result" not in left_targets:
                return runtime.call(target, event.inputs)
            context = deepcopy(event.inputs)
            reviewed = deepcopy(context.get("reviewed_result", {}))
            source = next((evidence.get((key[0], parent_id)) for parent_id in event.parents
                           if evidence.get((key[0], parent_id), None) is not None
                           and evidence[(key[0], parent_id)].target == "review_tool_result"), None)
            if source is not None:
                reviewed.update(runtime.call("review_tool_result", source.inputs))
            context["reviewed_result"] = reviewed
            decision = runtime.call(target, context)
            action = decision.get("action", "continue")
            if action in ("retry", "switch"):
                return {"action": action,
                        "tool_call": decision.get("tool_call", context.get("tool_call"))}
            return {"observation": reviewed.get("observation")}

        return all(effect(left_runtime, target, key, event) ==
                   effect(right_runtime, target, key, event)
                   for target, values in events.items() for key, event in values)
    except (HookError, ValueError):
        return False


class Optimizer(Protocol):
    def diagnose(self, parent: Harness, trajectories: tuple[Trajectory, ...],
                 history: tuple[dict, ...], round_number: int, *,
                 current_evaluation: dict | None = None) -> DiagnosisBatch | FreeformBatch: ...

    def generate(self, parent: Harness, failures: tuple[FailureGroup, ...],
                 history: tuple[dict, ...], round_number: int, *,
                 current_evaluation: dict | None = None) -> tuple[Candidate, ...]: ...

    def revise_batch(self, parent: Harness, failures: tuple[FailureGroup, ...],
                     history: tuple[dict, ...], round_number: int, feedback: dict, *,
                     current_evaluation: dict | None = None) -> tuple[Candidate, ...] | None: ...


class MASH:
    def __init__(self, evaluator: Evaluator, optimizer: Optimizer, split: TaskSplit,
                 config: SearchConfig):
        self.evaluator, self.optimizer = evaluator, optimizer
        self.split, self.config, self.store = split, config, evaluator.store
        if evaluator.seed != config.seed:
            raise ValueError("evaluator seed must match the recorded search configuration")
        missing = set((*split.diagnosis, *split.validation, *split.test)) - evaluator.tasks.keys()
        if missing:
            raise ValueError(f"partition task IDs not found: {sorted(missing)}")
        self.frozen: FrozenHarness | None = None
        self.editable_targets = tuple(k for k in evaluator.runner.exposed_targets
                                      if TARGETS[k].level in config.editable_levels)
        self.task_strata = {task_id: task.stratum for task_id, task in evaluator.tasks.items()}
        self.behavior_evidence = {}
        self.hook_contracts = dict(getattr(evaluator.runner, "hook_contracts", {}))
        if not set(self.hook_contracts) <= set(evaluator.runner.exposed_targets):
            raise ValueError("native hook contracts must belong to exposed targets")

    def optimize(self, initial: Harness = Harness()) -> FrozenHarness:
        if self.frozen is not None:
            raise ValueError("this search is already frozen")
        for hook in initial.hooks:
            if hook.target not in self.evaluator.runner.exposed_targets:
                raise ValueError(f"initial harness target not exposed: {hook.target}")
            check_source(hook.target, hook.source)
        parent = initial
        pending: tuple[Candidate, ...] = ()
        history: list[dict] = []
        selected: list[str] = []
        for t in range(1, self.config.rounds + 1):
            fixed_parent = parent
            record = {"round": t, "parent_before": fixed_parent.harness_id,
                      "normal_evaluations": [], "diagnoses": None, "filtering": [],
                      "adopted": None, "adopted_version_id": None, "generation_batches": []}
            trajectories: list[Trajectory] = []
            comparison_trajectories: list[Trajectory] = []
            if not pending:
                trajectories.extend(self.evaluator.collect(
                    fixed_parent, self.split.diagnosis,
                    phase="initialization" if t == 1 else "diagnosis", round_number=t,
                    repetitions=self.config.initialization_repetitions if t == 1 else 1))
            else:
                best_gain = 0.0
                winner = None
                for candidate in pending:
                    if candidate.parent_id != fixed_parent.harness_id:
                        raise ValueError("pending candidate belongs to a different parent")
                    harness = fixed_parent.apply(candidate.patches)
                    pair = self._pair(
                        fixed_parent, harness, self.split.diagnosis, phase="normal",
                        round_number=t, repetitions=self.config.normal_repetitions)

                    trajectories.extend(pair.candidate)
                    comparison_trajectories.extend(pair.parent)
                    record["normal_evaluations"].append({"candidate": candidate.candidate_id,
                        "version_id": candidate.submission_version_id,
                        "candidate_harness_id": harness.harness_id,
                        "parent_harness_id": fixed_parent.harness_id,
                        "complete": True, "delta": pair.delta,
                        "parent_runs": [p.run_id for p in pair.parent],
                        "candidate_runs": [c.run_id for c in pair.candidate],
                        "task_outcomes": task_outcomes(pair, self.task_strata),
                        "repair_feedback": repair_feedback(candidate, history[-1], pair, parent=fixed_parent),
                        "mechanism_feedback": filtering_feedback(pair, candidate, self.task_strata, parent=fixed_parent),
                        "errors": [{"run_id": tr.run_id, "task_id": tr.task.task_id,
                                    "side": side, "error": tr.error}
                                   for side, runs in (("parent", pair.parent), ("candidate", pair.candidate))
                                   for tr in runs if tr.error is not None]})
                    if pair.delta > best_gain:
                        best_gain, winner = pair.delta, candidate
                if winner is not None:
                    parent = fixed_parent.apply(winner.patches)
                    record["adopted"] = winner.candidate_id
                    record["adopted_version_id"] = winner.submission_version_id
                    selected.append(winner.candidate_id)
            record["parent_after"] = parent.harness_id
            self._remember_behavior_inputs((*trajectories, *comparison_trajectories))
            record["diagnosis_runs"] = [{"run_id": tr.run_id, "task_id": tr.task.task_id,
                                          "harness_id": tr.harness_id, "score": tr.score}
                                         for tr in trajectories]
            pending = ()
            if t != self.config.rounds:
                current_evaluation = {key: deepcopy(record[key]) for key in
                    ("round", "parent_before", "parent_after", "adopted", "normal_evaluations")}

                current_evaluation["comparison_trajectories"] = [tr.model_dump() for tr in comparison_trajectories]
                batch = self.optimizer.diagnose(parent, tuple(trajectories), tuple(history), t,
                                                current_evaluation=current_evaluation)
                failures = organize(batch, tuple(trajectories))
                record["diagnoses"] = batch.model_dump()
                record["repair_reviews"] = [review.model_dump() for review in batch.repair_reviews]
                current_evaluation["repair_reviews"] = deepcopy(record["repair_reviews"])
                record["failure_groups"] = [g.model_dump() for g in failures]
                candidates = self.optimizer.generate(parent, failures, tuple(history), t,
                                                     current_evaluation=current_evaluation)
                groups = {f.group_id: f for f in failures}
                retained = []
                submitted = set()
                def validity(candidate):
                    boundary = list(candidate_boundary_errors(
                        candidate, parent, self.editable_targets, failures, self.config.variant, submitted))
                    errors = list(boundary)
                    matches = []
                    if not boundary and 1 in self.config.filtering_stages:
                        errors += list(cross_group_case_errors(
                            candidate, tuple(groups.values()), self.behavior_evidence))
                        errors += list(stage1(candidate, parent, self.editable_targets,
                                             evidence=self.behavior_evidence, contracts=self.hook_contracts,
                                             require_frozen_plan=self.config.variant in ("F", "O")))
                        previous = [Candidate.model_validate(entry["candidate"])
                                    for old in history for entry in old.get("filtering", [])]
                        previous += [Candidate.model_validate(entry["candidate"])
                                     for entry in record["filtering"]]
                        if not errors:
                            counts = {patch.target: sum(event.target == patch.target
                                      for event in self.behavior_evidence.values())
                                      for patch in candidate.patches}
                            matches = [{"version_id": old.submission_version_id,
                                        "candidate_id": old.candidate_id,
                                        "observed_inputs_by_target": counts,
                                        "observed_input_count": sum(counts.values())}
                                       for old in previous if behaviorally_equivalent(
                                           parent, candidate, old, self.behavior_evidence)]
                    return boundary, errors, matches

                def submit(candidate):
                    if candidate.candidate_id in submitted:
                        raise ValueError("submitted version IDs must be unique within a loop")
                    boundary, errors, matches = validity(candidate)
                    submitted.add(candidate.candidate_id)


                    entry = {"candidate": candidate.model_dump(), "stage1": None,
                             "version_id": candidate.submission_version_id,
                             "stage2_delta": None, "stage3_delta": None, "retained": False,
                             "enabled_stages": list(self.config.filtering_stages), "rejected_at": None,
                             "batch_index": batch_index}
                    if 1 in self.config.filtering_stages:
                        entry["behavior_feedback"] = {"scope": "observed_inputs_only", "matches": matches}
                    record["filtering"].append(entry)
                    entries.append((candidate, entry, boundary, errors))
                for batch_index in range(4):
                    if batch_index:
                        revision = getattr(self.optimizer, "revise_batch", None)
                        if revision is None or not failures:
                            break
                        feedback = {
                            "revision_batch": batch_index,
                            "thresholds": {"stage2": self.config.theta_eff, "stage3": self.config.theta_gen},
                            "submitted_versions": deepcopy(record["filtering"]),
                            "observed_inputs": [deepcopy(row) for entry in record["filtering"]
                                                for row in entry.get("observed_inputs", ())],
                        }
                        candidates = revision(parent, failures, tuple(history), t, feedback,
                                              current_evaluation=current_evaluation)
                        if candidates is None:
                            break
                    candidates = tuple(self.store.submit_candidate(candidate, t) for candidate in candidates)
                    if len(candidates) > self.config.max_candidates:
                        raise ValueError("optimizer exceeded the candidate limit for a batch")
                    if len({c.candidate_id for c in candidates}) != len(candidates):
                        raise ValueError("candidate IDs must be unique within a generation batch")
                    batch_record = {"batch_index": batch_index,
                                    "candidate_ids": [candidate.candidate_id for candidate in candidates],
                                    "retained_ids": []}
                    record["generation_batches"].append(batch_record)
                    entries = []
                    for candidate in candidates:
                        submit(candidate)
                    for candidate, entry, boundary, errors in entries:
                        if 1 in self.config.filtering_stages:
                            entry["stage1"] = list(errors)
                        if errors:
                            entry["rejected_at"] = "boundary" if boundary else "stage1"
                            entry["boundary_errors"] = list(boundary)
                            if not boundary:
                                entry["observed_inputs"] = self._behavior_case_inputs(candidate)
                            continue
                        harness = parent.apply(candidate.patches)
                        if 2 in self.config.filtering_stages:
                            pair = self._pair(parent, harness, self.split.small, phase="stage2",
                                              round_number=t, repetitions=1)
                            entry["stage2_delta"] = pair.delta
                            entry["stage2_task_outcomes"] = task_outcomes(pair, self.task_strata)
                            entry["stage2_feedback"] = filtering_feedback(pair, candidate, self.task_strata, parent=parent)
                            self._remember_behavior_inputs((*pair.parent, *pair.candidate))
                            observer = getattr(self.optimizer, "observe_screening", None)
                            if observer is not None:
                                observer(pair)
                            if pair.delta < self.config.theta_eff:
                                entry["rejected_at"] = "stage2"
                                continue
                        if 3 in self.config.filtering_stages:
                            pair = self._pair(parent, harness, self.split.validation, phase="stage3",
                                              round_number=t, repetitions=1)
                            entry["stage3_delta"] = pair.delta
                            entry["stage3_task_outcomes"] = task_outcomes(pair, self.task_strata)
                            entry["stage3_feedback"] = filtering_feedback(pair, candidate, self.task_strata, parent=parent)
                            observer = getattr(self.optimizer, "observe_validation", None)
                            if observer is not None:
                                observer(pair)
                            if pair.delta < self.config.theta_gen:
                                entry["rejected_at"] = "stage3"
                                continue
                        entry["retained"] = True
                        retained.append(candidate)
                    batch_record["retained_ids"] = [candidate.candidate_id for candidate in retained]
                    if retained:
                        break
                pending = tuple(retained)
            record["preflight_rejections"] = self.store.preflight_rejections(t)
            history.append(record)
        self.evaluator.check_system()
        frozen = FrozenHarness(harness=parent, fixed_system_id=self.evaluator.fixed_system_id,
                               split=self.split, config=self.config, rounds_completed=t,
                               selected_candidates=tuple(selected))
        self.frozen = frozen
        return frozen

    def _pair(self, parent, harness, ids, **kwargs):
        return self.evaluator.pair(parent, harness, ids, **kwargs)

    def _remember_behavior_inputs(self, trajectories):

        self.behavior_evidence.update({(tr.run_id, event.event_id): event
            for tr in trajectories for event in tr.execution.events if event.target in TARGETS})

    def _behavior_case_inputs(self, candidate):

        inputs = []
        for change in candidate.changes:
            for case in change.behavior_cases:
                if case.source is None:
                    continue
                event = self.behavior_evidence.get((case.source.run_id, case.source.event_id))
                if event is not None and event.target == change.target:
                    inputs.append({"target": change.target, "source": case.source.model_dump(),
                                   "context": deepcopy(event.inputs)})
        return inputs
