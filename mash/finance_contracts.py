from __future__ import annotations

def string_list(value, field):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must contain strings")


def bounded_string_list(output, field, limit):
    if field not in output:
        return
    string_list(output[field], field)
    if len(output[field]) > limit:
        raise ValueError(f"{field} exceeds the native limit")


def bounded_text(output, field, limit):
    if field in output and len(output[field]) > limit:
        raise ValueError(f"{field} exceeds the native limit")


def slot_ids(context):
    contract = context.get("contract", {})
    if not isinstance(contract, dict):
        raise ValueError("contract must be an object")
    slots = contract.get("required_slots", [])
    if not isinstance(slots, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("slot_id", ""), str)
            for item in slots):
        raise ValueError("contract.required_slots must contain slot objects")
    return {item.get("slot_id", "") for item in slots}


def worker_ids(context):
    envelopes = context.get("envelopes", [])
    if not isinstance(envelopes, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("worker_id", ""), str)
            for item in envelopes):
        raise ValueError("envelopes must contain worker objects")
    return {item.get("worker_id", "") for item in envelopes}


def observed_texts(value):
    if isinstance(value, dict):
        return tuple(text for item in value.values() for text in observed_texts(item))
    if isinstance(value, (list, tuple)):
        return tuple(text for item in value for text in observed_texts(item))
    if value is None:
        return ()
    return (str(value),)


def preserves_observed_text(value, source):
    folded = value.casefold()
    return any(folded in text.casefold() for text in observed_texts(source))


def observation_view(value):
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else value


def restored_observation(native, view, replacement):
    if replacement == view:
        return native
    restore = getattr(type(native), "model_validate", None)
    if callable(restore) and isinstance(replacement, dict):
        return restore(replacement)
    return replacement


def infer_task_contract(context, output):
    for field in ("requirements", "expected_units", "expected_periods",
                  "source_requirements", "calculation_requirements"):
        if field in output:
            bounded_string_list(output, field, 64)
    bounded_text(output, "guidance", 6000)


def build_shared_context(context, output):
    for field in ("finance_snapshot", "tool_context", "subagent_context",
                  "subagent_round_guidance", "guidance"):
        bounded_text(output, field, 6000)
    combined = "\n".join(value for value in
        (output.get("subagent_context", ""), output.get("guidance", "")) if value)
    if len(combined) > 6000:
        raise ValueError("combined subagent_context exceeds the native limit")


def build_graph_context(context, output):
    bounded_text(output, "guidance", 6000)


def review_team_plan(context, output):
    for field in ("covered_slots", "missing_slots", "duplicated_assignments", "dependency_warnings"):
        bounded_string_list(output, field, 64)
    bounded_text(output, "revision_guidance", 6000)
    if "revised_plan" not in output:
        return
    plan = output["revised_plan"]
    if set(plan) != {"subtasks", "reasoning"} or not isinstance(plan["reasoning"], str) or not plan["reasoning"]:
        raise ValueError("revised_plan requires subtasks and nonempty reasoning")
    if len(plan["reasoning"]) > 6000:
        raise ValueError("revised_plan reasoning exceeds the native limit")
    subtasks = plan["subtasks"]
    if not isinstance(subtasks, list) or len(subtasks) != context["worker_count"]:
        raise ValueError("revised_plan must assign every fixed worker exactly once")
    if any(set(item) != {"agent_id", "objective", "focus"} or
           any(not isinstance(item[field], str) or not item[field]
               for field in ("agent_id", "objective", "focus")) for item in subtasks):
        raise ValueError("revised_plan subtasks require agent_id, objective and focus")
    original = {str(item.get("agent_id", "")) for item in context["lead_plan"].get("subtasks", [])}
    revised = [item["agent_id"] for item in subtasks]
    if len(set(revised)) != len(revised) or set(revised) != original:
        raise ValueError("revised_plan cannot change worker identities")


def regeneration_contract(context, output):
    for field in ("tool_issue_codes", "self_check_findings"):
        bounded_string_list(output, field, 64)
    for field in ("regeneration_guidance", "guidance"):
        bounded_text(output, field, 6000)
    checks = output.get("self_check_findings", [])
    guidance = output.get("guidance", "")
    if guidance and guidance not in checks and len(checks) >= 64:
        raise ValueError("combined self_check_findings exceeds the native limit")
    if output.get("regeneration_requested") and not output.get("regeneration_guidance"):
        raise ValueError("regeneration requires guidance")


def normalize_worker_handoff(context, output):
    regeneration_contract(context, output)
    observed = context.get("result", {})
    for field in ("findings", "source_handles", "tool_errors"):
        if field not in output:
            continue
        string_list(output[field], field)
        if any(not preserves_observed_text(item, observed) for item in output[field]):
            raise ValueError(f"{field} must preserve observed worker data")
    known_slots = slot_ids(context)
    bounded_string_list(output, "findings", 128)
    bounded_string_list(output, "covered_slots", 64)
    bounded_string_list(output, "missing_information", 64)
    bounded_string_list(output, "source_handles", 128)
    bounded_string_list(output, "tool_errors", 128)
    if output.get("guidance") and len(output.get("findings", [])) >= 128:
        raise ValueError("combined normalized_findings exceeds the native limit")
    if len(output.get("evidence_items", [])) > 256:
        raise ValueError("evidence_items exceeds the native limit")
    if not set(output.get("covered_slots", [])) <= known_slots:
        raise ValueError("covered_slots contains an unknown task-contract slot")
    allowed = {"evidence_id", "agent_id", "trace_reference", "content", "evidence_role",
               "supports_slots", "source_url", "source_type", "metric", "value", "unit",
               "period", "confidence", "raw_excerpt", "metadata"}
    roles = {"candidate_answer", "source_fact", "calculation", "negative_finding", "supporting_context"}
    for item in output.get("evidence_items", []):
        if not isinstance(item, dict) or not set(item) <= allowed:
            raise ValueError("evidence_items contains an invalid record")
        content = item.get("content")
        if not isinstance(content, str) or not content or not preserves_observed_text(content, observed):
            raise ValueError("evidence content must originate in the worker result")
        if item.get("evidence_role") not in roles:
            raise ValueError("evidence_role is invalid")
        supports = item.get("supports_slots", [])
        string_list(supports, "supports_slots")
        if not set(supports) <= known_slots:
            raise ValueError("evidence references an unknown task-contract slot")
        for field in ("value", "unit", "period", "source_url", "raw_excerpt"):
            value = item.get(field)
            if value not in (None, "") and (not isinstance(value, str)
                    or not preserves_observed_text(value, observed)):
                raise ValueError(f"evidence must preserve the observed {field} binding")
        confidence = item.get("confidence", 0.5)
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise ValueError("evidence confidence must be between zero and one")


def resolve_edge_semantics(context, output):
    bounded_string_list(output, "conflicts", 64)
    bounded_string_list(output, "preferred_worker_ids", 16)
    bounded_string_list(output, "missing_information", 64)
    bounded_text(output, "guidance", 6000)
    if output.get("guidance") and len(output.get("missing_information", [])) >= 64:
        raise ValueError("combined synthesis_guidance exceeds the native limit")
    if not set(output.get("preferred_worker_ids", [])) <= worker_ids(context):
        raise ValueError("preferred_worker_ids contains an unknown worker")


def update_graph_state(context, output):
    bounded_string_list(output, "active_worker_ids", 16)
    bounded_string_list(output, "covered_slots", 64)
    bounded_string_list(output, "missing_information", 64)
    bounded_string_list(output, "state_notes", 64)
    if not set(output.get("active_worker_ids", [])) <= worker_ids(context):
        raise ValueError("active_worker_ids contains an unknown worker")
    if not set(output.get("covered_slots", [])) <= slot_ids(context):
        raise ValueError("covered_slots contains an unknown task-contract slot")


def plan_graph_readout(context, output):
    bounded_string_list(output, "prioritized_worker_ids", 16)
    bounded_text(output, "guidance", 6000)
    if not set(output.get("prioritized_worker_ids", [])) <= worker_ids(context):
        raise ValueError("prioritized_worker_ids contains an unknown worker")


def model_review(context, output):
    bounded_text(output, "retry_instruction", 6000)
    if output.get("action") == "retry" and not output.get("retry_instruction"):
        raise ValueError("model retry requires retry_instruction")


def decide_orchestration(context, output):
    bounded_string_list(output, "target_workers", 16)
    bounded_string_list(output, "missing_information", 64)
    bounded_string_list(output, "synthesis_guidance", 64)
    bounded_text(output, "guidance", 6000)
    bounded_text(output, "reason", 2000)


def prepare_tool_call(context, output):
    if "tool_call" not in output:
        return
    replacement = output["tool_call"]
    if not isinstance(replacement.get("name"), str) or not replacement["name"]:
        raise ValueError("tool_call requires a nonempty name")
    if not isinstance(replacement.get("args"), dict):
        raise ValueError("tool_call.args must be an object")
    original = context.get("tool_call", {})
    if original.get("type") == "tool_call":
        for field in ("type", "id"):
            if field not in replacement or replacement[field] != original[field]:
                raise ValueError(f"tool_call must preserve native {field}")


def decide_tool_transition(context, output):
    if output.get("action") not in ("retry", "switch") or "tool_call" not in output:
        return
    prepare_tool_call(context, output)
    original = context.get("tool_call", {}).get("args", {})
    replacement = output["tool_call"].get("args", {})
    for field in ("start_date", "end_date"):
        if field in original and replacement.get(field) != original[field]:
            raise ValueError(f"tool transition must preserve {field}")
    query = original.get("search_query")
    replacement_query = replacement.get("search_query")
    if (isinstance(query, str) and query.strip()
            and (not isinstance(replacement_query, str)
                 or query.casefold() not in replacement_query.casefold())):
        raise ValueError("tool transition must preserve search_query")


FINANCE_HOOK_CONTRACTS = {
    "infer_task_contract": infer_task_contract,
    "build_graph_context": build_graph_context,
    "build_shared_context": build_shared_context,
    "review_team_plan": review_team_plan,
    "review_node_execution": regeneration_contract,
    "normalize_worker_handoff": normalize_worker_handoff,
    "resolve_edge_semantics": resolve_edge_semantics,
    "update_graph_state": update_graph_state,
    "plan_graph_readout": plan_graph_readout,
    "decide_orchestration": decide_orchestration,
    "prepare_tool_call": prepare_tool_call,
    "decide_tool_transition": decide_tool_transition,
    "review_model_result": model_review,
    "review_team_model_result": model_review,
}
