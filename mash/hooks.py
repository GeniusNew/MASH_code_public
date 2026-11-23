
from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .models import Candidate, Event, Execution, Harness, Level

HookContract = Callable[[dict, dict], None]


@dataclass(frozen=True)
class Target:
    level: Level
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]



TARGETS = {
    "infer_task_contract": Target("graph", ("task",),
                                  ("requirements", "expected_units", "expected_periods",
                                   "source_requirements", "calculation_requirements", "guidance")),
    "build_graph_context": Target("graph", ("task", "contract", "tools"), ("guidance",)),
    "build_shared_context": Target("node", ("task", "contract", "role", "tools"),
                                   ("finance_snapshot", "tool_context", "subagent_context",
                                    "subagent_round_guidance", "guidance")),
    "prepare_worker_request": Target("edge", ("task", "contract", "assignment"), ("guidance",)),
    "review_team_plan": Target("graph", ("task", "contract", "lead_plan", "worker_count"),
                               ("covered_slots", "missing_slots", "duplicated_assignments",
                                "dependency_warnings", "revision_needed", "revision_guidance",
                                "revised_plan")),
    "review_node_execution": Target("node", ("task", "contract", "assignment", "result"),
                                    ("tool_issue_codes", "self_check_findings",
                                     "regeneration_requested", "regeneration_guidance", "guidance")),
    "prepare_node_action": Target("node", ("task", "contract", "assignment", "action"),
                                  ("allow", "arguments", "reason")),
    "normalize_worker_handoff": Target("edge", ("task", "contract", "assignment", "result"),
                                      ("findings", "evidence_items", "covered_slots", "missing_information",
                                       "source_handles", "tool_errors", "regeneration_requested",
                                       "regeneration_guidance", "guidance")),
    "resolve_edge_semantics": Target("graph", ("task", "contract", "envelopes"),
                                    ("conflicts", "preferred_worker_ids", "missing_information", "guidance")),
    "update_graph_state": Target("graph", ("task", "contract", "round_index", "envelopes", "previous_state"),
                                 ("active_worker_ids", "covered_slots", "missing_information", "state_notes")),
    "plan_graph_readout": Target("graph", ("task", "contract", "envelopes", "resolution", "graph_state"),
                                ("prioritized_worker_ids", "guidance")),
    "decide_orchestration": Target("graph", ("task", "contract", "round_index", "envelopes", "remaining_budget"),
                                    ("action", "target_workers", "missing_information", "guidance",
                                     "synthesis_guidance", "reason")),
    "prepare_model_call": Target("node", ("messages", "role", "phase", "round_index"), ("guidance",)),
    "review_model_result": Target("node", ("result", "role", "phase", "round_index"),
                                  ("action", "retry_instruction")),
    "prepare_team_model_call": Target("graph", ("messages", "role", "phase", "round_index"), ("guidance",)),
    "review_team_model_result": Target("graph", ("result", "role", "phase", "round_index", "readout"),
                                       ("action", "retry_instruction")),
    "prepare_tool_call": Target("node", ("task", "contract", "tool_call", "role", "phase"),
                                ("tool_call",)),
    "review_tool_result": Target("node", ("task", "contract", "tool_call", "observation",
                                            "failure_kind", "role", "phase"),
                                 ("observation", "failure_kind")),
    "decide_tool_transition": Target("node", ("task", "contract", "tool_call", "reviewed_result",
                                               "role", "phase"),
                                     ("action", "tool_call")),
}

SAFE_BUILTINS = {"len": len, "str": str, "int": int, "float": float, "bool": bool,
                 "list": list, "dict": dict, "tuple": tuple, "sorted": sorted,
                 "min": min, "max": max, "sum": sum, "any": any, "all": all,
                 "abs": abs, "round": round, "enumerate": enumerate, "zip": zip,
                 "isinstance": isinstance}
SAFE_METHODS = {"get", "keys", "values", "items", "lower", "upper", "strip", "split",
                "join", "startswith", "endswith", "replace", "casefold", "find"}
ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.If,
    ast.Expr, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Dict, ast.List, ast.Tuple,
    ast.Subscript, ast.Slice, ast.Call, ast.keyword, ast.Attribute, ast.IfExp,
    ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In,
    ast.NotIn, ast.Is, ast.IsNot, ast.ListComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension,
}

OUTPUT_TYPES = {"guidance": str, "action": str, "allow": bool, "arguments": dict, "reason": str,
                "tool_call": dict, "observation": object, "failure_kind": str,
                "revision_needed": bool, "revision_guidance": str, "revised_plan": dict,
                "regeneration_requested": bool, "regeneration_guidance": str,
                "retry_instruction": str, "finance_snapshot": str, "tool_context": str,
                "subagent_context": str, "subagent_round_guidance": str}
ORCHESTRATION_ACTIONS = ("native", "continue", "stop")
TOOL_TRANSITION_ACTIONS = ("continue", "retry", "switch", "stop")
TOOL_FAILURE_KINDS = ("none", "soft", "exception")
MODEL_REVIEW_ACTIONS = ("accept", "retry")


def target_spec(target: str) -> dict[str, Any]:
    spec = {
        "level": TARGETS[target].level,
        "inputs": TARGETS[target].inputs,
        "outputs": TARGETS[target].outputs,
        "output_types": {name: OUTPUT_TYPES.get(name, list).__name__
                         for name in TARGETS[target].outputs},
    }
    enums = {}
    if target == "infer_task_contract":
        spec["effect"] = (
            "requirements become ordered req-i slots for later hooks; this target does not insert the "
            "contract into model messages")
        spec["composition_semantics"] = (
            "to change model behavior, pair it with the context, request or readout target at the diagnosed boundary")
    elif target == "build_graph_context":
        spec["effect"] = (
            "guidance is injected into the lead task context before planning and remains in the lead prompt "
            "templates used for final synthesis; it does not reach worker prompts")
    elif target == "build_shared_context":
        spec["effect"] = (
            "finance_snapshot, tool_context, subagent_context and legacy guidance enter worker task context; "
            "subagent_round_guidance is worker-round guidance; none of these fields reaches the lead")
    elif target == "prepare_worker_request":
        spec["effect"] = (
            "guidance is appended to the already assigned worker request; recipient identity, role, topology "
            "and the native assignment remain fixed")
    elif target == "update_graph_state":
        spec["effect"] = (
            "records coverage and state for graph readout; active_worker_ids is descriptive and does not "
            "schedule, rerun or reassign a worker")
        spec["input_semantics"] = (
            "previous_state is an earlier hook-produced summary, not independent evidence. A worker id in "
            "active_worker_ids alone does not establish missing work; ground missing_information in the "
            "current envelopes, contract requirements or an explicit observed failure.")
    elif target == "plan_graph_readout":
        spec["effect"] = (
            "prioritized_worker_ids reorder existing worker findings for synthesis and guidance enters the "
            "final structured synthesis context; this target cannot create or replace worker evidence")
        spec["input_semantics"] = (
            "graph_state.active_worker_ids is descriptive hook-produced state and alone does not establish "
            "that work is incomplete. Require current missing_information, unresolved contract slots, tool "
            "errors or explicit unfinished output before adding unresolved guidance. Identify answer-bearing "
            "envelopes from structural evidence and completion markers, not literal benchmark units, years, "
            "entities or answer-format tokens. A supporting envelope is not incomplete merely because it lacks "
            "a final-answer marker.")
        spec["guidance_semantics"] = (
            "refer to the task's requested format generically; do not hardcode the count or kind of answer "
            "sections, periods, explanations, units or metrics from a discovery task")
        spec["behavior_case_guidance"] = (
            "include a preserve_parent case where an answer-bearing envelope coexists with a useful supporting "
            "envelope that has no unresolved marker")
    elif target == "decide_orchestration":
        enums["action"] = ORCHESTRATION_ACTIONS
    elif target == "decide_tool_transition":
        enums["action"] = TOOL_TRANSITION_ACTIONS
        spec["effect"] = ("continue and stop both return the reviewed observation; retry and switch "
                          "add one bounded attempt and are the only actions that can recover a returned failure")
        spec["action_semantics"] = {
            "retry": ("executes the same tool call unless tool_call supplies a replacement; use it when "
                      "another attempt against the same provider can plausibly succeed"),
            "switch": ("executes the supplied alternative tool_call once; use an observed existing tool "
                       "for persistent provider failures and preserve the original query and time constraints"),
        }
        spec["behavior_case_guidance"] = (
            "when observed successful calls exist, include one as a source-backed preserve_parent case; "
            "when switching tools that share search or date fields, include a case preserving search_query, "
            "start_date and end_date")
    elif target == "review_tool_result":
        enums["failure_kind"] = TOOL_FAILURE_KINDS
        spec["effect"] = (
            "failure_kind records classification but does not by itself change the observation returned "
            "to the worker or the native continue transition; recovery requires a compatible "
            "decide_tool_transition action")
        spec["observation_semantics"] = (
            "observation may be a JSON object, list or string; guard object methods with isinstance; "
            "content may contain arbitrary retrieved documents; classify a returned failure "
            "from the payload envelope or an exact structural marker, not a matching phrase anywhere "
            "inside retrieved evidence")
        spec["behavior_case_guidance"] = (
            "when observed successful calls exist, include one as a source-backed preserve_parent case")
    elif target == "normalize_worker_handoff":
        spec["effect"] = (
            "findings and evidence_items reach graph resolution and final synthesis; guidance is appended "
            "to normalized findings; regeneration_requested uses the existing bounded regeneration path")
        spec["consumption_semantics"] = (
            "native final synthesis renders normalized findings before evidence_items and tool_errors, then "
            "truncates the combined structured handoff to 6000 characters; place compact source-bearing "
            "successful evidence before diagnostics and avoid duplicating the same text across fields")
        spec["provenance_semantics"] = (
            "every findings, source_handles and tool_errors string must be copied as an exact substring "
            "from result; every evidence_items content, value, unit, period, source_url and raw_excerpt "
            "must likewise preserve an exact observed result substring; do not wrap observed text in new prose")
        spec["input_semantics"] = (
            "result.status is the worker conversation lifecycle state; active alone does not mean the "
            "submitted handoff is incomplete. source_records and calculation_records contain only records "
            "recognized from explicitly structured tool payload keys and may be empty for a usable answer. "
            "Do not trigger from any of these fields alone; require an evidenced non-answer, tool failure, "
            "timeout, error, or task-relevant missing output. tool_observations.success records dispatcher "
            "completion, while output_excerpt may contain either a structured inner success=false envelope "
            "or arbitrary retrieved text; do not classify generic error or failure words inside documents.")
        spec["evidence_item_semantics"] = {
            "required": ("content", "evidence_role"),
            "roles": ("candidate_answer", "source_fact", "calculation", "negative_finding",
                      "supporting_context"),
            "supports_slots": "must name only contract.required_slots slot_id values",
        }
        spec["behavior_case_guidance"] = (
            "for empty submitted findings with useful raw result data, copy explicit_candidate_answer or "
            "an exact tool observation excerpt into findings or evidence_items; synthesized instructions "
            "belong in missing_information or guidance")
    elif target in ("review_model_result", "review_team_model_result"):
        enums["action"] = MODEL_REVIEW_ACTIONS
        spec["effect"] = (
            "retry adds one same-lifecycle model call with retry_instruction; each native model-call "
            "lifecycle has its own bounded retry opportunity, so use a narrow observed trigger")
        spec["result_semantics"] = (
            "result may be a JSON object or string; guard object methods with isinstance; result.content "
            "may be empty when result.tool_calls or additional_kwargs.tool_calls contains "
            "a valid tool request; that is not an empty model response and must be preserved")
        spec["behavior_case_guidance"] = (
            "include a source-backed trigger case and a preserve_parent case for a substantive result; "
            "for worker results, also preserve an empty-content result carrying a tool call")
    if enums:
        spec["output_enums"] = enums
    if target in ("prepare_tool_call", "decide_tool_transition"):
        spec["tool_call_semantics"] = (
            "preserve the observed native call envelope when replacing name or args, including type and id; "
            "dict(context['tool_call'], name=existing_tool_name, args=replacement_args) preserves those fields. "
            "A bare name/args object can be interpreted as raw arguments by the native dispatcher.")
    return spec


class HookError(ValueError):
    def __init__(self, message: str, execution: Execution | None = None):
        super().__init__(message)
        self.execution = execution


def _numeric_multiply(left, right):

    if type(left) not in (int, float) or type(right) not in (int, float):
        raise TypeError("multiplication requires numeric operands")
    return left * right


def executable_tree(tree: ast.Module) -> ast.Module:

    class NumericMultiplication(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Mult):
                return ast.copy_location(ast.Call(func=ast.Name(id="_numeric_multiply", ctx=ast.Load()),
                                                  args=[node.left, node.right], keywords=[]), node)
            return node

    return ast.fix_missing_locations(NumericMultiplication().visit(copy.deepcopy(tree)))


def check_literal_multiplication(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for operand in (node.left, node.right):
                try:
                    value = ast.literal_eval(operand)
                except (ValueError, TypeError):
                    continue
                if type(value) not in (int, float):
                    raise HookError("multiplication requires numeric operands")


def check_output_value(target: str, name: str, value: Any):
    expected = OUTPUT_TYPES.get(name, list)
    if not isinstance(value, expected):
        raise HookError(f"{target}.{name} must be {expected.__name__}")
    if target == "decide_orchestration":
        if name == "action" and value not in ORCHESTRATION_ACTIONS:
            raise HookError("orchestration action must be native, continue or stop")
        if name in ("target_workers", "missing_information") and any(not isinstance(v, str) for v in value):
            raise HookError(f"{target}.{name} must contain strings")
    if name in ("tool_issue_codes", "self_check_findings", "synthesis_guidance"):
        if any(not isinstance(v, str) for v in value):
            raise HookError(f"{target}.{name} must contain strings")
    if target == "decide_tool_transition" and name == "action" and value not in TOOL_TRANSITION_ACTIONS:
        raise HookError("tool transition action must be continue, retry, switch or stop")
    if target == "review_tool_result" and name == "failure_kind" and value not in TOOL_FAILURE_KINDS:
        raise HookError("tool failure kind must be none, soft or exception")
    if target in ("review_model_result", "review_team_model_result"):
        if name == "action" and value not in MODEL_REVIEW_ACTIONS:
            raise HookError("model review action must be accept or retry")


def check_literal_outputs(tree: ast.Module, target: str):

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        if isinstance(node.value, (ast.Constant, ast.List, ast.Tuple)) or node.value is None:
            raise HookError(f"{target} must return a dict")
        if isinstance(node.value, ast.Dict):
            for key, value_node in zip(node.value.keys, node.value.values):
                if not isinstance(key, ast.Constant):
                    continue
                if key.value not in TARGETS[target].outputs:
                    raise HookError("return fields fall outside target's output contract")
                try:
                    value = ast.literal_eval(value_node)
                except (ValueError, TypeError):
                    continue
                check_output_value(target, key.value, value)


def check_source(target: str, source: str) -> ast.Module:
    if target not in TARGETS:
        raise HookError(f"unknown target: {target}")
    try:
        tree = ast.parse(source)
        compile(tree, f"<{target}>", "exec")
    except SyntaxError as exc:
        raise HookError(f"syntax: {exc.msg}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise HookError("source must contain exactly one function")
    function = tree.body[0]
    args = function.args
    if (function.name != target or function.decorator_list or function.returns
            or [a.arg for a in args.args] != ["context"] or args.posonlyargs
            or args.kwonlyargs or args.vararg or args.kwarg or args.defaults
            or any(a.annotation for a in args.args)):
        raise HookError(f"expected undecorated def {target}(context)")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise HookError(f"forbidden operation: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise HookError("nested functions are forbidden")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id == target):
            raise HookError("private names and recursion are forbidden")
        if isinstance(node, ast.Assign) and any(not isinstance(t, ast.Name) for t in node.targets):
            raise HookError("assignments may only bind local names")
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, ast.Store):
            raise HookError("mutation is forbidden, including comprehension targets")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_METHODS:
            raise HookError(f"forbidden attribute: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS:
                    raise HookError(f"forbidden call: {node.func.id}")
            elif not isinstance(node.func, ast.Attribute):
                raise HookError("indirect calls are forbidden")
        if isinstance(node, ast.comprehension) and node.is_async:
            raise HookError("async comprehension is forbidden")
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = node.value.keys
            if any(not isinstance(k, ast.Constant) or k.value not in TARGETS[target].outputs
                   for k in keys):
                raise HookError("return fields fall outside target's output contract")
    return tree


def check_declared_names(tree: ast.Module):

    local_names = {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   for target in node.targets}

    def visit(node, available):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in available:
                raise HookError(f"undeclared name: {node.id}")
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.GeneratorExp)):

            visit(node.generators[0].iter, available)
            names = {item.id for generator in node.generators
                     for item in ast.walk(generator.target) if isinstance(item, ast.Name)}
            scope = available | names
            for index, generator in enumerate(node.generators):
                if index:
                    visit(generator.iter, scope)
                for condition in generator.ifs:
                    visit(condition, scope)
            if isinstance(node, ast.DictComp):
                visit(node.key, scope)
                visit(node.value, scope)
            else:
                visit(node.elt, scope)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, available)

    visit(tree, set(SAFE_BUILTINS) | local_names | {"context"})


def check_context_inputs(tree: ast.Module, target: str):


    if any(isinstance(node, ast.Assign) and any(t.id == "context" for t in node.targets)
           for node in ast.walk(tree)):
        return

    def visit(node, supplied_context=True):
        if isinstance(node, (ast.ListComp, ast.DictComp, ast.GeneratorExp)):
            visit(node.generators[0].iter, supplied_context)
            shadowed = any(isinstance(n, ast.Name) and n.id == "context"
                          for g in node.generators for n in ast.walk(g.target))
            inner_context = supplied_context and not shadowed
            for i, generator in enumerate(node.generators):
                if i:
                    visit(generator.iter, inner_context)
                for condition in generator.ifs:
                    visit(condition, inner_context)
            for expression in ((node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)):
                visit(expression, inner_context)
            return

        base, key = None, None
        if isinstance(node, ast.Subscript):
            base, key = node.value, node.slice
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "get" and node.args):
            base, key = node.func.value, node.args[0]
        if (supplied_context and isinstance(base, ast.Name) and base.id == "context"
                and isinstance(key, ast.Constant) and key.value not in TARGETS[target].inputs):
            raise HookError(f"unregistered context input: {key.value!r}")
        for child in ast.iter_child_nodes(node):
            visit(child, supplied_context)

    visit(tree)


def check_task_dispatch(tree: ast.Module):
    class Check(ast.NodeVisitor):
        def __init__(self):
            self.aliases = {"context": ()}

        def path(self, node):
            if isinstance(node, ast.Name):
                return self.aliases.get(node.id)
            base, key = None, None
            if isinstance(node, ast.Subscript):
                base, key = node.value, node.slice
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("lower", "upper", "strip", "casefold"):
                    return self.path(node.func.value)
                if node.func.attr == "get" and node.args:
                    base, key = node.func.value, node.args[0]
            if base is not None and isinstance(key, ast.Constant):
                parent = self.path(base)
                if parent is not None:
                    return (*parent, key.value)
            return None

        def identity(self, node):
            return self.path(node) in (("task", "task_id"), ("task", "question"),
                                       ("task", "payload", "question"))

        def visit_Assign(self, node):
            self.visit(node.value)
            path = self.path(node.value)
            for target in node.targets:
                self.aliases[target.id] = path

        def visit_If(self, node):
            self.visit(node.test)
            before = self.aliases.copy()
            for statement in node.body:
                self.visit(statement)
            left = self.aliases
            self.aliases = before
            for statement in node.orelse:
                self.visit(statement)
            self.aliases = {k: v for k, v in left.items() if k in self.aliases and self.aliases[k] == v}

        def visit_ListComp(self, node):
            before = self.aliases.copy()
            for generator in node.generators:
                self.visit(generator.iter)
                for name in ast.walk(generator.target):
                    if isinstance(name, ast.Name):
                        self.aliases.pop(name.id, None)
                for condition in generator.ifs:
                    self.visit(condition)
            for value in ((node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)):
                self.visit(value)
            self.aliases = before

        visit_DictComp = visit_GeneratorExp = visit_ListComp

        def visit_Compare(self, node):
            for left, operation, right in zip((node.left, *node.comparators), node.ops, node.comparators):
                pairs = [(left, right)] if isinstance(operation, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) else []
                if isinstance(operation, (ast.Eq, ast.NotEq)):
                    pairs.append((right, left))
                for identity, other in pairs:
                    if not self.identity(identity):
                        continue
                    try:
                        literal = ast.literal_eval(other)
                    except (ValueError, TypeError):
                        continue
                    values = literal if isinstance(literal, (list, tuple, dict)) else (literal,)
                    if any(isinstance(value, str) and value.strip() for value in values):
                        raise HookError("task-specific dispatch on a literal task_id or whole question is forbidden")
            self.generic_visit(node)

    Check().visit(tree)


def simplify_literals(tree: ast.Module) -> ast.Module:






    result = copy.deepcopy(tree)


    propagate = not any(isinstance(node, (ast.Is, ast.IsNot)) for node in ast.walk(tree))
    dict_is_builtin = not any(isinstance(node, ast.Assign) and any(t.id == "dict" for t in node.targets)
                              for node in ast.walk(tree))

    class Substitute(ast.NodeTransformer):
        def __init__(self, bindings):
            self.bindings = bindings

        def visit_Name(self, node):
            if propagate and isinstance(node.ctx, ast.Load) and node.id in self.bindings:
                return copy.deepcopy(self.bindings[node.id])
            return node

        def visit_IfExp(self, node):
            node.test = self.visit(node.test)
            try:
                test = ast.literal_eval(node.test)
            except (ValueError, TypeError):
                node.body, node.orelse = self.visit(node.body), self.visit(node.orelse)
                return node
            return self.visit(node.body if test else node.orelse)

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (not dict_is_builtin or not isinstance(node.func, ast.Name) or node.func.id != "dict"
                    or any(keyword.arg is None for keyword in node.keywords)):
                return node
            try:
                args = [ast.literal_eval(arg) for arg in node.args]
                kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
            except (ValueError, TypeError):
                return node
            try:
                value = dict(*args, **kwargs)
            except (ValueError, TypeError):


                return node
            return ast.copy_location(ast.parse(repr(value), mode="eval").body, node)


        def visit_ListComp(self, node):
            return node

        visit_DictComp = visit_GeneratorExp = visit_ListComp

    def block(statements, bindings):
        output = []
        for statement in statements:
            substitute = Substitute(bindings)
            if isinstance(statement, ast.If):
                statement.test = substitute.visit(statement.test)
                try:
                    test = ast.literal_eval(statement.test)
                except (ValueError, TypeError):
                    left, right = bindings.copy(), bindings.copy()
                    statement.body, left_returns = block(statement.body, left)
                    statement.orelse, right_returns = block(statement.orelse, right)
                    output.append(statement)
                    continuing = [env for env, returns in ((left, left_returns), (right, right_returns)) if not returns]
                    if not continuing:
                        return output, True
                    shared = {key: value for key, value in continuing[0].items()
                              if all(key in env and ast.dump(value) == ast.dump(env[key]) for env in continuing[1:])}
                    bindings.clear()
                    bindings.update(shared)
                else:
                    branch, returns = block(statement.body if test else statement.orelse, bindings)
                    output.extend(branch)
                    if returns:
                        return output, True
            elif isinstance(statement, ast.Assign):
                statement.value = substitute.visit(statement.value)
                for target in statement.targets:
                    bindings.pop(target.id, None)
                    try:
                        ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        pass
                    else:
                        bindings[target.id] = statement.value
                output.append(statement)
            else:
                output.append(substitute.visit(statement))
                if isinstance(statement, ast.Return):
                    return output, True
        return output, False

    result.body[0].body, returns = block(result.body[0].body, {})
    if not returns:
        result.body[0].body.append(ast.Return(value=ast.Constant(value=None)))
    return result


def substantive_ast(tree: ast.Module) -> str:






    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    class RemoveInertStatements(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return node

        def visit_Assign(self, node):
            if all(isinstance(t, ast.Name) and t.id not in loaded for t in node.targets):
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return node
                return None
            return node

    return ast.dump(RemoveInertStatements().visit(copy.deepcopy(tree)))


def source_preflight(candidate: Candidate, exposed: tuple[str, ...]) -> tuple[str, ...]:
    errors = []
    for patch in candidate.patches:
        if patch.target not in exposed:
            errors.append(f"target is not exposed by this workflow: {patch.target}")
            continue
        try:
            source_tree = check_source(patch.target, patch.source)
            check_declared_names(source_tree)
            check_context_inputs(source_tree, patch.target)
            check_task_dispatch(source_tree)
            tree = simplify_literals(source_tree)
            check_literal_multiplication(tree)
            check_literal_outputs(tree, patch.target)
        except HookError as exc:
            errors.append(f"{patch.target}: {exc}")
    return tuple(errors)


def check_behavior_assertion(target: str, assertion, result: dict[str, Any]):
    path = assertion.field.split(".")
    root = path[0]
    if root not in TARGETS[target].outputs:
        raise HookError(f"behavior assertion uses unregistered output field: {assertion.field}")
    present = root in result
    value = result.get(root)
    for name in path[1:]:
        if not present or not isinstance(value, dict) or name not in value:
            present, value = False, None
            break
        value = value[name]
    if assertion.operator == "omitted":
        if present:
            raise HookError(f"expected {assertion.field} to be omitted")
    elif assertion.operator == "equals":
        if len(path) == 1:
            check_output_value(target, root, assertion.value)
        if not present or value != assertion.value:
            raise HookError(f"expected {assertion.field} to equal {assertion.value!r}, received {value!r}")
    elif assertion.operator == "nonempty":
        if not present or value is None or value == "" or value == [] or value == {}:
            raise HookError(f"expected {assertion.field} to be nonempty")
    elif assertion.operator == "contains_all":
        if isinstance(value, str):
            matched = all(item.casefold() in value.casefold() for item in assertion.value)
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            folded = {item.casefold() for item in value}
            matched = all(item.casefold() in folded for item in assertion.value)
        else:
            matched = False
        if not matched:
            raise HookError(f"expected {assertion.field} to contain {assertion.value!r}, received {value!r}")
    elif assertion.operator == "length_at_most":
        if not present or not hasattr(value, "__len__") or len(value) > assertion.value:
            raise HookError(f"expected {assertion.field} length at most {assertion.value}, received {value!r}")


def candidate_boundary_errors(candidate, parent, exposed, failures, variant, candidate_ids=()):
    errors = []
    groups = {group.group_id: group for group in failures}
    planned = {change.target for change in candidate.changes}
    targets = {patch.target for patch in candidate.patches}
    if candidate.candidate_id in candidate_ids:
        errors.append("candidate ID duplicates another submitted candidate")
    if candidate.parent_id != parent.harness_id:
        errors.append("candidate belongs to another parent")
    if candidate.group_id not in groups:
        errors.append("candidate does not address a diagnosed failure group")
    if len(planned) != len(candidate.changes) or len(targets) != len(candidate.patches):
        errors.append("duplicate planned or implemented target")
    if not (planned | targets) <= set(exposed):
        errors.append("candidate changes a frozen or unavailable target")
    for change in candidate.changes:
        if change.target in TARGETS and change.level != TARGETS[change.target].level:
            errors.append(f"incorrect level for {change.target}")
    levels = {TARGETS[target].level for target in targets if target in TARGETS}
    if variant in ("U", "R", "L") and len(levels) > 1:
        errors.append("strategy requires a single-level candidate")
    if variant == "L" and candidate.group_id in groups:
        diagnosed = {getattr(record, "level", None) for record in groups[candidate.group_id].records}
        if levels != diagnosed:
            errors.append("candidate changes a different level from its analysis session")
    return tuple(errors)


def stage1(candidate: Candidate, parent: Harness, exposed: tuple[str, ...],
           *, evidence: Mapping[tuple[str, str], Event] | None = None,
           contracts: Mapping[str, HookContract] | None = None,
           require_frozen_plan: bool = False) -> tuple[str, ...]:

    errors: list[str] = []
    if require_frozen_plan and candidate.frozen_plan is None:
        errors.append("candidate requires a plan frozen before implementation")
    if candidate.frozen_plan is not None and candidate.revision_plan() != candidate.frozen_plan:
        errors.append("candidate differs from the frozen plan; replan before implementation")
    if candidate.parent_id != parent.harness_id:
        errors.append("candidate refers to a different parent")
    planned = {c.target: c for c in candidate.changes}
    patches = {p.target: p for p in candidate.patches}
    if len(planned) != len(candidate.changes) or len(patches) != len(candidate.patches):
        errors.append("duplicate planned or implemented target")
    if not set(patches) <= set(planned):
        errors.append("implementation contains changes outside the proposal")
    required = {c.target for c in candidate.changes if c.required}
    if not required <= set(patches):
        errors.append("implementation is missing required changes")
    existing = {h.target: h.source for h in parent.hooks}
    changed: set[str] = set()
    for target, change in planned.items():
        if target not in exposed or target not in TARGETS:
            errors.append(f"target is not exposed by this workflow: {target}")
        elif change.level != TARGETS[target].level:
            errors.append(f"incorrect level for {target}")
    for target, patch in patches.items():
        if target not in exposed:
            errors.append(f"target is not exposed by this workflow: {target}")
        try:
            source_tree = check_source(target, patch.source)
            check_declared_names(source_tree)
            check_context_inputs(source_tree, target)
            check_task_dispatch(source_tree)
            tree = simplify_literals(source_tree)
            check_literal_multiplication(tree)
            check_literal_outputs(tree, target)
            old = existing.get(target, f"def {target}(context):\n    return {{}}\n")
            if substantive_ast(tree) != substantive_ast(simplify_literals(ast.parse(old))):
                changed.add(target)
        except HookError as exc:
            errors.append(f"{target}: {exc}")
    if not changed:
        errors.append("detected no-op: no substantive AST change")
    if not required <= changed:
        errors.append("required target has no implemented change")

    if not errors:
        runtime = HookRuntime(Harness(hooks=candidate.patches), contracts=contracts)
        parent_runtime = HookRuntime(parent)
        for target in patches:
            cases = planned[target].behavior_cases
            if not cases:
                errors.append(f"{target}: proposal needs behavior_cases with input and expected output")
            observed = any(event.target == target for event in (evidence or {}).values())
            if observed and not any(case.source is not None for case in cases):
                errors.append(f"{target}: behavior_cases must cite an observed discovery hook input")
            for index, case in enumerate(cases, 1):
                try:
                    context = case.context
                    if case.source is not None:
                        event = (evidence or {}).get((case.source.run_id, case.source.event_id))
                        if event is None or event.target != target:
                            raise HookError("behavior source must cite this hook in supplied discovery evidence")
                        if (context is not None and json.dumps(context, sort_keys=True, allow_nan=False) !=
                                json.dumps(event.inputs, sort_keys=True, allow_nan=False)):
                            raise HookError("behavior context must equal the complete source event inputs")
                        context = event.inputs
                    if not set(context) <= set(TARGETS[target].inputs):
                        raise HookError("behavior case contains unregistered context fields")
                    if not set(case.expected or {}) <= set(TARGETS[target].outputs):
                        raise HookError("behavior case expects unregistered output fields")
                    for name, value in (case.expected or {}).items():
                        check_output_value(target, name, value)
                    json.dumps(context, allow_nan=False)
                    result = runtime.call(target, context)
                    if case.expected is not None:
                        mismatch = (bool(result) if not case.expected else
                                    any(name not in result or
                                        json.dumps(result[name], sort_keys=True, allow_nan=False) !=
                                        json.dumps(value, sort_keys=True, allow_nan=False)
                                        for name, value in case.expected.items()))
                        if mismatch:
                            raise HookError(f"expected {case.expected!r}, received {result!r}")
                    for assertion in case.assertions:
                        check_behavior_assertion(target, assertion, result)
                    if case.preserve_parent:
                        original = parent_runtime.call(target, context)
                        if (json.dumps(result, sort_keys=True, allow_nan=False) !=
                                json.dumps(original, sort_keys=True, allow_nan=False)):
                            raise HookError("preserve_parent case changes the parent's hook output")
                except (ValueError, TypeError) as exc:
                    errors.append(f"{target}: behavior case {index} ({case.description}): {exc}")
    return tuple(errors)


def cross_group_case_errors(candidate, failures, source_events) -> tuple[str, ...]:
    groups = {group.group_id: group for group in failures}
    if not groups or candidate.group_id not in groups:
        return (() if not groups else ("candidate group_id is not a supplied failure group",))
    planned = {change.target: change for change in candidate.changes}

    def cited(group, target):
        return {(ref.run_id, ref.event_id) for record in group.records for ref in record.evidence
                if (event := source_events.get((ref.run_id, ref.event_id))) is not None
                and event.target == target}

    errors = []
    for patch in candidate.patches:
        own = cited(groups[candidate.group_id], patch.target)
        foreign = set().union(*(cited(group, patch.target) for group_id, group in groups.items()
                                if group_id != candidate.group_id))
        if not own or not foreign:
            continue
        change = planned.get(patch.target)
        checked = {(case.source.run_id, case.source.event_id)
                   for case in (() if change is None else change.behavior_cases)
                   if case.source is not None}
        if not checked & foreign:
            errors.append(f"{patch.target}: shared-target groups require a source-backed "
                          "behavior case from another supplied failure group")
    return tuple(errors)


class HookRuntime:
    def __init__(self, harness: Harness, *, contracts: Mapping[str, HookContract] | None = None):
        self.contracts = dict(contracts or {})
        self.functions = {}
        for hook in harness.hooks:
            tree = check_source(hook.target, hook.source)
            namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS.copy(),
                                         "_numeric_multiply": _numeric_multiply}
            exec(compile(executable_tree(tree), f"<{hook.target}>", "exec"), namespace)
            self.functions[hook.target] = namespace[hook.target]

    def call(self, target: str, context: dict[str, Any]) -> dict[str, Any]:
        if target not in self.functions:
            return {}
        try:
            result = self.functions[target](copy.deepcopy(context))
        except Exception as exc:
            raise HookError(f"{target} execution failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(result, dict) or not set(result) <= set(TARGETS[target].outputs):
            raise HookError(f"{target} returned invalid output fields")
        for name, value in result.items():
            check_output_value(target, name, value)

        try:
            result = json.loads(json.dumps(result, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise HookError(f"{target} returned a non-JSON value: {exc}") from exc
        if target in self.contracts:
            try:
                self.contracts[target](copy.deepcopy(context), copy.deepcopy(result))
            except (ValueError, TypeError, LookupError, AssertionError) as exc:
                raise HookError(f"{target} violates native contract: {exc}") from exc
        return result
