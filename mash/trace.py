
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from threading import RLock
from typing import Any, Callable

from .hooks import HookContract, HookError, HookRuntime, TARGETS
from .models import Event, Execution, Harness, Level


@dataclass(frozen=True)
class Observation:
    event_id: str
    value: dict[str, Any]


@dataclass(frozen=True)
class Delivery:
    event_id: str
    sender: str
    recipients: tuple[str, ...]
    content: str
    round_number: int


class ExecutionTrace:







    def __init__(self, *, topology: str, workers: tuple[str, ...]):
        self.topology, self.workers = topology, workers
        self._events: list[Event] = []
        self._ids: set[str] = set()
        self._lock = RLock()

    def record(self, target: str, level: Level, inputs: dict, outputs: dict,
               *, parents: tuple[str, ...] = (), actor: str = "", round_number: int = 0) -> str:
        with self._lock:
            if not set(parents) <= self._ids:
                raise ValueError("a source event must exist before its consumer executes")
            event = Event(event_id=f"e{len(self._events):06d}", target=target, level=level,
                          inputs=deepcopy(inputs), outputs=deepcopy(outputs),
                          parents=tuple(dict.fromkeys(parents)), actor=actor, round_number=round_number)
            self._events.append(event)
            self._ids.add(event.event_id)
            return event.event_id

    def deliver(self, *, sender: str, recipients: tuple[str, ...], content: str,
                source_event: str, round_number: int) -> Delivery:

        if self.topology == "independent" and set(recipients) & set(self.workers):
            raise ValueError("independent workers cannot receive intermediate peer messages")
        event_id = self.record("deliver_contribution", "edge",
            {"sender": sender, "recipients": list(recipients)}, {"content": content},
            parents=(source_event,), actor=sender, round_number=round_number)
        return Delivery(event_id, sender, recipients, content, round_number)

    def model_call(self, *, actor: str, level: Level, round_number: int,
                   messages: list[dict], invoke: Callable[[list[dict]], dict],
                   parents: tuple[str, ...] = (), received: tuple[Delivery, ...] = ()) -> Observation:






        if any(actor not in message.recipients for message in received):
            raise ValueError("received evidence must be addressed to this consumer")
        supplied = deepcopy(messages)
        input_id = self.record("model_input", level,
            {"messages": supplied, "received_delivery_ids": [d.event_id for d in received]}, {},
            parents=(*parents, *(d.event_id for d in received)), actor=actor, round_number=round_number)
        try:
            output = invoke(deepcopy(supplied))
        except Exception as exc:
            self.record("model_error", level, {}, {"error_type": type(exc).__name__},
                        parents=(input_id,), actor=actor, round_number=round_number)
            raise
        event_id = self.record("model_output", level, {}, output, parents=(input_id,),
                               actor=actor, round_number=round_number)
        return Observation(event_id, output)

    @property
    def events(self) -> tuple[Event, ...]:
        with self._lock:
            return tuple(event.model_copy(deep=True) for event in self._events)


class TaskRuntime:


    def __init__(self, harness: Harness, exposed_targets: tuple[str, ...], *,
                  topology: str, workers: tuple[str, ...],
                  hook_contracts: dict[str, HookContract] | None = None):
        if not {h.target for h in harness.hooks} <= set(exposed_targets):
            raise ValueError("harness changes an interface outside the fixed workflow")
        if not set(hook_contracts or {}) <= set(exposed_targets):
            raise ValueError("native hook contracts must belong to exposed targets")
        self.hooks = HookRuntime(harness, contracts=hook_contracts)
        self.exposed_targets = exposed_targets
        self.trace = ExecutionTrace(topology=topology, workers=workers)

    def hook(self, target: str, context: dict, *, parents: tuple[str, ...] = (),
             actor: str = "", round_number: int = 0) -> Observation:
        if target not in self.exposed_targets:
            raise ValueError("hook is not exposed by this workflow")
        try:
            value = self.hooks.call(target, context)
        except HookError as exc:
            self.trace.record(target, TARGETS[target].level, context,
                {"error_type": type(exc).__name__, "error": str(exc)},
                parents=parents, actor=actor, round_number=round_number)
            exc.execution = self.finish("", task_error=str(exc))
            raise
        event_id = self.trace.record(target, TARGETS[target].level, context, value,
            parents=parents, actor=actor, round_number=round_number)
        return Observation(event_id, value)

    def finish(self, answer: str, **execution_fields) -> Execution:
        return Execution(answer=answer, events=self.trace.events, **execution_fields)

    def node_action(self, *, task: dict, contract: dict, assignment: dict,
                    tool: str, arguments: dict, invoke: Callable[[str, dict], dict],
                    parents: tuple[str, ...] = (), actor: str = "", round_number: int = 0) -> Observation:







        supplied = deepcopy(arguments)
        decision = self.hook("prepare_node_action", {
            "task": task, "contract": contract, "assignment": assignment,
            "action": {"tool": tool, "arguments": supplied},
        }, parents=parents, actor=actor, round_number=round_number)
        effective = deepcopy(decision.value.get("arguments", supplied))
        inputs = {"tool": tool, "arguments": effective}
        if not decision.value.get("allow", True):
            output = {"blocked": True, "reason": decision.value.get("reason", "")}
            event_id = self.trace.record("node_action_blocked", "node", inputs, output,
                parents=(decision.event_id,), actor=actor, round_number=round_number)
            return Observation(event_id, output)
        input_id = self.trace.record("tool_input", "node", inputs, {},
            parents=(decision.event_id,), actor=actor, round_number=round_number)
        try:
            output = invoke(tool, deepcopy(effective))
        except Exception as exc:
            self.trace.record("tool_error", "node", {}, {"error_type": type(exc).__name__},
                parents=(input_id,), actor=actor, round_number=round_number)
            raise
        event_id = self.trace.record("tool_output", "node", {}, output,
            parents=(input_id,), actor=actor, round_number=round_number)
        return Observation(event_id, output)

    def consume_deliveries(self, *, messages: list[dict], deliveries: tuple[Delivery, ...],
                          invoke: Callable[[list[dict]], dict], actor: str, level: Level,
                          round_number: int, parents: tuple[str, ...] = ()) -> Observation:





        actual = [*deepcopy(messages), *({"role": "user", "content": d.content} for d in deliveries)]
        return self.trace.model_call(actor=actor, level=level, round_number=round_number,
            messages=actual, invoke=invoke, parents=parents, received=deliveries)

    def worker_request(self, *, task: dict, contract: dict, assignment: dict,
                       messages: list[dict], invoke: Callable[[list[dict]], dict],
                       actor: str, round_number: int, parents: tuple[str, ...] = (),
                       received: tuple[Delivery, ...] = ()) -> Observation:





        if actor not in self.trace.workers:
            raise ValueError("request recipient must be an existing worker")
        decision = self.hook("prepare_worker_request",
            {"task": task, "contract": contract, "assignment": assignment},
            parents=parents, actor=actor, round_number=round_number)
        actual = deepcopy(messages)
        if decision.value.get("guidance"):
            actual.append({"role": "user", "content": decision.value["guidance"]})
        return self.consume_deliveries(messages=actual, deliveries=received, invoke=invoke,
            actor=actor, level="node", round_number=round_number, parents=(decision.event_id,))

    def handoff(self, *, task: dict, contract: dict, assignment: dict, result: dict,
                native_envelope: dict, source_event: str, sender: str,
                recipients: tuple[str, ...], round_number: int) -> Delivery:






        decision = self.hook("normalize_worker_handoff",
            {"task": task, "contract": contract, "assignment": assignment, "result": result},
            parents=(source_event,), actor=sender, round_number=round_number)
        effective = {**deepcopy(native_envelope), **decision.value}
        content = json.dumps(effective, ensure_ascii=False, allow_nan=False, sort_keys=True)
        event_id = self.trace.record("handoff_output", "edge", {"native_envelope": native_envelope},
            {"envelope": effective, "content": content}, parents=(decision.event_id,),
            actor=sender, round_number=round_number)
        return self.trace.deliver(sender=sender, recipients=recipients, content=content,
                                 source_event=event_id, round_number=round_number)

    def graph_readout(self, *, task: dict, contract: dict, native_resolution: dict,
                      messages: list[dict], deliveries: tuple[Delivery, ...],
                      invoke: Callable[[list[dict]], dict], actor: str,
                      round_number: int, parents: tuple[str, ...] = ()) -> Observation:





        envelopes = [json.loads(d.content) for d in deliveries]
        if any(not isinstance(e, dict) for e in envelopes):
            raise ValueError("graph_readout requires delivered JSON object envelopes")
        causes = (*parents, *(d.event_id for d in deliveries))
        resolution = deepcopy(native_resolution)
        if "resolve_edge_semantics" in self.exposed_targets:
            observed = self.hook("resolve_edge_semantics",
                {"task": task, "contract": contract, "envelopes": envelopes},
                parents=causes, actor=actor, round_number=round_number)
            resolution.update(observed.value)
            causes = (*causes, observed.event_id)
        actual = deepcopy(messages)
        if resolution:
            actual.append({"role": "user", "content": json.dumps(
                {"coordination": resolution}, ensure_ascii=False, allow_nan=False, sort_keys=True)})
        if "plan_graph_readout" in self.exposed_targets:
            observed = self.hook("plan_graph_readout",
                {"task": task, "contract": contract, "envelopes": envelopes, "resolution": resolution},
                parents=causes, actor=actor, round_number=round_number)
            causes = (*causes, observed.event_id)
            if observed.value.get("guidance"):
                actual.append({"role": "user", "content": observed.value["guidance"]})
        return self.consume_deliveries(messages=actual, deliveries=deliveries, invoke=invoke,
            actor=actor, level="graph", round_number=round_number, parents=causes)

    def orchestrate(self, *, task: dict, contract: dict, round_index: int, envelopes: list[dict],
                    remaining_budget: dict, native_decision: dict, eligible_workers: tuple[str, ...],
                    may_continue: bool, invoke: Callable[[dict, str], dict], actor: str,
                    parents: tuple[str, ...] = ()) -> Observation:







        if self.trace.topology == "independent":
            raise ValueError("independent execution has no intermediate team scheduling")
        if not set(eligible_workers) <= set(self.trace.workers):
            raise ValueError("eligible follow-ups must be existing workers")
        if native_decision.get("action") not in ("continue", "stop"):
            raise ValueError("native decision must resolve to continue or stop")
        decision = self.hook("decide_orchestration", {
            "task": task, "contract": contract, "round_index": round_index,
            "envelopes": envelopes, "remaining_budget": remaining_budget},
            parents=parents, actor=actor, round_number=round_index)
        effective = {**deepcopy(native_decision), **decision.value}
        if effective["action"] == "native":
            effective["action"] = native_decision["action"]
        targets = effective.get("target_workers", [])
        if not isinstance(targets, list) or any(not isinstance(w, str) for w in targets):
            raise ValueError("native target_workers must be a list of worker names")
        budget_stop = effective["action"] == "continue" and not may_continue
        if budget_stop:
            effective["action"] = "stop"
        if effective["action"] == "stop":
            effective["target_workers"] = []
        if not set(effective.get("target_workers", [])) <= set(eligible_workers):
            error = "orchestration selects a worker outside the native eligible follow-ups"
            self.trace.record("orchestration_error", "graph", {"decision": effective},
                {"error": error}, parents=(decision.event_id,), actor=actor, round_number=round_index)
            raise HookError(error, execution=self.finish("", task_error=error))
        input_id = self.trace.record("orchestration_input", "graph",
            {"decision": effective, "native_decision": native_decision,
             "eligible_workers": list(eligible_workers), "budget_stop": budget_stop}, {},
            parents=(decision.event_id,), actor=actor, round_number=round_index)
        try:
            output = invoke(deepcopy(effective), input_id)
        except Exception as exc:
            self.trace.record("orchestration_error", "graph", {}, {"error_type": type(exc).__name__},
                parents=(input_id,), actor=actor, round_number=round_index)
            raise
        event_id = self.trace.record("orchestration_output", "graph", {}, output,
            parents=(input_id,), actor=actor, round_number=round_index)
        return Observation(event_id, output)
