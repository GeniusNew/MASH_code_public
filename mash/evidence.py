
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

from .llm import ModelResponseError
from .models import Trajectory


EVIDENCE_PROMPT = '''
$evidence preserves complete original content. Read it with
{"read_evidence":[{"ref":"ID","offset":0,"length":12000}],"notes":"optional"}.
Current trajectory references include task/answer previews and the final recorded event.
Use these to select substantive-answer failures alongside execution signals; read the original
events and complete values before diagnosing a cause. Previews do not count as event reads.
Read an exact event with {"read_evidence":[{"run_id":"RUN","event_id":"EVENT"}]}.
For first inspection, add "view":"indexed" to expose event fields, direct_children provenance,
and long values stored as nested $evidence references. Follow relevant references before deriving behavior or claiming an
omission. The default "view":"raw" returns the complete original event. Offsets belong to the
chosen view; keep that view when following next_offset.
Use a $evidence value in ref, not event_id. ref reads return the referenced document as stored;
scope applies to run queries and view to exact events.
Current diagnosis, paired-parent and supplied Stage-2 repair events are readable. Read prior events
with "scope":"history" and Stage-3 validation with "scope":"screening". These support comparison
only; formal diagnoses cite current diagnosis trajectories. Test trajectories are unavailable.
Use direct IDs from decision_boundaries, execution_signals and contrast_runs before listing a run.
When a hook is known, list it with {"read_evidence":[{"run_id":"RUN","target":"HOOK"}]}.
Use target:"*" only when locating an unknown boundary; event_id must identify a recorded event.
Reads use character offsets and session requests. Follow next_offset and nested references; request
several together when useful. Partial or unread evidence cannot establish an omission. Follow event
parents and inspect complete decision inputs. Reuse evidence_reads and retained_evidence_reads.
pending_evidence_reads lists unreturned ranges; continue those instead of repeating the batch.
Requests are stateless: save observed values, exact IDs, parent edges and conclusions in notes
(8000 UTF-8 bytes), not reading instructions. Notes are not evidence.
session_requests_remaining includes this call. When it is 1, return the supplied result schema,
not another read request. Return an empty result if the available evidence cannot support a proposal.
'''


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def page_key(page):
    return (page.get("scope", "current"), page.get("ref"), page.get("run_id"),
            page.get("event_id"), page.get("target"), page.get("view", "raw"),
            page["offset"], len(page["content"]))


EVIDENCE_FIELDS = ("trajectories", "history", "current_evaluation", "filter_feedback",
                   "failures", "current_harness", "runtime_context", "behavior_sources",
                   "contrast_runs", "parent_behavior_examples", "screening_contrasts", "candidate_preflight_feedback",
                   "rejected_diagnosis", "execution_signals", "failure_target_index",
                   "failure_target_context", "handoff_propagation_index", "decision_boundaries", "targets",
                   "frozen_plans", "replanning_schema", "implementation_context", "rejected_response")

READ_CATALOG_FIELDS = ("targets", "runtime_context", "current_harness", "failure_target_context",
                       "behavior_sources", "contrast_runs", "parent_behavior_examples", "screening_contrasts")


class EvidenceSession:
    def __init__(self, payload, max_bytes, *, trajectories: tuple[Trajectory, ...] = (),
                 history_trajectories: Mapping[str, Trajectory] | None = None,
                 screening_trajectories: Mapping[str, Trajectory] | None = None):
        self.max_bytes = max_bytes
        self.documents = {}
        self.read_log = []
        self.read_counts = {}
        self.trajectories = {t.run_id: t for t in trajectories}
        self.events = {(t.run_id, e.event_id): (t, e)
                       for t in trajectories for e in t.execution.events}
        self.history_trajectories = dict(history_trajectories or {})
        self.screening_trajectories = dict(screening_trajectories or {})
        self.payload = payload.copy()
        if len(wire(payload).encode("utf-8")) > max_bytes // 2:
            for key in EVIDENCE_FIELDS:
                if key in payload:
                    value = payload[key]
                    if key in ("targets", "failure_target_context") and isinstance(value, dict):
                        self.payload[key] = {name: self._pack(spec) for name, spec in value.items()}
                    else:
                        self.payload[key] = ([self._pack(v) for v in value]
                                             if isinstance(value, list) else self._pack(value))

    def set_context(self, name, context):
        self.payload[name] = {target: self._pack(row) for target, row in context.items()}

    def _pack(self, value):
        if isinstance(value, dict):
            packed = {k: self._pack(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            packed = [self._pack(v) for v in value]
        else:
            packed = value
        if (isinstance(packed, dict) and "execution" in packed and "task" in packed
                and isinstance(packed.get("run_id"), str) and packed["run_id"] in self.trajectories):
            return self._reference(packed)
        content = packed if isinstance(packed, str) else wire(packed)
        threshold = 1024 if isinstance(packed, str) else 6000
        if len(content.encode("utf-8")) <= threshold:
            return packed
        return self._reference(packed)

    def _reference(self, value):
        if isinstance(value, dict) and "$evidence" in value:
            return value
        kind = "text" if isinstance(value, str) else "json"
        content = value if kind == "text" else wire(value)
        ref = hashlib.sha256((kind + ":" + content).encode("utf-8")).hexdigest()[:24]
        self.documents[ref] = {"kind": kind, "content": content}
        result = {"$evidence": ref, "kind": kind, "characters": len(content)}
        if isinstance(value, dict):
            for key in ("run_id", "event_id", "task_id", "harness_id", "group_id", "diagnosis_id",
                        "candidate_id", "version_id", "parent_id", "score", "target", "level", "parents", "round",
                        "decision_event_id",
                        "round_index", "remaining_budget", "normal_summary"):
                if key in value:
                    result[key] = value[key]
            if "execution" in value and "task" in value and isinstance(value.get("run_id"), str):
                trajectory = self.trajectories.get(value["run_id"])
                if trajectory is not None:
                    result.update({
                        "task_id": trajectory.task.task_id,
                        "task_preview": wire(trajectory.task.payload).encode("utf-8")[:300].decode("utf-8", "ignore"),
                        "answer_preview": trajectory.execution.answer.encode("utf-8")[:600].decode("utf-8", "ignore"),
                        "answer_characters": len(trajectory.execution.answer)})
                    if trajectory.execution.events:
                        event = trajectory.execution.events[-1]
                        result["terminal_event"] = {"event_id": event.event_id, "target": event.target}
        elif isinstance(value, (list, tuple)):
            result["items"] = len(value)
        return result

    def _compact(self, prompt, *, reserve=0):

        prompt_bytes = len(prompt.encode("utf-8"))
        target = self.max_bytes - prompt_bytes - reserve
        size = lambda: len(wire(self.payload).encode("utf-8"))
        retained = self.payload.get("retained_evidence_reads", [])
        while retained and size() > target:
            retained.pop()
        keys = sorted((k for k in EVIDENCE_FIELDS if k in self.payload),
                      key=lambda k: (k == "trajectories",
                                     -len(wire(self.payload[k]).encode("utf-8"))))
        for items_first in (True, False):
            for key in keys:
                if size() <= target:
                    break
                value = self.payload[key]
                if items_first:
                    if not isinstance(value, list):
                        continue
                    replacement = []
                    for item in value:
                        ref = self._reference(item)
                        replacement.append(ref if len(wire(ref).encode("utf-8")) < len(wire(item).encode("utf-8")) else item)
                else:
                    replacement = self._reference(value)
                if len(wire(replacement).encode("utf-8")) < len(wire(value).encode("utf-8")):
                    self.payload[key] = replacement
        if prompt_bytes + size() > self.max_bytes:
            raise ValueError("optimizer prompt and metadata exceed input budget after evidence indexing")

    def request(self, prompt, remaining):
        self.payload["session_requests_remaining"] = remaining
        for key in ("format_error", "format_or_evidence_error"):
            if key in self.payload:
                self.payload[key] = self.payload[key][:4000]
        self._compact(prompt)
        return self.payload

    def _compact_read_catalogs(self, prompt):
        target = self.max_bytes - len(prompt.encode("utf-8")) - 4096
        fields = sorted((key for key in READ_CATALOG_FIELDS if key in self.payload),
                        key=lambda key: -len(wire(self.payload[key]).encode("utf-8")))
        for key in fields:
            if len(wire(self.payload).encode("utf-8")) <= target:
                break
            value = self.payload[key]
            reference = self._reference(value)
            if len(wire(reference).encode("utf-8")) < len(wire(value).encode("utf-8")):
                self.payload[key] = reference

    def read(self, value, prompt):
        if not isinstance(value, dict) or "read_evidence" not in value:
            return False
        queries = value["read_evidence"]
        if not isinstance(queries, list) or not queries:
            raise ModelResponseError("read_evidence must be a nonempty list")
        notes = value.get("notes", self.payload.get("working_notes", ""))
        if not isinstance(notes, str) or len(notes.encode("utf-8")) > 8000:
            raise ModelResponseError("working notes must be a string of at most 8000 UTF-8 bytes")
        self.payload["working_notes"] = notes
        previous = [*self.payload.get("evidence_reads", []),
                    *self.payload.get("retained_evidence_reads", [])]
        self.payload["evidence_reads"] = []
        self.payload["retained_evidence_reads"] = []
        queries = sorted(queries, key=lambda query: isinstance(query, dict) and "run_id" in query
                         and "event_id" not in query and "ref" not in query)
        pending = []
        self.payload["pending_evidence_reads"] = queries
        self.payload.pop("format_error", None)
        self.payload.pop("format_or_evidence_error", None)
        self._compact(prompt, reserve=self.max_bytes // 2)
        previous.sort(key=lambda page:
                      -self.read_counts.get(page_key(page)[:-2], 0) / len(wire(page).encode("utf-8")))
        retention_budget = max(0, (self.max_bytes - len(prompt.encode("utf-8"))
                                  - len(wire(self.payload).encode("utf-8")) - 4096) // 4)
        for page in previous:
            size = len(wire(page).encode("utf-8"))
            if size <= retention_budget:
                self.payload["retained_evidence_reads"].append(page)
                retention_budget -= size
        for index, query in enumerate(queries):
            self.payload["pending_evidence_reads"] = [*pending, *queries[index + 1:]]
            metadata = {}
            if not isinstance(query, dict):
                raise ModelResponseError("evidence query must be an object")
            scope = query.get("scope", "current")
            if scope not in ("current", "history", "screening"):
                raise ModelResponseError("evidence scope must be current, history or screening")
            view = query.get("view", "raw")
            if view not in ("raw", "indexed"):
                raise ModelResponseError("event view must be raw or indexed")
            if "ref" in query:
                ref = query["ref"]
                if not isinstance(ref, str) or ref not in self.documents:
                    raise ModelResponseError("unknown evidence reference")
                document = self.documents[ref]
                identity = {"ref": ref}
            elif "event_id" not in query and isinstance(query.get("run_id"), str):
                run_id = query["run_id"]
                source = ({"current": self.trajectories, "history": self.history_trajectories,
                           "screening": self.screening_trajectories}[scope])
                trajectory = source.get(run_id)
                if trajectory is None:
                    raise ModelResponseError("unknown run in this session's supplied execution evidence")
                target = query.get("target")
                if target is not None and not isinstance(target, str):
                    raise ModelResponseError("event index target must be a string")
                if target in ("*", "events"):
                    target = None
                identity = {"run_id": run_id, "scope": scope, "target": target}
                document = {"kind": "json", "content": wire({**identity,
                    "task_id": trajectory.task.task_id, "harness_id": trajectory.harness_id,
                    "events": [{"event_id": event.event_id, "target": event.target,
                                "level": event.level, "actor": event.actor, "parents": event.parents}
                               for event in trajectory.execution.events if target is None or event.target == target]})}
            else:
                run_id, event_id = query.get("run_id"), query.get("event_id")
                if not isinstance(run_id, str) or not isinstance(event_id, str):
                    raise ModelResponseError(f"event queries require string run_id and event_id: {wire(query)}")
                if scope in ("history", "screening"):
                    source = self.history_trajectories if scope == "history" else self.screening_trajectories
                    trajectory = source.get(run_id)
                    event = next((e for e in trajectory.execution.events if e.event_id == event_id), None) if trajectory else None
                else:
                    trajectory, event = self.events.get((run_id, event_id), (None, None))
                if event is None:
                    if event_id in self.documents:
                        lookup = {"ref": event_id}
                        raise ModelResponseError(
                            f"{event_id!r} is a document reference, not a recorded event in run {run_id!r}; "
                            f"read its contents with {wire({'read_evidence': [lookup]})}")
                    lookup = {"run_id": run_id, "scope": scope, "target": "*"}
                    raise ModelResponseError(
                        f"unknown event {event_id!r} in run {run_id!r} (scope={scope!r}); "
                        f"inspect recorded IDs with {wire({'read_evidence': [lookup]})}")
                metadata = {"target": event.target, "parents": list(event.parents)}
                if event.inputs.get("phase") is not None:
                    metadata["phase"] = event.inputs["phase"]
                identity = {"run_id": run_id, "event_id": event_id}
                if scope != "current":
                    identity["scope"] = scope
                value = {**identity,
                    "harness_id": trajectory.harness_id, "task": trajectory.task.model_dump(),
                    "event": event.model_dump()}
                if view == "indexed":
                    metadata["view"] = view
                    value["task"] = self._pack(value["task"])
                    value["event"] = {key: self._pack(item) for key, item in value["event"].items()}
                    value["direct_children"] = [
                        {"event_id": child.event_id, "target": child.target,
                         "level": child.level, "actor": child.actor, "parents": child.parents}
                        for child in trajectory.execution.events if event_id in child.parents]
                document = {"kind": "json", "content": wire(value)}
            content = document["content"]
            offset, length = query.get("offset", 0), query.get("length", 12000)
            if type(offset) is not int or type(length) is not int or not 0 <= offset < len(content) or length < 1:
                raise ModelResponseError("invalid evidence character range")
            self.payload["pending_evidence_reads"] = [*pending, *queries[index:]]
            requested = min(length, len(content) - offset)
            high = requested
            request_reserve = 4096 if len(content) > 6000 else 64

            def read_value(size):
                end = offset + size
                return {**identity, **metadata, "kind": document["kind"], "offset": offset,
                        "content": content[offset:end], "characters": len(content),
                        "next_offset": end if end < len(content) else None}

            def fits(size):
                payload = {**self.payload,
                           "evidence_reads": [*self.payload["evidence_reads"], read_value(size)]}
                return (len(prompt.encode("utf-8")) + len(wire(payload).encode("utf-8"))
                        <= self.max_bytes - request_reserve)

            if fits(high):
                low = high
            else:
                low, high = 0, high - 1
                while low < high:
                    middle = (low + high + 1) // 2
                    if fits(middle):
                        low = middle
                    else:
                        high = middle - 1
            if not low:
                break
            read = read_value(low)
            self.payload["evidence_reads"].append(read)
            if low < requested:
                pending.append({**query, "offset": offset + low, "length": requested - low})
            self.payload["pending_evidence_reads"] = [*pending, *queries[index + 1:]]
            document_key = page_key(read)[:-2]
            self.read_counts[document_key] = self.read_counts.get(document_key, 0) + 1
            self.read_log.append({key: read[key] for key in
                                  ("run_id", "event_id", "scope", "target", "phase", "ref") if key in read})
        if not self.payload["pending_evidence_reads"]:
            self.payload.pop("pending_evidence_reads")
        seen = {page_key(page) for page in self.payload["evidence_reads"]}
        self.payload["retained_evidence_reads"] = [page for page in self.payload["retained_evidence_reads"]
                                                   if page_key(page) not in seen]
        retained = self.payload["retained_evidence_reads"]
        seen.update(page_key(page) for page in retained)
        for page in previous:
            key = page_key(page)
            if key in seen:
                continue
            seen.add(key)
            retained.append(page)
            if len(prompt.encode("utf-8")) + len(wire(self.payload).encode("utf-8")) > self.max_bytes - 4096:
                self._compact_read_catalogs(prompt)
            if len(prompt.encode("utf-8")) + len(wire(self.payload).encode("utf-8")) > self.max_bytes - 4096:
                retained.pop()
        return True
