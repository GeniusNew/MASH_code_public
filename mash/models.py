from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

Level = Literal["node", "edge", "graph"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


class TaskView(Model):
    task_id: str
    payload: dict[str, Any]


class Task(TaskView):
    reference: Any
    stratum: str | None = None

    def public(self) -> TaskView:
        return TaskView(task_id=self.task_id, payload=self.model_dump()["payload"])


class TaskSplit(Model):
    diagnosis: tuple[str, ...]
    small: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    @model_validator(mode="after")
    def check_partitions(self):
        for name in ("diagnosis", "small", "validation", "test"):
            ids = getattr(self, name)
            if not ids or len(ids) != len(set(ids)):
                raise ValueError(f"{name} must be nonempty and contain unique task IDs")
        dis, small, val, test = map(set, (self.diagnosis, self.small, self.validation, self.test))
        if not small <= dis:
            raise ValueError("small must be a fixed subset of diagnosis")
        if dis & val or dis & test or val & test:
            raise ValueError("diagnosis, validation and test must be disjoint")
        return self


class SearchConfig(Model):
    variant: Literal["F", "U", "O", "R", "L"] = "F"
    filtering_stages: tuple[Literal[1, 2, 3], ...] = (1, 2, 3)
    editable_levels: tuple[Level, ...] = ("node", "edge", "graph")
    initialization_repetitions: int = Field(default=1, ge=1)
    rounds: int = Field(default=20, ge=1)
    theta_eff: float = Field(default=0.005, ge=-1, le=1)
    theta_gen: float = Field(default=0.0025, ge=-1, le=1)
    normal_repetitions: int = Field(default=2, ge=1)
    max_candidates: int = Field(default=3, ge=1)
    analysis_calls: int = Field(default=11, ge=1)
    generation_calls: int = Field(default=13, ge=1)
    optimizer_input_max_bytes: int = Field(default=120000, ge=16000)
    seed: int = 0

    @model_validator(mode="after")
    def valid_experiment(self):
        if tuple(sorted(set(self.filtering_stages))) != self.filtering_stages:
            raise ValueError("filtering stages must be unique and ordered")
        if not self.editable_levels or len(set(self.editable_levels)) != len(self.editable_levels):
            raise ValueError("editable levels must be nonempty and unique")
        return self


class Hook(Model):
    target: str
    source: str


class Harness(Model):
    hooks: tuple[Hook, ...] = ()

    @model_validator(mode="after")
    def unique_targets(self):
        if len({h.target for h in self.hooks}) != len(self.hooks):
            raise ValueError("duplicate harness targets")
        return self

    @property
    def harness_id(self) -> str:
        return digest(sorted((h.target, h.source) for h in self.hooks))[:16]

    def apply(self, patches: tuple[Hook, ...]) -> Harness:
        merged = {h.target: h for h in self.hooks}
        merged.update({h.target: h for h in patches})
        return Harness(hooks=tuple(merged[k] for k in sorted(merged)))


class Event(Model):
    event_id: str
    target: str
    level: Level
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    parents: tuple[str, ...] = ()
    actor: str = ""
    round_number: int = 0


class Execution(Model):
    answer: str
    events: tuple[Event, ...]
    task_error: str | None = None
    scoring_error: str | None = None
    native_score: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def ordered_dag(self):
        if self.scoring_error is not None and self.native_score is not None:
            raise ValueError("invalid grading cannot produce a native score")
        seen: set[str] = set()
        for event in self.events:
            if event.event_id in seen or not set(event.parents) <= seen:
                raise ValueError("events must form an ordered DAG with unique IDs")
            seen.add(event.event_id)
        return self


class Trajectory(Model):
    run_id: str
    task: TaskView
    harness_id: str
    score: float = Field(ge=0, le=1)
    execution: Execution
    error: str | None = None


class EvidenceRef(Model):
    run_id: str
    event_id: str


class InformationGap(Model):
    decision: str = Field(min_length=1)
    required_information: tuple[str, ...]
    received_information: tuple[str, ...]
    missing_information: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)


class Diagnosis(InformationGap):
    diagnosis_id: str
    origin: EvidenceRef
    decision_event: EvidenceRef
    causal_path: tuple[EvidenceRef, ...] = Field(min_length=1)
    root_cause: str = Field(min_length=1)


class MergeClaim(Model):
    diagnosis_ids: tuple[str, ...] = Field(min_length=2)
    common_root_cause: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    consolidated: InformationGap


class RepairReview(Model):
    version_id: str = Field(min_length=1)
    status: Literal["addressed", "persists", "changed", "unverified"]
    reason: str = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = ()


class DiagnosisBatch(Model):
    records: tuple[Diagnosis, ...] = ()
    merges: tuple[MergeClaim, ...] = ()
    repair_reviews: tuple[RepairReview, ...] = ()


class FreeformRecord(Model):
    diagnosis_id: str
    analysis: str = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    level: Level | None = None


class FreeformBatch(Model):
    records: tuple[FreeformRecord, ...] = ()
    repair_reviews: tuple[RepairReview, ...] = ()


class FailureGroup(Model):
    group_id: str
    root_cause: str
    records: tuple[Diagnosis | FreeformRecord, ...]
    merge_justification: str = ""
    consolidated: InformationGap | None = None

    def generation_view(self) -> dict:

        if self.consolidated is None:
            return self.model_dump()
        return {"group_id": self.group_id, "root_cause": self.root_cause,
                "diagnosis": self.consolidated.model_dump(),
                "merge_justification": self.merge_justification,
                "propagation": [{"diagnosis_id": r.diagnosis_id,
                                 "origin": r.origin.model_dump(),
                                 "decision_event": r.decision_event.model_dump(),
                                 "causal_path": [e.model_dump() for e in r.causal_path]}
                                for r in self.records]}


class BehaviorAssertion(Model):
    field: str = Field(min_length=1)
    operator: Literal["equals", "nonempty", "contains_all", "omitted", "length_at_most"]
    value: Any = Field(default=None, description=(
        "contains_all requires a nonempty list of strings, never objects; length_at_most requires "
        "a nonnegative integer; nonempty and omitted require null. equals may compare structured values."))

    @model_validator(mode="after")
    def value_matches_operator(self):
        if self.operator == "contains_all" and (not isinstance(self.value, list)
                or not self.value or not all(isinstance(item, str) for item in self.value)):
            raise ValueError("contains_all requires a nonempty list of strings")
        if self.operator == "length_at_most" and (not isinstance(self.value, int) or self.value < 0):
            raise ValueError("length_at_most requires a nonnegative integer")
        if self.operator in ("nonempty", "omitted") and self.value is not None:
            raise ValueError(f"{self.operator} does not accept a value")
        return self


class BehaviorCase(Model):
    description: str = Field(min_length=1)
    context: dict[str, Any] | None = None
    expected: dict[str, Any] | None = None
    assertions: tuple[BehaviorAssertion, ...] = ()
    source: EvidenceRef | None = None
    preserve_parent: bool = False

    @model_validator(mode="after")
    def input_is_available(self):
        if self.context is None and self.source is None:
            raise ValueError("behavior case requires context or an original event source")
        if self.expected is None and not self.assertions and not self.preserve_parent:
            raise ValueError("behavior case requires expected output, assertions, or parent preservation")
        return self


class PlannedChange(Model):
    target: str
    level: Level
    behavior: str = Field(min_length=1)
    applicability: str = Field(min_length=1)
    required: bool = True


class Change(PlannedChange):
    behavior_cases: tuple[BehaviorCase, ...] = ()


class RevisionPlan(Model):
    parent_id: str
    group_id: str
    hypothesis: str = Field(min_length=1)
    changes: tuple[PlannedChange, ...] = Field(min_length=1)


class PlannedCandidate(RevisionPlan):
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")


class PlanBatch(Model):
    plans: tuple[PlannedCandidate, ...] = ()


class Candidate(PlannedCandidate):
    changes: tuple[Change, ...] = Field(min_length=1)
    patches: tuple[Hook, ...] = Field(min_length=1)
    frozen_plan: SkipJsonSchema[RevisionPlan | None] = None
    submission_version_id: SkipJsonSchema[str | None] = None

    def revision_plan(self) -> RevisionPlan:
        return RevisionPlan(parent_id=self.parent_id, group_id=self.group_id,
            hypothesis=self.hypothesis, changes=tuple(PlannedChange.model_validate(
                change.model_dump(exclude={"behavior_cases"})) for change in self.changes))


class CandidateBatch(Model):
    candidates: tuple[Candidate, ...] = ()


class ImplementationPatch(Hook):
    behavior_cases: tuple[BehaviorCase, ...] = ()


class CandidateImplementation(Model):
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    patches: tuple[ImplementationPatch, ...] = Field(min_length=1)

    def bind(self, plan: RevisionPlan) -> Candidate:
        cases = {patch.target: patch.behavior_cases for patch in self.patches}
        return Candidate(candidate_id=self.candidate_id,
            **plan.model_dump(exclude={"changes"}),
            changes=tuple(Change(**change.model_dump(), behavior_cases=cases.get(change.target, ()))
                          for change in plan.changes),
            patches=tuple(Hook(target=patch.target, source=patch.source) for patch in self.patches))


class ImplementationBatch(Model):
    implementations: tuple[CandidateImplementation, ...] = ()


class PairResult(Model):
    parent: tuple[Trajectory, ...]
    candidate: tuple[Trajectory, ...]
    delta: float


class FrozenHarness(Model):
    harness: Harness
    fixed_system_id: str
    split: TaskSplit
    config: SearchConfig
    rounds_completed: int
    selected_candidates: tuple[str, ...]
