
from .models import DiagnosisBatch, FreeformBatch, FailureGroup, InformationGap, Trajectory


class UnknownTrajectoryEvent(ValueError):
    pass


def organize(batch: DiagnosisBatch, trajectories: tuple[Trajectory, ...]) -> tuple[FailureGroup, ...]:
    events = {(t.run_id, e.event_id): e for t in trajectories for e in t.execution.events}
    records = {r.diagnosis_id: r for r in batch.records}
    if len(records) != len(batch.records):
        raise ValueError("diagnosis IDs must be unique")
    if isinstance(batch, FreeformBatch):
        for record in batch.records:
            if any((ref.run_id, ref.event_id) not in events for ref in record.evidence):
                raise UnknownTrajectoryEvent("diagnosis cites an unknown trajectory event")
        return tuple(FailureGroup(group_id=f"g{i}", root_cause=r.analysis, records=(r,))
                     for i, r in enumerate(batch.records))

    def key(ref):
        return ref.run_id, ref.event_id

    for record in batch.records:
        refs = (*record.evidence, record.origin, record.decision_event, *record.causal_path)
        if any(key(ref) not in events for ref in refs):
            raise UnknownTrajectoryEvent("diagnosis cites an unknown trajectory event")
        if record.causal_path[0] != record.origin or record.causal_path[-1] != record.decision_event:
            raise ValueError("causal path must run from origin to failed decision")
        if record.origin not in record.evidence or record.decision_event not in record.evidence:
            raise ValueError("evidence must include origin and failed decision")
        for before, after in zip(record.causal_path, record.causal_path[1:]):
            if before.run_id != after.run_id or before.event_id not in events[key(after)].parents:
                raise ValueError("causal path must follow recorded information-flow edges")

    used: set[str] = set()
    groups = []
    for claim in batch.merges:
        ids = set(claim.diagnosis_ids)
        if len(ids) != len(claim.diagnosis_ids) or not ids <= records.keys() or used & ids:
            raise ValueError("merge claims must contain unique, disjoint, existing diagnoses")
        origins = {key(records[i].origin) for i in ids}
        cited = {key(ref) for ref in claim.evidence}
        if not origins <= cited or not cited <= events.keys():
            missing = sorted(origins - cited)
            unknown = sorted(cited - events.keys())
            raise ValueError("merge evidence must cover each claimed root origin; "
                             f"diagnosis_ids={claim.diagnosis_ids!r}; "
                             f"missing_origins={missing!r}; unknown_events={unknown!r}. "
                             "Correct these references or omit this merge while retaining "
                             "individually supported diagnoses.")
        consolidated_evidence = {key(ref) for ref in claim.consolidated.evidence}
        if not origins <= consolidated_evidence or not consolidated_evidence <= cited:
            raise ValueError("consolidated evidence must preserve origins and use the merge evidence; "
                             f"diagnosis_ids={claim.diagnosis_ids!r}; "
                             f"missing_origins={sorted(origins - consolidated_evidence)!r}; "
                             f"outside_merge_evidence={sorted(consolidated_evidence - cited)!r}. "
                             "Correct consolidated.evidence while preserving the supported records.")


        ordered = tuple(records[i] for i in claim.diagnosis_ids)
        groups.append(FailureGroup(group_id=f"g{len(groups)}", root_cause=claim.common_root_cause,
                                   records=ordered, merge_justification=claim.justification,
                                   consolidated=claim.consolidated))
        used.update(ids)
    for record in batch.records:
        if record.diagnosis_id not in used:
            groups.append(FailureGroup(group_id=f"g{len(groups)}", root_cause=record.root_cause,
                                       records=(record,), consolidated=InformationGap.model_validate(
                                           {k: getattr(record, k) for k in InformationGap.model_fields})))
    return tuple(groups)
