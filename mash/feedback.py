
from collections import defaultdict
from decimal import Decimal

from .models import Candidate, FailureGroup, PairResult
from .effects import hook_effects


def repair_feedback(candidate: Candidate, submission: dict, pair: PairResult, *, parent=None) -> dict:
    group = next(FailureGroup.model_validate(g) for g in submission["failure_groups"]
                 if g["group_id"] == candidate.group_id)
    source_tasks = {row["run_id"]: row["task_id"] for row in submission["diagnosis_runs"]}
    diagnosed_tasks = sorted({source_tasks[ref.run_id] for record in group.records for ref in record.evidence})
    paired = defaultdict(list)
    for original, child in zip(pair.parent, pair.candidate, strict=True):
        paired[original.task.task_id].append((original, child))
    effects = []
    for task_id, runs in paired.items():
        parent_mean = sum((Decimal(str(p.score)) for p, _ in runs), Decimal(0)) / len(runs)
        candidate_mean = sum((Decimal(str(c.score)) for _, c in runs), Decimal(0)) / len(runs)
        effects.append({"task_id": task_id, "repetitions": len(runs),
                        "parent_mean": float(parent_mean), "candidate_mean": float(candidate_mean),
                        "gain": float(candidate_mean - parent_mean),
                        "diagnosis_evidence_task": task_id in diagnosed_tasks,
                        "parent_runs": [p.run_id for p, _ in runs],
                        "candidate_runs": [c.run_id for _, c in runs]})
    measured = {row["task_id"] for row in effects}
    return {"version_id": candidate.submission_version_id or f'{submission["round"]}:{candidate.candidate_id}',
            **({"hook_effects": hook_effects(pair, candidate, parent)} if parent is not None else {}),
            "diagnosis_round": submission["round"], "failure": group.generation_view(),
            "hypothesis": candidate.hypothesis,
            "intended_changes": [change.model_dump() for change in candidate.changes],
            "mechanism_evidence": {
                patch.target: {
                    side: [{"run_id": tr.run_id, "event_id": event.event_id,
                            "task_id": tr.task.task_id}
                           for tr in runs for event in tr.execution.events if event.target == patch.target]
                    for side, runs in (("parent", pair.parent), ("candidate", pair.candidate))}
                for patch in candidate.patches},
            "diagnosed_tasks": diagnosed_tasks,
            "unmeasured_diagnosed_tasks": sorted(set(diagnosed_tasks) - measured),
            "task_effects": effects,
            "improved_tasks": [row["task_id"] for row in effects if row["gain"] > 0],
            "regressed_tasks": [row["task_id"] for row in effects if row["gain"] < 0]}


def reviewable_repairs(current_evaluation: dict | None, level=None) -> dict[str, dict]:

    return {row["repair_feedback"]["version_id"]: row
            for row in (current_evaluation or {}).get("normal_evaluations", ())
            if row.get("complete") and row.get("repair_feedback") and
            (level is None or any(c["level"] == level
                                 for c in row["repair_feedback"]["intended_changes"]))}


def validate_repair_reviews(reviews, current_evaluation, trajectories, *, level=None):

    expected = reviewable_repairs(current_evaluation, level)
    if len(reviews) != len(expected) or {r.version_id for r in reviews} != set(expected):
        raise ValueError("provide one repair_review for each completed normal comparison in this session; "
                         f"expected_version_ids={sorted(expected)!r}; "
                         f"received_version_ids={[r.version_id for r in reviews]!r}")
    events = {tr.run_id: {e.event_id for e in tr.execution.events} for tr in trajectories}
    for review in reviews:
        row = expected[review.version_id]
        child_runs = set(row["candidate_runs"])
        paired_runs = child_runs | set(row["parent_runs"])
        if any(ref.run_id not in paired_runs or ref.event_id not in events.get(ref.run_id, ())
               for ref in review.evidence):
            raise ValueError("repair_review evidence must cite events from that repair's normal comparison")
        if review.status != "unverified" and not any(ref.run_id in child_runs for ref in review.evidence):
            raise ValueError("a verified repair_review requires candidate event evidence, not score alone")
