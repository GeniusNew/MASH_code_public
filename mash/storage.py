from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4


class Store:
    def __init__(self, root: Path):
        self.root = root
        self._candidate_submissions = {}
        root.mkdir(parents=True, exist_ok=False)

    def append(self, kind: str, **fields):
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, **fields},
                                    ensure_ascii=False, allow_nan=False) + "\n")

    def submit_candidate(self, candidate, round_number: int, *, fresh=False):
        version_id = candidate.submission_version_id
        content = candidate.model_dump(exclude={"candidate_id", "submission_version_id"})
        previous = self._candidate_submissions.get(version_id)
        if not fresh and previous is not None and previous[0] == round_number:
            if previous[1] != content:
                raise ValueError("a submitted implementation cannot change without a new version")
            return candidate
        version_id = uuid4().hex
        candidate = candidate.model_copy(update={"submission_version_id": version_id})
        self._candidate_submissions[version_id] = (round_number, content)
        self.append("candidate_submitted", round_number=round_number,
                    candidate_id=candidate.candidate_id, version_id=version_id,
                    candidate=candidate.model_dump())
        return candidate

    def preflight_rejections(self, round_number: int) -> list[dict]:
        path = self.root / "events.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        candidates, rejections = {}, {}
        for event in events:
            if event.get("round_number") != round_number:
                continue
            if event["kind"] == "candidate_submitted":
                candidates[event["version_id"]] = event["candidate"]
            elif event["kind"] == "candidate_preflight_result":
                rejections[event["version_id"]] = event
        return [{"version_id": version_id, "candidate": candidates[version_id],
                 "rejected_at": event["rejected_at"], "errors": event["errors"]}
                for version_id, event in rejections.items() if event["rejected_at"] is not None]
