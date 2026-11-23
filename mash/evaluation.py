from __future__ import annotations

from decimal import Decimal
from typing import Callable, Protocol

from .hooks import HookError
from .models import Execution, Harness, PairResult, Task, TaskView, Trajectory, digest
from .storage import Store


class TaskExecutionError(RuntimeError):

    def __init__(self, message, execution=None):
        super().__init__(message)
        self.execution = execution


class InfrastructureError(RuntimeError):

    def __init__(self, message, execution=None):
        super().__init__(message)
        self.execution = execution


class Runner(Protocol):
    exposed_targets: tuple[str, ...]

    @property
    def system_id(self) -> str: ...

    def execute(self, task: TaskView, harness: Harness, seed: int) -> Execution: ...


class Evaluator:
    def __init__(self, runner: Runner, tasks: tuple[Task, ...],
                 scorer: Callable[[Task, str], float] | None, store: Store, seed: int):
        self.runner, self.scorer, self.store, self.seed = runner, scorer, store, seed
        self.tasks = {t.task_id: t for t in tasks}
        if len(self.tasks) != len(tasks):
            raise ValueError("task IDs must be unique")
        self.fixed_system_id = runner.system_id
        self.serial = 0

    def check_system(self):
        if self.runner.system_id != self.fixed_system_id:
            raise ValueError("fixed MAS configuration changed during optimization")

    def run(self, harness: Harness, task_id: str, *, phase: str, round_number: int,
            repetition: int) -> Trajectory:
        self.check_system()
        task = self.tasks[task_id]
        self.serial += 1
        run_id = f"run-{self.serial:07d}"
        seed_key = ([self.seed, "test", repetition, task_id] if phase == "test"
                    else [self.seed, phase, round_number, repetition, task_id])
        seed = int(digest(seed_key)[:8], 16)
        error = None
        execution = None
        try:
            execution = self.runner.execute(task.public(), harness, seed)
            if execution.scoring_error is not None:
                raise InfrastructureError(execution.scoring_error)
            if execution.task_error is not None:
                score, error = 0.0, execution.task_error
            elif self.scorer is None:
                if execution.native_score is None:
                    raise ValueError("native scorer requires Execution.native_score")
                score = execution.native_score
            else:
                score = self.scorer(task.model_copy(deep=True), execution.answer)
        except (TaskExecutionError, HookError) as exc:
            execution = getattr(exc, "execution", None) or execution or Execution(answer="", events=())
            score = 0.0
            error = f"{type(exc).__name__}: {exc}"
        self.check_system()
        return Trajectory(run_id=run_id, task=task.public(), harness_id=harness.harness_id,
                          score=score, execution=execution, error=error)

    def collect(self, harness: Harness, ids: tuple[str, ...], *, phase: str,
                round_number: int, repetitions: int) -> tuple[Trajectory, ...]:
        return tuple(self.run(harness, task_id, phase=phase, round_number=round_number,
                              repetition=r)
                     for r in range(repetitions) for task_id in ids)

    def pair(self, parent: Harness, candidate: Harness, ids: tuple[str, ...], *,
             phase: str, round_number: int, repetitions: int) -> PairResult:
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("paired evaluation requires nonempty unique task IDs")
        parents, candidates = [], []
        for r in range(repetitions):
            for task_id in ids:
                request = dict(phase=phase, round_number=round_number, repetition=r)

                parents.append(self.run(parent, task_id, **request))
                candidates.append(self.run(candidate, task_id, **request))

        delta = float(sum((Decimal(str(c.score)) - Decimal(str(p.score))
                           for p, c in zip(parents, candidates, strict=True)), Decimal(0)) / len(parents))
        return PairResult(parent=tuple(parents), candidate=tuple(candidates), delta=delta)
