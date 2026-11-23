from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
from pathlib import Path

from mash import Harness, MASH, SearchConfig, Task, TaskSplit, validate_runtime_context
from mash.evaluation import Evaluator, Runner
from mash.finance_contracts import FINANCE_HOOK_CONTRACTS
from mash.optimizer import LLMOptimizer
from mash.storage import Store


def exact_match(task: Task, answer: str) -> float:
    references = task.reference if isinstance(task.reference, list) else [task.reference]
    normalize = lambda value: " ".join(str(value).casefold().split())
    return float(any(normalize(answer) == normalize(reference) for reference in references))


def optimize_finance_fold(*, runner: Runner, tasks: tuple[Task, ...], split: TaskSplit,
                          output: Path, runtime_context: dict,
                          optimizer_client_factory: Callable[[Store], object]):
    config = SearchConfig(initialization_repetitions=2)
    validate_runtime_context(runtime_context, runner.exposed_targets)
    contracts = {target: FINANCE_HOOK_CONTRACTS[target]
                 for target in runner.exposed_targets if target in FINANCE_HOOK_CONTRACTS}

    store = Store(output)
    optimizer_client = optimizer_client_factory(store)
    if getattr(optimizer_client, "store", None) is not store:
        raise ValueError("the optimizer client and evaluator must share the same Store")

    evaluator = Evaluator(runner, tasks, exact_match, store, config.seed)
    optimizer = LLMOptimizer(optimizer_client, config, runner.exposed_targets,
                             runtime_context=runtime_context, hook_contracts=contracts)
    return MASH(evaluator, optimizer, split, config).optimize(Harness())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Optimize one Finance fold with MASH.")
    parser.add_argument("--integration", required=True,
                        help="module exporting runner, tasks, split, runtime_context, and make_optimizer_client")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    integration = importlib.import_module(args.integration)
    frozen = optimize_finance_fold(
        runner=integration.runner,
        tasks=integration.tasks,
        split=integration.split,
        output=args.output,
        runtime_context=integration.runtime_context,
        optimizer_client_factory=integration.make_optimizer_client,
    )
    print(frozen.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
