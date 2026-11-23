from .hooks import TARGETS


def validate_runtime_context(runtime_context: dict, exposed_targets: tuple[str, ...]):

    hooks = runtime_context.get("hooks")
    if not isinstance(hooks, dict):
        raise ValueError("runtime_context.hooks must describe the runner's exposed targets")
    for target in exposed_targets:
        semantics = hooks.get(target)
        if not isinstance(semantics, dict):
            raise ValueError(f"runtime_context.hooks is missing {target}")
        invocation = semantics.get("invocation")
        if not isinstance(invocation, str) or not invocation.strip():
            raise ValueError(f"{target} requires a native invocation description")
        if "implementation" in semantics or target in (
                "prepare_node_action", "prepare_worker_request", "normalize_worker_handoff",
                "resolve_edge_semantics", "plan_graph_readout", "decide_orchestration"):
            source = semantics.get("implementation")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{target}.implementation must supply nonempty native consumer source")
        for field in ("inputs", "outputs"):
            descriptions = semantics.get(field)
            if (not isinstance(descriptions, dict)
                    or set(descriptions) != set(getattr(TARGETS[target], field))
                    or any(not isinstance(v, str) or not v.strip() for v in descriptions.values())):
                raise ValueError(f"{target}.{field} must describe every registered field")
