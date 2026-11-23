
import inspect
from textwrap import dedent


def native_implementation(*functions) -> str:





    parts = []
    for function in functions:
        lines, _ = inspect.getsourcelines(function)
        parts.append(dedent("".join(lines)))
    return "\n".join(parts)
