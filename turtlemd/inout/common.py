"""Methods for dealing with the creation of objects."""
from __future__ import annotations

import inspect
from collections import defaultdict
from inspect import Parameter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


def get_parameter_type(arg: Parameter) -> str:
    if arg.kind == Parameter.POSITIONAL_OR_KEYWORD:
        return "args" if arg.default is Parameter.empty else "kwargs"
    kind_mapping = {
        Parameter.POSITIONAL_ONLY: "args",
        Parameter.VAR_POSITIONAL: "varargs",
        Parameter.KEYWORD_ONLY: "kwargs",
        Parameter.VAR_KEYWORD: "keywords",
    }
    return kind_mapping[arg.kind]


def inspect_function(method: Callable) -> dict[str, list[Any]]:
    """Return arguments/kwargs of a given function.

    Args:
        method: The callable to inspect.

    Returns
        dict: A dictionary with parameter types as keys and their
        names as values. The types are:
        * `args` : list of the positional arguments
        * `kwargs` : list of keyword arguments
        * `varargs` : list of arguments
        * `keywords` : list of keyword arguments
    """
    arguments = inspect.signature(method)
    parameters_by_type = defaultdict(list)
    for name, param in arguments.parameters.items():
        parameters_by_type[get_parameter_type(param)].append(name)
    return parameters_by_type
