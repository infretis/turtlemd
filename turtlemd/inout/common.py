"""Methods for creating objects from settings."""
from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from inspect import Parameter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def generic_factory(
    settings: dict[str, Any],
    registry: dict[str, type],
    name: str = "generic",
) -> Any | None:
    """Create an instance of a class based on the given settings.

    Args:
        settings: Specifies arguments and keyword arguments for object
            initiation. Must contain a "class" key to specify which
            class should be instantiated.
        registry: A mapping from class names to classes.
        name: Labels the type of classes we create. Mainly
            used for error messages. Defaults to "generic".

    Returns:
        An instance of the specified class if successful. Returns
            None otherwise.

    Raises:
        ValueError: If the "class" key is missing in settings or if the
            specified class is not found in the registry.
    """
    klass_name = settings.get("class", "").lower()

    if not klass_name:
        msg = f'No "class" given for {name}! Could not create object!'
        LOGGER.error(msg)
        raise ValueError(msg)

    klass = registry.get(klass_name, None)

    if klass is None:
        msg = f'Could not create unknown {name} class "{settings["class"]}"!'
        LOGGER.error(msg)
        raise ValueError(msg)
    settings.pop("class", None)
    return initiate_instance(klass, settings)


def get_parameter_type(arg: Parameter) -> str:
    """Determine the type of a method's parameter.

    This method categorizes the given parameter based on its kind
    (e.g., positional, keyword-only, etc.).

    Args:
        arg: The parameter whose type will be determined.

    Returns:
        The type of the parameter. Can be one of the following:
            - "args": Positional parameters or positional-or-keyword
                parameters without default values.
            - "kwargs": Keyword-only parameters or positional-or-keyword
                parameters with default values.
            - "varargs": For variable positional parameters (i.e., *args).
            - "keywords": For variable keyword parameters (i.e., **kwargs).
    """
    if arg.kind == Parameter.POSITIONAL_OR_KEYWORD:
        return "args" if arg.default is Parameter.empty else "kwargs"
    kind_mapping = {
        Parameter.POSITIONAL_ONLY: "args",
        Parameter.VAR_POSITIONAL: "varargs",
        Parameter.KEYWORD_ONLY: "kwargs",
        Parameter.VAR_KEYWORD: "keywords",
    }
    return kind_mapping[arg.kind]


def get_args_and_kwargs(
    klass: Any, settings: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Extract arguments for a class's constructor from given settings.

    This method identifies the required arguments for the constuctor
    of `klass` and retrieves their values from the `settings` dictionary.

    Args:
        klass: The class for which the constructor arguments need to be
            extracted.
        settings: A dictionary containing potential positional and keyword
            arguments for `klass.__init__()`.

    Returns:
        A tuple containing:
            - Positional arguments extracted from `settings`.
            - Keyword arguments extracted from `settings`.

    Raises:
        ValueError: If a required positional argument for `klass` is not
            present in `settings` or if the method needs *args or **kwargs.
    """
    info = inspect_function(klass.__init__)
    used_args, positional_args, keyword_args = set(), [], {}
    missing_args = []

    # We disallow the usage of *args when creating instances from settings:
    if info.get("varargs"):
        raise ValueError(
            "*args are not supported when initiating classes from settings!"
        )
    # We disallow the usage of **kwargs when creating instances from settings:
    if info.get("keywords"):
        raise ValueError(
            "*kwargs are not supported when initiating classes from settings!"
        )

    for arg in info.get("args", []):
        if arg == "self":
            continue
        if arg in settings:
            positional_args.append(settings[arg])
            used_args.add(arg)
        else:
            missing_args.append(arg)

    if missing_args:
        raise ValueError(
            f"Required arguments {', '.join(missing_args)} for "
            f"{klass} were not found!"
        )

    for kwarg in info.get("kwargs", []):
        if kwarg != "self" and kwarg in settings:
            keyword_args[kwarg] = settings[kwarg]
            used_args.add(kwarg)

    for key in settings:
        if key not in used_args:
            LOGGER.warning(
                'Ignored extra argument "%s" when initiating %s', key, klass
            )
    return positional_args, keyword_args


def inspect_function(method: Callable) -> dict[str, list[Any]]:
    """Return arguments/kwargs of a given function.

    This function categorizes the parameters of the given function or method
    based on their types (e.g., positional, keyword-only, etc.).

    Args:
        method: The function or method to inspect.

    Returns:
        A dictionary categorizing parameters:
            - "args": Positional or positional-or-keyword parameters
                without default values.
            - "kwargs": Keyword-only or positional-or-keyword parameters
                with default values.
            - "varargs": Variable positional parameters (*args).
            - "keywords": Variable keyword parameters (**kwargs).
    """
    arguments = inspect.signature(method)
    parameters_by_type = defaultdict(list)
    for name, param in arguments.parameters.items():
        parameters_by_type[get_parameter_type(param)].append(name)
    return parameters_by_type


def initiate_instance(klass: type[Any], settings: dict[str, Any]) -> Any:
    """Initialize a class with arguments from settings.

    Args:
        klass: The class to initialize.
        settings): A dictionary containing positional and keyword arguments
            (if any) for `klass.__init__()`.

    Returns:
        An instance of the given class, initialized with the arguments
            extracted from `settings`.
    """
    args, kwargs = get_args_and_kwargs(klass, settings)
    msg = 'Initiated "%s" from "%s"'
    msg += "\n-arguments: %s"
    msg += "\n-kwargs: %s"
    LOGGER.debug(msg, klass.__name__, klass.__module__, args, kwargs)
    return klass(*args, **kwargs)
