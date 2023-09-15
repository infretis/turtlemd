"""Define the input file."""
import pathlib
from typing import Any

import toml

from turtlemd.system import Box

DEFAULT = pathlib.Path(__file__).resolve().parent / "default.toml"


def read_settings_file(settings_file: str) -> dict[str, Any]:
    """Read settings from the given file."""
    default = toml.load(DEFAULT)
    settings = toml.load(settings_file)
    return deep_update(default, settings)


def deep_update(
    target: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    for key, val in source.items():
        if isinstance(val, dict):
            target[key] = deep_update(target.get(key, {}), val)
        else:
            target[key] = val
    return target


def create_box_from_settings(settings: dict[str, Any]) -> Box:
    """Create a simulation box from settings."""
    low = settings["box"]["low"]
    high = settings["box"]["high"]
    periodic = settings["box"]["periodic"]

    try:
        len(periodic)
    except TypeError:
        periodic = [periodic] * len(low)
        settings["box"]["periodic"] = periodic

    return Box(low=low, high=high, periodic=periodic)
