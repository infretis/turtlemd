"""Define the input file."""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import toml

from turtlemd.inout.common import generic_factory
from turtlemd.integrators import INTEGRATORS, MDIntegrator
from turtlemd.system import Box

if TYPE_CHECKING:  # pragma: no cover
    pass


DEFAULT = pathlib.Path(__file__).resolve().parent / "default.toml"


def read_settings_file(settings_file: str | pathlib.Path) -> dict[str, Any]:
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


def create_integrator_from_settings(
    settings: dict[str, Any]
) -> MDIntegrator | None:
    """Create an integrator from settings."""
    return generic_factory(
        settings["integrator"], INTEGRATORS, name="integrator"
    )
