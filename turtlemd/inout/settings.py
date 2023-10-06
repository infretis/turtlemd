"""Define the input file."""
from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import toml

from turtlemd.inout.common import generic_factory
from turtlemd.integrators import INTEGRATORS, MDIntegrator
from turtlemd.system import Box, System
from turtlemd.tools.xyz import particles_from_xyz_file

if TYPE_CHECKING:  # pragma: no cover
    pass


DEFAULT = pathlib.Path(__file__).resolve().parent / "default.toml"

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def read_settings_file(settings_file: pathlib.Path) -> dict[str, Any]:
    """Read settings from the given file."""
    default = toml.load(DEFAULT)
    settings = toml.load(settings_file)
    settings = deep_update(default, settings)
    settings["settings"]["directory"] = settings_file.resolve().parent
    return settings


def deep_update(
    target: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    """Update the target dictionary with settings from the source."""
    for key, val in source.items():
        if isinstance(val, dict):
            target[key] = deep_update(target.get(key, {}), val)
        else:
            target[key] = val
    return target


def search_for_setting(
    settings: dict[str, Any], target: str
) -> list[dict[str, Any]]:
    """Search for dictionary items by a given key."""
    stack = [settings]
    found = []
    while stack:
        current = stack.pop()
        for key, val in current.items():
            if isinstance(val, dict):
                stack.append(val)
            elif key == target:
                found.append(current)
    return found


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


def create_system_from_settings(settings: dict[str, Any]) -> System:
    """Create a system from the given settings."""
    xyz_filename = pathlib.Path(settings["particles"]["file"])

    if xyz_filename.is_absolute() and xyz_filename.is_file():
        xyz_file = xyz_filename
    else:
        base_dir = settings["settings"]["directory"]
        xyz_file = (base_dir / xyz_filename).resolve()
        if not xyz_file.is_file():
            msg = "Coordinate file %s not found."
            LOGGER.critical(msg, xyz_file)
            raise FileNotFoundError(msg, xyz_file)

    msg = "Loading initial coordinates from file: %s"
    LOGGER.info(msg, xyz_file)

    system = System(
        box=create_box_from_settings(settings),
        particles=particles_from_xyz_file(xyz_file),
    )
    return system
