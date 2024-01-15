"""Define the input file."""
from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import numpy as np
import toml

from turtlemd.inout.common import generic_factory
from turtlemd.inout.xyz import configuration_from_xyz_file
from turtlemd.integrators import INTEGRATORS, MDIntegrator
from turtlemd.random import create_random_generator
from turtlemd.system import Box, Particles, System
from turtlemd.system.box import TriclinicBox
from turtlemd.system.particles import generate_maxwell_velocities

if TYPE_CHECKING:  # pragma: no cover
    pass

DEFAULT = pathlib.Path(__file__).resolve().parent / "default.toml"

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def read_settings_file(settings_file: pathlib.Path | str) -> dict[str, Any]:
    """Read settings from the given file."""
    settings_file = pathlib.Path(settings_file)
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


def create_box_from_settings(settings: dict[str, Any]) -> Box | TriclinicBox:
    """Create a simulation box from settings."""
    low = settings["box"]["low"]
    periodic = settings["box"]["periodic"]

    try:
        len(periodic)
    except TypeError:
        periodic = [periodic] * len(low)
        settings["box"]["periodic"] = periodic

    if any(angle in settings["box"] for angle in ("alpha", "beta", "gamma")):
        return TriclinicBox(**settings["box"])
    return Box(**settings["box"])


def create_integrator_from_settings(settings: dict[str, Any]) -> MDIntegrator:
    """Create an integrator from settings."""
    return generic_factory(
        settings["integrator"], INTEGRATORS, name="integrator"
    )


def look_for_file(settings: dict[str, Any], filename: str) -> pathlib.Path:
    """Find the file from a give input string."""
    file_path = pathlib.Path(filename)

    if file_path.is_absolute() and file_path.is_file():
        return file_path
    else:
        base_dir = settings["settings"]["directory"]
        file_path = (base_dir / file_path).resolve()
        if not file_path.is_file():
            msg = "File %s not found."
            LOGGER.critical(msg, file_path)
            raise FileNotFoundError(msg, file_path)
    return file_path


def get_particle_data_from_settings(
    settings: dict[str, Any],
    dict_key: str,
    list_key: str,
    file_key: str,
    dtype: type = float,
) -> tuple[dict[str, int], np.ndarray | None]:
    """Get masses or types from the settings."""
    data_dict = settings["particles"].get(dict_key, {})

    data_list = settings["particles"].get(list_key)
    if data_list:
        data_list = np.array(data_list, dtype=dtype)

    if file_key in settings["particles"]:
        filename = look_for_file(settings, settings["particles"][file_key])
        data_list = np.loadtxt(filename, dtype=dtype)

    return data_dict, data_list


def create_particles_from_settings(
    settings: dict[str, Any], dim: int = 3
) -> Particles:
    """Create particles from settings.

    Args:
        settings: The settings to create particles from.
        dim: The dimensionality of the system.
    """
    particles = Particles(dim=dim)

    xyz_file = look_for_file(settings, settings["particles"]["file"])
    msg = "Loading initial coordinates from file: %s"
    LOGGER.info(msg, xyz_file)
    atoms, pos, vel, force = configuration_from_xyz_file(xyz_file, dim=dim)

    mass_dict, mass_list = get_particle_data_from_settings(
        settings,
        dict_key="masses",
        list_key="mass_list",
        file_key="mass_file",
        dtype=float,
    )
    type_dict, type_list = get_particle_data_from_settings(
        settings,
        dict_key="types",
        list_key="type_list",
        file_key="type_file",
        dtype=float,
    )

    if not type_dict:
        type_dict = {atom: idx for idx, atom in enumerate(set(atoms))}

    for i, (atomi, posi, veli, forcei) in enumerate(
        zip(atoms, pos, vel, force)
    ):
        if mass_list is not None:
            massi = mass_list[i]
        else:
            massi = mass_dict.get(atomi, 1.0)
        if type_list is not None:
            typei = type_list[i]
        else:
            typei = type_dict.get(atomi, -1)
        particles.add_particle(
            pos=posi,
            vel=veli,
            force=forcei,
            mass=massi,
            name=atomi,
            ptype=typei,
        )
    return particles


def create_velocities(settings: dict[str, Any], system: System) -> None:
    """Create velocities for the particles in a system."""
    vel_settings = settings.get("particles", {}).get("velocity")
    if vel_settings is None:
        return

    rgen = create_random_generator(seed=vel_settings.get("seed"))

    generate_maxwell_velocities(
        system.particles,
        rgen,
        temperature=vel_settings["temperature"],
        boltzmann=1.0,
        dof=system.dof(),
        momentum=vel_settings["momentum"],
    )


def create_system_from_settings(settings: dict[str, Any]) -> System:
    """Create a system from the given settings."""
    # Set up the box:
    box = create_box_from_settings(settings)
    particles = create_particles_from_settings(settings, dim=box.dim)
    system = System(
        box=box,
        particles=particles,
    )
    create_velocities(settings, system)
    return system
