"""Methods to read particles from xyz-files."""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from turtlemd.system.particles import Particles

if TYPE_CHECKING:  # pragma: no cover
    from turtlemd.system import System

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def pad_to_nd(x: np.ndarray, dim: int = 3) -> np.ndarray:
    """Pad 1D and 2D vectors to 3D."""
    length = max(0, dim - len(x))
    return np.pad(x, (0, length), mode="constant")


@dataclass
class Snapshot:
    """Store coordinates and atoms for a snapshot."""

    natoms: int = 0
    comment: str = ""
    atoms: list[str] = field(default_factory=list)
    _xyz: list[list[float]] = field(default_factory=list)
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))


def read_xyz_file(filename: str | pathlib.Path) -> Iterator[Snapshot]:
    """Read configurations from a xyz-file."""
    lines_to_read = 0
    snapshot = None
    with open(filename, encoding="utf-8") as fileh:
        for lines in fileh:
            if lines_to_read == 0:
                # This is a new frame
                if snapshot is not None:
                    snapshot.xyz = np.array(snapshot._xyz)
                    yield snapshot
                snapshot = None
                try:
                    lines_to_read = int(lines.strip())  # Atoms on first line
                except ValueError:
                    LOGGER.error(
                        "Could not read the number of atoms "
                        "(first line) in file %s!",
                        filename,
                    )
                    raise
            else:
                if snapshot is None:
                    snapshot = Snapshot(
                        natoms=lines_to_read, comment=lines.strip()
                    )
                    continue
                lines_to_read -= 1
                data = lines.strip().split()
                snapshot.atoms.append(data[0])
                snapshot._xyz.append([float(i) for i in data[1:]])

    if snapshot is not None:
        snapshot.xyz = np.array(snapshot._xyz)
        yield snapshot


def particles_from_xyz_file(
    filename: str | pathlib.Path,
    dim: int = 3,
    masses: dict[str, float] | None = None,
) -> Particles:
    """Create particles from a given xyz-file.

    Args:
        dim: The number of dimensions to consider.
        masses: dict[str, float]
    """
    if masses is None:
        masses = {}
    snapshot = next(read_xyz_file(filename))
    particles = Particles(dim=dim)
    # We will just assign particle types from the atom name:
    ptypes = {atom: idx for idx, atom in enumerate(set(snapshot.atoms))}
    for atom, xyz in zip(snapshot.atoms, snapshot.xyz):
        pos = xyz[:dim]
        vel = None
        force = None
        if len(xyz) > dim:
            vel = xyz[dim : dim * 2]
            if len(vel) != len(pos):
                vel = None
            force = xyz[dim * 2 : dim * 3]
            if len(force) != len(pos):
                force = None
        particles.add_particle(
            pos=pos,
            vel=vel,
            force=force,
            mass=masses.get(atom, 1.0),
            name=atom,
            ptype=ptypes.get(atom, -1),
        )
    return particles


def system_to_xyz(
    system: System,
    filename: str | pathlib.Path,
    filemode: str = "w",
    title: str | None = None,
) -> None:
    """Write the system configuration to a xyz-file.

    Args:
        system: The system to get the particles from.
        filename: The path to the file to write.
        filemode: If "w" this method will overwrite if the `filename`
            already exists. Otherwise it will append to it.
        title: A string to use as the title for the frame.
    """
    if filemode != "w":
        filemode = "a"
    if title is None:
        box = " ".join([f"{i}" for i in system.box.box_matrix.flatten()])
        txt_title = f"# TurtleMD system. Box: {box}\n"
    else:
        txt_title = f"{title}\n"
    with open(filename, filemode, encoding="utf-8") as output_xyz:
        output_xyz.write(f"{system.particles.npart}\n")
        output_xyz.write(txt_title)
        for part in system.particles:
            name = f"{part['name']:5s}"
            pos = " ".join([f"{i:15.9f}" for i in pad_to_nd(part["pos"])])
            output_xyz.write(f"{name} {pos}\n")
