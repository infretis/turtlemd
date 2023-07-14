"""Methods to read particles from xyz-files."""
import logging
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from turtlemd.system.particles import Particles

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass
class Snapshot:
    """Store coordinates and atoms for a snapshot."""

    natoms: int = 0
    comment: str = ""
    atoms: list[str] = field(default_factory=list)
    _xyz: list[list[float]] = field(default_factory=list)
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))


def read_xyz_file(filename: str | pathlib.Path) -> Iterator[Snapshot]:
    """Read configurations from a xyz-file"""

    lines_to_read = 0
    snapshot = None
    with open(filename) as fileh:
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
