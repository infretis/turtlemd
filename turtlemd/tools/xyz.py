"""Methods to read particles from xyz-files."""
import logging
from dataclasses import dataclass, field

import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass
class Snapshot:
    """Store coordinates and atoms for a snapshot."""

    natoms: int = 0
    comment: str = ""
    atoms: list[str] = field(default_factory=list)
    _xyz: list[list[float]] = field(default_factory=list)
    xyz: np.ndarray = np.zeros(3)


def read_xyz_file(filename):
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
                        "Could not read the number of atoms (first line) in file %s!",
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
