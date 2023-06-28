"""Some helper methods for setting up simulations."""
import itertools

import numpy as np

UNIT_CELL = {
    "sc": np.array([[0.0, 0.0, 0.0]]),
    "sq": np.array([[0.0, 0.0]]),
    "sq2": np.array([[0.0, 0.0], [0.5, 0.5]]),
    "bcc": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    "fcc": np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]
    ),
    "hcp": np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 5.0 / 6.0, 0.5],
            [0.0, 1.0 / 3.0, 0.5],
        ]
    ),
    "diamond": np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ]
    ),
}


def generate_lattice(
    lattice: str = "fcc",
    repeat: list[int] | None = None,
    lattice_constant: float | None = None,
    density: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate points on a simple lattice.

    The lattice is one of the items in `UNIT_CELL`. The lattice is
    repeated `repeat` number of times. The lattice spacing can be
    set with the lattice constant or with the density.

    Args:
        lattice (str): The type of lattice to create. See `UNIT_CELL`.
        repeat (list[int]): Number of times to repeat the lattice.
        lattice_constant (float): Lattice constant to use.
        density (float): The number density of the lattice.

    Returns:
        out[0] (np.ndarray): The generated positions.
        out[1] (np.ndarray): The size (as in the box-vectors) the lattice.
    """
    unit_cell = UNIT_CELL.get(lattice.lower(), None)
    if unit_cell is None:
        msg = (
            f"Unknown lattice '{lattice}'. Expected one of {UNIT_CELL.keys()}"
        )
        raise ValueError(msg)
    npart, ndim = unit_cell.shape
    lcon = lattice_constant
    if density is not None:
        lcon = (npart / density) ** (1.0 / float(ndim))
    if lcon is None:
        lcon = 1
    if repeat is None:
        repeat = [1] * ndim
    positions = []
    for i in itertools.product(*[range(nri) for nri in repeat[:ndim]]):
        pos = lcon * (np.array(i) + unit_cell)
        positions.extend(pos)
    size = [[0.0, i * lcon] for i in repeat[:ndim]]
    return np.array(positions), np.array(size)
