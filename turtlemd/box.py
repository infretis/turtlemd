"""Define a simulation box."""
import logging
from abc import ABC, abstractmethod

import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def interpret_box_size(
    size: np.ndarray,
    periodicity: list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Figure out the input simulation box size."""
    size = size.astype(float)
    if size.ndim < 2:
        high = np.copy(size)
        low = np.zeros_like(high)
    else:
        low = size[:, 0]
        high = size[:, 1]
    length = high - low
    low[length == float("inf")] = -float("inf")
    high[length == float("inf")] = float("inf")
    # Adjust box lengths for non-periodic dimensions:
    if periodicity is not None:
        for i, peri in enumerate(periodicity):
            if not peri:
                low[i] = -float("inf")
                high[i] = float("inf")
                length[i] = float("inf")
    else:
        periodicity = [True] * len(low)
    return low, high, length, periodicity


class Box(ABC):
    """A generic simulation box.

    This class defines a generic simulation box.

    Attributes
    ----------
    dim : int
        The dimensionality of the box.
    periodic : list of bool
        Specifies which dimensions for which we should apply
        periodic boundaries.
    low : numpy.array
        The lower limits of the simulation box.
    high : numpy.array
        The upper limits of the simulation box.
    length : numpy.array
        The box lengths
    ilength : numpy.array
        The inverse box lengths.
    box_matrix : numpy.array
        2D matrix, representing the simulation cell.
    """

    dim: int
    periodic: list[bool]
    low: np.ndarray
    high: np.ndarray
    length: np.ndarray
    ilength: np.ndarray
    box_matrix: np.ndarray

    def __init__(
        self,
        size : np.ndarray,
        periodicity: list[bool] | None = None,
    ):
        """Initialise the Box class."""
        self.low, self.high, self.length, self.periodic = interpret_box_size(
            size, periodicity=periodicity
        )
        self.dim = len(self.periodic)
        assert len(self.periodic) == len(self.low)
        self.box_matrix = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            self.box_matrix[i, i] = self.length[i]
        self.ilength = 1.0 / self.length

    def volume(self) -> float:
        """Calculate volume of the simulation cell."""
        return np.linalg.det(self.box_matrix)

    @abstractmethod
    def pbc_wrap(self, pos: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to the given positions."""

    @abstractmethod
    def pbc_dist_matrix(self, distance: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to a matrix of distance vectors."""

    @abstractmethod
    def pbc_dist(self, distance: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to a distance vector."""

    def __str__(self) -> str:
        """Return a string describing the box."""
        return f"Hello, this is box. My matrix is: {self.box_matrix}"


class RectangularBox(Box):
    """An orthogonal box."""

    def pbc_wrap(self, pos: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to the given position.

        Parameters
        ----------
        pos : nump.array
            Positions to apply periodic boundaries to.

        Returns
        -------
        out : numpy.array, same shape as parameter `pos`
            The periodic-boundary wrapped positions.

        """
        pbcpos = np.zeros_like(pos)
        for i, periodic in enumerate(self.periodic):
            if periodic:
                low = self.low[i]
                length = self.length[i]
                ilength = self.ilength[i]
                relpos = pos[:, i] - low
                delta = np.where(
                    np.logical_or(relpos < 0.0, relpos >= length),
                    relpos - np.floor(relpos * ilength) * length,
                    relpos,
                )
                pbcpos[:, i] = delta + low
            else:
                pbcpos[:, i] = pos[:, i]
        return pbcpos

    def pbc_dist_matrix(self, distance: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to a matrix of distance vectors.

        Parameters
        ----------
        distance : numpy.array
            The distance vectors.

        Returns
        -------
        out : numpy.array, same shape as the `distance` parameter
            The pbc-wrapped distances.
        """
        pbcdist = np.copy(distance)
        for i, (periodic, length, ilength) in enumerate(
            zip(self.periodic, self.length, self.ilength)
        ):
            if periodic:
                dist = pbcdist[:, i]
                high = 0.5 * length
                k = np.where(np.abs(dist) >= high)[0]
                dist[k] -= np.rint(dist[k] * ilength) * length
        return pbcdist

    def pbc_dist(self, distance: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to a distance vector."""
        pbcdist = np.zeros_like(distance)
        for i, (periodic, length, ilength) in enumerate(
            zip(self.periodic, self.length, self.ilength)
        ):
            if periodic and np.abs(distance[i]) > 0.5 * length:
                pbcdist[i] = (
                    distance[i] - np.rint(distance[i] * ilength) * length
                )
            else:
                pbcdist[i] = distance[i]
        return pbcdist
