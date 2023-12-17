"""Define a simulation box."""
import logging
from abc import ABC, abstractmethod

import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def guess_dimensionality(
    low: np.ndarray | list[float] | list[int] | None = None,
    high: np.ndarray | list[float] | list[int] | None = None,
    periodic: list[bool] | None = None,
) -> int:
    """Figure out the number of dimensions from the box input."""
    dims = []
    if low is not None:
        dims.append(len(low))
    if high is not None:
        dims.append(len(high))
    if periodic is not None:
        dims.append(len(periodic))

    if len(dims) == 0:
        LOGGER.warning(
            "Missing low/high/periodic parameters for box: assuming 1D"
        )
        return 1
    if len(set(dims)) != 1:
        LOGGER.error("Inconsistent box dimensions for low/high/periodic!")
        raise ValueError("Inconsistent box dimensions for low/high/periodic!")
    return dims[0]  # They should all be equal, pick the first.

def cosine(angle: float) -> float:
    """Return cosine of an angle in degrees.

    Note:
        If the angle is close to 90.0 we return 0.0.

    Args:
        angle: The angle in degrees.

    Returns:
        The cosine of the angle.
    """
    radians = np.radians(angle)
    return np.where(np.isclose(angle, 90.0), 0.0, np.cos(radians))

class BoxBase(ABC):
    """Define a generic simulation box.

    Attributes:
        dim (int): The dimensionality of the box.
        dof (np.ndarray): The degrees of freedom removed by periodicity.
        periodic (list[bool]): Specifies which dimensions for
            which we should apply periodic boundaries.
        box_matrix (np.ndarray): 2D matrix, representing the simulation cell.
    """

    dim: int
    dof: np.ndarray
    periodic: list[bool]
    box_matrix: np.ndarray

    def __init__(
        self,
        dim: int,
        periodic: list[bool] | None,
    ) -> None:
        """Create a generic box.

        Args:
            dim: The dimensionality of the box.
            periodic: Specifies which dimensions for which we should
                apply periodic boundaries.
        """
        self.dim = dim
        if periodic is not None:
            self.periodic = periodic
        else:
            self.periodic = [True] * self.dim
        assert self.dim == len(self.periodic)
        # Keep track of the degrees of freedom removed by periodic
        # boundaries:
        self.dof = np.array([1 if i else 0 for i in self.periodic])
        self.box_matrix = np.zeros((self.dim, self.dim))

    def volume(self) -> float:
        """Calculate volume of the simulation cell."""
        return np.linalg.det(self.box_matrix)

    @abstractmethod
    def pbc_wrap(self, pos: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to positions."""

    @abstractmethod
    def pbc_dist(self, distance: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to a distance vector."""

    @abstractmethod
    def pbc_dist_matrix(self, distance: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to a matrix of distance vectors."""


class TriclinicBox(BoxBase):
    """An triclinic simulation box.

    Attributes:
        low (np.ndarray): The lower limits of the simulation box.
        high (np.ndarray): The upper limits of the simulation box.
        length (np.ndarray): The box lengths
        ilength (np.ndarray): The inverse box lengths.
    """

    low: np.ndarray
    high: np.ndarray
    length: np.ndarray
    ilength: np.ndarray
    alpha: float
    beta: float
    gamma: float

    def __init__(
        self,
        low: np.ndarray | list[float] | list[int] | None = None,
        high: np.ndarray | list[float] | list[int] | None = None,
        periodic: list[bool] | None = None,
        alpha: float | None=None,
        beta: float | None=None,
        gamma: float | None=None,
    ):
        """Create the box."""
        super().__init__(
            dim=guess_dimensionality(low=low, high=high, periodic=periodic),
            periodic=periodic,
        )
        if low is not None:
            self.low = np.asarray(low).astype(float)
        else:
            self.low = np.array(
                [0.0 if i else -float("inf") for i in self.periodic]
            )
            LOGGER.warning("Set box low values: %s", self.low)

        if high is not None:
            self.high = np.asarray(high).astype(float)
        else:
            self.high = np.array(
                [1.0 if i else float("inf") for i in self.periodic]
            )
            LOGGER.warning("Set box high values: %s", self.high)

        self.length = self.high - self.low
        self.ilength = 1.0 / self.length

        self.alpha = alpha if alpha is not None else 90
        self.beta = beta if beta is not None else 90
        self.gamma = gamma if gamma is not None else 90

    def pbc_wrap(self, pos: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to positions.

        Args:
            pos (np.ndarray): Positions to apply periodic
                boundaries to.

        Returns:
            np.ndarray: The periodic-boundary wrapped positions,
                same shape as parameter `pos`.
        """
        pass

    def pbc_dist(self, distance: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to a distance vector."""
        pass

    def pbc_dist_matrix(self, distance: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to a matrix of distance vectors.

        Args:
            distance (np.ndarray): The distance vectors.

        Returns:
            np.ndarray: The PBC-wrapped distances, same shape as the
                `distance` parameter.
        """

    def __str__(self) -> str:
        """Return a string describing the box."""
        msg = [
            f"Hello, this is triclinic box and my matrix is:\n{self.box_matrix}",
            f"Periodic? {self.periodic}",
        ]
        return "\n".join(msg)


class Box(TriclinicBox):
    """An orthogonal simulation box.

    Attributes:
        low (np.ndarray): The lower limits of the simulation box.
        high (np.ndarray): The upper limits of the simulation box.
        length (np.ndarray): The box lengths
        ilength (np.ndarray): The inverse box lengths.
    """

    low: np.ndarray
    high: np.ndarray
    length: np.ndarray
    ilength: np.ndarray

    def __init__(
        self,
        low: np.ndarray | list[float] | list[int] | None = None,
        high: np.ndarray | list[float] | list[int] | None = None,
        periodic: list[bool] | None = None,
    ):
        """Create the box."""
        super().__init__(
            low=low,
            high=high,
            periodic=periodic,
            alpha=None,
            beta=None,
            gamma=None,
        )

        self.box_matrix = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            self.box_matrix[i, i] = self.length[i]

    def pbc_wrap(self, pos: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to positions.

        Args:
            pos (np.ndarray): Positions to apply periodic
                boundaries to.

        Returns:
            np.ndarray: The periodic-boundary wrapped positions,
                same shape as parameter `pos`.
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

    def pbc_dist_matrix(self, distance: np.ndarray) -> np.ndarray:
        """Apply periodic boundaries to a matrix of distance vectors.

        Args:
            distance (np.ndarray): The distance vectors.

        Returns:
            np.ndarray: The PBC-wrapped distances, same shape as the
                `distance` parameter.
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

    def __str__(self) -> str:
        """Return a string describing the box."""
        msg = [
            f"Hello, this is box and my matrix is:\n{self.box_matrix}",
            f"Periodic? {self.periodic}",
        ]
        return "\n".join(msg)


