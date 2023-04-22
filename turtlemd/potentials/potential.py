"""Define a generic potential function.

This module defines the generic class for potential functions.
This class is sub-classed in all potential functions.
"""
import logging
from abc import ABC, abstractmethod

import numpy as np

from turtlemd.system import System

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class PotentialFunction(ABC):
    """Base class for potential functions.

    Attributes
    ----------
    desc : string
        Short description of the potential.
    dim : int
        Represents the spatial dimensionality of the potential.
    params : dict
        The parameters for the potential. This dict defines,
        on initiation, the parameters the potential will handle
        and store. This is assumed to be set in the subclasses.

    """

    desc: str  # Short description of the potential
    dim: int  # The dimensionality of the potential.
    params: dict[str, float]  # Parameters for the potential.

    def __init__(self, dim: int = 1, desc: str = ""):
        """Initialise the potential.

        Parameters
        ----------
        dim : int, optional
            Represents the dimensionality.
        desc : string, optional
            Description of the potential function. Used to print out
            information about the potential.

        """
        self.dim = dim
        self.desc = desc
        self.params = {}

    def update_parameters(self, parameters: dict[str, float]) -> bool:
        """Update/set parameters."""
        for key in parameters:
            if key in self.params:
                self.params[key] = parameters[key]
            else:
                msg = 'Could not find "%s" in parameters. Ignoring!'
                LOGGER.warning(msg % key)
        return self.check_parameters()

    def check_parameters(self) -> bool:
        """Check the consistency of the parameters.

        Returns
        -------
        out : boolean
            True if the check(s) pass.

        """
        if not self.params:
            LOGGER.warning("No parameters are set for the potential")
            return False
        return True

    @abstractmethod
    def potential(self, system: System) -> float:
        """Evaluate the potential energy."""

    @abstractmethod
    def force(self, system: System) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the forces and the virial."""

    def potential_and_force(
        self, system: System
    ) -> tuple[float, np.ndarray, np.ndarray]:
        vpot = self.potential(system)
        force, virial = self.force(system)
        return vpot, force, virial

    def __str__(self) -> str:
        """Return a description of the potential."""
        msg = [f"Potential: {self.desc}"]
        for key in sorted(self.params):
            msg.append(f"{key}: {self.params[key]}")
        return "\n".join(msg)
