"""Define a generic potential function.

This module defines the generic class for potential functions.
This class is sub-classed in all potential functions.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from turtlemd.system.system import System

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class Potential(ABC):
    """Base class for potential functions."""

    desc: str  # Short description of the potential
    dim: int  # The dimensionality of the potential.
    # Parameters for the potential:
    params: Any

    def __init__(self, dim: int = 1, desc: str = ""):
        """Initialise the potential."""
        self.dim = dim
        self.desc = desc
        self.params = None

    def set_parameters(self, parameters: Any):
        """Set parameters for the potential."""
        msg = (
            "Set parameters used, but it is not implemented by the"
            " potential - ignoring the given parameters: %s"
        )
        LOGGER.info(msg, parameters)

    @abstractmethod
    def potential(self, system: System) -> float:
        """Evaluate the potential energy."""

    @abstractmethod
    def force(self, system: System) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the forces and the virial."""

    def potential_and_force(
        self, system: System
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate potential & force.

        It may be more efficient to do both togehter. If not changed
        in the subclasses, it will just use the other functions.
        """
        vpot = self.potential(system)
        force, virial = self.force(system)
        return vpot, force, virial

    def __str__(self) -> str:
        """Return a description of the potential."""
        msg = [f"Potential: {self.desc}"]
        for key in sorted(self.params):
            msg.append(f"{key}: {self.params[key]}")
        return "\n".join(msg)
