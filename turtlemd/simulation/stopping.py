"""Definitions for stopping a simulation."""
from __future__ import annotations

import logging
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from turtlemd.simulation.mdsimulation import (
        MDSimulation,
    )


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class StopCondition(ABC):
    """Base class for stopping conditions."""

    @abstractmethod
    def stop(self, simulation: MDSimulation) -> bool:
        """Check the stopping condition."""

    def __call__(self, simulation: MDSimulation) -> bool:
        """Check the stopping condition."""
        return self.stop(simulation)


class SoftExit(StopCondition):
    """Stop if we detect a `EXIT` file."""

    def stop(self, simulation: MDSimulation) -> bool:
        """Check if we are to do a soft exit."""
        exe_dir = getattr(simulation, "exe_dir", pathlib.Path())
        exit_file = exe_dir / "EXIT"
        if exit_file.is_file():
            LOGGER.info("Found exit file - will exit at the next step.")
            return True
        return False


class MaxSteps(StopCondition):
    """Stop if we surpass the maximum number of steps."""

    def stop(self, simulation: MDSimulation) -> bool:
        """Check if we are done."""
        return simulation.cycle["stepno"] >= simulation.cycle["steps"]
