"""Definition of a simulation."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

from turtlemd.integrators import MDIntegrator
from turtlemd.simulation.stopping import MaxSteps, SoftExit
from turtlemd.system.system import System

if TYPE_CHECKING:  # pragma: no cover
    from turtlemd.simulation.stopping import StopCondition


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class CycleNumber(TypedDict):
    """For storing the current cycle number."""

    step: int  # Current step.
    steps: int  # Number of steps to do.
    start: int  # Step we start at.
    stepno: int  # The current step (counter).


class MDSimulation:
    """A generic MD simulation for a system with a integrator."""

    cycle: CycleNumber  # Information about the current step.
    first_step: bool  # If we need to initialize the potential and force.
    integrator: MDIntegrator  # The time integration of the system.
    steps: int  # The number of steps do run.
    start: int  # The step number we start from.
    stop_conditions: list[StopCondition]  # Conditions for stopping.
    system: System  # The system we simulate.

    def __init__(
        self,
        system: System,
        integrator: MDIntegrator,
        steps: int,
        start: int = 0,
        stop_conditions: None | list[StopCondition] = None,
    ):
        """Initialize the MD simulation."""
        self.system = system
        self.integrator = integrator
        self.steps = steps
        self.first_step = True

        # The start parameter is mainly used for output.
        self.cycle: CycleNumber = {
            "step": start,  # Current step.
            "steps": steps,  # Number of steps to do.
            "start": start,  # Step we start at.
            "stepno": 0,  # The current step (counter).
        }

        if stop_conditions is None:
            self.stop_conditions = [MaxSteps(), SoftExit()]
        else:
            self.stop_conditions = stop_conditions
        # Double check that we have at least one SoftExit:
        if not any(isinstance(i, SoftExit) for i in self.stop_conditions):
            self.stop_conditions.append(SoftExit())

    def step(self) -> System:
        """Do a single step of the MD simulation."""
        if self.stop():
            return self.system
        if self.first_step:
            # Just get the energies for step zero:
            self.system.potential_and_force()
            self.first_step = False
            return self.system
        self.integrator.integration_step(self.system)
        self.cycle["step"] += 1
        self.cycle["stepno"] += 1
        return self.system

    def stop(self) -> bool:
        """Check if we should stop the simulation."""
        return any(condition(self) for condition in self.stop_conditions)

    def run(self):
        """Run simulation until we should stop."""
        while not self.stop():
            yield self.step()
