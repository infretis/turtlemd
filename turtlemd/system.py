"""Definition of the simulation system."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from turtlemd.box import Box  # pragma: no cover
    from turtlemd.particles import Particles  # pragma: no cover



class System:
    """A system the MD is run on.

    The system bridges some other objects together, like the particles
    and the box.

    """

    box: Box  # The simulation box.
    particles: Particles  # The particles in the system.
    potentials: list[Any]  # The force field.

    def __init__(
        self,
        box: Box,
        particles: Particles,
        potentials: list[Any] | None = None,
    ):
        """Initialize a new system.

        Args:
            box (Box): The simulation box.
            particles (Particles): The particles we simulate on.
        """
        self.box = box
        self.particles = particles
        self.potentials = []
        if potentials is not None:
            self.potentials = list(potentials)

    def potential_and_force(self):
        """Evaluate the potential energy and the force."""
        v_pot, force, virial = [], None, None
        for pot in self.potentials:
            v_poti, forcei, viriali = pot.potential_and_force(self)
            v_pot.append(v_poti)
            if force is None:
                force = np.zeros_like(forcei)
            if virial is None:
                virial = np.zeros_like(viriali)
            force += forcei
            virial += viriali
        if force is not None:
            self.particles.force = force
        if virial is not None:
            self.particles.virial = virial
        v_pot = sum(v_pot)
        self.particles.v_pot = v_pot
        return v_pot, force, virial

    def potential(self):
        """ "Evaluate the potential energy."""
        v_pot = sum(pot.potential(self) for pot in self.potentials)
        self.particles.v_pot = v_pot
        return v_pot

    def force(self):
        """Evaluate the force."""
        force, virial = None, None
        for pot in self.potentials:
            forcei, viriali = pot.force(self)
            if force is None:
                force = np.zeros_like(forcei)
            if virial is None:
                virial = np.zeros_like(viriali)
            force += forcei
            virial += viriali
        if force is not None:
            self.particles.force = force
        if virial is not None:
            self.particles.virial = virial
        return force, virial
