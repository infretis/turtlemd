"""Definition of the simulation system."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from turtlemd.box import Box
    from turtlemd.particles import Particles
    from turtlemd.potentials.potential import Potential


class System:
    """A system the MD is run on.

    The system bridges some other objects together, like the particles
    and the box.

    """

    box: Box  # The simulation box.
    particles: Particles  # The particles in the system.
    potentials: list[Potential]

    def __init__(
        self,
        box: Box,
        particles: Particles,
        potentials: list[Potential] | None = None,
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
            self.potentials = [i for i in potentials]

    def potential_and_force(self):
        """Evaluate the potential energy and the force."""
        v_pot, force, virial = None, None, None
        for pot in self.potentials:
            v_poti, forcei, viriali = pot.potential_and_force(self)
            if v_pot is None:
                v_pot = v_poti
            else:
                v_pot += v_poti
            if force is None:
                force = forcei
            else:
                force += forcei
            if virial is None:
                virial = viriali
            else:
                virial += viriali
        return v_pot, force, virial

    def potential(self):
        """ "Evaluate the potential energy."""
        v_pot = None
        for pot in self.potentials:
            v_poti = pot.potential(self)
            if v_pot is None:
                v_pot = v_poti
            else:
                v_pot += v_poti
        return v_pot

    def force(self):
        """Evaluate the force."""
        force, virial = None, None
        for pot in self.potentials:
            forcei, viriali = pot.force(self)
            if force is None:
                force = forcei
            else:
                force += forcei
            if virial is None:
                virial = viriali
            else:
                virial += viriali
        return force, virial

    def get_dim(self):
        return self.particles.dim
