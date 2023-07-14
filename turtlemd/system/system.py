"""Definition of the simulation system."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from turtlemd.system.box import Box
    from turtlemd.system.particles import Particles

from turtlemd.system.particles import (
    kinetic_energy,
    kinetic_temperature,
    pressure_tensor,
)


class Thermo(TypedDict):
    """For storing calculation of thermodynamic properties for a system."""

    ekin: None | float
    pressure: None | float
    temperature: None | float
    vpot: None | float


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

    def thermo(self, boltzmann: float = 1.0) -> Thermo:
        """Evaluate simple thermodynamic properties for the system."""
        thermo: Thermo = {
            "ekin": None,
            "pressure": None,
            "temperature": None,
            "vpot": None,
        }
        if self.particles is None or self.particles.npart == 0:
            return thermo

        particles = self.particles
        vpot = particles.v_pot
        if vpot is not None:
            vpot /= particles.npart
        thermo["vpot"] = vpot

        kin_tensor, ekin = kinetic_energy(particles)
        ekin /= particles.npart
        thermo["ekin"] = ekin

        _, pressure = pressure_tensor(
            particles, self.box.volume(), kin_tensor=kin_tensor
        )
        thermo["pressure"] = pressure

        dof = getattr(self.box, "dof", None)
        temp, _, _ = kinetic_temperature(particles, boltzmann, dof=dof)
        thermo["temperature"] = float(temp)
        return thermo
