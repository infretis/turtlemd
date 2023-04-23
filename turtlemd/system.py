"""Definition of the simulation system."""
from turtlemd.box import Box
from turtlemd.particles import Particles


class System:
    """A system the MD is run on.

    The system bridges some other objects together, like the particles
    and the box.

    """

    box: Box  # The simulation box.
    particles: Particles  # The particles in the system.

    def __init__(self, box: Box, particles: Particles):
        """Initialize a new system.

        Args:
            box (Box): The simulation box.
            particles (Particles): The particles we simulate on.
        """
        self.box = box
        self.particles = particles
