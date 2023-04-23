"""Definition of the simulation system."""
from turtlemd.box import Box
from turtlemd.particles import Particles


class System:
    """A system the MD is run on.

    The system bridges some other objects together,
    like the particles and the box.

    Attributes
    ----------
    box : object like :py:class:`.Box`
        The simulation box
    particles : object like :py:class:`.Particles`
        The particles we simulate on.
    """

    box: Box
    particles: Particles

    def __init__(self, box: Box, particles: Particles):
        self.box = box
        self.particles = particles
