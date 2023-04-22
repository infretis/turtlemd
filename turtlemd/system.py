"""Definition of the simulation system."""
from turtlemd.particles import Particles

class System:
    """A system the MD is run on.

    The system bridges some other objects together,
    like the particles and the box.

    Attributes
    ----------
    box : object like :py:class:`.Box`
        The simulation box
    particles :
    """
    box : str
    particles : Particles

    def __init__(self):
        pass
