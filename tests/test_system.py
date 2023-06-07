import pytest

from turtlemd.box import Box
from turtlemd.particles import Particles
from turtlemd.potentials.well import DoubleWell
from turtlemd.system import System


def test_system_setup():
    box = Box()
    particles = Particles()
    potentials = [DoubleWell()]
    system = System(box=box, particles=particles, potentials=potentials)
    assert system.box is box
    assert system.particles is particles
    assert system.potentials[0] is potentials[0]
    assert len(system.potentials) == 1


def test_potential():
    """Test that we can evaluate the potential via the system."""
    box = Box()
    particles = Particles()
    particles.add_particle(
        pos=[1.0],
    )
    potentials = [DoubleWell(a=1, b=3, c=0.0), DoubleWell(a=1, b=2, c=0.0)]
    system = System(box=box, particles=particles, potentials=potentials)
    vpot = system.potential()
    assert pytest.approx(vpot) == -9.0
