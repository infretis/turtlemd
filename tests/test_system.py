"""Tests for the System class."""
import numpy as np
import pytest

from turtlemd.potentials.well import DoubleWell
from turtlemd.system.box import Box
from turtlemd.system.particles import Particles
from turtlemd.system.system import System


def test_system_setup():
    """Test that we can initiate a system."""
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


def test_force():
    """Test that we can evaluate the force via the system."""
    box = Box()
    particles = Particles()
    particles.add_particle(
        pos=[1.0],
    )
    potentials = [DoubleWell(a=1, b=3, c=0.0), DoubleWell(a=1, b=4, c=0.0)]
    system = System(box=box, particles=particles, potentials=potentials)
    force, virial = system.force()
    assert pytest.approx(force) == np.full_like(force, 6.0)
    assert pytest.approx(virial) == np.zeros_like(virial)


def test_potential_and_force():
    """Test that we can evaluate the potential, force, and virial."""
    box = Box()
    particles = Particles()
    particles.add_particle(
        pos=[1.0],
    )
    potentials = [DoubleWell(a=1, b=3, c=0.0), DoubleWell(a=1, b=5, c=0.0)]
    system = System(box=box, particles=particles, potentials=potentials)
    v_pot, force, virial = system.potential_and_force()
    assert pytest.approx(v_pot) == -18.0
    assert pytest.approx(force) == np.full_like(force, 8)
    assert pytest.approx(virial) == np.zeros_like(virial)
