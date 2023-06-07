import logging

import numpy as np

from turtlemd.box import RectangularBox
from turtlemd.particles import Particles
from turtlemd.potentials.potential import Potential
from turtlemd.potentials.well import (
    DoubleWell,
    RectangularWell,
)
from turtlemd.system import System


def create_test_system(n: int) -> System:
    """Create a test system for calulcating the potential and force."""
    particles = Particles(dim=1)
    box = RectangularBox(size=[10])
    for _ in range(n):
        particles.add_particle(np.zeros(1))
    particles.pos = np.random.random(particles.pos.shape)
    system = System(box, particles)
    return system


def test_init_doublewell():
    well = DoubleWell(a=1.0, b=2.0, c=-1.0)
    assert well.desc == "1D double well potential"
    assert isinstance(well, Potential)
    params = {"a": 1.0, "b": 2.0, "c": -1.0}
    assert well.params == params


def test_set_parameters(caplog):
    with caplog.at_level(logging.INFO):
        well = DoubleWell(a=0.0, b=0.0, c=0.0)
        well.set_parameters(parameters="this-is-just-ignored")
        assert "ignoring the given parameters" in caplog.text
        assert well.params == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_str(capsys):
    well = DoubleWell(a=1.0, b=2.0, c=3.0)
    print(well)
    captured = capsys.readouterr()
    assert "1D double well potential" in captured.out


def test_doublewell_potential_force():
    well = DoubleWell(a=1.0, b=2.0, c=3.0)
    a = well.params["a"]
    b = well.params["b"]
    c = well.params["c"]
    # Test for one particle:
    system = create_test_system(1)
    system.particles.pos *= 0.0
    vpot0 = -b * c * c
    force0 = -2.0 * b * c
    vpot1 = well.potential(system)
    force1, virial1 = well.force(system)
    vpot2, force2, virial2 = well.potential_and_force(system)
    assert vpot0 == vpot1
    assert vpot1 == vpot2
    assert force0 == force1[0][0]
    assert np.allclose(force1, force2)
    assert np.allclose(virial1, np.zeros_like(virial1))
    assert np.allclose(virial1, virial2)
    # Test for 11 particles:
    system = create_test_system(11)
    x = system.particles.pos
    vpot0 = np.sum(a * x**4 - b * (x - c) ** 2, axis=0)
    force0 = -4.0 * a * x**3 + 2.0 * b * (x - c)
    vpot1 = well.potential(system)
    force1, virial1 = well.force(system)
    vpot2, force2, virial2 = well.potential_and_force(system)
    assert vpot0.sum() == vpot1
    assert vpot1 == vpot2
    assert np.allclose(force0, force1)
    assert np.allclose(force1, force2)
    assert np.allclose(virial1, np.zeros_like(virial1))
    assert np.allclose(virial1, virial2)


def test_rectangular_well(caplog):
    with caplog.at_level(logging.WARNING):
        RectangularWell(left=124, right=123)
        assert "Setting left >= right in RectangularWell" in caplog.text
    well = RectangularWell(left=-1, right=123)
    system = create_test_system(1)
    system.particles.pos *= 0
    v_pot = well.potential(system)
    assert v_pot == 0.0
    system.particles.pos += 124
    v_pot = well.potential(system)
    assert v_pot == float("inf")
    system.particles.pos -= 125
    v_pot = well.potential(system)
    assert v_pot == float("inf")
    system.particles.pos += 0.001
    v_pot = well.potential(system)
    assert v_pot == 0.0
    with caplog.at_level(logging.WARNING):
        force, virial = well.force(system)
        assert np.allclose(force, np.zeros_like(force))
        assert np.allclose(virial, np.zeros_like(virial))
        assert "Calling force for 1D Rectangular well" in caplog.text
    assert "not implemented!" in caplog.text
