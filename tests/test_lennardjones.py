import math

import numpy as np
import pytest

from turtlemd.box import RectangularBox
from turtlemd.particles import Particles
from turtlemd.potentials.lennardjones import (
    LennardJonesCut,
    generate_pair_interactions,
    mix_parameters,
)
from turtlemd.system import System

CORRECT_VPOT = 32480.0489507


def create_test_system():
    """Create a test system for calulcating the potential and force."""
    box = RectangularBox(size=np.array([10, 10, 10]))
    particles = Particles(dim=3)
    particles.add_particle(
        name="Ar", pos=np.array([1.0, 1.0, 1.0]), mass=1.0, ptype=0
    )
    particles.add_particle(
        name="Ar", pos=np.array([1.5, 1.0, 1.0]), mass=1.0, ptype=0
    )
    particles.add_particle(
        name="Ar", pos=np.array([1.5, 1.5, 1.0]), mass=1.0, ptype=0
    )
    system = System(box, particles)
    return system


def test_mix_geometric():
    """Test geometric mixing rules."""
    epsilon_i, epsilon_j = 2.0, 3.0
    sigma_i, sigma_j = 5.0, 6.0
    rcut_i, rcut_j = 10.0, 11.0
    mixed = mix_parameters(
        epsilon_i,
        sigma_i,
        rcut_i,
        epsilon_j,
        sigma_j,
        rcut_j,
        mixing="geometric",
    )
    assert math.isclose(mixed[0], math.sqrt(epsilon_i * epsilon_j))
    assert math.isclose(mixed[1], math.sqrt(sigma_i * sigma_j))
    assert math.isclose(mixed[2], math.sqrt(rcut_i * rcut_j))


def test_mix_arithmetic():
    """Test the arithmetic mixing rules."""
    epsilon_i, epsilon_j = 2.0, 3.0
    sigma_i, sigma_j = 5.0, 6.0
    rcut_i, rcut_j = 10.0, 11.0
    mixed = mix_parameters(
        epsilon_i,
        sigma_i,
        rcut_i,
        epsilon_j,
        sigma_j,
        rcut_j,
        mixing="arithmetic",
    )
    assert math.isclose(mixed[0], math.sqrt(epsilon_i * epsilon_j))
    assert math.isclose(mixed[1], 0.5 * (sigma_i + sigma_j))
    assert math.isclose(mixed[2], 0.5 * (rcut_i + rcut_j))


def test_mix_sixthpower():
    """Test the sixthpower mixing rules."""
    epsilon_i, epsilon_j = 2.0, 3.0
    sigma_i, sigma_j = 5.0, 6.0
    rcut_i, rcut_j = 10.0, 11.0
    mixed = mix_parameters(
        epsilon_i,
        sigma_i,
        rcut_i,
        epsilon_j,
        sigma_j,
        rcut_j,
        mixing="sixthpower",
    )
    si3 = sigma_i**3
    si6 = si3**2
    sj3 = sigma_j**3
    sj6 = sj3**2
    avgs6 = 0.5 * (si6 + sj6)
    epsilon_ij = math.sqrt(epsilon_i * epsilon_j) * si3 * sj3 / avgs6
    sigma_ij = avgs6 ** (1.0 / 6.0)
    rcut_ij = (0.5 * (rcut_i**6 + rcut_j**6)) ** (1.0 / 6.0)
    assert math.isclose(mixed[0], epsilon_ij)
    assert math.isclose(mixed[1], sigma_ij)
    assert math.isclose(mixed[2], rcut_ij)


def test_unkown_mixing():
    """Test that we fail for an unknow mixing rule."""
    with pytest.raises(ValueError):
        mix_parameters(1, 2, 3, 4, 5, 6, mixing="?")


def test_generte_pair():
    """Test that we generate correct Lennard-Jones parameters."""
    parameters = {
        0: {"sigma": 1.0, "epsilon": 1.5, "rcut": 2.0},
        1: {"sigma": 2.0, "epsilon": 2.5, "rcut": 3.0},
        2: {"sigma": 3.0, "epsilon": 3.5, "rcut": 4.0},
    }
    mix = generate_pair_interactions(parameters, "arithmetic")
    correct_mix = {
        (0, 0): {"rcut": 2.0, "epsilon": 1.5, "sigma": 1.0},
        (1, 1): {"rcut": 3.0, "epsilon": 2.5, "sigma": 2.0},
        (2, 2): {"rcut": 4.0, "epsilon": 3.5, "sigma": 3.0},
        (0, 1): {"rcut": 2.5, "epsilon": 1.9364916731037085, "sigma": 1.5},
        (1, 0): {"rcut": 2.5, "epsilon": 1.9364916731037085, "sigma": 1.5},
        (0, 2): {"rcut": 3.0, "epsilon": 2.2912878474779199, "sigma": 2.0},
        (2, 0): {"rcut": 3.0, "epsilon": 2.2912878474779199, "sigma": 2.0},
        (1, 2): {"rcut": 3.5, "epsilon": 2.9580398915498081, "sigma": 2.5},
        (2, 1): {"rcut": 3.5, "epsilon": 2.9580398915498081, "sigma": 2.5},
    }
    for key in correct_mix:
        assert key in mix
        assert correct_mix[key] == mix[key]


def test_generate_pair_mix():
    """Test that we generate correct interactions for mixed input."""
    case1 = {
        0: {"sigma": 1.0, "epsilon": 1.5, "rcut": 2.0},
        1: {"sigma": 2.0, "epsilon": 2.5, "rcut": 3.0},
        (0, 1): {"sigma": 1.0, "epsilon": 1.0, "rcut": 100.0},
    }
    case2 = {
        0: {"sigma": 1.0, "epsilon": 1.5, "rcut": 2.0},
        1: {"sigma": 2.0, "epsilon": 2.5, "rcut": 3.0},
        (1, 0): {"sigma": 1.0, "epsilon": 1.0, "rcut": 100.0},
    }
    for case in (case1, case2):
        mix = generate_pair_interactions(case, mixing="arithmetic")
        assert mix[0, 0] == case[0]
        assert mix[1, 1] == case[1]
        assert mix[0, 1] == mix[1, 0]


def test_initiate_lennard_jones():
    """Test that we can initiate and set parameters for LJ."""
    pot = LennardJonesCut(dim=3, shift=True, mixing="geometric")
    parameters = {
        0: {"sigma": 1, "epsilon": 1, "rcut": 2.5},
        1: {"sigma": 2, "epsilon": 2, "rcut": 5.0},
    }
    pot.set_parameters(parameters)
    expected_parameters = {
        (0, 0): {"sigma": 1, "epsilon": 1, "rcut": 2.5},
        (1, 1): {"sigma": 2, "epsilon": 2, "rcut": 5.0},
        (0, 1): {
            "sigma": math.sqrt(1 * 2),
            "epsilon": math.sqrt(2 * 1),
            "rcut": math.sqrt(5.0 * 2.5),
        },
        (1, 0): {
            "sigma": math.sqrt(1 * 2),
            "epsilon": math.sqrt(2 * 1),
            "rcut": math.sqrt(5.0 * 2.5),
        },
    }
    assert len(expected_parameters) == len(pot.params)
    for key in expected_parameters:
        assert key in pot.params
        assert expected_parameters[key] == pot.params[key]
    # Test that we do not fail for zero:
    parameters = {
        0: {"sigma": 1, "epsilon": 1, "rcut": 0.0},
    }
    pot.set_parameters(parameters)
    assert math.isclose(pot.params[0, 0]["rcut"], 0.0)


def test_potential():
    """Test that we can calculate the LJ potential."""
    system = create_test_system()
    pot = LennardJonesCut(dim=3, shift=True, mixing="geometric")
    parameters = {
        0: {"sigma": 1, "epsilon": 1, "rcut": 2.5},
    }
    pot.set_parameters(parameters)
    vpot = pot.potential(system)
    assert math.isclose(vpot, CORRECT_VPOT)
