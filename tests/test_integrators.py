"""Test the integration methods."""
import numpy as np
import pytest

from turtlemd.box import Box
from turtlemd.integrators import VelocityVerlet, Verlet
from turtlemd.particles import Particles
from turtlemd.potentials.well import DoubleWell
from turtlemd.system import System

TISMOL_POS_VV = np.array(
    [
        -1.00000000,
        -0.99843984,
        -0.99687973,
        -0.99531972,
        -0.99375986,
        -0.99220019,
        -0.99064077,
        -0.98908165,
        -0.98752287,
        -0.98596448,
        -0.98440654,
        -0.98284908,
        -0.98129215,
        -0.97973581,
        -0.97818009,
        -0.97662505,
        -0.97507074,
        -0.97351719,
        -0.97196445,
        -0.97041258,
        -0.96886161,
    ]
)

TISMOL_VEL_VV = np.array(
    [
        0.78008020,
        0.78006775,
        0.78003045,
        0.77996842,
        0.77988179,
        0.77977066,
        0.77963517,
        0.77947542,
        0.77929154,
        0.77908365,
        0.77885188,
        0.77859634,
        0.77831715,
        0.77801444,
        0.77768833,
        0.77733895,
        0.77696642,
        0.77657086,
        0.77615240,
        0.77571116,
        0.77524727,
    ]
)


CORRECT_POS_V = np.array(
    [-0.99843984, -0.99687973, -0.99531972, -0.99375986, -0.99220019]
)
CORRECT_VEL_V = np.array(
    [0.78008018, 0.78006773, 0.78003043, 0.77996841, 0.77988177]
)


def create_test_system():
    """Create a system to test on."""
    box = Box(periodic=[False])
    potentials = [DoubleWell(a=1.0, b=2.0, c=0.0)]
    system = System(
        box=box,
        particles=Particles(dim=box.dim),
        potentials=potentials,
    )
    system.particles.add_particle(
        pos=np.array([-1.0]),
        vel=np.array([0.78008020]),
        name="Ar",
        mass=1.0,
        ptype=0,
    )
    return system


def test_velocity_verlet():
    """Test integration with Velocity Verlet."""
    system = create_test_system()
    integrator = VelocityVerlet(timestep=0.002)
    for i in range(21):
        assert pytest.approx(system.particles.pos[0][0]) == TISMOL_POS_VV[i]
        assert pytest.approx(system.particles.vel[0][0]) == TISMOL_VEL_VV[i]
        if i % 2 == 0:
            integrator.integration_step(system)
        else:
            integrator(system)  # Check that the integrator is callable.


def test_verlet():
    """Test integration with Verlet."""
    system = create_test_system()
    integrator = Verlet(timestep=0.002)
    assert integrator.previous_pos is None
    prev = np.copy(system.particles.pos)
    for i in range(5):
        if i % 2 == 0:
            integrator.integration_step(system)
        else:
            integrator(system)  # Chat that the integrator is callable
        assert pytest.approx(system.particles.pos[0][0]) == CORRECT_POS_V[i]
        assert pytest.approx(system.particles.vel[0][0]) == CORRECT_VEL_V[i]
        assert pytest.approx(prev) == integrator.previous_pos
        prev = np.copy(system.particles.pos)
