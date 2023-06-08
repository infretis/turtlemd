"""Test the integration methods."""
import numpy as np
import pytest
from help import FakeRandomGenerator

from turtlemd.box import Box
from turtlemd.integrators import LangevinOverdamped, VelocityVerlet, Verlet
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

TISMOL_POS_LANG_OVER = np.array(
    [
        -1.00000000,
        -0.95540084,
        -0.19165497,
        0.78018908,
        1.32632619,
        2.28604749,
        2.39185991,
        2.64430617,
        2.25348211,
        2.49825935,
        2.88324951,
        3.30550209,
        2.98182643,
        2.76033520,
        2.86756839,
        2.58347298,
        2.50423709,
        2.20295643,
        2.42537267,
        3.05261057,
        3.15554594,
        2.44639287,
        2.88716400,
        3.29914672,
        2.96754315,
        3.33636578,
        2.79841879,
        2.84221288,
        2.33746733,
        2.54907741,
        2.90952943,
        3.31484556,
        2.98322875,
        2.76077697,
        2.86775262,
        2.58354092,
        2.50427057,
        2.20297400,
        2.42538389,
        3.05261681,
        3.15554769,
        2.44639327,
        2.88716422,
        3.29914680,
        2.96754316,
        3.33636579,
        2.79841879,
        2.84221288,
        2.33746733,
        2.54907741,
        2.90952943,
    ]
)

TISMOL_VEL_LANG_OVER = np.array(
    [
        0.78008020,
        0.04459916,
        0.76596773,
        0.97676712,
        0.53799599,
        0.98657113,
        0.36343554,
        0.55356508,
        0.03172585,
        0.48984683,
        0.73416686,
        0.98453450,
        0.55129904,
        0.40598753,
        0.59448391,
        0.26823255,
        0.31168371,
        0.05072849,
        0.44876367,
        0.94301707,
        0.78008020,
        0.04459916,
        0.76596773,
        0.97676712,
        0.53799599,
        0.98657113,
        0.36343554,
        0.55356508,
        0.03172585,
        0.48984683,
        0.73416686,
        0.98453450,
        0.55129904,
        0.40598753,
        0.59448391,
        0.26823255,
        0.31168371,
        0.05072849,
        0.44876367,
        0.94301707,
        0.78008020,
        0.04459916,
        0.76596773,
        0.97676712,
        0.53799599,
        0.98657113,
        0.36343554,
        0.55356508,
        0.03172585,
        0.48984683,
        0.73416686,
    ]
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


def test_langevin_brownian():
    """Test the overdamped Langevin integrator."""
    system = create_test_system()
    integrator = LangevinOverdamped(
        timestep=0.002,
        gamma=0.3,
        rgen=FakeRandomGenerator(seed=1),
        beta=1.0,
    )
    for i in range(51):
        assert (
            pytest.approx(system.particles.pos[0][0])
            == TISMOL_POS_LANG_OVER[i]
        )
        assert (
            pytest.approx(system.particles.vel[0][0])
            == TISMOL_VEL_LANG_OVER[i]
        )
        integrator(system)
