"""Test the integration methods."""
import numpy as np
import pytest
from help import FakeRandomGenerator, fake_multivariate_normal
from numpy.random import default_rng

import turtlemd.integrators
from turtlemd.integrators import (
    LangevinIntertia,
    LangevinOverdamped,
    LangevinParameter,
    VelocityVerlet,
    Verlet,
)
from turtlemd.potentials.well import DoubleWell
from turtlemd.system.box import Box
from turtlemd.system.particles import Particles
from turtlemd.system.system import System

TISMOL_POS_LANG = np.array(
    [
        -1.00000000,
        -0.99799432,
        -0.98665264,
        -0.97520352,
        -0.96807892,
        -0.96159279,
        -0.95014721,
        -0.94447852,
        -0.94017809,
        -0.93804918,
        -0.92699119,
        -0.92490484,
        -0.91348463,
        -0.90195902,
        -0.89475989,
        -0.88820117,
        -0.87668488,
        -0.87094729,
        -0.86657973,
        -0.86438545,
        -0.85326381,
        -0.85111548,
        -0.83963493,
        -0.82805057,
        -0.82079420,
        -0.81417970,
        -0.80260905,
        -0.79681846,
        -0.79239923,
        -0.79015456,
        -0.77898382,
        -0.77678760,
        -0.76526037,
        -0.75363047,
        -0.74632964,
        -0.73967169,
        -0.72805861,
        -0.72222650,
        -0.71776665,
        -0.71548223,
        -0.70427262,
        -0.70203832,
        -0.69047381,
        -0.67880735,
        -0.67147062,
        -0.66477741,
        -0.65312967,
        -0.64726344,
        -0.64276997,
        -0.64045243,
        -0.62921016,
    ]
)

TISMOL_VEL_LANG = np.array(
    [
        0.78008020,
        0.78725597,
        0.79204310,
        0.79490675,
        0.79431301,
        0.80064533,
        0.80501899,
        0.80970383,
        0.81149089,
        0.81460505,
        0.82094536,
        0.82705648,
        0.83080101,
        0.83265977,
        0.83109156,
        0.83647149,
        0.83992168,
        0.84371050,
        0.84461736,
        0.84686145,
        0.85235247,
        0.85763502,
        0.86057204,
        0.86165886,
        0.85934749,
        0.86400519,
        0.86676039,
        0.86988007,
        0.87013281,
        0.87173251,
        0.87659871,
        0.88127588,
        0.88362726,
        0.88416154,
        0.88132438,
        0.88547579,
        0.88774997,
        0.89041257,
        0.89022221,
        0.89138792,
        0.89583821,
        0.90011745,
        0.90208908,
        0.90227395,
        0.89911189,
        0.90295628,
        0.90494650,
        0.90734699,
        0.90690729,
        0.90783201,
        0.91205777,
    ]
)


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

# timestep = 0.002, gamma=0.3, beta=1
LANGEVIN_PARAMETER1 = LangevinParameter(
    c0=0.9994001799640054,
    a1=0.0019994001199818983,
    a2=np.array([[1.99960006e-06]]),
    b1=np.array([[0.0009996000898119623]]),
    b2=np.array([[0.000999800030169936]]),
    mean=[np.array([0.0, 0.0])],
    cov=[
        np.array(
            [
                [1.59928143e-09, 1.19928025e-06],
                [1.19928025e-06, 1.19928029e-03],
            ]
        ),
    ],
    cho=[
        np.array(
            [
                [3.99910169e-05, 0.00000000e00],
                [2.99887411e-02, 1.73192291e-02],
            ]
        ),
    ],
)
# timestep = 0.0005, gamma=0.5, beta=1
LANGEVIN_PARAMETER2 = LangevinParameter(
    c0=0.999750031247396,
    a1=0.0004999375052079369,
    a2=np.array([[1.24989584e-07]]),
    b1=np.array([[0.0002499583369555136]]),
    b2=np.array([[0.0002499791682524233]]),
    mean=[np.array([0.0, 0.0])],
    cov=[
        np.array(
            [
                [4.16593409e-11, 1.24968755e-07],
                [1.24968755e-07, 4.99875021e-04],
            ]
        ),
    ],
    cho=[
        np.array(
            [
                [6.45440476e-06, 0.00000000e00],
                [1.93617784e-02, 1.11801860e-02],
            ]
        ),
    ],
)


def compare_parameters(par1: LangevinParameter, par2: LangevinParameter):
    assert pytest.approx(par1.c0) == par2.c0
    assert pytest.approx(par1.a1) == par2.a1
    assert pytest.approx(par1.a2) == par2.a2
    assert pytest.approx(par1.b1) == par2.b1
    assert pytest.approx(par1.b2) == par2.b2
    for i, j in zip(par1.mean, par2.mean):
        assert pytest.approx(i) == j
    for i, j in zip(par1.cov, par2.cov):
        assert pytest.approx(i) == j
    for i, j in zip(par1.cho, par2.cho):
        assert pytest.approx(i) == j


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


def test_langevin_parameters():
    """Test the Langevin inertia integrator and generation of parameters."""
    system = create_test_system()
    integrator = LangevinIntertia(
        timestep=0.002,
        gamma=0.3,
        beta=1.0,
    )
    integrator.initiate_parameters(system)
    compare_parameters(integrator.param, LANGEVIN_PARAMETER1)

    integrator = LangevinIntertia(
        timestep=0.0005,
        gamma=0.5,
        beta=1.0,
    )
    integrator.initiate_parameters(system)
    compare_parameters(integrator.param, LANGEVIN_PARAMETER2)


def test_langevin_random():
    """Test the drawing of random numbers for the Langevin integrator."""
    system = create_test_system()
    rgen = default_rng(seed=123)
    rgen2 = default_rng(seed=123)
    integrator = LangevinIntertia(
        timestep=0.002, gamma=0.3, beta=1.0, rgen=rgen
    )
    integrator.initiate_parameters(system)
    pos, vel = integrator.draw_random_numbers(system)
    # Compare the manual method to the numpy method:
    pos2, vel2 = rgen2.multivariate_normal(
        integrator.param.mean[0], integrator.param.cov[0], method="cholesky"
    )
    assert pytest.approx(pos[0][0]) == pos2
    assert pytest.approx(vel[0][0]) == vel2


def test_langevin_integration(monkeypatch):
    """Test the integration of the equations of motion."""
    system = create_test_system()
    integrator = LangevinIntertia(
        timestep=0.002,
        gamma=0.3,
        rgen=FakeRandomGenerator(seed=1),
        beta=1.0,
    )

    with monkeypatch.context() as m:
        m.setattr(
            turtlemd.integrators,
            "multivariate_normal",
            fake_multivariate_normal,
        )
        for i in range(51):
            pos, vel = system.particles.pos, system.particles.vel
            assert pytest.approx(pos[0][0]) == TISMOL_POS_LANG[i]
            assert pytest.approx(vel[0][0]) == TISMOL_VEL_LANG[i]
            integrator(system)
