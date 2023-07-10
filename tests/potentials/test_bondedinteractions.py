import pathlib

import numpy as np
import pytest
from jax.config import config

import turtlemd as tmd
import turtlemd.integrators
import turtlemd.potentials
import turtlemd.system
from turtlemd.potentials.jax_bondedinteractions import BondedInteractions

config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent


def rotation(xyz, a, theta):
    """Rotate particles coordinates xyz theta radians along vector a."""
    a = a / np.linalg.norm(a)
    C = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.identity(3) + C * np.sin(theta) + C @ C * (1 - np.cos(theta))
    return R @ xyz


def create_test_system():
    """Create a simple 4-particle test system containing 3 bonds,
    2 angles and 1 dihedral.

    y
    ^
    | (1)  (4)
    |  |    |
    | (2)--(3)
    ----------> z

    """
    box = tmd.system.box.Box(
        low=[-2, -2, -2], high=[2, 2, 2], periodic=[True, True, True]
    )

    # create particles
    particles = tmd.system.particles.Particles()
    initial_conf = np.array(
        [
            [-0.05, 0.10, 0.00],
            [-0.05, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.05, 0.10, 0.00],
        ]
    )
    for xyz in initial_conf:
        particles.add_particle(xyz, mass=1.008, name="H")

    # force field
    # units in kJ/mol, nm, and ps (same as gromacs)
    potentials = [
        BondedInteractions(
            bonds=[
                # k         b0   i  j
                (200000.0, 0.1, 0, 1),
                (200000.0, 0.1, 1, 2),
                (200000.0, 0.1, 2, 3),
            ],
            angles=[
                # k      ang0           i  j  k
                (400.0, np.deg2rad(90), 0, 1, 2),
                (400.0, np.deg2rad(90), 1, 2, 3),
            ],
            dihedrals=[
                # k    ang0           n    i  j  k  l
                (8.0, np.deg2rad(180), 2.0, 0, 1, 2, 3)
            ],
        )
    ]

    system = tmd.system.system.System(
        box=box, particles=particles, potentials=potentials
    )
    return system


def test_angle_potential():
    """Test that rotation of the  angle between atoms 1-2-3 from -45 to 45
    degrees gives the correct potential energy surface"""
    system = create_test_system()
    Npoints = 100
    ang = np.deg2rad(np.linspace(-45, 45, Npoints))
    potential = np.zeros(Npoints)
    k, ang0 = system.potentials[0].angles[0][:2]
    true_potential = (
        0.5 * k * (np.deg2rad(90) + ang - ang0) ** 2
    )  # we start at 90 degrees
    rij = system.particles.pos[0] - system.particles.pos[1]
    rkj = system.particles.pos[2] - system.particles.pos[1]
    pik = 0.5 * np.cross(rij, rkj)

    for i in range(ang.shape[0]):
        system.particles.pos[0] = (
            rotation(rij, pik, ang[i]) + system.particles.pos[1]
        )
        potential[i] = system.potential()

    assert pytest.approx(potential) == true_potential


def test_dihedral_potential():
    """Test that rotation of particle 4 from 0 to 360 degrees along the
    middle bond gives the correct potential energy surface"""
    system = create_test_system()
    Npoints = 100
    ang = np.deg2rad(np.linspace(0, 360, Npoints))
    potential = np.zeros(Npoints)
    k, ang0, n = system.potentials[0].dihedrals[0][:3]
    true_potential = k * (1.0 + np.cos(n * ang - ang0))
    # save original location
    pos0 = system.particles.pos[3] * 1
    rkj = system.particles.pos[1] - system.particles.pos[2]

    for i in range(ang.shape[0]):
        system.particles.pos[3] = rotation(pos0, rkj, ang[i])
        potential[i] = system.potential()

    assert pytest.approx(potential) == true_potential


def test_dynamics():
    """Test that we get the same trajectory as in gromacs by
    integrating 10 steps"""
    system = create_test_system()
    integrator = tmd.integrators.VelocityVerlet(timestep=0.0005)

    true_pos, true_vel, true_force = np.load(HERE / "gromacs_traj.npy")

    pos, vel, force = true_pos * 0, true_vel * 0, true_force * 0
    # initial conditions
    system.particles.pos = true_pos[0] * 1
    system.particles.force = true_force[0] * 1
    system.particles.vel = true_vel[0] * 1

    for step in range(11):
        pos[step] = system.particles.pos
        vel[step] = system.particles.vel
        force[step] = system.particles.force
        integrator(system)
    assert pytest.approx(pos, 0.001) == true_pos
    assert pytest.approx(vel, 0.001) == true_vel
    assert pytest.approx(force, 0.001) == true_force
