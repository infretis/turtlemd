import math

import numpy as np

from turtlemd.particles import Particles


def create_some_particles(masses):
    """Set up a particle object for testing."""
    particles = Particles(dim=3)
    for i in masses:
        particles.add_particle(np.zeros(3), mass=i)
    return particles


def test_creation():
    """Test the creation a particle list."""
    for dim in range(1, 4):
        particles = Particles(dim=dim)
        assert particles.dim == dim
        particles.add_particle(
            pos=np.zeros(dim),
            vel=np.zeros(dim),
            force=np.zeros(dim),
            name="A",
            ptype=1,
        )
        particles.add_particle(
            pos=np.ones(dim),
            vel=np.ones(dim),
            force=np.zeros(dim),
            name="B",
            ptype=2,
            mass=2,
        )
        assert particles.npart == 2
        assert particles.pos.shape == (2, dim)
        assert particles.mass.shape == (2, 1)
        assert math.isclose(particles.mass[0, 0], 1.0)
        assert math.isclose(particles.imass[1, 0], 0.5)


def test_empty_list():
    """Test that we can empty the list."""
    particles = Particles(dim=3)
    for _ in range(10):
        particles.add_particle(pos=np.zeros(3))
    assert particles.npart == 10
    assert particles.pos.shape == (10, 3)
    assert len(particles.ptype) == 10
    particles.empty()
    assert particles.npart == 0
    assert len(particles.ptype) == 0
    particles.add_particle(pos=np.ones(3))
    assert particles.npart == 1


def test_iterate():
    """Test that we can iterate over the particles."""
    particles = Particles(dim=3)
    for i in range(10):
        particles.add_particle(pos=np.ones(3) * i, name=f"A{i}")
    for i, part in enumerate(particles):
        assert np.allclose(part["pos"], particles.pos[i])


def test_pairs():
    """Test that we can iterate over pairs."""
    particles = Particles(dim=3)
    npart = 21
    for i in range(npart):
        particles.add_particle(pos=np.ones(3) * i, ptype=i)
    pairs = set([(pair[0], pair[1]) for pair in particles.pairs()])
    n_pairs = (npart * (npart - 1)) / 2
    assert len(pairs) == n_pairs


def test_linear_momentum():
    """Test calculation of linear momentum."""
    particles = create_some_particles([1, 0.5, 2, 1])
    particles.vel = np.array(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
    )
    _, _, mom = particles.linear_momentum()
    assert np.allclose(mom, [4, 4, 4])


def test_zero_momentum():
    """Test that we can zero out the momentum."""
    particles = create_some_particles([1, 0.5, 2, 1])
    particles.vel = np.array(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
    )
    vel = np.copy(particles.vel)
    particles.zero_momentum()
    mom = particles.linear_momentum()
    assert np.allclose(mom, np.zeros(3))
    particles.vel = vel
    particles.zero_momentum(dim=[False, True, False])
    mom = particles.linear_momentum()
    assert np.allclose(mom, [4.0, 0.0, 4.0])


def test_kinetic_energy():
    """Test calculation of kinetic energy."""
    particles = create_some_particles([1, 2, 1, 2])
    particles.vel = np.array(
        [[1.0, 1.0, 1.0], [2.1, 2.1, 2.1], [3.1, 3.1, 3.1], [4.1, 4.1, 4.1]]
    )
    correct = np.full((3, 3), 26.525)
    kin_tensor, kin = particles.kinetic_energy()
    assert math.isclose(kin, 79.575)
    assert np.allclose(kin_tensor, correct)
    mom = particles.vel * particles.mass
    kin_tensor2 = 0.5 * mom.T @ particles.vel
    assert np.allclose(kin_tensor, kin_tensor2)
    particles = create_some_particles([10])
    particles.vel = np.array([1.0, 2.0, 3.0])
    kin_tensor, kin = particles.kinetic_energy()
    assert math.isclose(70, kin)
    correct = np.array(
        [[5.0, 10.0, 15.0], [10.0, 20.0, 30.0], [15.0, 30.0, 45.0]]
    )
    assert np.allclose(kin_tensor, correct)


def test_kinetic_temp():
    """Test calculation of kinetic temperature."""
    particles = create_some_particles([1, 2, 1, 0.01])
    particles.vel = np.ones_like(particles.pos)
    temp1, temp2, _ = particles.kinetic_temperature(1.0)
    assert math.isclose(temp1, 1.0025)
    for i in temp2:
        assert math.isclose(i, 1.0025)
    temp1, temp2, _ = particles.kinetic_temperature(1.0, dof=[1.0, 0.0, 0.0])
    assert math.isclose(temp1, 1.11388888889)
    for i, j in zip(temp2, (1.3366666667, 1.0025, 1.0025)):
        assert math.isclose(i, j)


def test_pressure_tensors():
    """Test calculation of the pressure tensor and scalar."""
    particles = create_some_particles([1, 2, 3, 4])
    particles.vel = np.array(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
    )
    press, scalar = particles.pressure_tensor(1.0)
    for i in press.ravel():
        assert math.isclose(i, 13.75)
    assert math.isclose(scalar, 13.75)
