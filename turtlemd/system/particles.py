"""Define a list of particles.

Tha particle list collects the mass, positions, velocities,
forces, types, etc. for a collection of particles.
"""
from collections.abc import Iterator
from typing import SupportsIndex

import numpy as np
from numpy.random import Generator


class Particles:
    """A class representing a collection of particles."""

    npart: int  # The number of particles in the list
    dim: int  # The number of dimensions (1-3)
    pos: np.ndarray  # Positions for the particles
    vel: np.ndarray  # Velocities for the particles
    force: np.ndarray  # Force on the particles
    virial: np.ndarray  # Virial for the particles
    v_pot: float | None  # The potential energy of the particles.
    mass: np.ndarray  # The mass of the particles
    imass: np.ndarray  # The inverse mass of the particles
    name: np.ndarray  # Names of the particles
    ptype: np.ndarray  # Types of the particles

    def __init__(self, dim: int = 3):
        """Initialize an empty particle list.

        Args:
            dim: The number of dimensions we have.
        """
        self.dim = dim
        self.empty()

    def empty(self):
        """Empty the particle list and remove all data."""
        self.npart = 0
        self.v_pot = None
        self.pos = np.zeros((1, self.dim))
        self.vel = np.zeros_like(self.pos)
        self.force = np.zeros_like(self.pos)
        self.mass = np.zeros((1, 1))
        self.imass = np.zeros_like(self.mass)
        self.name = np.array([], dtype=str)
        self.ptype = np.array([], dtype=int)
        self.virial = np.zeros((self.dim, self.dim))

    def add_particle(
        self,
        pos: np.ndarray | list[float],
        vel: np.ndarray | None = None,
        force: np.ndarray | None = None,
        mass: float = 1.0,
        name: str = "?",
        ptype: int = 1,
    ):
        """Add a single particle to the particle list."""
        if vel is None:
            vel = np.zeros_like(pos)
        if force is None:
            force = np.zeros_like(pos)

        if self.npart == 0:
            self.pos[0] = pos
            self.vel[0] = vel
            self.force[0] = force
            self.mass[0] = mass
        else:
            self.pos = np.vstack([self.pos, pos])
            self.vel = np.vstack([self.vel, vel])
            self.force = np.vstack([self.force, force])
            self.mass = np.vstack([self.mass, mass])

        self.name = np.append(self.name, name)
        self.ptype = np.append(self.ptype, ptype)
        self.imass = 1.0 / self.mass
        self.npart += 1

    def __iter__(self) -> Iterator[dict]:
        """Yield the properties of the particles.

        Yields:
            out: The information in `self.pos`, `self.vel`, ... etc.
        """
        for i, pos in enumerate(self.pos):
            part = {
                "pos": pos,
                "vel": self.vel[i],
                "force": self.force[i],
                "mass": self.mass[i],
                "imass": self.imass[i],
                "name": self.name[i],
                "type": self.ptype[i],
            }
            yield part

    def __len__(self) -> int:
        """Just give the number of particles."""
        return self.npart

    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...]):
        """Support slicing for particles."""
        part = Particles(dim=self.dim)
        part.pos = self.pos[key]
        part.vel = self.vel[key]
        part.force = self.force[key]
        part.mass = self.mass[key]
        part.imass = self.imass[key]
        part.name = self.name[key]
        part.ptype = self.ptype[key]
        part.virial = self.virial = np.zeros(
            (self.dim, self.dim)
        )  # Should be recalculated.
        part.v_pot = None
        part.npart = len(part.name)
        return part

    def pairs(self) -> Iterator[tuple[int, int, int, int]]:
        """Iterate over all pairs of particles.

        Yields:
            out[0]: The index for the first particle in the pair.
            out[1]: The index for the second particle in the pair.
            out[2]: The particle type of the first particle.
            out[3]: The particle type of the second particle.

        """
        for i, itype in enumerate(self.ptype[:-1]):
            for j, jtype in enumerate(self.ptype[i + 1 :]):
                yield (i, i + 1 + j, itype, jtype)


def linear_momentum(particles: Particles) -> np.ndarray:
    """Return linear momentum of the particles."""
    return np.sum(particles.vel * particles.mass, axis=0)


def zero_momentum(particles: Particles, dim: list[bool] | None = None):
    """Set the linear momentum for the particles to zero.

    Args:
        dim: If None, the momentum will be reset for ALL dimensions.
            If a list is given, the momentum will only be reset where
            `dim[i]` is True.
    """
    mom = linear_momentum(particles)
    if dim is not None:
        for i, reset in enumerate(dim):
            if not reset:
                mom[i] = 0
    particles.vel -= mom / particles.mass.sum()


def kinetic_energy(particles: Particles) -> tuple[np.ndarray, float]:
    """Calculate kinetic energy of the particles.

    Returns:
        out[0] : The kinetic energy tensor.
        out[1] : The kinetic energy
    """
    mom = particles.vel * particles.mass
    if len(particles.mass) == 1:
        kin = 0.5 * np.outer(mom, particles.vel)
    else:
        kin = 0.5 * np.einsum("ij,ik->jk", mom, particles.vel)
    return kin, kin.trace()


def kinetic_temperature(
    particles: Particles,
    boltzmann: float,
    dof: list[float] | None = None,
    kin_tensor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the kinetic temperature of a collection of particles.

    Args:
        boltzmann: This is the Boltzmann factor/constant in correct units.
        dof: The degrees of freedom to subtract. Its shape should
            be equal to the number of dimensions.
        kin_tensor: The kinetic energy tensor. If the kinetic energy
            tensor is not given, it will be recalculated here.

    Returns:
        out[0]: The temperature averaged over all dimensions.
        out[1]: The temperature for each spatial dimension.
        out[2]: The kinetic energy tensor.

    """
    ndof = particles.npart * np.ones(particles.vel[0].shape)

    if kin_tensor is None:
        kin_tensor, _ = kinetic_energy(particles)
    if dof is not None:
        ndof = ndof - dof
    temperature = (2.0 * np.diagonal(kin_tensor) / ndof) / boltzmann
    return np.mean(temperature), temperature, kin_tensor


def pressure_tensor(
    particles: Particles, volume: float, kin_tensor: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Calculate the pressure tensor.

    The pressure tensor is obtained from the virial the kinetic
    energy tensor.

    Args:
        volume: The volume of the simulation box the particles are in.
        kin_tensor: The kinetic energy tensor. It will be calculate if not
            given here.

    Returns:
        out[0]: The symmetric pressure tensor.
        out[1] : The scalar pressure.

    """
    if kin_tensor is None:
        kin_tensor, _ = kinetic_energy(particles)
    pressure = (particles.virial + 2.0 * kin_tensor) / volume
    trace = pressure.trace() / float(particles.dim)
    return pressure, trace


def generate_maxwell_velocities(
    particles: Particles,
    rgen: Generator,
    temperature: float = 1.0,
    boltzmann: float = 1.0,
    dof: list[float] | None = None,
    momentum: bool = True,
):
    """Generate velocities from a Maxwell distribution.

    The velocities are drawn to match a given temperature and this
    function can be applied to a subset of the particles.

    The generation is done in three steps:

    1) We generate velocities from a standard normal distribution.

    2) We scale the velocity of particle `i` with
       ``1.0/sqrt(mass_i)`` and reset the momentum.

    3) We scale the velocities to the set temperature.

    Args:
        particles: The particles we will set the velocity for.
        rgen: The random number generator used for drawing velocities.
        temperature: The desired temperature.
        boltzmann: This is the Boltzmann factor/constant in
            correct units.
        dof: The degrees of freedom to subtract. Its shape should
            be equal to the number of dimensions.
        momentum: If False, we will not zero the linear momentum.
    """

    vel = np.sqrt(particles.imass) * rgen.normal(
        loc=0.0, scale=1.0, size=particles.vel.shape
    )
    particles.vel = vel
    if momentum:
        zero_momentum(particles)

    temp_gen, _, _ = kinetic_temperature(particles, boltzmann, dof=dof)
    scale_factor = np.sqrt(temperature / temp_gen)
    particles.vel *= scale_factor
