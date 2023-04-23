"""Define a list of particles.

Tha particle list collects the mass, positions, velocities,
forces, types, etc. for a collection of particles.
"""
from collections.abc import Iterator

import numpy as np


class Particles:
    npart: int  # The number of particles in the list
    dim: int  # The number of dimensions (1-3)
    pos: np.ndarray  # Positions for the particles
    vel: np.ndarray  # Velocities for the particles
    force: np.ndarray  # Force on the particles
    virial: np.ndarray  # Virial for the particles
    mass: np.ndarray  # The mass of the particles
    imass: np.ndarray  # The inverse mass of the particles
    name: list[str]  # Names of the particles
    ptype: np.ndarray  # Types of the particles

    def __init__(self, dim: int = 3):
        """Initialize an empty particle list.

        Parameters
        ----------
        dim : integer, optional
            The number of dimensions we have.
        """
        self.dim = dim
        self.empty()

    def empty(self):
        """Empty the particle list and remove all data."""
        self.npart = 0
        self.pos = np.zeros((1, self.dim))
        self.vel = np.zeros_like(self.pos)
        self.force = np.zeros_like(self.pos)
        self.mass = np.zeros((1, 1))
        self.imass = np.zeros_like(self.mass)
        self.name = []
        self.ptype = np.array([], dtype=int)
        self.virial = np.zeros((self.dim, self.dim))

    def add_particle(
        self,
        pos: np.ndarray,
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

        self.name.append(name)
        self.ptype = np.append(self.ptype, ptype)
        self.imass = 1.0 / self.mass
        self.npart += 1

    def __iter__(self) -> Iterator[dict]:
        """Yield the properties of the particles.

        Yields
        ------
        out : dict
            The information in `self.pos`, `self.vel`, ... etc.
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

    def pairs(self) -> Iterator[tuple[int, int, int, int]]:
        """Iterate over all pairs of particles.

        Yields
        ------
        out[0] : integer
            The index for the first particle in the pair.
        out[1] : integer
            The index for the second particle in the pair.
        out[2] : integer
            The particle type of the first particle.
        out[3] : integer
            The particle type of the second particle.

        """
        for i, itype in enumerate(self.ptype[:-1]):
            for j, jtype in enumerate(self.ptype[i + 1 :]):
                yield (i, i + 1 + j, itype, jtype)

    def linear_momentum(self) -> np.ndarray:
        """Return linear momentum of the particles"""
        mom = np.sum(self.vel * self.mass, axis=0)
        return mom

    def zero_momentum(self, dim: list[bool] | None = None):
        """Set the linear momentum for the particles to zero.

        Parameters
        ----------
        dim : list or None, optional
            If None, the momentum will be reset for ALL dimensions.
            If a list is given, the momentum will only be reset where
            it is True.
        """
        mom = self.linear_momentum()
        if dim is not None:
            for i, reset in enumerate(dim):
                if not reset:
                    mom[i] = 0
        self.vel -= mom / self.mass.sum()

    def kinetic_energy(self) -> tuple[np.ndarray, float]:
        """Calculate kinetic energy of the particles.

        Returns
        -------
        out[0] : numpy.array
            The kinetic energy tensor. Dimensionality is equal to (dim, dim)
            where dim is the number of dimensions used in the velocities.
        out[1] : float
            The kinetic energy
        """
        mom = self.vel * self.mass
        if len(self.mass) == 1:
            kin = 0.5 * np.outer(mom, self.vel)
        else:
            kin = 0.5 * np.einsum("ij,ik->jk", mom, self.vel)
        return kin, kin.trace()

    def kinetic_temperature(
        self,
        boltzmann: float,
        dof: list[float] | None = None,
        kin_tensor: np.ndarray | None = None,
    ) -> tuple[np.floating, np.ndarray, np.ndarray]:
        """Calculate the kinetic temperature of a collection of particles.

        Parameters
        ----------
        boltzmann : float
            This is the Boltzmann factor/constant in correct units.
        dof : list of floats, optional
            The degrees of freedom to subtract. Its shape should
            be equal to the number of dimensions.
        kin_tensor : numpy.array optional
            The kinetic energy tensor. If the kinetic energy tensor is not
            given, it will be recalculated here.

        Returns
        -------
        out[0] : float
            The temperature averaged over all dimensions.
        out[1] : numpy.array
            The temperature for each spatial dimension.
        out[2] : numpy.array
            The kinetic energy tensor.

        """
        ndof = self.npart * np.ones(self.vel[0].shape)

        if kin_tensor is None:
            kin_tensor, _ = self.kinetic_energy()
        if dof is not None:
            ndof = ndof - dof
        temperature = (2.0 * kin_tensor.diagonal() / ndof) / boltzmann
        return np.mean(temperature), temperature, kin_tensor

    def pressure_tensor(
        self, volume: float, kin_tensor: np.ndarray | None = None
    ) -> tuple[np.ndarray, float]:
        """Calculate the pressure tensor.

        The pressure tensor is obtained from the virial the kinetic
        energy tensor.

        Parameters
        ----------
        volume : float
            The volume of the simulation box the particles are in.
        kin_tensor : numpy.array, optional
            The kinetic energy tensor. It will be calculate if not
            given here.

        Returns
        -------
        out[0] : numpy.array
            The symmetric pressure tensor, dimensions (`dim`, `dim`), where
            `dim` = the number of dimensions considered in the simulation.
        out[1] : float
            The scalar pressure.

        """
        if kin_tensor is None:
            kin_tensor, _ = self.kinetic_energy()
        pressure = (self.virial + 2.0 * kin_tensor) / volume
        trace = pressure.trace() / float(self.dim)
        return pressure, trace
