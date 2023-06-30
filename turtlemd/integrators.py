"""Definition of time integrators."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator, default_rng

from turtlemd.system.system import System

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class MDIntegrator(ABC):
    """Base class for MD integrators."""

    timestep: float  # The timestep
    description: str  # Short description of the integrator
    dynamics: str  # String representation of the type of dynamics

    def __init__(self, timestep: float, description: str, dynamics: str):
        """Set up the integrator with."""
        self.timestep = timestep
        self.description = description
        self.dynamics = dynamics

    @abstractmethod
    def integration_step(self, system: System):
        """Integrate the system a single time step."""

    def __call__(self, system: System):
        return self.integration_step(system)


class Verlet(MDIntegrator):
    """The Verlet integrator."""

    timestep_squared: float  # Squared time step.
    half_idt: float  # Half of the inverse of the time step.
    previous_pos: np.ndarray | None  # Positions at the previous step.

    def __init__(self, timestep: float):
        super().__init__(
            timestep=timestep,
            description="Verlet integrator",
            dynamics="NVE",
        )
        self.timestep_squared = self.timestep**2
        self.half_idt = 0.5 / self.timestep
        self.previous_pos = None

    def integration_step(self, system: System):
        particles = system.particles
        if self.previous_pos is None:
            self.previous_pos = particles.pos - particles.vel * self.timestep
        acc = particles.force * particles.imass
        pos = (
            2.0 * particles.pos
            - self.previous_pos
            + acc * self.timestep_squared
        )
        # Update velocity:
        particles.vel = (pos - self.previous_pos) * self.half_idt
        # Update positions:
        self.previous_pos, particles.pos = particles.pos, pos
        # Update potential energy and force:
        system.potential_and_force()


class VelocityVerlet(MDIntegrator):
    """The Velocity Verlet integrator."""

    half_timestep: float  # Half of the time step

    def __init__(self, timestep: float):
        super().__init__(
            timestep=timestep,
            description="Velocity Verlet integrator",
            dynamics="NVE",
        )
        self.half_timestep = self.timestep * 0.5

    def integration_step(self, system: System):
        particles = system.particles
        imass = particles.imass
        # Update velocity
        particles.vel += self.half_timestep * particles.force * imass
        # Update position
        particles.pos += self.timestep * particles.vel
        # Update potential and force:
        system.potential_and_force()
        # Update velocity again:
        particles.vel += self.half_timestep * particles.force * imass


class LangevinOverdamped(MDIntegrator):
    """Overdamped version of the Langevin integrator.

    For the `overdamped`_ Langevin integrator, the equations of motion are
    integrated according to:

    ``r(t + dt) = r(t) + dt * f(t)/m*gamma + dr``,

    and

    ``v(t + dt) = dr``

    where ``dr`` are obtained from a normal distribution.

    .. _overdamped:
        https://en.wikipedia.org/wiki/Brownian_dynamics
    """

    gamma: float  # The gamma parameter
    sigma: np.ndarray | float  # Standard deviation for random numbers
    bddt: np.ndarray | float  # The factor dt/(m*gamma)
    rgen: Generator  # The random number generator to use here
    beta: float  # The kB*T factor
    _initiate: bool  # If True, we still need to set some parameters

    def __init__(
        self, timestep: float, gamma: float, rgen: Generator, beta: float
    ):
        super().__init__(
            timestep=timestep,
            description="Langevin overdamped integrator",
            dynamics="stochastic",
        )
        self.gamma = gamma
        self.sigma = 0.0
        self.rgen = rgen
        self.bddt = 0.0
        self.beta = beta
        self._initiate = True

    def initiate_parameters(self, system: System):
        """Initiate the parameters for the integrator.

        The initiation needs the masses of the particles, so we need
        the system to initiate all parameters.
        """
        if self._initiate:
            imass = system.particles.imass
            self.sigma = np.sqrt(
                2.0 * self.timestep * imass / (self.beta * self.gamma)
            )
            self.bddt = self.timestep * imass / self.gamma
            self._initiate = False

    def integration_step(self, system: System):
        """Do a single integration step."""
        self.initiate_parameters(system)
        system.force()
        particles = system.particles
        rands = self.rgen.normal(
            loc=0.0,
            scale=self.sigma,  # sigma is not None here.
            size=particles.vel.shape,
        )
        particles.pos += self.bddt * particles.force + rands
        particles.vel = rands
        system.potential()


@dataclass
class LangevinParameter:
    """Store parameters for the Langevin integrator."""

    c0: float = 0.0
    a1: float = 0.0
    a2: np.ndarray = field(default_factory=lambda: np.zeros(1))
    b1: np.ndarray = field(default_factory=lambda: np.zeros(1))
    b2: np.ndarray = field(default_factory=lambda: np.zeros(1))
    mean: list[np.ndarray] = field(default_factory=list)
    cov: list[np.ndarray] = field(default_factory=list)
    cho: list[np.ndarray] = field(default_factory=list)


class LangevinIntertia(MDIntegrator):
    """The `Langevin`_ integrator.

    The equations of motion are integrated according to,

    ``r(t + dt) = r(t) + c1 * dt * v(t) + c2*dt*dt*a(t) + dr``

    ``v(r + dt) = c0 * v(t) + (c1-c2)*dt*a(t) + c2*dt*a(t+dt) + dv``.

    where c0, c1, and c2 are parameters derived from gamma.

    .. _Langevin:
        https://en.wikipedia.org/wiki/Langevin_dynamics
    """

    gamma: float  # The gamma parameter
    rgen: Generator  # The random number generator to use here
    beta: float  # The kB*T factor.
    param: LangevinParameter  # Constants/parameters for the dynamics.
    _initiate: bool  # Determines if we need to initiate the parameters.

    def __init__(
        self,
        timestep: float,
        gamma: float,
        beta: float,
        rgen: Generator | None = None,
        seed: int = 0,
    ):
        super().__init__(
            timestep=timestep,
            description="Langevin overdamped integrator",
            dynamics="stochastic",
        )
        self.gamma = gamma
        if rgen is None:
            self.rgen = default_rng(seed=seed)
        else:
            self.rgen = rgen
        self.beta = beta
        self._initiate = True
        self.param = LangevinParameter()

    def initiate_parameters(self, system: System):
        """Initiate the parameters for the integrator.

        The initiation needs the masses of the particles, so we need
        the system to initiate all parameters.
        """
        if not self._initiate:
            return
        gamma_dt = self.gamma * self.timestep
        exp_gdt = np.exp(-gamma_dt)

        c_0 = exp_gdt
        c_1 = (1.0 - c_0) / gamma_dt
        c_2 = (1.0 - c_1) / gamma_dt

        imasses = system.particles.imass
        self.param.c0 = c_0
        self.param.a1 = c_1 * self.timestep
        self.param.a2 = c_2 * self.timestep**2 * imasses
        self.param.b1 = (c_1 - c_2) * self.timestep * imasses
        self.param.b2 = c_2 * self.timestep * imasses

        self.param.mean = []
        self.param.cho = []
        self.param.cov = []

        for imassi in imasses:
            sig_ri2 = (
                self.timestep * imassi[0] / (self.beta * self.gamma)
            ) * (2.0 - (3.0 - 4.0 * exp_gdt + exp_gdt**2) / gamma_dt)
            sig_vi2 = (1.0 - exp_gdt**2) * imassi[0] / self.beta
            cov_rvi = (imassi[0] / (self.beta * self.gamma)) * (
                1.0 - exp_gdt
            ) ** 2

            cov_matrix = np.array([[sig_ri2, cov_rvi], [cov_rvi, sig_vi2]])

            self.param.mean.append(np.zeros(2))
            self.param.cho.append(np.linalg.cholesky(cov_matrix))
            self.param.cov.append(cov_matrix)
        self._initiate = False

    def integration_step(self, system: System):
        """Do one Langevin integration step."""
        self.initiate_parameters(system)
        particles = system.particles
        pos_rand, vel_rand = self.draw_random_numbers(system)
        particles.pos += (
            self.param.a1 * particles.vel
            + self.param.a2 * particles.force
            + pos_rand
        )
        vel2 = (
            self.param.c0 * particles.vel
            + self.param.b1 * particles.force
            + vel_rand
        )
        system.force()
        particles.vel = vel2 + self.param.b2 * particles.force
        system.potential()

    def draw_random_numbers(
        self, system: System
    ) -> tuple[np.ndarray, np.ndarray]:
        """This method draws random numbers for the integration step."""
        particles = system.particles
        dim = particles.dim
        pos_rand = np.zeros_like(particles.pos)
        vel_rand = np.zeros_like(particles.vel)
        mean, cho = self.param.mean, self.param.cho
        for i, (meani, choi) in enumerate(zip(mean, cho)):
            randxv = multivariate_normal(self.rgen, meani, choi, dim)
            pos_rand[i] = randxv[:, 0]
            vel_rand[i] = randxv[:, 1]
        return pos_rand, vel_rand


def multivariate_normal(
    rgen: Generator, mean: np.ndarray, cho: np.ndarray, dim: int
) -> np.ndarray:
    """Draw numbers from a multivariate normal distribution.

    Here, we want to avoid redoing the Cholesky factorization we
    did during the initialization of the parameters. So this method
    should be equal to the `multivariate_normal` method of the
    random generators in numpy (with `method='cholesky'`) with the
    exception here that we assume we already have done the Cholesky
    factorization and know the input matrix `cho`.
    """
    norm = rgen.normal(loc=0.0, scale=1.0, size=2 * dim)
    norm = norm.reshape(dim, 2)
    meanm = np.array([mean, ] * dim)  # fmt: skip
    return meanm + norm @ cho.T
