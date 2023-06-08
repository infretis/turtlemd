"""Definition of time integrators."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.random import RandomState

from turtlemd.system import System

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
    sigma: np.ndarray | None  # Standard deviation for random numbers
    bddt: np.ndarray | None  # The factor dt/(m*gamma)
    rgen: RandomState  # The random number generator to use here
    beta: float  # The kB*T factor.

    def __init__(
        self, timestep: float, gamma: float, rgen: RandomState, beta: float
    ):
        super().__init__(
            timestep=timestep,
            description="Langevin overdamped integrator",
            dynamics="stochastic",
        )
        self.gamma = gamma
        self.sigma = None
        self.rgen = rgen
        self.bddt = None
        self.beta = beta

    def initiate_parameters(self, system: System):
        """Initiate the parameters for the integrator.

        The initiation needs the masses of the particles, so we need
        the system to initiate all parameters.
        """
        imass = system.particles.imass
        self.sigma = np.sqrt(
            2.0 * self.timestep * imass / (self.beta * self.gamma)
        )
        self.bddt = self.timestep * imass / self.gamma

    def integration_step(self, system: System):
        """Do a single integration step."""
        if self.sigma is None or self.bddt is None:
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


# @dataclass
# class LangevinParameterHigh:
#    scale: np.ndarray = np.zeros(1)
#    bddt: np.ndarray = np.zeros(1)
#
#
# @dataclass
# class LangevinParameterLow:
#    c0: float = 0.0
#    a1: float = 0.0
#    a2: np.ndarray = np.zeros(1)
#    b1: np.ndarray = np.zeros(1)
#    b2: np.ndarray = np.zeros(1)
#    mean: list[np.ndarray] = []
#    cov: list[np.ndarray] = []
#    cho: list[np.ndarray] = []
#
#
# class Langevin(MDIntegrator):
#    """The Langevin integrator."""
#
#    gamma: float  # The γ-parameter for the Langevin integrator
#    high_friction: bool  # If we should use the high friction variant
#    high: LangevinParameterHigh  # Parameters for high friciton limit
#    low: LangevinParameterLow  # Parameters for the low frinction variant
#    init_params: bool  # Should we initiate parameters?
#    rgen: RandomState  # Random generator to use here.
#    temperature: float  # The temperature of the system
#
#    def __init__(
#        self,
#        timestep: float,
#        gamma: float,
#        rgen,
#        temperature: float,
#        high_friction: bool = False,
#    ):
#        super().__init__(
#            timestep=timestep,
#            description="Langevin integrator",
#            dynamics="stochastic",
#        )
#        self.gamma = gamma
#        self.high_friction = high_friction
#        self.temperature = temperature
#        self.high = LangevinParameterHigh()
#        self.low = LangevinParameterLow()
#        self.init_params = True
#        self.rgen = rgen
#
#    def initiate_parameters(self, system: System):
#        """Initite the parameters for the integrator."""
#        beta = system.temperature["beta"]
#        imass = system.particles.imass
#        if self.high_friction:
#            self.high.scale = np.sqrt(
#                2.0 * self.timestep * imass / (beta * self.gamma)
#            )
#            self.high.bddt = self.timestep * imass / self.gamma
#        else:
#            gamma_dt = self.gamma * self.timestep
#            exp_gdt = np.exp(-gamma_dt)
#
#            c_0 = exp_gdt
#            c_1 = (1.0 - c_0) / gamma_dt
#            c_2 = (1.0 - c_1) / gamma_dt
#
#            self.low.c0 = c_0
#            self.low.a1 = c_1 * self.timestep
#            self.low.a2 = c_2 * self.timestep**2 * imass
#            self.low.b1 = (c_1 - c_2) * self.timestep * imass
#            self.low.b2 = c_2 * self.timestep * imass
#
#            self.low.mean = []
#            self.low.cho = []
#            self.low.cov = []
#
#            for imassi in imass:
#                sig_ri2 = (
#                    self.timestep * imassi[0] / (beta * self.gamma)
#                ) * (2.0 - (3.0 - 4.0 * exp_gdt + exp_gdt**2) / gamma_dt)
#                sig_vi2 = (1.0 - exp_gdt**2) * imassi[0] / beta
#                cov_rvi = (imassi[0] / (beta * self.gamma)) * (
#                    1.0 - exp_gdt
#                ) ** 2
#                cov_matrix = np.array(
#                    [[sig_ri2, cov_rvi], [cov_rvi, sig_vi2]]
#                )
#                self.low.mean.append(np.zeros(2))
#                self.low.cho.append(np.linalg.cholesky(cov_matrix))
#                self.low.cov.append(cov_matrix)
#
#    def overdamped_step(self, system: System):
#        """One time step of overdamped Langevin integration."""
#        system.force()
#        particles = system.particles
#        rands = self.rgen.normal(
#            loc=0.0,
#            scale=self.high.scale,
#            size=particles.vel.shape,
#        )
#        particles.pos += self.high.bddt * particles.force + rands
#        particles.vel = rands
#        system.potential()
#
#    def inertia_step(self, system: System):
#        """One time step of normal Langevin integration."""
#        particles = system.particles
#        ndim = system.get_dim()
#        pos_rand = np.zeros(particles.pos.shape)
#        vel_rand = np.zeros(particles.vel.shape)
#        if self.gamma > 0.0:
#            for i, (meani, covi, choi) in enumerate(
#                zip(self.low.mean, self.low.cov, self.low.cho)
#            ):
#                randxv = self.rgen.multivariate_normal(
#                    meani, covi, cho=choi, size=ndim
#                )
#                pos_rand[i] = randxv[:, 0]
#                vel_rand[i] = randxv[:, 1]
#        particles.pos += (
#            self.low.a1 * particles.vel
#            + self.low.a2 * particles.force
#            + pos_rand
#        )
#
#        vel2 = (
#            self.low.c0 * particles.vel
#            + self.low.b1 * particles.force
#            + vel_rand
#        )
#
#        system.force()  # Update forces.
#
#        particles.vel = vel2 + self.low.b2 * particles.force
#
#        system.potential()
#
#    def integration_step(self, system: System):
#        if self.init_params:
#            self.initiate_parameters(system)
#            self.init_params = False
