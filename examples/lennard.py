from collections import defaultdict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.random import default_rng
from tqdm import tqdm

from turtlemd.integrators import VelocityVerlet
from turtlemd.potentials import LennardJonesCut
from turtlemd.simulation import MDSimulation
from turtlemd.system import Box, Particles, System
from turtlemd.system.particles import generate_maxwell_velocities
from turtlemd.tools import generate_lattice

sns.set_context("talk")


def create_system():
    pos, size = generate_lattice(lattice="fcc", repeat=[4, 4, 4], density=0.8)
    box = Box(low=size[:, 0], high=size[:, 1])
    particles = Particles()
    for xyz in pos:
        particles.add_particle(pos=xyz, mass=1.0, ptype=0, name="Ar")

    generate_maxwell_velocities(
        particles, rgen=default_rng(), temperature=0.8
    )

    lennard = LennardJonesCut(dim=3, shift=True, mixing="geometric")
    parameters = {
        0: {"sigma": 1, "epsilon": 1, "rcut": 2.5},
    }
    lennard.set_parameters(parameters)

    system = System(box=box, particles=particles, potentials=[lennard])
    return system


def main():
    """Run a Lennard-Jones simulation."""
    system = create_system()
    simulation = MDSimulation(
        system=system, integrator=VelocityVerlet(0.005), steps=200
    )
    thermo = defaultdict(list)
    for step in tqdm(simulation.run(), total=simulation.steps):
        thermoi = step.thermo()
        for key, val in thermoi.items():
            thermo[key].append(val)
        thermo["step"].append(simulation.cycle["stepno"])

    for key, val in thermo.items():
        thermo[key] = np.array(val)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(thermo["step"], thermo["ekin"], label="Kinetic")
    ax.plot(thermo["step"], thermo["vpot"], label="Potential")
    ax.plot(
        thermo["step"], thermo["ekin"] + thermo["vpot"], label="Total energy"
    )
    ax.set(xlabel="Step", ylabel="Energy")
    sns.despine(fig=fig)
    plt.show()


if __name__ == "__main__":
    main()
