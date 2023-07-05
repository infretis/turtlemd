from collections import defaultdict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.random import default_rng
from tqdm import tqdm

from turtlemd.integrators import VelocityVerlet
from turtlemd.potentials.jax_bondedinteractions import BondedInteractions
from turtlemd.simulation import MDSimulation
from turtlemd.system import Box, Particles, System
from turtlemd.system.particles import generate_maxwell_velocities

sns.set_context("talk")


# units are ps, nm, kj/mol, g/mol

def create_system():
    mass = 1.008
    name = 'H'
    positions = [
        [0.303, 0.400, 0.372],
        [0.370, 0.424, 0.443],
        [0.432, 0.353, 0.408],
        [0.496, 0.424, 0.378],
    ]

    particles = Particles()
    for pos in positions:
        particles.add_particle(pos, mass=mass, name=name)

    generate_maxwell_velocities(
        particles,
        rgen=default_rng(),
        temperature=300.0,
        boltzmann=8.3145e-3,
        momentum=True,
    )

    box = Box(
        low=[0, 0, 0], high=[0.8, 0.8, 0.8], periodic=[True, True, True]
    )

    # force field
    potentials = [
        BondedInteractions(
            bonds=[
                # k         b0   i  j
                [200000.0, 0.1, 0, 1],
                [200000.0, 0.1, 1, 2],
                [200000.0, 0.1, 2, 3],
            ],
            angles=[
                # k      ang0           i  j  k
                [400.0, 1.57079632679, 0, 1, 2],
                [400.0, 1.57079632679, 1, 2, 3],
            ],
            dihedrals=[
                # k    ang0           n    i  j  k  l
                [8.0, 0.78539816339, 2.0, 0, 1, 2, 3]
            ],
        )
    ]

    system = System(box=box, particles=particles, potentials=potentials)
    return system


def main():
    """Run a simulation containing a four particle toy system."""
    system = create_system()
    simulation = MDSimulation(
        system=system, integrator=VelocityVerlet(0.0005), steps=2000
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
