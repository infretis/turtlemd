import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from numpy.random import default_rng

from turtlemd.integrators import VelocityVerlet
from turtlemd.potentials import DoubleWellPair, LennardJonesCut
from turtlemd.simulation import MDSimulation
from turtlemd.system import Box, Particles, System
from turtlemd.system.particles import generate_maxwell_velocities
from turtlemd.tools import generate_lattice

sns.set_context("talk", font_scale=0.8)


def create_system() -> System:
    """Create a system with 9 particles."""
    pos, size = generate_lattice(
        lattice="sq", repeat=[3, 3], lattice_constant=1.0
    )
    box = Box(low=size[:, 0], high=size[:, 1] * 2.0)
    # center positions in the box:
    pos -= np.mean(pos, axis=0) - 0.5 * box.length
    particles = Particles(dim=box.dim)
    for i, xyz in enumerate(pos):
        if i in (5, 7):
            particles.add_particle(pos=xyz, mass=1.0, ptype=1, name="B")
        else:
            particles.add_particle(pos=xyz, mass=1.0, ptype=0, name="A")

    generate_maxwell_velocities(
        particles,
        rgen=default_rng(),
        temperature=2.0,
    )

    lennard = LennardJonesCut(dim=3, shift=True, mixing="geometric")
    lj_parameters = {
        0: {"sigma": 1, "epsilon": 1, "rcut": 2 ** (1.0 / 6.0)},
        1: {"sigma": 1, "epsilon": 1, "rcut": 2 ** (1.0 / 6.0)},
    }
    lennard.set_parameters(lj_parameters)

    dwell = DoubleWellPair(types=(1, 1), dim=box.dim)
    well_parameters = {
        "rzero": 1.0 * (2.0 ** (1.0 / 6.0)),
        "height": 6.0,
        "width": 0.25,
    }
    dwell.set_parameters(well_parameters)

    system = System(box=box, particles=particles, potentials=[lennard, dwell])
    return system


def order_parameters(system: System) -> tuple[float, float, float]:
    """Calculate the bond length and the potential energy."""
    pot = system.potentials[1]
    particles = system.particles
    delta = system.box.pbc_dist(particles.pos[5] - particles.pos[7])
    bond_length = np.sqrt(np.dot(delta, delta))
    return delta, bond_length, pot.potential(system)


def draw_potential_energy(
    system: System, ax: plt.Axes
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the potential energy."""
    pot = system.potentials[1]
    maxpos = pot.params["rzero"] + pot.params["width"]
    rpos = np.linspace(maxpos - 0.37, maxpos + 0.37, 100)
    vpot = pot._potential_function(rpos)
    ax.plot(rpos, vpot)
    return rpos, np.array(vpot)


def spring_bond(
    delta: np.ndarray, dr: float, part1: np.ndarray, part2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Create positions for a zig-zag line.

    This is a function that will create positions which can be used to
    create a zig-zag bond. It is used here to illustrate a spring bond
    between two atoms

    Parameters
    ----------
    delta : numpy.array
        Distance vector between `part1` and `part2`, subjected to periodic
        boundary conditions.
    dr : float
        Length of `delta` vector.
    part1 : numpy.array
        Particle 1 position. Bond is drawn from `part1`.
    part2 : numpy.array
        Particle 2 position. Bond is drawn to `part2`.

    Returns
    -------
    out[0] : numpy.array
        X-coordinates for the line.
    out[1] : numpy.array
        Y-coordinates for the line.

    """
    delta_u = delta / dr
    xpos = [part1[0]]
    ypos = [part1[1]]
    for pidx, add in enumerate(np.linspace(0.0, dr - 1, 11)):
        point = part1 + (add + 0.5) * delta_u
        if pidx in [2, 4, 6, 8]:
            if delta_u[0] == 0:
                dperp = np.array([0.0, 0.0])
            else:
                dperp_v = np.array([-delta_u[1] / delta_u[0], 1.0])
                dperp = dperp_v / np.sqrt(np.dot(dperp_v, dperp_v))
            sig = 1 if delta_u[0] > 0.0 else -1.0
            if pidx in [2, 6]:
                dvec = sig * 0.2 * dperp
            else:
                dvec = -sig * 0.2 * dperp
            point = point + dvec
        xpos.append(point[0])
        ypos.append(point[1])
    xpos.append(part2[0])
    ypos.append(part2[1])
    xpos = np.array(xpos)
    ypos = np.array(ypos)
    return xpos, ypos


def update(
    frame: int,
    system: System,
    simulation: MDSimulation,
    patches: list,
    circles: list,
):
    """Update the animation."""
    updated_patches = []
    simulation.step()
    # Update the order parameter:
    delta, delr, vpot = order_parameters(system)
    patches[0].set_offsets([delr, vpot])
    updated_patches.append(patches[0])
    # Update the energies:
    thermo = system.thermo()
    thermo["total"] = thermo["vpot"] + thermo["ekin"]
    for i, item in enumerate(["vpot", "ekin", "total"]):
        step, value = patches[i + 1].get_data()
        step = np.append(step, simulation.cycle["stepno"])
        valuei = thermo[item]
        value = np.append(value, valuei)
        patches[i + 1].set_data(step, value)
        updated_patches.append(patches[i + 1])

    # Update the location of the particles:
    box = system.box
    particles = system.particles
    position = box.pbc_wrap(particles.pos)

    for ci, posi in zip(circles, position):
        ci.center = posi
        ci.set_visible(True)
        updated_patches.append(ci)

    # Bond length:
    circles[5].center = position[7] + delta
    xpos, ypos = spring_bond(delta, delr, position[7], position[7] + delta)
    patches[4].set_data(xpos, ypos)
    updated_patches.append(patches[4])

    return updated_patches


def main():
    """Run a Lennard-Jones simulation."""
    system = create_system()
    simulation = MDSimulation(
        system=system, integrator=VelocityVerlet(0.005), steps=1000
    )

    fig, axes = plt.subplot_mosaic(
        [["left", "upper right"], ["left", "lower right"]],
        figsize=(12, 5),
        layout="constrained",
    )

    axes["left"].set_aspect("equal")
    axes["left"].set_xticks([])
    axes["left"].set_yticks([])

    draw_potential_energy(system, axes["upper right"])

    order_scatter = axes["upper right"].scatter([], [], s=200)
    (line_pot,) = axes["lower right"].plot([], [], label="Potential")
    (line_kin,) = axes["lower right"].plot([], [], label="Kinetic")
    (line_etot,) = axes["lower right"].plot([], [], label="Total")
    axes["lower right"].set_xlim(0, simulation.steps)
    axes["lower right"].set_ylim(-0.2, 5.0)
    axes["lower right"].set_ylabel("Energy\n(arbitrary unit)")
    axes["lower right"].set_xlabel("Step")
    axes["lower right"].legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        columnspacing=1,
        labelspacing=1,
        fontsize="small",
    )
    axes["upper right"].set_ylabel("Potential energy\n(arbitrary unit)")
    axes["upper right"].set_xlabel("Bond length (arbitrary unit)")
    sns.despine(ax=axes["upper right"])
    sns.despine(ax=axes["lower right"])
    patches = [order_scatter, line_pot, line_kin, line_etot]

    circles = []
    for particle in system.particles:
        color = "#af8dc3" if particle["type"] == 0 else "#7fbf7b"
        circle = Circle(particle["pos"], radius=0.5, color=color)
        circle.set_visible(False)
        axes["left"].add_patch(circle)
        circles.append(circle)

    (bond_length,) = axes["left"].plot([], [])
    patches.append(bond_length)

    box_patch = Rectangle(
        system.box.low, system.box.length[0], system.box.length[1], fill=None
    )
    axes["left"].add_patch(box_patch)
    axes["left"].set_xlim(system.box.low[0], system.box.length[0])
    axes["left"].set_ylim(system.box.low[1], system.box.length[1])

    def init():
        return patches + circles

    _ = FuncAnimation(
        fig,
        update,
        frames=simulation.steps,
        fargs=[system, simulation, patches, circles],
        repeat=False,
        interval=1,
        blit=True,
        init_func=init,
    )
    plt.show()


if __name__ == "__main__":
    main()
