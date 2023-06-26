"""Test the MD simulation class."""
import logging
import os
import pathlib

import numpy as np
import pytest

from turtlemd.integrators import VelocityVerlet
from turtlemd.potentials.lennardjones import LennardJonesCut
from turtlemd.simulation.mdsimulation import MDSimulation
from turtlemd.simulation.stopping import MaxSteps, SoftExit
from turtlemd.system import Box, Particles, System
from turtlemd.tools.tools import generate_lattice

HERE = pathlib.Path(__file__).resolve().parent


CORRECT_ENERGIES = {
    "temperature": np.array(
        [
            2.0,
            1.999160440266409,
            1.9966296489360378,
            1.9923736990422694,
            1.9863390767180622,
            1.9784523258956044,
            1.968619736829698,
            1.9567271576071847,
            1.9426400497258036,
            1.9262095427759933,
            1.9072700985978948,
        ]
    ),
    "vpot": np.array(
        [
            -6.779703167843626,
            -6.778457352965096,
            -6.774700669524494,
            -6.768382753429658,
            -6.759424141357691,
            -6.747715738534962,
            -6.73311834845348,
            -6.715462381126993,
            -6.694547918387425,
            -6.670151227900626,
            -6.64203109238166,
        ]
    ),
    "ekin": np.array(
        [
            2.9722222222222223,
            2.9709745431736914,
            2.967213506057723,
            2.960888691632261,
            2.951920572344897,
            2.9401999843170783,
            2.925587664455245,
            2.9079139703328996,
            2.8869789627869578,
            2.8625614038476566,
            2.834415285416316,
        ]
    ),
    "pressure": np.array(
        [
            -4.050724889322912,
            -4.043813673394954,
            -4.023015683870377,
            -3.988171894211394,
            -3.9390375443837495,
            -3.875281440146006,
            -3.7964859272745066,
            -3.7021479523149554,
            -3.5916818266053743,
            -3.465513329534334,
            -3.322391707367211,
        ]
    ),
    "energy": np.array(
        [
            -3.8074809456214034,
            -3.807482809791404,
            -3.8074871634667713,
            -3.8074940617973967,
            -3.807503569012794,
            -3.8075157542178837,
            -3.8075306839982344,
            -3.807548410794093,
            -3.8075689556004675,
            -3.80758982405297,
            -3.807615806965344,
        ]
    ),
}


def create_test_system(repeat=None, vel=True):
    """Create a LJ system we can use for testing."""
    if repeat is None:
        repeat = [3, 3, 3]
    # Position of particles:
    xyz, size = generate_lattice("fcc", repeat, density=0.9)
    box = Box(low=size[:, 0], high=size[:, 1])
    particles = Particles(dim=box.dim)
    for pos in xyz:
        particles.add_particle(pos=pos, mass=1.0, name="Ar", ptype=0)
    if vel:
        # Velocity of particles:
        particles.vel = np.loadtxt(HERE / "velocities.txt.gz")
    # Potential:
    ljpot = LennardJonesCut(dim=3, shift=True, mixing="geometric")
    parameters = {
        0: {"sigma": 1.0, "epsilon": 1.0, "rcut": 2.5},
    }
    ljpot.set_parameters(parameters)
    # Create system:
    system = System(box=box, particles=particles, potentials=[ljpot])
    return system


def test_md_simulation():
    """Test that we can run a MD simulation."""
    system = create_test_system()
    integrator = VelocityVerlet(timestep=0.002)
    simulation = MDSimulation(system, integrator, 10)
    # Accumulate energies:
    for i, systemi in enumerate(simulation.run()):
        therm = systemi.thermo(boltzmann=1)
        assert (
            pytest.approx(therm["temperature"])
            == CORRECT_ENERGIES["temperature"][i]
        )
        assert (
            pytest.approx(therm["pressure"])
            == CORRECT_ENERGIES["pressure"][i]
        )
        assert pytest.approx(therm["ekin"]) == CORRECT_ENERGIES["ekin"][i]
        assert pytest.approx(therm["vpot"]) == CORRECT_ENERGIES["vpot"][i]


def test_stop_conditions_create():
    """Test that we can give stop conditions."""
    system = create_test_system(repeat=[1, 1, 1], vel=False)
    integrator = VelocityVerlet(timestep=0.002)
    simulation = MDSimulation(
        system, integrator, 2, stop_conditions=[MaxSteps()]
    )
    # Check that the MaxSteps is present:
    assert any(isinstance(i, MaxSteps) for i in simulation.stop_conditions)
    # Check that we got a soft exit also (by default):
    assert any(isinstance(i, SoftExit) for i in simulation.stop_conditions)


def test_max_steps_condition():
    """Test that the max steps works as planned."""
    system = create_test_system(repeat=[1, 1, 1], vel=False)
    integrator = VelocityVerlet(timestep=0.002)
    simulation = MDSimulation(
        system, integrator, 2, stop_conditions=[MaxSteps()]
    )
    # The zeroth step:
    prev = system.particles.pos.copy()
    simulation.step()
    assert pytest.approx(prev) == system.particles.pos
    assert simulation.cycle["stepno"] == 0
    # The first real step:
    prev = system.particles.pos.copy()
    simulation.step()
    assert pytest.approx(prev) != system.particles.pos
    assert simulation.cycle["stepno"] == 1
    # The second real step:
    prev = system.particles.pos.copy()
    simulation.step()
    assert pytest.approx(prev) != system.particles.pos
    assert simulation.cycle["stepno"] == 2
    # The simulation should be done now, and will not run more steps:
    prev = system.particles.pos.copy()
    simulation.step()  # These steps will not be performed.
    simulation.step()
    simulation.step()
    assert pytest.approx(prev) == system.particles.pos
    assert simulation.cycle["stepno"] == 2


def test_soft_exit_condition(caplog, tmp_path):
    """Test the soft exit stop condition."""
    system = create_test_system(repeat=[1, 1, 1], vel=False)
    integrator = VelocityVerlet(timestep=0.002)
    simulation = MDSimulation(
        system, integrator, 10, stop_conditions=[MaxSteps()]
    )

    run_in = tmp_path / "md"
    run_in.mkdir()
    os.chdir(run_in)
    # Step zero:
    simulation.step()
    for _ in range(5):
        simulation.step()
    assert simulation.cycle["stepno"] == 5
    # Create the exit file:
    exit_file = run_in / "EXIT"
    exit_file.touch()
    with caplog.at_level(logging.INFO):
        simulation.step()
        assert "Found exit file" in caplog.text
    assert simulation.cycle["stepno"] == 5
    simulation.step()
    assert simulation.cycle["stepno"] == 5
    exit_file.unlink()
    simulation.step()
    assert simulation.cycle["stepno"] == 6
