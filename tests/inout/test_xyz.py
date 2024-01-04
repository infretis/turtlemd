"""Test that we can read xyz-files."""
import logging
import pathlib

import numpy as np
import pytest

from turtlemd.inout.xyz import (
    configuration_from_xyz_file,
    pad_to_nd,
    read_xyz_file,
    system_to_xyz,
)
from turtlemd.system import Box, Particles, System
from turtlemd.tools import generate_lattice

HERE = pathlib.Path(__file__).resolve().parent
XYZDIR = HERE / "xyz"

CORRECT_XYZ = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
)


def test_read_single_config():
    """Test that we can read a single config from a xyz-file."""
    xyz_file = XYZDIR / "config.xyz"
    snapshot = next(read_xyz_file(xyz_file))
    assert snapshot.natoms == 5
    assert snapshot.atoms == ["Ba", "Hf", "O", "O", "O"]
    assert (
        snapshot.comment
        == "Example from: https://en.wikipedia.org/wiki/XYZ_file_format"
    )
    assert pytest.approx(snapshot.xyz) == CORRECT_XYZ


def test_read_trajectory():
    """Test that we can read a xyz-trajectory."""
    xyz_file = XYZDIR / "traj.xyz"
    for i, snapshot in enumerate(read_xyz_file(xyz_file)):
        assert snapshot.natoms == 3
        xyz = np.full_like(snapshot.xyz, 500 - i)
        assert pytest.approx(snapshot.xyz) == xyz
        assert snapshot.xyz.shape == (3, 6)
        assert snapshot.comment.startswith(f"# Step: {500-i} Box:")
        assert snapshot.atoms == ["A", "B", "C"]


def test_malformed_xyz(caplog):
    xyz_file = XYZDIR / "error.xyz"
    with pytest.raises(ValueError):
        with caplog.at_level(logging.ERROR):
            next(read_xyz_file(xyz_file))
            assert "Could not read the number of atoms" in caplog.text


def test_configuration_from_xyz():
    """Test that we can read configurations from a given xyz-file."""
    xyz_file = XYZDIR / "config.xyz"
    atoms, pos, _, _ = configuration_from_xyz_file(xyz_file, dim=3)
    assert atoms == ["Ba", "Hf", "O", "O", "O"]
    assert pytest.approx(pos) == CORRECT_XYZ
    # Test what happens if we use a 2D system:
    _, pos, vel, _ = configuration_from_xyz_file(xyz_file, dim=2)
    assert pos.shape == (5, 2)
    assert pytest.approx(vel) == np.zeros((5, 2))
    xyz_file = XYZDIR / "config2D.xyz"
    _, pos, _, _ = configuration_from_xyz_file(xyz_file, dim=2)
    assert pos.shape == (5, 2)
    # Test what happens if we have more columns:
    xyz_file = XYZDIR / "traj.xyz"
    _, pos, vel, _ = configuration_from_xyz_file(xyz_file, dim=3)
    assert pytest.approx(pos) == np.full((3, 3), 500)
    assert pytest.approx(vel) == np.full((3, 3), 500)
    _, pos, vel, force = configuration_from_xyz_file(xyz_file, dim=2)
    assert pytest.approx(pos) == np.full((3, 2), 500)
    assert pytest.approx(vel) == np.full((3, 2), 500)
    assert pytest.approx(force) == np.full((3, 2), 500)


def test_pad_to_nd():
    """Test that we can pad 1D and 2D coordinates."""
    vec1 = np.array(
        [
            1,
        ]
    )
    vec2 = np.array([1, 2])
    vec3 = np.array([1, 2, 3])
    vec4 = np.array([1, 2, 3, 4])
    pad_to_nd(vec1, dim=3)
    assert pytest.approx(np.array([1, 0, 0])) == pad_to_nd(vec1, dim=3)
    assert pytest.approx(np.array([1, 2, 0])) == pad_to_nd(vec2, dim=3)
    assert pytest.approx(vec3) == pad_to_nd(vec3, dim=3)
    assert pytest.approx(vec4) == pad_to_nd(vec4, dim=3)


def create_test_system(lattice="fcc"):
    """Create a test system."""
    if lattice == "fcc":
        xyz, size = generate_lattice("fcc", [3, 3, 3], density=0.9)
    elif lattice == "sq2":
        xyz, size = generate_lattice(lattice, [3, 3], density=0.9)
    else:
        xyz = [
            1.0,
        ]
        size = [None, None]
    box = Box(low=size[:, 0], high=size[:, 1])
    particles = Particles(dim=box.dim)
    for pos in xyz:
        particles.add_particle(pos=pos, mass=1.0, name="Ar", ptype=0)
    return System(box=box, particles=particles)


def test_system_to_xyz(tmp_path: pathlib.PosixPath):
    """Test that we can create a XYZ file from a system."""
    system = create_test_system()
    xyz_file = (tmp_path / "system.xyz").resolve()
    system_to_xyz(system, xyz_file)
    _, pos, _, _ = configuration_from_xyz_file(xyz_file)
    assert pytest.approx(pos) == system.particles.pos
    positions = [system.particles.pos.copy()]
    # Test that we can append to a file to create two frames:
    system.particles.pos += 1.234
    positions.append(system.particles.pos.copy())
    system_to_xyz(system, xyz_file, filemode="a", title="Second frame")
    for i, snapshot in enumerate(read_xyz_file(xyz_file)):
        assert pytest.approx(snapshot.xyz) == positions[i]
        if i == 0:
            assert snapshot.comment.startswith(
                "# TurtleMD system. Box: 4.9324"
            )
        elif i == 1:
            assert snapshot.comment.strip() == "Second frame"


def test_system_to_xyz_2dim(tmp_path: pathlib.PosixPath):
    """Test that we can write XYZ-files when the system is 2D or 1D."""
    # For 2D:
    system = create_test_system(lattice="sq2")
    xyz_file = (tmp_path / "systemsq2.xyz").resolve()
    system_to_xyz(system, xyz_file)
    _, pos, _, _ = configuration_from_xyz_file(xyz_file)
    assert pytest.approx(pos[:, 0]) == system.particles.pos[:, 0]
    assert pytest.approx(pos[:, 1]) == system.particles.pos[:, 1]
    assert pos.shape == (18, 3)
    assert system.particles.pos.shape == (18, 2)
    assert pytest.approx(pos[:, 2]) == np.zeros(18)

    # For 1D:
    box = Box(periodic=[False])
    particles = Particles(dim=box.dim)
    for i in range(3):
        particles.add_particle(
            pos=[i],
            mass=1.0,
            name="Ar",
            ptype=0,
        )
        system_1d = System(box=box, particles=particles)
        xyz_file = (tmp_path / "system1D.xyz").resolve()
        system_to_xyz(system_1d, xyz_file)
        _, pos, _, _ = configuration_from_xyz_file(xyz_file)
        pos1 = pos[:, 0]
        assert pytest.approx(pos1) == system_1d.particles.pos.flatten()
