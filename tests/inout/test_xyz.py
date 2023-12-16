"""Test that we can read xyz-files."""
import logging
import pathlib

import numpy as np
import pytest

from turtlemd.inout.xyz import particles_from_xyz_file, read_xyz_file

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


def test_particles_from_xyz():
    """Test that we can create particles from a given xyz-file."""
    xyz_file = XYZDIR / "config.xyz"
    # Set up some masses:
    masses = {
        "O": 16.0,
        "Hf": 178.49,
        "Ba": 137.33,
    }
    particles = particles_from_xyz_file(xyz_file, dim=3, masses=masses)
    assert list(particles.name) == ["Ba", "Hf", "O", "O", "O"]
    assert pytest.approx(particles.pos) == CORRECT_XYZ
    mass_table = np.array([137.33, 178.49, 16.0, 16.0, 16.0]).reshape(5, 1)
    assert pytest.approx(particles.mass) == mass_table
    assert particles.ptype[0] != particles.ptype[1]
    assert particles.ptype[0] != particles.ptype[2]
    assert particles.ptype[2] == particles.ptype[3]
    assert particles.ptype[2] == particles.ptype[4]
    # Test what happens if we use a 2D system:
    particles = particles_from_xyz_file(xyz_file, dim=2)
    assert particles.pos.shape == (5, 2)
    assert pytest.approx(particles.vel) == np.zeros((5, 2))
    xyz_file = XYZDIR / "config2D.xyz"
    particles = particles_from_xyz_file(xyz_file, dim=2)
    assert particles.pos.shape == (5, 2)
    # Test what happens if we have more columns:
    xyz_file = XYZDIR / "traj.xyz"
    particles = particles_from_xyz_file(xyz_file, dim=3)
    assert pytest.approx(particles.pos) == np.full((3, 3), 500)
    assert pytest.approx(particles.vel) == np.full((3, 3), 500)
    particles = particles_from_xyz_file(xyz_file, dim=2)
    assert pytest.approx(particles.pos) == np.full((3, 2), 500)
    assert pytest.approx(particles.vel) == np.full((3, 2), 500)
    assert pytest.approx(particles.force) == np.full((3, 2), 500)
