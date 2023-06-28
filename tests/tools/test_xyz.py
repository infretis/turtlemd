"""Test that we can read xyz-files."""
import logging
import pathlib

import numpy as np
import pytest

from turtlemd.tools.xyz import read_xyz_file

HERE = pathlib.Path(__file__).resolve().parent

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
    xyz_file = HERE / "config.xyz"
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
    xyz_file = HERE / "traj.xyz"
    for i, snapshot in enumerate(read_xyz_file(xyz_file)):
        assert snapshot.natoms == 3
        xyz = np.full_like(snapshot.xyz, 500 - i)
        assert pytest.approx(snapshot.xyz) == xyz
        assert snapshot.xyz.shape == (3, 6)
        assert snapshot.comment.startswith(f"# Step: {500-i} Box:")
        assert snapshot.atoms == ["A", "B", "C"]


def test_malformed_xyz(caplog):
    xyz_file = HERE / "error.xyz"
    with pytest.raises(ValueError):
        with caplog.at_level(logging.ERROR):
            next(read_xyz_file(xyz_file))
            assert "Could not read the number of atoms" in caplog.text
