"""Test that we can read and interpret the input file."""
import pathlib

import numpy as np
import pytest

from turtlemd.inout.settings import (
    create_box_from_settings,
    read_settings_file,
)

HERE = pathlib.Path(__file__).resolve().parent


def test_create_box():
    """Test that we can create a box from settings."""
    settings = read_settings_file(HERE / "box1.toml")
    box = create_box_from_settings(settings)
    assert pytest.approx(box.low) == np.array([10.0, 11.0, 12.0])
    assert pytest.approx(box.high) == np.array([23.0, 24.0, 25.0])
    assert len(box.periodic) == 3
    assert all(box.periodic)
    # Read a similar file, but periodic is given as only true
    # (that is, not a list).
    settings = read_settings_file(HERE / "box2.toml")
    box = create_box_from_settings(settings)
    assert len(box.periodic) == 3
    assert all(box.periodic)
