"""Test that we can read and interpret the input file."""
import logging
import pathlib

import numpy as np
import pytest

from turtlemd.inout.settings import (
    create_box_from_settings,
    create_integrator_from_settings,
    read_settings_file,
)
from turtlemd.integrators import Verlet

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


def test_create_integrator(caplog):
    """Test the creation of integrators from settings."""
    settings = read_settings_file(HERE / "verlet.toml")
    integ = create_integrator_from_settings(settings)
    assert integ.timestep == 1234.5678
    assert isinstance(integ, Verlet)

    settings = read_settings_file(HERE / "verlet.toml")
    settings["integrator"].pop("class")
    with pytest.raises(ValueError):
        with caplog.at_level(logging.ERROR):
            create_integrator_from_settings(settings)
            assert 'No "class" given for integrator' in caplog.text

    settings = read_settings_file(HERE / "integrator.toml")
    with pytest.raises(ValueError):
        with caplog.at_level(logging.ERROR):
            create_integrator_from_settings(settings)
            assert "Could not create unknown class" in caplog.text
