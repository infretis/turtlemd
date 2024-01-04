"""Test that we can read and interpret the input file."""
import logging
import pathlib
import shutil

import numpy as np
import pytest

from turtlemd.inout.settings import (
    create_box_from_settings,
    create_integrator_from_settings,
    create_system_from_settings,
    read_settings_file,
    search_for_setting,
)
from turtlemd.integrators import (
    LangevinInertia,
    LangevinOverdamped,
    VelocityVerlet,
    Verlet,
)

HERE = pathlib.Path(__file__).resolve().parent


def test_search_for_setting():
    """Test that we can search for settings."""
    settings = read_settings_file(HERE / "nesting.toml")
    assert settings["system"]["particles"]["file"] == "system.particles"
    assert settings["system"]["file"] == "system"
    assert settings["x"]["y"]["z"]["w"]["file"] == "x.y.z.w"
    match = search_for_setting(settings, "file")
    for item in match:
        item["file"] = "updated"
    assert settings["system"]["particles"]["file"] == "updated"
    assert settings["system"]["file"] == "updated"
    assert settings["x"]["y"]["z"]["w"]["file"] == "updated"


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
    # Read a triclinic box:
    settings = read_settings_file(HERE / "box3.toml")
    boxt = create_box_from_settings(settings)
    assert pytest.approx(boxt.alpha) == 75
    assert pytest.approx(boxt.beta) == 60
    assert pytest.approx(boxt.gamma) == 90


def test_create_integrator(caplog: pytest.LogCaptureFixture):
    """Test the creation of integrators from settings."""
    settings = read_settings_file(HERE / "integrators" / "verlet.toml")
    integ = create_integrator_from_settings(settings)
    assert integ is not None
    assert integ.timestep == 1234.5678
    assert isinstance(integ, Verlet)

    settings = read_settings_file(HERE / "integrators" / "verlet.toml")
    settings["integrator"].pop("class")
    with pytest.raises(ValueError):
        with caplog.at_level(logging.ERROR):
            create_integrator_from_settings(settings)
            assert 'No "class" given for integrator' in caplog.text

    settings = read_settings_file(HERE / "integrators" / "integrator.toml")
    with pytest.raises(ValueError):
        with caplog.at_level(logging.ERROR):
            create_integrator_from_settings(settings)
            assert "Could not create unknown class" in caplog.text

    # Test that we can create all integrators:
    classes = (Verlet, VelocityVerlet, LangevinOverdamped, LangevinInertia)
    files = (
        "verlet.toml",
        "velocityverlet.toml",
        "langevin1.toml",
        "langevin2.toml",
    )

    for klass, filei in zip(classes, files):
        settings = read_settings_file(HERE / "integrators" / filei)
        integ = create_integrator_from_settings(settings)
        assert integ is not None
        assert isinstance(integ, klass)


def test_create_system(tmp_path: pathlib.PosixPath):
    """Test that we can create systems."""
    settings_file = HERE / "system.toml"
    settings = read_settings_file(settings_file)
    system = create_system_from_settings(settings)
    correct = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert pytest.approx(system.particles.pos) == correct
    # Test that we can read relative paths:
    settings_file = HERE / "config" / "system_one_down.toml"
    settings = read_settings_file(settings_file)
    system = create_system_from_settings(settings)
    correct2 = np.full((2, 3), 8.0)
    assert pytest.approx(system.particles.pos) == correct2
    # Test a absolute path:
    new_file = (tmp_path / "start_absolute.xyz").resolve()
    shutil.copy(HERE / "config" / "start.xyz", new_file)
    settings["particles"] = {"file": new_file}
    system = create_system_from_settings(settings)
    assert pytest.approx(system.particles.pos) == correct
    # Test a missing file:
    with pytest.raises(FileNotFoundError):
        new_file = (tmp_path / "missing_file.xyz").resolve()
        settings["particles"] = {"file": new_file}
        create_system_from_settings(settings)
