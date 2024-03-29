"""Test the simulation box."""
import logging
import pathlib

import numpy as np
import pytest

from turtlemd.system.box import Box, TriclinicBox, guess_dimensionality

HERE = pathlib.Path(__file__).resolve().parent


def test_guess_dimensionality():
    """Test that we understand the box size input."""
    dim = guess_dimensionality()
    assert dim == 1

    dim = guess_dimensionality(low=[-10])
    assert dim == 1

    dim = guess_dimensionality(high=[10])
    assert dim == 1

    dim = guess_dimensionality(periodic=[True])
    assert dim == 1

    dim = guess_dimensionality(low=[-10], high=[10])
    assert dim == 1

    dim = guess_dimensionality(low=[-10, 10], high=[10, 10])
    assert dim == 2

    dim = guess_dimensionality(
        low=[-10, 10], high=[10, 10], periodic=[True, False]
    )
    assert dim == 2

    with pytest.raises(ValueError):
        guess_dimensionality(
            low=[-10, 10], high=[10, 10], periodic=[True, False, True]
        )


def test_initiate_box(caplog):
    """Test that we can initiate a box"""
    box = Box(low=[-5, 11], high=[5, 12])
    assert pytest.approx(box.low) == np.array([-5, 11])
    assert pytest.approx(box.high) == np.array([5, 12])
    assert pytest.approx(box.length) == np.array([10, 1])
    assert box.periodic[0]
    assert box.periodic[1]

    with caplog.at_level(logging.WARNING):
        box = Box()
        assert pytest.approx(box.low) == np.array([0.0])
        assert pytest.approx(box.high) == np.array([1.0])
        assert all(box.periodic)
        assert box.dim == 1
        assert "Missing low/high/periodic" in caplog.text
        assert "Set box low values" in caplog.text
        assert "Set box high values" in caplog.text

    box = Box(high=[10, 10, 10], periodic=[True, True, True])
    assert pytest.approx(box.low) == np.zeros(3)
    assert pytest.approx(box.high) == np.array([10, 10, 10])
    assert pytest.approx(box.length) == np.array([10, 10, 10])
    assert all(box.periodic)

    box = Box(high=[10, 10, 10], periodic=[True, False, False])
    assert pytest.approx(box.low[0]) == 0
    assert box.low[1] == -float("inf")
    assert box.low[2] == -float("inf")
    assert pytest.approx(box.length[0]) == 10.0
    assert box.length[1] == float("inf")
    assert box.length[2] == float("inf")

    box = Box(periodic=[False, False])
    assert box.low[0] == -float("inf")
    assert box.low[1] == -float("inf")
    assert box.high[0] == float("inf")
    assert box.high[1] == float("inf")
    assert box.length[0] == float("inf")
    assert box.length[1] == float("inf")

    with pytest.raises(ValueError):
        TriclinicBox(periodic=[True])

    with pytest.raises(ValueError):
        TriclinicBox(periodic=[True, True, True])

    with pytest.raises(ValueError):
        TriclinicBox(periodic=[True, True, True], alpha=90, beta=90, gamma=90)


def test_volume():
    """Test that we can calculate the volume."""
    box = Box(low=[-5, 11], high=[5, 12])
    vol = box.volume()
    assert pytest.approx(vol) == 10.0
    box = Box(low=[-5], high=[5])
    vol = box.volume()
    assert pytest.approx(vol) == 10.0
    box = Box(high=[5])
    vol = box.volume()
    assert pytest.approx(vol) == 5.0
    box = Box(
        low=[-5, -float("inf")],
        high=[5, float("inf")],
        periodic=[True, False],
    )
    vol = box.volume()
    assert vol == float("inf")


def test_print(capfd):
    """Test that we can print box information"""
    box = Box(high=[5])
    print(box)
    captured = capfd.readouterr()
    assert "Hello, this is box" in captured.out
    assert f"and my matrix is:\n{box.box_matrix}" in captured.out

    box = TriclinicBox(high=[10.0, 10.0], alpha=None, beta=None, gamma=45.0)
    print(box)
    captured = capfd.readouterr()
    assert "Hello, this is triclinic box and my matrix" in captured.out


def test_pbc_wrap():
    """Test that we can wrap coordinates."""
    box = Box(high=[10, 11, 12], periodic=[False, True, True])
    pos = np.array(
        [
            [11, 10, 14],
        ]
    )
    correct = np.array(
        [
            [11, 10, 2],
        ]
    )
    pbc_pos = box.pbc_wrap(pos)
    assert pytest.approx(correct) == pbc_pos


def test_pbc_dist_matrix():
    """Test that we can do pbc for a matrix"""
    box = Box(high=[10, 10, 10], periodic=[False, True, True])
    dist = np.array(
        [
            [8.0, 7.0, 9.0],
            [100.0, 1.0, 7.0],
        ]
    )
    correct = np.array(
        [
            [8.0, -3.0, -1.0],
            [100.0, 1.0, -3.0],
        ]
    )
    pbc_dist = box.pbc_dist_matrix(dist)
    assert pytest.approx(pbc_dist) == correct
    assert not pytest.approx(dist) == pbc_dist


def test_pbc_dist():
    """Test that we can do pbc for a single coordinate."""
    box = Box(high=[10], periodic=[True])
    dist = np.array([7])
    pbc_dist = box.pbc_dist(dist)
    assert pytest.approx(pbc_dist) == np.array([-3])

    box = Box(high=[2, 10, 10], periodic=[True, True, False])
    pos1 = np.array([1, 5, 6])
    pos2 = np.array([1, -3, 100])
    dist = pos1 - pos2
    pbc_dist = box.pbc_dist(dist)
    assert pytest.approx(pbc_dist) == np.array([0, -2, -94])


def test_triclinic_pbc_wrap_2d():
    """Test the triclinic PBC wrapping."""
    data_dir = HERE / "data"
    xyz = np.loadtxt(data_dir / "pbc_triclinic_raw.data.gz")
    box = TriclinicBox(high=[10.0, 10.0], alpha=None, beta=None, gamma=45.0)
    wrap = box.pbc_wrap(xyz)
    vmd = np.loadtxt(data_dir / "pbc_triclinic_wrapped_by_vmd.data.gz")
    assert pytest.approx(wrap, abs=1e-5) == vmd


def test_triclinic_pbc_wrap_3d():
    """Test the triclinic PBC wrapping."""
    data_dir = HERE / "data"
    xyz = np.loadtxt(data_dir / "distance" / "system2.txt.gz")
    box = TriclinicBox(
        high=[10, 10, 10],
        alpha=60,
        beta=60,
        gamma=75,
    )
    wrap = box.pbc_wrap(xyz)
    vmd = np.loadtxt(data_dir / "pbc_triclinic_wrapped_by_vmd_3d.data.gz")
    assert pytest.approx(wrap, abs=1e-5) == vmd


def test_triclinic_pbc_dist_2d():
    """Test the triclinic PBC for distances."""
    data_dir = HERE / "data" / "distance"
    xyz = np.loadtxt(data_dir / "system1.txt.gz")
    box = TriclinicBox(
        high=[10, 10],
        alpha=None,
        beta=None,
        gamma=75,
    )
    distances = []
    for xyzi in xyz:
        dist = [np.linalg.norm(box.pbc_dist(xyzi - xyzj)) for xyzj in xyz]
        distances.append(dist)
    distances = np.array(distances)
    correct = np.loadtxt(data_dir / "dist1.txt.gz")
    assert pytest.approx(distances, abs=1e-5) == correct
    raw_dist = [xyz[0] - xyzj for xyzj in xyz]
    distances2 = np.linalg.norm(box.pbc_dist_matrix(raw_dist), axis=1)
    assert pytest.approx(distances[0, :]) == distances2


def test_triclinic_pbc_dist_3d():
    """Test the triclinic PBC for distances."""
    data_dir = HERE / "data" / "distance"
    xyz = np.loadtxt(data_dir / "system2.txt.gz")
    box = TriclinicBox(
        high=[10, 10, 10],
        alpha=60,
        beta=60,
        gamma=75,
    )
    distances = []
    for xyzi in xyz:
        dist = [np.linalg.norm(box.pbc_dist(xyzi - xyzj)) for xyzj in xyz]
        distances.append(dist)
    distances = np.array(distances)
    correct = np.loadtxt(data_dir / "dist2.txt.gz")
    assert pytest.approx(distances, abs=1e-5) == correct
