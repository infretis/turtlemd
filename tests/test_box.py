import numpy as np
import pytest

from turtlemd.box import RectangularBox, interpret_box_size


def test_interpret_box_size():
    """Test that we understand the box size input."""
    size = np.array([10, 5, 1])
    low, high, length, periodic = interpret_box_size(size)
    for lowi in low:
        assert lowi == 0
    for sizei, highi in zip(size, high):
        assert highi == sizei
    for sizei, lengthi in zip(size, length):
        assert lengthi == sizei
    assert len(periodic) == 3
    assert all(periodic)

    low, high, length, periodic = interpret_box_size(
        size, periodicity=[True, True, False]
    )
    assert low[-1] == -float("inf")
    assert high[-1] == float("inf")
    assert length[-1] == float("inf")
    assert periodic[0]
    assert periodic[1]
    assert not periodic[2]

    size = np.array([[-1, 1], [0, 11]])
    low, high, length, periodic = interpret_box_size(size)
    assert low[0] == -1
    assert low[1] == 0
    assert high[0] == 1
    assert high[1] == 11
    assert length[0] == 2
    assert length[1] == 11
    assert periodic[0]
    assert periodic[1]


def test_initiate_box():
    """Test that we can initiate a box"""
    size = np.array([[-5, 5], [11, 12]])
    box = RectangularBox(size=size)
    assert pytest.approx(box.low) == np.array([-5, 11])
    assert pytest.approx(box.high) == np.array([5, 12])
    assert pytest.approx(box.length) == np.array([10, 1])
    assert box.periodic[0]
    assert box.periodic[1]


def test_volume():
    """Test that we can calculate the volume."""
    size = np.array([[-5, 5], [11, 12]])
    box = RectangularBox(size=size)
    vol = box.volume()
    assert pytest.approx(vol) == 10.0
    size = np.array([[-5, 5]])
    box = RectangularBox(size=size)
    vol = box.volume()
    assert pytest.approx(vol) == 10.0
    size = np.array([5])
    box = RectangularBox(size=size)
    vol = box.volume()
    assert pytest.approx(vol) == 5.0
    size = np.array([[-5, 5], [11, 12]])
    box = RectangularBox(size=size, periodicity=[True, False])
    vol = box.volume()
    assert vol == float("inf")


def test_print(capfd):
    """Test that we can print box information"""
    size = np.array([5])
    box = RectangularBox(size=size)
    print(box)
    captured = capfd.readouterr()
    assert "Hello, this is box." in captured.out
    assert f"My matrix is: {box.box_matrix}" in captured.out


def test_pbc_wrap():
    """Test that we can wrap coordinates."""
    length = np.array([10, 11, 12])
    box = RectangularBox(size=length, periodicity=[False, True, True])
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
    length = np.array([10, 10, 10])
    box = RectangularBox(size=length, periodicity=[False, True, True])
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
    length = np.array([10])
    box = RectangularBox(size=length, periodicity=[True])
    dist = np.array([7])
    pbc_dist = box.pbc_dist(dist)
    assert pytest.approx(pbc_dist) == np.array([-3])

    length = np.array([2, 10, 10])
    box = RectangularBox(size=length, periodicity=[True, True, False])
    pos1 = np.array([1, 5, 6])
    pos2 = np.array([1, -3, 100])
    dist = pos1 - pos2
    pbc_dist = box.pbc_dist(dist)
    assert pytest.approx(pbc_dist) == np.array([0, -2, -94])
