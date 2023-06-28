import numpy as np
import pytest

from turtlemd.tools.tools import UNIT_CELL, generate_lattice

CORRECT_XYZ = {
    "fcc": np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
    ),
    "bcc": np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [1.5, 0.5, 0.5]]
    ),
    "diamond": np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.705, 0.705],
            [0.705, 0.0, 0.705],
            [0.705, 0.705, 0.0],
            [0.3525, 0.3525, 0.3525],
            [0.3525, 1.0575, 1.0575],
            [1.0575, 0.3525, 1.0575],
            [1.0575, 1.0575, 0.3525],
        ]
    ),
    "hcp": np.array(
        [
            [0.0, 0.0, 0.0],
            [0.45, 0.45, 0.0],
            [0.45, 0.75, 0.45],
            [0.0, 0.3, 0.45],
        ]
    ),
    "sc": np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.14159],
            [0.0, 0.0, 6.28318],
            [0.0, 0.0, 9.42477],
        ]
    ),
    "sq": np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 0.0], [3.0, 3.0]]),
    "sq2": np.array(
        [
            [0.0, 0.0],
            [1.5, 1.5],
            [0.0, 3.0],
            [1.5, 4.5],
            [3.0, 0.0],
            [4.5, 1.5],
            [3.0, 3.0],
            [4.5, 4.5],
        ]
    ),
}


CORRECT_SIZE = {
    "fcc": np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]),
    "bcc": np.array([[0.0, 2.0], [0.0, 1.0], [0.0, 1.0]]),
    "diamond": np.array([[0.0, 1.41], [0.0, 1.41], [0.0, 1.41]]),
    "hcp": np.array([[0.0, 0.9], [0.0, 0.9], [0.0, 0.9]]),
    "sc": np.array([[0.0, 3.14159], [0.0, 3.14159], [0.0, 12.56636]]),
    "sq": np.array([[0.0, 6.0], [0.0, 6.0]]),
    "sq2": np.array([[0.0, 6.0], [0.0, 6.0]]),
}


def test_lattice_generation():
    """Test that we can generate lattices."""
    cases = [
        {"lattice": "bcc", "repeat": [2, 1, 1], "lattice_constant": 1.0},
        {"lattice": "fcc", "repeat": [1, 1, 1], "lattice_constant": 2.0},
        {"lattice": "hcp", "repeat": [1, 1, 1], "lattice_constant": 0.9},
        {"lattice": "diamond", "repeat": [1, 1, 1], "lattice_constant": 1.41},
        {"lattice": "sc", "repeat": [1, 1, 4], "lattice_constant": 3.14159},
        {"lattice": "sq", "repeat": [2, 2], "lattice_constant": 3},
        {"lattice": "sq2", "repeat": [2, 2], "lattice_constant": 3},
        {"lattice": "fcc", "repeat": [1, 1, 1], "density": 0.5},
    ]

    for case in cases:
        pos, size = generate_lattice(**case)
        assert pytest.approx(size) == CORRECT_SIZE[case["lattice"]]
        assert pytest.approx(pos) == CORRECT_XYZ[case["lattice"]]


def test_lattice_special():
    """Test all the 'special' cases for lattice generation."""
    # Test that we fail for a unknow lattice.
    with pytest.raises(ValueError):
        generate_lattice(lattice="this is not a lattice!")
    # Test that we set lcon to just one and repeat to ones, if
    # these are not given:
    pos, size = generate_lattice(lattice="fcc")
    assert pytest.approx(pos) == UNIT_CELL["fcc"]
    assert pytest.approx(size) == np.array([[0, 1], [0, 1], [0, 1]])
