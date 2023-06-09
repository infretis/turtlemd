"""Common methods for the tests."""
import numpy as np
from numpy.random import Generator


class FakeRandomGenerator:
    """A fake random generator for testing."""

    def __init__(self, seed=0, norm_shift=False):
        self.seed = seed
        self.rgen = [
            0.78008018,
            0.04459916,
            0.76596775,
            0.97676713,
            0.53799598,
            0.98657116,
            0.36343553,
            0.55356511,
            0.03172585,
            0.48984682,
            0.73416687,
            0.98453452,
            0.55129902,
            0.40598753,
            0.59448394,
            0.26823255,
            0.31168372,
            0.05072849,
            0.44876368,
            0.94301709,
        ]
        self.length = len(self.rgen)
        self.norm_shift = norm_shift

    def rand(self, shape: int = 1) -> np.ndarray:
        """Fake numbers in [0, 1)."""
        numbers = []
        for _ in range(shape):
            if self.seed >= self.length:
                self.seed = 0
            numbers.append(self.rgen[self.seed])
            self.seed += 1
        return np.array(numbers)

    def normal(self, loc=0.0, scale=1.0, size=None) -> np.ndarray:
        """Mimic the normal method of random generators."""
        if self.norm_shift:
            shift = loc - 0.5
        else:
            shift = 0.0
        if size is None:
            return self.rand(shape=1) + shift
        numbers = np.zeros(size)
        for i in np.nditer(numbers, op_flags=["readwrite"]):
            i[...] = self.rand(shape=1)[0] + shift
        return numbers


def fake_multivariate_normal(
    rgen: Generator, mean: np.ndarray, cho: np.ndarray, dim: int
) -> np.ndarray:
    """Draw multivariate numbers for testing."""
    norm = rgen.normal(loc=0.0, scale=1.0, size=2 * dim)
    norm = norm.reshape(dim, 2)
    meanm = np.array([mean,] * dim)  # fmt: skip
    return 0.01 * (meanm + norm)
