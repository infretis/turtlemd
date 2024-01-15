"""Make random number generators available.

This module is just present to make sure that all random
number generators are created by the same method.
"""
from __future__ import annotations

from numpy.random import Generator, default_rng


def create_random_generator(seed: int | None = None) -> Generator:
    """Create a random generator."""
    return default_rng(seed=seed)
