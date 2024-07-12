from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Distribution:
    """Base distribution type."""


@dataclass
class Uniform(Distribution):
    """A uniform distribution."""


@dataclass
class Normal(Distribution):
    """Represents a normal distribution.

    Attributes:
        mu: The mean of the distribution
        sigma: The standard deviation of the float
    """

    mu: float
    sigma: float


@dataclass
class Beta(Distribution):
    """Represents a beta distribution.

    Attributes:
        alpha: The alpha parameter of a beta distribution
        beta: The beta parameter of a beta distribution
    """

    alpha: float
    beta: float
