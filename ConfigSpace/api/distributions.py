from dataclasses import dataclass


@dataclass
class Distribution:
    """Base distribution type"""

    pass


@dataclass
class Uniform(Distribution):
    """A uniform distribution"""

    pass


@dataclass
class Normal(Distribution):
    """Represents a normal distribution.

    Parameters
    ----------
    mu: float
        The mean of the distribution

    sigma: float
        The standard deviation of the float
    """

    mu: float
    sigma: float


@dataclass
class Beta(Distribution):
    """Represents a beta distribution.

    Parameters
    ----------
    alpha: float
        The alpha parameter of a beta distribution

    beta: float
        The beta parameter of a beta distribution
    """

    alpha: float
    beta: float
