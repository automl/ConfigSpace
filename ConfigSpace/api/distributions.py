"""The base distributions currently available.

Note:
* A reason to make these classes is so we can seperate distributions from types so we don't
  have exploding classes with each new distribution, i.e. as when added a BetaInteger and BetaFloat.
  This doesn't solve it for now but leaves the API open to do so.

  Ideally, if someone wanted to implement a PoissonDistribution, they would only need to do the
  following and not have to submit a PR.

  .. code::python

    class Poisson(Distribution):
        ...

    Float("name", (1, 10), distribution=Poisson(...))

"""
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
