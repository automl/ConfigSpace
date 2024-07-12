from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.stats import truncnorm

from ConfigSpace.functional import is_close_to_integer
from ConfigSpace.hyperparameters.distributions import (
    DiscretizedContinuousScipyDistribution,
)
from ConfigSpace.hyperparameters.hp_components import ATOL, UnitScaler
from ConfigSpace.hyperparameters.hyperparameter import IntegerHyperparameter
from ConfigSpace.types import Number, f64, i64

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.normal_float import NormalFloatHyperparameter


@dataclass(init=False)
class NormalIntegerHyperparameter(IntegerHyperparameter):
    """A normally distributed integer hyperparameter.

    The 'mu' and 'sigma' parameters define the mean and standard deviation of the
    normal distribution. The 'lower' and 'upper' parameters move the distribution
    from the `[0, 1]`-range and scale it appropriately, but the shape of the
    distribution is preserved as if it were in `[0, 1]`-range.

    Its values are sampled from a normal distribution `N(mu, sigma)`.

    ```python exec="True" result="python" source="material-block"
    from ConfigSpace import NormalIntegerHyperparameter

    n = NormalIntegerHyperparameter('n', mu=150, sigma=20, lower=100, upper=200)
    print(n)
    """

    ORDERABLE: ClassVar[bool] = True

    mu: float
    """Mean of the normal distribution."""

    sigma: float
    """Standard deviation of the normal distribution."""

    lower: int
    """Lower bound of a range of values from which the hyperparameter represents."""

    upper: int
    """Upper bound of a range of values from which the hyperparameter represents."""

    log: bool
    """If `True` the values of the hyperparameter will be sampled on a log-scale."""

    name: str
    """Name of the hyperparameter, with which it can be accessed."""

    default_value: int
    """The default value of this hyperparameter."""

    meta: Mapping[Hashable, Any] | None
    """Field for holding meta data provided by the user. Not used by the ConfigSpace."""

    size: int = field(init=False)
    """Size of the hyperparameter, which is the count of ints between `upper` and
    `lower`, inclusive."""

    def __init__(
        self,
        name: str,
        mu: Number,
        sigma: Number,
        lower: Number,
        upper: Number,
        default_value: Number | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """A normally distributed integer hyperparameter.

        Args:
            name:
                Name of the hyperparameter, with which it can be accessed
            mu:
                Mean of the normal distribution
            sigma:
                Standard deviation of the normal distribution
            lower:
                Lower bound of of values from which the hyperparameter represents
            upper:
                Upper bound of of values from which the hyperparameter represents
            default_value:
                The default value of this hyperparameter
            log:
                If `True` the values will be sampled on a log-scale
            meta:
                Field for holding meta data provided by the user. Not used by
                ConfigSpace.
        """
        if mu <= 0 and log:
            raise ValueError(
                f"Hyperparameter '{name}' has illegal settings: "
                f"mu={mu} must be positive for log-scale.",
            )

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.lower = int(np.rint(lower))
        self.upper = int(np.rint(upper))
        self.log = bool(log)

        try:
            scaler = UnitScaler(
                i64(self.lower),
                i64(self.upper),
                log=self.log,
                dtype=i64,
            )
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            _default_value = int(np.rint(np.clip(self.mu, self.lower, self.upper)))
        else:
            if not is_close_to_integer(f64(default_value), atol=ATOL):
                raise TypeError(
                    f"`default_value` for hyperparameter '{name}' must be an integer."
                    f" Got '{type(default_value).__name__}' for {default_value=}.",
                )

            _default_value = int(np.rint(default_value))

        size = self.upper - self.lower + 1

        vectorized_mu = scaler.to_vector(np.array([self.mu]))[0]
        vectorized_sigma = scaler.vectorize_size(f64(self.sigma))

        vec_truncnorm_dist = truncnorm(  # type: ignore
            a=(0.0 - vectorized_mu) / vectorized_sigma,
            b=(1.0 - vectorized_mu) / vectorized_sigma,
            loc=vectorized_mu,
            scale=vectorized_sigma,
        )

        vector_dist = DiscretizedContinuousScipyDistribution(
            rv=vec_truncnorm_dist,  # type: ignore
            steps=int(size),
            lower_vectorized=f64(0.0),
            upper_vectorized=f64(1.0),
        )
        super().__init__(
            name=name,
            size=int(size),
            default_value=_default_value,
            meta=meta,
            transformer=scaler,
            vector_dist=vector_dist,
            neighborhood=vector_dist.neighborhood,
            neighborhood_size=self._integer_neighborhood_size,
            value_cast=int,
        )

    def to_float(self) -> NormalFloatHyperparameter:
        """Convert this hyperparameter to a normal float hyperparameter."""
        from ConfigSpace.hyperparameters.normal_float import NormalFloatHyperparameter

        return NormalFloatHyperparameter(
            name=self.name,
            mu=self.mu,
            sigma=self.sigma,
            lower=self.lower,
            upper=self.upper,
            default_value=self.default_value,
            log=self.log,
        )

    def __str__(self) -> str:
        parts = [
            self.name,
            f"Type: {str(self.__class__.__name__).replace('Hyperparameter', '')}",
            f"Mu: {self.mu}",
            f"Sigma: {self.sigma}",
            f"Range: [{self.lower}, {self.upper}]",
            f"Default: {self.default_value}",
        ]
        if self.log:
            parts.append("on log-scale")

        return ", ".join(parts)
