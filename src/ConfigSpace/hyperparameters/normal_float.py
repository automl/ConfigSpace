from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.stats import truncnorm
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from ConfigSpace.hyperparameters.distributions import ScipyContinuousDistribution
from ConfigSpace.hyperparameters.float_hyperparameter import FloatHyperparameter
from ConfigSpace.hyperparameters.hp_components import ROUND_PLACES, UnitScaler
from ConfigSpace.types import Number, f64

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.normal_integer import NormalIntegerHyperparameter


@dataclass(init=False)
class NormalFloatHyperparameter(FloatHyperparameter):
    """A normally distributed float hyperparameter.

    The 'mu' and 'sigma' parameters define the mean and standard deviation of the
    normal distribution. The 'lower' and 'upper' parameters move the distribution
    from the `[0, 1]`-range and scale it appropriately, but the shape of the
    distribution is preserved as if it were in `[0, 1]`-range.

    Its values are sampled from a normal distribution `N(mu, sigma)`.

    ```python exec="True" result="python" source="material-block"
    from ConfigSpace import NormalFloatHyperparameter

    n = NormalFloatHyperparameter('n', mu=5.5, sigma=2, lower=0, upper=11, log=False)
    print(n)
    ```
    """

    ORDERABLE: ClassVar[bool] = True

    mu: float
    """Mean of the normal distribution."""

    sigma: float
    """Standard deviation of the normal distribution."""

    lower: float
    """Lower bound of a range of values from which the hyperparameter represents."""

    upper: float
    """Upper bound of a range of values from which the hyperparameter represents."""

    log: bool
    """If `True` the values of the hyperparameter will be sampled on a log-scale."""

    name: str
    """Name of the hyperparameter, with which it can be accessed."""

    default_value: float
    """The default value of this hyperparameter."""

    meta: Mapping[Hashable, Any] | None
    """Field for holding meta data provided by the user. Not used by the ConfigSpace."""

    size: float = field(init=False)
    """Size of the hyperparameter, which is always infinity for a normal float."""

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
        """A normally distributed float hyperparameter.

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
                Field for meta data provided by the user. Not used by ConfigSpace.
        """
        if mu <= 0 and log:
            raise ValueError(
                f"Hyperparameter '{name}' has illegal settings: "
                f"mu={mu} must be positive for log-scale.",
            )

        self.lower = float(np.round(lower, ROUND_PLACES))
        self.upper = float(np.round(upper, ROUND_PLACES))
        self.mu = float(np.round(mu, ROUND_PLACES))
        self.sigma = float(np.round(sigma, ROUND_PLACES))
        self.log = bool(log)

        try:
            scaler = UnitScaler(f64(self.lower), f64(self.upper), log=log, dtype=f64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            _default_value = np.clip(self.mu, self.lower, self.upper)
        else:
            _default_value = default_value

        _default_value = float(np.round(_default_value, ROUND_PLACES))

        vectorized_mu = scaler.to_vector(np.array([self.mu]))[0]
        vectorized_sigma = scaler.vectorize_size(f64(self.sigma))

        vec_truncnorm_dist = truncnorm(  # type: ignore
            a=(0.0 - vectorized_mu) / vectorized_sigma,
            b=(1.0 - vectorized_mu) / vectorized_sigma,
            loc=vectorized_mu,
            scale=vectorized_sigma,
        )
        assert isinstance(vec_truncnorm_dist, rv_continuous_frozen)

        max_density_point = np.clip(vectorized_mu, 0.0, 1.0)
        vect_dist = ScipyContinuousDistribution(
            rv=vec_truncnorm_dist,
            lower_vectorized=f64(0.0),
            upper_vectorized=f64(1.0),
            _max_density=vec_truncnorm_dist.pdf(max_density_point),
        )
        super().__init__(
            name=name,
            size=np.inf,
            default_value=_default_value,
            meta=meta,
            transformer=scaler,
            vector_dist=vect_dist,
            neighborhood=vect_dist.neighborhood,
            neighborhood_size=np.inf,
            value_cast=float,
        )

    def to_integer(self) -> NormalIntegerHyperparameter:
        """Convert this hyperparameter to a normal integer hyperparameter.

        This is done by rounding the lower and upper bounds and the default value
        as required.

        Returns:
            A normal integer hyperparameter.
        """
        from ConfigSpace.hyperparameters.normal_integer import (
            NormalIntegerHyperparameter,
        )

        return NormalIntegerHyperparameter(
            name=self.name,
            mu=self.mu,
            sigma=self.sigma,
            lower=np.ceil(self.lower),
            upper=np.floor(self.upper),
            default_value=np.rint(self.default_value),
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
