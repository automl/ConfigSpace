from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.stats import truncnorm
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from ConfigSpace.hyperparameters._distributions import ScipyContinuousDistribution
from ConfigSpace.hyperparameters._hp_components import ROUND_PLACES, UnitScaler
from ConfigSpace.hyperparameters.float_hyperparameter import FloatHyperparameter
from ConfigSpace.hyperparameters.normal_integer import NormalIntegerHyperparameter
from ConfigSpace.types import Number, f64


@dataclass(init=False)
class NormalFloatHyperparameter(FloatHyperparameter):
    ORDERABLE: ClassVar[bool] = True

    mu: float
    sigma: float
    lower: float
    upper: float
    log: bool

    name: str
    default_value: float
    meta: Mapping[Hashable, Any] | None
    size: float

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
        if mu <= 0 and log:
            raise ValueError(
                f"Hyperparameter '{name}' has illegal settings: "
                f"mu={mu} must be positive for log-scale.",
            )

        self.lower = float(lower)
        self.upper = float(upper)
        self.mu = float(mu)
        self.sigma = float(sigma)
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
