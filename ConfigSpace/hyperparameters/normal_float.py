from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.stats import truncnorm
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from ConfigSpace.hyperparameters._distributions import ScipyContinuousDistribution
from ConfigSpace.hyperparameters._hp_components import (
    ROUND_PLACES,
    VECTORIZED_NUMERIC_LOWER,
    VECTORIZED_NUMERIC_UPPER,
    UnitScaler,
)
from ConfigSpace.hyperparameters.float_hyperparameter import FloatHyperparameter
from ConfigSpace.hyperparameters.hyperparameter import HyperparameterWithPrior
from ConfigSpace.hyperparameters.normal_integer import NormalIntegerHyperparameter
from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter


@dataclass(init=False)
class NormalFloatHyperparameter(
    FloatHyperparameter,
    HyperparameterWithPrior[UniformFloatHyperparameter],
):
    orderable: ClassVar[bool] = True
    mu: float
    sigma: float

    def __init__(
        self,
        name: str,
        mu: int | float,
        sigma: int | float,
        lower: float | int,
        upper: float | int,
        default_value: None | float = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        if mu < lower or mu > upper:
            raise ValueError(
                f"mu={mu} must be in the range [{lower}, {upper}] for hyperparameter"
                f"'{name}'",
            )

        self.lower = np.float64(lower)
        self.upper = np.float64(upper)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.log = bool(log)

        try:
            scaler = UnitScaler(self.lower, self.upper, log=log)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            _default_value = np.float64(self.mu)
        else:
            _default_value = np.float64(round(default_value, ROUND_PLACES))

        truncnorm_dist = truncnorm(  # type: ignore
            a=(self.lower - self.mu) / self.sigma,
            b=(self.upper - self.mu) / self.sigma,
            loc=self.mu,
            scale=self.sigma,
        )
        assert isinstance(truncnorm_dist, rv_continuous_frozen)

        max_density_point = np.clip(
            scaler.to_vector(np.array([self.mu]))[0],
            a_min=VECTORIZED_NUMERIC_LOWER,
            a_max=VECTORIZED_NUMERIC_UPPER,
        )
        max_density_value: float = truncnorm_dist.pdf(max_density_point)
        assert isinstance(max_density_value, float)

        vect_dist = ScipyContinuousDistribution(
            rv=truncnorm_dist,
            dtype=np.float64,
            max_density_value=max_density_value,
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
        )

    def to_uniform(self) -> UniformFloatHyperparameter:
        return UniformFloatHyperparameter(
            name=self.name,
            lower=self.lower,
            upper=self.upper,
            default_value=self.default_value,
            log=self.log,
            meta=self.meta,
        )

    def to_integer(self) -> NormalIntegerHyperparameter:
        return NormalIntegerHyperparameter(
            name=self.name,
            mu=round(self.mu),
            sigma=self.sigma,
            lower=np.ceil(self.lower),
            upper=np.floor(self.upper),
            default_value=round(self.default_value),
            log=self.log,
        )
