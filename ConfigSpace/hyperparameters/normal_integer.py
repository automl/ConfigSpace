from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.stats import truncnorm

from ConfigSpace.hyperparameters._distributions import (
    DiscretizedContinuousScipyDistribution,
)
from ConfigSpace.hyperparameters._hp_components import (
    VECTORIZED_NUMERIC_LOWER,
    VECTORIZED_NUMERIC_UPPER,
    UnitScaler,
)
from ConfigSpace.hyperparameters.hyperparameter import HyperparameterWithPrior
from ConfigSpace.hyperparameters.integer_hyperparameter import IntegerHyperparameter
from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter


@dataclass(init=False)
class NormalIntegerHyperparameter(
    IntegerHyperparameter,
    HyperparameterWithPrior[UniformIntegerHyperparameter],
):
    orderable: ClassVar[bool] = True
    mu: float
    sigma: float

    def __init__(
        self,
        name: str,
        mu: float,
        sigma: float,
        lower: int,
        upper: int,
        default_value: int | np.integer | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.log = bool(log)
        self.lower = np.int64(lower)
        self.upper = np.int64(upper)

        try:
            scaler = UnitScaler(self.lower, self.upper, log=self.log)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            _default_value = np.rint(
                scaler.to_value(np.array([self.mu]))[0],
            ).astype(np.int64)
        else:
            _default_value = np.rint(default_value).astype(np.int64)

        size = self.upper - self.lower + 1

        truncnorm_dist = truncnorm(  # type: ignore
            a=(self.lower - self.mu) / self.sigma,
            b=(self.upper - self.mu) / self.sigma,
            loc=self.mu,
            scale=self.sigma,
        )
        max_density_point = np.clip(
            scaler.to_vector(np.array([self.mu]))[0],
            a_min=VECTORIZED_NUMERIC_LOWER,
            a_max=VECTORIZED_NUMERIC_UPPER,
        )
        max_density_value: float = truncnorm_dist.pdf(max_density_point)  # type: ignore
        assert isinstance(max_density_value, float)
        vector_dist = DiscretizedContinuousScipyDistribution(
            dist=truncnorm_dist,  # type: ignore
            steps=size,
            max_density_value=max_density_value,
            normalization_constant_value=None,  # Will compute on demand
        )

        # Compute the normalization constant ahead of time
        constant = vector_dist._normalization_constant()
        vector_dist.normalization_constant_value = constant

        super().__init__(
            name=name,
            size=int(size),
            default_value=_default_value,
            meta=meta,
            transformer=scaler,
            vector_dist=vector_dist,
            neighborhood=vector_dist.neighborhood,
            neighborhood_size=self._neighborhood_size,
        )

    def to_uniform(self) -> UniformIntegerHyperparameter:
        return UniformIntegerHyperparameter(
            self.name,
            lower=self.lower,
            upper=self.upper,
            default_value=self.default_value,
            log=self.log,
            meta=self.meta,
        )
