from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.stats import truncnorm

from ConfigSpace.functional import is_close_to_integer
from ConfigSpace.hyperparameters._distributions import (
    DiscretizedContinuousScipyDistribution,
)
from ConfigSpace.hyperparameters._hp_components import ATOL, UnitScaler
from ConfigSpace.hyperparameters.hyperparameter import (
    HyperparameterWithPrior,
    IntegerHyperparameter,
)
from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter


@dataclass(init=False)
class NormalIntegerHyperparameter(
    IntegerHyperparameter,
    HyperparameterWithPrior[UniformIntegerHyperparameter],
):
    serializable_type_name: ClassVar[str] = "normal_int"
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
        self.lower = np.int64(np.rint(lower))
        self.upper = np.int64(np.rint(upper))

        try:
            scaler = UnitScaler(self.lower, self.upper, log=self.log, dtype=np.int64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            _default_value = np.rint(np.clip(self.mu, self.lower, self.upper)).astype(
                np.int64,
            )
        else:
            if not is_close_to_integer(default_value, atol=ATOL):
                raise TypeError(
                    f"`default_value` for hyperparameter '{name}' must be an integer."
                    f" Got '{type(default_value).__name__}' for {default_value=}.",
                )

            _default_value = np.rint(default_value).astype(np.int64)

        size = self.upper - self.lower + 1

        vectorized_mu = scaler.to_vector(np.array([self.mu]))[0]
        vectorized_sigma = scaler.vectorize_size(self.sigma)

        vec_truncnorm_dist = truncnorm(  # type: ignore
            a=(0.0 - vectorized_mu) / vectorized_sigma,
            b=(1.0 - vectorized_mu) / vectorized_sigma,
            loc=vectorized_mu,
            scale=vectorized_sigma,
        )

        vector_dist = DiscretizedContinuousScipyDistribution(
            rv=vec_truncnorm_dist,  # type: ignore
            steps=int(size),
            lower_vectorized=np.float64(0.0),
            upper_vectorized=np.float64(1.0),
        )
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.serializable_type_name,
            "log": self.log,
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "lower": int(self.lower),
            "upper": int(self.upper),
            "default_value": int(self.default_value),
        }
