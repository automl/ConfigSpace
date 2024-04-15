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
from ConfigSpace.hyperparameters.hyperparameter import IntegerHyperparameter
from ConfigSpace.types import Number, f64, i64


@dataclass(init=False)
class NormalIntegerHyperparameter(IntegerHyperparameter):
    ORDERABLE: ClassVar[bool] = True

    mu: f64
    sigma: f64
    lower: i64
    upper: i64
    log: bool

    name: str
    default_value: i64
    meta: Mapping[Hashable, Any] | None

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
        self.mu = f64(mu)
        self.sigma = f64(sigma)
        self.lower = i64(np.rint(lower))
        self.upper = i64(np.rint(upper))
        self.log = bool(log)

        try:
            scaler = UnitScaler(self.lower, self.upper, log=self.log, dtype=i64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            _default_value = np.rint(np.clip(self.mu, self.lower, self.upper)).astype(
                i64,
            )
        else:
            if not is_close_to_integer(f64(default_value), atol=ATOL):
                raise TypeError(
                    f"`default_value` for hyperparameter '{name}' must be an integer."
                    f" Got '{type(default_value).__name__}' for {default_value=}.",
                )

            _default_value = np.rint(default_value).astype(i64)

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
            neighborhood_size=self._neighborhood_size,
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
