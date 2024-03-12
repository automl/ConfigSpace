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
from ConfigSpace.hyperparameters.hyperparameter import HyperparameterWithPrior
from ConfigSpace.hyperparameters.normal_integer import NormalIntegerHyperparameter
from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter


@dataclass(init=False)
class NormalFloatHyperparameter(
    FloatHyperparameter,
    HyperparameterWithPrior[UniformFloatHyperparameter],
):
    serializable_type_name: ClassVar[str] = "normal_float"
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
        self.lower = np.float64(lower)
        self.upper = np.float64(upper)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.log = bool(log)

        try:
            scaler = UnitScaler(
                self.lower,
                self.upper,
                log=log,
                dtype=np.float64,
            )
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            _default_value = np.clip(np.float64(self.mu), self.lower, self.upper)
        else:
            _default_value = np.float64(default_value)

        _default_value = np.round(_default_value, ROUND_PLACES)

        vectorized_mu = scaler.to_vector(np.array([self.mu]))[0]
        vectorized_sigma = scaler.vectorize_size(self.sigma)

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
            dtype=np.float64,
            lower_vectorized=np.float64(0.0),
            upper_vectorized=np.float64(1.0),
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
            "lower": float(self.lower),
            "upper": float(self.upper),
            "default_value": float(self.default_value),
        }
