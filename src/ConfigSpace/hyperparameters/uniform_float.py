from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
)

import numpy as np

from ConfigSpace.hyperparameters._distributions import UnitUniformContinuousDistribution
from ConfigSpace.hyperparameters._hp_components import ROUND_PLACES, UnitScaler
from ConfigSpace.hyperparameters.hyperparameter import FloatHyperparameter
from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter


@dataclass(init=False)
class UniformFloatHyperparameter(FloatHyperparameter):
    serializable_type_name: ClassVar[str] = "uniform_float"
    orderable: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        lower: int | float | np.floating,
        upper: int | float | np.floating,
        default_value: int | float | np.floating | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        self.lower = np.float64(lower)
        self.upper = np.float64(upper)
        self.log = log

        try:
            scaler = UnitScaler(self.lower, self.upper, log=log, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        vect_dist = UnitUniformContinuousDistribution(
            pdf_max_density=1 / float(self.upper - self.lower),
        )
        super().__init__(
            name=name,
            size=np.inf,
            default_value=np.float64(
                np.round(
                    default_value
                    if default_value is not None
                    else scaler.to_value(np.array([0.5]))[0],
                    ROUND_PLACES,
                ),
            ),
            meta=meta,
            transformer=scaler,
            neighborhood=vect_dist.neighborhood,
            vector_dist=vect_dist,
            neighborhood_size=np.inf,
        )

    def to_integer(self) -> UniformIntegerHyperparameter:
        return UniformIntegerHyperparameter(
            name=self.name,
            lower=math.ceil(self.lower),
            upper=math.floor(self.upper),
            default_value=round(self.default_value),
            log=self.log,
            meta=None,
        )

    def __str__(self) -> str:
        parts = [
            self.name,
            f"Type: {str(self.__class__.__name__).replace('Hyperparameter', '')}",
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
            "lower": float(self.lower),
            "upper": float(self.upper),
            "default_value": float(self.default_value),
        }
