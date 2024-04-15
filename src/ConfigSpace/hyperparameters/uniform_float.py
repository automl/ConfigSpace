from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ConfigSpace.hyperparameters._distributions import UnitUniformContinuousDistribution
from ConfigSpace.hyperparameters._hp_components import ROUND_PLACES, UnitScaler
from ConfigSpace.hyperparameters.hyperparameter import FloatHyperparameter
from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter
from ConfigSpace.types import f64

if TYPE_CHECKING:
    from ConfigSpace.types import Number


@dataclass(init=False)
class UniformFloatHyperparameter(FloatHyperparameter):
    ORDERABLE: ClassVar[bool] = True

    lower: f64
    upper: f64
    log: bool

    name: str
    default_value: f64
    meta: Mapping[Hashable, Any] | None

    def __init__(
        self,
        name: str,
        lower: Number,
        upper: Number,
        default_value: Number | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        self.lower = f64(lower)
        self.upper = f64(upper)
        self.log = log

        try:
            scaler = UnitScaler(self.lower, self.upper, log=log, dtype=f64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        vect_dist = UnitUniformContinuousDistribution(
            pdf_max_density=1 / float(self.upper - self.lower),
        )
        super().__init__(
            name=name,
            size=np.inf,
            default_value=f64(
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
