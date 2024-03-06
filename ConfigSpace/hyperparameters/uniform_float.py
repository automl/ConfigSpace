from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
)

import numpy as np
from scipy.stats import (
    uniform,
)

from ConfigSpace.hyperparameters._distributions import ScipyContinuousDistribution
from ConfigSpace.hyperparameters._hp_components import ROUND_PLACES, UnitScaler
from ConfigSpace.hyperparameters.float_hyperparameter import FloatHyperparameter
from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter


@dataclass(init=False)
class UniformFloatHyperparameter(FloatHyperparameter):
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
            scaler = UnitScaler(self.lower, self.upper, log=log)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            _default_value = np.float64(scaler.to_value(np.array([0.5]))[0])
        else:
            _default_value = np.float64(round(default_value, ROUND_PLACES))

        vect_dist = ScipyContinuousDistribution(
            rv=uniform(a=0, b=1),  # type: ignore
            max_density_value=float(1 / (self.upper - self.lower)),
            dtype=np.float64,
        )
        super().__init__(
            name=name,
            size=np.inf,
            default_value=_default_value,
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
