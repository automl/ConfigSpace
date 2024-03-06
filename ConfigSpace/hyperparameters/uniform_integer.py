from __future__ import annotations

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

from ConfigSpace.hyperparameters._distributions import (
    DiscretizedContinuousScipyDistribution,
)
from ConfigSpace.hyperparameters._hp_components import (
    VECTORIZED_NUMERIC_LOWER,
    VECTORIZED_NUMERIC_UPPER,
    UnitScaler,
)
from ConfigSpace.hyperparameters.integer_hyperparameter import IntegerHyperparameter


@dataclass(init=False)
class UniformIntegerHyperparameter(IntegerHyperparameter):
    orderable: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        lower: int | float | np.number,
        upper: int | float | np.number,
        default_value: int | np.integer | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        self.lower = np.int64(np.rint(lower))
        self.upper = np.int64(np.rint(upper))
        self.log = bool(log)

        try:
            scaler = UnitScaler(lower, upper, log=log)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            _default_value = np.int64(scaler.to_value(np.array([0.5]))[0])
        else:
            _default_value = np.int64(round(default_value))

        self.log = log
        size = self.upper - self.lower + 1
        vector_dist = DiscretizedContinuousScipyDistribution(
            dist=uniform(VECTORIZED_NUMERIC_LOWER, VECTORIZED_NUMERIC_UPPER),  # type: ignore
            steps=size,
            max_density_value=float(1 / size),
            normalization_constant_value=1,
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

    def neighborhood_size(self, value: np.int64 | None) -> int | float:
        if value is None or self.lower <= value <= self.upper:
            return self.size
        return self.size - 1
