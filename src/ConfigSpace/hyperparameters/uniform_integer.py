from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.stats import uniform

from ConfigSpace.functional import is_close_to_integer
from ConfigSpace.hyperparameters._distributions import (
    DiscretizedContinuousScipyDistribution,
    UniformIntegerNormalizedDistribution,
)
from ConfigSpace.hyperparameters._hp_components import ATOL, UnitScaler
from ConfigSpace.hyperparameters.integer_hyperparameter import IntegerHyperparameter
from ConfigSpace.types import Number, f64, i64


@dataclass(init=False)
class UniformIntegerHyperparameter(IntegerHyperparameter):
    ORDERABLE: ClassVar[bool] = True

    lower: int
    upper: int
    log: bool

    name: str
    default_value: int
    meta: Mapping[Hashable, Any] | None
    size: int

    def __init__(
        self,
        name: str,
        lower: Number,
        upper: Number,
        default_value: Number | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        self.lower = int(np.rint(lower))
        self.upper = int(np.rint(upper))
        self.log = bool(log)

        if default_value is not None and not is_close_to_integer(
            f64(default_value),
            atol=ATOL,
        ):
            raise TypeError(
                f"`default_value` for hyperparameter '{name}' must be an integer."
                f" Got '{type(default_value).__name__}' for {default_value=}.",
            )

        try:
            scaler = UnitScaler(i64(self.lower), i64(self.upper), log=log, dtype=i64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        size = self.upper - self.lower + 1
        if not self.log:
            vector_dist = UniformIntegerNormalizedDistribution(size=int(size))
        else:
            vector_dist = DiscretizedContinuousScipyDistribution(
                rv=uniform(),  # type: ignore
                steps=int(size),
                _max_density=float(1 / size),
                _pdf_norm=float(size),
                lower_vectorized=f64(0.0),
                upper_vectorized=f64(1.0),
                log_scale=log,
                transformer=scaler,
            )

        super().__init__(
            name=name,
            size=int(size),
            default_value=(
                int(
                    default_value
                    if default_value is not None
                    else np.rint(scaler.to_value(np.array([0.5]))[0]),
                )
            ),
            meta=meta,
            transformer=scaler,
            vector_dist=vector_dist,
            neighborhood=vector_dist.neighborhood,
            neighborhood_size=self._integer_neighborhood_size,
            value_cast=int,
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
