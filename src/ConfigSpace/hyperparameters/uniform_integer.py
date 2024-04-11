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


@dataclass(init=False)
class UniformIntegerHyperparameter(IntegerHyperparameter):
    serializable_type_name: ClassVar[str] = "uniform_int"
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

        if default_value is not None and not is_close_to_integer(
            default_value,
            atol=ATOL,
        ):
            raise TypeError(
                f"`default_value` for hyperparameter '{name}' must be an integer."
                f" Got '{type(default_value).__name__}' for {default_value=}.",
            )

        try:
            scaler = UnitScaler(self.lower, self.upper, log=log, dtype=np.int64)
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
                lower_vectorized=np.float64(0.0),
                upper_vectorized=np.float64(1.0),
                log_scale=log,
                transformer=scaler,
            )

        super().__init__(
            name=name,
            size=int(size),
            default_value=(
                np.int64(
                    default_value
                    if default_value is not None
                    else np.rint(scaler.to_value(np.array([0.5]))[0]),
                )
            ),
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
            "lower": int(self.lower),
            "upper": int(self.upper),
            "default_value": int(self.default_value),
        }
