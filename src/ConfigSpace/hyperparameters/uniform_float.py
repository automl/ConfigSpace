from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ConfigSpace.hyperparameters.distributions import UnitUniformContinuousDistribution
from ConfigSpace.hyperparameters.hp_components import ROUND_PLACES, UnitScaler
from ConfigSpace.hyperparameters.hyperparameter import FloatHyperparameter
from ConfigSpace.types import f64

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter
    from ConfigSpace.types import Number


@dataclass(init=False)
class UniformFloatHyperparameter(FloatHyperparameter):
    """A uniformly distributed float hyperparameter.

    The 'lower' and 'upper' parameters define the range of values from which the
    hyperparameter represents. The 'log' parameter defines whether the values of the
    hyperparameter will be sampled on a log-scale.

    Its values are sampled from a uniform distribution `U(lower, upper)`.

    ```python exec="True" result="python" source="material-block"
    from ConfigSpace import UniformFloatHyperparameter

    u = UniformFloatHyperparameter('u', lower=11.3, upper=12.5, log=False)
    print(u)
    ```
    """

    ORDERABLE: ClassVar[bool] = True

    lower: float
    """Lower bound of a range of values from which the hyperparameter represents."""

    upper: float
    """Upper bound of a range of values from which the hyperparameter represents."""

    log: bool
    """If `True` the values of the hyperparameter will be sampled on a log-scale."""

    name: str
    """Name of the hyperparameter, with which it can be accessed."""

    default_value: float
    """The default value of this hyperparameter.

    If not specified, the default value is the midpoint of the range.
    """

    meta: Mapping[Hashable, Any] | None
    """Field for holding meta data provided by the user. Not used by the ConfigSpace."""

    size: float = field(init=False)
    """Size of hyperparameter. It is set to np.inf for continuous hyperparameters."""

    def __init__(
        self,
        name: str,
        lower: Number,
        upper: Number,
        default_value: Number | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """Initialize a uniformly distributed float hyperparameter.

        Args:
            name:
                Name of the hyperparameter, with which it can be accessed
            lower:
                Lower bound of a range of values from which the hyperparameter
                represents
            upper:
                Upper bound of a range of values from which the hyperparameter
                represents
            default_value:
                The default value of this hyperparameter. If not specified, the
                default value is the midpoint of the range
            log:
                If `True` the values of the hyperparameter will be sampled on a log-scale
            meta:
                Field for holding meta data provided by the user. Not used by
                ConfigSpace.
        """
        self.lower = float(np.round(lower, ROUND_PLACES))
        self.upper = float(np.round(upper, ROUND_PLACES))
        self.log = log

        try:
            scaler = UnitScaler(f64(self.lower), f64(self.upper), log=log, dtype=f64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        vect_dist = UnitUniformContinuousDistribution(
            pdf_max_density=1 / float(self.upper - self.lower),
        )
        super().__init__(
            name=name,
            size=np.inf,
            default_value=float(
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
            value_cast=float,
        )

    def to_integer(self) -> UniformIntegerHyperparameter:
        """Converts the hyperparameter to a uniformly integer hyperparameter.

        This is done by rounding the lower and upper bounds of the float hyperparameter
        and the default value.

        Returns:
            A uniformly distributed integer hyperparameter.
        """
        from ConfigSpace.hyperparameters.uniform_integer import (
            UniformIntegerHyperparameter,
        )

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
