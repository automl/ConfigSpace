from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.stats import uniform

from ConfigSpace.functional import is_close_to_integer
from ConfigSpace.hyperparameters.distributions import (
    DiscretizedContinuousScipyDistribution,
    UniformIntegerNormalizedDistribution,
)
from ConfigSpace.hyperparameters.hp_components import ATOL, UnitScaler
from ConfigSpace.hyperparameters.integer_hyperparameter import IntegerHyperparameter
from ConfigSpace.types import Number, f64, i64

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter


@dataclass(init=False)
class UniformIntegerHyperparameter(IntegerHyperparameter):
    """A uniformly distributed integer hyperparameter.

    The 'lower' and 'upper' parameters define the range of values from which the
    hyperparameter represents. The 'log' parameter defines whether the values of the
    hyperparameter will be sampled on a log-scale.

    Its values are sampled from a uniform distribution `U(lower, upper)`.

    ```python exec="True" result="python" source="material-block"
    from ConfigSpace import UniformIntegerHyperparameter

    u = UniformIntegerHyperparameter('u', lower=0, upper=10, log=False)
    print(u)
    ```
    """

    ORDERABLE: ClassVar[bool] = True

    lower: int
    """Lower bound of a range of values from which the hyperparameter represents."""

    upper: int
    """Upper bound of a range of values from which the hyperparameter represents."""

    log: bool
    """If `True` the values of the hyperparameter will be sampled on a log-scale."""

    name: str
    """Name of the hyperparameter, with which it can be accessed."""

    default_value: int
    """The default value of this hyperparameter.

    If not specified, the default value is the midpoint of the range.
    """

    meta: Mapping[Hashable, Any] | None
    """Field for holding meta data provided by the user. Not used by the ConfigSpace."""

    size: int = field(init=False)
    """The size of the hyperparameter's domain."""

    def __init__(
        self,
        name: str,
        lower: Number,
        upper: Number,
        default_value: Number | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """Initialize a uniformly distributed integer hyperparameter.

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
                default value is the midpoint of the range.
            log:
                If `True` the values of the hyperparameter will be sampled on a
                log-scale
            meta:
                Field for holding meta data provided by the user. Not used by
                ConfigSpace.
        """
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

    def to_float(self) -> UniformFloatHyperparameter:
        """Convert this hyperparameter to a uniform float hyperparameter.

        Returns:
            A uniform float hyperparameter with the same range as this.
        """
        from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter

        return UniformFloatHyperparameter(
            name=self.name,
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
            f"Range: [{self.lower}, {self.upper}]",
            f"Default: {self.default_value}",
        ]
        if self.log:
            parts.append("on log-scale")

        return ", ".join(parts)
