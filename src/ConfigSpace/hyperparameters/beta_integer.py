from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.stats import beta as spbeta

from ConfigSpace.functional import is_close_to_integer_single
from ConfigSpace.hyperparameters.distributions import (
    DiscretizedContinuousScipyDistribution,
)
from ConfigSpace.hyperparameters.hp_components import ATOL, UnitScaler
from ConfigSpace.hyperparameters.integer_hyperparameter import IntegerHyperparameter
from ConfigSpace.types import f64, i64

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.beta_float import BetaFloatHyperparameter
    from ConfigSpace.types import Number


@dataclass(init=False)
class BetaIntegerHyperparameter(IntegerHyperparameter):
    r"""A beta distributed integer hyperparameter. The 'lower' and 'upper'
    parameters move the distribution from the `[0, 1]`-range and scale it
    appropriately, but the shape of the distribution is preserved as if it were in
    `[0, 1]`-range.

    Its values are sampled from a beta distribution `Beta(alpha, beta)`.

    ```python exec="True" result="python" source="material-block"
    from ConfigSpace import BetaIntegerHyperparameter

    b = BetaIntegerHyperparameter('b', alpha=3, beta=2, lower=1, upper=4, log=False)
    print(b)
    ```
    """

    ORDERABLE: ClassVar[bool] = True

    alpha: float
    """Alpha parameter of the normalized beta distribution."""

    beta: float
    """Beta parameter of the normalized beta distribution."""

    lower: int
    """Lower bound of a range of values from which the hyperparameter represents."""

    upper: int
    """Upper bound of a range of values from which the hyperparameter represents."""

    log: bool
    """If `True` the values of the hyperparameter will be sampled on a log-scale."""

    name: str
    """Name of the hyperparameter, with which it can be accessed."""

    default_value: int
    """The default value of this hyperparameter."""

    meta: Mapping[Hashable, Any] | None
    """Field for holding meta data provided by the user. Not used by ConfigSpace."""

    size: int
    """Size of the hyperparameter. This is the number of possible values the
    hyperparameter can take on within the specified range."""

    def __init__(
        self,
        name: str,
        alpha: Number,
        beta: Number,
        lower: Number,
        upper: Number,
        default_value: Number | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """Initialize a beta distributed integer hyperparameter.

        Args:
            name:
                Name of the hyperparameter with which it can be accessed.
            alpha:
                Alpha parameter of the distribution, from which hyperparameter is
                sampled.
            beta:
                Beta parameter of the distribution, from which
                hyperparameter is sampled.
            lower:
                Lower bound of a range of values from which the hyperparameter will be
                sampled.
            upper:
                Upper bound of a range of values from which the hyperparameter will be
                sampled.
            default_value:
                Sets the default value of a hyperparameter to a given value.
            log:
                If `True` the values of the hyperparameter will be sampled
                on a logarithmic scale. Defaults to `False`.
            meta:
                Field for holding meta data provided by the user.
                Not used by the configuration space.
        """
        if (alpha < 1) or (beta < 1):
            raise ValueError(
                "Please provide values of alpha and beta larger than or equal to"
                "1 so that the probability density is finite.",
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lower = int(np.rint(lower))
        self.upper = int(np.rint(upper))
        self.log = bool(log)

        try:
            scaler = UnitScaler(i64(self.lower), i64(self.upper), log=log, dtype=i64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if default_value is None:
            if (self.alpha > 1) or (self.beta > 1):
                vectorized_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)
            else:
                # If both alpha and beta are 1, we have a uniform distribution.
                vectorized_mode = 0.5

            _default_value = np.rint(
                scaler.to_value(np.array([vectorized_mode]))[0],
            ).astype(i64)
        else:
            if not is_close_to_integer_single(default_value, atol=ATOL):
                raise TypeError(
                    f"`default_value` for hyperparameter '{name}' must be an integer."
                    f" Got '{type(default_value).__name__}' for {default_value=}.",
                )

            _default_value = np.rint(default_value).astype(i64)

        size = int(self.upper - self.lower + 1)
        vector_dist = DiscretizedContinuousScipyDistribution(
            rv=spbeta(self.alpha, self.beta),  # type: ignore
            steps=size,
            lower_vectorized=f64(0.0),
            upper_vectorized=f64(1.0),
        )

        super().__init__(
            name=name,
            size=size,
            default_value=_default_value,
            meta=meta,
            transformer=scaler,
            vector_dist=vector_dist,
            neighborhood=vector_dist.neighborhood,
            neighborhood_size=self._integer_neighborhood_size,
            value_cast=int,
        )

    def to_float(self) -> BetaFloatHyperparameter:
        """Converts the beta integer hyperparameter to a beta float hyperparameter.

        !!! warning

            `meta` information is not transferred to the new hyperparameter and
            must be transferred manually.

        Returns:
            A beta integer hyperparameter.
        """
        from ConfigSpace.hyperparameters.beta_float import BetaFloatHyperparameter

        return BetaFloatHyperparameter(
            name=self.name,
            lower=self.lower,
            upper=self.upper,
            default_value=self.default_value,
            log=self.log,
            meta=None,
            alpha=self.alpha,
            beta=self.beta,
        )

    def __str__(self) -> str:
        parts = [
            self.name,
            f"Type: {str(self.__class__.__name__).replace('Hyperparameter', '')}",
            f"Alpha: {self.alpha}",
            f"Beta: {self.beta}",
            f"Range: [{self.lower}, {self.upper}]",
            f"Default: {self.default_value}",
        ]
        if self.log:
            parts.append("on log-scale")

        return ", ".join(parts)
