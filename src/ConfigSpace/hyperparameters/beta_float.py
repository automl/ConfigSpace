from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.stats import beta as spbeta

from ConfigSpace.hyperparameters.distributions import (
    ScipyContinuousDistribution,
)
from ConfigSpace.hyperparameters.float_hyperparameter import FloatHyperparameter
from ConfigSpace.hyperparameters.hp_components import ROUND_PLACES, UnitScaler
from ConfigSpace.types import f64

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.beta_integer import BetaIntegerHyperparameter
    from ConfigSpace.types import Number


@dataclass(init=False)
class BetaFloatHyperparameter(FloatHyperparameter):
    """A beta distributed float hyperparameter. The 'lower' and 'upper'
    parameters move the distribution from the `[0, 1]`-range and scale
    it appropriately, but the shape of the distribution is preserved as
    if it were in `[0, 1]`-range.

    Its values are sampled from a beta distribution `Beta(alpha, beta)`.

    ```python exec="True" result="python" source="material-block"
    from ConfigSpace import BetaFloatHyperparameter

    b = BetaFloatHyperparameter('b', alpha=3, beta=2, lower=1, upper=4, log=False)
    print(b)
    ```
    """

    ORDERABLE: ClassVar[bool] = True

    alpha: float
    """Alpha parameter of the normalized beta distribution."""

    beta: float
    """Beta parameter of the normalized beta distribution."""

    lower: float
    """Lower bound of a range of values from which the hyperparameter represents."""

    upper: float
    """Upper bound of a range of values from which the hyperparameter represents."""

    log: bool
    """If `True` the values of the hyperparameter will be sampled on a log-scale."""

    name: str
    """Name of the hyperparameter, with which it can be accessed."""

    default_value: float
    """The default value of this hyperparameter."""

    meta: Mapping[Hashable, Any] | None
    """Field for holding meta data provided by the user. Not used by ConfigSpace."""

    size: float = field(init=False)
    """Size of the hyperparameter. Always `np.inf` for continuous hyperparameters."""

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
        """Initialize a beta distributed float hyperparameter.

        Args:
            name:
                Name of the hyperparameter, with which it can be accessed
            alpha:
                Alpha parameter of the normalized beta distribution
            beta:
                Beta parameter of the normalized beta distribution
            lower:
                Lower bound of a range of values from which the hyperparameter will be
                sampled. The Beta disribution gets scaled by the total range of the
                hyperparameter.
            upper:
                Upper bound of a range of values from which the hyperparameter will be
                sampled. The Beta disribution gets scaled by the total range of the
                hyperparameter.
            default_value:
                Sets the default value of a hyperparameter to a given value
            log:
                If `True` the values of the hyperparameter will be sampled
                on a logarithmic scale. Default to `False`
            meta:
                Field for holding meta data provided by the user.
                Not used by the configuration space.
        """
        if (alpha < 1) or (beta < 1):
            raise ValueError(
                "Please provide values of alpha and beta larger than or equal to"
                " 1 so that the probability density is finite.",
            )

        self.alpha = float(np.round(alpha, ROUND_PLACES))
        self.beta = float(np.round(beta, ROUND_PLACES))
        self.lower = float(np.round(lower, ROUND_PLACES))
        self.upper = float(np.round(upper, ROUND_PLACES))
        self.log = bool(log)

        try:
            scaler = UnitScaler(f64(self.lower), f64(self.upper), log=log, dtype=f64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        if (self.alpha > 1) or (self.beta > 1):
            normalized_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)
        else:
            # If both alpha and beta are 1, we have a uniform distribution.
            normalized_mode = 0.5

        if default_value is None:
            _default_value = scaler.to_value(np.array([normalized_mode]))[0]
        else:
            _default_value = default_value

        # This nicely behaves in 0, 1 range
        beta_rv = spbeta(self.alpha, self.beta)
        vector_dist = ScipyContinuousDistribution(
            rv=beta_rv,  # type: ignore
            lower_vectorized=f64(0.0),
            upper_vectorized=f64(1.0),
            _max_density=beta_rv.pdf(normalized_mode),  # type: ignore
        )

        super().__init__(
            name=name,
            size=np.inf,
            default_value=float(np.round(_default_value, ROUND_PLACES)),
            meta=meta,
            transformer=scaler,
            vector_dist=vector_dist,
            neighborhood=vector_dist.neighborhood,
            neighborhood_size=np.inf,
            value_cast=float,
        )

    def to_integer(self) -> BetaIntegerHyperparameter:
        """Converts the beta float hyperparameter to a beta integer hyperparameter.

        This happens by rounding the lower and upper bounds and the default value
        as required.

        !!! warning

            `meta` information is not transferred to the new hyperparameter and
            must be transferred manually.

        Returns:
            A beta integer hyperparameter.
        """
        from ConfigSpace.hyperparameters.beta_integer import BetaIntegerHyperparameter

        lower = int(np.ceil(self.lower))
        upper = int(np.floor(self.upper))
        default_value = int(np.rint(self.default_value))

        return BetaIntegerHyperparameter(
            name=self.name,
            lower=lower,
            upper=upper,
            default_value=default_value,
            log=self.log,
            meta=None,
            alpha=float(self.alpha),
            beta=float(self.beta),
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
