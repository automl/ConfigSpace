from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.stats import beta as spbeta

from ConfigSpace.hyperparameters._distributions import (
    ScipyContinuousDistribution,
)
from ConfigSpace.hyperparameters._hp_components import ROUND_PLACES, UnitScaler
from ConfigSpace.hyperparameters.beta_integer import BetaIntegerHyperparameter
from ConfigSpace.hyperparameters.float_hyperparameter import FloatHyperparameter
from ConfigSpace.types import f64

if TYPE_CHECKING:
    from ConfigSpace.types import Number


@dataclass(init=False)
class BetaFloatHyperparameter(FloatHyperparameter):
    ORDERABLE: ClassVar[bool] = True

    alpha: f64
    beta: f64
    lower: f64
    upper: f64
    log: bool

    name: str
    default_value: f64
    meta: Mapping[Hashable, Any] | None

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
        r"""A beta distributed float hyperparameter. The 'lower' and 'upper'
        parameters move the distribution from the [0, 1]-range and scale
        it appropriately, but the shape of the distribution is preserved as
        if it were in [0, 1]-range.

        Its values are sampled from a beta distribution
        :math:`Beta(\alpha, \beta)`.

        >>> from ConfigSpace import BetaFloatHyperparameter
        >>>
        >>> BetaFloatHyperparameter('b', alpha=3, beta=2, lower=1, upper=4, log=False)
        b, Type: BetaFloat, Alpha: 3.0 Beta: 2.0, Range: [1.0, 4.0], Default: 3.0

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

        self.alpha = f64(alpha)
        self.beta = f64(beta)
        self.lower = f64(lower)
        self.upper = f64(upper)
        self.log = bool(log)

        try:
            scaler = UnitScaler(self.lower, self.upper, log=log, dtype=f64)
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
            default_value=f64(np.round(_default_value, ROUND_PLACES)),
            meta=meta,
            transformer=scaler,
            vector_dist=vector_dist,
            neighborhood=vector_dist.neighborhood,
            neighborhood_size=np.inf,
        )

    def to_integer(self) -> BetaIntegerHyperparameter:
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
