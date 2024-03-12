from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, ClassVar

import numpy as np
from scipy.stats import beta as spbeta

from ConfigSpace.functional import is_close_to_integer
from ConfigSpace.hyperparameters._distributions import (
    DiscretizedContinuousScipyDistribution,
)
from ConfigSpace.hyperparameters._hp_components import ABS_ROUND_CLOSENESS, UnitScaler
from ConfigSpace.hyperparameters.hyperparameter import (
    HyperparameterWithPrior,
)
from ConfigSpace.hyperparameters.integer_hyperparameter import IntegerHyperparameter
from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter


class BetaIntegerHyperparameter(
    IntegerHyperparameter,
    HyperparameterWithPrior[UniformIntegerHyperparameter],
):
    serializable_type_name: ClassVar[str] = "beta_int"
    orderable: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        alpha: int | float,
        beta: int | float,
        lower: int | float,
        upper: int | float,
        default_value: int | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        r"""A beta distributed integer hyperparameter. The 'lower' and 'upper' parameters
        move the distribution from the [0, 1]-range and scale it appropriately, but the
        shape of the distribution is preserved as if it were in [0, 1]-range.

        Its values are sampled from a beta distribution
        :math:`Beta(\alpha, \beta)`.

        >>> from ConfigSpace import BetaIntegerHyperparameter
        >>>
        >>> BetaIntegerHyperparameter('b', alpha=3, beta=2, lower=1, upper=4, log=False)
        b, Type: BetaInteger, Alpha: 3.0 Beta: 2.0, Range: [1, 4], Default: 3


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
        self.lower = np.int64(lower)
        self.upper = np.int64(upper)
        self.log = bool(log)

        try:
            scaler = UnitScaler(self.lower, self.upper, log=log, dtype=np.int64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        spbeta(self.alpha, self.beta)
        if (self.alpha > 1) or (self.beta > 1):
            vectorized_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)
        else:
            # If both alpha and beta are 1, we have a uniform distribution.
            vectorized_mode = 0.5

        if default_value is None:
            _default_value = np.rint(
                scaler.to_value(np.array([vectorized_mode]))[0],
            ).astype(np.int64)
        else:
            if not is_close_to_integer(default_value, atol=ABS_ROUND_CLOSENESS):
                raise TypeError(
                    f"`default_value` for hyperparameter '{name}' must be an integer."
                    f" Got '{type(default_value).__name__}' for {default_value=}.",
                )

            _default_value = np.rint(default_value).astype(np.int64)

        size = int(self.upper - self.lower + 1)
        vector_dist = DiscretizedContinuousScipyDistribution(
            rv=spbeta(self.alpha, self.beta),  # type: ignore
            steps=size,
            lower_vectorized=np.float64(0.0),
            upper_vectorized=np.float64(1.0),
        )

        super().__init__(
            name=name,
            size=size,
            default_value=_default_value,
            meta=meta,
            transformer=scaler,
            vector_dist=vector_dist,
            neighborhood=vector_dist.neighborhood,
            neighborhood_size=self._neighborhood_size,
        )

    def to_uniform(self) -> UniformIntegerHyperparameter:
        return UniformIntegerHyperparameter(
            name=self.name,
            lower=self.lower,
            upper=self.upper,
            default_value=self.default_value,
            log=self.log,
            meta=None,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.serializable_type_name,
            "log": self.log,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "lower": int(self.lower),
            "upper": int(self.upper),
            "default_value": int(self.default_value),
        }
