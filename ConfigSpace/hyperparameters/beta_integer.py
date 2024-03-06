from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, ClassVar

import numpy as np
from scipy.stats import beta as spbeta

from ConfigSpace.hyperparameters._distributions import (
    DiscretizedContinuousScipyDistribution,
)
from ConfigSpace.hyperparameters._hp_components import UnitScaler
from ConfigSpace.hyperparameters.hyperparameter import (
    HyperparameterWithPrior,
)
from ConfigSpace.hyperparameters.integer_hyperparameter import IntegerHyperparameter
from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter


class BetaIntegerHyperparameter(
    IntegerHyperparameter,
    HyperparameterWithPrior[UniformIntegerHyperparameter],
):
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
            scaler = UnitScaler(self.lower, self.upper, log=log)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e

        beta_rv = spbeta(self.alpha, self.beta)
        if (self.alpha > 1) or (self.beta > 1):
            normalized_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)
        else:
            # If both alpha and beta are 1, we have a uniform distribution.
            normalized_mode = 0.5

        max_density_value = float(beta_rv.pdf(normalized_mode))  # type: ignore

        if default_value is None:
            _default_value = np.rint(
                scaler.to_value(np.array([normalized_mode]))[0],
            ).astype(np.int64)
        else:
            _default_value = np.rint(default_value).astype(np.int64)

        size = self.upper - self.lower + 1
        vector_dist = DiscretizedContinuousScipyDistribution(
            rv=spbeta(self.alpha, self.beta),  # type: ignore
            max_density_value=max_density_value,
            steps=size,
        )

        # Compute the normalization constant ahead of time
        constant = vector_dist._normalization_constant()
        vector_dist.normalization_constant_value = constant

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

    def to_uniform(self) -> UniformIntegerHyperparameter:
        return UniformIntegerHyperparameter(
            name=self.name,
            lower=self.lower,
            upper=self.upper,
            default_value=self.default_value,
            log=self.log,
            meta=None,
        )
