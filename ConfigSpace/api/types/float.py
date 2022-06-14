"""Interface to Float based hyperparameters

Most interactions occur through the Float function and constructs the Cpython types under the hood
"""
from __future__ import annotations

from typing import Any, overload

from ConfigSpace.api.distributions import Beta, Distribution, Normal, Uniform
from ConfigSpace.hyperparameters import (BetaFloatHyperparameter,
                                         NormalFloatHyperparameter,
                                         UniformFloatHyperparameter)


# Uniform | None -> UniformFloatHyperparameter
@overload
def Float(
    name: str,
    bounds: tuple[float, float] | None = ...,
    *,
    distribution: Uniform | None = ...,
    default: float | None = ...,
    q: int | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
    **kwargs: Any,
) -> UniformFloatHyperparameter:
    ...


# Normal -> NormalFloatHyperparameter
@overload
def Float(
    name: str,
    bounds: tuple[float, float] | None = ...,
    *,
    distribution: Normal,
    default: float | None = ...,
    q: int | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
    **kwargs: Any,
) -> NormalFloatHyperparameter:
    ...


# Beta -> BetaFloatHyperparameter
@overload
def Float(
    name: str,
    bounds: tuple[float, float] | None = ...,
    *,
    distribution: Beta,
    default: float | None = ...,
    q: int | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
    **kwargs: Any,
) -> BetaFloatHyperparameter:
    ...


def Float(
    name: str,
    bounds: tuple[float, float] | None = None,
    *,
    distribution: Distribution | None = None,
    default: float | None = None,
    q: int | None = None,
    log: bool = False,
    meta: dict | None = None,
    **kwargs: Any,
) -> UniformFloatHyperparameter | NormalFloatHyperparameter | BetaFloatHyperparameter:
    if distribution is None:
        distribution = Uniform()

    if bounds is None and isinstance(distribution, Uniform):
        raise ValueError("`bounds` must be specifed for Uniform distribution")

    if bounds is None:
        lower, upper = (None, None)
    else:
        lower, upper = bounds

    # NOTE: not very pretty but ensures we don't accidentally merge arguments in definition
    # and the **kwargs
    if isinstance(distribution, Uniform):
        return UniformFloatHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            default_value=default,
            q=q,
            log=log,
            meta=meta,
            **kwargs,
        )
    elif isinstance(distribution, Normal):
        return NormalFloatHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            default_value=default,
            mu=distribution.mu,
            sigma=distribution.sigma,
            q=q,
            log=log,
            meta=meta,
            **kwargs,
        )
    elif isinstance(distribution, Beta):
        return BetaFloatHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            alpha=distribution.alpha,
            beta=distribution.beta,
            default_value=default,
            q=q,
            log=log,
            meta=meta,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown distribution type {type(distribution)}")
