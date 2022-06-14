"""Interface to Int based hyperparameters.

Most interactions occur through the Int function and constructs the Cpython types under the hood
"""
from __future__ import annotations

from typing import Any, overload

from ConfigSpace.api.distributions import Beta, Distribution, Normal, Uniform
from ConfigSpace.hyperparameters import (BetaIntegerHyperparameter,
                                         NormalIntegerHyperparameter,
                                         UniformIntegerHyperparameter)


# Uniform | None -> UniformIntegerHyperparameter
@overload
def Int(
    name: str,
    bounds: tuple[int, int] | None = ...,
    *,
    distribution: Uniform | None = ...,
    default: int | None = ...,
    q: int | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
    **kwargs: Any,
) -> UniformIntegerHyperparameter:
    ...


# Normal
@overload
def Int(
    name: str,
    bounds: tuple[int, int] | None = ...,
    *,
    distribution: Normal,
    default: int | None = ...,
    q: int | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
    **kwargs: Any,
) -> NormalIntegerHyperparameter:
    ...


@overload
def Int(
    name: str,
    bounds: tuple[int, int] | None = ...,
    *,
    distribution: Beta,
    default: int | None = ...,
    q: int | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
    **kwargs: Any,
) -> BetaIntegerHyperparameter:
    ...


def Int(
    name: str,
    bounds: tuple[int, int] | None = None,
    *,
    distribution: Distribution | None = None,
    default: int | None = None,
    q: int | None = None,
    log: bool = False,
    meta: dict | None = None,
    **kwargs: Any,
) -> UniformIntegerHyperparameter | NormalIntegerHyperparameter | BetaIntegerHyperparameter:
    """TODO"""
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
        return UniformIntegerHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            q=q,
            log=log,
            default_value=default,
            meta=meta,
            **kwargs,
        )
    elif isinstance(distribution, Normal):
        return NormalIntegerHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            q=q,
            log=log,
            default_value=default,
            meta=meta,
            mu=distribution.mu,
            sigma=distribution.sigma,
            **kwargs,
        )
    elif isinstance(distribution, Beta):
        return BetaIntegerHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            q=q,
            log=log,
            default_value=default,
            meta=meta,
            alpha=distribution.alpha,
            beta=distribution.beta,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown distribution type {type(distribution)}")
