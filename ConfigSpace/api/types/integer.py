from __future__ import annotations

from typing import overload

from ConfigSpace.api.distributions import Beta, Distribution, Normal, Uniform
from ConfigSpace.hyperparameters import (
    BetaIntegerHyperparameter,
    NormalIntegerHyperparameter,
    UniformIntegerHyperparameter,
)


# Uniform | None -> UniformIntegerHyperparameter
@overload
def Integer(
    name: str,
    bounds: tuple[int, int],
    *,
    distribution: Uniform | None = ...,
    default: int | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
) -> UniformIntegerHyperparameter: ...


# Normal -> NormalIntegerHyperparameter
@overload
def Integer(
    name: str,
    bounds: tuple[int, int],
    *,
    distribution: Normal,
    default: int | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
) -> NormalIntegerHyperparameter: ...


# Beta -> BetaIntegerHyperparameter
@overload
def Integer(
    name: str,
    bounds: tuple[int, int],
    *,
    distribution: Beta,
    default: int | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
) -> BetaIntegerHyperparameter: ...


def Integer(
    name: str,
    bounds: tuple[int, int],
    *,
    distribution: Distribution | None = None,
    default: int | None = None,
    log: bool = False,
    meta: dict | None = None,
) -> (
    UniformIntegerHyperparameter
    | NormalIntegerHyperparameter
    | BetaIntegerHyperparameter
):
    """Create an IntegerHyperparameter.

    .. code:: python

        # Uniformly distributed
        Integer("a", (1, 10))
        Integer("a", (1, 10), distribution=Uniform())

        # Normally distributed at 2 with std 3
        Integer("b", (0, 5), distribution=Normal(2, 3))  # ... bounded

        # Beta distributed with alpha 1 and beta 2
        Integer("c", (0, 3), distribution=Beta(1, 2))  # ... bounded

        # Give it a default value
        Integer("a", (1, 10), default=4)

        # Sample on a log scale
        Integer("a", (1, 100), log=True)

        # Add meta info to the param
        Integer("a", (1, 10), meta={"use": "For counting chickens"})

    Note:
    ----
    `Integer` is actually a function, please use the corresponding return types if
    doing an `isinstance(param, type)` check and not `Integer`.

    Parameters
    ----------
    name : str
        The name to give to this hyperparameter

    bounds : tuple[int, int]
        The bounds to give to the integer.

    distribution : Uniform | Normal | Beta, = Uniform
        The distribution to use for the hyperparameter. See above

    default : int | None = None
        The default value to give to the hyperparameter.

    log : bool = False
        Whether to this parameter lives on a log scale

    meta : dict | None = None
        Any meta information you want to associate with this parameter

    Returns
    -------
    UniformIntegerHyperparameter | NormalIntegerHyperparameter | BetaIntegerHyperparameter
        Returns the corresponding hyperparameter type
    """
    if distribution is None:
        distribution = Uniform()

    if bounds is None and isinstance(distribution, Uniform):
        raise ValueError("`bounds` must be specifed for Uniform distribution")

    if bounds is None:
        lower, upper = (None, None)
    else:
        lower, upper = bounds

    if isinstance(distribution, Uniform):
        return UniformIntegerHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            log=log,
            default_value=default,
            meta=meta,
        )

    if isinstance(distribution, Normal):
        return NormalIntegerHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            log=log,
            default_value=default,
            meta=meta,
            mu=distribution.mu,
            sigma=distribution.sigma,
        )

    if isinstance(distribution, Beta):
        return BetaIntegerHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            log=log,
            default_value=default,
            meta=meta,
            alpha=distribution.alpha,
            beta=distribution.beta,
        )

    raise ValueError(f"Unknown distribution type {type(distribution)}")
