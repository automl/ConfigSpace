from __future__ import annotations

from typing import overload

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
) -> UniformFloatHyperparameter | NormalFloatHyperparameter | BetaFloatHyperparameter:
    """Create a FloatHyperparameter.

    .. code:: python

        # Uniformly distributed
        Float("a", (1, 10))
        Float("a", (1, 10), distribution=Uniform())

        # Normally distributed at 2 with std 3
        Float("b", distribution=Normal(2, 3))
        Float("b", (0, 5), distribution=Normal(2, 3))  # ... bounded

        # Beta distributed with alpha 1 and beta 2
        Float("c", distribution=Beta(1, 2))
        Float("c", (0, 3), distribution=Beta(1, 2))  # ... bounded

        # Give it a default value
        Float("a", (1, 10), default=4.3)

        # Sample on a log scale
        Float("a", (1, 100), log=True)

        # Quantized into three brackets
        Float("a", (1, 10), q=3)

        # Add meta info to the param
        Float("a", (1.0, 10), meta={"use": "For counting chickens"})

    Note
    ----
    `Float` is actually a function, please use the corresponding return types if
    doing an `isinstance(param, type)` check and not `Float`.

    Parameters
    ----------
    name : str
        The name to give to this hyperparameter

    bounds : tuple[float, float] | None = None
        The bounds to give to the float. Note that by default, this is required
        for Uniform distribution, which is the default distribution

    distribution : Uniform | Normal | Beta, = Uniform
        The distribution to use for the hyperparameter. See above

    default : float | None = None
        The default value to give to the hyperparameter.

    q : float | None = None
        The quantization factor, must evenly divide the boundaries.

        Note
        ----
        Quantization points act are not equal and require experimentation
        to be certain about

        * https://github.com/automl/ConfigSpace/issues/264

    log : bool = False
        Whether to this parameter lives on a log scale

    meta : dict | None = None
        Any meta information you want to associate with this parameter

    Returns
    -------
    UniformFloatHyperparameter | NormalFloatHyperparameter | BetaFloatHyperparameter
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
        return UniformFloatHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            default_value=default,
            q=q,
            log=log,
            meta=meta,
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
        )
    else:
        raise ValueError(f"Unknown distribution type {type(distribution)}")
