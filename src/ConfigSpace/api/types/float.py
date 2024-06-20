from __future__ import annotations

from typing import overload

from ConfigSpace.api.distributions import Beta, Distribution, Normal, Uniform
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    NormalFloatHyperparameter,
    UniformFloatHyperparameter,
)


# Uniform | None -> UniformFloatHyperparameter
@overload
def Float(
    name: str,
    bounds: tuple[float, float],
    *,
    distribution: Uniform | None = ...,
    default: float | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
) -> UniformFloatHyperparameter: ...


# Normal -> NormalFloatHyperparameter
@overload
def Float(
    name: str,
    bounds: tuple[float, float],
    *,
    distribution: Normal,
    default: float | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
) -> NormalFloatHyperparameter: ...


# Beta -> BetaFloatHyperparameter
@overload
def Float(
    name: str,
    bounds: tuple[float, float],
    *,
    distribution: Beta,
    default: float | None = ...,
    log: bool = ...,
    meta: dict | None = ...,
) -> BetaFloatHyperparameter: ...


def Float(
    name: str,
    bounds: tuple[float, float],
    *,
    distribution: Distribution | None = None,
    default: float | None = None,
    log: bool = False,
    meta: dict | None = None,
) -> UniformFloatHyperparameter | NormalFloatHyperparameter | BetaFloatHyperparameter:
    """Create a FloatHyperparameter.

    ```python
    # Uniformly distributed
    Float("a", (1, 10))
    Float("a", (1, 10), distribution=Uniform())

    # Normally distributed at 2 with std 3
    Float("b", (0, 5), distribution=Normal(2, 3))

    # Beta distributed with alpha 1 and beta 2
    Float("c", (0, 3), distribution=Beta(1, 2))

    # Give it a default value
    Float("a", (1, 10), default=4.3)

    # Sample on a log scale
    Float("a", (1, 100), log=True)

    # Add meta info to the param
    Float("a", (1.0, 10), meta={"use": "For counting chickens"})
    ```

    !!! note

        `Float` is actually a function, please use the corresponding return types if
        doing an `isinstance(param, type)` check and not `Float`.

    Args:
        name:
            The name to give to this hyperparameter

        bounds:
            The bounds to give to the float.

        distribution:
            The distribution to use for the hyperparameter. See above

        default:
            The default value to give to the hyperparameter.

        log:
            Whether to this parameter lives on a log scale

        meta:
            Any meta information you want to associate with this parameter

    Returns:
        Returns the corresponding hyperparameter type
    """
    if distribution is None:
        distribution = Uniform()

    lower, upper = bounds

    if isinstance(distribution, Uniform):
        return UniformFloatHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            default_value=default,
            log=log,
            meta=meta,
        )

    if isinstance(distribution, Normal):
        return NormalFloatHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            default_value=default,
            mu=distribution.mu,
            sigma=distribution.sigma,
            log=log,
            meta=meta,
        )

    if isinstance(distribution, Beta):
        return BetaFloatHyperparameter(
            name=name,
            lower=lower,
            upper=upper,
            alpha=distribution.alpha,
            beta=distribution.beta,
            default_value=default,
            log=log,
            meta=meta,
        )

    raise ValueError(f"Unknown distribution type {type(distribution)}")
