"""Api wrapper around Categorical type hyperparameters."""
from __future__ import annotations

from typing import Any, Sequence, Union, overload

from typing_extensions import (Literal,  # Move to `typing` when 3.8 minimum
                               TypeAlias)

from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         OrdinalHyperparameter)

# We only accept these types in `items`
T: TypeAlias = Union[str, int, float]


# ordered False -> CategoricalHyperparameter
@overload
def Categorical(
    name: str,
    items: Sequence[T],
    *,
    default: T | None = None,
    weights: Sequence[float] | None = None,
    ordered: Literal[False],
    meta: dict | None = None,
    **kwargs: Any,
) -> CategoricalHyperparameter:
    ...


# ordered True -> OrdinalHyperparameter
@overload
def Categorical(
    name: str,
    items: Sequence[T],
    *,
    default: T | None = None,
    weights: Sequence[float] | None = None,
    ordered: Literal[True],
    meta: dict | None = None,
    **kwargs: Any,
) -> OrdinalHyperparameter:
    ...


# ordered bool (unknown) -> Either
@overload
def Categorical(
    name: str,
    items: Sequence[T],
    *,
    default: T | None = None,
    weights: Sequence[float] | None = None,
    ordered: bool = ...,
    meta: dict | None = None,
    **kwargs: Any,
) -> CategoricalHyperparameter | OrdinalHyperparameter:
    ...


def Categorical(
    name: str,
    items: Sequence[T],
    *,
    default: T | None = None,
    weights: Sequence[float] | None = None,
    ordered: bool = False,
    meta: dict | None = None,
    **kwargs: Any,
) -> CategoricalHyperparameter | OrdinalHyperparameter:
    """TODO"""
    if ordered and weights is not None:
        raise ValueError("Can't apply `weights` to `ordered` Categorical")

    if ordered:
        return OrdinalHyperparameter(
            name=name,
            sequence=items,
            default_value=default,
            meta=meta,
            **kwargs,
        )
    else:
        return CategoricalHyperparameter(
            name=name,
            choices=items,
            default_value=default,
            weights=weights,
            meta=meta,
            **kwargs,
        )
