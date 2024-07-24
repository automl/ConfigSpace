from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeVar, overload

from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.types import NotSet, _NotSet

# We only accept these types in `items`
T = TypeVar("T")


# ordered False -> CategoricalHyperparameter
@overload
def Categorical(
    name: str,
    items: Sequence[T],
    *,
    default: T | _NotSet = NotSet,
    weights: Sequence[float] | None = None,
    ordered: Literal[False],
    meta: dict | None = None,
) -> CategoricalHyperparameter: ...


# ordered True -> OrdinalHyperparameter
@overload
def Categorical(
    name: str,
    items: Sequence[T],
    *,
    default: T | _NotSet = NotSet,
    weights: Sequence[float] | None = None,
    ordered: Literal[True],
    meta: dict | None = None,
) -> OrdinalHyperparameter: ...


# ordered bool (unknown) -> Either
@overload
def Categorical(
    name: str,
    items: Sequence[T],
    *,
    default: T | _NotSet = NotSet,
    weights: Sequence[float] | None = None,
    ordered: bool = ...,
    meta: dict | None = None,
) -> CategoricalHyperparameter | OrdinalHyperparameter: ...


def Categorical(
    name: str,
    items: Sequence[T],
    *,
    default: T | _NotSet = NotSet,
    weights: Sequence[float] | None = None,
    ordered: bool = False,
    meta: dict | None = None,
) -> CategoricalHyperparameter | OrdinalHyperparameter:
    """Creates a Categorical Hyperparameter.

    CategoricalHyperparameter's can be used to represent a discrete
    choice. Optionally, you can specify that these values are also ordered in
    some manner, e.g. `#!python ["small", "medium", "large"]`.

    ```python
    # A simple categorical hyperparameter
    c = Categorical("animals", ["cat", "dog", "mouse"])

    # With a default
    c = Categorical("animals", ["cat", "dog", "mouse"], default="mouse")

    # Make them weighted
    c = Categorical("animals", ["cat", "dog", "mouse"], weights=[0.1, 0.8, 3.14])

    # Specify it's an OrdinalHyperparameter (ordered categories)
    # ... note that you can't apply weights to an Ordinal
    o = Categorical("size", ["small", "medium", "large"], ordered=True)

    # Add some meta information for your own tracking
    c = Categorical("animals", ["cat", "dog", "mouse"], meta={"use": "Favourite Animal"})
    ```

    !!! note

        `Categorical` is actually a function, please use the corresponding return types if
        doing an `isinstance(param, type)` check with either
        [`CategoricalHyperparameter`][ConfigSpace.hyperparameters.CategoricalHyperparameter]
        and/or [`OrdinalHyperparameter`][ConfigSpace.hyperparameters.OrdinalHyperparameter].

    Args:
        name: The name of the hyperparameter
        items:
            A list of items to put in the category.

            !!! warning

                Can't have duplicate categories, use weights if required.

        default: The default value of the categorical hyperparameter.
        weights:
            The weights to apply to each categorical. Each item will be sampled according
            to these weights.
        ordered:
            Whether the categorical is ordered or not. If `True`, this will return an
            [`OrdinalHyperparameter`][ConfigSpace.hyperparameters.OrdinalHyperparameter],
            otherwise it remain a
            [`CategoricalHyperparameter`][ConfigSpace.hyperparameters.CategoricalHyperparameter].
        meta:
            Any additional meta information you would like to store along with the
            hyperparamter.
    """  # noqa: E501
    if ordered and weights is not None:
        raise ValueError("Can't apply `weights` to `ordered` Categorical")

    if ordered:
        return OrdinalHyperparameter(
            name=name,
            sequence=items,
            default_value=default,
            meta=meta,
        )

    return CategoricalHyperparameter(
        name=name,
        choices=items,
        default_value=default,
        weights=weights,
        meta=meta,
    )
