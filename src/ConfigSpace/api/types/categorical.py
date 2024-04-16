from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Union, overload
from typing_extensions import TypeAlias

from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.types import NotSet, _NotSet

# We only accept these types in `items`
T: TypeAlias = Union[str, int, float]


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
    some manner, e.g. ``["small", "medium", "large"]``.

    .. code:: python

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

    Note:
    ----
    ``Categorical`` is actually a function, please use the corresponding return types if
    doing an `isinstance(param, type)` check with either
    :py:class:`~ConfigSpace.hyperparameters.CategoricalHyperparameter`
    and/or :py:class:`~ConfigSpace.hyperparameters.OrdinalHyperparameter`.

    Parameters
    ----------
    name:
        The name of the hyperparameter

    items:
        A list of items to put in the category. Note that there are limitations:

        * Can't use `None`, use a string "None" instead and convert as required.
        * Can't have duplicate categories, use weights if required.

    default:
        The default value of the categorical hyperparameter

    weights:
        The weights to apply to each categorical. Each item will be sampled according
        to these weights.

    ordered:
        Whether the categorical is ordered or not. If True, this will return an
        :py:class:`OrdinalHyperparameter`, otherwise it remain a
        :py:class:`CategoricalHyperparameter`.

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
