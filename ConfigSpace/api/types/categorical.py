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

    Note
    ----
    ``Categorical`` is actually a function, please use the corresponding return types if
    doing an `isinstance(param, type)` check with either
    :py:class:`~ConfigSpace.hyperparameters.CategoricalHyperparameter`
    and/or :py:class:`~ConfigSpace.hyperparameters.OrdinalHyperparameter`.

    Parameters
    ----------
    name: str
        The name of the hyperparameter

    items: Sequence[T],
        A list of items to put in the category. Note that there are limitations:

        * Can't use `None`, use a string "None" instead and convert as required.
        * Can't have duplicate categories, use weights if required.

    default: T | None = None
        The default value of the categorical hyperparameter

    weights: Sequence[float] | None = None
        The weights to apply to each categorical. Each item will be sampled according
        to these weights.

    ordered: bool = False
        Whether the categorical is ordered or not. If True, this will return an
        :py:class:`OrdinalHyperparameter`, otherwise it remain a
        :py:class:`CategoricalHyperparameter`.

    meta: dict | None = None
        Any additional meta information you would like to store along with the hyperparamter.
    """
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
