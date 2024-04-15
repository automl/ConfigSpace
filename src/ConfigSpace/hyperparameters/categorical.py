from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Set
from typing_extensions import deprecated

import numpy as np

from ConfigSpace.hyperparameters._distributions import (
    UniformIntegerDistribution,
    WeightedIntegerDiscreteDistribution,
)
from ConfigSpace.hyperparameters._hp_components import TransformerSeq, _Neighborhood
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter
from ConfigSpace.types import Array, f64

if TYPE_CHECKING:
    from ConfigSpace.types import Array


@dataclass
class NeighborhoodCat(_Neighborhood):
    size: int

    def __call__(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,  # noqa: ARG002
        seed: np.random.RandomState | None = None,
    ) -> Array[f64]:
        seed = np.random.RandomState() if seed is None else seed
        pivot = int(vector)
        _range = np.arange(0, self.size, dtype=np.float64)
        bot = _range[:pivot]
        top = _range[pivot + 1 :]
        choices = np.concatenate((bot, top))
        seed.shuffle(choices)
        return choices[:n]


@dataclass(init=False)
class CategoricalHyperparameter(Hyperparameter[Any]):
    ORDERABLE: ClassVar[bool] = False

    choices: Sequence[Any]
    weights: tuple[float, ...] | None
    probabilities: Array[f64] = field(repr=False, init=False)

    name: str
    default_value: Any
    meta: Mapping[Hashable, Any] | None
    size: int = field(init=False)

    def __init__(
        self,
        name: str,
        choices: Sequence[Any],
        default_value: Any | None = None,
        meta: Mapping[Hashable, Any] | None = None,
        weights: Sequence[float] | Array[np.number] | None = None,
    ) -> None:
        # TODO: We can allow for None but we need to be sure it doesn't break
        # anything elsewhere.
        if any(choice is None for choice in choices):
            raise TypeError("Choice 'None' is not supported")

        if isinstance(choices, Set):
            raise TypeError(
                "Using a set of choices is prohibited as it can result in "
                "non-deterministic behavior. Please use a list or a tuple.",
            )

        # TODO:For now we assume hashable for choices to make the below check with
        # Counter work. We can probably relax this assumption
        choices = tuple(choices)
        counter = Counter(choices)
        for choice, count in counter.items():
            if count > 1:
                raise ValueError(
                    f"Choices for categorical hyperparameters {name} contain"
                    f" choice `{choice}` {count} times, while only a single oocurence"
                    " is allowed.",
                )

        if isinstance(weights, set):
            raise TypeError(
                "Using a set of weights is prohibited as it can result in "
                "non-deterministic behavior. Please use a list or a tuple.",
            )

        if isinstance(weights, Sequence):
            if len(weights) != len(choices):
                raise ValueError(
                    "The list of weights and the list of choices are required to be"
                    f" of same length. Gave {len(weights)} weights and"
                    f" {len(choices)} choices.",
                )
            if any(weight < 0 for weight in weights):
                raise ValueError(
                    f"Negative weights are not allowed. Got {weights}.",
                )
            if all(weight == 0 for weight in weights):
                raise ValueError(
                    "All weights are zero, at least one weight has to be strictly"
                    " positive.",
                )
            tupled_weights = tuple(weights)
        elif weights is not None:
            raise TypeError(
                f"The weights have to be a list, tuple or None. Got {weights!r}.",
            )
        else:
            tupled_weights = None

        if default_value is not None and default_value not in choices:
            raise ValueError(
                "The default value has to be one of the choices. "
                f"Got {default_value!r} which is not in {choices}.",
            )

        size = len(choices)
        if weights is None:
            probabilities = np.full(size, fill_value=1 / size, dtype=f64)
        else:
            _weights = np.asarray(weights, dtype=np.float64)
            probabilities = _weights / np.sum(_weights)

        if default_value is None and weights is None:
            default_value = choices[0]
        elif default_value is None:
            highest_prob_index = np.argmax(probabilities)
            default_value = choices[highest_prob_index]
        elif default_value in choices:
            pass
        else:
            raise ValueError(f"Illegal default value {default_value}")

        # We only need to pass probabilties is they are non-uniform...
        if probabilities is not None:
            vector_dist = WeightedIntegerDiscreteDistribution(
                size=size,
                probabilities=np.asarray(probabilities),
            )
        else:
            vector_dist = UniformIntegerDistribution(size=size)

        # NOTE: Unfortunatly, numpy will promote number types to str
        # if there are string types in the array, where we'd rather
        # stick to object type in that case. Hence the manual...
        seq_choices = np.asarray(choices)
        if seq_choices.dtype.kind in {"U", "S"} and not all(
            isinstance(choice, str) for choice in choices
        ):
            seq_choices = np.asarray(choices, dtype=object)

        self.probabilities = probabilities
        self.choices = choices
        self.weights = tupled_weights

        super().__init__(
            name=name,
            default_value=default_value,
            vector_dist=vector_dist,
            size=size,
            transformer=TransformerSeq(seq=seq_choices),
            neighborhood=NeighborhoodCat(size=size),
            neighborhood_size=self._neighborhood_size,
            meta=meta,
        )

    def to_uniform(self) -> CategoricalHyperparameter:
        return CategoricalHyperparameter(
            name=self.name,
            choices=self.choices,
            weights=None,
            default_value=self.default_value,
            meta=self.meta,
        )

    def __eq__(self, other: Any) -> bool:
        if (
            not isinstance(other, self.__class__)
            or self.name != other.name
            or self.default_value != other.default_value
            or len(self.choices) != len(other.choices)
        ):
            return False

        # Longer check
        for this_choice, this_prob in zip(
            self.choices,
            self.probabilities,
        ):
            if this_choice not in other.choices:
                return False

            index_of_choice_in_other = other.choices.index(this_choice)
            other_prob = other.probabilities[index_of_choice_in_other]
            if this_prob != other_prob:
                return False

        return True

    def _neighborhood_size(self, value: Any | None) -> int:
        if value is None or value not in self.choices:
            return self.size
        return self.size - 1

    @property
    @deprecated("Please use `len(hp.choices)` or 'hp.size' instead.")
    def num_choices(self) -> int:
        return self.size

    @property
    @deprecated("Please use `.probabilities`. Note, it's a np.ndarray now.")
    def _probabilities(self) -> tuple[float, ...]:
        return tuple(self.probabilities)

    def __str__(self) -> str:
        parts = [
            self.name,
            f"Type: {str(self.__class__.__name__).replace('Hyperparameter', '')}",
            "Choices: {" + ", ".join(map(str, self.choices)) + "}",
            f"Default: {self.default_value}",
        ]
        if not np.all(self.probabilities == self.probabilities[0]):
            parts.append(f"Probabilities: {self.probabilities}")

        return ", ".join(parts)
