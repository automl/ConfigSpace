from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Set

import numpy as np
import numpy.typing as npt

from ConfigSpace.hyperparameters._distributions import (
    UniformIntegerDistribution,
    WeightedIntegerDiscreteDistribution,
)
from ConfigSpace.hyperparameters._hp_components import (
    TransformerSeq,
    _Neighborhood,
)
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


@dataclass
class NeighborhoodCat(_Neighborhood):
    size: int

    def __call__(
        self,
        vector: np.float64,
        n: int,
        *,
        std: float | None = None,
        seed: np.random.RandomState | None = None,
    ) -> npt.NDArray[np.float64]:
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
    serializable_type_name: ClassVar[str] = "categorical"
    orderable: ClassVar[bool] = False

    choices: tuple[Any, ...]
    weights: tuple[float, ...] | None
    probabilities: tuple[float, ...] = field(repr=False, init=False)

    def __init__(
        self,
        name: str,
        choices: Sequence[Any],
        default_value: Any | None = None,
        weights: Sequence[int | float] | None = None,
        meta: Mapping[Hashable, Any] | None = None,
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
                    f"Choices for categorical hyperparameters {name} contain choice"
                    f" `{choice}` {count} times, while only a single oocurence is"
                    " allowed.",
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
            weights = tuple(weights)
        elif weights is not None:
            raise TypeError(
                f"The weights have to be a list, tuple or None. Got {weights!r}.",
            )

        if default_value is not None and default_value not in choices:
            raise ValueError(
                "The default value has to be one of the choices. "
                f"Got {default_value!r} which is not in {choices}.",
            )

        if weights is None:
            _weights = np.ones(len(choices), dtype=np.float64)
        else:
            _weights = np.asarray(weights, dtype=np.float64)

        probabilities = _weights / np.sum(_weights)

        self.choices = choices
        self.weights = weights
        self.probabilities = tuple(probabilities)
        size = len(choices)

        if default_value is None and weights is None:
            default_value = choices[0]
        elif default_value is None:
            default_value = choices[np.argmax(np.asarray(weights))]
        elif default_value in choices:
            default_value = default_value  # noqa: PLW0127
        else:
            raise ValueError(f"Illegal default value {default_value}")

        # We only need to pass probabilties is they are non-uniform...
        if self.weights is not None:
            vect_dist = WeightedIntegerDiscreteDistribution(
                size=size,
                probabilities=np.asarray(self.probabilities),
            )
        else:
            vect_dist = UniformIntegerDistribution(size=size)

        seq_choices = np.asarray(choices)
        # NOTE: Unfortunatly, numpy will promote number types to str
        # if there are string types in the array, where we'd rather
        # stick to object type in that case. Hence the manual...
        if seq_choices.dtype.kind in {"U", "S"} and not all(
            isinstance(choice, str) for choice in choices
        ):
            seq_choices = np.asarray(choices, dtype=object)

        super().__init__(
            name=name,
            size=size,
            default_value=default_value,
            meta=meta,
            transformer=TransformerSeq(seq=seq_choices),
            neighborhood=NeighborhoodCat(size=len(choices)),
            vector_dist=vect_dist,
            neighborhood_size=self._neighborhood_size,
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
        if not isinstance(other, self.__class__):
            return False

        # Quick checks first
        if len(self.choices) != len(other.choices):
            return False

        if self.default_value != other.default_value:
            return False

        if self.name != other.name:
            return False

        # Longer check
        for this_choice, this_prob in zip(
            self.choices,
            self.probabilities,
            strict=True,
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
            return int(self.size)
        return int(self.size) - 1

    @property
    def num_choices(self) -> int:
        warnings.warn(
            "The property 'num_choices' is deprecated and will be removed in a future"
            " release. Please use either `len(hp.choices)` or 'hp.size' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return len(self.choices)

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.serializable_type_name,
            "choices": list(self.choices),
            "default_value": self.default_value,
            "weights": self.weights,
            "meta": self.meta,
        }
