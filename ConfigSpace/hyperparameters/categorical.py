from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from scipy.stats import rv_discrete

from ConfigSpace.hyperparameters._distributions import ScipyDiscreteDistribution
from ConfigSpace.hyperparameters._hp_components import (
    NeighborhoodCat,
    TransformerSeq,
)
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


@dataclass(init=False)
class CategoricalHyperparameter(Hyperparameter[Any, np.int64]):
    orderable: ClassVar[bool] = False
    choices: Sequence[Any]
    weights: Sequence[int | float] | None
    probabilities: npt.NDArray[np.float64] = field(repr=False, init=False)

    def __init__(
        self,
        name: str,
        choices: Sequence[Any],
        weights: Sequence[int | float] | None = None,
        default_value: Any | None = None,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        # TODO check that there is no bullshit in the choices!
        choices = list(choices)
        counter = Counter(choices)
        for choice, count in counter.items():
            if count > 1:
                raise ValueError(
                    f"Choices for categorical hyperparameters {name} contain choice"
                    f" `{choice:!r}` {count} times, while only a single oocurence is"
                    " allowed.",
                )

        match weights:
            case set():
                raise TypeError(
                    "Using a set of weights is prohibited as it can result in "
                    "non-deterministic behavior. Please use a list or a tuple.",
                )
            case Sequence():
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
        self.probabilities = probabilities

        match default_value, weights:
            case None, None:
                default_value = choices[0]
            case None, _:
                default_value = choices[np.argmax(np.asarray(weights))]
            case _ if default_value in choices:
                default_value = default_value  # noqa: PLW0127
            case _:
                raise ValueError(f"Illegal default value {default_value}")

        size = len(choices)
        custom_discrete = rv_discrete(
            values=(np.arange(size), probabilities),
            a=0,
            b=size,
        ).freeze()
        vect_dist = ScipyDiscreteDistribution(
            rv=custom_discrete,  # type: ignore
            max_density_value=1 / size,
            dtype=np.int64,
        )

        super().__init__(
            name=name,
            size=size,
            default_value=default_value,
            meta=meta,
            transformer=TransformerSeq(seq=choices),
            neighborhood=NeighborhoodCat(n=len(choices)),
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

        if self.probabilities is not None:
            ordered_probabilities_self = {
                choice: self.probabilities[i] for i, choice in enumerate(self.choices)
            }
        else:
            ordered_probabilities_self = None

        if other.probabilities is not None:
            ordered_probabilities_other = {
                choice: (
                    other.probabilities[other.choices.index(choice)]
                    if choice in other.choices
                    else None
                )
                for choice in self.choices
            }
        else:
            ordered_probabilities_other = None

        return ordered_probabilities_self == ordered_probabilities_other

    def _neighborhood_size(self, value: Any | None) -> int:
        if value is None or value not in self.choices:
            return int(self.size)
        return int(self.size) - 1
