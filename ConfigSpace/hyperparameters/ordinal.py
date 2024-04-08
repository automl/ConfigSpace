from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ConfigSpace.hyperparameters._distributions import UniformIntegerDistribution
from ConfigSpace.hyperparameters._hp_components import (
    TransformerSeq,
    ordinal_neighborhood,
)
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


@dataclass(init=False)
class OrdinalHyperparameter(Hyperparameter[Any]):
    serializable_type_name: ClassVar[str] = "ordinal"
    orderable: ClassVar[bool] = True
    sequence: tuple[Any, ...] = field(hash=True)

    def __init__(
        self,
        name: str,
        sequence: Sequence[Any],
        default_value: Any | None = None,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        self.sequence = tuple(sequence)

        # TODO: Maybe give some way to not check this, i.e. for large sequences
        # of int...
        if any(i != sequence.index(x) for i, x in enumerate(sequence)):
            raise ValueError(
                "The sequence has to be a list of unique elements as defined"
                " by object equality."
                f"Got {sequence} which does not fulfill this requirement.",
            )

        size = len(sequence)
        if default_value is None:
            default_value = self.sequence[0]
        elif default_value not in sequence:
            raise ValueError(
                "The default value has to be one of the ordinal values. "
                f"Got {default_value!r} which is not in {sequence}.",
            )

        seq_choices = np.asarray(sequence)
        # NOTE: Unfortunatly, numpy will promote number types to str
        # if there are string types in the array, where we'd rather
        # stick to object type in that case. Hence the manual...
        if seq_choices.dtype.kind in {"U", "S"} and not all(
            isinstance(item, str) for item in sequence
        ):
            seq_choices = np.asarray(sequence, dtype=object)

        super().__init__(
            name=name,
            size=size,
            default_value=default_value,
            meta=meta,
            transformer=TransformerSeq(seq=seq_choices),
            neighborhood=partial(ordinal_neighborhood, size=int(size)),
            vector_dist=UniformIntegerDistribution(size=size),
            neighborhood_size=self._neighborhood_size,
        )

    def _neighborhood_size(self, value: Any | None) -> int:
        size = len(self.sequence)
        if value is None:
            return size

        # No neighbors if it's the only element
        if size == 1:
            return 0

        end_index = len(self.sequence) - 1
        index = self.sequence.index(value)

        # We have at least 2 elements
        if index in (0, end_index):
            return 1

        # We have at least 3 elements and the value is not at the ends
        return 2

    def check_order(self, value: Any, other: Any) -> bool:
        return self.sequence.index(value) < self.sequence.index(other)

    def get_order(self, value: Any) -> int:
        return self.sequence.index(value)

    def get_value(self, i: int | np.int64) -> Any:
        return self.sequence[int(i)]

    def get_seq_order(self) -> npt.NDArray[np.int64]:
        return np.arange(len(self.sequence))

    def __str__(self) -> str:
        parts = [
            self.name,
            f"Type: {str(self.__class__.__name__).replace('Hyperparameter', '')}",
            "Sequence: {" + ", ".join(map(str, self.sequence)) + "}",
            f"Default: {self.default_value}",
        ]
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.serializable_type_name,
            "sequence": self.sequence,
            "default_value": self.default_value,
        }

    @property
    def num_elements(self) -> int:
        warnings.warn(
            "The property 'num_elements' is deprecated and will be removed in a future"
            " release. Please use either `len(hp.sequence)` or 'hp.size' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return len(self.sequence)
