from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar

import numpy as np
from scipy.stats import randint

from ConfigSpace.hyperparameters._distributions import ScipyDiscreteDistribution
from ConfigSpace.hyperparameters._hp_components import (
    TransformerSeq,
    ordinal_neighborhood,
)
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


@dataclass(init=False)
class OrdinalHyperparameter(Hyperparameter[Any, np.int64]):
    orderable: ClassVar[bool] = True
    sequence: Sequence[Any] = field(hash=True)

    def __init__(
        self,
        name: str,
        sequence: Sequence[Any],
        default_value: Any | None = None,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        self.sequence = sequence
        size = len(sequence)
        if default_value is None:
            default_value = self.sequence[0]
        elif default_value not in sequence:
            raise ValueError(
                "The default value has to be one of the ordinal values. "
                f"Got {default_value!r} which is not in {sequence}.",
            )

        super().__init__(
            name=name,
            size=size,
            default_value=default_value,
            meta=meta,
            transformer=TransformerSeq(seq=sequence),
            neighborhood=partial(ordinal_neighborhood, size=int(size)),
            vector_dist=ScipyDiscreteDistribution(
                rv=randint(a=0, b=size),  # type: ignore
                max_density_value=1 / size,
                dtype=np.int64,
            ),
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
