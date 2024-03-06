from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ConfigSpace.hyperparameters._distributions import ConstantVectorDistribution
from ConfigSpace.hyperparameters._hp_components import (
    CONSTANT_VECTOR_VALUE,
    DType,
    TransformerConstant,
)
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


def _empty_neighborhood(*_: Any, **__: Any) -> npt.NDArray[np.integer]:
    return np.ndarray([], dtype=np.integer)


class Constant(Hyperparameter[DType, np.integer]):
    orderable: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        value: DType,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """Representing a constant hyperparameter in the configuration space.

        By sampling from the configuration space each time only a single,
        constant ``value`` will be drawn from this hyperparameter.

        Args:
            name:
                Name of the hyperparameter, with which it can be accessed
            value:
                value to sample hyperparameter from
            meta:
                Field for holding meta data provided by the user.
                Not used by the configuration space.
        """
        self.value = value

        super().__init__(
            name=name,
            default_value=value,
            size=1,
            meta=meta,
            transformer=TransformerConstant(value=value),
            vector_dist=ConstantVectorDistribution(value=CONSTANT_VECTOR_VALUE),
            neighborhood=_empty_neighborhood,
            neighborhood_size=0,
        )


class UnParametrizedHyperparameter(Constant):
    pass
