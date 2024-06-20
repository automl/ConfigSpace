from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ConfigSpace.hyperparameters._distributions import ConstantVectorDistribution
from ConfigSpace.hyperparameters._hp_components import TransformerConstant
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter
from ConfigSpace.types import Array, f64

CONSTANT_VECTOR_VALUE_YES = f64(1)
"""Vectorized value for constant when set."""

CONSTANT_VECTOR_VALUE_NO = f64(0)
"""Vectorized value for constant when not set."""


def _empty_neighborhood(*_: Any, **__: Any) -> Array[f64]:
    return np.ndarray([], dtype=f64)


@dataclass(init=False)
class Constant(Hyperparameter[Any, Any]):
    """Representing a constant hyperparameter in the configuration space.

    By sampling from the configuration space each time only a single,
    constant `value` will be drawn from this hyperparameter.
    """

    ORDERABLE: ClassVar[bool] = False

    name: str
    """Name of the hyperparameter, with which it can be accessed."""

    value: Any
    """Value to sample hyperparameter from."""

    meta: Mapping[Hashable, Any] | None
    """Field for holding meta data provided by the user. Not used by the ConfigSpace."""

    size: int
    """Size of the hyperparameter, which is always 1 for a constant hyperparameter."""

    def __init__(
        self,
        name: str,
        value: Any,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """Representing a constant hyperparameter in the configuration space.

        By sampling from the configuration space each time only a single,
        constant `value` will be drawn from this hyperparameter.

        Args:
            name:
                Name of the hyperparameter, with which it can be accessed
            value:
                value to sample hyperparameter from
            meta:
                Field for holding meta data provided by the user.
                Not used by the configuration space.
        """
        # TODO: This should be changed and allowed...
        if not isinstance(value, (int, float, str)) or isinstance(value, bool):
            raise TypeError(
                f"Constant hyperparameter '{name}' must be of type int, float or str, "
                f"but got {type(value).__name__}.",
            )

        self.value = value

        super().__init__(
            name=name,
            default_value=value,
            size=1,
            meta=meta,
            transformer=TransformerConstant(
                value=value,
                vector_value_yes=CONSTANT_VECTOR_VALUE_YES,
                vector_value_no=CONSTANT_VECTOR_VALUE_NO,
            ),
            vector_dist=ConstantVectorDistribution(
                vector_value=CONSTANT_VECTOR_VALUE_YES,
            ),
            neighborhood=_empty_neighborhood,
            neighborhood_size=0,
            value_cast=None,
        )

    def __str__(self) -> str:
        parts = [
            self.name,
            f"Type: {str(self.__class__.__name__).replace('Hyperparameter', '')}",
            f"Value: {self.value}",
        ]

        return ", ".join(parts)


UnParametrizedHyperparameter = Constant  # Legacy
