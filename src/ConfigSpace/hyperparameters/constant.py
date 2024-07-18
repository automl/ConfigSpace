from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar
from typing_extensions import override

import numpy as np

from ConfigSpace.hyperparameters.distributions import ConstantVectorDistribution
from ConfigSpace.hyperparameters.hp_components import TransformerConstant
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter
from ConfigSpace.types import Array, Mask, f64

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

    _contains_sequence_as_value: bool = False

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
        if isinstance(value, np.ndarray):
            raise ValueError(
                "Constant hyperparameter does not support numpy arrays as values",
            )
        self.value = value
        self._contains_sequence_as_value = isinstance(
            value,
            Sequence,
        ) and not isinstance(value, str)

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

    @override
    def legal_value(self, value: Any | Sequence[Any] | Array[Any]) -> bool | Mask:
        if isinstance(value, np.ndarray):
            return self._transformer.legal_value(value)

        if isinstance(value, str):
            return self._transformer.legal_value(np.array([value]))[0]

        # Got a sequence of things, could be a list of stuff or a single value which is
        # itself a list, e.g. a tuple (1, 2) indicating a single value
        # If we could have single values which are sequences, we need to do some
        # magic to get it into an array without numpy flattening it down
        if isinstance(value, Sequence):
            if self._contains_sequence_as_value:
                # https://stackoverflow.com/a/47389566/5332072
                _v = np.empty(1, dtype=object)
                _v[0] = value
                return self._transformer.legal_value(_v)[0]

            # A sequence of things containing different values
            return self._transformer.legal_value(np.asarray(value))

        # Single value that is not a sequence
        return self._transformer.legal_value(np.array([value]))[0]

    @override
    def pdf_values(self, values: Sequence[Any] | Array[Any]) -> Array[f64]:
        if isinstance(values, np.ndarray):
            if values.ndim != 1:
                raise ValueError("Method pdf expects a one-dimensional numpy array")

            vector = self.to_vector(values)  # type: ignore
            return self.pdf_vector(vector)

        if self._contains_sequence_as_value:
            # We have to convert it into a numpy array of objects carefully
            # https://stackoverflow.com/a/47389566/5332072
            _v = np.empty(len(values), dtype=object)
            _v[:] = values
            _vector: Array[f64] = self.to_vector(_v)  # type: ignore
            return self.pdf_vector(_vector)

        vector: Array[f64] = self.to_vector(values)  # type: ignore
        return self.pdf_vector(vector)

    @override
    def to_vector(self, value: Any | Sequence[Any] | Array[Any]) -> f64 | Array[f64]:
        if isinstance(value, np.ndarray):
            return self._transformer.to_vector(value)

        if isinstance(value, str):
            return self._transformer.to_vector(np.array([value]))[0]

        # Got a sequence of things, could be a list of stuff or a single value which is
        # itself a list, e.g. a tuple (1, 2) indicating a single value
        # If we could have single values which are sequences, we need to do some
        # magic to get it into an array without numpy flattening it down
        if isinstance(value, Sequence):
            if self._contains_sequence_as_value:
                # https://stackoverflow.com/a/47389566/5332072
                _v = np.empty(1, dtype=object)
                _v[0] = value
                return self._transformer.to_vector(_v)[0]

            # A sequence of things containing different values
            return self._transformer.to_vector(np.asarray(value))

        # Single value that is not a sequence
        return self._transformer.to_vector(np.array([value]))[0]


UnParametrizedHyperparameter = Constant  # Legacy
