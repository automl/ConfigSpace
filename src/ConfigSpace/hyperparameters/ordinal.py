from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar
from typing_extensions import deprecated, override

import numpy as np

from ConfigSpace.hyperparameters.distributions import UniformIntegerDistribution
from ConfigSpace.hyperparameters.hp_components import (
    TransformerSeq,
    ordinal_neighborhood,
)
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter
from ConfigSpace.types import Array, Mask, NotSet, _NotSet, f64, i64


@dataclass(init=False)
class OrdinalHyperparameter(Hyperparameter[Any, Any]):
    """Representing an ordinal hyperparameter in the configuration space.

    An ordinal hyperparameter is a hyperparameter that can take on one of a
    fixed number of arbitrary values. The values are ordered and the order is
    defined by the sequence in which they are passed to the hyperparameter.

    ```python exec="True" result="python" source="material-block"
    from ConfigSpace import OrdinalHyperparameter

    hp = OrdinalHyperparameter('hp', sequence=['small', 'medium', 'large'], default_value='medium')
    print(hp)
    ```
    """  # noqa: E501

    ORDERABLE: ClassVar[bool] = True

    sequence: tuple[Any, ...]
    """Sequence of values the hyperparameter can take on."""

    name: str
    """Name of the hyperparameter, with which it can be accessed."""

    default_value: Any
    """Default value of the hyperparameter."""

    meta: Mapping[Hashable, Any] | None
    """Field for holding meta data provided by the user. Not used by the ConfigSpace."""

    size: int = field(init=False)
    """Size of the hyperparameter, which is the number of possible values the
    hyperparameter can take on within the specified sequence."""

    _contains_sequence_as_value: bool

    def __init__(
        self,
        name: str,
        sequence: Sequence[Any],
        default_value: Any | _NotSet = NotSet,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """Initialize an ordinal hyperparameter.

        Args:
            name:
                Name of the hyperparameter, with which it can be accessed
            sequence:
                Sequence of values the hyperparameter can take on
            default_value:
                Default value of the hyperparameter
            meta:
                Field for holding meta data provided by the user
        """
        # TODO: Maybe give some way to not check this, i.e. for large sequences
        # of int...
        if any(i != sequence.index(x) for i, x in enumerate(sequence)):
            raise ValueError(
                "The sequence has to be a list of unique elements as defined"
                " by object equality."
                f"Got {sequence} which does not fulfill this requirement.",
            )

        size = len(sequence)
        if default_value is NotSet:
            default_value = sequence[0]
        elif default_value not in sequence:
            raise ValueError(
                "The default value has to be one of the ordinal values. "
                f"Got {default_value!r} which is not in {sequence}.",
            )

        try:
            # This can fail with a ValueError if the choices contain arbitrary objects
            # that are list like.
            seq_choices = np.asarray(sequence)
            if seq_choices.ndim != 1:
                raise ValueError

            # NOTE: Unfortunatly, numpy will promote number types to str
            # if there are string types in the array, where we'd rather
            # stick to object type in that case. Hence the manual...
            if seq_choices.dtype.kind in {"U", "S"} and not all(
                isinstance(item, str) for item in sequence
            ):
                seq_choices = np.array(sequence, dtype=object)

        except ValueError:
            seq_choices = list(sequence)

        self.sequence = tuple(sequence)

        # If the Hyperparameter recieves as a Sequence during legality checks or
        # conversions, we need to inform it that one of the values is a Sequence itself,
        # i.e. we should treat it as a single value and not a list of multiple values
        self._contains_sequence_as_value = any(
            isinstance(item, Sequence) and not isinstance(item, str)
            for item in self.sequence
        )

        super().__init__(
            name=name,
            size=size,
            default_value=default_value,
            meta=meta,
            transformer=TransformerSeq(seq=seq_choices),
            neighborhood=partial(ordinal_neighborhood, size=int(size)),
            vector_dist=UniformIntegerDistribution(size=size),
            neighborhood_size=self._ordinal_neighborhood_size,
            value_cast=None,
        )

    def _ordinal_neighborhood_size(self, value: Any | _NotSet) -> int:
        size = len(self.sequence)
        if value is NotSet:
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
        """Check if the value is smaller than the other value."""
        return self.sequence.index(value) < self.sequence.index(other)

    def get_order(self, value: Any) -> int:
        """Get the order of the value in the sequence."""
        return self.sequence.index(value)

    def get_value(self, i: int | np.integer) -> Any:
        """Get the value at the index in the sequence."""
        return self.sequence[int(i)]

    def get_seq_order(self) -> Array[i64]:
        """Get the order of the sequence."""
        return np.arange(len(self.sequence))

    def __str__(self) -> str:
        parts = [
            self.name,
            f"Type: {str(self.__class__.__name__).replace('Hyperparameter', '')}",
            "Sequence: {" + ", ".join(map(str, self.sequence)) + "}",
            f"Default: {self.default_value}",
        ]
        return ", ".join(parts)

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

    @property
    @deprecated("Please use 'len(hp.sequence)' or `hp.size` instead.")
    def num_elements(self) -> int:
        """Deprecated: Number of elements in the sequence."""
        return self.size
