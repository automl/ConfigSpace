from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np

from ConfigSpace.functional import (
    is_close_to_integer,
    is_close_to_integer_single,
    normalize,
    scale,
)
from ConfigSpace.types import DType, f64, i64

if TYPE_CHECKING:
    from ConfigSpace.types import Array, Mask

RandomState = np.random.RandomState

CONSTANT_VECTOR_VALUE = i64(0)
ROUND_PLACES = 9
ATOL = 1e-9

T_contra = TypeVar("T_contra", contravariant=True)


class _Transformer(Protocol[DType]):
    lower_vectorized: f64
    upper_vectorized: f64

    def to_value(self, vector: Array[f64]) -> Array[DType]: ...

    def to_vector(self, value: Array[DType]) -> Array[f64]: ...

    def legal_value(self, value: Array[DType]) -> Mask: ...

    def legal_vector(self, vector: Array[f64]) -> Mask: ...

    def legal_vector_single(self, vector: np.number) -> bool: ...

    def legal_value_single(self, value: np.number) -> bool: ...


class _Neighborhood(Protocol):
    def __call__(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
    ) -> Array[f64]: ...


@dataclass
class TransformerSeq(_Transformer[Any]):
    seq: Array[Any]
    _lookup: dict[Any, int] | None = field(init=False)

    lower_vectorized: f64 = field(init=False)
    upper_vectorized: f64 = field(init=False)

    def __post_init__(self) -> None:
        if len(self.seq) == 0:
            raise ValueError("Sequence must have at least one element.")

        try:
            self._lookup = {v: i for i, v in enumerate(self.seq)}
        except TypeError:
            self._lookup = None

        self.lower_vectorized = f64(0)
        self.upper_vectorized = f64(len(self.seq))

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, TransformerSeq):
            return False

        return (
            np.array_equal(self.seq, value.seq)
            and self.lower_vectorized == value.lower_vectorized
        )

    def to_value(self, vector: Array[f64]) -> Array[Any]:
        if not is_close_to_integer(vector, atol=ATOL).all():
            raise ValueError(
                "Got unexpected float value while trying to transform a vector"
                f" representation into a value in {self.seq}."
                f"Expected integers but got {vector} (dtype: {vector.dtype})",
            )
        indices: Array[np.intp] = np.rint(vector).astype(np.intp)
        return self.seq[indices]  # type: ignore

    def to_vector(self, value: Array[Any]) -> Array[f64]:
        if self._lookup is not None:
            return np.array([self._lookup[v] for v in value], dtype=f64)
        return np.flatnonzero(np.isin(self.seq, value)).astype(f64)

    def legal_value(self, value: Array[Any]) -> Mask:
        if self._lookup is not None:
            return np.array([v in self._lookup for v in value], dtype=np.bool_)
        return np.isin(value, self.seq)

    def legal_vector(self, vector: Array[f64]) -> Mask:
        return (
            (vector >= 0)
            & (vector < len(self.seq))
            & is_close_to_integer(vector, atol=ATOL)
        )

    def legal_value_single(self, value: Any) -> bool:
        return value in self.seq

    def legal_vector_single(self, vector: np.number) -> bool:
        return bool(
            vector >= 0
            and vector < len(self.seq)
            and is_close_to_integer_single(vector, atol=ATOL),
        )


@dataclass
class UnitScaler(_Transformer[DType]):
    lower_value: DType
    upper_value: DType
    dtype: type[DType]
    log: bool = False

    lower_vectorized: f64 = field(init=False)
    upper_vectorized: f64 = field(init=False)
    _scale_vec_to_int: DType = field(init=False)

    def __post_init__(self) -> None:
        if self.lower_value >= self.upper_value:
            raise ValueError(
                f"Upper bound {self.upper_value:f} must be larger than"
                f" lower bound {self.lower_value:f}",
            )

        if self.log and self.lower_value <= 0:
            raise ValueError(
                f"Negative lower bound {self.lower_value:f} for log-scale is not"
                " possible.",
            )

        self.lower_vectorized = f64(0)
        self.upper_vectorized = f64(1)

        self._scale_vec_to_int = self.upper_value - self.lower_value

    def to_value(self, vector: Array[f64]) -> Array[DType]:
        """Transform a value from the unit interval to the range."""
        unchecked_values = self._unsafe_to_value(vector)
        if np.issubdtype(self.dtype, np.integer):
            return unchecked_values.round().astype(self.dtype)

        return np.clip(  # type: ignore
            unchecked_values,
            self.lower_value,
            self.upper_value,
            dtype=self.dtype,
        )

    def _unsafe_to_value_single(self, vector: f64) -> f64:
        if self.log:
            _l = np.log(self.lower_value)
            _u = np.log(self.upper_value)
            scaled = vector * (_u - _l) + _l
            return np.exp(scaled)  # type: ignore

        _l = self.lower_value
        _u = self.upper_value
        return vector * (_u - _l) + _l  # type: ignore

    def _unsafe_to_value(self, vector: Array[f64]) -> Array[f64]:
        # NOTE: Unsafe as it does not check boundaries, clip or integer'ness
        # linear (0-1) space to log scaling (0-1)
        if self.log:
            scaled_to_log_bounds = scale(
                vector,
                to=(np.log(self.lower_value), np.log(self.upper_value)),
            )
            return np.exp(scaled_to_log_bounds)

        return scale(vector, to=(self.lower_value, self.upper_value))

    def to_vector(self, value: Array[DType]) -> Array[f64]:
        """Transform a value from the range to the unit interval."""
        if self.log:
            return normalize(
                np.log(value),
                bounds=(np.log(self.lower_value), np.log(self.upper_value)),
            )

        return normalize(value, bounds=(self.lower_value, self.upper_value))

    def vectorize_size(self, size: f64) -> f64:
        """Vectorize to the correct scale but is not necessarily in the range."""
        if self.log:
            return np.abs(  # type: ignore
                np.log(size) / (np.log(self.upper_value) - np.log(self.lower_value)),
            )

        return f64(size / (self.upper_value - self.lower_value))

    def legal_value(self, value: Array[Any]) -> Mask:
        # If we have a non numeric dtype, we have to unfortunatly go through but by bit
        if value.dtype.kind not in "iuf":
            return np.array([self.legal_value_single(v) for v in value], dtype=np.bool_)

        if np.issubdtype(self.dtype, np.integer):
            rints = np.rint(value)
            return (  # type: ignore
                (rints >= self.lower_value)
                & (rints <= self.upper_value)
                & is_close_to_integer(value, atol=ATOL)
            )

        return (value >= self.lower_value) & (value <= self.upper_value)

    def legal_vector(self, vector: Array[f64]) -> Mask:
        if np.issubdtype(self.dtype, np.integer):
            # NOTE: Unfortunatly for integers, we have to transform back to original
            # space to check if the vector value is indeed close to being an integer.
            # With a non-log spaced vector, we can multiply by the size of the range
            # as a quick check, giving us back integer values. However this does
            # not apply generally to a log-scale vector as the non-linear scaling
            # will not give us back integer values when doing the above.
            unchecked_values = self._unsafe_to_value(vector.astype(f64))
            return self.legal_value(unchecked_values)

        return (vector >= self.lower_vectorized) & (vector <= self.upper_vectorized)

    def legal_value_single(self, value: Any) -> bool:
        if not isinstance(value, (int, float, np.number)):
            return False

        if np.issubdtype(self.dtype, np.integer):
            rint = np.rint(value)
            return bool(
                (self.lower_value <= rint)
                & (rint <= self.upper_value)
                & is_close_to_integer_single(value, atol=ATOL),
            )

        return bool(self.lower_value <= value <= self.upper_value)

    def legal_vector_single(self, vector: np.number) -> bool:
        if not np.issubdtype(self.dtype, np.integer):
            return bool(self.lower_vectorized <= vector <= self.upper_vectorized)

        if not self.log:
            inbounds = bool(self.lower_vectorized <= vector <= self.upper_vectorized)
            scaled = vector * self._scale_vec_to_int
            return bool(is_close_to_integer_single(scaled, atol=ATOL) and inbounds)

        value = self._unsafe_to_value_single(vector)  # type: ignore
        return self.legal_value_single(value)


def ordinal_neighborhood(
    vector: f64,
    n: int,
    *,
    size: int,
    std: float | None = None,  # noqa: ARG001
    seed: RandomState | None = None,
) -> Array[f64]:
    end_index = size - 1
    assert 0 <= vector <= end_index

    # No neighbors if it's the only element
    if size == 1:
        return np.array([], dtype=f64)

    # We have at least 2 elements,
    # in this case it's only neighbor is the one beside it
    if vector == 0:
        return np.array([1.0], dtype=f64)

    # Also only one neighbor for the other end
    if np.rint(vector) == end_index:
        return np.array([end_index - 1], dtype=f64)

    # We have at least 3 elements and the value is not at the ends
    neighbors: Array[f64] = np.array([vector - 1, vector + 1], dtype=f64)
    if n >= 2:
        return neighbors

    seed = np.random.RandomState() if seed is None else seed
    return np.array([seed.choice(neighbors)], dtype=f64)


@dataclass
class TransformerConstant(_Transformer[DType]):
    value: DType
    vector_value_yes: f64
    vector_value_no: f64

    lower_vectorized: f64 = field(init=False)
    upper_vectorized: f64 = field(init=False)

    def __post_init__(self) -> None:
        self.lower_vectorized = self.vector_value_no
        self.upper_vectorized = self.vector_value_yes

    def to_vector(self, value: Array[DType]) -> Array[f64]:
        return np.where(
            value == self.value,
            self.vector_value_yes,
            self.vector_value_no,
        )

    def to_value(self, vector: Array[f64]) -> Array[DType]:
        try:
            return np.full_like(vector, self.value, dtype=type(self.value))
        except TypeError:
            # Let numpy figure it out
            return np.array([self.value] * len(vector))

    def legal_value(self, value: Array[DType]) -> Mask:
        return value == self.value  # type: ignore

    def legal_vector(self, vector: Array[f64]) -> Mask:
        return vector == self.vector_value_yes  # type: ignore

    def legal_value_single(self, value: Any) -> bool:
        return value == self.value  # type: ignore

    def legal_vector_single(self, vector: np.number) -> bool:
        return vector == self.vector_value_yes  # type: ignore
