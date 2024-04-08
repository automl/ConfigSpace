from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from ConfigSpace.functional import (
    is_close_to_integer,
    is_close_to_integer_single,
    normalize,
    scale,
)

RandomState = np.random.RandomState

CONSTANT_VECTOR_VALUE = np.int64(0)
ROUND_PLACES = 9
ATOL = 1e-9

DType = TypeVar("DType", bound=np.number)
"""Type variable for the data type of the hyperparameter."""

T_contra = TypeVar("T_contra", contravariant=True)


class _Transformer(Protocol[DType]):
    lower_vectorized: np.float64
    upper_vectorized: np.float64

    def to_value(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[DType]: ...

    def to_vector(self, value: npt.NDArray[DType]) -> npt.NDArray[np.float64]: ...

    def legal_value(self, value: npt.NDArray[DType]) -> npt.NDArray[np.bool_]: ...

    def legal_vector(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.bool_]: ...

    def legal_vector_single(self, vector: np.number) -> bool: ...

    def legal_value_single(self, vector: np.number) -> bool: ...


class _Neighborhood(Protocol):
    def __call__(
        self,
        vector: np.float64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
    ) -> npt.NDArray[np.float64]: ...


@dataclass
class TransformerSeq(_Transformer[Any]):
    lower_vectorized: ClassVar[np.int64] = np.int64(0)
    seq: npt.NDArray[Any]
    _lookup: dict[Any, int] | None = field(init=False)

    def __post_init__(self) -> None:
        if len(self.seq) == 0:
            raise ValueError("Sequence must have at least one element.")

        try:
            self._lookup = {v: i for i, v in enumerate(self.seq)}
        except TypeError:
            self._lookup = None

    @property
    def upper_vectorized(self) -> np.int64:
        return np.int64(len(self.seq))

    def to_value(self, vector: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        if not is_close_to_integer(vector, atol=ATOL).all():
            raise ValueError(
                "Got unexpected float value while trying to transform a vector"
                f" representation into a value in {self.seq}."
                f"Expected integers but got {vector} (dtype: {vector.dtype})",
            )
        indices = np.rint(vector).astype(np.int64)
        return self.seq[indices]

    def to_vector(self, value: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        if self._lookup is not None:
            return np.array([self._lookup[v] for v in value], dtype=np.int64)
        return np.flatnonzero(np.isin(self.seq, value))

    def legal_value(self, value: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
        if self._lookup is not None:
            return np.array([v in self._lookup for v in value], dtype=np.bool_)
        return np.isin(value, self.seq)

    def legal_vector(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.bool_]:
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


class UnitScaler(_Transformer[DType]):
    lower_vectorized: ClassVar[np.float64] = np.float64(0.0)
    upper_vectorized: ClassVar[np.float64] = np.float64(1.0)

    def __init__(
        self,
        lower_value: float | int | np.number,
        upper_value: float | int | np.number,
        *,
        dtype: type[DType],
        log: bool = False,
    ):
        if lower_value >= upper_value:
            raise ValueError(
                f"Upper bound {upper_value:f} must be larger than"
                f" lower bound {lower_value:f}",
            )

        if log and lower_value <= 0:
            raise ValueError(
                f"Negative lower bound {lower_value:f} for log-scale is not possible.",
            )

        self.lower_value: DType = dtype(lower_value)
        self.upper_value: DType = dtype(upper_value)
        self.log = log
        self.dtype = dtype
        self._scale_vec_to_int = upper_value - lower_value

    def to_value(
        self,
        vector: npt.NDArray[np.float64],
    ) -> npt.NDArray[DType]:
        """Transform a value from the unit interval to the range."""
        unchecked_values = self._unsafe_to_value(vector)
        if np.issubdtype(self.dtype, np.integer):
            return unchecked_values.round().astype(self.dtype)

        return np.clip(
            unchecked_values,
            self.lower_value,
            self.upper_value,
            dtype=self.dtype,
        )

    def _unsafe_to_value_single(
        self,
        vector: np.float64,
    ) -> np.float64:
        if self.log:
            _l = np.log(self.lower_value)
            _u = np.log(self.upper_value)
            scaled = vector * (_u - _l) + _l
            return np.exp(scaled)

        _l = self.lower_value
        _u = self.upper_value
        return vector * (_u - _l) + _l

    def _unsafe_to_value(
        self,
        vector: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # NOTE: Unsafe as it does not check boundaries, clip or integer'ness
        # linear (0-1) space to log scaling (0-1)
        if self.log:
            scaled_to_log_bounds = scale(
                vector,
                to=(np.log(self.lower_value), np.log(self.upper_value)),
            )
            return np.exp(scaled_to_log_bounds)

        return scale(vector, to=(self.lower_value, self.upper_value))

    def to_vector(self, value: npt.NDArray[DType]) -> npt.NDArray[np.float64]:
        """Transform a value from the range to the unit interval."""
        if self.log:
            return normalize(
                np.log(value),
                bounds=(np.log(self.lower_value), np.log(self.upper_value)),
            )

        return normalize(value, bounds=(self.lower_value, self.upper_value))

    def vectorize_size(self, size: float) -> np.float64:
        """Vectorize to the correct scale but is not necessarily in the range."""
        if self.log:
            return np.log(size) / (np.log(self.upper_value) - np.log(self.lower_value))

        return np.float64(size / (self.upper_value - self.lower_value))

    def legal_value(self, value: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
        # If we have a non numeric dtype, we have to unfortunatly go through but by bit
        if value.dtype.kind not in "iuf":
            return np.array([self.legal_value_single(v) for v in value], dtype=np.bool_)

        if np.issubdtype(self.dtype, np.integer):
            rints = np.rint(value)
            return (
                (rints >= self.lower_value)
                & (rints <= self.upper_value)
                & is_close_to_integer(value, atol=ATOL)
            )

        return (value >= self.lower_value) & (value <= self.upper_value)

    def legal_vector(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.bool_]:
        if np.issubdtype(self.dtype, np.integer):
            # NOTE: Unfortunatly for integers, we have to transform back to original
            # space to check if the vector value is indeed close to being an integer.
            # With a non-log spaced vector, we can multiply by the size of the range
            # as a quick check, giving us back integer values. However this does
            # not apply generally to a log-scale vector as the non-linear scaling
            # will not give us back integer values when doing the above.
            unchecked_values = self._unsafe_to_value(vector.astype(np.float64))
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
            return is_close_to_integer_single(scaled, atol=ATOL) and inbounds

        value = self._unsafe_to_value_single(vector)  # type: ignore
        return self.legal_value_single(value)


def ordinal_neighborhood(
    vector: np.int64,
    n: int,
    *,
    size: int,
    std: float | None = None,
    seed: RandomState | None = None,
) -> npt.NDArray[np.int64]:
    end_index = size - 1
    assert 0 <= vector <= end_index

    # No neighbors if it's the only element
    if size == 1:
        return np.array([])

    # We have at least 2 elements
    if vector == 0:
        return np.array([1])

    if vector == end_index:
        return np.array([end_index - 1])

    # We have at least 3 elements and the value is not at the ends
    neighbors = np.array([vector - 1, vector + 1])
    if n == 1:
        seed = np.random.RandomState() if seed is None else seed
        return np.array([seed.choice(neighbors)])

    return neighbors


@dataclass
class TransformerConstant(_Transformer[DType]):
    value: DType
    vector_value_yes: np.float64
    vector_value_no: np.float64

    @property
    def lower_vectorized(self) -> np.float64:
        return self.vector_value_no

    @property
    def upper_vectorized(self) -> np.float64:
        return self.vector_value_yes

    def to_vector(
        self,
        value: npt.NDArray[DType],
    ) -> np.float64 | npt.NDArray[np.float64]:
        return np.where(
            value == self.value,
            self.vector_value_yes,
            self.vector_value_no,
        ).astype(np.float64)

    def to_value(
        self,
        vector: np.float64 | npt.NDArray[np.float64],
    ) -> DType | npt.NDArray[DType]:
        if isinstance(vector, np.ndarray):
            try:
                return np.full_like(vector, self.value, dtype=type(self.value))
            except TypeError:
                # Let numpy figure it out
                return np.array([self.value] * len(vector))

        return self.value

    def legal_value(self, value: Any) -> bool:
        return value == self.value

    def legal_vector(self, vector: np.float64) -> bool:
        return vector == self.vector_value_yes

    def legal_value_single(self, vector: np.number) -> bool:
        return vector == self.value

    def legal_vector_single(self, vector: np.number) -> bool:
        return vector == self.vector_value_yes
