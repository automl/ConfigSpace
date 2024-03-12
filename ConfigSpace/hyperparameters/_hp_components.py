from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from ConfigSpace.functional import (
    is_close_to_integer,
    normalize,
    scale,
    split_arange,
)

RandomState = np.random.RandomState

CONSTANT_VECTOR_VALUE = np.int64(0)
ROUND_PLACES = 9
ABS_ROUND_CLOSENESS = 1 / 10**ROUND_PLACES

DType = TypeVar("DType", bound=np.number)
"""Type variable for the data type of the hyperparameter."""

VDType = TypeVar("VDType", bound=np.number)
"""Type variable for the data type of the vectorized hyperparameter."""

T_contra = TypeVar("T_contra", contravariant=True)


class _Transformer(Protocol[DType, VDType]):
    lower_vectorized: VDType
    upper_vectorized: VDType

    def to_value(self, vector: npt.NDArray[VDType]) -> npt.NDArray[DType]: ...

    def to_vector(
        self,
        value: Sequence[DType] | npt.NDArray[DType],
    ) -> npt.NDArray[VDType]: ...

    def legal_value(self, value: npt.NDArray[DType]) -> npt.NDArray[np.bool_]: ...

    def legal_vector(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.bool_]: ...


class _Neighborhood(Protocol[VDType]):
    def __call__(
        self,
        vector: VDType,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
    ) -> npt.NDArray[VDType]: ...


@dataclass
class TransformerSeq(_Transformer[Any, np.int64]):
    lower_vectorized: ClassVar[np.int64] = np.int64(0)
    seq: Sequence[Any]

    @property
    def upper_vectorized(self) -> np.int64:
        return np.int64(len(self.seq))

    def to_value(self, vector: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        if not (np.round(vector, ROUND_PLACES) == vector).all():
            raise ValueError(
                "Got unexpected float value while trying to transform a vector"
                f" representation into a value in {self.seq}."
                f"Expected integers but got {vector} (dtype: {vector.dtype})",
            )
        return np.array([self.seq[v] for v in vector.astype(int)])

    def to_vector(self, value: npt.NDArray) -> npt.NDArray[np.int64]:
        return np.array([self.seq.index(v) for v in value], dtype=np.int64)

    def legal_value(self, value: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
        return np.array([v in self.seq for v in value], dtype=np.bool_)

    def legal_vector(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.bool_]:
        return np.where(
            (vector >= 0)
            & (vector < len(self.seq))
            & ~np.isnan(vector)
            & is_close_to_integer(vector, atol=ABS_ROUND_CLOSENESS),
            True,
            False,
        )


class UnitScaler(_Transformer[DType, np.float64]):
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
            value = np.array(
                [
                    v if isinstance(v, (int, float, np.number)) else np.nan
                    for v in value
                ],
            )

        inbounds = (value >= self.lower_value) & (value <= self.upper_value)
        notnan = ~np.isnan(value)
        if np.issubdtype(self.dtype, np.integer):
            return np.where(
                inbounds
                & notnan
                & is_close_to_integer(value, atol=ABS_ROUND_CLOSENESS),
                True,
                False,
            )

        return np.where(inbounds & notnan, True, False)

    def legal_vector(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.bool_]:
        notnan = ~np.isnan(vector)
        in_bounds = (vector >= self.lower_vectorized) & (
            vector <= self.upper_vectorized
        )
        if np.issubdtype(self.dtype, np.integer):
            # NOTE: Unfortunatly for integers, we have to transform back to original
            # space to check if the vector value is indeed close to being an integer.
            # With a non-log spaced vector, we can multiply by the size of the range
            # as a quick check, giving us back integer values. However this does
            # not apply generally to a log-scale vector as the non-linear scaling
            # will not give us back integer values when doing the above.
            values = self.to_value(vector.astype(np.float64))
            return self.legal_value(values) & notnan & in_bounds

        return notnan & in_bounds


@dataclass
class NeighborhoodCat(_Neighborhood[np.int64]):
    n: int

    def __call__(
        self,
        vector: np.int64,
        n: int,
        *,
        std: float | None = None,  # noqa: ARG002
        seed: RandomState | None = None,
    ) -> npt.NDArray[np.int64]:
        seed = np.random.RandomState() if seed is None else seed
        choices = split_arange((0, vector), (vector + 1, self.n))

        if n >= len(choices):
            seed.shuffle(choices)
            return choices

        return seed.choice(choices, n, replace=False)


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
class TransformerConstant(_Transformer[DType, VDType]):
    value: DType
    vector_value_yes: VDType
    vector_value_no: VDType

    @property
    def lower_vectorized(self) -> VDType:
        return self.vector_value_no

    @property
    def upper_vectorized(self) -> VDType:
        return self.vector_value_yes

    def to_vector(
        self,
        value: DType | npt.NDArray[DType],
    ) -> VDType | npt.NDArray[VDType]:
        if isinstance(value, np.ndarray | Sequence):
            return np.where(
                value == self.value,
                self.vector_value_yes,
                self.vector_value_no,
            ).astype(type(self.vector_value_yes))

        return self.vector_value_yes if value == self.value else self.vector_value_no

    def to_value(
        self,
        vector: VDType | npt.NDArray[VDType],
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

    def legal_vector(self, vector: VDType) -> bool:
        return vector == self.vector_value_yes
