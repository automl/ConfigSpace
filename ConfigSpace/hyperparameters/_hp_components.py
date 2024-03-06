from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from ConfigSpace.functional import split_arange

RandomState = np.random.RandomState

ROUND_PLACES = 9
VECTORIZED_NUMERIC_LOWER = 0.0
VECTORIZED_NUMERIC_UPPER = 1.0
DEFAULT_VECTORIZED_NUMERIC_STD = 0.2
CONSTANT_VECTOR_VALUE = np.int64(0)

DType = TypeVar("DType", bound=np.number)
"""Type variable for the data type of the hyperparameter."""

VDType = TypeVar("VDType", bound=np.number)
"""Type variable for the data type of the vectorized hyperparameter."""

T_contra = TypeVar("T_contra", contravariant=True)


class _Transformer(Protocol[DType, VDType]):
    def to_value(self, vector: npt.NDArray[VDType]) -> npt.NDArray[DType]: ...

    def to_vector(
        self,
        value: Sequence[DType] | npt.NDArray[DType],
    ) -> npt.NDArray[VDType]: ...


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
class TransformerSeq(_Transformer[DType, np.int64]):
    seq: Sequence[DType]

    def to_value(self, vector: npt.NDArray[np.int64]) -> DType | Sequence[DType]:
        return self.seq[vector]

    def to_vector(
        self,
        value: Sequence[DType] | npt.NDArray,
    ) -> npt.NDArray[np.int64]:
        return np.array([self.seq.index(v) for v in value], dtype=np.int64)


class UnitScaler(_Transformer[DType, np.float64]):
    def __init__(
        self,
        low: float | int | np.number,
        high: float | int | np.number,
        *,
        log: bool = False,
    ):
        if low >= high:
            raise ValueError(
                f"Upper bound {high:f} must be larger than lower bound {low:f}",
            )

        if log and low <= 0:
            raise ValueError(
                f"Negative lower bound {low:f} for log-scale is not possible.",
            )

        self.low = low
        self.high = high
        self.log = log
        self.diff = high - low

    def to_value(
        self,
        vector: npt.NDArray[np.float64],
    ) -> npt.NDArray[DType]:
        """Transform a value from the unit interval to the range."""
        # linear (0-1) space to log scaling (0-1)
        if self.log:
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            x = vector * (log_high - log_low) + log_low
            x = np.exp(x)
        else:
            x = vector * (self.high - self.low) + self.low

        return np.clip(x, self.low, self.high, dtype=np.float64)

    def to_vector(self, value: npt.NDArray[DType]) -> npt.NDArray[np.float64]:
        """Transform a value from the range to the unit interval."""
        x = value
        if self.log:
            x = (np.log(x) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        else:
            x = (x - self.low) / (self.high - self.low)

        return np.clip(x, 0.0, 1.0, dtype=np.float64)


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
        choices = split_arange((0, vector), (vector, self.n))
        seed = np.random.RandomState() if seed is None else seed
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
class TransformerConstant(_Transformer[DType, np.integer]):
    value: DType

    def to_vector(
        self,
        value: DType | npt.NDArray[DType],
    ) -> np.integer | npt.NDArray[np.integer]:
        if isinstance(value, np.ndarray | Sequence):
            return np.full_like(value, CONSTANT_VECTOR_VALUE, dtype=np.integer)

        return CONSTANT_VECTOR_VALUE

    def to_value(
        self,
        vector: np.integer | npt.NDArray[np.integer],
    ) -> DType | npt.NDArray[DType]:
        if isinstance(vector, np.ndarray):
            try:
                return np.full_like(vector, self.value, dtype=type(self.value))
            except TypeError:
                # Let numpy figure it out
                return np.array([self.value] * len(vector))

        return self.value
