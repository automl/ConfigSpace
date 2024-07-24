from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from typing_extensions import override

import numpy as np

from ConfigSpace.functional import (
    is_close_to_integer,
    is_close_to_integer_single,
    normalize,
    scale,
)
from ConfigSpace.types import DType, ObjectArray, f64, i64

if TYPE_CHECKING:
    from ConfigSpace.types import Array, Mask

RandomState = np.random.RandomState

CONSTANT_VECTOR_VALUE = i64(0)

# NOTE: Beyond this point, tests start to fail on equality checks
# due to transforms. This seems to be a relatively stable point
ROUND_PLACES = 13
ATOL = 1e-13

T_contra = TypeVar("T_contra", contravariant=True)


class Transformer(Protocol[DType]):
    """Protocol for a transformer.

    This protocol defines how to move from **vectorized** representation to
    **value** representation and vice versa. It also defines how to check if
    a value or vector is legal.

    With this it also include
    [`.lower_vectorized`][`ConfigSpace.hp_components.Transformer.lower_vectorized`]
    and
    [`.upper_vectorized`][`ConfigSpace.hp_components.Transformer.upper_vectorized`]
    which defines the upper and lower bounds of the vectorized representation.

    !!! note

        All vectorized representations should be of type `np.float64`, even
        if they are integer values. This helps with calculations and being
        able to store configurations in a single array.
    """

    lower_vectorized: f64
    """Lower bound of the vectorized representation."""

    upper_vectorized: f64
    """Upper bound of the vectorized representation."""

    def to_value(self, vector: Array[f64]) -> Array[DType]:
        """Transform a vector representation into its value representation."""
        ...

    def to_vector(self, value: Array[DType]) -> Array[f64]:
        """Transform a value representation into its vector representation."""
        ...

    def legal_value(self, value: Array[DType]) -> Mask:
        """Returns a boolean mask of which values are legal."""
        ...

    def legal_vector(self, vector: Array[f64]) -> Mask:
        """Returns a boolean mask of which vectors are legal."""
        ...

    def legal_vector_single(self, vector: np.number) -> bool:
        """Check if a single vector value is legal.

        This is used as an optimization instead of having to wrap
        a single vector value into an array and then unpack it.
        """
        ...

    def legal_value_single(self, value: np.number) -> bool:
        """Check if a single value is legal.

        This is used as an optimization instead of having to wrap
        a single value into an array and then unpack it.
        """
        ...


class Neighborhood(Protocol):
    """Protocol for a neighborhood function.

    This protocol defines how to get the neighborhood of a value in a
    vectorized representation. This is used for the `neighborhood=` argument
    in the `ConfigSpace.hyperparameters.Hyperparameter` class, letting the
    hyperparameter know how to get the neighborhood of a vectorized value.
    """

    def __call__(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        """Get the neighborhood of a vectorized value.

        Args:
            vector: The vectorized value to get the neighborhood of.
            n: The number of neighbors to get.
            std: The standard deviation of the neighborhood.
            seed: The seed for the random number generator.
        """
        ...


@dataclass
class TransformerSeq(Transformer[Any]):
    """Implmentation of a transformer for a sequence of values.

    This uses an integer range from 0 to `len(seq) - 1` to represent the
    sequence of values in vectorized space.

    This is useful primarily for categorical and ordinal hyperparameters.

    Args:
        seq: The sequence of values to transform.
    """

    seq: Array[Any] | list[Any]  # If `list`, assumed to contain sequence objects
    """The original sequence of values."""

    lower_vectorized: f64 = field(init=False)
    """Lower bound of the vectorized representation.

    Always 0.
    """

    upper_vectorized: f64 = field(init=False)
    """Upper bound of the vectorized representation.

    Always `len(seq) - 1`.
    """

    _lookup: dict[Any, int] | None = field(init=False)

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

    @override
    def to_value(self, vector: Array[f64]) -> Array[Any]:
        if not is_close_to_integer(vector, atol=ATOL).all():
            raise ValueError(
                "Got unexpected float value while trying to transform a vector"
                f" representation into a value in {self.seq}."
                f"Expected integers but got {vector} (dtype: {vector.dtype})",
            )

        if isinstance(self.seq, np.ndarray):
            indices = np.rint(vector).astype(i64)
            return self.seq[indices]

        items = [self.seq[int(np.rint(i))] for i in vector]
        if isinstance(self.seq, list):
            # We have to convert it into a numpy array of objects carefully
            # https://stackoverflow.com/a/47389566/5332072
            _v = np.empty(len(items), dtype=object)
            _v[:] = items
            return _v

        return np.array(items, dtype=object)

    @override
    def to_vector(self, value: Array[Any]) -> Array[f64]:
        if self._lookup is not None:
            return np.array([self._lookup[v] for v in value], dtype=f64)

        if isinstance(self.seq, np.ndarray):
            return np.flatnonzero(np.isin(self.seq, value)).astype(f64)

        return np.array([self.seq.index(v) for v in value], dtype=f64)

    @override
    def legal_value(self, value: Array[Any]) -> Mask:
        if self._lookup is not None:
            return np.array([v in self._lookup for v in value], dtype=np.bool_)

        if isinstance(self.seq, np.ndarray):
            return np.isin(value, self.seq)

        return np.array([v in self.seq for v in value], dtype=np.bool_)

    @override
    def legal_vector(self, vector: Array[f64]) -> Mask:
        return (
            (vector >= 0)
            & (vector < len(self.seq))
            & is_close_to_integer(vector, atol=ATOL)
        )

    @override
    def legal_value_single(self, value: Any) -> bool:
        if self._lookup is not None:
            return value in self._lookup
        return value in self.seq

    @override
    def legal_vector_single(self, vector: np.number) -> bool:
        return bool(
            vector >= 0
            and vector < len(self.seq)
            and is_close_to_integer_single(vector, atol=ATOL),
        )


@dataclass
class UnitScaler(Transformer[DType]):
    """Implementation of a transformer from a vectorized continuous range `(0, 1)`
    to another specified range in value space.

    Args:
        lower_value: Lower bound of the range.
        upper_value: Upper bound of the range.
        dtype: The type of the values in the range.
        log: Whether to use a log scale or not.
    """

    lower_value: DType
    """Lower bound of the value range."""

    upper_value: DType
    """Upper bound of the value range."""

    dtype: type[DType]
    """What type we should return when transforming to value space."""

    log: bool = False
    """Whether to use a log scale or not."""

    lower_vectorized: f64 = field(init=False)
    """Lower bound of the vectorized representation.

    Always 0.
    """

    upper_vectorized: f64 = field(init=False)
    """Lower bound of the vectorized representation.

    Always 1.
    """

    _size: f64 = field(init=False)

    # NOTE(eddiebergman): This is required as it's easy to overflow on Windows
    # when normalizing or performing other operations on large boundaries.
    _lower_value_f64: f64 = field(init=False)
    _upper_value_f64: f64 = field(init=False)

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
        self._lower_value_f64 = f64(self.lower_value)
        self._upper_value_f64 = f64(self.upper_value)
        self._size = self._upper_value_f64 - self._lower_value_f64

    @override
    def to_value(self, vector: Array[f64]) -> Array[DType]:
        """Transform a value from the unit interval to the range."""
        unchecked_values = self._unsafe_to_value(vector)
        if np.issubdtype(self.dtype, np.integer):
            return unchecked_values.round().astype(self.dtype)

        return np.clip(  # type: ignore
            unchecked_values,
            self._lower_value_f64,
            self._upper_value_f64,
            dtype=self.dtype,
        )

    def _unsafe_to_value_single(self, vector: f64) -> f64:
        # NOTE: Unsafe as it does not check boundaries, clip or integer'ness
        # linear (0-1) space to log scaling (0-1)
        if self.log:
            _l = np.log(self.lower_value)
            _u = np.log(self.upper_value)
            scaled = vector * (_u - _l) + _l
            return np.exp(scaled)  # type: ignore

        return vector * self._size + _l  # type: ignore

    def _unsafe_to_value(self, vector: Array[f64]) -> Array[f64]:
        # NOTE: Unsafe as it does not check boundaries, clip or integer'ness
        # linear (0-1) space to log scaling (0-1)
        if self.log:
            scaled_to_log_bounds = scale(
                vector,
                to=(np.log(self._lower_value_f64), np.log(self._upper_value_f64)),
            )
            return np.exp(scaled_to_log_bounds)

        return scale(vector, to=(self._lower_value_f64, self._upper_value_f64))

    @override
    def to_vector(self, value: Array[DType]) -> Array[f64]:
        if self.log:
            return normalize(
                np.log(value),
                bounds=(np.log(self._lower_value_f64), np.log(self._upper_value_f64)),
            )

        return normalize(value, bounds=(self._lower_value_f64, self._upper_value_f64))

    def vectorize_size(self, size: f64) -> f64:
        """Vectorize to the correct scale but is not necessarily in the range.

        Mainly useful for scaling things like `std` in value space to it's equivalent
        in vector space.
        """
        if self.log:
            return np.abs(  # type: ignore
                np.log(self._lower_value_f64 + size)
                / (np.log(self._upper_value_f64) - np.log(self._lower_value_f64)),
            )

        return f64(size / self._size)

    @override
    def legal_value(self, value: Array[Any]) -> Mask:
        # If we have a non numeric dtype, we have to unfortunatly go through but by bit
        if value.dtype.kind not in "iuf":
            return np.array([self.legal_value_single(v) for v in value], dtype=np.bool_)

        if np.issubdtype(self.dtype, np.integer):
            rints = np.rint(value)
            return (  # type: ignore
                (rints >= self._lower_value_f64)
                & (rints <= self._upper_value_f64)
                & is_close_to_integer(value, atol=ATOL)
            )

        return (value >= self._lower_value_f64) & (value <= self._upper_value_f64)

    @override
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

    @override
    def legal_value_single(self, value: Any) -> bool:
        if not isinstance(value, (int, float, np.number)):
            return False

        if np.issubdtype(self.dtype, np.integer):
            rint = np.rint(value)
            return bool(
                (self._lower_value_f64 <= rint)
                & (rint <= self._upper_value_f64)
                & is_close_to_integer_single(value, atol=ATOL),
            )

        return bool(self._lower_value_f64 <= value <= self._upper_value_f64)

    @override
    def legal_vector_single(self, vector: np.number) -> bool:
        if not np.issubdtype(self.dtype, np.integer):
            return bool(self.lower_vectorized <= vector <= self.upper_vectorized)

        if not self.log:
            inbounds = bool(self.lower_vectorized <= vector <= self.upper_vectorized)
            scaled = vector * self._size
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
    """Get the neighborhood of a vectorized ordinal value.

    This is used for the `neighborhood=` argument in the
    [`ConfigSpace.hyperparameters.OrdinalHyperparameter`][ConfigSpace.hyperparameters.OrdinalHyperparameter]

    Args:
        vector: The vectorized value to get the neighborhood of.

            !!! warning

                This is assumed to be an integer in the range `[0, size - 1]`.

        n: The number of neighbors to get.
        size: The size of the sequence.
        std: The standard deviation of the neighborhood.
        seed: The seed for the random number generator.

    Returns:
        The neighborhood of the vectorized value.
    """
    end_index = size - 1
    assert 0 <= vector <= end_index

    # No neighbors if it's the only element
    if size == 1:
        return np.array([], dtype=f64)

    # We have at least 2 elements,
    # in this case it's only neighbor is the one beside it
    # which is itself +1
    if vector == 0:
        return np.array([1.0], dtype=f64)

    # Also only one neighbor for the other end
    if np.rint(vector) == end_index:
        return np.array([end_index - 1], dtype=f64)

    # We have at least 3 elements and the value is not at the ends
    neighbors: Array[f64] = np.rint([vector - 1, vector + 1], dtype=f64)
    if n >= 2:
        return neighbors

    seed = np.random.RandomState() if seed is None else seed
    return np.array([seed.choice(neighbors)], dtype=f64)


# HACK: Technically `Any` isn't an `np.number` that the Transformer expects
# as it's type variable. However for a Constant, we can like with this typing
# hack.
@dataclass
class TransformerConstant(Transformer[Any]):
    """Implementation of a transformer for a constant value."""

    value: Any
    """The constant value."""

    vector_value_yes: f64
    """The vectorized value for the constant value.

    This is the value that represents the constant value and
    is asserted to be greater than the `vector_value_no`.
    """

    vector_value_no: f64
    """The vectorized value for anything but the constant value.

    This is the value that represents anything but the constant value
    and is asserted to be less than the `vector_value_yes`.
    """

    lower_vectorized: f64 = field(init=False)
    """Lower bound of the vectorized representation."""

    upper_vectorized: f64 = field(init=False)
    """Upper bound of the vectorized representation."""

    def __post_init__(self) -> None:
        assert self.vector_value_yes > self.vector_value_no
        self.lower_vectorized = self.vector_value_no
        self.upper_vectorized = self.vector_value_yes

    @override
    def to_vector(self, value: ObjectArray) -> Array[f64]:
        if isinstance(self.value, np.ndarray):
            return np.flatnonzero(np.equal(self.value, value)).astype(f64)

        return np.array([v == self.value for v in value], dtype=f64)

    @override
    def to_value(self, vector: Array[f64]) -> ObjectArray:
        if isinstance(self.value, Sequence) and not isinstance(self.value, str):
            # We have to convert it into a numpy array of objects carefully
            # https://stackoverflow.com/a/47389566/5332072
            _v = np.empty(len(vector), dtype=object)
            _v[:] = [self.value] * len(vector)
            return _v
        return np.full_like(vector, self.value, dtype=object)

    @override
    def legal_value(self, value: ObjectArray) -> Mask:
        return np.array([v == self.value for v in value], dtype=np.bool_)

    @override
    def legal_vector(self, vector: Array[f64]) -> Mask:
        return vector == self.vector_value_yes  # type: ignore

    @override
    def legal_value_single(self, value: Any) -> bool:
        return value == self.value  # type: ignore

    @override
    def legal_vector_single(self, vector: np.number) -> bool:
        return vector == self.vector_value_yes  # type: ignore
