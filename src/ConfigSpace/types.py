from __future__ import annotations

from typing import Final, TypeVar, Union
from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt

Mask: TypeAlias = npt.NDArray[np.bool_]
"""Mask, a numpy array of bools."""

ValueT = TypeVar("ValueT")
"""Some value type.

This represents a single value from the value space,
not contained within some numpy array, for example
a raw `int`.
"""

DType = TypeVar("DType", bound=np.number)
"""Some numpy number type.

This represents a numpy array of values from the value space,
for example a `np.int64`.
"""

Array: TypeAlias = npt.NDArray[DType]
"""Array, a numpy array of a specific dtype."""

ObjectArray: TypeAlias = npt.NDArray[np.object_]
"""Object array, a numpy array of objects."""

f64: TypeAlias = np.float64
"""64-bit floating point number."""

i64: TypeAlias = np.int64
"""64-bit integer."""

Number: TypeAlias = Union[int, float, np.number]
"""Number, an integer, float, or numpy number."""


class _NotSet:
    def __repr__(self) -> str:
        return "ValueNotSetObject"


NotSet: Final = _NotSet()  # Sentinal value for unset values
"""Sentinal value for unset values.

This is useful in cases where `None` is a valid value and should not be used to
indicate that something was not set.
"""
