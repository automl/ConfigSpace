from __future__ import annotations

from typing import Final, TypeVar, Union
from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt

Mask: TypeAlias = npt.NDArray[np.bool_]
"""Mask, vector of bools."""

DType = TypeVar("DType", bound=np.number)

Array: TypeAlias = npt.NDArray[DType]

f64: TypeAlias = np.float64
i64: TypeAlias = np.int64

Number: TypeAlias = Union[int, float, np.number]


class _NotSet:
    def __repr__(self) -> str:
        return "ValueNotSetObject"


NotSet: Final = _NotSet()  # Sentinal value for unset values
