from __future__ import annotations

from typing import Final
from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt

Vector: TypeAlias = npt.NDArray[np.float64]
"""Vectorized representation of a configuration."""


class _NotSet:
    def __repr__(self):
        return "ValueNotSetObject"


NotSet: Final = _NotSet()  # Sentinal value for unset values
