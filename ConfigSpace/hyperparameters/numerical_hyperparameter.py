from __future__ import annotations

from dataclasses import field
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

NumberDType = TypeVar("NumberDType", bound=np.number)


@runtime_checkable
class NumericalHyperparameter(Protocol[NumberDType]):
    lower: NumberDType = field(hash=True)
    upper: NumberDType = field(hash=True)
    log: bool = field(hash=True)
