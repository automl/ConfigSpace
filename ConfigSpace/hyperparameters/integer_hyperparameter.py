from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter
from ConfigSpace.hyperparameters.numerical_hyperparameter import NumericalHyperparameter


@dataclass(init=False)
class IntegerHyperparameter(
    Hyperparameter[np.int64, np.float64],
    NumericalHyperparameter[np.int64],
):
    def _neighborhood_size(self, value: np.int64 | None) -> int:
        if value is None or self.lower <= value <= self.upper:
            return int(self.size)
        return int(self.size) - 1
