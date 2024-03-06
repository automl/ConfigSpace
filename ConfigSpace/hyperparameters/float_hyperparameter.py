from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter
from ConfigSpace.hyperparameters.numerical_hyperparameter import NumericalHyperparameter


@dataclass(init=False)
class FloatHyperparameter(
    Hyperparameter[np.float64, np.float64],
    NumericalHyperparameter[np.float64],
):
    pass
