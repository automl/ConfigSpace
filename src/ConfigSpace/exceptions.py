from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ConfigSpace.configuration_space import ConfigurationSpace
    from ConfigSpace.hyperparameters import Hyperparameter


class ForbiddenValueError(ValueError):
    """Raised when a combination of values is forbidden for a Configuration."""


class IllegalValueError(ValueError):
    def __init__(self, hyperparameter: Hyperparameter, value: Any):
        super().__init__()
        self.hyperparameter = hyperparameter
        self.value = value

    def __str__(self) -> str:
        return (
            f"Value {self.value}: ({type(self.value)}) is not allowed for"
            f" hyperparameter with name '{self.hyperparameter.name}'"
            f"\n{self.hyperparameter}"
        )


class IllegalVectorizedValueError(ValueError):
    def __init__(self, hyperparameter: Hyperparameter, vector: Any):
        super().__init__()
        self.hyperparameter = hyperparameter
        self.vector = vector

    def __str__(self) -> str:
        return (
            f"Vectorized value '{self.vector}': ({type(self.vector)}) is not allowed"
            f" for hyperparameter with name '{self.hyperparameter.name}'"
            f"\n{self.hyperparameter}"
        )


class ActiveHyperparameterNotSetError(ValueError):
    def __init__(self, hyperparameter: Hyperparameter) -> None:
        super().__init__(hyperparameter)
        self.hyperparameter = hyperparameter

    def __str__(self) -> str:
        return f"Hyperparameter is active but has no value set.\n{self.hyperparameter}"


class InactiveHyperparameterSetError(ValueError):
    def __init__(self, hyperparameter: Hyperparameter, value: Any) -> None:
        super().__init__(hyperparameter)
        self.hyperparameter = hyperparameter
        self.value = value

    def __str__(self) -> str:
        return (
            f"Hyperparameter is inactive but has a value set as {self.value}.\n"
            f"{self.hyperparameter}"
        )


class HyperparameterNotFoundError(ValueError):
    pass


class ChildNotFoundError(HyperparameterNotFoundError):
    pass


class ParentNotFoundError(HyperparameterNotFoundError):
    pass


class HyperparameterIndexError(KeyError):
    def __init__(self, idx: int, space: ConfigurationSpace, *args: Any):
        super().__init__(idx, space, *args)
        self.idx = idx
        self.space = space

    def __str__(self) -> str:
        return (
            f"Hyperparameter #'{self.idx}' does not exist in this space."
            f"\n{self.space}"
        )


class AmbiguousConditionError(ValueError):
    pass


class HyperparameterAlreadyExistsError(ValueError):
    pass


class CyclicDependancyError(ValueError):
    pass


class NoPossibleNeighborsError(ValueError):
    pass
