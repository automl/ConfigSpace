from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ConfigSpace.conditions import ConditionComponent
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
            f" hyperparameter {self.hyperparameter}"
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
    def __init__(
        self,
        hyperparameter: Hyperparameter | str,
        space: ConfigurationSpace,
        preamble: str | None = None,
    ):
        super().__init__(hyperparameter, space, preamble)
        self.preamble = preamble
        self.hp_name = hyperparameter if isinstance(hyperparameter, str) else hyperparameter.name
        self.space = space

    def __str__(self) -> str:
        pre = f"{self.preamble}\n" if self.preamble is not None else ""
        return f"{pre}" f"Hyperparameter {self.hp_name} not found in space." f"\n{self.space}"


class ChildNotFoundError(HyperparameterNotFoundError):
    def __str__(self) -> str:
        return "Child " + super().__str__()


class ParentNotFoundError(HyperparameterNotFoundError):
    def __str__(self) -> str:
        return "Parent " + super().__str__()


class HyperparameterIndexError(KeyError):
    def __init__(self, idx: int, space: ConfigurationSpace):
        super().__init__(idx, space)
        self.idx = idx
        self.space = space

    def __str__(self) -> str:
        raise KeyError(
            f"Hyperparameter #'{self.idx}' does not exist in this space." f"\n{self.space}",
        )


class AmbiguousConditionError(ValueError):
    def __init__(self, present: ConditionComponent, new_condition: ConditionComponent):
        super().__init__(present, new_condition)
        self.present = present
        self.new_condition = new_condition

    def __str__(self) -> str:
        return (
            "Adding a second condition (different) for a hyperparameter is ambiguous"
            " and therefore forbidden. Add a conjunction instead!"
            f"\nAlready inserted: {self.present}"
            f"\nNew one: {self.new_condition}"
        )


class HyperparameterAlreadyExistsError(ValueError):
    def __init__(
        self,
        existing: Hyperparameter,
        other: Hyperparameter,
        space: ConfigurationSpace,
    ):
        super().__init__(existing, other, space)
        self.existing = existing
        self.other = other
        self.space = space

    def __str__(self) -> str:
        return (
            f"Hyperparameter {self.existing.name} already exists in space."
            f"\nExisting: {self.existing}"
            f"\nNew one: {self.other}"
            f"{self.space}"
        )


class CyclicDependancyError(ValueError):
    def __init__(self, cycles: list[list[str]]) -> None:
        super().__init__(cycles)
        self.cycles = cycles

    def __str__(self) -> str:
        return f"Hyperparameter configuration contains a cycle {self.cycles}"
