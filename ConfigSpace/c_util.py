from __future__ import annotations

from collections import deque

import numpy as np

from ConfigSpace.conditions import ConditionComponent, OrConjunction
from ConfigSpace.exceptions import (
    ActiveHyperparameterNotSetError,
    ForbiddenValueError,
    IllegalValueError,
    InactiveHyperparameterSetError,
)
from ConfigSpace.forbidden import AbstractForbiddenComponent
from ConfigSpace.hyperparameters import Hyperparameter
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


def check_forbidden(forbidden_clauses: list, vector: np.ndarray) -> int:
    Iforbidden: int = len(forbidden_clauses)
    clause: AbstractForbiddenComponent

    for i in range(Iforbidden):
        clause = forbidden_clauses[i]
        if clause.c_is_forbidden_vector(vector, strict=False):
            raise ForbiddenValueError("Given vector violates forbidden clause %s" % (str(clause)))


def check_configuration(
    self,
    vector: np.ndarray,
    allow_inactive_with_values: bool,
) -> int:
    hp_name: str
    hyperparameter: Hyperparameter
    hyperparameter_idx: int
    hp_value: float | int
    add: int
    condition: ConditionComponent
    child: Hyperparameter
    conditions: list
    children: list
    inactive: set
    visited: set

    active: np.ndarray = np.zeros(len(vector), dtype=int)

    unconditional_hyperparameters = self.get_all_unconditional_hyperparameters()
    to_visit = deque()
    visited = set()
    to_visit.extendleft(unconditional_hyperparameters)
    inactive = set()

    for ch in unconditional_hyperparameters:
        active[self._hyperparameter_idx[ch]] = 1

    while len(to_visit) > 0:
        hp_name = to_visit.pop()
        visited.add(hp_name)
        hp_idx = self._hyperparameter_idx[hp_name]
        hyperparameter = self._hyperparameters[hp_name]
        hp_value = vector[hp_idx]

        if not np.isnan(hp_value) and not hyperparameter.is_legal_vector(hp_value):
            raise IllegalValueError(hyperparameter, hp_value)

        children = self._children_of[hp_name]
        for child in children:
            if child.name not in inactive:
                conditions = self._parent_conditions_of[child.name]
                add = True
                for condition in conditions:
                    if not condition._evaluate_vector(vector):
                        add = False
                        inactive.add(child.name)
                        break
                if add:
                    hyperparameter_idx = self._hyperparameter_idx[child.name]
                    active[hyperparameter_idx] = 1
                    to_visit.appendleft(child.name)

        if active[hp_idx] and np.isnan(hp_value):
            raise ActiveHyperparameterNotSetError(hyperparameter)

    for hp_idx in self._idx_to_hyperparameter:
        if not allow_inactive_with_values and not active[hp_idx] and not np.isnan(vector[hp_idx]):
            # Only look up the value (in the line above) if the hyperparameter is inactive!
            hp_name = self._idx_to_hyperparameter[hp_idx]
            hp_value = vector[hp_idx]
            raise InactiveHyperparameterSetError(hyperparameter, hp_value)

    self._check_forbidden(vector)


def correct_sampled_array(
    vector: np.ndarray,
    forbidden_clauses_unconditionals: list,
    forbidden_clauses_conditionals: list,
    hyperparameters_with_children: list,
    num_hyperparameters: int,
    unconditional_hyperparameters: list,
    hyperparameter_to_idx: dict,
    parent_conditions_of: dict,
    parents_of: dict,
    children_of: dict,
) -> np.ndarray:
    clause: AbstractForbiddenComponent
    condition: ConditionComponent
    hyperparameter_idx: int
    NaN: float = np.NaN
    visited: set
    inactive: set
    child: Hyperparameter
    children: list
    child_name: str
    parents: list
    parent: Hyperparameter
    parents_visited: int
    conditions: list
    add: int

    active: np.ndarray = np.zeros(len(vector), dtype=int)

    for j in range(len(forbidden_clauses_unconditionals)):
        clause = forbidden_clauses_unconditionals[j]
        if clause.c_is_forbidden_vector(vector, strict=False):
            msg = "Given vector violates forbidden clause %s" % str(clause)
            raise ForbiddenValueError(msg)

    hps = deque()
    visited = set()
    hps.extendleft(hyperparameters_with_children)

    for ch in unconditional_hyperparameters:
        active[hyperparameter_to_idx[ch]] = 1

    inactive = set()

    while len(hps) > 0:
        hp = hps.pop()
        visited.add(hp)
        children = children_of[hp]
        for child in children:
            child_name = child.name
            if child_name not in inactive:
                parents = parents_of[child_name]
                hyperparameter_idx = hyperparameter_to_idx[child_name]
                if len(parents) == 1:
                    conditions = parent_conditions_of[child_name]
                    add = True
                    for j in range(len(conditions)):
                        condition = conditions[j]
                        if not condition._evaluate_vector(vector):
                            add = False
                            vector[hyperparameter_idx] = NaN
                            inactive.add(child_name)
                            break
                    if add is True:
                        active[hyperparameter_idx] = 1
                        hps.appendleft(child_name)

                else:
                    parents_visited = 0
                    for parent in parents:
                        if parent.name in visited:
                            parents_visited += 1
                    if parents_visited > 0:  # make sure at least one parent was visited
                        conditions = parent_conditions_of[child_name]
                        if isinstance(conditions[0], OrConjunction):
                            pass
                        else:  # AndCondition
                            if parents_visited != len(parents):
                                continue

                        add = True
                        for j in range(len(conditions)):
                            condition = conditions[j]
                            if not condition._evaluate_vector(vector):
                                add = False
                                vector[hyperparameter_idx] = NaN
                                inactive.add(child_name)
                                break

                        if add is True:
                            active[hyperparameter_idx] = 1
                            hps.appendleft(child_name)

                    else:
                        continue

    for j in range(len(vector)):
        if not active[j]:
            vector[j] = NaN

    for j in range(len(forbidden_clauses_conditionals)):
        clause = forbidden_clauses_conditionals[j]
        if clause.c_is_forbidden_vector(vector, strict=False):
            msg = "Given vector violates forbidden clause %s" % str(clause)
            raise ForbiddenValueError(msg)

    return vector


def change_hp_value(
    configuration_space,
    configuration_array: np.ndarray,
    hp_name: str,
    hp_value: float,
    index: int,
) -> np.ndarray:
    """Change hyperparameter value in configuration array to given value.

    Does not check if the new value is legal. Activates and deactivates other
    hyperparameters if necessary. Does not check if new hyperparameter value
    results in the violation of any forbidden clauses.

    Parameters
    ----------
    configuration_space : ConfigurationSpace

    configuration_array : np.ndarray

    hp_name : str

    hp_value : float

    index : int

    Returns
    -------
    np.ndarray
    """
    current: Hyperparameter
    current_name: str
    disabled: list
    hps_to_be_activate: set
    visited: set
    active: int
    condition: ConditionComponent
    current_idx: int
    current_value: float
    default_value: float
    children: list
    children_: list
    ch: Hyperparameter
    child: str
    to_disable: set
    NaN: float = np.NaN
    children_of: dict = configuration_space._children_of

    configuration_array[index] = hp_value

    # Hyperparameters which are going to be set to inactive
    disabled = []

    # Hyperparameters which are going to be set activate, we introduce this to resolve the conflict that might be raised
    # by OrConjunction:
    # Suppose that we have a parent HP_p whose possible values are A, B, C; a child HP_d is activate if
    # HP_p is A or B. Then when HP_p switches from A to B, HP_d needs to remain activate.
    hps_to_be_activate = set()

    # Activate hyperparameters if their parent node got activated
    children = children_of[hp_name]
    if len(children) > 0:
        to_visit = deque()  # type: deque
        to_visit.extendleft(children)
        visited = set()  # type: Set[str]

        while len(to_visit) > 0:
            current = to_visit.pop()
            current_name = current.name
            if current_name in visited:
                continue
            visited.add(current_name)
            if current_name in hps_to_be_activate:
                continue

            current_idx = configuration_space._hyperparameter_idx[current_name]
            current_value = configuration_array[current_idx]

            conditions = configuration_space._parent_conditions_of[current_name]

            active = True
            for condition in conditions:
                if not condition._evaluate_vector(configuration_array):
                    active = False
                    break

            if active:
                hps_to_be_activate.add(current_idx)
                if current_value == current_value:
                    children_ = children_of[current_name]
                    if len(children_) > 0:
                        to_visit.extendleft(children_)

            if current_name in disabled:
                continue

            if active and current_value != current_value:
                default_value = current.normalized_default_value
                configuration_array[current_idx] = default_value
                children_ = children_of[current_name]
                if len(children_) > 0:
                    to_visit.extendleft(children_)

            # If the hyperparameter was made inactive,
            # all its children need to be deactivade as well
            if not active and current_value == current_value:
                configuration_array[current_idx] = NaN

                children = children_of[current_name]

                if len(children) > 0:
                    to_disable = set()
                    for ch in children:
                        to_disable.add(ch.name)
                    while len(to_disable) > 0:
                        child = to_disable.pop()
                        child_idx = configuration_space._hyperparameter_idx[child]
                        disabled.append(child_idx)
                        children = children_of[child]

                        for ch in children:
                            to_disable.add(ch.name)

    for idx in disabled:
        if idx not in hps_to_be_activate:
            configuration_array[idx] = NaN

    return configuration_array
