from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from ConfigSpace.exceptions import (
    ActiveHyperparameterNotSetError,
    ForbiddenValueError,
    IllegalVectorizedValueError,
    InactiveHyperparameterSetError,
)
from ConfigSpace.hyperparameters import Hyperparameter

if TYPE_CHECKING:
    from ConfigSpace.conditions import Condition
    from ConfigSpace.configuration_space import ConfigurationSpace
    from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


def topological_sort(dependancy_graph: dict[str, list[Hyperparameter]]) -> deque[str]:
    # https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    # Assumptions:
    # key (node): value (things node depends on)
    # * All nodes are given as a key, even if they have no dependancies (empty value)
    # * All nodes are unique
    # * No cycles
    # ----
    # An example case to think about it
    # - D: [B, C]
    # - A: []
    # - B: [A]
    # - C: [A]
    marked: set[str] = set()
    L: deque[str] = deque()

    def visit(node: str) -> None:
        if node in marked:
            return
        marked.add(node)
        for successor in dependancy_graph[node]:
            visit(successor.name)
        L.append(node)

    for node in dependancy_graph:
        visit(node)

    return L


def check_configuration(
    space: ConfigurationSpace,
    vector: np.ndarray,
    allow_inactive_with_values: bool = False,
) -> None:
    for hp_name, conditions_effecting_hp in space._conditional_dep_graph.items():
        print(hp_name, conditions_effecting_hp)
        print(space._parents_of[hp_name])
        hp_idx = space._hyperparameter_idx[hp_name]
        hp = space._hyperparameters[hp_name]
        hp_vector_val = vector[hp_idx]
        is_active = ~np.isnan(hp_vector_val)

        if is_active and not hp.legal_vector(hp_vector_val):
            raise IllegalVectorizedValueError(hp, hp_vector_val)

        # If all conditions pass, then the hyperparameter should remain active
        should_be_active = True
        for condition in conditions_effecting_hp:
            if not condition.satisfied_by_vector(vector):
                if not allow_inactive_with_values:
                    raise InactiveHyperparameterSetError(hp, hp_vector_val)
                should_be_active = False

        # If all condition checks above are satisfied, then the hyperparameter
        # should be active
        if should_be_active and not is_active:
            raise ActiveHyperparameterNotSetError(hp)

    for clause in space.forbidden_clauses:
        if clause.is_forbidden_vector(vector):
            raise ForbiddenValueError(
                f"Given vector violates forbidden clause {clause}",
            )


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

    Returns:
    -------
    np.ndarray
    """
    current: Hyperparameter
    current_name: str
    disabled: list
    hps_to_be_activate: set
    visited: set
    active: int
    condition: Condition
    current_idx: int
    current_value: float
    default_value: float
    children: list
    children_: list
    ch: Hyperparameter
    child: str
    to_disable: set
    NaN: float = np.nan
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
                if condition.satisfied_by_vector(configuration_array) is False:
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
