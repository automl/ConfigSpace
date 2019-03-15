# cython: language_level=3

from collections import deque

import numpy as np
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

from libc.stdlib cimport malloc, free

from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.forbidden import AbstractForbiddenComponent

from ConfigSpace.forbidden import AbstractForbiddenComponent
from ConfigSpace.forbidden cimport AbstractForbiddenComponent
from ConfigSpace.hyperparameters import Hyperparameter
from ConfigSpace.hyperparameters cimport Hyperparameter
from ConfigSpace.conditions import ConditionComponent
from ConfigSpace.conditions cimport ConditionComponent


cpdef int check_forbidden(list forbidden_clauses, np.ndarray vector) except 1:
    cdef int I = len(forbidden_clauses)
    cdef AbstractForbiddenComponent clause

    for i in range(I):
        clause = forbidden_clauses[i]
        if clause.c_is_forbidden_vector(vector, strict=False):
            raise ForbiddenValueError("Given vector violates forbidden clause %s" % (str(clause)))


cpdef int check_configuration(
    self,
    np.ndarray vector,
    bint allow_inactive_with_values
) except 1:
    cdef str hp_name
    cdef int hp_index
    cdef Hyperparameter hyperparameter
    cdef int hyperparameter_idx
    cdef DTYPE_t hp_value
    cdef int add
    cdef ConditionComponent condition
    cdef Hyperparameter child
    cdef list conditions
    cdef list parents
    cdef list children
    cdef set inactive

    cdef int* active
    active = <int*> malloc(sizeof(int) * len(vector))
    for i in range(len(vector)):
        active[i] = 0

    unconditional_hyperparameters = self.get_all_unconditional_hyperparameters()
    to_visit = deque()
    to_visit.extendleft(unconditional_hyperparameters)
    inactive = set()

    for ch in unconditional_hyperparameters:
        active[self._hyperparameter_idx[ch]] = 1

    while len(to_visit) > 0:
        hp_name = to_visit.pop()
        hp_idx = self._hyperparameter_idx[hp_name]
        hyperparameter = self._hyperparameters[hp_name]
        hp_value = vector[hp_idx]

        if not np.isnan(hp_value) and not hyperparameter.is_legal_vector(hp_value):
            free(active)
            raise ValueError("Hyperparameter instantiation '%s' "
                             "(type: %s) is illegal for hyperparameter %s" %
                             (hp_value, str(type(hp_value)),
                              hyperparameter))

        children = self._children_of[hp_name]
        for child in children:
            if child.name not in inactive:
                parents = self._parents_of[child.name]
                if len(parents) == 1:
                    conditions = self._parent_conditions_of[child.name]
                    add = True
                    for condition in conditions:
                        if not condition._evaluate_vector(vector):
                            add = False
                            inactive.add(child.name)
                            break
                    if add:
                        hyperparameter_idx = self._hyperparameter_idx[
                            child.name]
                        active[hyperparameter_idx] = 1
                        to_visit.appendleft(child.name)

                else:
                    parent_names = set([p.name for p in parents])
                    if not parent_names <= set(to_visit):  # make sure no parents are still unvisited
                        conditions = self._parent_conditions_of[child.name]
                        add = True
                        for condition in conditions:
                            if not condition._evaluate_vector(vector):
                                add = False
                                inactive.add(child.name)
                                break

                        if add:
                            hyperparameter_idx = self._hyperparameter_idx[
                                child.name]
                            active[hyperparameter_idx] = 1
                            to_visit.appendleft(child.name)

                    else:
                        continue

        if active[hp_idx] and np.isnan(hp_value):
            free(active)
            raise ValueError("Active hyperparameter '%s' not specified!" %
                             hyperparameter.name)

    for hp_idx in self._idx_to_hyperparameter:

        if not allow_inactive_with_values and not active[hp_idx] and \
                not np.isnan(vector[hp_idx]):
                # Only look up the value (in the line above) if the
                # hyperparameter is inactive!
            hp_name = self._idx_to_hyperparameter[hp_idx]
            hp_value = vector[hp_idx]
            free(active)
            raise ValueError("Inactive hyperparameter '%s' must not be "
                             "specified, but has the vector value: '%s'." %
                             (hp_name, hp_value))
    free(active)
    self._check_forbidden(vector)


cpdef np.ndarray correct_sampled_array(
    np.ndarray[DTYPE_t, ndim=1] vector,
    list forbidden_clauses_unconditionals,
    list forbidden_clauses_conditionals,
    list hyperparameters_with_children,
    int num_hyperparameters,
    list unconditional_hyperparameters,
    dict hyperparameter_to_idx,
    dict parent_conditions_of,
    dict parents_of,
    dict children_of,
):
    cdef AbstractForbiddenComponent clause
    cdef ConditionComponent condition
    cdef int hyperparameter_idx
    cdef DTYPE_t NaN = np.NaN
    cdef set visited
    cdef set inactive
    cdef Hyperparameter child
    cdef list children
    cdef str child_name
    cdef list parents
    cdef set parent_names
    cdef list conditions
    cdef int add

    cdef int* active
    active = <int*> malloc(sizeof(int) * num_hyperparameters)
    for j in range(num_hyperparameters):
        active[j] = 0

    for j in range(len(forbidden_clauses_unconditionals)):
        clause = forbidden_clauses_unconditionals[j]
        if clause.c_is_forbidden_vector(vector, strict=False):
            free(active)
            raise ForbiddenValueError(
                "Given vector violates forbidden clause %s" % (
                str(clause)))

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
                    if add == True:
                        active[hyperparameter_idx] = 1
                        hps.appendleft(child_name)

                else:
                    parent_names = set([p.name for p in parents])
                    if parent_names.issubset(visited):  # make sure no parents are still unvisited
                        conditions = parent_conditions_of[child_name]
                        add = True
                        for j in range(len(conditions)):
                            condition = conditions[j]
                            if not condition._evaluate_vector(vector):
                                add = False
                                vector[hyperparameter_idx] = NaN
                                inactive.add(child_name)
                                break

                        if add == True:
                            active[hyperparameter_idx] = 1
                            hps.appendleft(child_name)

                    else:
                        continue

    for j in range(len(vector)):
        if not active[j]:
            vector[j] = NaN

    free(active)
    for j in range(len(forbidden_clauses_conditionals)):
        clause = forbidden_clauses_conditionals[j]
        if clause.c_is_forbidden_vector(vector, strict=False):
            raise ForbiddenValueError(
                "Given vector violates forbidden clause %s" % (
                    str(clause)))

    return vector


cpdef np.ndarray change_hp_value(
    configuration_space,
    np.ndarray[DTYPE_t, ndim=1] configuration_array,
    str hp_name,
    DTYPE_t hp_value,
    int index,
):
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
    cdef Hyperparameter current
    cdef str current_name
    cdef list disabled
    cdef set visited
    cdef dict activated_values
    cdef int active
    cdef ConditionComponent condition
    cdef int current_idx
    cdef DTYPE_t current_value
    cdef DTYPE_t default_value
    cdef list children
    cdef list children_
    cdef Hyperparameter ch
    cdef str child
    cdef set to_disable
    cdef DTYPE_t NaN = np.NaN
    cdef dict children_of = configuration_space._children_of

    configuration_array[index] = hp_value

    # Hyperparameters which are going to be set to inactive
    disabled = []

    # Activate hyperparameters if their parent node got activated
    children = children_of[hp_name]
    if len(children) > 0:
        to_visit = deque()  # type: deque
        to_visit.extendleft(children)
        visited = set()  # type: Set[str]
        activated_values = dict()  # type: Dict[str, Union[int, float, str]]

        while len(to_visit) > 0:
            current = to_visit.pop()
            current_name = current.name
            if current_name in visited:
                continue
            visited.add(current_name)
            if current_name in disabled:
                continue

            current_idx = configuration_space._hyperparameter_idx[current_name]
            current_value = configuration_array[current_idx]

            conditions = configuration_space._parent_conditions_of[current_name]

            active = True
            for condition in conditions:
                if not condition._evaluate_vector(configuration_array):
                    active = False
                    break

            if active and not current_value == current_value:
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
        configuration_array[idx] = NaN

    return configuration_array
