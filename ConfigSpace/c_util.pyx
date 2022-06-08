# cython: language_level=3

import heapq
from collections import deque

import numpy as np
from ConfigSpace.forbidden import AbstractForbiddenComponent
from ConfigSpace.forbidden cimport AbstractForbiddenComponent
from ConfigSpace.hyperparameters import Hyperparameter
from ConfigSpace.hyperparameters cimport Hyperparameter
from ConfigSpace.conditions import ConditionComponent
from ConfigSpace.conditions cimport ConditionComponent
from ConfigSpace.exceptions import ForbiddenValueError

from libc.stdlib cimport malloc, free
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t


cpdef int check_forbidden(list forbidden_clauses, np.ndarray vector) except 1:
    cdef int Iforbidden = len(forbidden_clauses)
    cdef AbstractForbiddenComponent clause

    for i in range(Iforbidden):
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
    cdef set visited

    cdef int* active
    active = <int*> malloc(sizeof(int) * len(vector))
    for i in range(len(vector)):
        active[i] = 0

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
            free(active)
            raise ValueError("Hyperparameter instantiation '%s' "
                             "(type: %s) is illegal for hyperparameter %s" %
                             (hp_value, str(type(hp_value)),
                              hyperparameter))

        children = self._children_of[hp_name]
        for child in children:
            if child.name not in inactive:
                parents = self._parents_of[child.name]
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
            free(active)
            raise ValueError("Active hyperparameter '%s' not specified!" %
                             hyperparameter.name)

    for hp_idx in self._idx_to_hyperparameter:

        if not allow_inactive_with_values and not active[hp_idx] and not np.isnan(vector[hp_idx]):
            # Only look up the value (in the line above) if the hyperparameter is inactive!
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
    list conditional_hyperparameters,
    dict hyperparameter_to_idx,
    dict parent_conditions_of,
):
    """Ensure that the array values of inactive hyperparameters are NaN.
    
    The output array does not violate any condition or forbidden clause.

    Parameters
    ----------
    vector : np.ndarray
        Vector of hyperparameter values. It is assumed that none of the active hyperparameters has a NaN value assigned.
    
    conditional_hyperparameters : list[str]
        Names of conditional hyperparameters ordered topologically

    Returns
    -------
    np.ndarray
        Updated vector
    """
    cdef AbstractForbiddenComponent clause
    cdef ConditionComponent condition
    cdef DTYPE_t NaN = np.NaN
    cdef str current_name

    for j in range(len(forbidden_clauses_unconditionals)):
        clause = forbidden_clauses_unconditionals[j]
        if clause.c_is_forbidden_vector(vector, strict=False):
            raise ForbiddenValueError(
                "Given vector violates forbidden clause %s" % (
                    str(clause)
                )
            )
    
    # We assume that the conditional hyperparameters are ordered in topological order.
    for current_name in conditional_hyperparameters:
        for condition in parent_conditions_of[current_name]:
            if not condition._evaluate_vector(vector):
                vector[hyperparameter_to_idx[current_name]] = NaN
                break            

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
    cdef int active
    cdef int update
    cdef ConditionComponent condition
    cdef int current_idx
    cdef DTYPE_t current_value
    cdef DTYPE_t target_value
    cdef DTYPE_t default_value
    cdef DTYPE_t NaN = np.NaN
    cdef dict children_of = configuration_space._children_of

    # We maintain `to_visit` as a minimum heap of indices of hyperparameters that may need to be updated.
    # We assume that the hyperparameters are sorted topologically with respect to the conditions by the hyperparameter indices.
    # Initially, we know that the hyperparameter with the index `index` may need to be updated (by changing its value to `hp_value`).
    to_visit = [index]
    
    # Since one hyperparameter may be reachable in more than one way, we need to make sure we don't schedule it for inspection more than once.
    scheduled = np.zeros(len(configuration_space), dtype=bool)
    scheduled[index] = True

    # Activate hyperparameters if their parent node got activated.
    while len(to_visit) > 0:
        assert np.all(scheduled[to_visit])
        current_idx = heapq.heappop(to_visit)
        current_name = configuration_space._idx_to_hyperparameter[current_idx]
        conditions = configuration_space._parent_conditions_of[current_name]

        # Should the current hyperparameter be active?
        active = True
        for condition in conditions:
            if not condition._evaluate_vector(configuration_array):
                # The current hyperparameter should be inactive because `condition` is not satisfied. 
                active = False
                break

        # Should the value of the current hyperparameter be updated?
        update = False
        if current_idx == index:
            # The current hyperparameter should be updated because the caller requested this update.
            if not active:
                raise ValueError(
                    "Attempting to change the value of the inactive hyperparameter '%s' to '%s'." % (hp_name, hp_value))
            target_value = hp_value
            update = True
        else:
            current_value = configuration_array[current_idx]
            if active and not current_value == current_value:
                # The current hyperparameter should be active but is inactive.
                current = configuration_space._hyperparameters[current_name]
                target_value = current.normalized_default_value
                update = True
            elif not active and current_value == current_value:
                # The current hyperparameter should be inactive but is active.
                # If the hyperparameter was made inactive,
                # all its children need to be deactivated as well
                target_value = NaN
                update = True
        
        if update:
            configuration_array[current_idx] = target_value
            for child in children_of[current_name]:
                child_idx = configuration_space._hyperparameter_idx[child.name]
                # We assume that the hyperparameters are ordered topologically by index.
                # This means that every child must have an index greater than its parent.
                assert child_idx > current_idx
                if not scheduled[child_idx]:
                    heapq.heappush(to_visit, child_idx)
                    scheduled[child_idx] = True
        assert len(to_visit) == 0 or to_visit[0] > current_idx

    return configuration_array
