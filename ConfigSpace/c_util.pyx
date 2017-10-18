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