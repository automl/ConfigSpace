# cython: language_level=3

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


cdef class AbstractForbiddenComponent(object):

    cdef public hyperparameter
    cdef public int vector_id
    cdef public value
    cdef public DTYPE_t vector_value

    cdef int c_is_forbidden_vector(self, np.ndarray instantiated_hyperparameters, int strict)
    cpdef get_descendant_literal_clauses(self)
    cpdef set_vector_idx(self, hyperparameter_to_idx)
    cpdef is_forbidden(self, instantiated_hyperparameters, strict)
