# cython: language_level=3
from typing import Union
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

cdef class Hyperparameter(object):
    cdef public str name
    cdef public default_value
    cdef public DTYPE_t normalized_default_value
    cdef public dict meta

    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2)
    cpdef bint is_legal_vector(self, DTYPE_t value)

cdef class NumericalHyperparameter(Hyperparameter):
    cdef public lower
    cdef public upper
    cdef public q
    cdef public log
    cdef public _lower
    cdef public _upper
    cpdef int compare(self, value: Union[int, float, str], value2: Union[int, float, str])
    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2)

cdef class IntegerHyperparameter(NumericalHyperparameter):
    cdef ufhp
    cpdef int _transform_scalar(self, double scalar)
    cpdef np.ndarray _transform_vector(self, np.ndarray vector)

cdef class FloatHyperparameter(NumericalHyperparameter):
    cpdef double _transform_scalar(self, double scalar)
    cpdef np.ndarray _transform_vector(self, np.ndarray vector)
