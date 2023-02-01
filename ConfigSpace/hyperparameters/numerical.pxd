from typing import Union
import numpy as np
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

from .hyperparameter import Hyperparameter
from .hyperparameter cimport Hyperparameter


cdef class NumericalHyperparameter(Hyperparameter):
    cdef public lower
    cdef public upper
    cdef public q
    cdef public log
    cdef public _lower
    cdef public _upper
    cpdef int compare(self, value: Union[int, float, str], value2: Union[int, float, str])
    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2)
