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

from .numerical import NumericalHyperparameter
from .numerical cimport NumericalHyperparameter


cdef class FloatHyperparameter(NumericalHyperparameter):
    cpdef double _transform_scalar(self, double scalar)
    cpdef np.ndarray _transform_vector(self, np.ndarray vector)
