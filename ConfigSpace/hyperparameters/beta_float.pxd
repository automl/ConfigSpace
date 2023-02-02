import numpy as np
cimport numpy as np
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

from .uniform_float cimport UniformFloatHyperparameter


cdef class BetaFloatHyperparameter(UniformFloatHyperparameter):
    cdef public alpha
    cdef public beta
