from typing import Any, Dict, Optional, Union

import numpy as np
cimport numpy as np
np.import_array()


cdef class NumericalHyperparameter(Hyperparameter):

    def __init__(self, name: str, default_value: Any, meta: Optional[Dict]) -> None:
        super(NumericalHyperparameter, self).__init__(name, meta)
        self.default_value = default_value

    def has_neighbors(self) -> bool:
        return True

    def get_num_neighbors(self, value = None) -> float:

        return np.inf

    cpdef int compare(self, value: Union[int, float, str], value2: Union[int, float, str]):
        if value < value2:
            return -1
        elif value > value2:
            return 1
        elif value == value2:
            return 0

    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2):
        if value < value2:
            return -1
        elif value > value2:
            return 1
        elif value == value2:
            return 0

    def allow_greater_less_comparison(self) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        """
        This method implements a comparison between self and another
        object.

        Additionally, it defines the __ne__() as stated in the
        documentation from python:
            By default, object implements __eq__() by using is, returning NotImplemented
            in the case of a false comparison: True if x is y else NotImplemented.
            For __ne__(), by default it delegates to __eq__() and inverts the result
            unless it is NotImplemented.

        """
        if not isinstance(other, self.__class__):
            return False

        return (
            self.name == other.name and
            self.default_value == other.default_value and
            self.lower == other.lower and
            self.upper == other.upper and
            self.log == other.log and
            self.q == other.q
        )

    def __hash__(self):
        return hash(
            (
                self.name,
                self.lower,
                self.upper,
                self.log,
                self.q
            )
        )

    def __copy__(self):
        return self.__class__(
            name=self.name,
            default_value=self.default_value,
            lower=self.lower,
            upper=self.upper,
            log=self.log,
            q=self.q,
            meta=self.meta
        )
