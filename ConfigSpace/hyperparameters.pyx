# Copyright (c) 2014-2016, ConfigSpace developers
# Matthias Feurer
# Katharina Eggensperger
# and others (see commit history).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# cython: language_level=3

from abc import ABCMeta, abstractmethod
import warnings

from ConfigSpace.hyperparameters cimport Hyperparameter

from collections import OrderedDict, Counter
import copy
from typing import List, Any, Dict, Union, Tuple, Optional
import io
import numpy as np
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
#DTYPE = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
#ctypedef np.float_t DTYPE_t


cdef class Hyperparameter(object):
    #cdef public str name
    #cdef public default_value

    def __init__(self, name: str, meta: Optional[Dict]) -> None:
        if not isinstance(name, str):
            raise TypeError(
                "The name of a hyperparameter must be an instance of"
                " %s, but is %s." % (str(str), type(name)))
        self.name = name # type : str
        self.meta = meta

    def __repr__(self):
        raise NotImplementedError()

    def is_legal(self, value):
        raise NotImplementedError()

    cpdef bint is_legal_vector(self, DTYPE_t value):
        """
        Check whether the given value is a legal value for the vector
        representation of this hyperparameter.

        Parameters
        ----------
        value
            the vector value to check

        Returns
        -------
        bool
            True if the given value is a legal vector value, otherwise False

        """
        raise NotImplementedError()

    def sample(self, rs):
        vector = self._sample(rs)
        return self._transform(vector)

    def _sample(self, rs, size):
        raise NotImplementedError()

    def _transform(self, vector: Union[np.ndarray, float, int]) -> Optional[Union[np.ndarray, float, int]]:
        raise NotImplementedError()

    def _inverse_transform(self, vector):
        raise NotImplementedError()

    def has_neighbors(self):
        raise NotImplementedError()

    def get_neighbors(self, value, rs, number, transform=False):
        raise NotImplementedError()

    def get_num_neighbors(self, value):
        raise NotImplementedError()

    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2):
        raise NotImplementedError()



cdef class Constant(Hyperparameter):
    cdef public value
    cdef DTYPE_t value_vector

    def __init__(self, name: str, value: Union[str, int, float], meta: Optional[Dict]=None) -> None:
        """
        Representing a constant hyperparameter in the configuration space.

        By sampling from the configuration space each time only a single,
        constant ``value`` will be drawn from this hyperparameter.

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        value : (str, int, float)
            value to sample hyperparameter from
        meta : (Dict, optional)
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        super(Constant, self).__init__(name, meta)
        allowed_types = (int, float, str)

        if not isinstance(value, allowed_types) or \
                isinstance(value, bool):
            raise TypeError("Constant value is of type %s, but only the "
                            "following types are allowed: %s" %
                            (type(value), allowed_types))  # type: ignore

        self.value = value
        self.value_vector = 0.
        self.default_value = value
        self.normalized_default_value = 0.

    def __repr__(self) -> str:
        repr_str = ["%s" % self.name,
                    "Type: Constant",
                    "Value: %s" % self.value]
        return ", ".join(repr_str)

    def __richcmp__(self, other: Any, int op):
        """
        Todo: With Cython 2.7 this function can be replaced by using the six
              Python special methods (__eq__(), __lt__(),...).
        (--> https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#rich-comparisons)

        Before 2.7 there were no separate methods for the individual rich
        comparison operations. Instead there is a single method __richcmp__()
        which takes an integer indicating which operation is to be performed,
        as follows:
        < 	0
        == 2
        > 	4
        <=	1
        !=	3
        >=	5
        """

        if isinstance(other, self.__class__):
            if op == 2:
                return self.value == other.value and self.name == other.name

            elif op == 3:
                return self.value != other.value or self.name != other.name

        return NotImplemented

    def __copy__(self):
        return Constant(self.name, self.value)

    def __hash__(self):
        return hash((self.name, self.value))

    def is_legal(self, value: Union[str, int, float]) -> bool:
        return value == self.value

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return value == self.value_vector

    def _sample(self, rs: None, size: Optional[int]=None) -> Union[int, np.ndarray]:
        return 0 if size == 1 else np.zeros((size,))

    def _transform(self, vector: Union[np.ndarray, float, int]) -> Optional[Union[np.ndarray, float, int]]:
        if not np.isfinite(vector):
            return None
        return self.value

    def _inverse_transform(self, vector: Union[np.ndarray, float, int]) -> Union[np.ndarray, int, float]:
        if vector != self.value:
            return np.NaN
        return 0

    def has_neighbors(self) -> bool:
        return False

    def get_num_neighbors(self, value=None) -> int:
        return 0

    def get_neighbors(self, value: Any, rs: np.random.RandomState, number: int, transform: bool = False) -> List:
        return []


cdef class UnParametrizedHyperparameter(Constant):
    pass


cdef class NumericalHyperparameter(Hyperparameter):
    cdef public lower
    cdef public upper
    cdef public q
    cdef public log
    cdef public _lower
    cdef public _upper

    def __init__(self, name: str, default_value: Any, meta: Optional[Dict]) -> None:
        super(NumericalHyperparameter, self).__init__(name, meta)
        self.default_value = default_value

    def has_neighbors(self) -> bool:
        return True

    def get_num_neighbors(self, value=None) -> float:

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

    def __richcmp__(self, other: Any, int op):
        """
        Todo: With Cython 2.7 this function can be replaced by using the six
              Python special methods (__eq__(), __lt__(),...).
        (--> https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#rich-comparisons)

        Before 2.7 there were no separate methods for the individual rich
        comparison operations. Instead there is a single method __richcmp__()
        which takes an integer indicating which operation is to be performed,
        as follows:
        < 	0
        == 2
        > 	4
        <=	1
        !=	3
        >=	5
        """

        if isinstance(other, self.__class__):
            if op == 2:
                return (
                    self.name == other.name and
                    self.default_value == other.default_value and
                    self.lower == other.lower and
                    self.upper == other.upper and
                    self.log == other.log and
                    self.q == other.q
                )

            elif op == 3:
                return (
                    self.name != other.name or
                    self.default_value != other.default_value or
                    self.lower != other.lower or
                    self.upper != other.upper or
                    self.log != other.log or
                    self.q != other.q
                )

        return NotImplemented

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
            q=self.q
        )


cdef class FloatHyperparameter(NumericalHyperparameter):
    def __init__(self, name: str, default_value: Union[int, float], meta: Optional[Dict]=None) -> None:
        super(FloatHyperparameter, self).__init__(name, default_value, meta)

    def is_legal(self, value: Union[int, float]) -> bool:
        raise NotImplemented

    cpdef bint is_legal_vector(self, DTYPE_t value):
        raise NotImplemented

    def check_default(self, default_value: Union[int, float]) -> float:
        raise NotImplemented


cdef class IntegerHyperparameter(NumericalHyperparameter):
    cdef ufhp

    def __init__(self, name: str, default_value: int, meta: Optional[Dict]=None) -> None:
        super(IntegerHyperparameter, self).__init__(name, default_value, meta)

    def is_legal(self, value: int) -> bool:
        raise NotImplemented

    cpdef bint is_legal_vector(self, DTYPE_t value):
        raise NotImplemented

    def check_default(self, default_value) -> int:
        raise NotImplemented

    def check_int(self, parameter: int, name: str) -> int:
        if abs(int(parameter) - parameter) > 0.00000001 and \
                        type(parameter) is not int:
            raise ValueError("For the Integer parameter %s, the value must be "
                             "an Integer, too. Right now it is a %s with value"
                             " %s." % (name, type(parameter), str(parameter)))
        return int(parameter)


cdef class UniformFloatHyperparameter(FloatHyperparameter):
    def __init__(self, name: str, lower: Union[int, float], upper: Union[int, float],
                 default_value: Union[int, float, None] = None, q: Union[int, float, None] = None, log: bool = False,
                 meta: Optional[Dict]=None) -> None:
        """
        A float hyperparameter.

        Its values are sampled from a uniform distribution with values
        from ``lower`` to ``upper``.

        Example
        -------

        >>> import ConfigSpace as CS
        >>> import ConfigSpace.hyperparameters as CSH
        >>> cs = CS.ConfigurationSpace()
        >>> uniform_float_hp = CSH.UniformFloatHyperparameter('uni_float', lower=10,
        ...                                                   upper=100, log=False)
        >>> cs.add_hyperparameter(uniform_float_hp)

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        lower : (int, floor)
            Lower bound of a range of values from which the hyperparameter will be sampled
        upper : (int, float)
            Upper bound
        default_value : (int, float, optional)
            Sets the default value of a hyperparameter to a given value
        q : (int, float, optional)
            Quantization factor
        log : (bool, optional)
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Default to ``False``
        meta : (Dict, optional)
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        super(UniformFloatHyperparameter, self).__init__(name, default_value, meta)
        self.lower = float(lower)
        self.upper = float(upper)
        self.q = float(q) if q is not None else None
        self.log = bool(log)

        if self.lower >= self.upper:
            raise ValueError("Upper bound %f must be larger than lower bound "
                             "%f for hyperparameter %s" %
                             (self.lower, self.upper, name))
        elif log and self.lower <= 0:
            raise ValueError("Negative lower bound (%f) for log-scale "
                             "hyperparameter %s is forbidden." %
                             (self.lower, name))

        self.default_value = self.check_default(default_value)

        if self.log:
            if self.q is not None:
                lower = self.lower - (np.float64(self.q) / 2. - 0.0001)
                upper = self.upper + (np.float64(self.q) / 2. - 0.0001)
            else:
                lower = self.lower
                upper = self.upper
            self._lower = np.log(lower)
            self._upper = np.log(upper)
        else:
            if self.q is not None:
                self._lower = self.lower - (self.q / 2. - 0.0001)
                self._upper = self.upper + (self.q / 2. - 0.0001)
            else:
                self._lower = self.lower
                self._upper = self.upper

        self.normalized_default_value = self._inverse_transform(self.default_value)

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: UniformFloat, Range: [%s, %s], Default: %s" %
                       (self.name, repr(self.lower), repr(self.upper),
                        repr(self.default_value)))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

    def is_legal(self, value: Union[float]) -> bool:
        if not (isinstance(value, float) or isinstance(value, int)):
            return False
        elif self.upper >= value >= self.lower:
            return True
        else:
            return False

    cpdef bint is_legal_vector(self, DTYPE_t value):
        if 1.0 >= value >= 0.0:
            return True
        else:
            return False

    def check_default(self, default_value: Optional[float]) -> float:
        if default_value is None:
            if self.log:
                default_value = np.exp((np.log(self.lower) + np.log(self.upper)) / 2.)
            else:
                default_value = (self.lower + self.upper) / 2.
        default_value = np.round(float(default_value), 10)

        if self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def to_integer(self) -> 'UniformIntegerHyperparameter':
        # TODO check if conversion makes sense at all (at least two integer values possible!)
        # todo check if params should be converted to int while class initialization or inside class itself
        return UniformIntegerHyperparameter(self.name, int(self.lower),
                                            int(self.upper),
                                            int(np.round(self.default_value)), int(self.q),
                                            self.log)

    def _sample(self, rs: np.random, size: Optional[int]=None) -> Union[float, np.ndarray]:
        return rs.uniform(size=size)

    def _transform(self, vector: Union[np.ndarray, float, int]) -> Optional[Union[np.ndarray, float, int]]:
        if np.any(np.isnan(vector)):
            return None
        vector *= (self._upper - self._lower)
        vector += self._lower
        if self.log:
            vector = np.exp(vector)
        if self.q is not None:
            vector = int(np.round(vector / self.q, 0)) * self.q
        vector = np.minimum(self.upper, vector)
        vector = np.maximum(self.lower, vector)
        return vector

    def _inverse_transform(self, vector: Union[np.ndarray, None]) -> Union[np.ndarray, float, int]:
        if vector is None:
            return np.NaN
        if self.log:
            vector = np.log(vector)
        vector = (vector - self._lower) / (self._upper - self._lower)
        vector = np.minimum(1.0, vector)
        vector = np.maximum(0.0, vector)
        return vector

    def get_neighbors(
        self,
        value: Any,
        rs: np.random.RandomState,
        number: int=4,
        transform: bool=False,
        std: float=0.2
    ) -> List[float]:
        neighbors = []  # type: List[float]
        while len(neighbors) < number:
            neighbor = rs.normal(value, std)  # type: float
            if neighbor < 0 or neighbor > 1:
                continue
            if transform:
                neighbors.append(self._transform(neighbor))
            else:
                neighbors.append(neighbor)
        return neighbors


cdef class NormalFloatHyperparameter(FloatHyperparameter):
    cdef public mu
    cdef public sigma

    def __init__(self, name: str, mu: Union[int, float], sigma: Union[int, float],
                 default_value: Union[None, float] = None, q: Union[int, float, None] = None, log: bool = False,
                 meta: Optional[Dict]=None) -> None:
        """
        A float hyperparameter.

        Its values are sampled from a normal distribution
        :math:`\mathcal{N}(\mu, \sigma^2)`.

        Example
        -------

        >>> import ConfigSpace as CS
        >>> import ConfigSpace.hyperparameters as CSH
        >>> cs = CS.ConfigurationSpace()
        >>> normal_float_hp = CSH.NormalFloatHyperparameter('normal_float', mu=0,
        ...                                                 sigma=1, log=False)
        >>> cs.add_hyperparameter(normal_float_hp)

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        mu : (int, float)
            Mean of the distribution
        sigma : (int, float)
            Standard deviation of the distribution
        default_value : (int, float, optional)
            Sets the default value of a hyperparameter to a given value
        q : (int, float, optional)
            Quantization factor
        log : (bool, optional)
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Default to ``False``
        meta : (Dict, optional)
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        super(NormalFloatHyperparameter, self).__init__(name, default_value, meta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.q = float(q) if q is not None else None
        self.log = bool(log)
        self.default_value = self.check_default(default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: NormalFloat, Mu: %s Sigma: %s, Default: %s" %
                       (self.name, repr(self.mu), repr(self.sigma),
                        repr(self.default_value)))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

    def __richcmp__(self, other: Any, int op):
        """
        Todo: With Cython 2.7 this function can be replaced by using the six
              Python special methods (__eq__(), __lt__(),...).
        (--> https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#rich-comparisons)

        Before 2.7 there were no separate methods for the individual rich
        comparison operations. Instead there is a single method __richcmp__()
        which takes an integer indicating which operation is to be performed,
        as follows:
        < 	0
        == 2
        > 	4
        <=	1
        !=	3
        >=	5
        """

        if isinstance(other, self.__class__):
            if op == 2:
                return (
                    self.name == other.name and
                    self.default_value == other.default_value and
                    self.mu == other.mu and
                    self.sigma == other.sigma and
                    self.log == other.log and
                    self.q == other.q
                )

            elif op == 3:
                return (
                    self.name != other.name or
                    self.default_value != other.default_value or
                    self.mu != other.mu or
                    self.sigma != other.sigma or
                    self.log != other.log or
                    self.q != other.q
                )

        return NotImplemented

    def __copy__(self):
        return NormalFloatHyperparameter(
            name=self.name,
            default_value=self.default_value,
            mu=self.mu,
            sigma=self.sigma,
            log=self.log,
            q=self.q,
        )

    def __hash__(self):
        return hash((self.name, self.mu, self.sigma, self.log, self.q))

    def to_uniform(self, z: int = 3) -> 'UniformFloatHyperparameter':
        return UniformFloatHyperparameter(self.name,
                                          self.mu - (z * self.sigma),
                                          self.mu + (z * self.sigma),
                                          default_value=int(
                                              np.round(self.default_value, 0)),
                                          q=self.q, log=self.log)

    def check_default(self, default_value: Union[int, float]) -> Union[int, float]:
        if default_value is None:
            return self.mu

        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def to_integer(self) -> 'NormalIntegerHyperparameter':
        if self.q is None:
            q_int = None
        else:
            q_int = int(self.q)
        return NormalIntegerHyperparameter(self.name, int(self.mu), self.sigma,
                                           default_value=int(np.round(self.default_value, 0)),
                                           q=q_int, log=self.log)

    def is_legal(self, value: Union[float]) -> bool:
        return isinstance(value, float) or isinstance(value, int)

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return isinstance(value, float) or isinstance(value, int)

    def _sample(self, rs: np.random.RandomState, size: Optional[int]=None) -> Union[np.ndarray, float]:
        mu = self.mu
        sigma = self.sigma
        return rs.normal(mu, sigma, size=size)

    def _transform(self, vector: Union[np.ndarray, float, int]) -> Optional[Union[np.ndarray, float, int]]:
        if np.isnan(vector):
            return None
        if self.log:
            vector = np.exp(vector)
        if self.q is not None:
            vector = int(np.round(vector / self.q, 0)) * self.q
        return vector

    def _inverse_transform(self, vector: Optional[np.ndarray]) -> Union[float, np.ndarray]:
        if vector is None:
            return np.NaN

        if self.log:
            vector = np.log(vector)
        return vector

    def get_neighbors(self, value: float, rs: np.random.RandomState, number: int = 4, transform: bool = False) -> List[float]:
        neighbors = []
        for i in range(number):
            neighbors.append(rs.normal(value, self.sigma))
        return neighbors


cdef class UniformIntegerHyperparameter(IntegerHyperparameter):
    def __init__(self, name: str, lower: int, upper: int, default_value: Union[int, None] = None,
                 q: Union[int, None] = None, log: bool = False, meta: Optional[Dict]=None) -> None:
        """
        An integer hyperparameter.

        Its values are sampled from a uniform distribution
        with bounds ``lower`` and ``upper``.

        Example
        -------

        >>> import ConfigSpace as CS
        >>> import ConfigSpace.hyperparameters as CSH
        >>> cs = CS.ConfigurationSpace()
        >>> uniform_integer_hp = CSH.UniformIntegerHyperparameter(name='uni_int', lower=10,
        ...                                                       upper=100, log=False)
        >>> cs.add_hyperparameter(uniform_integer_hp)

        Parameters
        ----------
        name : str
            Name of the hyperparameter with which it can be accessed
        lower : int
            Lower bound of a range of values from which the hyperparameter will be sampled
        upper : int
            upper bound
        default_value : (int, optional)
            Sets the default value of a hyperparameter to a given value
        q : (int, optional)
            Quantization factor
        log : (bool, optional)
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Defaults to ``False``
        meta : (Dict, optional)
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """

        super(UniformIntegerHyperparameter, self).__init__(name, default_value, meta)
        self.lower = self.check_int(lower, "lower")
        self.upper = self.check_int(upper, "upper")
        if default_value is not None:
            default_value = self.check_int(default_value, name)

        if q is not None:
            if q < 1:
                warnings.warn("Setting quantization < 1 for Integer "
                              "Hyperparameter '%s' has no effect." %
                              name)
                self.q = None
            else:
                self.q = self.check_int(q, "q")
        else:
            self.q = None
        self.log = bool(log)

        if self.lower >= self.upper:
            raise ValueError("Upper bound %d must be larger than lower bound "
                             "%d for hyperparameter %s" %
                             (self.lower, self.upper, name))
        elif log and self.lower <= 0:
            raise ValueError("Negative lower bound (%d) for log-scale "
                             "hyperparameter %s is forbidden." %
                             (self.lower, name))

        self.default_value = self.check_default(default_value)

        self.ufhp = UniformFloatHyperparameter(self.name,
                                               self.lower - 0.49999,
                                               self.upper + 0.49999,
                                               log=self.log, q=self.q,
                                               default_value=self.default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: UniformInteger, Range: [%s, %s], Default: %s"
                       % (self.name, repr(self.lower),
                          repr(self.upper), repr(self.default_value)))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % repr(np.int(self.q)))
        repr_str.seek(0)
        return repr_str.getvalue()

    def _sample(self, rs: np.random.RandomState, size: Optional[int]=None) -> Union[np.ndarray, float]:
        value = self.ufhp._sample(rs, size=size)
        # Map all floats which belong to the same integer value to the same
        # float value by first transforming it to an integer and then
        # transforming it back to a float between zero and one
        value = self._transform(value)
        value = self._inverse_transform(value)
        return value

    def _transform(self, vector: Union[np.ndarray, float, int]) -> Optional[Union[np.ndarray, float, int]]:
        if np.any(np.isnan(vector)):
            return None
        vector = self.ufhp._transform(vector)
        if self.q is not None:
            vector = (np.round(vector / self.q, 0)).astype(int) * self.q
        vector = (np.round(vector, 0)).astype(int)
        # Convert to regular float to avoid handling different data types
        if isinstance(vector, (np.int, np.int32, np.int64)):
            vector = int(vector)
        return vector

    def _inverse_transform(self, vector: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
        return self.ufhp._inverse_transform(vector)

    def is_legal(self, value: int) -> bool:
        if not (isinstance(value, (int, np.int, np.int32, np.int64))):
            return False
        elif self.upper >= value >= self.lower:
            return True
        else:
            return False

    cpdef bint is_legal_vector(self, DTYPE_t value):
        if 1.0 >= value >= 0.0:
            return True
        else:
            return False

    def check_default(self, default_value: Union[int, float]) -> int:
        if default_value is None:
            if self.log:
                default_value = np.exp((np.log(self.lower) + np.log(self.upper)) / 2.)
            else:
                default_value = (self.lower + self.upper) / 2.
        default_value = int(np.round(default_value, 0))

        if self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def has_neighbors(self) -> bool:
        if self.log:
            upper = np.exp(self.ufhp._upper)
            lower = np.exp(self.ufhp._lower)
        else:
            upper = self.ufhp._upper
            lower = self.ufhp._lower

        # If there is only one active value, this is not enough
        if upper - lower >= 1:
            return True
        else:
            return False

    def get_neighbors(
        self,
        value: Union[int, float],
        rs: np.random.RandomState,
        number: int=4,
        transform: bool=False,
        std: float=0.2,
    ) -> List[
        int]:
        neighbors = []  # type: List[int]
        while len(neighbors) < number:
            rejected = True  # type: bool
            iteration = 0  # type: int
            while rejected:
                new_min_value = np.min([1, rs.normal(loc=value, scale=std)])
                new_value = np.max((0, new_min_value))
                int_value = self._transform(value)
                new_int_value = self._transform(new_value)
                if int_value != new_int_value:
                    rejected = False
                elif iteration > 100000:
                    raise ValueError('Probably caught in an infinite loop.')

            if transform:
                neighbors.append(self._transform(new_value))
            else:
                new_value = self._transform(new_value)
                new_value = self._inverse_transform(new_value)
                neighbors.append(new_value)

        return neighbors


cdef class NormalIntegerHyperparameter(IntegerHyperparameter):
    cdef public mu
    cdef public sigma
    cdef nfhp

    def __init__(self, name: str, mu: int, sigma: Union[int, float],
                 default_value: Union[int, None] = None, q: Union[None, int] = None, log: bool = False,
                 meta: Optional[Dict]=None) -> None:
        """
        An integer hyperparameter.

        Its values are sampled from a normal distribution
        :math:`\mathcal{N}(\mu, \sigma^2)`.

        Example
        -------

            >>> import ConfigSpace as CS
            >>> import ConfigSpace.hyperparameters as CSH
            >>> cs = CS.ConfigurationSpace()
            >>> normal_int_hp = CSH.NormalIntegerHyperparameter(name='normal_int', mu=0.,
            ...                                                 sigma=1., log=False)
            >>> cs.add_hyperparameter(normal_int_hp)

        Parameters
        ----------
        name : str
            Name of the hyperparameter with which it can be accessed
        mu : int
            Mean of the distribution, from which hyperparameter is sampled
        sigma : (int, float)
            Standard deviation of the distribution, from which
            hyperparameter is sampled
        default_value : (int, optional)
            Sets the default value of a hyperparameter to a given value
        q : (int, optional)
            Quantization factor
        log : (bool, optional)
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Defaults to ``False``
        meta : (Dict, optional)
            Field for holding meta data provided by the user.
            Not used by the configuration space.

        """
        super(NormalIntegerHyperparameter, self).__init__(name, default_value, meta)
        self.mu = mu
        self.sigma = sigma

        if default_value is not None:
            default_value = self.check_int(default_value, self.name)

        if q is not None:
            if q < 1:
                warnings.warn("Setting quantization < 1 for Integer "
                              "Hyperparameter '%s' has no effect." %
                              name)
                self.q = None
            else:
                self.q = self.check_int(q, "q")
        else:
            self.q = None
        self.log = bool(log)

        self.default_value = self.check_default(default_value)

        self.nfhp = NormalFloatHyperparameter(self.name,
                                              self.mu,
                                              self.sigma,
                                              log=self.log,
                                              q=self.q,
                                              default_value=self.default_value)

        self.normalized_default_value = self._inverse_transform(self.default_value)

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: NormalInteger, Mu: %s Sigma: %s, Default: "
                       "%s" % (self.name, repr(self.mu),
                               repr(self.sigma), repr(self.default_value)))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

    def __richcmp__(self, other: Any, int op):
        """
        Todo: With Cython 2.7 this function can be replaced by using the six
              Python special methods (__eq__(), __lt__(),...).
        (--> https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#rich-comparisons)

        Before 2.7 there were no separate methods for the individual rich
        comparison operations. Instead there is a single method __richcmp__()
        which takes an integer indicating which operation is to be performed,
        as follows:
        < 	0
        == 2
        > 	4
        <=	1
        !=	3
        >=	5
        """

        if isinstance(other, self.__class__):
            if op == 2:
                return (
                    self.name == other.name and
                    self.mu == other.mu and
                    self.sigma == other.sigma and
                    self.log == other.log and
                    self.q == other.q
                )

            elif op == 3:
                return (
                    self.name != other.name or
                    self.mu != other.mu or
                    self.sigma != other.sigma or
                    self.log != other.log or
                    self.q != other.q
                )

        return NotImplemented

    def __hash__(self):
        return hash((self.name, self.mu, self.sigma, self.log, self.q))

    def __copy__(self):
        return NormalIntegerHyperparameter(
            name=self.name,
            default_value=self.default_value,
            mu=self.mu,
            sigma=self.sigma,
            log=self.log,
            q=self.q,
        )

    # todo check if conversion should be done in initiation call or inside class itsel
    def to_uniform(self, z: int = 3) -> 'UniformIntegerHyperparameter':
        return UniformIntegerHyperparameter(self.name,
                                            np.round(int(self.mu - (z * self.sigma))),
                                            np.round(int(self.mu + (z * self.sigma))),
                                            default_value=self.default_value,
                                            q=self.q, log=self.log)

    def is_legal(self, value: int) -> bool:
        return isinstance(value, (int, np.int, np.int32, np.int64))

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return isinstance(value, float) or isinstance(value, int)

    def check_default(self, default_value: int) -> int:
        if default_value is None:
            return self.mu

        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def _sample(self, rs: np.random.RandomState, size: Optional[int]=None) -> Union[np.ndarray, float]:
        value = self.nfhp._sample(rs, size=size)
        # Map all floats which belong to the same integer value to the same
        # float value by first transforming it to an integer and then
        # transforming it back to a float between zero and one
        value = self._transform(value)
        value = self._inverse_transform(value)
        return value

    def _transform(self, vector: Union[np.ndarray, float, int]) -> Optional[Union[np.ndarray, int]]:
        if np.isnan(vector):
            return None
        vector = self.nfhp._transform(vector)
        vector = (np.round(vector, 0)).astype(int)
        if isinstance(vector, (np.int, np.int32, np.int64)):
            vector = int(vector)
        return vector

    def _inverse_transform(self, vector: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
        return self.nfhp._inverse_transform(vector)

    def has_neighbors(self) -> bool:
        return True

    def get_neighbors(
        self,
        value: Union[int, float],
        rs: np.random.RandomState,
        number: int=4,
        transform: bool=False,
    ) -> List[Union[np.ndarray, float, int]]:
        neighbors = []  # type: List[Union[np.ndarray, float, int]]
        while len(neighbors) < number:
            rejected = True
            iteration = 0
            while rejected:
                iteration += 1
                new_value = rs.normal(value, self.sigma)
                int_value = self._transform(value)
                new_int_value = self._transform(new_value)
                if int_value != new_int_value:
                    rejected = False
                elif iteration > 100000:
                    raise ValueError('Probably caught in an infinite loop.')

            if transform:
                neighbors.append(self._transform(new_value))
            else:
                neighbors.append(new_value)
        return neighbors


cdef class CategoricalHyperparameter(Hyperparameter):
    cdef public tuple choices
    cdef public int num_choices
    cdef list choices_vector
    cdef set _choices_set

    # TODO add more magic for automated type recognition
    # TODO move from list to tuple for choices argument
    def __init__(
        self,
        name: str,
        choices: Union[List[Union[str, float, int]], Tuple[Union[float, int, str]]],
        default_value: Union[int, float, str, None]=None,
        meta: Optional[Dict]=None
    ) -> None:
        """
        A categorical hyperparameter.

        Its values are sampled from a set of ``values``.

        Example
        -------

        >>> import ConfigSpace as CS
        >>> import ConfigSpace.hyperparameters as CSH
        >>> cs = CS.ConfigurationSpace()
        >>> cat_hp = CSH.CategoricalHyperparameter('cat_hp', choices=['red', 'green', 'blue'])
        >>> cs.add_hyperparameter(cat_hp)

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        choices : (list([str, float, int]), tuple([str, float, int]))
            Collection of values to sample hyperparameter from
        default_value : (int, float, str, optional)
            Sets the default value of the hyperparameter to a given value
        meta : (Dict, optional)
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """

        super(CategoricalHyperparameter, self).__init__(name, meta)
        # TODO check that there is no bullshit in the choices!
        counter = Counter(choices)
        for choice in choices:
            if counter[choice] > 1:
                raise ValueError(
                    "Choices for categorical hyperparameters %s contain choice '%s' %d "
                    "times, while only a single oocurence is allowed."
                    % (name, choice, counter[choice])
                )
            if choice is None:
                raise TypeError("Choice 'None' is not supported")
        self.choices = tuple(choices)
        self.num_choices = len(choices)
        self.choices_vector = list(range(self.num_choices))
        self._choices_set = set(self.choices_vector)
        self.default_value = self.check_default(default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: Categorical, Choices: {" % (self.name))
        for idx, choice in enumerate(self.choices):
            repr_str.write(str(choice))
            if idx < len(self.choices) - 1:
                repr_str.write(", ")
        repr_str.write("}")
        repr_str.write(", Default: ")
        repr_str.write(str(self.default_value))
        repr_str.seek(0)
        return repr_str.getvalue()

    def __richcmp__(self, other: Any, int op):
        """
        Todo: With Cython 2.7 this function can be replaced by using the six
              Python special methods (__eq__(), __lt__(),...).
        (--> https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#rich-comparisons)

        Before 2.7 there were no separate methods for the individual rich
        comparison operations. Instead there is a single method __richcmp__()
        which takes an integer indicating which operation is to be performed,
        as follows:
        < 	0
        == 2
        > 	4
        <=	1
        !=	3
        >=	5
        """

        if isinstance(other, self.__class__):
            if op == 2:
                return (
                    self.name == other.name and
                    self.choices == other.choices
                )

            elif op == 3:
                return (
                    self.name != other.name or
                    self.choices != other.choices
                )

        return NotImplemented

    def __hash__(self):
        return hash((self.name, self.choices))

    def __copy__(self):
        return CategoricalHyperparameter(
            name=self.name,
            choices=copy.deepcopy(self.choices),
            default_value=self.default_value,
        )

    cpdef int compare(self, value: Union[int, float, str], value2: Union[int, float, str]):
        if value == value2:
            return 0
        else:
            return 1

    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2):
        if value == value2:
            return 0
        else:
            return 1

    def is_legal(self, value: Union[None, str, float, int]) -> bool:
        if value in self.choices:
            return True
        else:
            return False

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return value in self._choices_set

    def check_default(self, default_value: Union[None, str, float, int]) -> Union[str, float, int]:
        if default_value is None:
            return self.choices[0]
        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def _sample(self, rs: np.random.RandomState, size: Optional[int]=None) -> Union[int, np.ndarray]:
        return rs.randint(0, self.num_choices, size=size)

    def _transform(self, vector: Union[np.ndarray, float, int]) -> Optional[Union[str, int, float]]:
        if not np.isfinite(vector):
            return None
        if np.equal(np.mod(vector, 1), 0):
            return self.choices[int(vector)]
        else:
            raise ValueError('Can only index the choices of the categorical '
                             'hyperparameter %s with an integer, but provided '
                             'the following float: %f' % (self, vector))

    def _inverse_transform(self, vector: Union[None, str, float, int]) -> Union[int, float]:
        if vector is None:
            return np.NaN
        return self.choices.index(vector)

    def has_neighbors(self) -> bool:
        return len(self.choices) > 1

    def get_num_neighbors(self, value=None) -> int:
        return len(self.choices) - 1

    def get_neighbors(self, value: int, rs: np.random.RandomState, number: Union[int, float] = np.inf, transform: bool = False) -> \
            List[Union[float, int, str]]:
        neighbors = []  # type: List[Union[float, int, str]]
        if number < len(self.choices):
            while len(neighbors) < number:
                rejected = True
                index = int(value)
                while rejected:
                    neighbor_idx = rs.randint(0, self.num_choices)
                    if neighbor_idx != index:
                        rejected = False

                if transform:
                    candidate = self._transform(neighbor_idx)
                else:
                    candidate = float(neighbor_idx)

                if candidate in neighbors:
                    continue
                else:
                    neighbors.append(candidate)
        else:
            for candidate_idx, candidate_value in enumerate(self.choices):
                if int(value) == candidate_idx:
                    continue
                else:
                    if transform:
                        candidate = self._transform(candidate_idx)
                    else:
                        candidate = float(candidate_idx)

                    neighbors.append(candidate)

        return neighbors

    def allow_greater_less_comparison(self) -> bool:
        raise ValueError("Parent hyperparameter in a > or < "
                         "condition must be a subclass of "
                         "NumericalHyperparameter or "
                         "OrdinalHyperparameter, but is "
                         "<cdef class 'ConfigSpace.hyperparameters.CategoricalHyperparameter'>")


cdef class OrdinalHyperparameter(Hyperparameter):
    cdef public tuple sequence
    cdef public int num_elements
    cdef sequence_vector
    cdef value_dict
    def __init__(
        self,
        name: str,
        sequence: Union[List[Union[float, int, str]], Tuple[Union[float, int, str]]],
        default_value: Union[str, int, float, None]=None,
        meta: Optional[Dict]=None
    ) -> None:
        """
        An ordinal hyperparameter.

        Its values are sampled form a ``sequence`` of values.

        Example
        -------

        >>> import ConfigSpace as CS
        >>> import ConfigSpace.hyperparameters as CSH
        >>> cs = CS.ConfigurationSpace()
        >>> ord_hp = CSH.OrdinalHyperparameter('ordinal_hp', sequence=['10', '20', '30'])
        >>> cs.add_hyperparameter(ord_hp)

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed.
        sequence : (list([str, float, int]), tuple([str, float, int]))
            collection of values to sample hyperparameter from.
        default_value : (int, float, str, optional)
            Sets the default value of a hyperparameter to a given value.
        meta : (Dict, optional)
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """

        # Remark
        # Since the sequence can consist of elements from different types,
        # they are stored into a dictionary in order to handle them as a
        # numeric sequence according to their order/position.
        super(OrdinalHyperparameter, self).__init__(name, meta)
        if len(sequence) > len(set(sequence)):
            raise ValueError("Ordinal Hyperparameter Sequence %s contain duplicate values." % sequence)
        self.sequence = tuple(sequence)
        self.num_elements = len(sequence)
        self.sequence_vector = list(range(self.num_elements))
        self.default_value = self.check_default(default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)
        self.value_dict = OrderedDict()  # type: OrderedDict[Union[int, float, str], int]
        counter = 0
        for element in self.sequence:
            self.value_dict[element] = counter
            counter += 1

    def __hash__(self):
            return hash((self.name, self.sequence))

    def __repr__(self) -> str:
        """
        write out the parameter definition
        """
        repr_str = io.StringIO()
        repr_str.write("%s, Type: Ordinal, Sequence: {" % (self.name))
        for idx, seq in enumerate(self.sequence):
            repr_str.write(str(seq))
            if idx < len(self.sequence) - 1:
                repr_str.write(", ")
        repr_str.write("}")
        repr_str.write(", Default: ")
        repr_str.write(str(self.default_value))
        repr_str.seek(0)
        return repr_str.getvalue()

    def __richcmp__(self, other: Any, int op):
        """
        Todo: With Cython 2.7 this function can be replaced by using the six
              Python special methods (__eq__(), __lt__(),...).
        (--> https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#rich-comparisons)

        Before 2.7 there were no separate methods for the individual rich
        comparison operations. Instead there is a single method __richcmp__()
        which takes an integer indicating which operation is to be performed,
        as follows:
        < 	0
        == 2
        > 	4
        <=	1
        !=	3
        >=	5
        """

        if isinstance(other, self.__class__):
            if op == 2:
                return (
                    self.name == other.name and
                    self.sequence == other.sequence
                )

            elif op == 3:
                return (
                    self.name != other.name or
                    self.sequence != other.sequence
                )

        return NotImplemented

    def __copy__(self):
        return OrdinalHyperparameter(
                name=self.name,
                sequence=copy.deepcopy(self.sequence),
                default_value=self.default_value,
            )

    cpdef int compare(self, value: Union[int, float, str], value2: Union[int, float, str]):
        if self.value_dict[value] < self.value_dict[value2]:
            return -1
        elif self.value_dict[value] > self.value_dict[value2]:
            return 1
        elif self.value_dict[value] == self.value_dict[value2]:
            return 0

    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2):
        if value < value2:
            return -1
        elif value > value2:
            return 1
        elif value == value2:
            return 0

    def is_legal(self, value: Union[int, float, str]) -> bool:
        """
        check if a certain value is represented in the sequence
        """
        return value in self.sequence

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return value in self.sequence_vector

    def check_default(self, default_value: Optional[Union[int, float, str]]) -> Union[int, float, str]:
        """
        check if given default value is represented in the sequence.
        If there's no default value we simply choose the
        first element in our sequence as default.
        """
        if default_value is None:
            return self.sequence[0]
        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def _transform(self, vector: Union[np.ndarray, float, int]) -> Optional[Union[np.ndarray, float, int]]:
        if vector != vector:
            return None
        if np.equal(np.mod(vector, 1), 0):
            return self.sequence[int(vector)]
        else:
            raise ValueError('Can only index the choices of the ordinal '
                             'hyperparameter %s with an integer, but provided '
                             'the following float: %f' % (self, vector))

    def _inverse_transform(self, vector: Optional[Union[np.ndarray, List, int, str, float]]) -> Union[float, List[int], List[str], List[float]]:
        if vector is None:
            return np.NaN
        return self.sequence.index(vector)

    def get_seq_order(self) -> np.ndarray:
        """
        return the ordinal sequence as numeric sequence
        (according to the the ordering) from 1 to length of our sequence.
        """
        return np.arange(0, self.num_elements)

    def get_order(self, value: Optional[Union[int, str, float]]) -> int:
        """
        return the seuence position/order of a certain value from the sequence
        """
        return self.value_dict[value]

    def get_value(self, idx: int) -> Union[int, str, float]:
        """
        return the sequence value of a given order/position
        """
        return list(self.value_dict.keys())[list(self.value_dict.values()).index(idx)]

    def check_order(self, val1: Union[int, str, float], val2: Union[int, str, float]) -> bool:
        """
        check whether value1 is smaller than value2.
        """
        idx1 = self.get_order(val1)
        idx2 = self.get_order(val2)
        if idx1 < idx2:
            return True
        else:
            return False

    def _sample(self, rs: np.random.RandomState, size: Optional[int]=None) -> int:
        """
        return a random sample from our sequence as order/position index
        """
        return rs.randint(0, self.num_elements, size=size)

    def has_neighbors(self) -> bool:
        """
        check if there are neighbors or we're only dealing with an
        one-element sequence
        """
        return len(self.sequence) > 1

    def get_num_neighbors(self, value: Union[int, float, str]) -> int:
        """
        return the number of existing neighbors in the sequence
        """
        max_idx = len(self.sequence) - 1
        if value == self.sequence[0] and value == self.sequence[max_idx]:# check if there is only one value
            return 0
        elif value == self.sequence[0] or value == self.sequence[max_idx]:
            return 1
        else:
            return 2

    def get_neighbors(self, value: Union[int, str, float], rs: None, number: int = 0, transform: bool = False) \
            -> List[Union[str, float, int]]:
        """
        Return the neighbors of a given value.
        Value must be in vector form. Ordinal name will not work.
        """
        neighbors = []
        if transform:
            if self.get_num_neighbors(value) < len(self.sequence):
                index = self.get_order(value)
                neighbor_idx1 = index - 1
                neighbor_idx2 = index + 1
                seq = self.get_seq_order()

                if neighbor_idx1 >= seq[0]:
                    candidate1 = self.get_value(neighbor_idx1)
                    if self.check_order(candidate1, value):
                        neighbors.append(candidate1)
                if neighbor_idx2 < self.num_elements:
                    candidate2 = self.get_value(neighbor_idx2)
                    if self.check_order(value, candidate2):
                        neighbors.append(candidate2)

        else:
            if self.get_num_neighbors(self.get_value(value)) < len(self.sequence):
                index = value
                neighbor_idx1 = index - 1
                neighbor_idx2 = index + 1
                seq = self.get_seq_order()

                if neighbor_idx1 < index and neighbor_idx1 >= seq[0]:
                    neighbors.append(neighbor_idx1)
                if neighbor_idx2 > index and neighbor_idx2 < self.num_elements:
                    neighbors.append(neighbor_idx2)

        return neighbors

    def allow_greater_less_comparison(self) -> bool:
        return True
