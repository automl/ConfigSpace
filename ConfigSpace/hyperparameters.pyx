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
import copy
import io
import math
import warnings
from collections import OrderedDict, Counter
from itertools import count
from more_itertools import roundrobin, duplicates_everseen
from typing import List, Any, Dict, Union, Set, Tuple, Optional, Sequence

from scipy.stats import truncnorm, beta as spbeta, norm
import numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
cimport numpy as np
np.import_array()

from ConfigSpace.functional import center_range, arange_chunked

# OPTIM: Some operations generate an arange which could blowup memory if
# done over the entire space of integers (int32/64).
# To combat this, `arange_chunked` is used in scenarios where reducion
# operations over all the elments could be done in partial steps independantly.
# For example, a sum over the pdf values could be done in chunks.
# This may add some small overhead for smaller ranges but is unlikely to
# be noticable.
ARANGE_CHUNKSIZE = 10_000_000

cdef class Hyperparameter(object):

    def __init__(self, name: str, meta: Optional[Dict]) -> None:
        if not isinstance(name, str):
            raise TypeError(
                "The name of a hyperparameter must be an instance of"
                " %s, but is %s." % (str(str), type(name)))
        self.name: str = name
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

    def rvs(
        self,
        size: Optional[int] = None,
        random_state: Optional[Union[int, np.random, np.random.RandomState]] = None
    ) -> Union[float, np.ndarray]:
        """
        scipy compatibility wrapper for ``_sample``,
        allowing the hyperparameter to be used in sklearn API
        hyperparameter searchers, eg. GridSearchCV.

        """

        # copy-pasted from scikit-learn utils/validation.py
        def check_random_state(seed):
            """
            Turn seed into a np.random.RandomState instance
            If seed is None (or np.random), return the RandomState singleton used
            by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            If seed is a new-style np.random.Generator, return it.
            Otherwise, raise ValueError.

            """
            if seed is None or seed is np.random:
                return np.random.mtrand._rand
            if isinstance(seed, (int, np.integer)):
                return np.random.RandomState(seed)
            if isinstance(seed, np.random.RandomState):
                return seed
            try:
                # Generator is only available in numpy >= 1.17
                if isinstance(seed, np.random.Generator):
                    return seed
            except AttributeError:
                pass
            raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                             ' instance' % seed)

        # if size=None, return a value, but if size=1, return a 1-element array

        vector = self._sample(
            rs=check_random_state(random_state),
            size=size if size is not None else 1
        )
        if size is None:
            vector = vector[0]

        return self._transform(vector)

    def _sample(self, rs, size):
        raise NotImplementedError()

    def _transform(
        self,
        vector: Union[np.ndarray, float, int]
    ) -> Optional[Union[np.ndarray, float, int]]:
        raise NotImplementedError()

    def _inverse_transform(self, vector):
        raise NotImplementedError()

    def has_neighbors(self):
        raise NotImplementedError()

    def get_neighbors(self, value, rs, number, transform = False):
        raise NotImplementedError()

    def get_num_neighbors(self, value):
        raise NotImplementedError()

    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2):
        raise NotImplementedError()

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the hyperparameter in
        the hyperparameter space (the one specified by the user).
        For each hyperparameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) hyperparameter
        space. Only legal values return a positive probability density,
        otherwise zero.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        raise NotImplementedError()

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the hyperparameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
         in the pdf method handles these).

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        raise NotImplementedError()

    def get_size(self) -> float:
        raise NotImplementedError()


cdef class Constant(Hyperparameter):
    cdef public value
    cdef DTYPE_t value_vector

    def __init__(self, name: str, value: Union[str, int, float], meta: Optional[Dict] = None
                 ) -> None:
        """
        Representing a constant hyperparameter in the configuration space.

        By sampling from the configuration space each time only a single,
        constant ``value`` will be drawn from this hyperparameter.

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        value : str, int, float
            value to sample hyperparameter from
        meta : Dict, optional
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
            self.value == other.value and
            self.name == other.name and
            self.default_value == other.default_value
        )

    def __copy__(self):
        return Constant(self.name, self.value, meta=self.meta)

    def __hash__(self):
        return hash((self.name, self.value))

    def is_legal(self, value: Union[str, int, float]) -> bool:
        return value == self.value

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return value == self.value_vector

    def _sample(self, rs: None, size: Optional[int] = None) -> Union[int, np.ndarray]:
        return 0 if size == 1 else np.zeros((size,))

    def _transform(self, vector: Optional[Union[np.ndarray, float, int]]) \
            -> Optional[Union[np.ndarray, float, int]]:
        return self.value

    def _transform_vector(self, vector: Optional[np.ndarray]) \
            -> Optional[Union[np.ndarray, float, int]]:
        return self.value

    def _transform_scalar(self, vector: Optional[Union[float, int]]) \
            -> Optional[Union[np.ndarray, float, int]]:
        return self.value

    def _inverse_transform(self, vector: Union[np.ndarray, float, int]
                           ) -> Union[np.ndarray, int, float]:
        if vector != self.value:
            return np.NaN
        return 0

    def has_neighbors(self) -> bool:
        return False

    def get_num_neighbors(self, value = None) -> int:
        return 0

    def get_neighbors(self, value: Any, rs: np.random.RandomState, number: int,
                      transform: bool = False) -> List:
        return []

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the original parameter space (the one specified by the user).
        For each hyperparameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) parameter
        space. Only legal values return a positive probability density,
        otherwise zero.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        if vector.ndim != 1:
            raise ValueError("Method pdf expects a one-dimensional numpy array")
        return self._pdf(vector)

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these).

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        return (vector == self.value).astype(float)

    def get_max_density(self):
        return 1.0

    def get_size(self) -> float:
        return 1.0

cdef class UnParametrizedHyperparameter(Constant):
    pass


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


cdef class FloatHyperparameter(NumericalHyperparameter):
    def __init__(self, name: str, default_value: Union[int, float], meta: Optional[Dict] = None
                 ) -> None:
        super(FloatHyperparameter, self).__init__(name, default_value, meta)

    def is_legal(self, value: Union[int, float]) -> bool:
        raise NotImplementedError()

    cpdef bint is_legal_vector(self, DTYPE_t value):
        raise NotImplementedError()

    def check_default(self, default_value: Union[int, float]) -> float:
        raise NotImplementedError()

    def _transform(self, vector: Union[np.ndarray, float, int]
                   ) -> Optional[Union[np.ndarray, float, int]]:
        try:
            if isinstance(vector, np.ndarray):
                return self._transform_vector(vector)
            return self._transform_scalar(vector)
        except ValueError:
            return None

    cpdef double _transform_scalar(self, double scalar):
        raise NotImplementedError()

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        raise NotImplementedError()

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the original parameter space (the one specified by the user).
        For each parameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) parameter
        space. Only legal values return a positive probability density,
        otherwise zero.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        if vector.ndim != 1:
            raise ValueError("Method pdf expects a one-dimensional numpy array")
        vector = self._inverse_transform(vector)
        return self._pdf(vector)

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these).

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        raise NotImplementedError()

    def get_max_density(self) -> float:
        """
        Returns the maximal density on the pdf for the parameter (so not
        the mode, but the value of the pdf on the mode).
        """
        raise NotImplementedError()



cdef class IntegerHyperparameter(NumericalHyperparameter):
    def __init__(self, name: str, default_value: int, meta: Optional[Dict] = None) -> None:
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

    def _transform(self, vector: Union[np.ndarray, float, int]
                   ) -> Optional[Union[np.ndarray, float, int]]:
        try:
            if isinstance(vector, np.ndarray):
                return self._transform_vector(vector)
            return self._transform_scalar(vector)
        except ValueError:
            return None

    cpdef long long _transform_scalar(self, double scalar):
        raise NotImplementedError()

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        raise NotImplementedError()

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the hyperparameter in
        the hyperparameter space (the one specified by the user).
        For each hyperparameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) hyperparameter
        space. Only legal values return a positive probability density,
        otherwise zero.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        if vector.ndim != 1:
            raise ValueError("Method pdf expects a one-dimensional numpy array")
        is_integer = (np.round(vector) == vector).astype(int)
        vector = self._inverse_transform(vector)
        return self._pdf(vector) * is_integer

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these). Optimally, an IntegerHyperparameter
        should have a corresponding float, which can be utlized for the calls
        to the probability density function (see e.g. NormalIntegerHyperparameter)

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        raise NotImplementedError()

    def get_max_density(self) -> float:
        """
        Returns the maximal density on the pdf for the parameter (so not
        the mode, but the value of the pdf on the mode).
        """
        raise NotImplementedError()


cdef class UniformFloatHyperparameter(FloatHyperparameter):
    def __init__(self, name: str, lower: Union[int, float], upper: Union[int, float],
                 default_value: Union[int, float, None] = None,
                 q: Union[int, float, None] = None, log: bool = False,
                 meta: Optional[Dict] = None) -> None:
        """
        A uniformly distributed float hyperparameter.

        Its values are sampled from a uniform distribution with values
        from ``lower`` to ``upper``.

        >>> from ConfigSpace import UniformFloatHyperparameter
        >>>
        >>> UniformFloatHyperparameter('u', lower=10, upper=100, log = False)
        u, Type: UniformFloat, Range: [10.0, 100.0], Default: 55.0

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        lower : int, float
            Lower bound of a range of values from which the hyperparameter will be sampled
        upper : int, float
            Upper bound
        default_value : int, float, optional
            Sets the default value of a hyperparameter to a given value
        q : int, float, optional
            Quantization factor
        log : bool, optional
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Default to ``False``
        meta : Dict, optional
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
                             (self.upper, self.lower, name))
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
        if self.q is not None:
            # There can be weird rounding errors, so we compare the result against self.q, see
            # In [13]: 2.4 % 0.2
            # Out[13]: 0.1999999999999998
            if np.round((self.upper - self.lower) % self.q, 10) not in (0, self.q):
                raise ValueError(
                    'Upper bound (%f) - lower bound (%f) must be a multiple of q (%f)'
                    % (self.upper, self.lower, self.q)
                )

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
        # todo check if params should be converted to int while class initialization
        # or inside class itself
        return UniformIntegerHyperparameter(
            name=self.name,
            lower=int(np.ceil(self.lower)),
            upper=int(np.floor(self.upper)),
            default_value=int(np.rint(self.default_value)),
            q=int(np.rint(self.q)),
            log=self.log,
        )

    def _sample(self, rs: np.random, size: Optional[int] = None) -> Union[float, np.ndarray]:
        return rs.uniform(size=size)

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        if np.isnan(vector).any():
            raise ValueError('Vector %s contains NaN\'s' % vector)
        vector = vector * (self._upper - self._lower) + self._lower
        if self.log:
            vector = np.exp(vector)
        if self.q is not None:
            vector = np.rint((vector - self.lower) / self.q) * self.q + self.lower
            vector = np.minimum(vector, self.upper)
            vector = np.maximum(vector, self.lower)
        return np.maximum(self.lower, np.minimum(self.upper, vector))

    cpdef double _transform_scalar(self, double scalar):
        if scalar != scalar:
            raise ValueError('Number %s is NaN' % scalar)
        scalar = scalar * (self._upper - self._lower) + self._lower
        if self.log:
            scalar = math.exp(scalar)
        if self.q is not None:
            scalar = np.round((scalar - self.lower) / self.q) * self.q + self.lower
            scalar = min(scalar, self.upper)
            scalar = max(scalar, self.lower)
        scalar = min(self.upper, max(self.lower, scalar))
        return scalar

    def _inverse_transform(self, vector: Union[np.ndarray, None]
                           ) -> Union[np.ndarray, float, int]:
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
        number: int = 4,
        transform: bool = False,
        std: float = 0.2
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

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these).

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        # everything that comes into _pdf for a uniform variable should
        # already be in [0, 1]-range, and if not, it's outside the upper
        # or lower bound.
        ub = 1
        lb = 0
        inside_range = ((lb <= vector) & (vector <= ub)).astype(int)
        return inside_range / (self.upper - self.lower)

    def get_max_density(self) -> float:
        return 1 / (self.upper - self.lower)

    def get_size(self) -> float:
        if self.q is None:
            return np.inf
        else:
            return np.rint((self.upper - self.lower) / self.q) + 1

cdef class NormalFloatHyperparameter(FloatHyperparameter):
    cdef public mu
    cdef public sigma

    def __init__(self, name: str, mu: Union[int, float], sigma: Union[int, float],
                 default_value: Union[None, float] = None,
                 q: Union[int, float, None] = None, log: bool = False,
                 lower: Optional[Union[float, int]] = None,
                 upper: Optional[Union[float, int]] = None,
                 meta: Optional[Dict] = None) -> None:
        r"""
        A normally distributed float hyperparameter.

        Its values are sampled from a normal distribution
        :math:`\mathcal{N}(\mu, \sigma^2)`.

        >>> from ConfigSpace import NormalFloatHyperparameter
        >>>
        >>> NormalFloatHyperparameter('n', mu=0, sigma=1, log=False)
        n, Type: NormalFloat, Mu: 0.0 Sigma: 1.0, Default: 0.0

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        mu : int, float
            Mean of the distribution
        sigma : int, float
            Standard deviation of the distribution
        default_value : int, float, optional
            Sets the default value of a hyperparameter to a given value
        q : int, float, optional
            Quantization factor
        log : bool, optional
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Default to ``False``
        lower : int, float, optional
            Lower bound of a range of values from which the hyperparameter will be sampled
        upper : int, float, optional
            Upper bound of a range of values from which the hyperparameter will be sampled
        meta : Dict, optional
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

        if (lower is not None) ^ (upper is not None):
            raise ValueError("Only one bound was provided when both lower and upper bounds must be provided.")

        if lower is not None and upper is not None:
            self.lower = float(lower)
            self.upper = float(upper)

            if self.lower >= self.upper:
                raise ValueError("Upper bound %f must be larger than lower bound "
                                "%f for hyperparameter %s" %
                                (self.upper, self.lower, name))
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
            if self.q is not None:
                # There can be weird rounding errors, so we compare the result against self.q, see
                # In [13]: 2.4 % 0.2
                # Out[13]: 0.1999999999999998
                if np.round((self.upper - self.lower) % self.q, 10) not in (0, self.q):
                    raise ValueError(
                        'Upper bound (%f) - lower bound (%f) must be a multiple of q (%f)'
                        % (self.upper, self.lower, self.q)
                    )

    def __repr__(self) -> str:
        repr_str = io.StringIO()

        if self.lower is None or self.upper is None:
            repr_str.write("%s, Type: NormalFloat, Mu: %s Sigma: %s, Default: %s" % (self.name, repr(self.mu), repr(self.sigma), repr(self.default_value)))
        else:
            repr_str.write("%s, Type: NormalFloat, Mu: %s Sigma: %s, Range: [%s, %s], Default: %s" % (self.name, repr(self.mu), repr(self.sigma), repr(self.lower), repr(self.upper), repr(self.default_value)))

        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

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
            self.mu == other.mu and
            self.sigma == other.sigma and
            self.log == other.log and
            self.q == other.q and
            self.lower == other.lower and
            self.upper == other.upper
        )

    def __copy__(self):
        return NormalFloatHyperparameter(
            name=self.name,
            default_value=self.default_value,
            mu=self.mu,
            sigma=self.sigma,
            log=self.log,
            q=self.q,
            lower=self.lower,
            upper=self.upper,
            meta=self.meta
        )

    def __hash__(self):
        return hash((self.name, self.mu, self.sigma, self.log, self.q))

    def to_uniform(self, z: int = 3) -> 'UniformFloatHyperparameter':
        if self.lower is None or self.upper is None:
            lb = self.mu - (z * self.sigma)
            ub = self.mu + (z * self.sigma)
        else:
            lb = self.lower
            ub = self.upper

        return UniformFloatHyperparameter(self.name,
                                          lb,
                                          ub,
                                          default_value=self.default_value,
                                          q=self.q, log=self.log, meta=self.meta)

    def check_default(self, default_value: Union[int, float]) -> Union[int, float]:
        if default_value is None:
            if self.log:
                return self._transform_scalar(self.mu)
            else:
                return self.mu

        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def to_integer(self) -> 'NormalIntegerHyperparameter':
        if self.q is None:
            q_int = None
        else:
            q_int = int(np.rint(self.q))
        if self.lower is None:
            lower = None
            upper = None
        else:
            lower=np.ceil(self.lower)
            upper=np.floor(self.upper)

        return NormalIntegerHyperparameter(self.name, int(np.rint(self.mu)), self.sigma,
                                           lower=lower, upper=upper,
                                           default_value=int(np.rint(self.default_value)),
                                           q=q_int, log=self.log)

    def is_legal(self, value: Union[float]) -> bool:
        return isinstance(value, float) or isinstance(value, int)

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return isinstance(value, float) or isinstance(value, int)

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None
                ) -> Union[np.ndarray, float]:

        if self.lower == None:
            mu = self.mu
            sigma = self.sigma
            return rs.normal(mu, sigma, size=size)
        else:
            mu = self.mu
            sigma = self.sigma
            lower = self._lower
            upper = self._upper
            a = (lower - mu) / sigma
            b = (upper - mu) / sigma

            return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rs)

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        if np.isnan(vector).any():
            raise ValueError('Vector %s contains NaN\'s' % vector)
        if self.log:
            vector = np.exp(vector)
        if self.q is not None:
            vector = np.rint(vector / self.q) * self.q
        return vector

    cpdef double _transform_scalar(self, double scalar):
        if scalar != scalar:
            raise ValueError('Number %s is NaN' % scalar)
        if self.log:
            scalar = math.exp(scalar)
        if self.q is not None:
            scalar = np.round(scalar / self.q) * self.q
        return scalar

    def _inverse_transform(self, vector: Optional[np.ndarray]) -> Union[float, np.ndarray]:
        if vector is None:
            return np.NaN

        if self.log:
            vector = np.log(vector)
        return vector

    def get_neighbors(self, value: float, rs: np.random.RandomState, number: int = 4,
                      transform: bool = False) -> List[float]:
        neighbors = []
        for i in range(number):
            new_value = rs.normal(value, self.sigma)

            if self.lower is not None and self.upper is not None:
                    new_value = min(max(new_value, self.lower), self.upper)

            neighbors.append(new_value)
        return neighbors

    def get_size(self) -> float:
        if self.q is None:
            return np.inf
        elif self.lower is None:
            return np.inf
        else:
            return np.rint((self.upper - self.lower) / self.q) + 1

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these).

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        mu = self.mu
        sigma = self.sigma
        if self.lower == None:
            return norm(loc=mu, scale=sigma).pdf(vector)
        else:
            mu = self.mu
            sigma = self.sigma
            lower = self._lower
            upper = self._upper
            a = (lower - mu) / sigma
            b = (upper - mu) / sigma

            return truncnorm(a, b, loc=mu, scale=sigma).pdf(vector)

    def get_max_density(self) -> float:
        if self.lower is None:
            return self._pdf(np.array([self.mu]))[0]

        if self.mu < self._lower:
            return self._pdf(np.array([self._lower]))[0]
        elif self.mu > self._upper:
            return self._pdf(np.array([self._upper]))[0]
        else:
            return self._pdf(np.array([self.mu]))[0]


cdef class BetaFloatHyperparameter(UniformFloatHyperparameter):
    cdef public alpha
    cdef public beta

    def __init__(self, name: str, alpha: Union[int, float], beta: Union[int, float],
                 lower: Union[float, int],
                 upper: Union[float, int],
                 default_value: Union[None, float] = None,
                 q: Union[int, float, None] = None, log: bool = False,
                 meta: Optional[Dict] = None) -> None:
        r"""
        A beta distributed float hyperparameter. The 'lower' and 'upper' parameters move the
        distribution from the [0, 1]-range and scale it appropriately, but the shape of the
        distribution is preserved as if it were in [0, 1]-range.

        Its values are sampled from a beta distribution
        :math:`Beta(\alpha, \beta)`.

        >>> from ConfigSpace import BetaFloatHyperparameter
        >>>
        >>> BetaFloatHyperparameter('b', alpha=3, beta=2, lower=1, upper=4, log=False)
        b, Type: BetaFloat, Alpha: 3.0 Beta: 2.0, Range: [1.0, 4.0], Default: 3.0

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        alpha : int, float
            Alpha parameter of the normalized beta distribution
        beta : int, float
            Beta parameter of the normalized beta distribution
        lower : int, float
            Lower bound of a range of values from which the hyperparameter will be sampled.
            The Beta disribution gets scaled by the total range of the hyperparameter.
        upper : int, float
            Upper bound of a range of values from which the hyperparameter will be sampled.
            The Beta disribution gets scaled by the total range of the hyperparameter.
        default_value : int, float, optional
            Sets the default value of a hyperparameter to a given value
        q : int, float, optional
            Quantization factor
        log : bool, optional
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Default to ``False``
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        # TODO - we cannot use the check_default of UniformFloat (but everything else),
        # but we still need to overwrite it. Thus, we first just need it not to raise an
        # error, which we do by setting default_value = upper - lower / 2 to not raise an error,
        # then actually call check_default once we have alpha and beta, and are not inside
        # UniformFloatHP.
        super(BetaFloatHyperparameter, self).__init__(
            name, lower, upper, (upper + lower) / 2, q, log, meta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        if (alpha < 1) or (beta < 1):
            raise ValueError("Please provide values of alpha and beta larger than or equal to\
             1 so that the probability density is finite.")

        if (self.q is not None) and (self.log is not None) and (default_value is None):
            warnings.warn('Logscale and quantization together results in incorrect default values. '
                          'We recommend specifying a default value manually for this specific case.')

        self.default_value = self.check_default(default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: BetaFloat, Alpha: %s Beta: %s, Range: [%s, %s], Default: %s" % (self.name, repr(self.alpha), repr(self.beta), repr(self.lower), repr(self.upper), repr(self.default_value)))

        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

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
            self.alpha == other.alpha and
            self.beta == other.beta and
            self.log == other.log and
            self.q == other.q and
            self.lower == other.lower and
            self.upper == other.upper
        )

    def __copy__(self):
        return BetaFloatHyperparameter(
            name=self.name,
            default_value=self.default_value,
            alpha=self.alpha,
            beta=self.beta,
            log=self.log,
            q=self.q,
            lower=self.lower,
            upper=self.upper,
            meta=self.meta
        )

    def __hash__(self):
        return hash((self.name, self.alpha, self.beta, self.lower, self.upper, self.log, self.q))

    def to_uniform(self) -> 'UniformFloatHyperparameter':
        return UniformFloatHyperparameter(self.name,
                                          self.lower,
                                          self.upper,
                                          default_value=self.default_value,
                                          q=self.q, log=self.log, meta=self.meta)

    def check_default(self, default_value: Union[int, float, None]) -> Union[int, float]:
        # return mode as default
        # TODO - for log AND quantization together specifially, this does not give the exact right
        # value, due to the bounds _lower and _upper being adjusted when quantizing in
        # UniformFloat.
        if default_value is None:
            if (self.alpha > 1) or (self.beta > 1):
                normalized_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)
            else:
                # If both alpha and beta are 1, we have a uniform distribution.
                normalized_mode = 0.5

            ub = self._inverse_transform(self.upper)
            lb = self._inverse_transform(self.lower)
            scaled_mode = normalized_mode * (ub - lb) + lb
            return self._transform_scalar(scaled_mode)

        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def to_integer(self) -> 'BetaIntegerHyperparameter':
        if self.q is None:
            q_int = None
        else:
            q_int = int(np.rint(self.q))

        lower = int(np.ceil(self.lower))
        upper = int(np.floor(self.upper))
        default_value = int(np.rint(self.default_value))
        return BetaIntegerHyperparameter(self.name, lower=lower, upper=upper, alpha=self.alpha, beta=self.beta,
                                           default_value=int(np.rint(self.default_value)),
                                           q=q_int, log=self.log)

    def is_legal(self, value: Union[float]) -> bool:
        if isinstance(value, (float, int)):
            return self.upper >= value >= self.lower
        return False


    cpdef bint is_legal_vector(self, DTYPE_t value):
        return self._upper >= value >= self._lower

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None
                ) -> Union[np.ndarray, float]:
        alpha = self.alpha
        beta = self.beta
        return spbeta(alpha, beta).rvs(size=size, random_state=rs)

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these).

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        ub = self._inverse_transform(self.upper)
        lb = self._inverse_transform(self.lower)
        alpha = self.alpha
        beta = self.beta
        return spbeta(alpha, beta, loc=lb, scale=ub-lb).pdf(vector) \
        * (ub-lb) / (self._upper - self._lower)

    def get_max_density(self) -> float:
        if (self.alpha > 1) or (self.beta > 1):
            normalized_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)
        elif self.alpha < self.beta:
            normalized_mode = 0
        elif self.alpha > self.beta:
            normalized_mode = 1
        else:
            normalized_mode = 0.5

        ub = self._inverse_transform(self.upper)
        lb = self._inverse_transform(self.lower)
        scaled_mode = normalized_mode * (ub - lb) + lb

        # Since _pdf takes only a numpy array, we have to create the array,
        # and retrieve the element in the first (and only) spot in the array
        return self._pdf(np.array([scaled_mode]))[0]

cdef class UniformIntegerHyperparameter(IntegerHyperparameter):
    def __init__(self, name: str, lower: int, upper: int, default_value: Union[int, None] = None,
                 q: Union[int, None] = None, log: bool = False,
                 meta: Optional[Dict] = None) -> None:
        """
        A uniformly distributed integer hyperparameter.

        Its values are sampled from a uniform distribution
        with bounds ``lower`` and ``upper``.

        >>> from ConfigSpace import UniformIntegerHyperparameter
        >>>
        >>> UniformIntegerHyperparameter(name='u', lower=10, upper=100, log=False)
        u, Type: UniformInteger, Range: [10, 100], Default: 55

        Parameters
        ----------
        name : str
            Name of the hyperparameter with which it can be accessed
        lower : int
            Lower bound of a range of values from which the hyperparameter will be sampled
        upper : int
            upper bound
        default_value : int, optional
            Sets the default value of a hyperparameter to a given value
        q : int, optional
            Quantization factor
        log : bool, optional
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Defaults to ``False``
        meta : Dict, optional
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
                if (self.upper - self.lower) % self.q != 0:
                    raise ValueError(
                        'Upper bound (%d) - lower bound (%d) must be a multiple of q (%d)'
                        % (self.upper, self.lower, self.q)
                    )
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
                                               log=self.log,
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
            repr_str.write(", Q: %s" % repr(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None
                ) -> Union[np.ndarray, float]:
        value = self.ufhp._sample(rs, size=size)
        # Map all floats which belong to the same integer value to the same
        # float value by first transforming it to an integer and then
        # transforming it back to a float between zero and one
        value = self._transform(value)
        value = self._inverse_transform(value)
        return value

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        vector = self.ufhp._transform_vector(vector)
        if self.q is not None:
            vector = np.rint((vector - self.lower) / self.q) * self.q + self.lower
            vector = np.minimum(vector, self.upper)
            vector = np.maximum(vector, self.lower)

        return np.rint(vector)

    cpdef long long _transform_scalar(self, double scalar):
        scalar = self.ufhp._transform_scalar(scalar)
        if self.q is not None:
            scalar = np.round((scalar - self.lower) / self.q) * self.q + self.lower
            scalar = min(scalar, self.upper)
            scalar = max(scalar, self.lower)
        return int(np.round(scalar))

    def _inverse_transform(self, vector: Union[np.ndarray, float, int]
                           ) -> Union[np.ndarray, float, int]:
        return self.ufhp._inverse_transform(vector)

    def is_legal(self, value: int) -> bool:
        if not (isinstance(value, (int, np.int32, np.int64))):
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

    def get_num_neighbors(self, value = None) -> int:
        # If there is a value in the range, then that value is not a neighbor of itself
        # so we need to remove one
        if value is not None and self.lower <= value <= self.upper:
            return self.upper - self.lower - 1
        else:
            return self.upper - self.lower

    def get_neighbors(
        self,
        value: float,
        rs: np.random.RandomState,
        number: int = 4,
        transform: bool = False,
        std: float = 0.2,
    ) -> List[int]:
        """Get the neighbors of a value

        NOTE
        ----
        **This assumes the value is in the unit-hypercube [0, 1]**

        Parameters
        ----------
        value: float
            The value to get neighbors around. This assume the ``value`` has been
            converted to the [0, 1] range which can be done with ``_inverse_transform``.

        rs: RandomState
            The random state to use

        number: int = 4
            How many neighbors to get

        transform: bool = False
            Whether to transform this value from the unit cube, back to the
            hyperparameter's specified range of values.

        std: float = 0.2
            The std. dev. to use in the [0, 1] hypercube space while sampling
            for neighbors.

        Returns
        -------
        List[int]
            Some ``number`` of neighbors centered around ``value``.
        """
        assert 0 <= value <= 1, (
            "For get neighbors of UniformIntegerHyperparameter, the value"
            " if assumed to be in the unit-hypercube [0, 1]. If this was not"
            " the behaviour assumed, please raise a ticket on github."
        )
        assert number < 1000000, (
            "Can only generate less than 1 million neighbors."
        )
        # Convert python values to cython ones
        cdef long long center = self._transform(value)
        cdef long long lower = self.lower
        cdef long long upper = self.upper
        cdef unsigned int n_requested = number
        cdef unsigned long long n_neighbors = upper - lower - 1
        cdef long long stepsize = self.q if self.q is not None else 1

        neighbors = []

        cdef long long v  # A value that's possible to return
        if n_neighbors < n_requested:

            for v in range(lower, center):
                neighbors.append(v)

            for v in range(center + 1, upper):
                neighbors.append(v)

            if transform:
                return neighbors
            else:
                return self._inverse_transform(np.asarray(neighbors)).tolist()

        # A truncated normal between 0 and 1, centered on the value with a scale of std.
        # This will be sampled from and converted to the corresponding int value
        # However, this is too slow - we use the "poor man's truncnorm below"
        # cdef np.ndarray float_indices = truncnorm.rvs(
        #     a=(0 - value) / std,
        #     b=(1 - value) / std,
        #     loc=value,
        #     scale=std,
        #     size=number,
        #     random_state=rs
        # )
        # We sample five times as many values as needed and weed them out below
        # (perform rejection sampling and make sure we don't sample any neighbor twice)
        # This increases our chances of not having to fill the neighbors list by calling
        # `center_range`
        # Five is an arbitrary number and can probably be tuned to reduce overhead
        cdef np.ndarray float_indices = rs.normal(value, std, size=number * 5)
        cdef np.ndarray mask = (float_indices >= 0) & (float_indices <= 1)
        float_indices = float_indices[mask]

        cdef np.ndarray possible_neighbors_as_array = self._transform_vector(float_indices).astype(np.longlong)
        cdef long long [:] possible_neighbors = possible_neighbors_as_array

        cdef unsigned int n_neighbors_generated = 0
        cdef unsigned int n_candidates = len(float_indices)
        cdef unsigned int candidate_index = 0
        cdef set seen = {center}
        while n_neighbors_generated < n_requested and candidate_index < n_candidates:
            v = possible_neighbors[candidate_index]
            if v not in seen:
                seen.add(v)
                n_neighbors_generated += 1
            candidate_index += 1

        if n_neighbors_generated < n_requested:
            numbers_around = center_range(center, lower, upper, stepsize)

            while n_neighbors_generated < n_requested:
                v = next(numbers_around)
                if v not in seen:
                    seen.add(v)
                    n_neighbors_generated += 1

        seen.remove(center)
        neighbors = list(seen)
        if transform:
            return neighbors
        else:
            return self._inverse_transform(np.array(neighbors)).tolist()


    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these). Optimally, an IntegerHyperparameter
        should have a corresponding float, which can be utlized for the calls
        to the probability density function (see e.g. NormalIntegerHyperparameter)

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        return self.ufhp._pdf(vector)

    def get_max_density(self) -> float:
        lb = self.lower
        ub = self.upper
        return 1 / (ub - lb + 1)

    def get_size(self) -> float:
        if self.q is None:
            q = 1
        else:
            q = self.q
        return np.rint((self.upper - self.lower) / q) + 1


cdef class NormalIntegerHyperparameter(IntegerHyperparameter):
    cdef public mu
    cdef public sigma
    cdef public nfhp
    cdef normalization_constant


    def __init__(self, name: str, mu: int, sigma: Union[int, float],
                 default_value: Union[int, None] = None, q: Union[None, int] = None,
                 log: bool = False,
                 lower: Optional[int] = None,
                 upper: Optional[int] = None,
                 meta: Optional[Dict] = None) -> None:
        r"""
        A normally distributed integer hyperparameter.

        Its values are sampled from a normal distribution
        :math:`\mathcal{N}(\mu, \sigma^2)`.

        >>> from ConfigSpace import NormalIntegerHyperparameter
        >>>
        >>> NormalIntegerHyperparameter(name='n', mu=0, sigma=1, log=False)
        n, Type: NormalInteger, Mu: 0 Sigma: 1, Default: 0

        Parameters
        ----------
        name : str
            Name of the hyperparameter with which it can be accessed
        mu : int
            Mean of the distribution, from which hyperparameter is sampled
        sigma : int, float
            Standard deviation of the distribution, from which
            hyperparameter is sampled
        default_value : int, optional
            Sets the default value of a hyperparameter to a given value
        q : int, optional
            Quantization factor
        log : bool, optional
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Defaults to ``False``
        lower : int, float, optional
            Lower bound of a range of values from which the hyperparameter will be sampled
        upper : int, float, optional
            Upper bound of a range of values from which the hyperparameter will be sampled
        meta : Dict, optional
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

        if (lower is not None) ^ (upper is not None):
            raise ValueError("Only one bound was provided when both lower and upper bounds must be provided.")

        if lower is not None and upper is not None:
            self.upper = self.check_int(upper, "upper")
            self.lower = self.check_int(lower, "lower")
            if self.lower >= self.upper:
                raise ValueError("Upper bound %d must be larger than lower bound "
                                "%d for hyperparameter %s" %
                                (self.lower, self.upper, name))
            elif log and self.lower <= 0:
                raise ValueError("Negative lower bound (%d) for log-scale "
                                "hyperparameter %s is forbidden." %
                                (self.lower, name))
            self.lower = lower
            self.upper = upper

        self.nfhp = NormalFloatHyperparameter(self.name,
                                              self.mu,
                                              self.sigma,
                                              log=self.log,
                                              q=self.q,
                                              lower=self.lower,
                                              upper=self.upper,
                                              default_value=default_value)

        self.default_value = self.check_default(default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)

        if (self.lower is None) or (self.upper is None):
            # Since a bound is missing, the pdf cannot be normalized. Working with the unnormalized variant)
            self.normalization_constant = 1
        else:
            self.normalization_constant = self._compute_normalization()

    def __repr__(self) -> str:
        repr_str = io.StringIO()

        if self.lower is None or self.upper is None:
            repr_str.write("%s, Type: NormalInteger, Mu: %s Sigma: %s, Default: %s" % (self.name, repr(self.mu), repr(self.sigma), repr(self.default_value)))
        else:
            repr_str.write("%s, Type: NormalInteger, Mu: %s Sigma: %s, Range: [%s, %s], Default: %s" % (self.name, repr(self.mu), repr(self.sigma), repr(self.lower), repr(self.upper), repr(self.default_value)))

        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

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
            self.mu == other.mu and
            self.sigma == other.sigma and
            self.log == other.log and
            self.q == other.q and
            self.lower == other.lower and
            self.upper == other.upper and
            self.default_value == other.default_value
        )

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
            lower=self.lower,
            upper=self.upper,
            meta=self.meta
        )

    def to_uniform(self, z: int = 3) -> 'UniformIntegerHyperparameter':
        if self.lower is None or self.upper is None:
            lb = np.round(int(self.mu - (z * self.sigma)))
            ub = np.round(int(self.mu + (z * self.sigma)))
        else:
            lb = self.lower
            ub = self.upper

        return UniformIntegerHyperparameter(self.name,
                                            lb,
                                            ub,
                                            default_value=self.default_value,
                                            q=self.q, log=self.log, meta=self.meta)

    def is_legal(self, value: int) -> bool:
        return isinstance(value, (int, np.int32, np.int64))

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return isinstance(value, float) or isinstance(value, int)

    def check_default(self, default_value: int) -> int:
        if default_value is None:
            if self.log:
                return self._transform_scalar(self.mu)
            else:
                return self.mu

        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None
                ) -> Union[np.ndarray, float]:
        value = self.nfhp._sample(rs, size=size)
        # Map all floats which belong to the same integer value to the same
        # float value by first transforming it to an integer and then
        # transforming it back to a float between zero and one
        value = self._transform(value)
        value = self._inverse_transform(value)
        return value

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        vector = self.nfhp._transform_vector(vector)
        return np.rint(vector)

    cpdef long long _transform_scalar(self, double scalar):
        scalar = self.nfhp._transform_scalar(scalar)
        return int(np.round(scalar))

    def _inverse_transform(self, vector: Union[np.ndarray, float, int]
                           ) -> Union[np.ndarray, float]:
        return self.nfhp._inverse_transform(vector)

    def has_neighbors(self) -> bool:
        return True

    def get_neighbors(
        self,
        value: Union[int, float],
        rs: np.random.RandomState,
        number: int = 4,
        transform: bool = False,
    ) -> List[int]:
        stepsize = self.q if self.q is not None else 1
        bounded = self.lower is not None
        mu = self.mu
        sigma = self.sigma

        neighbors: set[int] = set()
        center = self._transform(value)

        if bounded:
            float_indices = norm.rvs(
                loc=mu,
                scale=sigma,
                size=number,
                random_state=rs,
            )
        else:
            float_indices = truncnorm(
                a = (self.lower - mu) / sigma,
                b = (self.upper - mu) / sigma,
                loc=center,
                scale=sigma,
                size=number,
                random_state=rs,
            )

        possible_neighbors = self._transform_vector(float_indices).astype(np.longlong)

        for possible_neighbor in possible_neighbors:
            # If we already happen to have this neighbor, pick the closest
            # number around it that is not arelady included
            if possible_neighbor in neighbors or possible_neighbor == center:

                if bounded:
                    numbers_around = center_range(possible_neighbor, self.lower, self.upper, stepsize)
                else:
                    decrement_count = count(possible_neighbor - stepsize, step=-stepsize)
                    increment_count = count(possible_neighbor + stepsize, step=stepsize)
                    numbers_around = roundrobin(decrement_count, increment_count)

                valid_numbers_around = (
                    n for n in numbers_around
                    if (n not in neighbors and n != center)
                )
                possible_neighbor = next(valid_numbers_around, None)

                if possible_neighbor is None:
                    raise ValueError(
                        f"Found no more eligble neighbors for value {center}"
                        f"\nfound {neighbors}"
                    )

            # We now have a valid sample, add it to the list of neighbors
            neighbors.add(possible_neighbor)

            if transform:
                return [self._transform(neighbor) for neighbor in neighbors]
            else:
                return list(neighbors)

    def _compute_normalization(self):
        if self.lower is None:
            warnings.warn('Cannot normalize the pdf exactly for a NormalIntegerHyperparameter'
            f' {self.name} without bounds. Skipping normalization for that hyperparameter.')
            return 1

        else:
            chunks = arange_chunked(self.lower, self.upper + 1, chunk_size=ARANGE_CHUNKSIZE)
            return sum(self.nfhp.pdf(chunk).sum() for chunk in chunks)

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these). Optimally, an IntegerHyperparameter
        should have a corresponding float, which can be utlized for the calls
        to the probability density function (see e.g. NormalIntegerHyperparameter)

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        return self.nfhp._pdf(vector) / self.normalization_constant

    def get_max_density(self):
        chunks = arange_chunked(self.lower, self.upper + 1, chunk_size=ARANGE_CHUNKSIZE)
        maximum = max(self.nfhp.pdf(chunk).max() for chunk in chunks)
        return maximum / self.normalization_constant

    def get_size(self) -> float:
        if self.lower is None:
            return np.inf
        else:
            if self.q is None:
                q = 1
            else:
                q = self.q
            return np.rint((self.upper - self.lower) / self.q) + 1


cdef class BetaIntegerHyperparameter(UniformIntegerHyperparameter):
    cdef public alpha
    cdef public beta
    cdef public bfhp
    cdef normalization_constant


    def __init__(self, name: str, alpha: Union[int, float], beta: Union[int, float],
                 lower: Union[int, float],
                 upper: Union[int, float],
                 default_value: Union[int, None] = None, q: Union[None, int] = None,
                 log: bool = False,
                 meta: Optional[Dict] = None) -> None:
        r"""
        A beta distributed integer hyperparameter. The 'lower' and 'upper' parameters move the
        distribution from the [0, 1]-range and scale it appropriately, but the shape of the
        distribution is preserved as if it were in [0, 1]-range.

        Its values are sampled from a beta distribution
        :math:`Beta(\alpha, \beta)`.

        >>> from ConfigSpace import BetaIntegerHyperparameter
        >>>
        >>> BetaIntegerHyperparameter('b', alpha=3, beta=2, lower=1, upper=4, log=False)
        b, Type: BetaInteger, Alpha: 3.0 Beta: 2.0, Range: [1, 4], Default: 3


        Parameters
        ----------
        name : str
            Name of the hyperparameter with which it can be accessed
        alpha : int, float
            Alpha parameter of the distribution, from which hyperparameter is sampled
        beta : int, float
            Beta parameter of the distribution, from which
            hyperparameter is sampled
        lower : int, float
            Lower bound of a range of values from which the hyperparameter will be sampled
        upper : int, float
            Upper bound of a range of values from which the hyperparameter will be sampled
        default_value : int, optional
            Sets the default value of a hyperparameter to a given value
        q : int, optional
            Quantization factor
        log : bool, optional
            If ``True``, the values of the hyperparameter will be sampled
            on a logarithmic scale. Defaults to ``False``
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.

        """
        super(BetaIntegerHyperparameter, self).__init__(
            name, lower, upper, np.round((upper + lower) / 2), q, log, meta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        if (alpha < 1) or (beta < 1):
            raise ValueError("Please provide values of alpha and beta larger than or equal to\
             1 so that the probability density is finite.")
        if self.q is None:
            q = 1
        else:
            q = self.q
        self.bfhp = BetaFloatHyperparameter(self.name,
                                              self.alpha,
                                              self.beta,
                                              log=self.log,
                                              q=q,
                                              lower=self.lower,
                                              upper=self.upper,
                                              default_value=self.default_value)

        self.default_value = self.check_default(default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)
        self.normalization_constant = self._compute_normalization()

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: BetaInteger, Alpha: %s Beta: %s, Range: [%s, %s], Default: %s" % (self.name, repr(self.alpha), repr(self.beta), repr(self.lower), repr(self.upper), repr(self.default_value)))

        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

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
            self.alpha == other.alpha and
            self.beta == other.beta and
            self.log == other.log and
            self.q == other.q and
            self.lower == other.lower and
            self.upper == other.upper
        )

    def __hash__(self):
        return hash((self.name, self.alpha, self.beta, self.lower, self.upper, self.log, self.q))

    def __copy__(self):
        return BetaIntegerHyperparameter(
            name=self.name,
            default_value=self.default_value,
            alpha=self.alpha,
            beta=self.beta,
            log=self.log,
            q=self.q,
            lower=self.lower,
            upper=self.upper,
            meta=self.meta
        )

    def to_uniform(self) -> 'UniformIntegerHyperparameter':
        return UniformIntegerHyperparameter(self.name,
                                            self.lower,
                                            self.upper,
                                            default_value=self.default_value,
                                            q=self.q, log=self.log, meta=self.meta)


    def check_default(self, default_value: Union[int, float, None]) -> int:
        if default_value is None:
            # Here, we just let the BetaFloat take care of the default value
            # computation, and just tansform it accordingly
            value = self.bfhp.check_default(None)
            value = self._inverse_transform(value)
            value = self._transform(value)
            return value

        if self.is_legal(default_value):
            return default_value
        else:
            raise ValueError('Illegal default value {}'.format(default_value))

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None
                ) -> Union[np.ndarray, float]:
        value = self.bfhp._sample(rs, size=size)
        # Map all floats which belong to the same integer value to the same
        # float value by first transforming it to an integer and then
        # transforming it back to a float between zero and one
        value = self._transform(value)
        value = self._inverse_transform(value)
        return value

    def _compute_normalization(self):
        chunks = arange_chunked(self.lower, self.upper + 1, chunk_size=ARANGE_CHUNKSIZE)
        return sum(self.bfhp.pdf(chunk).sum() for chunk in chunks)

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these). Optimally, an IntegerHyperparameter
        should have a corresponding float, which can be utlized for the calls
        to the probability density function (see e.g. NormalIntegerHyperparameter)

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        return self.bfhp._pdf(vector) / self.normalization_constant

    def get_max_density(self):
        chunks = arange_chunked(self.lower, self.upper + 1, chunk_size=ARANGE_CHUNKSIZE)
        maximum = max(self.bfhp.pdf(chunk).max() for chunk in chunks)
        return maximum / self.normalization_constant


cdef class CategoricalHyperparameter(Hyperparameter):
    cdef public tuple choices
    cdef public tuple weights
    cdef public int num_choices
    cdef public tuple probabilities
    cdef list choices_vector
    cdef set _choices_set

    # TODO add more magic for automated type recognition
    # TODO move from list to tuple for choices argument
    def __init__(
        self,
        name: str,
        choices: Union[List[Union[str, float, int]], Tuple[Union[float, int, str]]],
        default_value: Union[int, float, str, None] = None,
        meta: Optional[Dict] = None,
        weights: Optional[Sequence[Union[int, float]]] = None
    ) -> None:
        """
        A categorical hyperparameter.

        Its values are sampled from a set of ``values``.

        ``None`` is a forbidden value, please use a string constant instead and parse
        it in your own code, see `here <https://github.com/automl/ConfigSpace/issues/159>_`
        for further details.

        >>> from ConfigSpace import CategoricalHyperparameter
        >>>
        >>> CategoricalHyperparameter('c', choices=['red', 'green', 'blue'])
        c, Type: Categorical, Choices: {red, green, blue}, Default: red

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        choices : list or tuple with str, float, int
            Collection of values to sample hyperparameter from
        default_value : int, float, str, optional
            Sets the default value of the hyperparameter to a given value
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        weights: Sequence[int | float] | None = None
            List of weights for the choices to be used (after normalization) as
            probabilities during sampling, no negative values allowed
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
        if isinstance(choices, set):
            raise TypeError('Using a set of choices is prohibited as it can result in '
                            'non-deterministic behavior. Please use a list or a tuple.')
        if isinstance(weights, set):
            raise TypeError('Using a set of weights is prohibited as it can result in '
                            'non-deterministic behavior. Please use a list or a tuple.')
        self.choices = tuple(choices)
        if weights is not None:
            self.weights = tuple(weights)
        self.probabilities = self._get_probabilities(choices=self.choices, weights=weights)
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
        # if the probability distribution is not uniform, write out the probabilities
        if not np.all(self.probabilities == self.probabilities[0]):
            repr_str.write(", Probabilities: %s" % str(self.probabilities))
        repr_str.seek(0)
        return repr_str.getvalue()

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

        if self.probabilities is not None:
            ordered_probabilities_self = {
                choice: self.probabilities[i] for i, choice in enumerate(self.choices)
            }
        else:
            ordered_probabilities_self = None
        if other.probabilities is not None:
            ordered_probabilities_other = {
                choice: (
                    other.probabilities[other.choices.index(choice)]
                    if choice in other.choices else
                    None
                )
                for choice in self.choices
            }
        else:
            ordered_probabilities_other = None

        return (
            self.name == other.name and
            set(self.choices) == set(other.choices) and
            self.default_value == other.default_value and
            (
                (ordered_probabilities_self is None and ordered_probabilities_other is None) or
                ordered_probabilities_self == ordered_probabilities_other or
                (
                    ordered_probabilities_self is None
                    and len(np.unique(list(ordered_probabilities_other.values()))) == 1
                ) or
                (
                    ordered_probabilities_other is None
                    and len(np.unique(list(ordered_probabilities_self.values()))) == 1
                )
             )
        )

    def __hash__(self):
        return hash((self.name, self.choices))

    def __copy__(self):
        return CategoricalHyperparameter(
            name=self.name,
            choices=copy.deepcopy(self.choices),
            default_value=self.default_value,
            weights=copy.deepcopy(self.weights),
            meta=self.meta
        )

    def to_uniform(self) -> 'CategoricalHyperparameter':
        """
        Creates a categorical parameter with equal weights for all choices
        This is used for the uniform configspace when sampling configurations in the local search
        in PiBO: https://openreview.net/forum?id=MMAeCXIa89

        Returns
        ----------
        CategoricalHyperparameter
            An identical parameter as the original, except that all weights are uniform.
        """
        return CategoricalHyperparameter(
            name=self.name,
            choices=copy.deepcopy(self.choices),
            default_value=self.default_value,
            meta=self.meta
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

    def _get_probabilities(self, choices: Tuple[Union[None, str, float, int]],
                           weights: Union[None, List[float]]) -> Union[None, List[float]]:
        if weights is None:
            return tuple(np.ones(len(choices)) / len(choices))

        if len(weights) != len(choices):
            raise ValueError(
                "The list of weights and the list of choices are required to be of same length.")

        weights = np.array(weights)

        if np.all(weights == 0):
            raise ValueError("At least one weight has to be strictly positive.")

        if np.any(weights < 0):
            raise ValueError("Negative weights are not allowed.")

        return tuple(weights / np.sum(weights))

    def check_default(self, default_value: Union[None, str, float, int]
                      ) -> Union[str, float, int]:
        if default_value is None:
            return self.choices[0]
        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None
                ) -> Union[int, np.ndarray]:
        return rs.choice(a=self.num_choices, size=size, replace=True, p=self.probabilities)

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        if np.isnan(vector).any():
            raise ValueError('Vector %s contains NaN\'s' % vector)

        if np.equal(np.mod(vector, 1), 0):
            return self.choices[vector.astype(int)]

        raise ValueError('Can only index the choices of the ordinal '
                         'hyperparameter %s with an integer, but provided '
                         'the following float: %f' % (self, vector))

    def _transform_scalar(self, scalar: Union[float, int]) -> Union[float, int, str]:
        if scalar != scalar:
            raise ValueError('Number %s is NaN' % scalar)

        if scalar % 1 == 0:
            return self.choices[int(scalar)]

        raise ValueError('Can only index the choices of the ordinal '
                         'hyperparameter %s with an integer, but provided '
                         'the following float: %f' % (self, scalar))

    def _transform(self, vector: Union[np.ndarray, float, int, str]
                   ) -> Optional[Union[np.ndarray, float, int]]:
        try:
            if isinstance(vector, np.ndarray):
                return self._transform_vector(vector)
            return self._transform_scalar(vector)
        except ValueError:
            return None

    def _inverse_transform(self, vector: Union[None, str, float, int]) -> Union[int, float]:
        if vector is None:
            return np.NaN
        return self.choices.index(vector)

    def has_neighbors(self) -> bool:
        return len(self.choices) > 1

    def get_num_neighbors(self, value = None) -> int:
        return len(self.choices) - 1

    def get_neighbors(self, value: int, rs: np.random.RandomState,
                      number: Union[int, float] = np.inf, transform: bool = False
                      ) -> List[Union[float, int, str]]:
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

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the original parameter space (the one specified by the user).
        For each parameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) parameter
        space. Only legal values return a positive probability density,
        otherwise zero.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        # this check is to ensure shape is right (and np.shape does not work in cython)
        if vector.ndim != 1:
            raise ValueError("Method pdf expects a one-dimensional numpy array")
        vector = np.array(self._inverse_transform(vector))
        return self._pdf(vector)

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these). For categoricals, each vector gets
        transformed to its corresponding index (but in float form). To be
        able to retrieve the element corresponding to the index, the float
        must be cast to int.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        probs = np.array(self.probabilities)
        res = np.array(probs[vector.astype(int)])
        if res.ndim == 0:
            return res.reshape(-1)
        return res

    def get_max_density(self) -> float:
        return np.max(self.probabilities)

    def get_size(self) -> float:
        return len(self.choices)


cdef class OrdinalHyperparameter(Hyperparameter):
    cdef public tuple sequence
    cdef public int num_elements
    cdef sequence_vector
    cdef value_dict

    def __init__(
        self,
        name: str,
        sequence: Union[List[Union[float, int, str]], Tuple[Union[float, int, str]]],
        default_value: Union[str, int, float, None] = None,
        meta: Optional[Dict] = None
    ) -> None:
        """
        An ordinal hyperparameter.

        Its values are sampled form a ``sequence`` of values.
        The sequence of values from a ordinal hyperparameter is ordered.

        ``None`` is a forbidden value, please use a string constant instead and parse
        it in your own code, see `here <https://github.com/automl/ConfigSpace/issues/159>`_
        for further details.

        >>> from ConfigSpace import OrdinalHyperparameter
        >>>
        >>> OrdinalHyperparameter('o', sequence=['10', '20', '30'])
        o, Type: Ordinal, Sequence: {10, 20, 30}, Default: 10

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed.
        sequence : list or tuple with (str, float, int)
            ordered collection of values to sample hyperparameter from.
        default_value : int, float, str, optional
            Sets the default value of a hyperparameter to a given value.
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """

        # Remark
        # Since the sequence can consist of elements from different types,
        # they are stored into a dictionary in order to handle them as a
        # numeric sequence according to their order/position.
        super(OrdinalHyperparameter, self).__init__(name, meta)
        if len(sequence) > len(set(sequence)):
            raise ValueError(
                "Ordinal Hyperparameter Sequence %s contain duplicate values." % sequence)
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
            self.sequence == other.sequence and
            self.default_value == other.default_value
        )

    def __copy__(self):
        return OrdinalHyperparameter(
                name=self.name,
                sequence=copy.deepcopy(self.sequence),
                default_value=self.default_value,
                meta=self.meta
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

    def check_default(self, default_value: Optional[Union[int, float, str]]
                      ) -> Union[int, float, str]:
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

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        if np.isnan(vector).any():
            raise ValueError('Vector %s contains NaN\'s' % vector)

        if np.equal(np.mod(vector, 1), 0):
            return self.sequence[vector.astype(int)]

        raise ValueError('Can only index the choices of the ordinal '
                         'hyperparameter %s with an integer, but provided '
                         'the following float: %f' % (self, vector))

    def _transform_scalar(self, scalar: Union[float, int]) -> Union[float, int, str]:
        if scalar != scalar:
            raise ValueError('Number %s is NaN' % scalar)

        if scalar % 1 == 0:
            return self.sequence[int(scalar)]

        raise ValueError('Can only index the choices of the ordinal '
                         'hyperparameter %s with an integer, but provided '
                         'the following float: %f' % (self, scalar))

    def _transform(self, vector: Union[np.ndarray, float, int]
                   ) -> Optional[Union[np.ndarray, float, int]]:
        try:
            if isinstance(vector, np.ndarray):
                return self._transform_vector(vector)
            return self._transform_scalar(vector)
        except ValueError:
            return None

    def _inverse_transform(self, vector: Optional[Union[np.ndarray, List, int, str, float]]
                           ) -> Union[float, List[int], List[str], List[float]]:
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

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None) -> int:
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
        # check if there is only one value
        if value == self.sequence[0] and value == self.sequence[max_idx]:
            return 0
        elif value == self.sequence[0] or value == self.sequence[max_idx]:
            return 1
        else:
            return 2

    def get_neighbors(self, value: Union[int, str, float], rs: None, number: int = 0,
                      transform: bool = False) -> List[Union[str, float, int]]:
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

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the hyperparameter in
        the original hyperparameter space (the one specified by the user).
        For each parameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) hyperparameter
        space. Only legal values return a positive probability density,
        otherwise zero. The OrdinalHyperparameter is treated
        as a UniformHyperparameter with regard to its probability density.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        if vector.ndim != 1:
            raise ValueError("Method pdf expects a one-dimensional numpy array")
        return self._pdf(vector)

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the hyperparameter in
        the transformed (and possibly normalized, depends on the hyperparameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these). The OrdinalHyperparameter is treated
        as a UniformHyperparameter with regard to its probability density.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        if not np.all(np.isin(vector, self.sequence)):
            raise ValueError(f'Some element in the vector {vector} is not in the sequence {self.sequence}.')
        return np.ones_like(vector, dtype=np.float64) / self.num_elements

    def get_max_density(self) -> float:
        return 1 / self.num_elements

    def get_size(self) -> float:
        return len(self.sequence)
