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
from ConfigSpace.hyperparameters_.hyperparameter cimport Hyperparameter
from ConfigSpace.hyperparameters_.numerical cimport NumericalHyperparameter
from ConfigSpace.hyperparameters_.integer_hyperparameter cimport IntegerHyperparameter
from ConfigSpace.hyperparameters_.float_hyperparameter cimport FloatHyperparameter
from ConfigSpace.hyperparameters_.uniform_float cimport UniformFloatHyperparameter
from ConfigSpace.hyperparameters_.uniform_integer cimport UniformIntegerHyperparameter
from ConfigSpace.hyperparameters_ import (
    Constant,
    UnParametrizedHyperparameter,
    OrdinalHyperparameter,
    CategoricalHyperparameter,
)

# OPTIM: Some operations generate an arange which could blowup memory if
# done over the entire space of integers (int32/64).
# To combat this, `arange_chunked` is used in scenarios where reducion
# operations over all the elments could be done in partial steps independantly.
# For example, a sum over the pdf values could be done in chunks.
# This may add some small overhead for smaller ranges but is unlikely to
# be noticable.
ARANGE_CHUNKSIZE = 10_000_000


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
