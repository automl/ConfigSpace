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
from ConfigSpace.hyperparameters.hyperparameter cimport Hyperparameter
from ConfigSpace.hyperparameters.numerical cimport NumericalHyperparameter
from ConfigSpace.hyperparameters.integer_hyperparameter cimport IntegerHyperparameter
from ConfigSpace.hyperparameters.float_hyperparameter cimport FloatHyperparameter
from ConfigSpace.hyperparameters.uniform_float cimport UniformFloatHyperparameter
from ConfigSpace.hyperparameters.uniform_integer cimport UniformIntegerHyperparameter
from ConfigSpace.hyperparameters.normal_float cimport NormalFloatHyperparameter
from ConfigSpace.hyperparameters.beta_integer cimport BetaIntegerHyperparameter

# OPTIM: Some operations generate an arange which could blowup memory if
# done over the entire space of integers (int32/64).
# To combat this, `arange_chunked` is used in scenarios where reducion
# operations over all the elments could be done in partial steps independantly.
# For example, a sum over the pdf values could be done in chunks.
# This may add some small overhead for smaller ranges but is unlikely to
# be noticable.
ARANGE_CHUNKSIZE = 10_000_000



cdef class BetaFloatHyperparameter(UniformFloatHyperparameter):

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
