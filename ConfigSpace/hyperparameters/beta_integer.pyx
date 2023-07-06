import io
from typing import Any, Dict, Optional, Union

from scipy.stats import beta as spbeta

import numpy as np
cimport numpy as np
np.import_array()

from ConfigSpace.functional import arange_chunked
from ConfigSpace.hyperparameters.beta_float cimport BetaFloatHyperparameter

# OPTIM: Some operations generate an arange which could blowup memory if
# done over the entire space of integers (int32/64).
# To combat this, `arange_chunked` is used in scenarios where reducion
# operations over all the elments could be done in partial steps independantly.
# For example, a sum over the pdf values could be done in chunks.
# This may add some small overhead for smaller ranges but is unlikely to
# be noticable.
ARANGE_CHUNKSIZE = 10_000_000


cdef class BetaIntegerHyperparameter(UniformIntegerHyperparameter):

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

    def to_uniform(self) -> "UniformIntegerHyperparameter":
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
            raise ValueError("Illegal default value {}".format(default_value))

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
        if self.upper - self.lower > ARANGE_CHUNKSIZE:
            a = self.bfhp._inverse_transform(self.lower)
            b = self.bfhp._inverse_transform(self.upper)
            u, v = spbeta(self.alpha, self.beta, loc=a, scale=b-a).interval(alpha=0.999999)
            lb = max(self.bfhp._transform(u), self.lower)
            ub = min(self.bfhp._transform(v), self.upper + 1)
        else:
            lb = self.lower
            ub = self.upper + 1
            
        chunks = arange_chunked(lb, ub, chunk_size=ARANGE_CHUNKSIZE)
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
