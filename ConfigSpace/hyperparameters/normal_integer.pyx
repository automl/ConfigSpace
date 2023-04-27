from itertools import count
import io
from more_itertools import roundrobin
from typing import List, Any, Dict, Union, Optional
import warnings

from scipy.stats import truncnorm, norm
import numpy as np
cimport numpy as np
np.import_array()

from ConfigSpace.functional import center_range, arange_chunked
from ConfigSpace.hyperparameters.uniform_integer cimport UniformIntegerHyperparameter
from ConfigSpace.hyperparameters.normal_float cimport NormalFloatHyperparameter

# OPTIM: Some operations generate an arange which could blowup memory if
# done over the entire space of integers (int32/64).
# To combat this, `arange_chunked` is used in scenarios where reducion
# operations over all the elments could be done in partial steps independantly.
# For example, a sum over the pdf values could be done in chunks.
# This may add some small overhead for smaller ranges but is unlikely to
# be noticable.
ARANGE_CHUNKSIZE = 10_000_000


cdef class NormalIntegerHyperparameter(IntegerHyperparameter):

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
        return hash((self.name, self.mu, self.sigma, self.log, self.q, self.lower, self.upper))

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

    def to_uniform(self, z: int = 3) -> "UniformIntegerHyperparameter":
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

        if not bounded:
            float_indices = norm.rvs(
                loc=mu,
                scale=sigma,
                size=number,
                random_state=rs,
            )
        else:
            dist = truncnorm(
                a = (self.lower - mu) / sigma,
                b = (self.upper - mu) / sigma,
                loc=center,
                scale=sigma,
            )

            float_indices = dist.rvs(
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
            warnings.warn("Cannot normalize the pdf exactly for a NormalIntegerHyperparameter"
                          f" {self.name} without bounds. Skipping normalization for that hyperparameter.")
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
