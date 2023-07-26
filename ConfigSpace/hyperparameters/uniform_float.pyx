import io
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
cimport numpy as np
np.import_array()

from ConfigSpace.hyperparameters.uniform_integer cimport UniformIntegerHyperparameter


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
                    "Upper bound (%f) - lower bound (%f) must be a multiple of q (%f)"
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
                default_value = float(np.exp((np.log(self.lower) + np.log(self.upper)) / 2.))
            else:
                default_value = (self.lower + self.upper) / 2.
        default_value = float(np.round(float(default_value), 10))

        if self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def to_integer(self) -> "UniformIntegerHyperparameter":
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
            raise ValueError("Number %s is NaN" % scalar)
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
