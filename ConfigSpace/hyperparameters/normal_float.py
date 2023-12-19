from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm, truncnorm

from ConfigSpace.hyperparameters.float_hyperparameter import FloatHyperparameter
from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.normal_integer import NormalIntegerHyperparameter


class NormalFloatHyperparameter(FloatHyperparameter):
    def __init__(
        self,
        name: str,
        mu: int | float,
        sigma: int | float,
        default_value: None | float = None,
        q: int | float | None = None,
        log: bool = False,
        lower: float | int | None = None,
        upper: float | int | None = None,
        meta: dict | None = None,
    ) -> None:
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
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.q = float(q) if q is not None else None
        self.log = bool(log)
        self.lower: float | None = None
        self.upper: float | None = None

        if (lower is not None) ^ (upper is not None):
            raise ValueError(
                "Only one bound was provided when both lower and upper bounds must be provided.",
            )

        if lower is not None and upper is not None:
            self.lower = float(lower)
            self.upper = float(upper)

            if self.lower >= self.upper:
                raise ValueError(
                    f"Upper bound {self.upper:f} must be larger than lower bound "
                    f"{self.lower:f} for hyperparameter {name}",
                )
            if log and self.lower <= 0:
                raise ValueError(
                    f"Negative lower bound ({self.lower:f}) for log-scale "
                    f"hyperparameter {name} is forbidden.",
                )

            self.default_value = self.check_default(default_value)

            if self.log:
                if self.q is not None:
                    lower = self.lower - (np.float64(self.q) / 2.0 - 0.0001)
                    upper = self.upper + (np.float64(self.q) / 2.0 - 0.0001)
                else:
                    lower = self.lower
                    upper = self.upper
                self._lower = np.log(lower)
                self._upper = np.log(upper)
            else:
                if self.q is not None:
                    self._lower = self.lower - (self.q / 2.0 - 0.0001)
                    self._upper = self.upper + (self.q / 2.0 - 0.0001)
                else:
                    self._lower = self.lower
                    self._upper = self.upper

            if self.q is not None:
                # There can be weird rounding errors, so we compare the result against self.q, see
                # In [13]: 2.4 % 0.2
                # Out[13]: 0.1999999999999998
                if np.round((self.upper - self.lower) % self.q, 10) not in (0, self.q):
                    raise ValueError(
                        f"Upper bound ({self.upper:f}) - lower bound ({self.lower:f}) must be a multiple of q ({self.q:f})",
                    )

        default_value = self.check_default(default_value)
        super().__init__(name, default_value, meta)
        self.normalized_default_value = self._inverse_transform(self.default_value)

    def __repr__(self) -> str:
        repr_str = io.StringIO()

        if self.lower is None or self.upper is None:
            repr_str.write(
                f"{self.name}, Type: NormalFloat, Mu: {self.mu!r} Sigma: {self.sigma!r}, Default: {self.default_value!r}",
            )
        else:
            repr_str.write(
                "{}, Type: NormalFloat, Mu: {} Sigma: {}, Range: [{}, {}], Default: {}".format(
                    self.name,
                    repr(self.mu),
                    repr(self.sigma),
                    repr(self.lower),
                    repr(self.upper),
                    repr(self.default_value),
                ),
            )

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
            self.name == other.name
            and self.default_value == other.default_value
            and self.mu == other.mu
            and self.sigma == other.sigma
            and self.log == other.log
            and self.q == other.q
            and self.lower == other.lower
            and self.upper == other.upper
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
            meta=self.meta,
        )

    def __hash__(self):
        return hash((self.name, self.mu, self.sigma, self.log, self.q, self.lower, self.upper))

    def to_uniform(self, z: int = 3) -> UniformFloatHyperparameter:
        if self.lower is None or self.upper is None:
            lb = self.mu - (z * self.sigma)
            ub = self.mu + (z * self.sigma)
        else:
            lb = self.lower
            ub = self.upper

        return UniformFloatHyperparameter(
            self.name,
            lb,
            ub,
            default_value=self.default_value,
            q=self.q,
            log=self.log,
            meta=self.meta,
        )

    def check_default(self, default_value: int | float | None) -> int | float:
        if default_value is None:
            if self.log:
                return self._transform_scalar(self.mu)
            else:
                return self.mu

        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def to_integer(self) -> NormalIntegerHyperparameter:
        q_int = None if self.q is None else int(np.rint(self.q))
        if self.lower is None:
            lower = None
            upper = None
        else:
            lower = np.ceil(self.lower)
            upper = np.floor(self.upper)

        from ConfigSpace.hyperparameters.normal_integer import NormalIntegerHyperparameter

        return NormalIntegerHyperparameter(
            self.name,
            int(np.rint(self.mu)),
            self.sigma,
            lower=lower,
            upper=upper,
            default_value=int(np.rint(self.default_value)),
            q=q_int,
            log=self.log,
        )

    def is_legal(self, value: float) -> bool:
        return (
            (isinstance(value, (float, int, np.number)))
            and (self.lower is None or value >= self.lower)
            and (self.upper is None or value <= self.upper)
        )

    def is_legal_vector(self, value) -> int:
        return isinstance(value, (float, int))

    def _sample(
        self,
        rs: np.random.RandomState,
        size: int | None = None,
    ) -> np.ndarray | float:
        if self.lower is None:
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

    def _transform_vector(self, vector: np.ndarray) -> np.ndarray:
        if np.isnan(vector).any():
            raise ValueError("Vector %s contains NaN's" % vector)
        if self.log:
            vector = np.exp(vector)
        if self.q is not None:
            vector = np.rint(vector / self.q) * self.q
        return vector

    def _transform_scalar(self, scalar: float) -> float:
        if scalar != scalar:
            raise ValueError("Number %s is NaN" % scalar)
        if self.log:
            scalar = math.exp(scalar)
        if self.q is not None:
            scalar = np.round(scalar / self.q) * self.q
        return scalar

    def _inverse_transform(
        self,
        vector: float | np.ndarray | None,
    ) -> float | np.ndarray:
        # TODO: Should probably use generics here
        if vector is None:
            return np.NaN

        if self.log:
            vector = np.log(vector)

        return vector

    def get_neighbors(
        self,
        value: float,
        rs: np.random.RandomState,
        number: int = 4,
        transform: bool = False,
    ) -> list[float]:
        neighbors = []
        for _i in range(number):
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
        -------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        mu = self.mu
        sigma = self.sigma
        if self.lower is None:
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
