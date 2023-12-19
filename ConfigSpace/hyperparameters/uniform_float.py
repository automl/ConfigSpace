from __future__ import annotations

import io
from typing import TYPE_CHECKING, overload

import numpy as np
import numpy.typing as npt

from ConfigSpace.deprecate import deprecate
from ConfigSpace.hyperparameters.float_hyperparameter import FloatHyperparameter

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter


class UniformFloatHyperparameter(FloatHyperparameter):
    def __init__(
        self,
        name: str,
        lower: int | float,
        upper: int | float,
        default_value: int | float | None = None,
        q: int | float | None = None,
        log: bool = False,
        meta: dict | None = None,
    ) -> None:
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
        lower = float(lower)
        upper = float(upper)
        q = float(q) if q is not None else None
        log = bool(log)
        if lower >= upper:
            raise ValueError(
                f"Upper bound {upper:f} must be larger than lower bound "
                f"{lower:f} for hyperparameter {name}",
            )

        if log and lower <= 0:
            raise ValueError(
                f"Negative lower bound ({lower:f}) for log-scale "
                f"hyperparameter {name} is forbidden.",
            )

        if q is not None and (np.round((upper - lower) % q, 10) not in (0, q)):
            # There can be weird rounding errors, so we compare the result against self.q
            # for example, 2.4 % 0.2 = 0.1999999999999998
            diff = upper - lower
            raise ValueError(
                f"Upper bound minus lower bound ({upper:f} - {lower:f} = {diff}) must be"
                f" a multiple of q ({q})",
            )

        self.lower = lower
        self.upper = upper
        self.q = q
        self.log = log

        q_lower, q_upper = (lower, upper)
        if q is not None:
            q_lower = lower - (q / 2.0 - 0.0001)
            q_upper = upper + (q / 2.0 - 0.0001)

        if log:
            self._lower = np.log(q_lower)
            self._upper = np.log(q_upper)
        else:
            self._lower = q_lower
            self._upper = q_upper

        if default_value is None:
            if log:
                default_value = float(np.exp((np.log(lower) + np.log(upper)) / 2.0))
            else:
                default_value = (lower + upper) / 2.0

        default_value = float(np.round(default_value, 10))
        if not self.is_legal(default_value):
            raise ValueError(f"Illegal default value {default_value}")

        self.default_value = default_value
        self.normalized_default_value = self._inverse_transform(self.default_value)
        super().__init__(name=name, default_value=default_value, meta=meta)

    def is_legal(self, value: float) -> bool:
        if not isinstance(value, (float, int)):
            return False
        return self.upper >= value >= self.lower

    def is_legal_vector(self, value: float) -> bool:
        # NOTE: This really needs a better name as it doesn't operate on vectors,
        # it means that it operates on the normalzied space, i.e. what is gotten
        # by inverse_transform
        return 1.0 >= value >= 0.0

    def to_integer(self) -> UniformIntegerHyperparameter:
        from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter

        return UniformIntegerHyperparameter(
            name=self.name,
            lower=int(np.ceil(self.lower)),
            upper=int(np.floor(self.upper)),
            default_value=int(np.rint(self.default_value)),
            q=int(np.rint(self.q)) if self.q is not None else None,
            log=self.log,
            meta=None,
        )

    @overload
    def _sample(self, rs: np.random.RandomState, size: None = None) -> float:
        ...

    @overload
    def _sample(self, rs: np.random.RandomState, size: int) -> npt.NDArray[np.float64]:
        ...

    def _sample(
        self,
        rs: np.random.RandomState,
        size: int | None = None,
    ) -> float | npt.NDArray[np.float64]:
        return rs.uniform(size=size)

    def _transform_scalar(self, scalar: float) -> np.float64:
        deprecate(self._transform_scalar, "Please use _transform instead")
        return self._transform(scalar)

    def _transform_vector(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.float64]:
        deprecate(self._transform_scalar, "Please use _transform instead")
        return self._transform(vector)

    @overload
    def _transform(self, vector: float) -> np.float64:
        ...

    @overload
    def _transform(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.float64]:
        ...

    def _transform(
        self,
        vector: npt.NDArray[np.number] | float,
    ) -> npt.NDArray[np.float64] | np.float64:
        vector = vector * (self._upper - self._lower) + self._lower
        if self.log:
            vector = np.exp(vector, dtype=np.float64)

        if self.q is not None:
            quantized = (vector - self.lower) / self.q
            vector = np.rint(quantized) * self.q + self.lower

        return np.clip(vector, self.lower, self.upper, dtype=np.float64)

    @overload
    def _inverse_transform(self, vector: float) -> np.float64:
        ...

    @overload
    def _inverse_transform(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.float64]:
        ...

    def _inverse_transform(
        self,
        vector: npt.NDArray[np.number] | float,
    ) -> npt.NDArray[np.float64] | np.float64:
        """Converts a value from the original space to the transformed space (0, 1)."""
        if self.log:
            vector = np.log(vector)
        vector = (vector - self._lower) / (self._upper - self._lower)
        return np.clip(vector, 0.0, 1.0, dtype=np.float64)

    def get_neighbors(
        self,
        value: float,  # Should be normalized closely into 0, 1!
        rs: np.random.RandomState,
        number: int = 4,
        transform: bool = False,
        std: float = 0.2,
    ) -> npt.NDArray[np.float64]:
        BUFFER_MULTIPLIER = 2
        SAMPLE_SIZE = number * BUFFER_MULTIPLIER
        # Make sure we can accomidate the number of (neighbors - 1) + a new sample set
        BUFFER_SIZE = number + number * BUFFER_MULTIPLIER

        neighbors = np.empty(BUFFER_SIZE, dtype=np.float64)
        offset = 0

        # Generate batches of number * 2 candidates, filling the above
        # buffer until we have enough valid candidates.
        # We should not overflow as the buffer
        while offset <= number:
            candidates = rs.normal(value, std, size=SAMPLE_SIZE)
            valid_candidates = candidates[(candidates >= 0) & (candidates <= 1)]

            n_candidates = len(valid_candidates)
            neighbors[offset:n_candidates] = valid_candidates
            offset += n_candidates

        neighbors = neighbors[:number]
        if transform:
            return self._transform(neighbors)

        return neighbors

    def _pdf(self, vector: npt.NDArray[np.number]) -> npt.NDArray[np.float64]:
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
        # everything that comes into _pdf for a uniform variable should
        # already be in [0, 1]-range, and if not, it's outside the upper
        # or lower bound.
        ub = 1
        lb = 0
        inside_range = ((lb <= vector) & (vector <= ub)).astype(dtype=np.uint64)
        return np.true_divide(inside_range, self.upper - self.lower, dtype=np.float64)

    def get_max_density(self) -> float:
        return 1 / (self.upper - self.lower)

    def get_size(self) -> float:
        if self.q is None:
            return np.inf

        return np.rint((self.upper - self.lower) / self.q) + 1

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write(
            f"{self.name}, Type: UniformFloat, "
            f"Range: [{self.lower!r}, {self.upper!r}], "
            f"Default: {self.default_value!r}",
        )
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()
