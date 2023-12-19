from __future__ import annotations

import io

import numpy as np

from ConfigSpace.functional import center_range
from ConfigSpace.hyperparameters.integer_hyperparameter import IntegerHyperparameter
from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter


class UniformIntegerHyperparameter(IntegerHyperparameter):
    def __init__(
        self,
        name: str,
        lower: int | float,
        upper: int | float,
        default_value: int | None = None,
        q: int | None = None,
        log: bool = False,
        meta: dict | None = None,
    ) -> None:
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
        self.log = bool(log)
        self.lower = self.check_int(lower, "lower")
        self.upper = self.check_int(upper, "upper")

        self.q = None
        if q is not None:
            if q < 1:
                raise ValueError(
                    "Setting quantization < 1 for Integer "
                    f"Hyperparameter {name} has no effect."
                )

            self.q = self.check_int(q, "q")
            if (self.upper - self.lower) % self.q != 0:
                raise ValueError(
                    "Upper bound (%d) - lower bound (%d) must be a multiple of q (%d)"
                    % (self.upper, self.lower, self.q),
                )

        if self.lower >= self.upper:
            raise ValueError(
                "Upper bound %d must be larger than lower bound "
                "%d for hyperparameter %s" % (self.lower, self.upper, name),
            )
        if log and self.lower <= 0:
            raise ValueError(
                "Negative lower bound (%d) for log-scale "
                "hyperparameter %s is forbidden." % (self.lower, name),
            )

        # Requires `log` to be set first
        if default_value is not None:
            default_value = self.check_int(default_value, name)
        else:
            default_value = self.check_default(default_value)

        # NOTE: Placed after the default value check to ensure it's set and not None
        super().__init__(name, default_value, meta)

        self.ufhp = UniformFloatHyperparameter(
            self.name,
            self.lower - 0.49999,
            self.upper + 0.49999,
            log=self.log,
            default_value=self.default_value,
        )

        self.normalized_default_value = self._inverse_transform(self.default_value)

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write(
            f"{self.name}, Type: UniformInteger, Range: [{self.lower!r}, {self.upper!r}], Default: {self.default_value!r}",
        )
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % repr(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

    def _sample(
        self,
        rs: np.random.RandomState,
        size: int | None = None,
    ) -> np.ndarray | float:
        value = self.ufhp._sample(rs, size=size)
        # Map all floats which belong to the same integer value to the same
        # float value by first transforming it to an integer and then
        # transforming it back to a float between zero and one
        value = self._transform(value)
        return self._inverse_transform(value)

    def _transform_vector(self, vector: np.ndarray) -> np.ndarray:
        vector = self.ufhp._transform_vector(vector)
        if self.q is not None:
            vector = np.rint((vector - self.lower) / self.q) * self.q + self.lower
            vector = np.minimum(vector, self.upper)
            vector = np.maximum(vector, self.lower)

        return np.rint(vector)

    def _transform_scalar(self, scalar: float) -> float:
        scalar = self.ufhp._transform_scalar(scalar)
        if self.q is not None:
            scalar = np.round((scalar - self.lower) / self.q) * self.q + self.lower
            scalar = min(scalar, self.upper)
            scalar = max(scalar, self.lower)
        return int(np.round(scalar))

    def _inverse_transform(
        self,
        vector: np.ndarray | float | int,
    ) -> np.ndarray | float | int:
        return self.ufhp._inverse_transform(vector)

    def is_legal(self, value: int) -> bool:
        if not (isinstance(value, (int, np.int32, np.int64))):
            return False
        return self.upper >= value >= self.lower

    def is_legal_vector(self, value) -> int:
        return 1.0 >= value >= 0.0

    def check_default(self, default_value: int | float | None) -> int:
        # Doesn't seem to quantize with q?
        if default_value is None:
            if self.log:
                default_value = np.exp((np.log(self.lower) + np.log(self.upper)) / 2.0)
            else:
                default_value = (self.lower + self.upper) / 2.0
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
        return upper - lower >= 1

    def get_num_neighbors(self, value=None) -> int:
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
    ) -> list[int]:
        """Get the neighbors of a value.

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
        assert number < 1000000, "Can only generate less than 1 million neighbors."
        # Convert python values to cython ones
        center = self._transform(value)
        lower = self.lower
        upper = self.upper
        n_requested = number
        n_neighbors = upper - lower - 1
        stepsize = self.q if self.q is not None else 1

        neighbors = []

        v: int  # A value that's possible to return
        if n_neighbors < n_requested:
            for v in range(lower, center):
                neighbors.append(v)

            for v in range(center + 1, upper + 1):
                neighbors.append(v)

            if transform:
                return neighbors
            else:
                return self._inverse_transform(np.asarray(neighbors)).tolist()

        # A truncated normal between 0 and 1, centered on the value with a scale of std.
        # This will be sampled from and converted to the corresponding int value
        # However, this is too slow - we use the "poor man's truncnorm below"
        # cdef np.ndarray float_indices = truncnorm.rvs(
        # We sample five times as many values as needed and weed them out below
        # (perform rejection sampling and make sure we don't sample any neighbor twice)
        # This increases our chances of not having to fill the neighbors list by calling
        # `center_range`
        # Five is an arbitrary number and can probably be tuned to reduce overhead
        float_indices: np.ndarray = rs.normal(value, std, size=number * 5)
        mask: np.ndarray = (float_indices >= 0) & (float_indices <= 1)
        float_indices = float_indices[mask]

        possible_neighbors_as_array: np.ndarray = self._transform_vector(float_indices).astype(
            np.longlong,
        )
        possible_neighbors: np.ndarray = possible_neighbors_as_array

        n_neighbors_generated: int = 0
        n_candidates: int = len(float_indices)
        candidate_index: int = 0
        seen: set[int] = {center}
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
        to the probability density function (see e.g. NormalIntegerHyperparameter).

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
        return self.ufhp._pdf(vector)

    def get_max_density(self) -> float:
        lb = self.lower
        ub = self.upper
        return 1 / (ub - lb + 1)

    def get_size(self) -> float:
        q = 1 if self.q is None else self.q
        return np.rint((self.upper - self.lower) / q) + 1
