# NOTE: Unfortunatly scipy.stats does not allow discrete distributions whose support
# is not integers,
# e.g. can't have a discrete distribution over [0.0, 1.0] with 10 bins.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt
from scipy.stats import randint
from scipy.stats._discrete_distns import randint_gen

from ConfigSpace.functional import arange_chunked, split_arange
from ConfigSpace.hyperparameters._hp_components import (
    DEFAULT_VECTORIZED_NUMERIC_STD,
    ROUND_PLACES,
    VDType,
)

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import (
        rv_continuous_frozen,
        rv_discrete_frozen,
    )

# OPTIM: Some operations generate an arange which could blowup memory if
# done over the entire space of integers (int32/64).
# To combat this, `arange_chunked` is used in scenarios where reducion
# operations over all the elments could be done in partial steps independantly.
# For example, a sum over the pdf values could be done in chunks.
# This may add some small overhead for smaller ranges but is unlikely to
# be noticable.
ARANGE_CHUNKSIZE = 10_000_000

CONFIDENCE_FOR_NORMALIZATION_OF_DISCRETE = 0.999999
NEIGHBOR_GENERATOR_N_RETRIES = 5
NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER = 2
RandomState = np.random.RandomState


class VectorDistribution(Protocol[VDType]):
    lower: VDType
    upper: VDType

    def max_density(self) -> float: ...

    def sample(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> npt.NDArray[VDType]: ...

    def in_support(self, vector: VDType) -> bool: ...

    def pdf(self, vector: npt.NDArray[VDType]) -> npt.NDArray[np.float64]: ...


@dataclass
class DiscretizedContinuousScipyDistribution(VectorDistribution[np.float64]):
    steps: int | np.int64
    dist: rv_continuous_frozen
    max_density_value: float | None = None
    normalization_constant_value: float | None = None
    int_dist: randint_gen = field(init=False)

    def __post_init__(self):
        int_gen = randint(0, self.steps)
        assert isinstance(int_gen, randint_gen)
        self.int_dist = int_gen

    @property
    def lower(self) -> np.float64:
        return self.dist.a

    @property
    def upper(self) -> np.float64:
        return self.dist.b

    def max_density(self) -> float:
        if self.max_density_value is not None:
            return self.max_density_value

        # Otherwise, we generate all possible integers and find the maximum
        lower, upper = self._bounds_with_confidence()
        lower_int, upper_int = self._as_integers(np.array([lower, upper]))
        chunks = arange_chunked(lower_int, upper_int, chunk_size=ARANGE_CHUNKSIZE)
        max_density = max(
            self.pdf(self._rescale_integers(chunk)).max() for chunk in chunks
        )
        self.max_density_value = max_density
        return max_density

    def _rescale_integers(
        self,
        integers: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        # the steps - 1 is because the range above is exclusive w.r.t. the upper bound
        # e.g.
        # Suppose setps = 5
        # then possible integers = [0, 1, 2, 3, 4]
        # then possible values = [0, 0.25, 0.5, 0.75, 1]
        # ... which span the 0, 1 range as intended
        unit_normed = integers / (self.steps - 1)
        return np.clip(
            self.lower + unit_normed * (self.upper - self.lower),
            self.lower,
            self.upper,
        )

    def _as_integers(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        unit_normed = (vector - self.lower) / (self.upper - self.lower)
        return np.rint(unit_normed * (self.steps - 1)).astype(int)

    def _bounds_with_confidence(
        self,
        confidence: float = CONFIDENCE_FOR_NORMALIZATION_OF_DISCRETE,
    ) -> tuple[np.float64, np.float64]:
        lower, upper = (
            self.dist.ppf((1 - confidence) / 2),
            self.dist.ppf((1 + confidence) / 2),
        )
        return max(lower, self.lower), min(upper, self.upper)

    def _normalization_constant(self) -> float:
        if self.normalization_constant_value is not None:
            return self.normalization_constant_value

        lower_int, upper_int = self._as_integers(np.array([self.lower, self.upper]))
        # If there's a lot of possible values, we want to find
        # the support for where the distribution is above some
        # minimal pdf value, and only compute the normalization constant
        # w.r.t. to those values. It is an approximation
        if upper_int - lower_int > ARANGE_CHUNKSIZE:
            l_bound, u_bound = self._bounds_with_confidence()
            lower_int, upper_int = self._as_integers(np.array([l_bound, u_bound]))

        chunks = arange_chunked(lower_int, upper_int, chunk_size=ARANGE_CHUNKSIZE)
        normalization_constant = sum(
            self.dist.pdf(self._rescale_integers(chunk)).sum() for chunk in chunks
        )
        self.normalization_constant_value = normalization_constant
        return self.normalization_constant_value

    def sample(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> npt.NDArray[np.float64]:
        integers = self.int_dist.rvs(size=n, random_state=seed)
        assert isinstance(integers, np.ndarray)
        return self._rescale_integers(integers)

    def in_support(self, vector: np.float64) -> bool:
        return self.pdf(np.array([vector]))[0] != 0

    def pdf(self, vector: npt.NDArray[VDType]) -> npt.NDArray[np.float64]:
        unit_normed = (vector - self.lower) / (self.upper - self.lower)
        int_scaled = unit_normed * (self.steps - 1)
        close_to_int = np.round(int_scaled, ROUND_PLACES)
        rounded_as_int = np.rint(int_scaled)

        valid_entries = np.where(
            (close_to_int == rounded_as_int)
            & (vector >= self.lower)
            & (vector <= self.upper),
            rounded_as_int,
            np.nan,
        )

        return self.dist.pdf(valid_entries) / self._normalization_constant()

    def neighborhood(
        self,
        vector: np.float64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
        n_retries: int = NEIGHBOR_GENERATOR_N_RETRIES,
        sample_multiplier: int = NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
    ) -> npt.NDArray[np.float64]:
        if std is None:
            std = DEFAULT_VECTORIZED_NUMERIC_STD

        assert n < 1000000, "Can only generate less than 1 million neighbors."
        seed = np.random.RandomState() if seed is None else seed

        center_int = self._as_integers(np.array([vector]))[0]

        # In the easiest case, the amount of neighbors we need is more than the amount
        # possible, in this case, we can skip our sampling and just generate all
        # neighbors, excluding the current value
        if n >= self.steps - 1:
            values = split_arange((0, center_int), (center_int, self.steps))
            return self._rescale_integers(values)

        # Otherwise, we use a repeated sampling strategy where we slowly increase the
        # std of a normal, centered on `center`, slowly expanding `std` such that
        # rejection won't fail.

        # We set up a buffer that can hold the number of neighbors we need, plus some
        # extra excess from sampling, preventing us from having to reallocate memory.
        SAMPLE_SIZE = n * sample_multiplier
        BUFFER_SIZE = n * (sample_multiplier + 1)
        neighbors = np.empty(BUFFER_SIZE, dtype=np.int64)
        offset = 0  # Indexes into current progress of filling buffer

        # We extend the range of stds to try to find neighbors
        stds = np.linspace(std, 1.0, n_retries + 1, endpoint=True)

        # The span over which we want std to cover
        range_size = self.upper - self.lower

        for _std in stds:
            # Generate candidates in vectorized space
            candidates = seed.normal(vector, _std * range_size, size=SAMPLE_SIZE)
            valid_candidates = candidates[
                (candidates >= self.lower) & (candidates <= self.upper)
            ]

            # Transform to integers and get uniques
            candidates_int = self._as_integers(valid_candidates)

            if offset == 0:
                uniq = np.unique(candidates_int)

                n_candidates = len(uniq)
                neighbors[:n_candidates] = uniq
                offset += n_candidates
            else:
                uniq = np.unique(candidates_int)
                new_uniq = np.setdiff1d(uniq, neighbors[:offset], assume_unique=True)

                n_new_unique = len(new_uniq)
                neighbors[offset : offset + n_new_unique] = new_uniq
                offset += n_new_unique

            # We have enough neighbors, we can stop and return the vectorized values
            if offset >= n:
                return self._rescale_integers(neighbors[:n])

        raise ValueError(
            f"Failed to find enough neighbors with {n_retries} retries."
            f"Given {n} neighbors, we only found {offset}.",
            f"The normal's for sampling neighbors were Normal({vector}, {list(stds)})"
            f" which were meant to find neighbors of {vector}. in the range",
            f" ({self.lower}, {self.upper}).",
        )


@dataclass
class ScipyDiscreteDistribution(VectorDistribution[VDType]):
    rv: rv_discrete_frozen
    max_density_value: float | Callable[[], float]
    dtype: type[VDType]

    @property
    def lower(self) -> VDType:
        return self.rv.a

    @property
    def upper(self) -> VDType:
        return self.rv.b

    def sample(
        self,
        size: int | None = None,
        *,
        seed: RandomState | None = None,
    ) -> npt.NDArray[VDType]:
        return self.rv.rvs(size=size, random_state=seed).astype(self.dtype)

    def in_support(self, vector: VDType) -> bool:
        return self.rv.a <= vector <= self.rv.b

    def max_density(self) -> float:
        match self.max_density_value:
            case float() | int():
                return self.max_density_value
            case _:
                max_density = self.max_density_value()
                self.max_density_value = max_density
                return max_density

    def pdf(self, vector: npt.NDArray[VDType]) -> npt.NDArray[np.float64]:
        return self.rv.pmf(vector)


@dataclass
class ScipyContinuousDistribution(VectorDistribution[VDType]):
    rv: rv_continuous_frozen
    max_density_value: float | Callable[[], float]
    dtype: type[VDType]

    @property
    def lower(self) -> VDType:
        return self.rv.a

    @property
    def upper(self) -> VDType:
        return self.rv.b

    def sample(
        self,
        size: int | None = None,
        *,
        seed: RandomState | None = None,
    ) -> npt.NDArray[VDType]:
        return self.rv.rvs(size=size, random_state=seed).astype(self.dtype)

    def in_support(self, vector: VDType) -> bool:
        return self.rv.a <= vector <= self.rv.b

    def max_density(self) -> float:
        match self.max_density_value:
            case float() | int():
                return self.max_density_value
            case _:
                max_density = self.max_density_value()
                self.max_density_value = max_density
                return max_density

    def pdf(self, vector: npt.NDArray[VDType]) -> npt.NDArray[np.float64]:
        return self.rv.pdf(vector)

    def neighborhood(
        self,
        vector: np.float64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
        n_retries: int = NEIGHBOR_GENERATOR_N_RETRIES,
        sample_multiplier: int = NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
    ) -> npt.NDArray[np.float64]:
        if std is None:
            std = DEFAULT_VECTORIZED_NUMERIC_STD

        seed = np.random.RandomState() if seed is None else seed

        SAMPLE_SIZE = n * sample_multiplier
        BUFFER_SIZE = n + n * sample_multiplier
        neighbors = np.empty(BUFFER_SIZE, dtype=np.float64)
        offset = 0

        # We extend the range of stds to try to find neighbors
        stds = np.linspace(std, 1.0, n_retries + 1, endpoint=True)

        # Generate batches of n * BUFFER_MULTIPLIER candidates, filling the above
        # buffer until we have enough valid candidates.
        # We should not overflow as the buffer
        range_size = self.upper - self.lower
        for _std in stds:
            candidates = seed.normal(vector, _std * range_size, size=(SAMPLE_SIZE,))
            valid_candidates = candidates[
                (candidates >= self.lower) & (candidates <= self.upper)
            ]

            n_candidates = len(valid_candidates)
            neighbors[offset:n_candidates] = valid_candidates
            offset += n_candidates

            # We have enough neighbors, we can stop and return the vectorized values
            if offset >= n:
                return neighbors[:n]

        raise ValueError(
            f"Failed to find enough neighbors with {n_retries} retries."
            f"Given {n} neighbors, we only found {offset}.",
            f"The normal's for sampling neighbors were Normal({vector}, {list(stds)})"
            f" which were meant to find neighbors of {vector}. in the range",
            f" ({self.lower}, {self.upper}).",
        )


@dataclass
class ConstantVectorDistribution(VectorDistribution[np.integer]):
    value: np.integer

    @property
    def lower(self) -> np.integer:
        return self.value

    @property
    def upper(self) -> np.integer:
        return self.value

    def max_density(self) -> float:
        return 1.0

    def sample(
        self,
        n: int | None = None,
        *,
        seed: RandomState | None = None,
    ) -> np.integer | npt.NDArray[np.integer]:
        if n is None:
            return self.value

        return np.full((n,), self.value, dtype=np.integer)

    def in_support(self, vector: np.integer) -> bool:
        return vector == self.value

    def pdf(self, vector: npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
        return (vector == self.value).astype(float)
