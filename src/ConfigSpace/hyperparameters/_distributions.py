from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterator
from typing_extensions import Protocol

import numpy as np

from ConfigSpace.functional import (
    is_close_to_integer,
    linspace_chunked,
    quantize,
    quantize_log,
)
from ConfigSpace.hyperparameters._hp_components import ATOL, _Transformer
from ConfigSpace.types import DType, f64, i64

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import (
        rv_continuous_frozen,
        rv_discrete_frozen,
    )

    from ConfigSpace.types import Array

# OPTIM: Some operations generate an arange which could blowup memory if
# done over the entire space of integers (int32/64).
# To combat this, `arange_chunked` is used in scenarios where reducion
# operations over all the elments could be done in partial steps independantly.
# For example, a sum over the pdf values could be done in chunks.
# This may add some small overhead for smaller ranges but is unlikely to
# be noticable.
ARANGE_CHUNKSIZE = 10_000_000

CONFIDENCE_FOR_NORMALIZATION_OF_DISCRETE = 0.999999
DEFAULT_VECTORIZED_NUMERIC_STD = 0.2
NEIGHBOR_GENERATOR_N_RETRIES = 8
NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER = 4
RandomState = np.random.RandomState

# OPTIM: No need to keep regnerating this as most of the time we use defaults
# for generating a linspace for neighborhoods
_CACHED_LINSPACE = np.linspace(
    DEFAULT_VECTORIZED_NUMERIC_STD,
    1.0,
    NEIGHBOR_GENERATOR_N_RETRIES + 1,
    endpoint=True,
)


def _compare_rv(a: Any, b: Any) -> bool:
    # scipy stats object don't compare nicely...
    adict = a.__dict__
    bdict = b.__dict__
    for key in adict:
        if key == "dist":
            if adict[key]._ctor_param != bdict[key]._ctor_param:
                return False

        elif adict[key] != bdict[key]:
            return False

    return True


def quantized_neighborhood(
    vector: f64,
    n: int,
    *,
    std: float | None = None,
    seed: RandomState | None = None,
    n_retries: int = NEIGHBOR_GENERATOR_N_RETRIES,
    sample_multiplier: int = NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
    lower: f64,
    upper: f64,
    bins: int,
) -> Array[f64]:
    if std is None:
        std = DEFAULT_VECTORIZED_NUMERIC_STD

    assert n < 1000000, "Can only generate less than 1 million neighbors."
    seed = np.random.RandomState() if seed is None else seed

    qvector = quantize(vector, bounds=(lower, upper), bins=bins)

    # In the easiest case, the amount of neighbors we need is more than the amount
    # possible, in this case, we can skip our sampling and just generate all
    # neighbors, excluding the current value
    n_available = bins - 1
    if n >= n_available:
        if qvector == 0:
            return np.arange(1, bins) / (bins - 1)
        if qvector == bins - 1:
            return np.arange(0, bins - 1) / (bins - 1)

        qint = i64(np.rint(vector * (bins - 1)))

        _range = np.arange(0, bins, dtype=np.float64)
        bot = _range[:qint]
        top = _range[qint + 1 :]
        return np.concatenate((bot, top)) / (bins - 1)

    # Otherwise, we use a repeated sampling strategy where we slowly increase the
    # std of a normal, centered on `center`, slowly expanding `std` such that
    # rejection won't fail.

    # We set up a buffer that can hold the number of neighbors we need, plus some
    # extra excess from sampling, preventing us from having to reallocate memory.
    # We also include the initial value in the buffer, as we will remove it later.
    # OPTIM: If you really need to optimize this more, one could consider using `bins`
    # as the more bins there are, the less need there is over sample for uniques
    SAMPLE_SIZE = n * sample_multiplier
    BUFFER_SIZE = n * (sample_multiplier + 1)
    neighbors = np.empty(BUFFER_SIZE + 1, dtype=f64)
    neighbors[0] = qvector
    offset = 1  # Indexes into current progress of filling buffer

    # We extend the range of stds to try to find neighbors
    if (
        std == DEFAULT_VECTORIZED_NUMERIC_STD
        and n_retries == NEIGHBOR_GENERATOR_N_RETRIES
    ):
        stds = _CACHED_LINSPACE
    else:
        stds = np.linspace(std, 1.0, n_retries + 1, endpoint=True)

    range_size = upper - lower
    for _std in stds:
        # Generate candidates in vectorized space
        candidates = seed.normal(qvector, _std * range_size, size=SAMPLE_SIZE)
        valid_candidates = candidates[(candidates >= lower) & (candidates <= upper)]

        # Transform to quantized space
        candidates_quantized = quantize(
            valid_candidates,
            bounds=(lower, upper),
            bins=bins,
        )

        uniq = np.unique(candidates_quantized)
        new_uniq = np.setdiff1d(uniq, neighbors[:offset], assume_unique=True)

        n_new_unique = len(new_uniq)
        neighbors[offset : offset + n_new_unique] = new_uniq
        offset += n_new_unique

        # We have enough neighbors, we can stop
        if offset - 1 >= n:
            # Ensure we don't include the initial value point
            return neighbors[1 : n + 1]

    raise ValueError(
        f"Failed to find enough neighbors with {n_retries} retries."
        f" Given {n=} neighbors to generate, we only found {offset - 1}."
        f" The normal's for sampling neighbors were Normal({vector}, {list(stds)})"
        f" which were meant to find neighbors of {vector}. in the range"
        f" ({lower}, {upper}).",
    )


def continuous_neighborhood(
    vector: f64,
    n: int,
    *,
    lower: f64,
    upper: f64,
    std: float | None = None,
    seed: RandomState | None = None,
    n_retries: int = NEIGHBOR_GENERATOR_N_RETRIES,
    sample_multiplier: int = NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
) -> Array[f64]:
    if std is None:
        std = DEFAULT_VECTORIZED_NUMERIC_STD

    seed = np.random.RandomState() if seed is None else seed

    SAMPLE_SIZE = n * sample_multiplier
    BUFFER_SIZE = n + n * sample_multiplier
    neighbors = np.empty(BUFFER_SIZE, dtype=f64)
    offset = 0

    # We extend the range of stds to try to find neighbors
    if (
        std == DEFAULT_VECTORIZED_NUMERIC_STD
        and n_retries == NEIGHBOR_GENERATOR_N_RETRIES
    ):
        stds = _CACHED_LINSPACE
    else:
        stds = np.linspace(std, 1.0, n_retries + 1, endpoint=True)

    # Generate batches of n * BUFFER_MULTIPLIER candidates, filling the above
    # buffer until we have enough valid candidates.
    # We should not overflow as the buffer
    range_size = upper - lower

    for _std in stds:
        candidates = seed.normal(vector, _std * range_size, size=(SAMPLE_SIZE,))

        # Select all those which are not at the lower boundary
        valid_mask = np.logical_and(candidates >= lower, candidates <= upper)
        valid_candidates = candidates[valid_mask]
        n_candidates = len(valid_candidates)
        neighbors[offset : offset + n_candidates] = valid_candidates
        offset += n_candidates

        # We have enough neighbors, we can stop and return the vectorized values
        if offset >= n:
            return neighbors[:n]

    raise ValueError(
        f"Failed to find enough neighbors with {n_retries} retries."
        f" Given {n} neighbors, we only found {offset}."
        f" The `Normals` for sampling neighbors were"
        f" Normal(mu={vector}, sigma={list(stds)})"
        f" which were meant to find vectorized neighbors of the vector {vector},"
        " which was expected to be in the range"
        f" ({lower}, {lower}).",
    )


class Distribution(Protocol):
    lower_vectorized: f64
    upper_vectorized: f64

    def max_density(self) -> float: ...

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]: ...

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]: ...


@dataclass
class DiscretizedContinuousScipyDistribution(Distribution, Generic[DType]):
    steps: int

    rv: rv_continuous_frozen
    lower_vectorized: f64
    upper_vectorized: f64

    log_scale: bool = False
    # NOTE: Only required if you require log scaled quantization
    transformer: _Transformer[DType] | None = None

    _max_density: float | None = None
    _pdf_norm: float | None = None

    original_value_scale: tuple[DType, DType] | None = None

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("The number of steps must be at least 1.")

        if self.log_scale:
            if self.transformer is None:
                raise ValueError(
                    "A transformer is required for log-scaled distributions.",
                )

            # TODO: Not a hard requirement but simplifies a lot
            if (
                self.lower_vectorized != self.transformer.lower_vectorized
                or self.upper_vectorized != self.transformer.upper_vectorized
            ):
                raise ValueError("Vectorized scales of transformers must match.")

            orig = self.transformer.to_value(
                np.array(
                    [
                        self.transformer.lower_vectorized,
                        self.transformer.upper_vectorized,
                    ],
                ),
            )
            self.original_value_scale = tuple(orig)

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, self.__class__):
            return NotImplemented

        for key in self.__dict__:
            if key == "rv":
                if not _compare_rv(self.__dict__[key], value.__dict__[key]):
                    return False
            elif self.__dict__[key] != value.__dict__[key]:
                return False

        return True

    def max_density(self) -> float:
        if self._max_density is not None:
            return self._max_density

        _max, _sum = self._max_density_and_normalization_constant()
        self._max_density = _max
        self._pdf_norm = _sum
        return _max

    def _pdf_normalization_constant(self) -> float:
        if self._pdf_norm is not None:
            return self._pdf_norm

        _max, _sum = self._max_density_and_normalization_constant()
        self._max_density = _max
        self._pdf_norm = _sum
        return _sum

    def _max_density_and_normalization_constant(self) -> tuple[float, float]:
        # NOTE: It's likely when either one is needed, so will the other.
        # We just compute both at the same time as it's likely more cache friendly.
        _sum = 0.0
        _max = 0.0
        for chunk in self._meaningful_pdf_values():
            pdf = self.rv.pdf(chunk)
            _sum += pdf.sum()
            _max = max(_max, pdf.max())

        return _max, _sum

    def _meaningful_pdf_values(
        self,
        confidence: float = CONFIDENCE_FOR_NORMALIZATION_OF_DISCRETE,
    ) -> Iterator[Array[f64]]:
        if self.steps > ARANGE_CHUNKSIZE:
            lower, upper = (
                self.rv.ppf((1 - confidence) / 2),
                self.rv.ppf((1 + confidence) / 2),
            )
            lower = max(lower, self.lower_vectorized)
            upper = min(upper, self.upper_vectorized)
        else:
            lower, upper = self.lower_vectorized, self.upper_vectorized

        qlow, qhigh = self._quantize(x=np.array([lower, upper]))

        # If we're not on a log-scale, we do simple uniform distance steps
        if not self.log_scale:
            stepsize = (upper - lower) / (self.steps - 1)
            steps_intermediate = (qhigh - qlow) / stepsize + 1
            return linspace_chunked(
                qlow,
                qhigh,
                steps_intermediate,
                chunk_size=ARANGE_CHUNKSIZE,
            )

        # Problem is when we're on a log scale and the steps between qlow and qhigh
        # are not uniform. We have no idea how many points lie between qhigh and qlow
        # based on their values alone. For this we are forced to transform back to
        # the original scale and use the information there to determine the number of
        # steps.
        assert self.transformer is not None
        assert self.original_value_scale is not None
        orig_low, orig_high = self.original_value_scale
        qlow_orig, qhigh_orig = self.transformer.to_value(np.array([qlow, qhigh]))

        # Now we can calculate the stepsize between the original values
        # and hence see where qhigh and qlow lie in the original space to
        # calculate how many intermediate steps we need.
        stepsize = (orig_high - orig_low) / (self.steps - 1)
        steps_intermediate = (qhigh_orig - qlow_orig) / stepsize + 1

        return iter(
            self.transformer.to_vector(chunk)
            for chunk in linspace_chunked(
                qlow_orig,
                qhigh_orig,
                steps_intermediate,
                chunk_size=ARANGE_CHUNKSIZE,
            )
        )

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        return self._quantize(x=self.rv.rvs(size=n, random_state=seed))

    def _quantize(self, x: Array[f64]) -> Array[f64]:
        vectorized_bounds = (self.lower_vectorized, self.upper_vectorized)
        if not self.log_scale:
            return quantize(x, bounds=vectorized_bounds, bins=self.steps)

        assert self.original_value_scale
        return quantize_log(
            x,
            bounds=vectorized_bounds,
            scale_slice=(self.original_value_scale),
            bins=self.steps,
        )

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        valid_entries = np.where(
            (vector >= self.lower_vectorized) & (vector <= self.upper_vectorized),
            vector,
            np.nan,
        )
        pdf = self.rv.pdf(valid_entries) / self._pdf_normalization_constant()

        # By definition, we don't allow NaNs in the pdf
        return np.nan_to_num(pdf, nan=0)

    def neighborhood(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
        n_retries: int = NEIGHBOR_GENERATOR_N_RETRIES,
        sample_multiplier: int = NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
    ) -> Array[f64]:
        if std is None:
            std = DEFAULT_VECTORIZED_NUMERIC_STD

        assert n < 1000000, "Can only generate less than 1 million neighbors."
        seed = np.random.RandomState() if seed is None else seed
        lower, upper = self.lower_vectorized, self.upper_vectorized
        steps = self.steps

        qvector = self._quantize(np.array([vector]))[0]

        # In the easiest case, the amount of neighbors we need is more than the amount
        # possible, in this case, we can skip our sampling and just generate all
        # neighbors, excluding the current value
        n_available = steps - 1
        if n >= n_available:
            if self.log_scale:
                assert self.transformer is not None
                assert self.original_value_scale is not None
                orig_low, orig_high = self.original_value_scale
                qvector_orig: DType = self.transformer.to_value(np.array([qvector]))[0]

                # Now we can calculate the stepsize between the original values
                # and hence see where qhigh and qlow lie in the original space to
                # calculate how many intermediate steps we need.
                stepsize = (orig_high - orig_low) / (self.steps - 1)

                # Edge case for when qcenter is the lower bound
                steps_to_take = int((qvector_orig - orig_low) / stepsize) + 1
                if steps_to_take == 1:
                    bottom = np.array([])
                else:
                    bottom_orig = np.linspace(
                        orig_low,  # type: ignore
                        qvector_orig,  # type: ignore
                        steps_to_take - 1,
                        endpoint=False,
                    )
                    bottom = self.transformer.to_vector(bottom_orig)

                top_orig = np.linspace(
                    qvector_orig + stepsize,
                    orig_high,  # type: ignore
                    steps - steps_to_take,
                )

                top = self.transformer.to_vector(top_orig)
            else:
                stepsize = (upper - lower) / (self.steps - 1)

                # Edge case for when qcenter is the lower bound
                steps_to_take = int((qvector - lower) / stepsize) + 1
                if steps_to_take == 1:
                    bottom = np.array([])
                else:
                    bottom = np.linspace(lower, qvector, steps_to_take, endpoint=False)

                top = np.linspace(qvector + stepsize, upper, steps - steps_to_take)

            return np.concatenate((bottom, top))

        # Otherwise, we use a repeated sampling strategy where we slowly increase the
        # std of a normal, centered on `center`, slowly expanding `std` such that
        # rejection won't fail.

        # We set up a buffer that can hold the number of neighbors we need, plus some
        # extra excess from sampling, preventing us from having to reallocate memory.
        # We also include the initial value in the buffer, as we will remove it later.
        SAMPLE_SIZE = n * sample_multiplier
        BUFFER_SIZE = n * (sample_multiplier + 1)
        neighbors = np.empty(BUFFER_SIZE + 1, dtype=f64)
        neighbors[0] = qvector
        offset = 1  # Indexes into current progress of filling buffer

        # We extend the range of stds to try to find neighbors
        if (
            std == DEFAULT_VECTORIZED_NUMERIC_STD
            and n_retries == NEIGHBOR_GENERATOR_N_RETRIES
        ):
            stds = _CACHED_LINSPACE
        else:
            stds = np.linspace(std, 1.0, n_retries + 1, endpoint=True)

        range_size = upper - lower
        for _std in stds:
            # Generate candidates in vectorized space
            candidates = seed.normal(qvector, _std * range_size, size=SAMPLE_SIZE)
            valid_candidates = candidates[(candidates >= lower) & (candidates <= upper)]

            # Transform to quantized space
            candidates_quantized = self._quantize(valid_candidates)

            uniq = np.unique(candidates_quantized)
            new_uniq = np.setdiff1d(uniq, neighbors[:offset], assume_unique=True)

            n_new_unique = len(new_uniq)
            neighbors[offset : offset + n_new_unique] = new_uniq
            offset += n_new_unique

            # We have enough neighbors, we can stop
            if offset - 1 >= n:
                # Ensure we don't include the initial value point
                return neighbors[1 : n + 1]

        raise ValueError(
            f"Failed to find enough neighbors with {n_retries} retries."
            f" Given {n=} neighbors to generate, we only found {offset - 1}."
            f" The normal's for sampling neighbors were Normal({vector}, {list(stds)})"
            f" which were meant to find neighbors of {vector}. in the range"
            f" ({self.lower_vectorized}, {self.upper_vectorized}).",
        )


@dataclass
class ScipyDiscreteDistribution(Distribution):
    rv: rv_discrete_frozen
    lower_vectorized: f64
    upper_vectorized: f64
    _max_density: float

    def sample_vector(
        self,
        n: int | None = None,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        return self.rv.rvs(size=n, random_state=seed).astype(f64)

    def max_density(self) -> float:
        return float(self._max_density)

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        # By definition, we don't allow NaNs in the pdf
        pdf = self.rv.pmf(vector)
        return np.nan_to_num(pdf, nan=0)

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, self.__class__):
            return NotImplemented

        for key in self.__dict__:
            if key == "rv":
                if not _compare_rv(self.__dict__[key], value.__dict__[key]):
                    return False
            elif self.__dict__[key] != value.__dict__[key]:
                return False

        return True


@dataclass
class UniformIntegerNormalizedDistribution(Distribution):
    lower_vectorized: ClassVar[f64] = f64(0.0)
    upper_vectorized: ClassVar[f64] = f64(1.0)

    size: int

    def sample_vector(
        self,
        n: int | None = None,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        seed = np.random.RandomState() if seed is None else seed
        ints = seed.randint(low=0, high=self.size, size=n, dtype=i64)
        return np.true_divide(ints, (self.size - 1), dtype=f64)

    def max_density(self) -> float:
        return 1 / self.size

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        valid_mask = (
            (vector >= self.lower_vectorized)
            & (vector <= self.upper_vectorized)
            & is_close_to_integer(vector * (self.size - 1), atol=ATOL)
        )
        return np.where(valid_mask, self.max_density(), 0)

    def neighborhood(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
        n_retries: int = NEIGHBOR_GENERATOR_N_RETRIES,
        sample_multiplier: int = NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
    ) -> Array[f64]:
        return quantized_neighborhood(
            vector,
            n,
            std=std,
            seed=seed,
            n_retries=n_retries,
            sample_multiplier=sample_multiplier,
            lower=self.lower_vectorized,
            upper=self.upper_vectorized,
            bins=self.size,
        )


@dataclass
class UnitUniformContinuousDistribution(Distribution):
    lower_vectorized: ClassVar[f64] = f64(0.0)
    upper_vectorized: ClassVar[f64] = f64(1.0)

    pdf_max_density: float

    def sample_vector(
        self,
        n: int | None = None,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        seed = np.random.RandomState() if seed is None else seed
        return seed.uniform(self.lower_vectorized, self.upper_vectorized, size=n)

    def max_density(self) -> float:
        return self.pdf_max_density

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        # By definition, we don't allow NaNs in the pdf
        return np.where(
            (vector >= self.lower_vectorized) & (vector <= self.upper_vectorized),
            self.pdf_max_density,
            0,
        )

    def neighborhood(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
        n_retries: int = NEIGHBOR_GENERATOR_N_RETRIES,
        sample_multiplier: int = NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
    ) -> Array[f64]:
        return continuous_neighborhood(
            vector,
            n,
            std=std,
            seed=seed,
            n_retries=n_retries,
            sample_multiplier=sample_multiplier,
            lower=self.lower_vectorized,
            upper=self.upper_vectorized,
        )


@dataclass
class UniformIntegerDistribution(Distribution):
    size: int

    lower_vectorized: f64 = field(init=False)
    upper_vectorized: f64 = field(init=False)

    def __post_init__(self) -> None:
        self.lower_vectorized = f64(0)
        self.upper_vectorized = f64(self.size - 1)

    def sample_vector(
        self,
        n: int | None = None,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        seed = np.random.RandomState() if seed is None else seed
        ints = seed.randint(low=0, high=self.size, size=n, dtype=i64)
        return ints.astype(f64)

    def max_density(self) -> float:
        return 1 / self.size

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        # By definition, we don't allow NaNs in the pdf
        valid_mask = (
            (vector >= self.lower_vectorized)
            & (vector <= self.upper_vectorized)
            & is_close_to_integer(vector, atol=ATOL)
        )
        return np.where(valid_mask, self.max_density(), 0)

    def neighborhood(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
        n_retries: int = NEIGHBOR_GENERATOR_N_RETRIES,
        sample_multiplier: int = NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
    ) -> Array[f64]:
        # Different than other neighborhoods as it's unnormalized and
        # the quantization is directly integers.
        if std is None:
            std = DEFAULT_VECTORIZED_NUMERIC_STD

        assert n < 1000000, "Can only generate less than 1 million neighbors."
        seed = np.random.RandomState() if seed is None else seed
        lower, upper = self.lower_vectorized, self.upper_vectorized
        steps = self.size

        qvector = np.rint(vector).astype(i64)

        # In the easiest case, the amount of neighbors we need is more than the amount
        # possible, in this case, we can skip our sampling and just generate all
        # neighbors, excluding the current value
        n_available = steps - 1
        if n >= n_available:
            if qvector == 0:
                return np.arange(1, steps, dtype=f64)
            if qvector == steps - 1:
                return np.arange(0, steps - 1, dtype=f64)

            bottom = np.arange(0, qvector)
            top = np.arange(qvector + 1, steps)
            return np.concatenate((bottom, top)).astype(f64)

        # Otherwise, we use a repeated sampling strategy where we slowly increase the
        # std of a normal, centered on `center`, slowly expanding `std` such that
        # rejection won't fail.

        # We set up a buffer that can hold the number of neighbors we need, plus some
        # extra excess from sampling, preventing us from having to reallocate memory.
        # We also include the initial value in the buffer, as we will remove it later.
        SAMPLE_SIZE = n * sample_multiplier
        BUFFER_SIZE = n * (sample_multiplier + 1)
        neighbors = np.empty(BUFFER_SIZE + 1, dtype=f64)
        neighbors[0] = qvector
        offset = 1  # Indexes into current progress of filling buffer

        # We extend the range of stds to try to find neighbors
        if (
            std == DEFAULT_VECTORIZED_NUMERIC_STD
            and n_retries == NEIGHBOR_GENERATOR_N_RETRIES
        ):
            stds = _CACHED_LINSPACE
        else:
            stds = np.linspace(std, 1.0, n_retries + 1, endpoint=True)

        range_size = upper - lower
        for _std in stds:
            # Generate candidates in vectorized space
            candidates = seed.normal(qvector, _std * range_size, size=SAMPLE_SIZE)
            valid_candidates = candidates[(candidates >= lower) & (candidates <= upper)]

            # Transform to quantized space
            candidates_quantized = quantize(
                valid_candidates,
                bounds=(lower, upper),
                bins=steps,
            )

            uniq = np.unique(candidates_quantized)
            new_uniq = np.setdiff1d(uniq, neighbors[:offset], assume_unique=True)

            n_new_unique = len(new_uniq)
            neighbors[offset : offset + n_new_unique] = new_uniq
            offset += n_new_unique

            # We have enough neighbors, we can stop
            if offset - 1 >= n:
                # Ensure we don't include the initial value point
                return neighbors[1 : n + 1]

        raise ValueError(
            f"Failed to find enough neighbors with {n_retries} retries."
            f" Given {n=} neighbors to generate, we only found {offset - 1}."
            f" The normal's for sampling neighbors were Normal({vector}, {list(stds)})"
            f" which were meant to find neighbors of {vector}. in the range"
            f" ({self.lower_vectorized}, {self.upper_vectorized}).",
        )


@dataclass
class WeightedIntegerDiscreteDistribution(Distribution):
    size: int
    probabilities: Array[f64]

    _max_density: float = field(init=False)
    lower_vectorized: f64 = field(init=False)
    upper_vectorized: f64 = field(init=False)

    def __post_init__(self) -> None:
        self._max_density = float(self.probabilities.max())
        self.lower_vectorized = f64(0)
        self.upper_vectorized = f64(self.size - 1)

    def sample_vector(
        self,
        n: int | None = None,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        seed = np.random.RandomState() if seed is None else seed
        return seed.choice(self.size, size=n, p=self.probabilities, replace=True)  # type: ignore

    def max_density(self) -> float:
        return float(self._max_density)

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        # By definition, we don't allow NaNs in the pdf
        valid_mask = (
            (vector >= self.lower_vectorized)
            & (vector <= self.upper_vectorized)
            & is_close_to_integer(vector, atol=ATOL)
        )

        # Bring it all into range to index by
        nan_filled = np.nan_to_num(vector, nan=0)
        xx = np.clip(nan_filled, 0, self.size - 1).astype(i64)
        pdf = self.probabilities[xx]
        return np.where(valid_mask, pdf, 0)

    def neighborhood(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,  # noqa: ARG002
        seed: RandomState | None = None,
    ) -> Array[f64]:
        seed = np.random.RandomState() if seed is None else seed
        pivot = int(np.rint(vector))
        bot = np.arange(0, pivot)
        top = np.arange(pivot + 1, self.size)
        choices = np.concatenate((bot, top)).astype(f64)
        seed.shuffle(choices)
        return choices[:n]


@dataclass
class ScipyContinuousDistribution(Distribution):
    rv: rv_continuous_frozen
    lower_vectorized: f64
    upper_vectorized: f64

    _max_density: float
    _pdf_norm: float = 1

    def sample_vector(
        self,
        n: int | None = None,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        return self.rv.rvs(size=n, random_state=seed).astype(f64)

    def max_density(self) -> float:
        return float(self._max_density)

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        pdf = self.rv.pdf(vector) / self._pdf_norm

        # By definition, we don't allow NaNs in the pdf
        return np.nan_to_num(pdf, nan=0)

    def neighborhood(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,
        seed: RandomState | None = None,
        n_retries: int = NEIGHBOR_GENERATOR_N_RETRIES,
        sample_multiplier: int = NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
    ) -> Array[f64]:
        return continuous_neighborhood(
            vector,
            n,
            std=std,
            seed=seed,
            n_retries=n_retries,
            sample_multiplier=sample_multiplier,
            lower=self.lower_vectorized,
            upper=self.upper_vectorized,
        )

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, self.__class__):
            return NotImplemented

        for key in self.__dict__:
            if key == "rv":
                if not _compare_rv(self.__dict__[key], value.__dict__[key]):
                    return False
            elif self.__dict__[key] != value.__dict__[key]:
                return False

        return True


@dataclass
class ConstantVectorDistribution(Distribution):
    vector_value: f64

    @property
    def lower_vectorized(self) -> f64:
        return self.vector_value

    @property
    def upper_vectorized(self) -> f64:
        return self.vector_value

    def max_density(self) -> float:
        return 1.0

    def sample_vector(
        self,
        n: int | None = None,
        *,
        seed: RandomState | None = None,  # noqa: ARG002
    ) -> f64 | Array[f64]:
        if n is None:
            return self.vector_value

        return np.full((n,), self.vector_value, dtype=f64)

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        return (vector == self.vector_value).astype(f64)
