from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Iterator
from typing_extensions import Protocol

import numpy as np

from ConfigSpace.functional import (
    is_close_to_integer,
    linspace_chunked,
    quantize,
    quantize_log,
)
from ConfigSpace.hyperparameters.hp_components import ATOL, Transformer
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
"""The default standard deviation for generating neighborhoods of vectorized values."""

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
    """Create a neighborhood of `n` neighbors around `vector` with a normal distribution.

    The neighborhood is created by sampling from a normal distribution centered around
    `vector` with a standard deviation of `std`. The samples are then quantized to the
    range `[lower, upper]` with `bins` bins. The number of samples is `n`.

    !!! warning

        If there are not enough unique neighbors to sample from, the function will
        return less than `n` neighbors.

    Args:
        vector: The center of the neighborhood.
        n: The number of neighbors to generate.
        lower: The lower bound of the quantized range.
        upper: The upper bound of the quantized range.
        bins: The number of bins to quantize the samples into.
        std: The standard deviation of the normal distribution. If `None` will use
            [`DEFAULT_VECTORIZED_NUMERIC_STD`][ConfigSpace.hyperparameters.distributions.DEFAULT_VECTORIZED_NUMERIC_STD].
        seed: The random seed to use.
        n_retries:
            The number of retries to attempt to generate unique neighbors.
            Each retry increases the standard deviation of the normal distribution to prevent
            rejection sampling from failing.
        sample_multiplier:
            A multiplier which multiplies by `n` to determine the number of samples to
            generate for try. By oversampling, we prevent having to repeated calls to
            both sampling and unique checking.

            However, oversampling makes a tradeoff when the `std` is not high enough to
            generate `n` unique neighbors, effectively sampling more of the same duplicates.

            Tuning this may be beneficial in unique circumstances, however we advise leaving
            this as a default.

    Returns:
        An array of `n` neighbors around `vector`.
    """  # noqa: E501
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

        _range: Array[f64] = np.arange(0, bins, dtype=np.float64)
        bot = _range[:qint]
        top = _range[qint + 1 :]
        return np.concatenate((bot, top)) / (bins - 1)  # type: ignore

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
    neighbors: Array[f64] = np.empty(BUFFER_SIZE + 1, dtype=f64)
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
    """Create a neighborhood of `n` neighbors around `vector` with a normal distribution.

    Args:
        vector: The center of the neighborhood.
        n: The number of neighbors to generate.
        lower: The lower bound of the neighborhood range.
        upper: The upper bound of the neighborhood range.
        std: The standard deviation of the normal distribution. If `None` will use
            [`DEFAULT_VECTORIZED_NUMERIC_STD`][ConfigSpace.hyperparameters.distributions.DEFAULT_VECTORIZED_NUMERIC_STD].
        seed: The random seed to use.
        n_retries:
            The number of retries to attempt to generate unique neighbors.
            Each retry increases the standard deviation of the normal distribution to
            prevent rejection sampling from failing.
        sample_multiplier:
            A multiplier which multiplies by `n` to determine the number of samples to
            generate for try. By oversampling, we prevent having to repeated calls to
            sampling. This prevents having to do more rounds of sampling when too many
            samples are out of bounds, useful for when the `vector` is near the bounds.

            Tuning this may be beneficial in unique circumstances, however we advise
            leaving this as a default.

    Returns:
        An array of `n` neighbors around `vector`.
    """  # noqa: E501
    if std is None:
        std = DEFAULT_VECTORIZED_NUMERIC_STD

    seed = np.random.RandomState() if seed is None else seed

    SAMPLE_SIZE = n * sample_multiplier
    BUFFER_SIZE = n + n * sample_multiplier
    neighbors: Array[f64] = np.empty(BUFFER_SIZE, dtype=f64)
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
    """A protocol for distributions.

    A distribution is defined by some **vectorized** space and allows us to
    draw samples from it, calculate the probability density function (pdf) and
    provide a maximum density.
    """

    lower_vectorized: f64
    """The lower bound of the vectorized space."""

    upper_vectorized: f64
    """The lower bound of the vectorized space."""

    def max_density(self) -> float:
        """Return the maximum density of the distribution."""
        ...

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        """Sample `n` values from the distribution.

        Samples generated do not have to be unique.

        !!! note
            Generated samples must be within the bounds defined by `lower_vectorized`
            and `upper_vectorized`.

        Args:
            n: The number of samples to generate.
            seed: The random seed to use.

        Returns:
            An array of `n` samples, **not** guaranteed to be unique.
        """
        ...

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        """Calculate the probability density function (pdf) of all elements in `vector`.

        Args:
            vector: The vectorized values to calculate the pdf for.

        Returns:
            The pdf of all elements in `vector`. If an element is outside the bounds
            defined by `lower_vectorized` and `upper_vectorized`, the pdf should be 0.
        """
        ...


@dataclass
class DiscretizedContinuousScipyDistribution(Distribution, Generic[DType]):
    """A wrapper to create discrete samples from a continuous scipy distribution.

    This class allows us to take pre-existing scipy distributions which are
    defined over a continuous space, and transform them into discrete intervals.

    This can also handle adapt a distribution, such that the discrete bins
    can be distributed on a log scale over the distributrion.

    !!! note
        If providing `log=True`, you must also provide the `Transformer` which
        will be used to transform from vectorized space to the value space.
    """

    steps: int
    """The number of steps to discretize the distribution into."""

    rv: rv_continuous_frozen
    """The continuous scipy distribution to discretize."""

    lower_vectorized: f64
    """The lower bound of the vectorized space."""

    upper_vectorized: f64
    """The upper bound of the vectorized space."""

    log_scale: bool = False
    """Whether the distribution is on a log scale."""

    # NOTE: Only required if you require log scaled quantization
    transformer: Transformer[DType] | None = None
    """The transformer to use for log-scaled distributions.

    Only required if `log_scale=True`.
    """

    _max_density: float | None = None
    """The maximum density of the distribution.

    If left as `None`, will be calculated on first call.
    """

    _pdf_norm: float | None = None
    """The normalization constant for the pdf.

    If left as `None`, will be calculated on first call.
    """

    original_value_scale: tuple[DType, DType] | None = None
    """The original value scale of the transformer.

    This is used on log-scale transformation.
    """

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
        """Return the maximum density of the distribution."""
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
        """Sample `n` values from the distribution."""
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
        """Calculate the probability density function (pdf) of all elements in `vector`.

        Args:
            vector: The vectorized values to calculate the pdf for.

        Returns:
            The pdf of all elements in `vector`. If an element is outside the bounds
            defined by `lower_vectorized` and `upper_vectorized`, the pdf is `0`.
        """
        valid_entries = np.where(
            (vector >= self.lower_vectorized) & (vector <= self.upper_vectorized),
            vector,
            np.nan,
        )
        pdf = self.rv.pdf(valid_entries) / self._pdf_normalization_constant()

        # By definition, we don't allow NaNs in the pdf
        return np.nan_to_num(pdf, nan=0)  # type: ignore

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
        """Create a neighborhood of `n` neighbors around `vector` with a normal distribution.

        The neighborhood is created by sampling from a normal distribution centered
        around `vector` with a standard deviation of `std`. The samples are then
        quantized to the range `[lower, upper]` with `bins` bins. The number of samples
        is `n`.

        !!! warning

            If there are not enough unique neighbors to sample from, the function will
            return less than `n` neighbors.

        Args:
            vector: The center of the neighborhood.
            n: The number of neighbors to generate.
            std: The standard deviation of the normal distribution. If `None` will use
                [`DEFAULT_VECTORIZED_NUMERIC_STD`][ConfigSpace.hyperparameters.distributions.DEFAULT_VECTORIZED_NUMERIC_STD].
            seed: The random seed to use.
            n_retries:
                The number of retries to attempt to generate unique neighbors.
                Each retry increases the standard deviation of the normal distribution
                to prevent rejection sampling from failing.
            sample_multiplier:
                A multiplier which multiplies by `n` to determine the number of samples
                to generate for try. By oversampling, we prevent having to repeated
                calls to both sampling and unique checking.

                However, oversampling makes a tradeoff when the `std` is not high
                enough to generate `n` unique neighbors,
                effectively sampling more of the same duplicates.

                Tuning this may be beneficial in unique circumstances, however we advise
                leaving this as a default.

        Returns:
            An array of `n` neighbors around `vector`.
        """  # noqa: E501
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
                        dtype=f64,
                    )
                    bottom = self.transformer.to_vector(bottom_orig)

                top_orig = np.linspace(
                    qvector_orig + stepsize,
                    orig_high,  # type: ignore
                    steps - steps_to_take,
                    dtype=f64,
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
        neighbors: Array[f64] = np.empty(BUFFER_SIZE + 1, dtype=f64)
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
    """A wrapper to create discrete samples from a scipy discrete distribution."""

    rv: rv_discrete_frozen
    """The discrete scipy distribution to use."""

    _max_density: float
    """The maximum density of the distribution."""

    lower_vectorized: f64
    """The lower bound of the vectorized space."""

    upper_vectorized: f64
    """The upper bound of the vectorized space."""

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        """Sample `n` values from the distribution."""
        return self.rv.rvs(size=n, random_state=seed).astype(f64)  # type: ignore

    def max_density(self) -> float:
        """Return the maximum density of the distribution."""
        return float(self._max_density)

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        """Calculate the probability density function (pdf) of all elements in `vector`.

        Args:
            vector: The vectorized values to calculate the pdf for.

        Returns:
            The pdf of all elements in `vector`. If an element is outside the bounds
            defined by `lower_vectorized` and `upper_vectorized`, the pdf is `0`.
        """
        pdf = self.rv.pmf(vector)
        return np.nan_to_num(pdf, nan=0)  # type: ignore

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
    """A uniform over (0, 1) that is quantized to provide `size` different bins."""

    size: int
    """The number of steps to discretize the distribution into."""

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("The number of steps must be at least 1.")
        self.lower_vectorized = f64(0.0)
        self.upper_vectorized = f64(1.0)

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        """Sample `n` values from the distribution."""
        seed = np.random.RandomState() if seed is None else seed
        ints = seed.randint(low=0, high=self.size, size=n, dtype=i64)
        return np.true_divide(ints, (self.size - 1), dtype=f64)

    def max_density(self) -> float:
        """Return the maximum density of the distribution."""
        return 1 / self.size

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        """Calculate the probability density function (pdf) of all elements in `vector`.

        Args:
            vector: The vectorized values to calculate the pdf for.

        Returns:
            The pdf of all elements in `vector`. If an element is outside the bounds
            of `(0, 1)`, or does not fall onto one of the quantized bins,
            the `pdf` is `0`.
        """
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
        """Please see [`quantized_neighborhood`][ConfigSpace.hyperparameters.distributions.quantized_neighborhood]."""  # noqa: E501
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
    """A uniform distribution over the unit interval (0, 1)."""

    lower_vectorized: f64 = field(init=False)
    """The lower bound of the vectorized space. In this case always 0."""

    upper_vectorized: f64 = field(init=False)
    """The upper bound of the vectorized space. In this case always 1."""

    pdf_max_density: float
    """The maximum density of the distribution provided by the consumer of this"""

    def __post_init__(self) -> None:
        self.lower_vectorized = f64(0.0)
        self.upper_vectorized = f64(1.0)

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        """Sample `n` values from the distribution."""
        seed = np.random.RandomState() if seed is None else seed
        return seed.uniform(self.lower_vectorized, self.upper_vectorized, size=n)

    def max_density(self) -> float:
        """Return the maximum density of the distribution."""
        return self.pdf_max_density

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        # By definition, we don't allow NaNs in the pdf
        """Calculate the probability density function (pdf) of all elements in `vector`.

        Args:
            vector: The vectorized values to calculate the pdf for.

        Returns:
            The pdf of all elements in `vector`. If an element is outside the bounds
            defined by `(0, 1)` the pdf is 0.
        """
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
        """Please see [`continuous_neighborhood`][ConfigSpace.hyperparameters.distributions.continuous_neighborhood]."""  # noqa: E501
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
    """A uniform distribution over the integers in the range `[0, size - 1]`."""

    size: int
    """The number of steps to discretize the distribution into."""

    lower_vectorized: f64 = field(init=False)
    """The lower bound of the vectorized space. In this case always 0."""

    upper_vectorized: f64 = field(init=False)
    """The upper bound of the vectorized space. In this case `size - 1`."""

    def __post_init__(self) -> None:
        self.lower_vectorized = f64(0)
        self.upper_vectorized = f64(self.size - 1)

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        """Sample `n` values from the distribution."""
        seed = np.random.RandomState() if seed is None else seed
        ints = seed.randint(low=0, high=self.size, size=n, dtype=i64)
        return ints.astype(f64)

    def max_density(self) -> float:
        """Return the maximum density of the distribution."""
        return 1 / self.size

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        """Calculate the probability density function (pdf) of all elements in `vector`.

        Args:
            vector: The vectorized values to calculate the pdf for.

        Returns:
            The pdf of all elements in `vector`. If an element is outside the bounds
            defined by `(0, size - 1)` the pdf is 0. If the element is not close to
            an integer, the pdf is 0.
        """
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
        """Create a neighborhood of `n` neighbors around `vector` with a normal distribution.

        The neighborhood is created by sampling from a normal distribution centered around
        `vector` with a standard deviation of `std`. The samples are then quantized to the
        range `[lower, upper]` with `bins` bins. The number of samples is `n`.

        !!! warning

            If there are not enough unique neighbors to sample from, the function will
            return less than `n` neighbors.

        Args:
            vector: The center of the neighborhood.
            n: The number of neighbors to generate.
            std: The standard deviation of the normal distribution. If `None` will use
                [`DEFAULT_VECTORIZED_NUMERIC_STD`][ConfigSpace.hyperparameters.distributions.DEFAULT_VECTORIZED_NUMERIC_STD].
            seed: The random seed to use.
            n_retries:
                The number of retries to attempt to generate unique neighbors.
                Each retry increases the standard deviation of the normal distribution to prevent
                rejection sampling from failing.
            sample_multiplier:
                A multiplier which multiplies by `n` to determine the number of samples to
                generate for try. By oversampling, we prevent having to repeated calls to
                both sampling and unique checking.

                However, oversampling makes a tradeoff when the `std` is not high enough to
                generate `n` unique neighbors, effectively sampling more of the same duplicates.

                Tuning this may be beneficial in unique circumstances, however we advise leaving
                this as a default.

        Returns:
            An array of `n` neighbors around `vector`.
        """  # noqa: E501
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
        neighbors: Array[f64] = np.empty(BUFFER_SIZE + 1, dtype=f64)
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
    """A discrete distribution over integers with weights to each.

    This can be primarily used for defining a weighted distribution over a set
    of choices, like a `Categorical`.
    """

    size: int
    """The number of steps to discretize the distribution into."""

    probabilities: Array[f64]
    """The probabilities of each integer in the range `[0, size - 1]`."""

    lower_vectorized: f64 = field(init=False)
    """The lower bound of the vectorized space. In this case always 0."""

    upper_vectorized: f64 = field(init=False)
    """The upper bound of the vectorized space. In this case `size - 1`."""

    _max_density: float = field(init=False)

    def __post_init__(self) -> None:
        self._max_density = float(self.probabilities.max())
        self.lower_vectorized = f64(0)
        self.upper_vectorized = f64(self.size - 1)

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        """Sample `n` values from the distribution."""
        seed = np.random.RandomState() if seed is None else seed
        return seed.choice(
            self.size,
            size=n,
            p=self.probabilities,
            replace=True,
        )  # type: ignore

    def max_density(self) -> float:
        """Return the maximum density of the distribution.

        In this case, it will be the maximum probability provided.
        """
        return float(self._max_density)

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        """Calculate the probability density function (pdf) of all elements in `vector`.

        Args:
            vector: The vectorized values to calculate the pdf for.

        Returns:
            The pdf of all elements in `vector`. If an element is outside the bounds
            defined by `(0, size - 1)` the pdf is 0. If the element is not close to
            an integer, the pdf is 0.
        """
        # By definition, we don't allow NaNs in the pdf
        valid_mask = (
            (vector >= self.lower_vectorized)
            & (vector <= self.upper_vectorized)
            & is_close_to_integer(vector, atol=ATOL)
        )

        # Bring it all into range to index by
        nan_filled: Array[f64] = np.nan_to_num(vector, nan=0)
        xx: Array[i64] = np.clip(nan_filled, 0, self.size - 1).astype(np.intp)
        pdf = self.probabilities[xx]
        return np.where(valid_mask, pdf, 0)


@dataclass
class ScipyContinuousDistribution(Distribution):
    """A wrapper to create continuous samples from a scipy continuous distribution."""

    rv: rv_continuous_frozen
    """The continuous scipy distribution to use."""

    lower_vectorized: f64
    """The lower bound of the vectorized space."""

    upper_vectorized: f64
    """The upper bound of the vectorized space."""

    _max_density: float
    _pdf_norm: float = 1

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,
    ) -> Array[f64]:
        """Sample `n` values from the distribution."""
        return self.rv.rvs(size=n, random_state=seed).astype(f64)  # type: ignore

    def max_density(self) -> float:
        """Return the maximum density of the distribution."""
        return float(self._max_density)

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        """Calculate the probability density function (pdf) of all elements in `vector`.

        Args:
            vector: The vectorized values to calculate the pdf for.

        Returns:
            The pdf of all elements in `vector`. If an element is outside the bounds
            defined by `lower_vectorized` and `upper_vectorized`, the pdf is `0`.
        """
        pdf = self.rv.pdf(vector) / self._pdf_norm

        # By definition, we don't allow NaNs in the pdf
        return np.nan_to_num(pdf, nan=0)  # type: ignore

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
        """Please see [`continuous_neighborhood`][ConfigSpace.hyperparameters.distributions.continuous_neighborhood]."""  # noqa: E501
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
    """A distribution that always returns the same constant value."""

    vector_value: f64
    """The constant vector value to return."""

    def __post_init__(self) -> None:
        if not np.isfinite(self.vector_value):
            raise ValueError("The constant value must be finite.")
        self.lower_vectorized = self.vector_value
        self.upper_vectorized = self.vector_value

    def max_density(self) -> float:
        """Return the maximum density of the distribution.

        Always 1.0 as the density is always 1 at the constant value.
        """
        return 1.0

    def sample_vector(
        self,
        n: int,
        *,
        seed: RandomState | None = None,  # noqa: ARG002
    ) -> Array[f64]:
        """Sample `n` values from the distribution."""
        return np.full((n,), self.vector_value, dtype=f64)

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        """Calculate the probability density function (pdf) of all elements in `vector`.

        Args:
            vector: The vectorized values to calculate the pdf for.

        Returns:
            The pdf of all elements in `vector`. If an element is not equal to the
            constant value, the pdf is `0`.
        """
        return (vector == self.vector_value).astype(f64)  # type: ignore
