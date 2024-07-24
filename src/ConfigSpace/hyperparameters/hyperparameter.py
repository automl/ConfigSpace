from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    overload,
)
from typing_extensions import Self, deprecated

import numpy as np

from ConfigSpace.types import DType, NotSet, Number, ValueT, _NotSet, f64, i64

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.distributions import Distribution
    from ConfigSpace.hyperparameters.hp_components import Neighborhood, Transformer
    from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter
    from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter
    from ConfigSpace.types import Array, Mask


@dataclass(init=False)
class Hyperparameter(ABC, Generic[ValueT, DType]):
    """Base class for all hyperparameters in the configuration space.

    Please see the [reference page](../../../reference/hyperparameters.md) for more.
    """

    ORDERABLE: ClassVar[bool] = False
    """If the hyperparameter values have an order. This is used for
    conditionals and forbiddens relying on relationships.
    """

    LEGAL_VALUE_TYPES: ClassVar[tuple[type, ...] | Literal["all"]] = "all"
    """The types of values that are legal for this hyperparameter. If set to
    `"all"` any type is legal. Otherwise, a tuple of types can be provided.
    """

    name: str
    """Name of the hyperparameter, with which it can be accessed."""

    default_value: ValueT
    """The default value of this hyperparameter."""

    meta: Mapping[Hashable, Any] | None
    """Field for holding meta data provided by the user. Not used by the ConfigSpace."""

    size: int | float
    """Size of the hyperparameter. For integer and choice hyperparameters this
    is the number of possible values the hyperparameter can take on within the
    specified range. For continuous hyperparameters this is usually `np.inf`.
    """

    _vector_dist: Distribution = field(repr=False)
    _normalized_default_value: f64 = field(repr=False)
    _transformer: Transformer[DType] = field(repr=False)
    _neighborhood: Neighborhood = field(repr=False, compare=False)
    _value_cast: Callable[[DType], ValueT] | None = field(repr=False, compare=False)
    _neighborhood_size: (
        float | Callable[[ValueT | DType | _NotSet | None], int | float]
    ) = field(
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        name: str,
        default_value: ValueT,
        vector_dist: Distribution,
        transformer: Transformer[DType],
        neighborhood: Neighborhood,
        size: int | float,
        neighborhood_size: float | int | Callable[[DType | ValueT | None], int | float],
        value_cast: Callable[[DType], ValueT] | None,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """Initialize a hyperparameter.

        Args:
            name:
                Name of the hyperparameter, with which it can be accessed.
            default_value:
                The default value of this hyperparameter.
            vector_dist:
                The distribution of the hyperparameter in vector space.
            transformer:
                The transformer to convert between value and vector space.
            neighborhood:
                The function to sample neighbors from the hyperparameter.
            size:
                Size of the hyperparameter. For integer and choice hyperparameters
                this is the number of possible values the hyperparameter can take on
                within the specified range. For continuous hyperparameters this is
                usually `np.inf`.
            neighborhood_size:
                The number of neighbors to sample from the hyperparameter. This can
                be a fixed number or a function that takes the current value and
                returns the number of neighbors to sample.
            value_cast:
                A function to cast the value to a different type. This is useful
                for ensuring when removing from a nunmpy array of hyperparameter values,
                that the type is preserved.
            meta:
                Field for holding meta data provided by the user. Not used by the
                ConfigSpace.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"Name must be a string, got {name} ({type(name)})",
            )

        self.name = name
        self.default_value = default_value
        self.meta = meta
        self.size = size

        self._vector_dist = vector_dist
        self._transformer = transformer
        self._neighborhood = neighborhood
        self._neighborhood_size = neighborhood_size  # type: ignore
        self._value_cast = value_cast

        if not self.legal_value(self.default_value):
            raise ValueError(
                f"Illegal default value {self.default_value} for"
                f" hyperparamter '{self.name}'.",
            )

        self._normalized_default_value = self.to_vector(self.default_value)

    @property
    def lower_vectorized(self) -> f64:
        """Lower bound of the hyperparameter in vector space."""
        return self._vector_dist.lower_vectorized

    @property
    def upper_vectorized(self) -> f64:
        """Upper bound of the hyperparameter in vector space."""
        return self._vector_dist.upper_vectorized

    @overload
    def sample_value(
        self,
        size: None = None,
        *,
        seed: np.random.RandomState | None = None,
    ) -> ValueT: ...

    @overload
    def sample_value(
        self,
        size: int,
        *,
        seed: np.random.RandomState | None = None,
    ) -> Array[DType]: ...

    def sample_value(
        self,
        size: int | None = None,
        *,
        seed: np.random.RandomState | None = None,
    ) -> ValueT | Array[DType]:
        """Sample a value from this hyperparameter.

        Args:
            size:
                The number of values to sample. If `None` a single value is
                sampled. Defaults to `None`.
            seed:
                The random state to use for sampling. If `None` the global
                random state is used. Defaults to `None`.

        Returns:
            The sampled value or an array of sampled values, depending on `size=`.
        """
        samples = self.sample_vector(size=size, seed=seed)
        return self.to_value(samples)

    @overload
    def sample_vector(
        self,
        size: None = None,
        *,
        seed: np.random.RandomState | None = None,
    ) -> f64: ...

    @overload
    def sample_vector(
        self,
        size: int,
        *,
        seed: np.random.RandomState | None = None,
    ) -> Array[f64]: ...

    def sample_vector(
        self,
        size: int | None = None,
        *,
        seed: np.random.RandomState | None = None,
    ) -> f64 | Array[f64]:
        """Sample a vectorized value from this hyperparameter.

        Args:
            size:
                The number of values to sample. If `None` a single value is
                sampled. Defaults to `None`.
            seed:
                The random state to use for sampling. If `None` the global
                random state is used. Defaults to `None`.

        Returns:
            The sampled vector or an array of sampled vectors, depending on `size=`.
        """
        if size is None:
            return self._vector_dist.sample_vector(n=1, seed=seed)[0]  # type: ignore
        return self._vector_dist.sample_vector(n=size, seed=seed)

    @overload
    def legal_vector(self, vector: Number) -> bool: ...

    @overload
    def legal_vector(self, vector: Array[f64]) -> Mask: ...

    def legal_vector(self, vector: Number | Array[f64]) -> Mask | bool:
        """Check if a vectorized value is legal for this hyperparameter.

        Args:
            vector:
                The vectorized value to check.

        Returns:
            `True` if the vector is legal, `False` otherwise. If `vector` is an
            array of vectors, a mask of legal values is returned.
        """
        if isinstance(vector, np.ndarray):
            if not np.issubdtype(vector.dtype, np.number):
                raise ValueError(
                    "The vector must be of a numeric dtype to check for legality."
                    f"Got {vector.dtype=} for {vector=}.",
                )
            return self._transformer.legal_vector(vector)

        if not isinstance(vector, (int, float, np.number)):
            return False

        return self._transformer.legal_vector_single(f64(vector))

    # NOTE: @overload, mypy seems to thing this will overlap with below
    @overload
    def legal_value(self, value: ValueT | DType) -> bool: ...  # type: ignore

    @overload
    def legal_value(
        self,
        value: Sequence[ValueT | DType] | Array[Any],
    ) -> Mask: ...

    def legal_value(
        self,
        value: ValueT | DType | Sequence[ValueT | DType] | Array[Any],
    ) -> bool | Mask:
        """Check if a value is legal for this hyperparameter.

        Args:
            value:
                The value to check.

        Returns:
            `True` if the value is legal, `False` otherwise. If `value` is an
            array of values, a mask of legal values is returned.
        """
        if isinstance(value, np.ndarray):
            return self._transformer.legal_value(value)

        if isinstance(value, Sequence) and not isinstance(value, str):
            return self._transformer.legal_value(np.asarray(value))

        return self._transformer.legal_value_single(value)  # type: ignore

    @overload
    def rvs(
        self,
        size: None = None,
        *,
        random_state: np.random.Generator | np.random.RandomState | int | None = None,
    ) -> ValueT: ...

    @overload
    def rvs(
        self,
        size: int,
        *,
        random_state: np.random.Generator | np.random.RandomState | int | None = None,
    ) -> Array[DType]: ...

    def rvs(
        self,
        size: int | None = None,
        *,
        random_state: np.random.Generator | np.random.RandomState | int | None = None,
    ) -> ValueT | Array[DType]:
        """Sample a value from this hyperparameter, compatbile with scipy.stats.rvs.

        Args:
            size:
                The number of values to sample. If `None` a single value is
                sampled. Defaults to `None`.
            random_state:
                The random state to use for sampling. If `None` the global
                random state is used. Defaults to `None`.


        Returns:
            The sampled value or an array of sampled values, depending on `size=`.
        """
        if isinstance(random_state, int) or random_state is None:
            random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.Generator):
            # HACK: This is to enable backwards compatibliity with numpy<=2.0,
            # where the default integer type is np.int32.
            MAX_INT = np.iinfo(np.int32).max
            random_state = np.random.RandomState(int(random_state.integers(0, MAX_INT)))

        vector = self.sample_vector(size=size, seed=random_state)
        return self.to_value(vector)

    @overload
    def to_value(self, vector: Number) -> ValueT: ...

    @overload
    def to_value(self, vector: Array[f64]) -> Array[DType]: ...

    def to_value(
        self,
        vector: Number | Array[f64],
    ) -> ValueT | Array[DType]:
        """Transform a vectorized value to a value in value space.

        Args:
            vector:
                The vectorized value to transform.

        Returns:
            The value in value space.
        """
        if isinstance(vector, np.ndarray):
            return self._transformer.to_value(vector)

        value: DType = self._transformer.to_value(np.array([vector]))[0]
        if self._value_cast is not None:
            return self._value_cast(value)

        return value  # type: ignore

    # NOTE: @overload, mypy seems to complain that this overlaps with below...
    @overload
    def to_vector(  # type: ignore
        self,
        value: Sequence[ValueT | DType] | Array[Any],
    ) -> Array[f64]: ...

    @overload
    def to_vector(self, value: ValueT | DType | Sequence[ValueT | DType]) -> f64: ...

    def to_vector(
        self,
        value: ValueT | DType | Sequence[ValueT | DType] | Array[Any],
    ) -> f64 | Array[f64]:
        """Transform a value to a vectorized value.

        Args:
            value:
                The value to transform.

        Returns:
            The vectorized value.
        """
        if isinstance(value, np.ndarray):
            return self._transformer.to_vector(value)

        if isinstance(value, Sequence) and not isinstance(value, str):
            return self._transformer.to_vector(np.asarray(value))

        return self._transformer.to_vector(np.array([value]))[0]  # type: ignore

    def neighbors_vectorized(
        self,
        vector: Number,
        n: int,
        *,
        std: float | None = None,
        seed: np.random.RandomState | None = None,
    ) -> Array[f64]:
        """Sample neighbors of a vectorized value.

        Args:
            vector:
                The vectorized value to sample neighbors from.
            n:
                The number of **unique** neighbors to sample.

                !!! warning

                    If there are less than `n` legal neighbors, then all legal
                    neighbors are returned, which is some number less than `n`.
            std:
                The standard deviation of the neighborhood. If `None` the
                neighborhood is deterministic. Defaults to `None`.

                !!! warning

                    Hyperparameter subclasses are under no obligation to use
                    this if it does not make sense, i.e. for an
                    [`OrdinalHyperparameter`][ConfigSpace.hyperparameters.OrdinalHyperparameter]
                    or a
                    [`CategoricalHyperparameter`][ConfigSpace.hyperparameters.CategoricalHyperparameter].

            seed:
                The random state to use for sampling. If `None` the global
                random state is used. Defaults to `None`.

        Returns:
            The sampled neighbors in vectorized space.
        """
        if std is not None:
            assert 0.0 <= std <= 1.0, f"std must be in [0, 1], got {std}"

        if not self.legal_vector(vector):
            raise ValueError(
                f"Vector value {vector} is not legal for hyperparameter '{self.name}'."
                f"\n{self}",
            )

        return self._neighborhood(f64(vector), n, std=std, seed=seed)

    def get_max_density(self) -> float:
        """Get the maximum density of the hyperparameter distribution."""
        return self._vector_dist.max_density()

    def neighbors_values(
        self,
        value: ValueT | DType,
        n: int,
        *,
        std: float | None = None,
        seed: np.random.RandomState | None = None,
    ) -> Array[DType]:
        """Sample neighbors of a value.

        Args:
            value:
                The value to sample neighbors from.
            n:
                The number of **unique** neighbors to sample.

                !!! warning

                    If there are less than `n` legal neighbors, then all legal
                    neighbors are returned, which is some number less than `n`.
            std:
                The standard deviation of the neighborhood. If `None` the
                neighborhood is deterministic. Defaults to `None`.

                !!! warning

                    Hyperparameter subclasses are under no obligation to use
                    this if it does not make sense, i.e. for an
                    [`OrdinalHyperparameter`][ConfigSpace.hyperparameters.OrdinalHyperparameter]
                    or a
                    [`CategoricalHyperparameter`][ConfigSpace.hyperparameters.CategoricalHyperparameter].

            seed:
                The random state to use for sampling. If `None` the global
                random state is used. Defaults to `None`.

        Returns:
            The sampled neighbors in value space.
        """
        vector = self.to_vector(value)
        return self.to_value(
            vector=self.neighbors_vectorized(vector, n, std=std, seed=seed),
        )

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        """Get the probability density of an array of vectorized values.

        Args:
            vector:
                The vectorized values to get the probability density of.

        Returns:
            The probability density of the vectorized values. Where vectorized
            values are not legal, the probability density is zero.
        """
        legal_mask: Array[f64] = self.legal_vector(vector).astype(f64)
        return self._vector_dist.pdf_vector(vector) * legal_mask

    def pdf_values(
        self,
        values: Sequence[ValueT | DType] | Array[DType],
    ) -> Array[f64]:
        """Get the probability density of an array of values.

        Args:
            values:
                The values to get the probability density of.

        Returns:
            The probability density of the values. Where values are not legal,
            the probability density is zero.
        """
        if isinstance(values, np.ndarray) and values.ndim != 1:
            raise ValueError("Method pdf expects a one-dimensional numpy array")

        vector = self.to_vector(values)
        return self.pdf_vector(vector)

    def copy(self, **kwargs: Any) -> Self:
        """Create a copy of the hyperparameter with updated attributes.

        Args:
            **kwargs:
                The attributes to update.

        Returns:
            A copy of the hyperparameter with the updated attributes.
        """
        # HACK: Really the only thing implementing Hyperparameter should be a dataclass
        # If a hyperparameter is somehow not a dataclass, it will likely need to
        # overwrite this.
        return replace(self, **kwargs)  # type: ignore

    def get_num_neighbors(
        self,
        value: ValueT | DType | _NotSet = NotSet,
    ) -> int | float:
        """Get the number of neighbors to sample for a given value.

        Args:
            value:
                The value to get the number of neighbors for. If `None` the
                default value is used. Defaults to `None`.

        Returns:
            The number of neighbors to sample.
        """
        return (
            self._neighborhood_size(value)
            if callable(self._neighborhood_size)
            else self._neighborhood_size
        )

    # ------------- Deprecations
    @deprecated("Please use `get_num_neighbors() > 0` or `hp.size > 1` instead.")
    def has_neighbors(self) -> bool:
        """Deprecated."""
        return self.get_num_neighbors() > 0

    @deprecated("Please use `sample_value(seed=rs)` instead.")
    def sample(self, rs: np.random.RandomState) -> ValueT:
        """Deprecated."""
        return self.sample_value(seed=rs)

    @deprecated("Please use `sample_vector(size, seed=rs)` instead.")
    def _sample(
        self,
        rs: np.random.RandomState,
        size: int | None = None,
    ) -> Array[f64]:
        if size is None:
            warnings.warn(
                "Private method is deprecated, please use"
                "`sample_vector(size=1, seed=rs)` for old behaviour."
                " This will be removed in the future.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.sample_vector(size=1, seed=rs)

        return self.sample_vector(size=size, seed=rs)

    @deprecated("Please use `pdf_values(value)` instead.")
    def pdf(
        self,
        vector: DType | Array[DType],  # NOTE: New convention this should be value
    ) -> f64 | Array[f64]:
        """Deprecated."""
        if isinstance(vector, np.ndarray):
            return self.pdf_values(vector)

        return self.pdf_values(np.asarray([vector]))[0]  # type: ignore

    @deprecated("Please use `pdf_vector(vector)` instead.")
    def _pdf(
        self,
        vector: f64 | Array[f64],
    ) -> f64 | Array[f64]:
        if isinstance(vector, np.ndarray):
            return self.pdf_vector(vector)

        return self.pdf_vector(np.asarray([vector]))[0]  # type: ignore

    @deprecated("Please use `.size` attribute instead.")
    def get_size(self) -> int | float:
        """Deprecated."""
        return self.size

    @deprecated("Please use `legal_value(value)` instead")
    def is_legal(self, value: DType) -> bool:
        """Deprecated."""
        return self.legal_value(value)

    @deprecated("Please use `legal_vector(vector)` instead.")
    def is_legal_vector(self, value: f64) -> bool:
        """Deprecated."""
        return self.legal_vector(value)

    @property
    @deprecated("Please use `.upper_vectorized` instead.")
    def _upper(self) -> f64:
        return self.upper_vectorized

    @property
    @deprecated("Please use `.lower_vectorized` instead.")
    def _lower(self) -> f64:
        return self.lower_vectorized

    @deprecated("Please use `neighbors_vectorized`  instead.")
    def get_neighbors(
        self,
        value: f64,
        rs: np.random.RandomState,
        number: int | None = None,
        std: float | None = None,
        transform: bool = False,
    ) -> Array[f64]:
        """Deprecated."""
        if transform is True:
            raise RuntimeError(
                "Previous `get_neighbors` with `transform=True` had different"
                " behaviour depending on the hyperparameter. Notably numerics"
                " were still considered to be in vectorized form while for ordinals"
                " they were considered to be in value form."
                "\nPlease use either `neighbors_vectorized` or `neighbors_values`"
                " instead, depending on your need. You can use `to_value` or"
                " `to_vector` to switch between the results of the two.",
            )

        if number is None:
            warnings.warn(
                "Please provide a number of neighbors to sample. The"
                " default used to be `4` but will be explicitly required"
                " in the futurefuture.",
                DeprecationWarning,
                stacklevel=2,
            )
            number = 4

        return self.neighbors_vectorized(value, number, std=std, seed=rs)

    @deprecated("Please use `hp.to_value(v)` instead.")
    def _transform_scalar(self, value: f64) -> ValueT:
        """Deprecated."""
        return self.to_value(value)

    @deprecated("Please use `hp.to_value(vector)` instead.")
    def _transform_vector(self, vector: Array[f64]) -> Array[DType]:
        """Deprecated."""
        return self.to_value(vector)

    @deprecated("Please use `hp.to_vector(value)` instead.")
    def _inverse_transform(
        self,
        value: ValueT | DType | Array[DType],
    ) -> f64 | Array[f64]:
        return self.to_vector(value)

    @deprecated("Please use `hp.to_value(vector)` instead.")
    def _transform(
        self,
        vector: f64 | Array[f64],
    ) -> ValueT | Array[DType]:
        return self.to_value(vector)


NumberT = TypeVar("NumberT", int, float)
"""Some number type that represents a single value in value space
for a numerical hyperparameter.
"""


@dataclass(init=False)
class NumericalHyperparameter(Hyperparameter[NumberT, DType]):
    """Base class for numerical hyperparameters in the configuration space.

    Should likely not be used directly and instead inherit from
    [`IntegerHyperparameter`][ConfigSpace.hyperparameters.IntegerHyperparameter]
    or
    [`FloatHyperparameter`][ConfigSpace.hyperparameters.FloatHyperparameter].
    """

    LEGAL_VALUE_TYPES: ClassVar[tuple[type, ...]] = (int, float, np.number)

    lower: NumberT
    """Lower bound of the hyperparameter in value space."""
    upper: NumberT
    """Upper bound of the hyperparameter in value space."""
    log: bool
    """If `True` the hyperparameter is sampled on a logarithmic scale."""

    @abstractmethod
    def to_uniform(
        self,
    ) -> UniformFloatHyperparameter | UniformIntegerHyperparameter:
        """Convert the hyperparameter to its uniform equivalent."""
        ...


@dataclass(init=False)
class IntegerHyperparameter(NumericalHyperparameter[int, i64]):
    """Base class for integer hyperparameters in the configuration space."""

    def _integer_neighborhood_size(self, value: int | i64 | _NotSet) -> int:
        if value is NotSet:
            return int(self.size)

        if self.lower <= value <= self.upper:  # type: ignore
            return int(self.size) - 1

        return int(self.size)

    def to_uniform(self) -> UniformIntegerHyperparameter:
        """Convert the hyperparameter to its uniform equivalent.

        This will remove any distribution associated with it's vectorized
        representation.
        """
        from ConfigSpace.hyperparameters.uniform_integer import (
            UniformIntegerHyperparameter,
        )

        return UniformIntegerHyperparameter(
            name=self.name,
            lower=self.lower,
            upper=self.upper,
            default_value=self.default_value,
            log=self.log,
            meta=self.meta,
        )


@dataclass(init=False)
class FloatHyperparameter(NumericalHyperparameter[float, f64]):
    """Base class for float hyperparameters in the configuration space."""

    def to_uniform(self) -> UniformFloatHyperparameter:
        """Convert the hyperparameter to its uniform equivalent.

        This will remove any distribution associated with it's vectorized
        representation.
        """
        from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter

        return UniformFloatHyperparameter(
            name=self.name,
            lower=self.lower,
            upper=self.upper,
            default_value=self.default_value,
            log=self.log,
            meta=self.meta,
        )
