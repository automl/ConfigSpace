from __future__ import annotations

import warnings
from abc import ABC
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    overload,
)
from typing_extensions import Self, deprecated

import numpy as np

from ConfigSpace.types import DType, f64, i64

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters._distributions import Distribution
    from ConfigSpace.hyperparameters._hp_components import _Neighborhood, _Transformer
    from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter
    from ConfigSpace.hyperparameters.uniform_integer import UniformIntegerHyperparameter
    from ConfigSpace.types import Array, Mask


@dataclass(init=False)
class Hyperparameter(ABC, Generic[DType]):
    ORDERABLE: ClassVar[bool] = False
    LEGAL_VALUE_TYPES: ClassVar[tuple[type, ...] | Literal["all"]] = "all"

    name: str
    default_value: DType
    meta: Mapping[Hashable, Any] | None
    size: int | float

    _vector_dist: Distribution = field(repr=False)
    _normalized_default_value: f64 = field(repr=False)
    _transformer: _Transformer[DType] = field(repr=False)
    _neighborhood: _Neighborhood = field(repr=False, compare=False)
    _neighborhood_size: float | int | Callable[[DType | None], int | float] = field(
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        name: str,
        default_value: DType,
        vector_dist: Distribution,
        transformer: _Transformer[DType],
        neighborhood: _Neighborhood,
        size: int | float,
        neighborhood_size: float | int | Callable[[DType | None], int | float],
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
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
        self._neighborhood_size = neighborhood_size

        if not self.legal_value(self.default_value):
            raise ValueError(
                f"Illegal default value {self.default_value} for"
                f" hyperparamter '{self.name}'.",
            )

        self._normalized_default_value = self.to_vector(self.default_value)

    @property
    def lower_vectorized(self) -> f64:
        return self._vector_dist.lower_vectorized

    @property
    def upper_vectorized(self) -> f64:
        return self._vector_dist.upper_vectorized

    @overload
    def sample_value(
        self,
        size: None = None,
        *,
        seed: np.random.RandomState | None = None,
    ) -> DType: ...

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
    ) -> DType | Array[DType]:
        """Sample a value from this hyperparameter."""
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
        if size is None:
            return self._vector_dist.sample_vector(n=1, seed=seed)[0]
        return self._vector_dist.sample_vector(n=size, seed=seed)

    @overload
    def legal_vector(self, vector: f64) -> bool: ...

    @overload
    def legal_vector(self, vector: Array[f64]) -> Mask: ...

    def legal_vector(self, vector: f64 | Array[f64]) -> Mask | bool:
        if isinstance(vector, np.ndarray):
            if not np.issubdtype(vector.dtype, np.number):
                raise ValueError(
                    "The vector must be of a numeric dtype to check for legality."
                    f"Got {vector.dtype=} for {vector=}.",
                )
            return self._transformer.legal_vector(vector)

        if not isinstance(vector, (int, float, np.number)):
            return False

        return self._transformer.legal_vector_single(vector)

    @overload
    def legal_value(self, value: DType) -> bool: ...

    @overload
    def legal_value(
        self,
        value: Sequence[DType] | Array[Any],
    ) -> Mask: ...

    def legal_value(
        self,
        value: DType | Sequence[DType] | Array[Any],
    ) -> bool | Mask:
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
    ) -> DType: ...

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
    ) -> DType | Array[DType]:
        if isinstance(random_state, int) or random_state is None:
            random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.Generator):
            MAX_INT = np.iinfo(np.int32).max
            random_state = np.random.RandomState(int(random_state.integers(0, MAX_INT)))

        vector = self.sample_vector(size=size, seed=random_state)
        return self.to_value(vector)

    @overload
    def to_value(self, vector: f64) -> DType: ...

    @overload
    def to_value(self, vector: Array[f64]) -> Array[DType]: ...

    def to_value(
        self,
        vector: f64 | Array[f64],
    ) -> DType | Array[DType]:
        if isinstance(vector, np.ndarray):
            return self._transformer.to_value(vector)

        return self._transformer.to_value(np.array([vector]))[0]

    @overload
    def to_vector(self, value: DType | int | float) -> f64: ...

    @overload
    def to_vector(
        self,
        value: Sequence[int | float | DType] | Array,
    ) -> Array[f64]: ...

    def to_vector(
        self,
        value: DType | int | float | Sequence[DType | int | float] | Array,
    ) -> f64 | Array[f64]:
        if isinstance(value, np.ndarray):
            return self._transformer.to_vector(value)

        if isinstance(value, Sequence) and not isinstance(value, str):
            return self._transformer.to_vector(np.asarray(value))

        return self._transformer.to_vector(np.array([value]))[0]

    def neighbors_vectorized(
        self,
        vector: f64,
        n: int,
        *,
        std: float | None = None,
        seed: np.random.RandomState | None = None,
    ) -> Array[f64]:
        if std is not None:
            assert 0.0 <= std <= 1.0, f"std must be in [0, 1], got {std}"

        if not self.legal_vector(vector):
            raise ValueError(
                f"Vector value {vector} is not legal for hyperparameter '{self.name}'."
                f"\n{self}",
            )

        return self._neighborhood(vector, n, std=std, seed=seed)

    def get_max_density(self) -> float:
        return self._vector_dist.max_density()

    def neighbors_values(
        self,
        value: DType,
        n: int,
        *,
        std: float | None = None,
        seed: np.random.RandomState | None = None,
    ) -> Array[DType]:
        vector = self.to_vector(value)
        return self.to_value(
            vector=self.neighbors_vectorized(vector, n, std=std, seed=seed),
        )

    def pdf_vector(self, vector: Array[f64]) -> Array[f64]:
        legal_mask = self.legal_vector(vector).astype(f64)
        return self._vector_dist.pdf_vector(vector) * legal_mask

    def pdf_values(
        self,
        values: Sequence[DType] | Array[DType],
    ) -> Array[f64]:
        # TODO: why this restriction?
        _values = np.asarray(values)
        if _values.ndim != 1:
            raise ValueError(
                "Method pdf expects a one-dimensional numpy array but got"
                f" {_values.ndim} dimensions."
                f"\n{_values}",
            )
        vector = self.to_vector(_values)
        return self.pdf_vector(vector)

    def copy(self, **kwargs: Any) -> Self:
        # HACK: Really the only thing implementing Hyperparameter should be a dataclass
        return replace(self, **kwargs)  # type: ignore

    def get_num_neighbors(self, value: DType | None = None) -> int | float:
        return (
            self._neighborhood_size(value)
            if callable(self._neighborhood_size)
            else self._neighborhood_size
        )

    # ------------- Deprecations
    @deprecated("Please use `get_num_neighbors() > 0` or `hp.size > 1` instead.")
    def has_neighbors(self) -> bool:
        return self.get_num_neighbors() > 0

    @deprecated("Please use `to_vector(value)` instead.")
    def _inverse_transform(
        self,
        value: DType | Array[DType],
    ) -> f64 | Array[f64]:
        return self.to_vector(value)

    @deprecated("Please use `sample_value(seed=rs)` instead.")
    def sample(self, rs: np.random.RandomState) -> DType:
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

    @deprecated("Please use `pdf_value(value)` instead.")
    def pdf(
        self,
        vector: DType | Array[DType],  # NOTE: New convention this should be value
    ) -> f64 | Array[f64]:
        if isinstance(vector, np.ndarray):
            return self.pdf_values(vector)

        return self.pdf_values(np.asarray([vector]))[0]

    @deprecated("Please use `pdf_vector(vector)` instead.")
    def _pdf(
        self,
        vector: f64 | Array[f64],
    ) -> f64 | Array[f64]:
        if isinstance(vector, np.ndarray):
            return self.pdf_vector(vector)

        return self.pdf_vector(np.asarray([vector]))[0]

    @deprecated("Please use `.size` attribute instead.")
    def get_size(self) -> int | float:
        return self.size

    @deprecated("Please use `legal_value(value)` instead")
    def is_legal(self, value: DType) -> bool:
        return self.legal_value(value)

    @deprecated("Please use `legal_vector(vector)` instead.")
    def is_legal_vector(self, value: f64) -> bool:
        return self.legal_vector(value)

    @deprecated("Please use `to_value(vector)` instead.")
    def _transform(
        self,
        vector: f64 | Array[f64],
    ) -> DType | Array[DType]:
        return self.to_value(vector)

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


@dataclass(init=False)
class NumericalHyperparameter(Hyperparameter[DType]):
    LEGAL_VALUE_TYPES: ClassVar[tuple[type, ...]] = (int, float, np.number)

    lower: DType
    upper: DType
    log: bool

    def to_uniform(
        self,
    ) -> UniformFloatHyperparameter | UniformIntegerHyperparameter: ...


@dataclass(init=False)
class IntegerHyperparameter(NumericalHyperparameter[i64]):
    def _neighborhood_size(self, value: i64 | None) -> int:
        if value is None:
            return int(self.size)

        if self.lower <= value <= self.upper:
            return int(self.size) - 1

        return int(self.size)

    def to_uniform(self) -> UniformIntegerHyperparameter:
        from ConfigSpace.hyperparameters.uniform_float import (
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
class FloatHyperparameter(NumericalHyperparameter[f64]):
    def to_uniform(self) -> UniformFloatHyperparameter:
        from ConfigSpace.hyperparameters.uniform_float import UniformFloatHyperparameter

        return UniformFloatHyperparameter(
            name=self.name,
            lower=self.lower,
            upper=self.upper,
            default_value=self.default_value,
            log=self.log,
            meta=self.meta,
        )
