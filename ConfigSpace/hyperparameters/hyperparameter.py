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
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)
from typing_extensions import Self, deprecated

import numpy as np
import numpy.typing as npt

from ConfigSpace.hyperparameters._hp_components import (
    DType,
    _Neighborhood,
    _Transformer,
)

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters._distributions import Distribution


@dataclass(init=False)
class Hyperparameter(ABC, Generic[DType]):
    serializable_type_name: ClassVar[str]
    orderable: ClassVar[bool] = False
    legal_value_types: ClassVar[tuple[type, ...] | Literal["all"]] = "all"

    name: str = field(hash=True)
    default_value: DType = field(hash=True)
    meta: Mapping[Hashable, Any] | None = field(hash=True)

    size: int | float = field(hash=True, repr=False)

    vector_dist: Distribution = field(hash=True, compare=False)
    normalized_default_value: np.float64 = field(hash=True, init=False, repr=False)

    _transformer: _Transformer[DType] = field(hash=True, compare=False)
    _neighborhood: _Neighborhood = field(hash=True, compare=False)
    _neighborhood_size: Callable[[DType | None], int | float] | float | int = field(
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        name: str,
        *,
        default_value: DType,
        size: int | float,
        vector_dist: Distribution,
        transformer: _Transformer[DType],
        neighborhood: _Neighborhood,
        neighborhood_size: Callable[[DType | None], int] | int | float = np.inf,
        meta: Mapping[Hashable, Any] | None = None,
    ):
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {name} ({type(name)})")

        self.name = name
        self.default_value = default_value
        self.vector_dist = vector_dist
        self.meta = meta if meta is not None else {}

        self.size = size

        self._transformer = transformer
        self._neighborhood = neighborhood
        self._neighborhood_size = neighborhood_size

        if not self.legal_value(self.default_value):
            raise ValueError(
                f"Illegal default value {self.default_value} for"
                f" hyperparamter '{self.name}'.\n{self}",
            )

        self.normalized_default_value = self.to_vector(default_value)

    @property
    def lower_vectorized(self) -> np.float64:
        return self.vector_dist.lower_vectorized

    @property
    def upper_vectorized(self) -> np.float64:
        return self.vector_dist.upper_vectorized

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
    ) -> npt.NDArray[DType]: ...

    def sample_value(
        self,
        size: int | None = None,
        *,
        seed: np.random.RandomState | None = None,
    ) -> DType | npt.NDArray[DType]:
        """Sample a value from this hyperparameter."""
        samples = self.sample_vector(size=size, seed=seed)
        return self.to_value(samples)

    @overload
    def sample_vector(
        self,
        size: None = None,
        *,
        seed: np.random.RandomState | None = None,
    ) -> np.float64: ...

    @overload
    def sample_vector(
        self,
        size: int,
        *,
        seed: np.random.RandomState | None = None,
    ) -> npt.NDArray[np.float64]: ...

    def sample_vector(
        self,
        size: int | None = None,
        *,
        seed: np.random.RandomState | None = None,
    ) -> np.float64 | npt.NDArray[np.float64]:
        if size is None:
            return self.vector_dist.sample_vector(n=1, seed=seed)[0]
        return self.vector_dist.sample_vector(n=size, seed=seed)

    @overload
    def legal_vector(self, vector: np.float64) -> bool: ...

    @overload
    def legal_vector(
        self,
        vector: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.bool_]: ...

    def legal_vector(
        self,
        vector: np.float64 | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.bool_] | bool:
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
        value: Sequence[DType] | npt.NDArray[DType],
    ) -> npt.NDArray[np.bool_]: ...

    def legal_value(
        self,
        value: DType | Sequence[DType] | npt.NDArray[DType],
    ) -> bool | npt.NDArray[np.bool_]:
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
    ) -> npt.NDArray[DType]: ...

    def rvs(
        self,
        size: int | None = None,
        *,
        random_state: np.random.Generator | np.random.RandomState | int | None = None,
    ) -> DType | npt.NDArray[DType]:
        if isinstance(random_state, int) or random_state is None:
            random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.Generator):
            MAX_INT = np.iinfo(np.int32).max
            random_state = np.random.RandomState(int(random_state.integers(0, MAX_INT)))

        vector = self.sample_vector(size=size, seed=random_state)
        return self.to_value(vector)

    @overload
    def to_value(self, vector: np.float64) -> DType: ...

    @overload
    def to_value(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[DType]: ...

    def to_value(
        self,
        vector: np.float64 | npt.NDArray[np.float64],
    ) -> DType | npt.NDArray[DType]:
        if isinstance(vector, np.ndarray):
            return self._transformer.to_value(vector)

        return self._transformer.to_value(np.array([vector]))[0]

    @overload
    def to_vector(self, value: DType | int | float) -> np.float64: ...

    @overload
    def to_vector(
        self,
        value: Sequence[int | float | DType] | npt.NDArray,
    ) -> npt.NDArray[np.float64]: ...

    def to_vector(
        self,
        value: DType | int | float | Sequence[DType | int | float] | npt.NDArray,
    ) -> np.float64 | npt.NDArray[np.float64]:
        if isinstance(value, np.ndarray):
            return self._transformer.to_vector(value)

        if isinstance(value, Sequence) and not isinstance(value, str):
            return self._transformer.to_vector(np.asarray(value))

        return self._transformer.to_vector(np.array([value]))[0]

    def neighbors_vectorized(
        self,
        vector: np.float64,
        n: int,
        *,
        std: float | None = None,
        seed: np.random.RandomState | None = None,
    ) -> npt.NDArray[np.float64]:
        if std is not None:
            assert 0.0 <= std <= 1.0, f"std must be in [0, 1], got {std}"

        if not self.legal_vector(vector):
            raise ValueError(
                f"Vector value {vector} is not legal for hyperparameter '{self.name}'."
                f"\n{self}",
            )

        return self._neighborhood(vector, n, std=std, seed=seed)

    def get_max_density(self) -> float:
        return self.vector_dist.max_density()

    def neighbors_values(
        self,
        value: DType,
        n: int,
        *,
        std: float | None = None,
        seed: np.random.RandomState | None = None,
    ) -> npt.NDArray[DType]:
        vector = self.to_vector(value)
        return self.to_value(
            vector=self.neighbors_vectorized(vector, n, std=std, seed=seed),
        )

    def pdf_vector(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        legal_mask = self.legal_vector(vector).astype(np.float64)
        return self.vector_dist.pdf_vector(vector) * legal_mask

    def pdf_values(
        self,
        values: Sequence[DType] | npt.NDArray,
    ) -> npt.NDArray[np.float64]:
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
        return replace(self, **kwargs)

    def get_num_neighbors(self, value: DType | None = None) -> int | float:
        return (
            self._neighborhood_size(value)
            if callable(self._neighborhood_size)
            else self._neighborhood_size
        )

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]) -> Self:
        if "q" in data:
            warnings.warn(
                "The key 'q' for quantized hyperparameters is deprecated and will"
                " and will be removed in the future. Ignoring for now.",
                DeprecationWarning,
                stacklevel=2,
            )
            data.pop("q")

        data.pop("type", None)

        # Legacy
        if "default" in data:
            data["default_value"] = data.pop("default")

        return cls(**data)

    # ------------- Deprecations
    @deprecated("Please use `get_num_neighbors() > 0` or `hp.size > 1` instead.")
    def has_neighbors(self) -> bool:
        return self.get_num_neighbors() > 0

    @deprecated("Please use `to_vector(value)` instead.")
    def _inverse_transform(
        self,
        value: DType | npt.NDArray[DType],
    ) -> np.float64 | npt.NDArray[np.float64]:
        return self.to_vector(value)

    @deprecated("Please use `sample_value(seed=rs)` instead.")
    def sample(self, rs: np.random.RandomState) -> DType:
        return self.sample_value(seed=rs)

    @deprecated("Please use `sample_vector(size, seed=rs)` instead.")
    def _sample(
        self,
        rs: np.random.RandomState,
        size: int | None = None,
    ) -> npt.NDArray[np.float64]:
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
        vector: DType | npt.NDArray[DType],  # NOTE: New convention this should be value
    ) -> np.float64 | npt.NDArray[np.float64]:
        if isinstance(vector, np.ndarray):
            return self.pdf_values(vector)

        return self.pdf_values(np.asarray([vector]))[0]

    @deprecated("Please use `pdf_vector(vector)` instead.")
    def _pdf(
        self,
        vector: np.float64 | npt.NDArray[np.float64],
    ) -> np.float64 | npt.NDArray[np.float64]:
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
    def is_legal_vector(self, value: np.float64) -> bool:
        return self.legal_vector(value)

    @deprecated("Please use `to_value(vector)` instead.")
    def _transform(
        self,
        vector: np.float64 | npt.NDArray[np.float64],
    ) -> DType | npt.NDArray[DType]:
        return self.to_value(vector)

    @property
    @deprecated("Please use `.upper_vectorized` instead.")
    def _upper(self) -> np.float64:
        return self.upper_vectorized

    @property
    @deprecated("Please use `.lower_vectorized` instead.")
    def _lower(self) -> np.float64:
        return self.lower_vectorized

    @deprecated("Please use `neighbors_vectorized`  instead.")
    def get_neighbors(
        self,
        value: np.float64,
        rs: np.random.RandomState,
        number: int | None = None,
        std: float | None = None,
        transform: bool = False,
    ) -> npt.NDArray:
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


HPType = TypeVar("HPType", bound=Hyperparameter)


@runtime_checkable
class HyperparameterWithPrior(Protocol[HPType]):
    def to_uniform(self) -> HPType: ...


@dataclass(init=False)
class NumericalHyperparameter(Hyperparameter[DType]):
    legal_value_types: ClassVar[tuple[type, ...]] = (int, float, np.number)

    lower: DType = field(hash=True)
    upper: DType = field(hash=True)
    log: bool = field(hash=True)


@dataclass(init=False)
class IntegerHyperparameter(NumericalHyperparameter[np.int64]):
    def _neighborhood_size(self, value: np.int64 | None) -> int:
        if value is None:
            return int(self.size)

        if self.lower <= value <= self.upper:
            return int(self.size) - 1

        return int(self.size)


@dataclass(init=False)
class FloatHyperparameter(NumericalHyperparameter[np.float64]):
    pass
