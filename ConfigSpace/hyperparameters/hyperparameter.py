from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

from ConfigSpace.hyperparameters._hp_components import (
    DType,
    VDType,
    _Neighborhood,
    _Transformer,
)

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters._distributions import VectorDistribution


class Comparison(str, Enum):
    LESS_THAN = "less"
    GREATER_THAN = "greater"
    EQUAL = "equal"
    UNEQUAL = "unequal"


@dataclass(init=False)
class Hyperparameter(Generic[DType, VDType]):
    orderable: ClassVar[bool] = False

    name: str = field(hash=True)
    default_value: DType = field(hash=True)
    vector_dist: VectorDistribution[VDType] = field(hash=True)
    meta: Mapping[Hashable, Any] | None = field(hash=True)

    size: int | float = field(hash=True, repr=False)

    normalized_default_value: VDType = field(hash=True, init=False, repr=False)

    _legal_vector: Callable[[VDType], bool] = field(hash=True)
    _transformer: _Transformer[DType, VDType] = field(hash=True)
    _neighborhood: _Neighborhood[VDType] = field(hash=True)
    _neighborhood_size: Callable[[DType | None], int | float] | float | int = field(
        repr=False,
    )

    def __init__(
        self,
        name: str,
        *,
        default_value: DType,
        size: int | float,
        vector_dist: VectorDistribution[VDType],
        transformer: _Transformer[DType, VDType],
        neighborhood: _Neighborhood[VDType],
        neighborhood_size: Callable[[DType | None], int] | int | float = np.inf,
        meta: Mapping[Hashable, Any] | None = None,
    ):
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
                f"Default value {self.default_value} is not within the legal range.",
            )

        self.normalized_default_value = self.to_vector(default_value)

    @property
    def lower_vectorized(self) -> VDType:
        return self.vector_dist.lower

    @property
    def upper_vectorized(self) -> VDType:
        return self.vector_dist.upper

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
    ) -> VDType: ...

    @overload
    def sample_vector(
        self,
        size: int,
        *,
        seed: np.random.RandomState | None = None,
    ) -> npt.NDArray[VDType]: ...

    def sample_vector(
        self,
        size: int | None = None,
        *,
        seed: np.random.RandomState | None = None,
    ) -> VDType | npt.NDArray[VDType]:
        if size is None:
            return self.vector_dist.sample(n=1, seed=seed)[0]
        return self.vector_dist.sample(n=size, seed=seed)

    def legal_vector(self, vector: VDType) -> bool:
        return self.vector_dist.in_support(vector)

    def legal_value(self, value: DType) -> bool:
        vector = self.to_vector(value)
        return self.legal_vector(vector)

    def rvs(self, random_state: np.random.RandomState) -> DType:
        vector = self.sample_vector(seed=random_state)
        return self.to_value(vector)

    @overload
    def to_value(self, vector: VDType) -> DType: ...

    @overload
    def to_value(self, vector: npt.NDArray[VDType]) -> npt.NDArray[DType]: ...

    def to_value(
        self,
        vector: VDType | npt.NDArray[VDType],
    ) -> DType | npt.NDArray[DType]:
        match vector:
            case np.ndarray():
                return self._transformer.to_value(vector)
            case _:
                return self._transformer.to_value(np.array([vector]))[0]

    @overload
    def to_vector(self, value: DType) -> VDType: ...

    @overload
    def to_vector(
        self,
        value: Sequence[DType] | npt.NDArray,
    ) -> npt.NDArray[VDType]: ...

    def to_vector(
        self,
        value: DType | Sequence[DType] | npt.NDArray,
    ) -> VDType | npt.NDArray[VDType]:
        match value:
            case np.ndarray():
                return self._transformer.to_vector(value)
            case Sequence():
                return self._transformer.to_vector(value)
            case _:
                return self._transformer.to_vector(np.array([value]))[0]

    def neighbors_vectorized(
        self,
        vector: VDType,
        n: int,
        *,
        std: float | None = None,
        seed: np.random.RandomState | None = None,
    ) -> npt.NDArray[VDType]:
        if std is not None:
            assert 0.0 <= std <= 1.0, f"std must be in [0, 1], got {std}"

        return self._neighborhood(vector, n, std=std, seed=seed)

    def neighbors_values(
        self,
        value: DType,
        n: int,
        *,
        std: float,
        seed: np.random.RandomState | None = None,
    ) -> npt.NDArray[DType]:
        vector = self.to_vector(value)
        return self.to_value(
            vector=self.neighbors_vectorized(vector, n, std=std, seed=seed),
        )

    def get_neighbors(
        self,
        value: VDType,
        rs: np.random.RandomState,
        number: int | None = None,
        std: float | None = None,
        transform: bool = False,
    ) -> npt.NDArray:
        warnings.warn(
            "Please use"
            "`neighbors_vectorized(value=value, n=number, seed=rs, std=str)`"
            " instead. This is deprecated and will be removed in the future."
            " If you need `transform=True`, please apply `to_value` to the result.",
            DeprecationWarning,
            stacklevel=2,
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

        neighbors = self.neighbors_vectorized(value, number, std=std, seed=rs)
        if transform:
            return self.to_value(neighbors)
        return neighbors

    def vector_pdf(self, vector: npt.NDArray[VDType]) -> npt.NDArray[np.float64]:
        match self.vector_dist:
            case rv_continuous_frozen():
                return self.vector_dist.pdf(vector)
            case rv_discrete_frozen():
                return self.vector_dist.pmf(vector)
            case _:
                raise NotImplementedError(
                    "Only continuous and discrete distributions are supported."
                    f"Got {self.vector_dist}",
                )

    def value_pdf(
        self,
        values: Sequence[DType] | npt.NDArray,
    ) -> npt.NDArray[np.float64]:
        vector = self.to_vector(values)
        return self.vector_pdf(vector)

    def copy(self, **kwargs: Any) -> Self:
        return replace(self, **kwargs)

    def get_size(self) -> int | float:
        warnings.warn(
            "Please just use the `.size` attribute directly",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.size

    def compare_value(
        self,
        value: DType,
        other_value: DType,
    ) -> Comparison:
        vector = self.to_vector(value)
        other_vector = self.to_vector(other_value)
        return self.compare_vector(vector, other_vector)

    def compare_vector(
        self,
        vector: VDType,
        other_vector: VDType,
    ) -> Comparison:
        if vector == other_vector:
            return Comparison.EQUAL

        if not self.orderable:
            return Comparison.UNEQUAL

        if vector < other_vector:
            return Comparison.LESS_THAN
        return Comparison.GREATER_THAN

    def get_num_neighbors(self, value: DType | None = None) -> int | float:
        return (
            self._neighborhood_size(value)
            if callable(self._neighborhood_size)
            else self._neighborhood_size
        )

    def has_neighbors(self) -> bool:
        warnings.warn(
            "Please use `get_num_neighbors() > 0` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_neighbors() > 0


HPType = TypeVar("HPType", bound=Hyperparameter)


@runtime_checkable
class HyperparameterWithPrior(Protocol[HPType]):
    def to_uniform(self) -> HPType: ...
