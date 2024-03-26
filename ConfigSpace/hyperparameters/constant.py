from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ConfigSpace.hyperparameters._distributions import ConstantVectorDistribution
from ConfigSpace.hyperparameters._hp_components import TransformerConstant
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter

CONSTANT_VECTOR_VALUE_YES = np.float64(1)
CONSTANT_VECTOR_VALUE_NO = np.float64(0)


def _empty_neighborhood(*_: Any, **__: Any) -> npt.NDArray[np.float64]:
    return np.ndarray([], dtype=np.float64)


class Constant(Hyperparameter[Any]):
    serializable_type_name: ClassVar[str] = "constant"
    orderable: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        value: Any,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """Representing a constant hyperparameter in the configuration space.

        By sampling from the configuration space each time only a single,
        constant ``value`` will be drawn from this hyperparameter.

        Args:
            name:
                Name of the hyperparameter, with which it can be accessed
            value:
                value to sample hyperparameter from
            meta:
                Field for holding meta data provided by the user.
                Not used by the configuration space.
        """
        # TODO: This should be changed and allowed...
        if not isinstance(value, (int, float, str)) or isinstance(value, bool):
            raise TypeError(
                f"Constant hyperparameter '{name}' must be of type int, float or str, "
                f"but got {type(value).__name__}.",
            )

        self.value = value

        super().__init__(
            name=name,
            default_value=value,
            size=1,
            meta=meta,
            transformer=TransformerConstant(
                value=value,
                vector_value_yes=CONSTANT_VECTOR_VALUE_YES,
                vector_value_no=CONSTANT_VECTOR_VALUE_NO,
            ),
            vector_dist=ConstantVectorDistribution(
                vector_value=CONSTANT_VECTOR_VALUE_YES,
            ),
            neighborhood=_empty_neighborhood,
            neighborhood_size=0,
        )

    def __str__(self) -> str:
        parts = [
            self.name,
            f"Type: {str(self.__class__.__name__).replace('Hyperparameter', '')}",
            f"Value: {self.value}",
        ]

        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.serializable_type_name,
            "value": self.value,
        }


UnParametrizedHyperparameter = Constant
