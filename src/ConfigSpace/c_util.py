from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
from typing_extensions import deprecated

from ConfigSpace.util import change_hp_value, check_configuration

if TYPE_CHECKING:
    from ConfigSpace.forbidden import ForbiddenLike
    from ConfigSpace.types import Array, f64


@deprecated(
    "Please use `configuration.check_valid_configuration()`"
    " or `space.check_configuration_vector_representation(configuration)` instead.",
)
def check_forbidden(
    forbidden_clauses: Iterable[ForbiddenLike],
    vector: Array[f64],
) -> bool:
    return any(clause.is_forbidden_vector(vector) for clause in forbidden_clauses)


# Backwards compatibility
__all__ = ["check_configuration", "change_hp_value"]
