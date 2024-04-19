# Copyright (c) 2014-2016, ConfigSpace developers
# Matthias Feurer
# Katharina Eggensperger
# and others (see commit history).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of itConfigurationSpaces contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Mapping, Union
from typing_extensions import Self, deprecated

import numpy as np
from more_itertools import unique_everseen

from ConfigSpace.hyperparameters import Hyperparameter
from ConfigSpace.types import Array, Mask, f64

if TYPE_CHECKING:
    from ConfigSpace.types import Array

_SENTINEL = object()


class ForbiddenClause(ABC):
    def __init__(self, hyperparameter: Hyperparameter) -> None:
        self.hyperparameter = hyperparameter
        self.vector_id: np.intp | None = None

    def set_vector_idx(self, hyperparameter_to_idx: Mapping[str, Any]) -> None:
        self.vector_id = np.intp(hyperparameter_to_idx[self.hyperparameter.name])

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        pass

    @abstractmethod
    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        pass

    @abstractmethod
    def __copy__(self) -> Self:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


class ForbiddenRelation(ABC):
    _RELATION_STR: ClassVar[str]

    def __init__(self, left: Hyperparameter, right: Hyperparameter):
        if not isinstance(left, Hyperparameter):
            raise TypeError("Argument 'left' is not of type %s." % Hyperparameter)
        if not isinstance(right, Hyperparameter):
            raise TypeError("Argument 'right' is not of type %s." % Hyperparameter)

        self.left = left
        self.right = right
        self.vector_ids: tuple[None, None] | tuple[np.intp, np.intp] = (None, None)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.left == other.left and self.right == other.right

    def __copy__(self) -> Self:
        return self.__class__(left=copy(self.left), right=copy(self.right))

    def set_vector_idx(self, hyperparameter_to_idx: Mapping[str, int]) -> None:
        self.vector_ids = (
            np.intp(hyperparameter_to_idx[self.left.name]),
            np.intp(hyperparameter_to_idx[self.right.name]),
        )

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        pass

    @abstractmethod
    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.name,
            "right": self.right.name,
            "type": "RELATION",
            "lambda": self._RELATION_STR,
        }


class ForbiddenConjunction(ABC):
    def __init__(
        self,
        *args: ForbiddenClause | ForbiddenConjunction | ForbiddenRelation,
    ) -> None:
        # Test the classes
        acceptable = (ForbiddenClause, ForbiddenConjunction, ForbiddenRelation)
        for idx, component in enumerate(args):
            if not isinstance(component, acceptable):
                raise TypeError(
                    "Argument #%d is not an instance of %s, "
                    "but %s" % (idx, acceptable, type(component)),
                )

        self.components = args
        dlcs: list[ForbiddenClause | ForbiddenRelation] = []
        for component in self.components:
            if isinstance(component, ForbiddenConjunction):
                dlcs.extend(component.dlcs)
            else:
                dlcs.append(component)

        self.dlcs: tuple[ForbiddenClause | ForbiddenRelation, ...] = tuple(
            unique_everseen(dlcs, key=lambda x: id(x)),
        )

    def __copy__(self) -> Self:
        return self.__class__(*[copy(comp) for comp in self.components])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        if len(self.components) != len(other.components):
            return False

        return all(c in other.components for c in self.components) and all(
            oc in self.components for oc in other.components
        )

    def set_vector_idx(self, hyperparameter_to_idx: Mapping[str, int]) -> None:
        for component in self.components:
            component.set_vector_idx(hyperparameter_to_idx)

    @deprecated("Use `.dlcs` instead")
    def get_descendant_literal_clauses(
        self,
    ) -> tuple[ForbiddenClause | ForbiddenRelation, ...]:
        return self.dlcs

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        pass

    @abstractmethod
    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


class ForbiddenEqualsClause(ForbiddenClause):
    """A ForbiddenEqualsClause.

    It forbids a value from the value range of a hyperparameter to be
    *equal to* ``value``.

    Forbids the value 2 for the hyperparameter *a*

    >>> from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause
    >>>
    >>> cs = ConfigurationSpace({"a": [1, 2, 3]})
    >>> forbidden_clause_a = ForbiddenEqualsClause(cs["a"], 2)
    >>> cs.add(forbidden_clause_a)
    Forbidden: a == 2

    Parameters
    ----------
    hyperparameter : :ref:`Hyperparameters`
        Methods on which a restriction will be made
    value : Any
        forbidden value
    """

    def __init__(self, hyperparameter: Hyperparameter, value: Any) -> None:
        if not hyperparameter.legal_value(value):
            raise ValueError(
                "Forbidden clause must be instantiated with a "
                f"legal hyperparameter value for '{hyperparameter}', but got "
                f"'{value!s}'",
            )
        super().__init__(hyperparameter)
        self.value = value

        # OPTIM: Since forbiddens are used in sampling which converts everything to
        # f64, we pre-convert the value here to make the comparison check faster
        self.vector_value = f64(self.hyperparameter.to_vector(self.value))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return self.hyperparameter == other.hyperparameter and self.value == other.value

    def __repr__(self) -> str:
        return f"Forbidden: {self.hyperparameter.name} == {self.value!r}"

    def __copy__(self) -> Self:
        return self.__class__(hyperparameter=self.hyperparameter, value=self.value)

    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        return (  # type: ignore
            values.get(self.hyperparameter.name, _SENTINEL) == self.value
        )

    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        return vector[self.vector_id] == self.vector_value  # type: ignore

    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        return np.equal(arr[self.vector_id], self.vector_value, dtype=np.bool_)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.hyperparameter.name,
            "type": "EQUALS",
            "value": self.value,
        }


class ForbiddenInClause(ForbiddenClause):
    def __init__(self, hyperparameter: Hyperparameter, values: Iterable[Any]) -> None:
        """A ForbiddenInClause.

        It forbids a value from the value range of a hyperparameter to be
        *in* a collection of ``values``.

        Forbids the values 2, 3 for the hyperparameter *a*

        >>> from ConfigSpace import ConfigurationSpace, ForbiddenInClause
        >>>
        >>> cs = ConfigurationSpace({"a": [1, 2, 3]})
        >>> forbidden_clause_a = ForbiddenInClause(cs['a'], [2, 3])
        >>> cs.add(forbidden_clause_a)
        Forbidden: a in {2, 3}

        Note:
        ----
        The forbidden values have to be a subset of the hyperparameter's values.

        Parameters
        ----------
        hyperparameter : (:ref:`Hyperparameters`, dict)
            Hyperparameter on which a restriction will be made

        values : Any
            Collection of forbidden values
        """
        values = tuple(values)
        for v in values:
            if not hyperparameter.legal_value(v):
                raise ValueError(
                    "Forbidden clause must be instantiated with a "
                    f"legal hyperparameter value for '{hyperparameter}', but got "
                    f"'{v!s}'",
                )
        super().__init__(hyperparameter)
        self.values = values
        self.vector_values = tuple(hyperparameter.to_vector(v) for v in values)

    def __repr__(self) -> str:
        return "Forbidden: {} in {}".format(
            self.hyperparameter.name,
            "{" + ", ".join(repr(value) for value in self.values) + "}",
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if not self.hyperparameter == other.hyperparameter:
            return False

        if not len(self.values) == len(other.values):
            return False

        return all(value in other.values for value in self.values)

    def __copy__(self) -> Self:
        return self.__class__(hyperparameter=self.hyperparameter, values=self.values)

    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        return values.get(self.hyperparameter.name, _SENTINEL) in self.values

    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        return vector[self.vector_id] in self.vector_values

    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        return np.isin(arr[self.vector_id], self.vector_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.hyperparameter.name,
            "type": "IN",
            "values": self.values,
        }


class ForbiddenAndConjunction(ForbiddenConjunction):
    """A ForbiddenAndConjunction.

    The ForbiddenAndConjunction combines forbidden-clauses, which allows to
    build powerful constraints.

    >>> from ConfigSpace import (
    ...     ConfigurationSpace,
    ...     ForbiddenEqualsClause,
    ...     ForbiddenInClause,
    ...     ForbiddenAndConjunction
    ... )
    >>>
    >>> cs = ConfigurationSpace({"a": [1, 2, 3], "b": [2, 5, 6]})
    >>>
    >>> forbidden_clause_a = ForbiddenEqualsClause(cs["a"], 2)
    >>> forbidden_clause_b = ForbiddenInClause(cs["b"], [2])
    >>>
    >>> forbidden_clause = ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_b)
    >>>
    >>> cs.add(forbidden_clause)
    (Forbidden: a == 2 && Forbidden: b in {2})

    Parameters
    ----------
    *args : list(:ref:`Forbidden clauses`)
        forbidden clauses, which should be combined
    """  # noqa: E501

    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        for forbidden in self.components:
            if not forbidden.is_forbidden_value(values):
                return False

        return True

    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        return all(
            forbidden.is_forbidden_vector(vector) for forbidden in self.components
        )

    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        forbidden_mask: Mask = np.ones(shape=arr.shape[1], dtype=np.bool_)
        for forbidden in self.components:
            forbidden_mask &= forbidden.is_forbidden_vector_array(arr)

        return forbidden_mask

    def to_dict(self) -> dict[str, Any]:
        return {
            # name:
            "type": "AND",
            "clauses": [component.to_dict() for component in self.components],
        }


class ForbiddenLessThanRelation(ForbiddenRelation):
    """A ForbiddenLessThan relation between two hyperparameters.

    The ForbiddenLessThan compares the values of two hyperparameters.

    >>> from ConfigSpace import ConfigurationSpace, ForbiddenLessThanRelation
    >>>
    >>> cs = ConfigurationSpace({"a": [1, 2, 3], "b": [2, 5, 6]})
    >>>
    >>> forbidden_clause = ForbiddenLessThanRelation(cs['a'], cs['b'])
    >>> cs.add(forbidden_clause)
    Forbidden: a < b

    Note:
    ----
    If the values of the both hyperparameters are not comparible
    (e.g. comparing int and str), a TypeError is raised. For OrdinalHyperparameters
    the actual values are used for comparison **not** their ordinal value.

    Parameters
    ----------
     left : :ref:`Hyperparameters`
         left side of the comparison

     right : :ref:`Hyperparameters`
         right side of the comparison
    """

    _RELATION_STR = "LESS"

    def __repr__(self) -> str:
        return f"Forbidden: {self.left.name} < {self.right.name}"

    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        # Relation is always evaluated against actual value and not vector rep
        left = values.get(self.left.name, _SENTINEL)
        if left is _SENTINEL:
            return False

        right = values.get(self.right.name, _SENTINEL)
        if right is _SENTINEL:
            return False

        return left < right  # type: ignore

    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        # Relation is always evaluated against actual value and not vector rep
        left: f64 = vector[self.vector_ids[0]]  # type: ignore
        right: f64 = vector[self.vector_ids[1]]  # type: ignore
        return self.left.to_value(left) < self.right.to_value(right)  # type: ignore

    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        left = arr[self.vector_ids[0]]
        right = arr[self.vector_ids[1]]
        return self.left.to_value(left) < self.right.to_value(right)


class ForbiddenEqualsRelation(ForbiddenRelation):
    """A ForbiddenEquals relation between two hyperparameters.

    The ForbiddenEquals compares the values of two hyperparameters.

    >>> from ConfigSpace import ConfigurationSpace, ForbiddenEqualsRelation
    >>>
    >>> cs = ConfigurationSpace({"a": [1, 2, 3], "b": [2, 5, 6]})
    >>>
    >>> forbidden_clause = ForbiddenEqualsRelation(cs['a'], cs['b'])
    >>> cs.add(forbidden_clause)
    Forbidden: a == b

    Note:
    ----
    If the values of the both hyperparameters are not comparible
    (e.g. comparing int and str), a TypeError is raised. For OrdinalHyperparameters
    the actual values are used for comparison **not** their ordinal value.

    Parameters
    ----------
     left : :ref:`Hyperparameters`
         left side of the comparison
     right : :ref:`Hyperparameters`
         right side of the comparison
    """

    _RELATION_STR = "EQUALS"

    def __repr__(self) -> str:
        return f"Forbidden: {self.left.name} == {self.right.name}"

    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        left = values.get(self.left.name, _SENTINEL)
        if left is _SENTINEL:
            return False

        right = values.get(self.right.name, _SENTINEL)
        if right is _SENTINEL:
            return False

        return left == right  # type: ignore

    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        # Relation is always evaluated against actual value and not vector rep
        left = vector[self.vector_ids[0]]
        right = vector[self.vector_ids[1]]
        return self.left.to_value(left) == self.right.to_value(right)  # type: ignore

    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        left = arr[self.vector_ids[0]]
        right = arr[self.vector_ids[1]]
        return self.left.to_value(left) < self.right.to_value(right)  # type: ignore


class ForbiddenGreaterThanRelation(ForbiddenRelation):
    """A ForbiddenGreaterThan relation between two hyperparameters.

    The ForbiddenGreaterThan compares the values of two hyperparameters.

    >>> from ConfigSpace import ConfigurationSpace, ForbiddenGreaterThanRelation
    >>>
    >>> cs = ConfigurationSpace({"a": [1, 2, 3], "b": [2, 5, 6]})
    >>> forbidden_clause = ForbiddenGreaterThanRelation(cs['a'], cs['b'])
    >>>
    >>> cs.add(forbidden_clause)
    Forbidden: a > b

    Note:
    ----
    If the values of the both hyperparameters are not comparible
    (e.g. comparing int and str), a TypeError is raised. For OrdinalHyperparameters
    the actual values are used for comparison **not** their ordinal value.

    Parameters
    ----------
     left : :ref:`Hyperparameters`
         left side of the comparison
     right : :ref:`Hyperparameters`
         right side of the comparison
    """

    _RELATION_STR = "GREATER"

    def __repr__(self) -> str:
        return f"Forbidden: {self.left.name} > {self.right.name}"

    def is_forbidden_value(self, values: dict[str, Any]) -> bool:
        left = values.get(self.left.name, _SENTINEL)
        if left is _SENTINEL:
            return False

        right = values.get(self.right.name, _SENTINEL)
        if right is _SENTINEL:
            return False

        return left > right  # type: ignore

    def is_forbidden_vector(self, vector: Array[f64]) -> bool:
        # Relation is always evaluated against actual value and not vector rep
        left: f64 = vector[self.vector_ids[0]]  # type: ignore
        right: f64 = vector[self.vector_ids[1]]  # type: ignore
        return self.left.to_value(left) > self.right.to_value(right)  # type: ignore

    def is_forbidden_vector_array(self, arr: Array[f64]) -> Mask:
        left = arr[self.vector_ids[0]]
        right = arr[self.vector_ids[1]]
        return self.left.to_value(left) > self.right.to_value(right)


ForbiddenLike = Union[
    ForbiddenClause,
    ForbiddenConjunction,
    ForbiddenRelation,
]
