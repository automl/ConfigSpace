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
#       names of itConfigurationSpaces contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
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

import copy
import io
from typing import Any

import numpy as np

from ConfigSpace.hyperparameters import Hyperparameter


class AbstractForbiddenComponent:
    def __init__(self):
        pass

    def __repr__(self):
        pass

    def __eq__(self, other: Any) -> bool:
        """
        This method implements a comparison between self and another
        object.

        Additionally, it defines the __ne__() as stated in the
        documentation from python:
            By default, object implements __eq__() by using is, returning NotImplemented
            in the case of a false comparison: True if x is y else NotImplemented.
            For __ne__(), by default it delegates to __eq__() and inverts the result
            unless it is NotImplemented.

        """
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.value == other.value and self.hyperparameter.name == other.hyperparameter.name

    def __hash__(self) -> int:
        """Override the default hash behavior (that returns the id or the object)."""
        return hash(tuple(sorted(self.__dict__.items())))

    def __copy__(self):
        raise NotImplementedError()

    def get_descendant_literal_clauses(self):
        pass

    def set_vector_idx(self, hyperparameter_to_idx):
        pass

    def is_forbidden(self, instantiated_hyperparameters, strict):
        pass

    def is_forbidden_vector(self, instantiated_hyperparameters, strict):
        return bool(self.c_is_forbidden_vector(instantiated_hyperparameters, strict))

    def c_is_forbidden_vector(self, instantiated_hyperparameters: np.ndarray, strict: int) -> int:
        pass


class AbstractForbiddenClause(AbstractForbiddenComponent):
    def __init__(self, hyperparameter: Hyperparameter):
        if not isinstance(hyperparameter, Hyperparameter):
            raise TypeError("Argument 'hyperparameter' is not of type %s." % Hyperparameter)
        self.hyperparameter = hyperparameter
        self.vector_id = -1

    def get_descendant_literal_clauses(self):
        return (self,)

    def set_vector_idx(self, hyperparameter_to_idx):
        self.vector_id = hyperparameter_to_idx[self.hyperparameter.name]


class SingleValueForbiddenClause(AbstractForbiddenClause):
    def __init__(self, hyperparameter: Hyperparameter, value: Any) -> None:
        super().__init__(hyperparameter)
        if not self.hyperparameter.is_legal(value):
            raise ValueError(
                "Forbidden clause must be instantiated with a "
                f"legal hyperparameter value for '{self.hyperparameter}', but got "
                f"'{value!s}'",
            )
        self.value = value
        self.vector_value = self.hyperparameter._inverse_transform(self.value)

    def __copy__(self):
        return self.__class__(
            hyperparameter=copy.copy(self.hyperparameter),
            value=self.value,
        )

    def is_forbidden(self, instantiated_hyperparameters, strict) -> int:
        value = instantiated_hyperparameters.get(self.hyperparameter.name)
        if value is None:
            if strict:
                raise ValueError(
                    "Is_forbidden must be called with the "
                    "instantiated hyperparameter in the "
                    "forbidden clause; you are missing "
                    "'%s'" % self.hyperparameter.name,
                )
            else:
                return False

        return self._is_forbidden(value)

    def c_is_forbidden_vector(self, instantiated_vector: np.ndarray, strict: int) -> int:
        value = instantiated_vector[self.vector_id]
        if value != value:
            if strict:
                raise ValueError(
                    "Is_forbidden must be called with the "
                    "instantiated vector id in the "
                    "forbidden clause; you are missing "
                    "'%s'" % self.vector_id,
                )
            else:
                return False

        return self._is_forbidden_vector(value)

    def _is_forbidden(self, value) -> int:
        pass

    def _is_forbidden_vector(self, value) -> int:
        pass


class MultipleValueForbiddenClause(AbstractForbiddenClause):
    def __init__(self, hyperparameter: Hyperparameter, values: Any) -> None:
        super().__init__(hyperparameter)

        for value in values:
            if not self.hyperparameter.is_legal(value):
                raise ValueError(
                    "Forbidden clause must be instantiated with a "
                    f"legal hyperparameter value for '{self.hyperparameter}', but got "
                    f"'{value!s}'",
                )
        self.values = values
        self.vector_values = [
            self.hyperparameter._inverse_transform(value) for value in self.values
        ]

    def __copy__(self):
        return self.__class__(
            hyperparameter=copy.copy(self.hyperparameter),
            values=copy.deepcopy(self.values),
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return (
            self.hyperparameter == other.hyperparameter
            and self.values == other.values
        )

    def is_forbidden(self, instantiated_hyperparameters, strict) -> bool:
        value = instantiated_hyperparameters.get(self.hyperparameter.name)
        if value is None:
            if strict:
                raise ValueError(
                    "Is_forbidden must be called with the "
                    "instantiated hyperparameter in the "
                    "forbidden clause; you are missing "
                    "'%s'." % self.hyperparameter.name,
                )
            return False

        return self._is_forbidden(value)

    def c_is_forbidden_vector(self, instantiated_vector: np.ndarray, strict: int) -> int:
        value = instantiated_vector[self.vector_id]

        if value != value:
            if strict:
                raise ValueError(
                    "Is_forbidden must be called with the "
                    "instantiated vector id in the "
                    "forbidden clause; you are missing "
                    "'%s'" % self.vector_id,
                )
            else:
                return False

        return self._is_forbidden_vector(value)

    def _is_forbidden(self, value) -> int:
        pass

    def _is_forbidden_vector(self, value) -> int:
        pass


class ForbiddenEqualsClause(SingleValueForbiddenClause):
    """A ForbiddenEqualsClause.

    It forbids a value from the value range of a hyperparameter to be
    *equal to* ``value``.

    Forbids the value 2 for the hyperparameter *a*

    >>> from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause
    >>>
    >>> cs = ConfigurationSpace({"a": [1, 2, 3]})
    >>> forbidden_clause_a = ForbiddenEqualsClause(cs["a"], 2)
    >>> cs.add_forbidden_clause(forbidden_clause_a)
    Forbidden: a == 2

    Parameters
    ----------
    hyperparameter : :ref:`Hyperparameters`
        Methods on which a restriction will be made
    value : Any
        forbidden value
    """

    def __repr__(self):
        return f"Forbidden: {self.hyperparameter.name} == {self.value!r}"

    def _is_forbidden(self, value) -> bool:
        return value == self.value

    def _is_forbidden_vector(self, value) -> bool:
        return value == self.vector_value


class ForbiddenInClause(MultipleValueForbiddenClause):
    def __init__(
        self,
        hyperparameter: dict[str, None | str | float | int],
        values: Any,
    ) -> None:
        """A ForbiddenInClause.

        It forbids a value from the value range of a hyperparameter to be
        *in* a collection of ``values``.

        Forbids the values 2, 3 for the hyperparameter *a*

        >>> from ConfigSpace import ConfigurationSpace, ForbiddenInClause
        >>>
        >>> cs = ConfigurationSpace({"a": [1, 2, 3]})
        >>> forbidden_clause_a = ForbiddenInClause(cs['a'], [2, 3])
        >>> cs.add_forbidden_clause(forbidden_clause_a)
        Forbidden: a in {2, 3}

        Note
        ----
        The forbidden values have to be a subset of the hyperparameter's values.

        Parameters
        ----------
        hyperparameter : (:ref:`Hyperparameters`, dict)
            Hyperparameter on which a restriction will be made

        values : Any
            Collection of forbidden values
        """
        super().__init__(hyperparameter, values)
        self.values = set(self.values)
        self.vector_values = set(self.vector_values)

    def __repr__(self) -> str:
        return "Forbidden: {} in {}".format(
            self.hyperparameter.name,
            "{" + ", ".join(repr(value) for value in sorted(self.values)) + "}",
        )

    def _is_forbidden(self, value) -> int:
        return value in self.values

    def _is_forbidden_vector(self, value) -> int:
        return value in self.vector_values


class AbstractForbiddenConjunction(AbstractForbiddenComponent):
    def __init__(self, *args: AbstractForbiddenComponent) -> None:
        super().__init__()
        # Test the classes
        for idx, component in enumerate(args):
            if not isinstance(component, AbstractForbiddenComponent):
                raise TypeError(
                    "Argument #%d is not an instance of %s, "
                    "but %s" % (idx, AbstractForbiddenComponent, type(component)),
                )

        self.components = args
        self.n_components = len(self.components)
        self.dlcs = self.get_descendant_literal_clauses()

    def __repr__(self):
        pass

    def __copy__(self):
        return self.__class__([copy(comp) for comp in self.components])

    def __eq__(self, other: Any) -> bool:
        """Comparison between self and another object.

        Additionally, it defines the __ne__() as stated in the
        documentation from python:
            By default, object implements __eq__() by using is, returning NotImplemented
            in the case of a false comparison: True if x is y else NotImplemented.
            For __ne__(), by default it delegates to __eq__() and inverts the result
            unless it is NotImplemented.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self.n_components != other.n_components:
            return False

        return all(self.components[i] == other.components[i] for i in range(self.n_components))

    def set_vector_idx(self, hyperparameter_to_idx) -> None:
        for component in self.components:
            component.set_vector_idx(hyperparameter_to_idx)

    def get_descendant_literal_clauses(self):
        children = []
        for component in self.components:
            if isinstance(component, AbstractForbiddenConjunction):
                children.extend(component.get_descendant_literal_clauses())
            else:
                children.append(component)
        return tuple(children)

    def is_forbidden(self, instantiated_hyperparameters, strict) -> bool:
        ihp_names = list(instantiated_hyperparameters.keys())

        for dlc in self.dlcs:
            if dlc.hyperparameter.name not in ihp_names:
                if strict:
                    raise ValueError(
                        "Is_forbidden must be called with all "
                        "instantiated hyperparameters in the "
                        "and conjunction of forbidden clauses; "
                        "you are (at least) missing "
                        "'%s'" % dlc.hyperparameter.name,
                    )
                else:
                    return False

        values = np.empty(self.n_components, dtype=np.int32)

        # Finally, call is_forbidden for all direct descendents and combine the
        # outcomes
        np_index = 0
        for component in self.components:
            e = component.is_forbidden(instantiated_hyperparameters, strict=strict)
            values[np_index] = e
            np_index += 1

        return self._is_forbidden(self.n_components, values)

    def c_is_forbidden_vector(self, instantiated_vector: np.ndarray, strict: int) -> int:
        e: int = 0
        values = np.empty(self.n_components, dtype=np.int32)

        # Finally, call is_forbidden for all direct descendents and combine the
        # outcomes. Check only as many forbidden clauses as the actual
        # evaluation function queries for (e.g. and conditions are False
        # if only one of the components evaluates to False).

        for i in range(self.n_components):
            component = self.components[i]
            e = component.c_is_forbidden_vector(instantiated_vector, strict)
            values[i] = e

        return self._is_forbidden(self.n_components, values)

    def _is_forbidden(self, I: int, evaluations) -> int:
        pass


class ForbiddenAndConjunction(AbstractForbiddenConjunction):
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
    >>> cs.add_forbidden_clause(forbidden_clause)
    (Forbidden: a == 2 && Forbidden: b in {2})

    Parameters
    ----------
    *args : list(:ref:`Forbidden clauses`)
        forbidden clauses, which should be combined
    """

    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    def _is_forbidden(self, I: int, evaluations) -> int:
        # Return False if one of the components evaluates to False

        for i in range(I):
            if evaluations[i] == 0:
                return 0
        return 1

    def c_is_forbidden_vector(self, instantiated_vector: np.ndarray, strict: int) -> int:
        # Copy from above to have early stopping of the evaluation of clauses -
        # gave only very modest improvements of ~5%; should probably be reworked
        # if adding more conjunctions in order to use better software design to
        # avoid code duplication.
        e: int = 0

        # Finally, call is_forbidden for all direct descendents and combine the
        # outcomes. Check only as many forbidden clauses as the actual
        # evaluation function queries for (e.g. and conditions are False
        # if only one of the components evaluates to False).

        for i in range(self.n_components):
            component = self.components[i]
            e = component.c_is_forbidden_vector(instantiated_vector, strict)
            if e == 0:
                return 0

        return 1


class ForbiddenRelation(AbstractForbiddenComponent):
    def __init__(self, left: Hyperparameter, right: Hyperparameter):
        if not isinstance(left, Hyperparameter):
            raise TypeError("Argument 'left' is not of type %s." % Hyperparameter)
        if not isinstance(right, Hyperparameter):
            raise TypeError("Argument 'right' is not of type %s." % Hyperparameter)

        self.left = left
        self.right = right
        self.vector_ids = (-1, -1)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.left == other.left and self.right == other.right

    def __copy__(self):
        return self.__class__(
            a=copy.copy(self.left),
            b=copy.copy(self.right),
        )

    def get_descendant_literal_clauses(self):
        return (self,)

    def set_vector_idx(self, hyperparameter_to_idx):
        self.vector_ids = (
            hyperparameter_to_idx[self.left.name],
            hyperparameter_to_idx[self.right.name],
        )

    def is_forbidden(self, instantiated_hyperparameters, strict):
        left = instantiated_hyperparameters.get(self.left.name)
        right = instantiated_hyperparameters.get(self.right.name)
        if left is None:
            if strict:
                raise ValueError(
                    "Is_forbidden must be called with the "
                    "instantiated hyperparameters in the "
                    "forbidden clause; you are missing "
                    "'%s'" % self.left.name,
                )
            else:
                return False
        if right is None:
            if strict:
                raise ValueError(
                    "Is_forbidden must be called with the "
                    "instantiated hyperparameters in the "
                    "forbidden clause; you are missing "
                    "'%s'" % self.right.name,
                )
            else:
                return False

        return self._is_forbidden(left, right)

    def _is_forbidden(self, left, right) -> int:
        pass

    def c_is_forbidden_vector(self, instantiated_vector: np.ndarray, strict: int) -> int:
        left = instantiated_vector[self.vector_ids[0]]
        right = instantiated_vector[self.vector_ids[1]]

        if left != left:
            if strict:
                raise ValueError(
                    "Is_forbidden must be called with the "
                    "instantiated vector id in the "
                    "forbidden clause; you are missing "
                    "'%s'" % self.vector_ids[0],
                )
            else:
                return False

        if right != right:
            if strict:
                raise ValueError(
                    "Is_forbidden must be called with the "
                    "instantiated vector id in the "
                    "forbidden clause; you are missing "
                    "'%s'" % self.vector_ids[1],
                )
            else:
                return False

        # Relation is always evaluated against actual value and not vector representation
        return self._is_forbidden(self.left._transform(left), self.right._transform(right))

    def _is_forbidden_vector(self, left, right) -> int:
        pass


class ForbiddenLessThanRelation(ForbiddenRelation):
    """A ForbiddenLessThan relation between two hyperparameters.

    The ForbiddenLessThan compares the values of two hyperparameters.

    >>> from ConfigSpace import ConfigurationSpace, ForbiddenLessThanRelation
    >>>
    >>> cs = ConfigurationSpace({"a": [1, 2, 3], "b": [2, 5, 6]})
    >>>
    >>> forbidden_clause = ForbiddenLessThanRelation(cs['a'], cs['b'])
    >>> cs.add_forbidden_clause(forbidden_clause)
    Forbidden: a < b

    Note
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

    def __repr__(self):
        return f"Forbidden: {self.left.name} < {self.right.name}"

    def _is_forbidden(self, left, right) -> int:
        return left < right

    def _is_forbidden_vector(self, left, right) -> int:
        return left < right


class ForbiddenEqualsRelation(ForbiddenRelation):
    """A ForbiddenEquals relation between two hyperparameters.

    The ForbiddenEquals compares the values of two hyperparameters.

    >>> from ConfigSpace import ConfigurationSpace, ForbiddenEqualsRelation
    >>>
    >>> cs = ConfigurationSpace({"a": [1, 2, 3], "b": [2, 5, 6]})
    >>>
    >>> forbidden_clause = ForbiddenEqualsRelation(cs['a'], cs['b'])
    >>> cs.add_forbidden_clause(forbidden_clause)
    Forbidden: a == b

    Note
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

    def __repr__(self):
        return f"Forbidden: {self.left.name} == {self.right.name}"

    def _is_forbidden(self, left, right) -> int:
        return left == right

    def _is_forbidden_vector(self, left, right) -> int:
        return left == right


class ForbiddenGreaterThanRelation(ForbiddenRelation):
    """A ForbiddenGreaterThan relation between two hyperparameters.

    The ForbiddenGreaterThan compares the values of two hyperparameters.

    >>> from ConfigSpace import ConfigurationSpace, ForbiddenGreaterThanRelation
    >>>
    >>> cs = ConfigurationSpace({"a": [1, 2, 3], "b": [2, 5, 6]})
    >>> forbidden_clause = ForbiddenGreaterThanRelation(cs['a'], cs['b'])
    >>>
    >>> cs.add_forbidden_clause(forbidden_clause)
    Forbidden: a > b

    Note
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

    def __repr__(self):
        return f"Forbidden: {self.left.name} > {self.right.name}"

    def _is_forbidden(self, left, right) -> int:
        return left > right

    def _is_forbidden_vector(self, left, right) -> int:
        return left > right
