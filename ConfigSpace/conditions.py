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
#       names of its contributors may be used to endorse or promote products
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
from abc import ABC, abstractmethod
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

from ConfigSpace.hyperparameters.hyperparameter import Comparison

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


class ConditionComponent(ABC):
    @abstractmethod
    def set_vector_idx(self, hyperparameter_to_idx) -> None:
        pass

    @abstractmethod
    def get_children_vector(self) -> list[int]:
        pass

    @abstractmethod
    def get_parents_vector(self) -> list[int]:
        pass

    @abstractmethod
    def get_children(self) -> list[ConditionComponent]:
        pass

    @abstractmethod
    def get_parents(self) -> list[ConditionComponent]:
        pass

    @abstractmethod
    def get_descendant_literal_conditions(self) -> list[AbstractCondition]:
        pass

    @abstractmethod
    def evaluate(
        self,
        instantiated_parent_hyperparameter: dict[str, None | int | float | str],
    ) -> bool:
        pass

    def evaluate_vector(self, instantiated_vector):
        return bool(self._evaluate_vector(instantiated_vector))

    @abstractmethod
    def _evaluate_vector(self, value: np.ndarray) -> bool:
        pass

    def __hash__(self) -> int:
        """Override the default hash behavior (that returns the id or the object)."""
        return hash(tuple(sorted(self.__dict__.items())))


class AbstractCondition(ConditionComponent):
    def __init__(self, child: Hyperparameter, parent: Hyperparameter) -> None:
        if child == parent:
            raise ValueError(
                "The child and parent hyperparameter must be different "
                "hyperparameters.",
            )
        self.child = child
        self.parent = parent
        self.child_vector_id = -1
        self.parent_vector_id = -1

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self.child != other.child or self.parent != other.parent:
            return False

        return self.value == other.value

    def set_vector_idx(self, hyperparameter_to_idx: dict):
        self.child_vector_id = hyperparameter_to_idx[self.child.name]
        self.parent_vector_id = hyperparameter_to_idx[self.parent.name]

    def get_children_vector(self) -> list[int]:
        return [self.child_vector_id]

    def get_parents_vector(self) -> list[int]:
        return [self.parent_vector_id]

    def get_children(self) -> list[Hyperparameter]:
        return [self.child]

    def get_parents(self) -> list[Hyperparameter]:
        return [self.parent]

    def get_descendant_literal_conditions(self) -> list[AbstractCondition]:
        return [self]

    def evaluate(
        self,
        instantiated_parent_hyperparameter: dict[str, int | float | str],
    ) -> bool:
        value = instantiated_parent_hyperparameter[self.parent.name]
        if isinstance(value, float | int | np.number) and np.isnan(value):
            return False
        return self._evaluate(value)

    def _evaluate_vector(self, instantiated_vector: np.ndarray) -> bool:
        if self.parent_vector_id is None:
            raise ValueError(
                "Parent vector id should not be None when calling evaluate vector",
            )

        return self._inner_evaluate_vector(instantiated_vector[self.parent_vector_id])

    @abstractmethod
    def _evaluate(self, instantiated_parent_hyperparameter: str | int | float) -> bool:
        pass

    @abstractmethod
    def _inner_evaluate_vector(self, value) -> bool:
        pass


class EqualsCondition(AbstractCondition):
    def __init__(
        self,
        child: Hyperparameter,
        parent: Hyperparameter,
        value: str | float | int,
    ) -> None:
        """Hyperparameter ``child`` is conditional on the ``parent`` hyperparameter
        being *equal* to ``value``.

        Make *b* an active hyperparameter if *a* has the value 1

        >>> from ConfigSpace import ConfigurationSpace, EqualsCondition
        >>>
        >>> cs = ConfigurationSpace({
        ...     "a": [1, 2, 3],
        ...     "b": (1.0, 8.0)
        ... })
        >>> cond = EqualsCondition(cs['b'], cs['a'], 1)
        >>> cs.add_condition(cond)
        b | a == 1

        Parameters
        ----------
        child : :ref:`Hyperparameters`
            This hyperparameter will be sampled in the configspace
            if the *equal condition* is satisfied
        parent : :ref:`Hyperparameters`
            The hyperparameter, which has to satisfy the *equal condition*
        value : str, float, int
            Value, which the parent is compared to
        """
        super().__init__(child, parent)
        if not parent.legal_value(value):
            raise ValueError(
                f"Hyperparameter '{child.name}' is "
                f"conditional on the illegal value '{value}' of "
                f"its parent hyperparameter '{parent.name}'",
            )
        self.value = value
        self.vector_value = self.parent.to_vector(self.value)

    def __repr__(self) -> str:
        return f"{self.child.name} | {self.parent.name} == {self.value!r}"

    def __copy__(self):
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def _evaluate(self, value: str | float | int) -> bool:
        # No need to check if the value to compare is a legal value; either it
        # is equal (and thus legal), or it would evaluate to False anyway
        return self.parent.compare_value(value, self.value) is Comparison.EQUAL

    def _inner_evaluate_vector(self, value) -> bool:
        # No need to check if the value to compare is a legal value; either it
        # is equal (and thus legal), or it would evaluate to False anyway
        return self.parent.compare_vector(value, self.vector_value) is Comparison.EQUAL


class NotEqualsCondition(AbstractCondition):
    def __init__(
        self,
        child: Hyperparameter,
        parent: Hyperparameter,
        value: str | float | int,
    ) -> None:
        """Hyperparameter ``child`` is conditional on the ``parent`` hyperparameter
        being *not equal* to ``value``.

        Make *b* an active hyperparameter if *a* has **not** the value 1

        >>> from ConfigSpace import ConfigurationSpace, NotEqualsCondition
        >>>
        >>> cs = ConfigurationSpace({
        ...     "a": [1, 2, 3],
        ...     "b": (1.0, 8.0)
        ... })
        >>> cond = NotEqualsCondition(cs['b'], cs['a'], 1)
        >>> cs.add_condition(cond)
        b | a != 1

        Parameters
        ----------
        child : :ref:`Hyperparameters`
            This hyperparameter will be sampled in the configspace
            if the not-equals condition is satisfied
        parent : :ref:`Hyperparameters`
            The hyperparameter, which has to satisfy the
            *not equal condition*
        value : str, float, int
            Value, which the parent is compared to
        """
        super().__init__(child, parent)
        if not parent.legal_value(value):
            raise ValueError(
                f"Hyperparameter '{child.name}' is "
                f"conditional on the illegal value '{value}' of "
                f"its parent hyperparameter '{parent.name}'",
            )
        self.value = value
        self.vector_value = self.parent.to_vector(self.value)

    def __repr__(self) -> str:
        return f"{self.child.name} | {self.parent.name} != {self.value!r}"

    def __copy__(self):
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def _evaluate(self, value: str | float | int) -> bool:
        if not self.parent.legal_value(value):
            return False

        return self.parent.compare_value(value, self.value) is not Comparison.EQUAL

    def _inner_evaluate_vector(self, value) -> bool:
        if not self.parent.legal_vector(value):
            return False
        return (
            self.parent.compare_vector(value, self.vector_value) is not Comparison.EQUAL
        )


class LessThanCondition(AbstractCondition):
    def __init__(
        self,
        child: Hyperparameter,
        parent: Hyperparameter,
        value: str | float | int,
    ) -> None:
        """Hyperparameter ``child`` is conditional on the ``parent`` hyperparameter
        being *less than* ``value``.

        Make *b* an active hyperparameter if *a* is less than 5

        >>> from ConfigSpace import ConfigurationSpace, LessThanCondition
        >>>
        >>> cs = ConfigurationSpace({
        ...    "a": (0, 10),
        ...    "b": (1.0, 8.0)
        ... })
        >>> cond = LessThanCondition(cs['b'], cs['a'], 5)
        >>> cs.add_condition(cond)
        b | a < 5

        Parameters
        ----------
        child : :ref:`Hyperparameters`
            This hyperparameter will be sampled in the configspace,
            if the *LessThanCondition* is satisfied
        parent : :ref:`Hyperparameters`
            The hyperparameter, which has to satisfy the *LessThanCondition*
        value : str, float, int
            Value, which the parent is compared to
        """
        super().__init__(child, parent)
        if not self.parent.orderable:
            raise ValueError(
                f"The parent hyperparameter must be orderable to use "
                f"GreaterThanCondition, however {self.parent} is not.",
            )
        if not parent.legal_value(value):
            raise ValueError(
                f"Hyperparameter '{child.name}' is "
                f"conditional on the illegal value '{value}' of "
                f"its parent hyperparameter '{parent.name}'",
            )
        self.value = value
        self.vector_value = self.parent.to_vector(self.value)

    def __repr__(self) -> str:
        return f"{self.child.name} | {self.parent.name} < {self.value!r}"

    def __copy__(self):
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def _evaluate(self, value: str | float | int) -> bool:
        if not self.parent.legal_value(value):
            return False

        return self.parent.compare_value(value, self.value) is Comparison.LESS_THAN

    def _inner_evaluate_vector(self, value) -> bool:
        if not self.parent.legal_vector(value):
            return False

        return (
            self.parent.compare_vector(value, self.vector_value) is Comparison.LESS_THAN
        )


class GreaterThanCondition(AbstractCondition):
    def __init__(
        self,
        child: Hyperparameter,
        parent: Hyperparameter,
        value: str | float | int,
    ) -> None:
        """Hyperparameter ``child`` is conditional on the ``parent`` hyperparameter
        being *greater than* ``value``.

        Make *b* an active hyperparameter if *a* is greater than 5

        >>> from ConfigSpace import ConfigurationSpace, GreaterThanCondition
        >>>
        >>> cs = ConfigurationSpace({
        ...     "a": (0, 10),
        ...     "b": (1.0, 8.0)
        ... })
        >>> cond = GreaterThanCondition(cs['b'], cs['a'], 5)
        >>> cs.add_condition(cond)
        b | a > 5

        Parameters
        ----------
        child : :ref:`Hyperparameters`
            This hyperparameter will be sampled in the configspace,
            if the *GreaterThanCondition* is satisfied
        parent : :ref:`Hyperparameters`
            The hyperparameter, which has to satisfy the *GreaterThanCondition*
        value : str, float, int
            Value, which the parent is compared to
        """
        super().__init__(child, parent)

        if not self.parent.orderable:
            raise ValueError(
                f"The parent hyperparameter must be orderable to use "
                f"GreaterThanCondition, however {self.parent} is not.",
            )
        if not parent.legal_value(value):
            raise ValueError(
                f"Hyperparameter '{child.name}' is "
                f"conditional on the illegal value '{value}' of "
                f"its parent hyperparameter '{parent.name}'",
            )
        self.value = value
        self.vector_value = self.parent.to_vector(self.value)

    def __repr__(self) -> str:
        return f"{self.child.name} | {self.parent.name} > {self.value!r}"

    def __copy__(self):
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def _evaluate(self, value: None | str | float | int) -> bool:
        if not self.parent.legal_value(value):
            return False

        return self.parent.compare_value(value, self.value) is Comparison.GREATER_THAN

    def _inner_evaluate_vector(self, value) -> int:
        if not self.parent.legal_vector(value):
            return False

        return (
            self.parent.compare_vector(value, self.vector_value)
            is Comparison.GREATER_THAN
        )


class InCondition(AbstractCondition):
    def __init__(
        self,
        child: Hyperparameter,
        parent: Hyperparameter,
        values: list[str | float | int],
    ) -> None:
        """Hyperparameter ``child`` is conditional on the ``parent`` hyperparameter
        being *in* a set of ``values``.

        make *b* an active hyperparameter if *a* is in the set [1, 2, 3, 4]

        >>> from ConfigSpace import ConfigurationSpace, InCondition
        >>>
        >>> cs = ConfigurationSpace({
        ...     "a": (0, 10),
        ...     "b": (1.0, 8.0)
        ... })
        >>> cond = InCondition(cs['b'], cs['a'], [1, 2, 3, 4])
        >>> cs.add_condition(cond)
        b | a in {1, 2, 3, 4}

        Parameters
        ----------
        child : :ref:`Hyperparameters`
            This hyperparameter will be sampled in the configspace,
            if the *InCondition* is satisfied
        parent : :ref:`Hyperparameters`
            The hyperparameter, which has to satisfy the *InCondition*
        values : list(str, float, int)
            Collection of values, which the parent is compared to

        """
        super().__init__(child, parent)
        for value in values:
            if not parent.legal_value(value):
                raise ValueError(
                    f"Hyperparameter '{child.name}' is "
                    f"conditional on the illegal value '{value}' of "
                    f"its parent hyperparameter '{parent.name}'",
                )
        self.values = values
        self.value = values
        self.vector_values = [self.parent.to_vector(value) for value in self.values]

    def __repr__(self) -> str:
        return "{} | {} in {{{}}}".format(
            self.child.name,
            self.parent.name,
            ", ".join([repr(value) for value in self.values]),
        )

    def _evaluate(self, value: str | float | int) -> bool:
        return value in self.values

    def _inner_evaluate_vector(self, value) -> bool:
        return value in self.vector_values


class AbstractConjunction(ConditionComponent):
    def __init__(self, *args: AbstractCondition) -> None:
        super().__init__()
        self.components = args
        self.n_components = len(self.components)
        self.dlcs = self.get_descendant_literal_conditions()

        # Test the classes
        for idx, component in enumerate(self.components):
            if not isinstance(component, ConditionComponent):
                raise TypeError(
                    "Argument #%d is not an instance of %s, "
                    "but %s" % (idx, ConditionComponent, type(component)),
                )

        # Test that all conjunctions and conditions have the same child!
        children = self.get_children()
        for c1, c2 in combinations(children, 2):
            if c1 != c2:
                raise ValueError(
                    "All Conjunctions and Conditions must have " "the same child.",
                )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if len(self.components) != len(other.components):
            return False

        for component, other_component in zip(
            self.components,
            other.components,
            strict=True,
        ):
            if component != other_component:
                return False

        return True

    def __copy__(self):
        return self.__class__(*[copy.copy(comp) for comp in self.components])

    def get_descendant_literal_conditions(self) -> tuple[AbstractCondition]:
        children = []
        for component in self.components:
            if isinstance(component, AbstractConjunction):
                children.extend(component.get_descendant_literal_conditions())
            else:
                children.append(component)
        return tuple(children)

    def set_vector_idx(self, hyperparameter_to_idx: dict):
        for component in self.components:
            component.set_vector_idx(hyperparameter_to_idx)

    def get_children_vector(self) -> list[int]:
        children_vector = []
        for component in self.components:
            children_vector.extend(component.get_children_vector())
        return children_vector

    def get_parents_vector(self) -> list[int]:
        parents_vector = []
        for component in self.components:
            parents_vector.extend(component.get_parents_vector())
        return parents_vector

    def get_children(self) -> list[ConditionComponent]:
        children = []
        for component in self.components:
            children.extend(component.get_children())
        return children

    def get_parents(self) -> list[ConditionComponent]:
        parents = []
        for component in self.components:
            parents.extend(component.get_parents())
        return parents

    def evaluate(
        self,
        instantiated_hyperparameters: dict[str, None | int | float | str],
    ) -> bool:
        # Then, check if all parents were passed
        conditions = self.dlcs
        for condition in conditions:
            if condition.parent.name not in instantiated_hyperparameters:
                raise ValueError(
                    "Evaluate must be called with all "
                    "instanstatiated parent hyperparameters in "
                    "the conjunction; you are (at least) missing "
                    "'%s'" % condition.parent.name,
                )

        # Finally, call evaluate for all direct descendents and combine the
        # outcomes
        values = np.empty(self.n_components, dtype=np.int32)
        for i, component in enumerate(self.components):
            e = component.evaluate(instantiated_hyperparameters)
            values[i] = e

        return self._evaluate(values)

    def _evaluate_vector(self, instantiated_vector: np.ndarray) -> bool:
        values = np.empty(self.n_components, dtype=np.int32)

        # Finally, call evaluate for all direct descendents and combine the
        # outcomes
        for i in range(self.n_components):
            component = self.components[i]
            e = component._evaluate_vector(instantiated_vector)
            values[i] = e

        return self._evaluate(values)

    @abstractmethod
    def _evaluate(self, evaluations) -> bool:
        pass


class AndConjunction(AbstractConjunction):
    # TODO: test if an AndConjunction results in an illegal state or a
    #       Tautology! -> SAT solver
    def __init__(self, *args: AbstractCondition) -> None:
        """By using the *AndConjunction*, constraints can easily be connected.

        The following example shows how two constraints with an *AndConjunction*
        can be combined.

        >>> from ConfigSpace import (
        ...     ConfigurationSpace,
        ...     LessThanCondition,
        ...     GreaterThanCondition,
        ...     AndConjunction
        ... )
        >>>
        >>> cs = ConfigurationSpace({
        ...     "a": (5, 15),
        ...     "b": (0, 10),
        ...     "c": (0.0, 1.0)
        ... })
        >>> less_cond = LessThanCondition(cs['c'], cs['a'], 10)
        >>> greater_cond = GreaterThanCondition(cs['c'], cs['b'], 5)
        >>> cs.add_condition(AndConjunction(less_cond, greater_cond))
        (c | a < 10 && c | b > 5)

        Parameters
        ----------
        *args : :ref:`Conditions`
            conditions, which will be combined with an *AndConjunction*
        """
        if len(args) < 2:
            raise ValueError("AndConjunction must at least have two Conditions.")
        super().__init__(*args)

    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    def _evaluate_vector(self, instantiated_vector: np.ndarray) -> bool:
        return all(c._evaluate_vector(instantiated_vector) for c in self.components)

    def _evaluate(self, evaluations: np.ndarray) -> bool:
        return bool(evaluations.all())


class OrConjunction(AbstractConjunction):
    def __init__(self, *args: AbstractCondition) -> None:
        """Similar to the *AndConjunction*, constraints can be combined by
        using the *OrConjunction*.

        >>> from ConfigSpace import (
        ...     ConfigurationSpace,
        ...     LessThanCondition,
        ...     GreaterThanCondition,
        ...     OrConjunction
        ... )
        >>>
        >>> cs = ConfigurationSpace({
        ...     "a": (5, 15),
        ...     "b": (0, 10),
        ...     "c": (0.0, 1.0)
        ... })
        >>> less_cond = LessThanCondition(cs['c'], cs['a'], 10)
        >>> greater_cond = GreaterThanCondition(cs['c'], cs['b'], 5)
        >>> cs.add_condition(OrConjunction(less_cond, greater_cond))
        (c | a < 10 || c | b > 5)

        Parameters
        ----------
        *args : :ref:`Conditions`
            conditions, which will be combined with an *OrConjunction*
        """
        if len(args) < 2:
            raise ValueError("OrConjunction must at least have two Conditions.")
        super().__init__(*args)

    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" || ")
        retval.write(")")
        return retval.getvalue()

    def _evaluate(self, evaluations) -> bool:
        return any(evaluations)

    def _evaluate_vector(self, instantiated_vector: np.ndarray) -> bool:
        return any(c._evaluate_vector(instantiated_vector) for c in self.components)
