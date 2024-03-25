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
# TODO: Apparently there's sort of 'value' expected on the subclasses.
# * `evaluate()` relies on subclass `evaluate()`.
# * Remove legality checks from runtime functions ..., conditions shouldn't
#  be validating this so much, expensive...
# * `InCondition` is weird due to it's use of `values` and not `value`.
# * Using iterators where possible instead of lists might save a lot of time...
# * children and parent iteration can be pre-computed ...
# * Can we lift the vector indices out of the conditionals?
# * Just pass the raw value to the conditionals themselves.
# * See if we can pass in other conjunctions to conjunctions?
# * Fixup the old usage of AbstractX
from __future__ import annotations

import copy
import io
import operator
from abc import abstractmethod
from collections.abc import Sequence
from itertools import combinations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterator,
    TypeVar,
)
from typing_extensions import Self

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


class _NotSet:
    def __repr__(self):
        return "ValueNotSetObject"


NotSet = _NotSet()  # Sentinal value for unset values


DP = TypeVar("DP", bound=np.number)
DC = TypeVar("DC", bound=np.number)
"""Type variable user data type for the parent and child."""

VP = TypeVar("VP", bound=np.number)
VC = TypeVar("VC", bound=np.number)
"""Type variable vectorized data type for the parent and child."""


# TODO: Used to signify joint operations between condition conjuctions and
# but also singlure conditionals
# Might be able to just unify into one but keep serpeate for now
class Condition(Generic[DC, VC, DP, VP]):
    def __init__(
        self,
        child: Hyperparameter[DC, VC],
        parent: Hyperparameter[DP, VP],
        value: Any,
    ) -> None:
        if child == parent:
            raise ValueError(
                "The child and parent hyperparameter must be different "
                "hyperparameters.",
            )
        self.child = child
        self.parent = parent

        self.child_vector_id: np.int64 | None = None
        self.parent_vector_id: np.int64 | None = None

        self.value = value

    def set_vector_idx(self, hyperparameter_to_idx: dict):
        """Sets the index of the hyperparameter for the vectorized form.

        This is sort of a second-stage init that is called when a condition is
        added to the search space.
        """
        self.child_vector_id = np.int64(hyperparameter_to_idx[self.child.name])
        self.parent_vector_id = np.int64(hyperparameter_to_idx[self.parent.name])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self.child != other.child or self.parent != other.parent:
            return False

        return self.value == other.value

    def conditionally_equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return self.parent == other.parent and self.value == other.value

    @abstractmethod
    def satisfied_by_value(
        self,
        instantiated_parent_hyperparameter: dict[str, Any],
    ) -> bool:
        pass

    @abstractmethod
    def satisfied_by_vector(
        self,
        instantiated_parent_hyperparameter: (
            Sequence[np.number] | npt.NDArray[np.number]
        ),
    ) -> bool:
        pass

    @abstractmethod
    def satisfied_by_vector_array(
        self,
        arr: npt.NDArray[np.number],
    ) -> npt.NDArray[np.bool_]:
        pass


class _BinaryOpCondition(Condition[DC, VC, DP, VP]):
    _op_str: ClassVar[str]
    _requires_orderable_parent: ClassVar[bool]
    _op: Callable[[Any, Any], bool]
    _vector_op: Callable[[npt.NDArray[np.number], np.float64], npt.NDArray[np.bool_]]

    def __init__(
        self,
        child: Hyperparameter[DC, VC],
        parent: Hyperparameter[DP, VP],
        value: Any,  # HACK: Typing here is to allow in conditional
        *,
        check_condition_legality: bool = True,
    ) -> None:
        super().__init__(child, parent, value)

        if self._requires_orderable_parent and not parent.orderable:
            _clsname = self.__class__.__name__
            raise ValueError(
                f"The parent hyperparameter must be orderable to use "
                f"{_clsname}, however {self.parent} is not.",
            )
        if check_condition_legality and not parent.legal_value(value):
            raise ValueError(
                f"Hyperparameter '{child.name}' is "
                f"conditional on the illegal value '{value}' of "
                f"its parent hyperparameter '{parent.name}'",
            )

        # TODO: If we can change this to just vector
        self.vector_value = np.float64(self.parent.to_vector(value))

    def __repr__(self) -> str:
        return f"{self.child.name} | {self.parent.name} {self._op_str} {self.value!r}"

    # TODO: Dataclassing would make this obsolete.
    def __copy__(self) -> Self:
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def satisfied_by_vector(
        self,
        instantiated_parent_hyperparameter: (
            Sequence[np.number] | npt.NDArray[np.number]
        ),
    ) -> bool:
        vector = instantiated_parent_hyperparameter[self.parent_vector_id]
        val = self._op(vector, self.vector_value)
        return bool(val)

    def satisfied_by_vector_array(
        self,
        arr: npt.NDArray[np.number],
    ) -> npt.NDArray[np.bool_]:
        vector = arr[:, self.parent_vector_id]
        return self._vector_op(vector, self.vector_value)

    def satisfied_by_value(
        self,
        instantiated_parent_hyperparameter: dict[str, Any],
    ) -> bool:
        value = instantiated_parent_hyperparameter[self.parent.name]
        if value is NotSet:
            return False

        # TODO: This can be sped up by using the value in some cases but it's
        # likely only a marginal gain.
        vector = self.parent.to_vector(value)
        return bool(self._op(vector, self.vector_value))


class EqualsCondition(_BinaryOpCondition[DC, VC, DP, VP]):
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

    _op_str = "=="
    _requires_orderable_parent = False
    _op = operator.eq
    _vector_op = np.equal


class NotEqualsCondition(_BinaryOpCondition[DC, VC, DP, VP]):
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

    _op_str = "!="
    _requires_orderable_parent = False
    _op = operator.ne
    _vector_op = np.not_equal


class LessThanCondition(_BinaryOpCondition[DC, VC, DP, VP]):
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

    _op_str = "<"
    _requires_orderable_parent = True
    _op = operator.lt
    _vector_op = np.less


class GreaterThanCondition(_BinaryOpCondition[DC, VC, DP, VP]):
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

    _op_str = ">"
    _requires_orderable_parent = True
    _op = operator.gt
    _vector_op = np.greater


class InCondition(Condition[DC, VC, DP, VP]):
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

    def __init__(
        self,
        child: Hyperparameter[DC, VC],
        parent: Hyperparameter[DP, VP],
        values: list[DP],
    ) -> None:
        super().__init__(child, parent, values)
        for value in values:
            if not parent.legal_value(value):
                raise ValueError(
                    f"Hyperparameter '{child.name}' is "
                    f"conditional on the illegal value '{value}' of "
                    f"its parent hyperparameter '{parent.name}'",
                )

        self.values = values
        self.vector_values = [self.parent.to_vector(value) for value in self.values]

    def __repr__(self) -> str:
        return "{} | {} in {{{}}}".format(
            self.child.name,
            self.parent.name,
            ", ".join([repr(value) for value in self.values]),
        )

    def __copy__(self) -> Self:
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            values=copy.copy(self.values),
        )

    def satisfied_by_vector(
        self,
        instantiated_parent_hyperparameter: (
            Sequence[np.number] | npt.NDArray[np.number]
        ),
    ) -> bool:
        vector = instantiated_parent_hyperparameter[self.parent_vector_id]
        return bool(vector in self.vector_values)

    def satisfied_by_vector_array(
        self,
        arr: npt.NDArray[np.number],
    ) -> npt.NDArray[np.bool_]:
        vector = arr[:, self.parent_vector_id]
        return np.isin(vector, self.vector_values)

    def satisfied_by_value(
        self,
        instantiated_parent_hyperparameter: dict[str, Any],
    ) -> bool:
        value = instantiated_parent_hyperparameter[self.parent.name]
        if value is NotSet:
            return False
        return bool(value in self.values)


class Conjunction:
    def __init__(self, *args: Condition | Conjunction) -> None:
        self.components = args
        self.n_components = len(self.components)
        self.dlcs = self.get_descendant_literal_conditions()

        # Test the classes
        for idx, component in enumerate(self.components):
            if not isinstance(component, Condition | Conjunction):
                raise TypeError(
                    "Argument #%d is not an instance of Condition or Conjunction, "
                    "but %s" % (idx, type(component)),
                )

        # Test that all conjunctions and conditions have the same child!
        children = self.get_children()
        for c1, c2 in combinations(children, 2):
            if c1 != c2:
                raise ValueError(
                    "All Conjunctions and Conditions must have the same child.",
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

    def conditionally_equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        this_dlcs = self.get_descendant_literal_conditions()
        other_dlcs = other.get_descendant_literal_conditions()

        if len(this_dlcs) != len(other_dlcs):
            return False

        return all(
            c.conditionally_equal(oc)
            for c, oc in zip(this_dlcs, other_dlcs, strict=True)
        )

    def __copy__(self):
        return self.__class__(*[copy.copy(comp) for comp in self.components])

    def get_descendant_literal_conditions(self) -> tuple[Condition, ...]:
        children = []
        for component in self.components:
            if isinstance(component, Conjunction):
                children.extend(component.get_descendant_literal_conditions())
            else:
                children.append(component)
        return tuple(children)

    def iter(self) -> Iterator[Condition | Conjunction]:
        for component in self.components:
            yield component
            if isinstance(component, Conjunction):
                yield from component.iter()

    def set_vector_idx(self, hyperparameter_to_idx: dict):
        for component in self.components:
            component.set_vector_idx(hyperparameter_to_idx)

    def get_children_vector(self) -> list[int]:
        return [c.child_vector_id for c in self.get_descendant_literal_conditions()]

    def get_parents_vector(self) -> list[int]:
        return [c.parent_vector_id for c in self.get_descendant_literal_conditions()]

    def get_children(self) -> list[Hyperparameter]:
        return [c.child for c in self.iter() if isinstance(c, Condition)]

    def get_parents(self) -> list[Hyperparameter]:
        return [c.parent for c in self.iter() if isinstance(c, Condition)]

    @abstractmethod
    def satisfied_by_value(self, instantiated_hyperparameters: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def satisfied_by_vector(self, instantiated_vector: np.ndarray) -> bool:
        pass

    @abstractmethod
    def satisfied_by_vector_array(
        self,
        arr: npt.NDArray[np.number],
    ) -> npt.NDArray[np.bool_]:
        pass


class AndConjunction(Conjunction):
    # TODO: test if an AndConjunction results in an illegal state or a
    #       Tautology! -> SAT solver
    def __init__(self, *args: Condition | Conjunction) -> None:
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

    def satisfied_by_value(self, instantiated_hyperparameters: dict[str, Any]) -> bool:
        # Then, check if all parents were passed
        # TODO: Speed this up
        conditions = self.dlcs
        for condition in conditions:
            if condition.parent.name not in instantiated_hyperparameters:
                raise ValueError(
                    "Evaluate must be called with all "
                    "instanstatiated parent hyperparameters in "
                    "the conjunction; you are (at least) missing "
                    "'%s'" % condition.parent.name,
                )

        for c in self.components:
            if not c.satisfied_by_value(instantiated_hyperparameters):
                return False

        return True

    def satisfied_by_vector(self, instantiated_vector: np.ndarray) -> bool:
        for c in self.components:  # noqa: SIM110
            if not c.satisfied_by_vector(instantiated_vector):
                return False

        return True

    def satisfied_by_vector_array(
        self,
        arr: npt.NDArray[np.number],
    ) -> npt.NDArray[np.bool_]:
        satisfied = np.ones(arr.shape[0], dtype=np.bool_)
        for c in self.components:
            satisfied &= c.satisfied_by_vector_array(arr)

        return satisfied


class OrConjunction(Conjunction):
    def __init__(self, *args: Condition | Conjunction) -> None:
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

    def satisfied_by_value(self, instantiated_hyperparameters: dict[str, Any]) -> bool:
        # Then, check if all parents were passed
        # TODO: Speed this up
        conditions = self.dlcs
        for condition in conditions:
            if condition.parent.name not in instantiated_hyperparameters:
                raise ValueError(
                    "Evaluate must be called with all "
                    "instanstatiated parent hyperparameters in "
                    "the conjunction; you are (at least) missing "
                    "'%s'" % condition.parent.name,
                )

        for c in self.components:
            if c.satisfied_by_value(instantiated_hyperparameters):
                return True

        return False

    def satisfied_by_vector(self, instantiated_vector: np.ndarray) -> bool:
        for c in self.components:  # noqa: SIM110
            if c.satisfied_by_vector(instantiated_vector):
                return True

        return False

    def satisfied_by_vector_array(
        self,
        arr: npt.NDArray[np.number],
    ) -> npt.NDArray[np.bool_]:
        satisfied = np.ones(arr.shape[0], dtype=np.bool_)
        for c in self.components:
            satisfied |= c.satisfied_by_vector_array(arr)

        return satisfied


# Backwards compatibility
AbstractCondition = Condition
AbstractConjunction = Conjunction
