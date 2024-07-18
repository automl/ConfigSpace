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
import operator
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Union
from typing_extensions import Self, deprecated, override

import numpy as np
from more_itertools import all_equal, unique_everseen

from ConfigSpace.types import NotSet, f64

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter
    from ConfigSpace.types import Array, Mask


class Condition(ABC):
    def __init__(
        self,
        child: Hyperparameter,
        parent: Hyperparameter,
        value: Any,
    ) -> None:
        if child == parent:
            raise ValueError(
                "The child and parent hyperparameter must be different "
                "hyperparameters.",
            )
        self.child = child
        self.parent = parent

        self.child_vector_id: np.intp | None = None
        self.parent_vector_id: np.intp | None = None

        self.value = value

    def set_vector_idx(self, hyperparameter_to_idx: Mapping[str, int]) -> None:
        """Sets the index of the hyperparameter for the vectorized form.

        This is sort of a second-stage init that is called when a condition is
        added to the search space.
        """
        self.child_vector_id = np.intp(hyperparameter_to_idx[self.child.name])
        self.parent_vector_id = np.intp(hyperparameter_to_idx[self.parent.name])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self.child != other.child or self.parent != other.parent:
            return False

        return self.value == other.value  # type: ignore

    def equivalent_condition_on_parent(self, other: ConditionLike) -> bool:
        if isinstance(other, self.__class__):
            return self.parent == other.parent and self.value == other.value

        return False

    @abstractmethod
    def satisfied_by_value(self, values: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def satisfied_by_vector(self, vector: Array[f64]) -> bool:
        pass

    @abstractmethod
    def satisfied_by_vector_array(self, arr: Array[f64]) -> Mask:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


class BinaryOpCondition(Condition):
    """This is a base class for conditions that are based on a binary operations."""

    _OP_STR: ClassVar[str]
    _REQUIRES_ORDERABLE_PARENT: ClassVar[bool]
    _OP: ClassVar[Callable[[Any, Any], bool]]
    _VECTOR_OP: ClassVar[Callable[..., Mask]]

    def __init__(
        self,
        child: Hyperparameter,
        parent: Hyperparameter,
        value: Any,  # HACK: Typing here is to allow in conditional
        *,
        check_condition_legality: bool = True,
    ) -> None:
        super().__init__(child, parent, value)

        if self._REQUIRES_ORDERABLE_PARENT and not parent.ORDERABLE:
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

        self.vector_value = f64(self.parent.to_vector(value))

        # HACK: For now, the only kind of hyperparameter that can **not** be ordered
        # as a value type but can be ordered as a vector type is an
        # OrdinalHyperparameter. This is an explicit hack for that, but if the
        # needs arise, we can make this more generic
        from ConfigSpace.hyperparameters.ordinal import OrdinalHyperparameter

        self.need_compare_as_vector = isinstance(self.parent, OrdinalHyperparameter)

    def __repr__(self) -> str:
        return f"{self.child.name} | {self.parent.name} {self._OP_STR} {self.value!r}"

    def __copy__(self) -> Self:
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def satisfied_by_vector(self, vector: Array[f64]) -> bool:
        parent_value = vector[self.parent_vector_id]
        return self._OP(parent_value, self.vector_value)  # type: ignore

    def satisfied_by_vector_array(self, arr: Array[f64]) -> Mask:
        vector = arr[self.parent_vector_id]
        return self._VECTOR_OP(vector, self.vector_value)

    def satisfied_by_value(self, values: dict[str, Any]) -> bool:
        value = values[self.parent.name]
        if value is NotSet:
            return False

        if not self.need_compare_as_vector:
            return bool(self._OP(value, self.value))  # type: ignore

        vector_value = self.parent.to_vector(value)
        return bool(self._VECTOR_OP(vector_value, self.vector_value))

    @override
    def to_dict(self) -> dict[str, Any]:
        return {
            "child": self.child.name,
            "parent": self.parent.name,
            "value": self.value,
        }


class EqualsCondition(BinaryOpCondition):
    """Hyperparameter `child` is conditional on the `parent` hyperparameter
    being *equal* to `value`.

    Make *b* an active hyperparameter if *a* has the value 1

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace import ConfigurationSpace, EqualsCondition
    cs = ConfigurationSpace({ "a": [1, 2, 3], "b": (1.0, 8.0)})
    cond = EqualsCondition(cs['b'], cs['a'], 1)
    cs.add(cond)
    print(cs)
    ```

    Args:
        child: This hyperparameter will be sampled in the configspace
            if the *equal condition* is satisfied
        parent: The hyperparameter, which has to satisfy the *equal condition*
        value: Value, which the parent is compared to
    """

    _OP_STR = "=="
    _REQUIRES_ORDERABLE_PARENT = False
    _OP = operator.eq
    _VECTOR_OP = operator.eq


class NotEqualsCondition(BinaryOpCondition):
    """Hyperparameter `child` is conditional on the `parent` hyperparameter
    being *not equal* to `value`.

    Make *b* an active hyperparameter if *a* has **not** the value 1

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace import ConfigurationSpace, NotEqualsCondition

    cs = ConfigurationSpace({ "a": [1, 2, 3], "b": (1.0, 8.0) })
    cond = NotEqualsCondition(cs['b'], cs['a'], 1)
    cs.add(cond)
    print(cs)
    ```

    Args:
        child:
            This hyperparameter will be sampled in the configspace
            if the not-equals condition is satisfied
        parent: The hyperparameter, which has to satisfy the *not equal condition*
        value: Value, which the parent is compared to
    """

    _OP_STR = "!="
    _REQUIRES_ORDERABLE_PARENT = False
    _OP = operator.ne
    _VECTOR_OP = operator.ne


class LessThanCondition(BinaryOpCondition):
    """Hyperparameter `child` is conditional on the `parent` hyperparameter
    being *less than* `value`.

    Make *b* an active hyperparameter if *a* is less than 5

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace import ConfigurationSpace, LessThanCondition

    cs = ConfigurationSpace({ "a": (0, 10), "b": (1.0, 8.0) })
    cond = LessThanCondition(cs['b'], cs['a'], 5)
    cs.add(cond)
    print(cs)
    ```

    Args:
        child:
            This hyperparameter will be sampled in the configspace,
            if the *LessThanCondition* is satisfied
        parent: The hyperparameter, which has to satisfy the *LessThanCondition*
        value: Value, which the parent is compared to
    """

    _OP_STR = "<"
    _REQUIRES_ORDERABLE_PARENT = True
    _OP = operator.lt
    _VECTOR_OP = operator.lt


class GreaterThanCondition(BinaryOpCondition):
    """Hyperparameter `child` is conditional on the `parent` hyperparameter
    being *greater than* `value`.

    Make *b* an active hyperparameter if *a* is greater than 5

    from ConfigSpace import ConfigurationSpace, GreaterThanCondition

    cs = ConfigurationSpace({ "a": (0, 10), "b": (1.0, 8.0) })
    cond = GreaterThanCondition(cs['b'], cs['a'], 5)
    cs.add(cond)
    print(cs)
    ```

    Args:
        child:
            This hyperparameter will be sampled in the configspace,
            if the *GreaterThanCondition* is satisfied
        parent: The hyperparameter, which has to satisfy the *GreaterThanCondition*
        value: Value, which the parent is compared to
    """

    _OP_STR = ">"
    _REQUIRES_ORDERABLE_PARENT = True
    _OP = operator.gt
    _VECTOR_OP = operator.gt


class InCondition(Condition):
    """Hyperparameter `child` is conditional on the `parent` hyperparameter
    being *in* a set of `values`.

    make *b* an active hyperparameter if *a* is in the set [1, 2, 3, 4]

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace import ConfigurationSpace, InCondition
    cs = ConfigurationSpace({ "a": (0, 10), "b": (1.0, 8.0) })
    cond = InCondition(cs['b'], cs['a'], [1, 2, 3, 4])
    cs.add(cond)
    print(cs)
    ```

    Args:
        child:
            This hyperparameter will be sampled in the configspace,
            if the *InCondition* is satisfied
        parent: The hyperparameter, which has to satisfy the *InCondition*
        values: Collection of values, which the parent is compared to
    """

    def __init__(
        self,
        child: Hyperparameter,
        parent: Hyperparameter,
        values: list[Any],
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

    def satisfied_by_vector(self, vector: Array[f64]) -> bool:
        return vector[self.parent_vector_id] in self.vector_values

    def satisfied_by_vector_array(self, arr: Array[f64]) -> Mask:
        vector = arr[self.parent_vector_id]
        return np.isin(vector, self.vector_values)

    def satisfied_by_value(self, values: dict[str, Any]) -> bool:
        value = values[self.parent.name]
        if value is NotSet:
            return False
        return bool(value in self.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "child": self.child.name,
            "parent": self.parent.name,
            "values": self.values,
        }


class Conjunction:
    def __init__(self, *args: Condition | Conjunction) -> None:
        # Test the classes
        for idx, component in enumerate(args):
            if not isinstance(component, (Condition, Conjunction)):
                raise TypeError(
                    "Argument #%d is not an instance of Condition or Conjunction, "
                    "but %s" % (idx, type(component)),
                )

        self.components = args
        self.n_components = len(self.components)

        dlcs: list[Condition] = []
        for component in self.components:
            if isinstance(component, Conjunction):
                dlcs.extend(component.dlcs)
            else:
                dlcs.append(component)

        self.dlcs: tuple[Condition, ...] = tuple(
            unique_everseen(dlcs, key=lambda x: id(x)),
        )

        # Test that all conjunctions and conditions have the same child!
        children = [dlc.child for dlc in self.dlcs]
        if not all_equal(children):
            raise ValueError("All Conjunctions, Conditions must have the same child.")

        self.child = children[0]
        self.child_vector_id: np.intp | None = None
        self.parent_vector_ids: Array[np.intp] | None = None
        self.parents = [c.parent for c in self.dlcs]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if len(self.components) != len(other.components):
            return False

        # We need to check that all of this components exist in the other
        # and vice versa...
        return all(c in other.components for c in self.components) and all(
            oc in self.components for oc in other.components
        )

    def __copy__(self) -> Self:
        return self.__class__(*[copy.copy(comp) for comp in self.components])

    @deprecated("Please just use `.dlcs` instead.")
    def get_descendant_literal_conditions(self) -> tuple[Condition, ...]:
        return self.dlcs

    def set_vector_idx(self, hyperparameter_to_idx: Mapping[str, int]) -> None:
        for component in self.components:
            component.set_vector_idx(hyperparameter_to_idx)
        self.child_vector_id = np.intp(hyperparameter_to_idx[self.child.name])
        self.parent_vector_ids = np.array(
            [c.parent_vector_id for c in self.dlcs],
            dtype=np.intp,
        )

    @deprecated(
        "Conjunctions of conditions can only every have one child."
        " Please use .child_vector_id instead.",
    )
    def get_children_vector(self) -> Array[np.intp]:
        return np.array([self.child_vector_id], dtype=np.intp)

    @deprecated("Please use .parent_vector_ids instead.")
    def get_parents_vector(self) -> Array[np.intp]:
        return self.parent_vector_ids  # type: ignore

    @deprecated(
        "Conjunctions of conditions can only every have one child."
        " Please use .child instead.",
    )
    def get_children(self) -> list[Hyperparameter]:
        return [self.child]

    @deprecated("Please use .parents instead.")
    def get_parents(self) -> list[Hyperparameter]:
        return self.parents

    def equivalent_condition_on_parent(self, other: ConditionLike) -> bool:
        # Not entirely true but it's a good enough approximation
        if not isinstance(other, Conjunction):
            return False

        if len(self.components) != len(other.components):
            return False

        # Each condition must match at one of the other conditions, which is of the same
        # conjunction type and has equal amount of components
        return all(
            any(c.equivalent_condition_on_parent(o) for o in other.components)
            for c in self.components
        )

    @abstractmethod
    def satisfied_by_value(self, values: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def satisfied_by_vector(self, vector: np.ndarray) -> bool:
        pass

    @abstractmethod
    def satisfied_by_vector_array(self, arr: Array[f64]) -> Mask:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


class AndConjunction(Conjunction):
    # TODO: test if an AndConjunction results in an illegal state or a
    #       Tautology! -> SAT solver
    def __init__(self, *args: Condition | Conjunction) -> None:
        """By using the *AndConjunction*, constraints can easily be connected.

        The following example shows how two constraints with an *AndConjunction*
        can be combined.

        ```python exec="true", source="material-block" result="python"
        from ConfigSpace import (
            ConfigurationSpace,
            LessThanCondition,
            GreaterThanCondition,
            AndConjunction
        )

        cs = ConfigurationSpace({ "a": (5, 15), "b": (0, 10), "c": (0.0, 1.0) })

        less_cond = LessThanCondition(cs['c'], cs['a'], 10)
        greater_cond = GreaterThanCondition(cs['c'], cs['b'], 5)
        cs.add(AndConjunction(less_cond, greater_cond))

        print(cs)
        ```

        Args:
            *args: conditions, which will be combined with an *AndConjunction*
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

    def satisfied_by_value(self, values: dict[str, Any]) -> bool:
        return all(c.satisfied_by_value(values) for c in self.components)

    def satisfied_by_vector(self, vector: Array[f64]) -> bool:
        for c in self.components:  # noqa: SIM110
            if not c.satisfied_by_vector(vector):
                return False

        return True

    def satisfied_by_vector_array(self, arr: Array[f64]) -> Mask:
        satisfied: Mask = np.ones(arr.shape[1], dtype=np.bool_)
        for c in self.components:
            satisfied &= c.satisfied_by_vector_array(arr)

        return satisfied

    def to_dict(self) -> dict[str, Any]:
        return {
            "child": self.child.name,
            "conditions": [component.to_dict() for component in self.components],
        }


class OrConjunction(Conjunction):
    """Similar to the [AndConjunction][ConfigSpace.conditions.AndConjunction],
    constraints can be combined by using the *OrConjunction*.

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace import (
        ConfigurationSpace,
        LessThanCondition,
        GreaterThanCondition,
        OrConjunction
    )

    cs = ConfigurationSpace({ "a": (5, 15), "b": (0, 10), "c": (0.0, 1.0) })

    less_cond = LessThanCondition(cs['c'], cs['a'], 10)
    greater_cond = GreaterThanCondition(cs['c'], cs['b'], 5)
    cs.add(OrConjunction(less_cond, greater_cond))

    print(cs)
    ```
    """

    def __init__(self, *args: ConditionLike) -> None:
        """Initialize the *OrConjunction*.

        Args:
            *args: conditions, which will be combined with an *OrConjunction*.
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

    def satisfied_by_value(self, values: dict[str, Any]) -> bool:
        return any(c.satisfied_by_value(values) for c in self.components)

    def satisfied_by_vector(self, vector: Array[f64]) -> bool:
        return any(c.satisfied_by_vector(vector) for c in self.components)

    def satisfied_by_vector_array(self, arr: Array[f64]) -> Mask:
        satisfied: Mask = np.zeros(arr.shape[1], dtype=np.bool_)
        for c in self.components:
            satisfied |= c.satisfied_by_vector_array(arr)

        return satisfied

    def to_dict(self) -> dict[str, Any]:
        return {
            "child": self.child.name,
            "conditions": [component.to_dict() for component in self.components],
        }


# Backwards compatibility
AbstractCondition = Condition
AbstractConjunction = Conjunction

ConditionLike = Union[Condition, Conjunction]
