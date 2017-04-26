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

from abc import ABCMeta, abstractmethod
from itertools import combinations
from typing import Any, List, Union
import operator

import numpy as np

import io
from functools import reduce
from ConfigSpace.hyperparameters import Hyperparameter, \
    NumericalHyperparameter, OrdinalHyperparameter


class ConditionComponent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
    @abstractmethod
    def set_vector_idx(self, hyperparameter_to_idx) -> None:
        pass

    @abstractmethod
    def get_children_vector(self) -> List[int]:
        pass

    @abstractmethod
    def get_parents_vector(self) -> List[int]:
        pass

    @abstractmethod
    def get_children(self) -> List['ConditionComponent']:
        pass

    @abstractmethod
    def get_parents(self) -> List['ConditionComponent']:
        pass

    @abstractmethod
    def get_descendant_literal_conditions(self) ->List['AbstractCondition']:
        pass

    @abstractmethod
    def evaluate(self, instantiated_parent_hyperparameter: Hyperparameter) -> bool:
        pass

    # http://stackoverflow.com/a/25176504/4636294
    def __eq__(self, other: Any) -> bool:
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))


class AbstractCondition(ConditionComponent):
    # TODO create a condition evaluator!

    @abstractmethod
    def __init__(self, child: Hyperparameter, parent: Hyperparameter) -> None:
        if not isinstance(child, Hyperparameter):
            raise ValueError("Argument 'child' is not an instance of "
                             "HPOlibConfigSpace.hyperparameter.Hyperparameter.")
        if not isinstance(parent, Hyperparameter):
            raise ValueError("Argument 'parent' is not an instance of "
                             "HPOlibConfigSpace.hyperparameter.Hyperparameter.")
        if child == parent:
            raise ValueError("The child and parent hyperparameter must be "
                             "different hyperparameters.")
        self.child = child
        self.parent = parent
        self.child_vector_id = None
        self.parent_vector_id = None

    def set_vector_idx(self, hyperparameter_to_idx):
        self.child_vector_id = hyperparameter_to_idx[self.child.name]
        self.parent_vector_id = hyperparameter_to_idx[self.parent.name]

    def get_children_vector(self) -> List[int]:
        return [self.child_vector_id]

    def get_parents_vector(self) -> List[int]:
        return [self.parent_vector_id]

    def get_children(self) -> List[Hyperparameter]:
        return [self.child]

    def get_parents(self) -> List[Hyperparameter]:
        return [self.parent]

    def get_descendant_literal_conditions(self) -> List['AbstractCondition']:
        return [self]

    def evaluate(self, instantiated_parent_hyperparameter: Hyperparameter) -> bool:
        hp_name = self.parent.name
        return self._evaluate(instantiated_parent_hyperparameter[hp_name])

    def evaluate_vector(self, instantiated_vector: np.ndarray) -> bool:
        if self.parent_vector_id is None:
            raise ValueError("Parent vector id should not be None when calling evaluate vector")
        return self._evaluate_vector(instantiated_vector[self.parent_vector_id])

    @abstractmethod
    def _evaluate(self, instantiated_parent_hyperparameter: Union[str, int, float]) -> bool:
        pass


class AbstractConjunction(ConditionComponent):
    def __init__(self, *args: AbstractCondition) -> None:
        super(AbstractConjunction, self).__init__()
        self.components = args

        # Test the classes
        for idx, component in enumerate(self.components):
            if not isinstance(component, ConditionComponent):
                raise TypeError("Argument #%d is not an instance of %s, "
                                "but %s" % (
                                    idx, ConditionComponent, type(component)))

        # Test that all conjunctions and conditions have the same child!
        children = self.get_children()
        for c1, c2 in combinations(children, 2):
            if c1 != c2:
                raise ValueError("All Conjunctions and Conditions must have "
                                 "the same child.")

    def get_descendant_literal_conditions(self) -> List[AbstractCondition]:
        children = []  # type: List[AbstractCondition]
        for component in self.components:
            if isinstance(component, AbstractConjunction):
                children.extend(component.get_descendant_literal_conditions())
            else:
                children.append(component)
        return children

    def set_vector_idx(self, hyperparameter_to_idx):
        for component in self.components:
            component.set_vector_idx(hyperparameter_to_idx)

    def get_children_vector(self) -> List[int]:
        children_vector = []
        for component in self.components:
            children_vector.extend(component.get_children_vector())
        return children_vector

    def get_parents_vector(self) -> List[int]:
        parents_vector = []
        for component in self.components:
            parents_vector.extend(component.get_parents_vector())
        return parents_vector

    def get_children(self) -> List[ConditionComponent]:
        children = []  # type: List[ConditionComponent]
        for component in self.components:
            children.extend(component.get_children())
        return children

    def get_parents(self) -> List[ConditionComponent]:
        parents = []  # type: List[ConditionComponent]
        for component in self.components:
            parents.extend(component.get_parents())
        return parents

    def evaluate(self, instantiated_hyperparameters: Hyperparameter) -> bool:
        # Then, check if all parents were passed
        conditions = self.get_descendant_literal_conditions()
        for condition in conditions:
            if condition.parent.name not in instantiated_hyperparameters:
                raise ValueError("Evaluate must be called with all "
                                 "instanstatiated parent hyperparameters in "
                                 "the conjunction; you are (at least) missing "
                                 "'%s'" % condition.parent.name)

        # Finally, call evaluate for all direct descendents and combine the
        # outcomes
        evaluations = []
        for component in self.components:
            e = component.evaluate(instantiated_hyperparameters)
            evaluations.append(e)

        return self._evaluate(evaluations)

    def evaluate_vector(self, instantiated_vector: np.ndarray) -> bool:
        # Then, check if all parents were passed
        conditions = self.get_descendant_literal_conditions()
        for condition in conditions:
            if condition.parent_vector_id is None:
                raise ValueError("Parent vector id should not be None when calling evaluate vector")

        # Finally, call evaluate for all direct descendents and combine the
        # outcomes
        evaluations = []
        for component in self.components:
            e = component.evaluate_vector(instantiated_vector)
            evaluations.append(e)

        return self._evaluate(evaluations)

    @abstractmethod
    def _evaluate(self, evaluations: List[bool]) -> bool:
        pass


class EqualsCondition(AbstractCondition):
    def __init__(self, child: Hyperparameter, parent: Hyperparameter, value: Union[str, float, int]) -> None:
        super(EqualsCondition, self).__init__(child, parent)
        if not parent.is_legal(value):
            raise ValueError("Hyperparameter '%s' is "
                             "conditional on the illegal value '%s' of "
                             "its parent hyperparameter '%s'" %
                             (child.name, value, parent.name))
        self.value = value
        self.vector_value = self.parent._inverse_transform(self.value)

    def __repr__(self) -> str:
        return "%s | %s == %s" % (self.child.name, self.parent.name,
                                  repr(self.value))

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal(value):
            return False

        cmp = self.parent.compare(value, self.value)
        if cmp == 0:
            return True
        else:
            return False

    def _evaluate_vector(self, value: Union[float, int]) -> bool:
        if not self.parent.is_legal_vector(value):
            return False

        cmp = self.parent.compare_vector(value, self.vector_value)
        if cmp == 0:
            return True
        else:
            return False


class NotEqualsCondition(AbstractCondition):
    def __init__(self, child: Hyperparameter, parent: Hyperparameter, value: Union[str, float, int]) -> None:
        super(NotEqualsCondition, self).__init__(child, parent)
        if not parent.is_legal(value):
            raise ValueError("Hyperparameter '%s' is "
                             "conditional on the illegal value '%s' of "
                             "its parent hyperparameter '%s'" %
                             (child.name, value, parent.name))
        self.value = value
        self.vector_value = self.parent._inverse_transform(self.value)

    def __repr__(self) -> str:
        return "%s | %s != %s" % (self.child.name, self.parent.name,
                                  repr(self.value))

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal(value):
            return False

        cmp = self.parent.compare(value, self.value)
        if cmp != 0:
            return True
        else:
            return False

    def _evaluate_vector(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal_vector(value):
            return False

        cmp = self.parent.compare_vector(value, self.vector_value)
        if cmp != 0:
            return True
        else:
            return False


class LessThanCondition(AbstractCondition):
    def __init__(self, child: Hyperparameter, parent: Hyperparameter, value: Union[str, float, int]) -> None:
        super(LessThanCondition, self).__init__(child, parent)
        self.parent.allow_greater_less_comparison()
        if not parent.is_legal(value):
            raise ValueError("Hyperparameter '%s' is "
                             "conditional on the illegal value '%s' of "
                             "its parent hyperparameter '%s'" %
                             (child.name, value, parent.name))
        self.value = value
        self.vector_value = self.parent._inverse_transform(self.value)

    def __repr__(self) -> str:
        return "%s | %s < %s" % (self.child.name, self.parent.name,
                                 repr(self.value))

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal(value):
            return False

        cmp = self.parent.compare(value, self.value)
        if cmp == -1:
            return True
        else:
            return False

    def _evaluate_vector(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal_vector(value):
            return False

        cmp = self.parent.compare_vector(value, self.vector_value)
        if cmp == -1:
            return True
        else:
            return False


class GreaterThanCondition(AbstractCondition):
    def __init__(self, child: Hyperparameter, parent: Hyperparameter, value: Union[str, float, int]) -> None:
        super(GreaterThanCondition, self).__init__(child, parent)
        self.parent.allow_greater_less_comparison()
        if not parent.is_legal(value):
            raise ValueError("Hyperparameter '%s' is "
                             "conditional on the illegal value '%s' of "
                             "its parent hyperparameter '%s'" %
                             (child.name, value, parent.name))
        self.value = value
        self.vector_value = self.parent._inverse_transform(self.value)

    def __repr__(self) -> str:
        return "%s | %s > %s" % (self.child.name, self.parent.name,
                                 repr(self.value))

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal(value):
            return False

        cmp = self.parent.compare(value, self.value)
        if cmp == 1:
            return True
        else:
            return False

    def _evaluate_vector(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal_vector(value):
            return False

        cmp = self.parent.compare_vector(value, self.vector_value)
        if cmp == 1:
            return True
        else:
            return False

class InCondition(AbstractCondition):
    def __init__(self, child: Hyperparameter, parent: Hyperparameter, values: List[Union[str, float, int]]) -> None:
        super(InCondition, self).__init__(child, parent)
        for value in values:
            if not parent.is_legal(value):
                raise ValueError("Hyperparameter '%s' is "
                                 "conditional on the illegal value '%s' of "
                                 "its parent hyperparameter '%s'" %
                                 (child.name, value, parent.name))
        self.values = values
        self.vector_values = [self.parent._inverse_transform(value) for value in self.values]

    def __repr__(self) -> str:
        return "%s | %s in {%s}" % (self.child.name, self.parent.name,
                                    ", ".join(
                                        [repr(value) for value in self.values]))

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        return value in self.values

    def _evaluate_vector(self, value: Union[float, int]) -> bool:
        return value in self.vector_values


class AndConjunction(AbstractConjunction):
    # TODO: test if an AndConjunction results in an illegal state or a
    # Tautology! -> SAT solver
    def __init__(self, *args: AbstractCondition) -> None:
        if len(args) < 2:
            raise ValueError("AndConjunction must at least have two "
                             "Conditions.")
        super(AndConjunction, self).__init__(*args)

    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    def _evaluate(self, evaluations: Any) -> bool:
        return reduce(operator.and_, evaluations)


class OrConjunction(AbstractConjunction):
    def __init__(self, *args: AbstractCondition) -> None:
        if len(args) < 2:
            raise ValueError("OrConjunction must at least have two "
                             "Conditions.")
        super(OrConjunction, self).__init__(*args)

    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" || ")
        retval.write(")")
        return retval.getvalue()

    def _evaluate(self, evaluations: Any) -> bool:
        return reduce(operator.or_, evaluations)
