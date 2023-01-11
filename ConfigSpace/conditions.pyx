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

import io
from functools import reduce
from abc import ABCMeta, abstractmethod
import copy
from itertools import combinations
from typing import Any, List, Union, Tuple, Dict, Optional
import operator

from libc.stdlib cimport malloc, free

import numpy as np

from ConfigSpace.hyperparameters cimport Hyperparameter

cimport numpy as np


cdef class ConditionComponent(object):

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        pass

    def set_vector_idx(self, hyperparameter_to_idx) -> None:
        pass

    def get_children_vector(self) -> List[int]:
        pass

    def get_parents_vector(self) -> List[int]:
        pass

    def get_children(self) -> List['ConditionComponent']:
        pass

    def get_parents(self) -> List['ConditionComponent']:
        pass

    def get_descendant_literal_conditions(self) ->List['AbstractCondition']:
        pass

    def evaluate(self,
                 instantiated_parent_hyperparameter: Dict[str, Union[None, int, float, str]]
                 ) -> bool:
        pass

    def evaluate_vector(self, instantiated_vector):
        return bool(self._evaluate_vector(instantiated_vector))

    cdef int _evaluate_vector(self, np.ndarray value):
        pass

    def __hash__(self) -> int:
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))


cdef class AbstractCondition(ConditionComponent):
    cdef public Hyperparameter child
    cdef public Hyperparameter parent
    cdef public int child_vector_id
    cdef public int parent_vector_id
    cdef public value
    cdef public DTYPE_t vector_value

    def __init__(self, child: Hyperparameter, parent: Hyperparameter) -> None:
        if child == parent:
            raise ValueError(
                "The child and parent hyperparameter must be different "
                "hyperparameters."
            )
        self.child = child
        self.parent = parent
        self.child_vector_id = -1
        self.parent_vector_id = -1

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
            return False

        if self.child != other.child:
            return False
        elif self.parent != other.parent:
            return False
        return self.value == other.value

    def set_vector_idx(self, hyperparameter_to_idx: dict):
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

    def evaluate(self, instantiated_parent_hyperparameter: Dict[str, Union[int, float, str]]
                 ) -> bool:
        hp_name = self.parent.name
        return self._evaluate(instantiated_parent_hyperparameter[hp_name])

    cdef int _evaluate_vector(self, np.ndarray instantiated_vector):
        if self.parent_vector_id is None:
            raise ValueError("Parent vector id should not be None when calling evaluate vector")
        return self._inner_evaluate_vector(instantiated_vector[self.parent_vector_id])

    def _evaluate(self, instantiated_parent_hyperparameter: Union[str, int, float]) -> bool:
        pass

    cdef int _inner_evaluate_vector(self, DTYPE_t value):
        pass


cdef class EqualsCondition(AbstractCondition):

    def __init__(self, child: Hyperparameter, parent: Hyperparameter,
                 value: Union[str, float, int]) -> None:
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

    def __copy__(self):
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        # No need to check if the value to compare is a legal value; either it
        # is equal (and thus legal), or it would evaluate to False anyway

        cmp = self.parent.compare(value, self.value)
        if cmp == 0:
            return True
        else:
            return False

    cdef int _inner_evaluate_vector(self, DTYPE_t value):
        # No need to check if the value to compare is a legal value; either it
        # is equal (and thus legal), or it would evaluate to False anyway

        cdef int cmp = self.parent.compare_vector(value, self.vector_value)
        if cmp == 0:
            return True
        else:
            return False


cdef class NotEqualsCondition(AbstractCondition):
    def __init__(self, child: Hyperparameter, parent: Hyperparameter,
                 value: Union[str, float, int]) -> None:
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

    def __copy__(self):
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal(value):
            return False

        cmp = self.parent.compare(value, self.value)
        if cmp != 0:
            return True
        else:
            return False

    cdef int _inner_evaluate_vector(self, DTYPE_t value):
        if not self.parent.is_legal_vector(value):
            return False

        cdef int cmp = self.parent.compare_vector(value, self.vector_value)
        if cmp != 0:
            return True
        else:
            return False


cdef class LessThanCondition(AbstractCondition):
    def __init__(self, child: Hyperparameter, parent: Hyperparameter,
                 value: Union[str, float, int]) -> None:
        """
        Hyperparameter ``child`` is conditional on the ``parent`` hyperparameter
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

    def __copy__(self):
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal(value):
            return False

        cmp = self.parent.compare(value, self.value)
        if cmp == -1:
            return True
        else:
            return False

    cdef int _inner_evaluate_vector(self, DTYPE_t value):
        if not self.parent.is_legal_vector(value):
            return False

        cdef int cmp = self.parent.compare_vector(value, self.vector_value)
        if cmp == -1:
            return True
        else:
            return False


cdef class GreaterThanCondition(AbstractCondition):
    def __init__(self, child: Hyperparameter, parent: Hyperparameter,
                 value: Union[str, float, int]) -> None:
        """
        Hyperparameter ``child`` is conditional on the ``parent`` hyperparameter
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

    def __copy__(self):
        return self.__class__(
            child=copy.copy(self.child),
            parent=copy.copy(self.parent),
            value=copy.copy(self.value),
        )

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        if not self.parent.is_legal(value):
            return False

        cmp = self.parent.compare(value, self.value)
        if cmp == 1:
            return True
        else:
            return False

    cdef int _inner_evaluate_vector(self, DTYPE_t value):
        if not self.parent.is_legal_vector(value):
            return False

        cdef int cmp = self.parent.compare_vector(value, self.vector_value)
        if cmp == 1:
            return True
        else:
            return False

cdef class InCondition(AbstractCondition):
    cdef public values
    cdef public vector_values

    def __init__(self, child: Hyperparameter, parent: Hyperparameter,
                 values: List[Union[str, float, int]]) -> None:
        """
        Hyperparameter ``child`` is conditional on the ``parent`` hyperparameter
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
        super(InCondition, self).__init__(child, parent)
        for value in values:
            if not parent.is_legal(value):
                raise ValueError("Hyperparameter '%s' is "
                                 "conditional on the illegal value '%s' of "
                                 "its parent hyperparameter '%s'" %
                                 (child.name, value, parent.name))
        self.values = values
        self.value = values
        self.vector_values = [self.parent._inverse_transform(value) for value in self.values]

    def __repr__(self) -> str:
        return "%s | %s in {%s}" % (self.child.name, self.parent.name,
                                    ", ".join(
                                        [repr(value) for value in self.values]))

    def _evaluate(self, value: Union[str, float, int]) -> bool:
        return value in self.values

    cdef int _inner_evaluate_vector(self, DTYPE_t value):
        return value in self.vector_values


cdef class AbstractConjunction(ConditionComponent):
    cdef public tuple components
    cdef int n_components
    cdef tuple dlcs

    def __init__(self, *args: AbstractCondition) -> None:
        super(AbstractConjunction, self).__init__()
        self.components = args
        self.n_components = len(self.components)
        self.dlcs = self.get_descendant_literal_conditions()

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
            return False

        if len(self.components) != len(other.components):
            return False

        for component, other_component in zip(self.components, other.components):
            if (component != other_component):
                return False

        return True


    def __copy__(self):
        return self.__class__(*[copy.copy(comp) for comp in self.components])

    def get_descendant_literal_conditions(self) -> Tuple[AbstractCondition]:
        children = []  # type: List[AbstractCondition]
        for component in self.components:
            if isinstance(component, AbstractConjunction):
                children.extend(component.get_descendant_literal_conditions())
            else:
                children.append(component)
        return tuple(children)

    def set_vector_idx(self, hyperparameter_to_idx: dict):
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

    def evaluate(self, instantiated_hyperparameters: Dict[str, Union[None, int, float, str]]
                 ) -> bool:
        cdef int* arrptr
        arrptr = <int*> malloc(sizeof(int) * self.n_components)

        # Then, check if all parents were passed
        conditions = self.dlcs
        for condition in conditions:
            if condition.parent.name not in instantiated_hyperparameters:
                raise ValueError("Evaluate must be called with all "
                                 "instanstatiated parent hyperparameters in "
                                 "the conjunction; you are (at least) missing "
                                 "'%s'" % condition.parent.name)

        # Finally, call evaluate for all direct descendents and combine the
        # outcomes
        for i, component in enumerate(self.components):
            e = component.evaluate(instantiated_hyperparameters)
            arrptr[i] = (e)

        rval = self._evaluate(self.n_components, arrptr)
        free(arrptr)
        return rval

    cdef int _evaluate_vector(self, np.ndarray instantiated_vector):
        cdef ConditionComponent component
        cdef int e
        cdef int rval
        cdef int* arrptr
        arrptr = <int*> malloc(sizeof(int) * self.n_components)

        # Finally, call evaluate for all direct descendents and combine the
        # outcomes
        for i in range(self.n_components):
            component = self.components[i]
            e = component._evaluate_vector(instantiated_vector)
            arrptr[i] = e

        rval = self._evaluate(self.n_components, arrptr)
        free(arrptr)
        return rval

    cdef int _evaluate(self, int I, int* evaluations):
        pass


cdef class AndConjunction(AbstractConjunction):
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

    cdef int _evaluate_vector(self, np.ndarray instantiated_vector):
        cdef ConditionComponent component
        cdef int e

        for i in range(self.n_components):
            component = self.components[i]
            e = component._evaluate_vector(instantiated_vector)
            if e == 0:
                return 0

        return 1

    cdef int _evaluate(self, int I, int* evaluations):
        for i in range(I):
            if evaluations[i] == 0:
                return 0
        return 1


cdef class OrConjunction(AbstractConjunction):
    def __init__(self, *args: AbstractCondition) -> None:
        """
        Similar to the *AndConjunction*, constraints can be combined by
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

    cdef int _evaluate(self, int I, int* evaluations):
        for i in range(I):
            if evaluations[i] == 1:
                return 1
        return 0

    cdef int _evaluate_vector(self, np.ndarray instantiated_vector):
        cdef ConditionComponent component
        cdef int e

        for i in range(self.n_components):
            component = self.components[i]
            e = component._evaluate_vector(instantiated_vector)
            if e == 1:
                return 1

        return 0
