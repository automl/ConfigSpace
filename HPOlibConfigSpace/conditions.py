__author__ = 'feurerm'

from abc import ABCMeta, abstractmethod
from itertools import combinations
import operator

import six

from HPOlibConfigSpace.hyperparameters import Hyperparameter


class ConditionComponent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_descendant_literal_conditions(self):
        pass

    @abstractmethod
    def evaluate(self, instantiated_parent_hyperparameter):
        pass

    # http://stackoverflow.com/a/25176504/4636294
    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))


class AbstractCondition(ConditionComponent):
    # TODO create a condition evaluator!

    @abstractmethod
    def __init__(self, child, parent):
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

    def get_children(self):
        return [self.child]

    def get_descendant_literal_conditions(self):
        return [self]

    def evaluate(self, instantiated_parent_hyperparameter):
        hp_name = self.parent.name
        return self._evaluate(instantiated_parent_hyperparameter[hp_name])

    @abstractmethod
    def _evaluate(self, instantiated_parent_hyperparameter):
        pass


class AbstractConjunction(ConditionComponent):
    def __init__(self, *args):
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

    def get_descendant_literal_conditions(self):
        children = []
        for component in self.components:
            if isinstance(component, AbstractConjunction):
                children.extend(component.get_descendant_literal_conditions())
            else:
                children.append(component)
        return children

    def get_children(self):
        children = []
        for component in self.components:
            children.extend(component.get_children())
        return children

    def evaluate(self, instantiated_hyperparameters):
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

    @abstractmethod
    def _evaluate(self, evaluations):
        pass


class EqualsCondition(AbstractCondition):
    def __init__(self, child, parent, value):
        super(EqualsCondition, self).__init__(child, parent)
        if not parent.is_legal(value):
            raise ValueError("Hyperparameter '%s' is "
                             "conditional on the illegal value '%s' of "
                             "its parent hyperparameter '%s'" %
                             (child.name, value, parent.name))
        self.value = value

    def __repr__(self):
        return "%s | %s == %s" % (self.child.name, self.parent.name,
                                  str(self.value))

    def _evaluate(self, value):
        return value == self.value


class NotEqualsCondition(AbstractCondition):
    def __init__(self, child, parent, value):
        super(NotEqualsCondition, self).__init__(child, parent)
        if not parent.is_legal(value):
            raise ValueError("Hyperparameter '%s' is "
                             "conditional on the illegal value '%s' of "
                             "its parent hyperparameter '%s'" %
                             (child.name, value, parent.name))
        self.value = value

    def __repr__(self):
        return "%s | %s != %s" % (self.child.name, self.parent.name,
                                  str(self.value))

    def _evaluate(self, value):
        return value != self.value


class InCondition(AbstractCondition):
    def __init__(self, child, parent, values):
        super(InCondition, self).__init__(child, parent)
        for value in values:
            if not parent.is_legal(value):
                raise ValueError("Hyperparameter '%s' is "
                                 "conditional on the illegal value '%s' of "
                                 "its parent hyperparameter '%s'" %
                                 (child.name, value, parent.name))
        self.values = values

    def __repr__(self):
        return "%s | %s in {%s}" % (self.child.name, self.parent.name,
                                    ", ".join(
                                        [str(value) for value in self.values]))

    def _evaluate(self, value):
        return value in self.values


class AndConjunction(AbstractConjunction):
    # TODO: test if an AndConjunction results in an illegal state or a
    # Tautology! -> SAT solver
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("AndConjunction must at least have two "
                             "Conditions.")
        super(AndConjunction, self).__init__(*args)

    def __repr__(self):
        retval = six.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    def _evaluate(self, evaluations):
        return six.moves.reduce(operator.and_, evaluations)


class OrConjunction(AbstractConjunction):
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("OrConjunction must at least have two "
                             "Conditions.")
        super(OrConjunction, self).__init__(*args)

    def __repr__(self):
        retval = six.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" || ")
        retval.write(")")
        return retval.getvalue()

    def _evaluate(self, evaluations):
        return six.moves.reduce(operator.or_, evaluations)