__author__ = 'feurerm'

from abc import ABCMeta, abstractmethod
from itertools import combinations, izip
import operator
import StringIO


from HPOlibConfigSpace.hyperparameters import Hyperparameter, \
    InstantiatedHyperparameter


class ConditionComponent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    def __ne__(self, other):
        return not self.__eq__(other)

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_descendant_literal_conditions(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


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
        if not isinstance(instantiated_parent_hyperparameter,
                          InstantiatedHyperparameter):
            raise TypeError("Evaluate must be called with an instance of %s, "
                            "not %s" % (InstantiatedHyperparameter,
                                        type(instantiated_parent_hyperparameter)))

        if instantiated_parent_hyperparameter.hyperparameter.name != \
                self.parent.name:
            raise ValueError("Evaluate must be called with the "
                             "instanstatiated parent hyperparameter '%s', "
                             "not '%s'." % (self.parent.name,
                                          instantiated_parent_hyperparameter.
                                          hyperparameter.name))

        return self._evaluate(instantiated_parent_hyperparameter)

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
                                "but %s" % (idx, ConditionComponent, type(component)))

        # Test that all conjunctions and conditions have the same child!
        children = self.get_children()
        for c1, c2 in combinations(children, 2):
            if c1 != c2:
                raise ValueError("All Conjunctions and Conditions must have "
                                 "the same child.")

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif len(self.components) != len(other.components):
            return False
        else:
            for comp1, comp2 in izip(sorted(self.components,
                                            key=lambda t:t.__repr__()),
                                     sorted(other.components,
                                            key=lambda t: t.__repr__())):
                if comp1 != comp2:
                    return False
        return True

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
        # First, check if evaluate was called only with instantiated
        # hyperparameters
        ihps = {}
        for ihp in instantiated_hyperparameters:
            if not isinstance(ihp, InstantiatedHyperparameter):
                raise TypeError("Evaluate must be called with an instances of "
                                "%s, you provided one instance of %s" % (
                                    InstantiatedHyperparameter, type(ihp)))
            ihps[ihp.hyperparameter.name] = ihp

        # Then, check if all parents were passed
        conditions = self.get_descendant_literal_conditions()
        for condition in conditions:
            if condition.parent.name not in ihps:
                raise ValueError("Evaluate must be called with all "
                                 "instanstatiated parent hyperparameters in "
                                 "the conjunction; you are (at least) missing "
                                 "'%s'" % condition.parent.name)

        # Finally, call evaluate for all direct descendents and combine the
        # outcomes
        evaluations = []
        for component in self.components:
            # If it's a condition, we must only pass the parent hyperparameter
            if isinstance(component, AbstractCondition):
                parent_name = component.parent.name
                e = component.evaluate(ihps[parent_name])
                evaluations.append(e)
            else:
                e = component.evaluate(instantiated_hyperparameters)
                evaluations.append(e)

        return self._evaluate(evaluations)

    @abstractmethod
    def _evaluate(self, evaluations):
        pass


class EqualsCondition(AbstractCondition):
    def __init__(self, child, parent, value):
        super(EqualsCondition, self).__init__(child, parent)
        parent.is_legal(value)
        self.value = value

    def __repr__(self):
        return "%s | %s == %s" % (self.child.name, self.parent.name,
                                  str(self.value))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.child == other.child and \
                self.parent == other.parent and \
                self.value == other.value

    def _evaluate(self, instantiated_parent_hyperparameter):
        return instantiated_parent_hyperparameter.value == self.value


class NotEqualsCondition(AbstractCondition):
    def __init__(self, child, parent, value):
        super(NotEqualsCondition, self).__init__(child, parent)
        parent.is_legal(value)
        self.value = value

    def __repr__(self):
        return "%s | %s != %s" % (self.child.name, self.parent.name,
                                  str(self.value))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.child == other.child and \
                   self.parent == other.parent and \
                   self.value == other.value

    def _evaluate(self, instantiated_parent_hyperparameter):
        return instantiated_parent_hyperparameter.value != self.value


class InCondition(AbstractCondition):
    def __init__(self, child, parent, values):
        super(InCondition, self).__init__(child, parent)
        for value in values:
            parent.is_legal(value)
        self.values = values

    def __repr__(self):
        return "%s | %s in {%s}" % (self.child.name, self.parent.name,
            ", ".join([str(value) for value in self.values]))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.child == other.child and \
                   self.parent == other.parent and \
                   self.values == other.values

    def _evaluate(self, instantiated_parent_hyperparameter):
        return instantiated_parent_hyperparameter.value in self.values


class AndConjunction(AbstractConjunction):
    # TODO: test if an AndConjunction results in an illegal state or a
    # Tautology! -> SAT solver
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("AndConjunction must at least have two "
                             "Conditions.")
        super(AndConjunction, self).__init__(*args)

    def __repr__(self):
        retval = StringIO.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    def _evaluate(self, evaluations):
        return reduce(operator.and_, evaluations)


class OrConjunction(AbstractConjunction):
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("AndConjunction must at least have two "
                             "Conditions.")
        super(OrConjunction, self).__init__(*args)

    def __repr__(self):
        retval = StringIO.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" || ")
        retval.write(")")
        return retval.getvalue()

    def _evaluate(self, evaluations):
        return reduce(operator.or_, evaluations)