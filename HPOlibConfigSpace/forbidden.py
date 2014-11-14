from abc import ABCMeta, abstractmethod
from itertools import izip
import operator
import StringIO

from HPOlibConfigSpace.hyperparameters import Hyperparameter, \
    InstantiatedHyperparameter


class AbstractForbiddenComponent(object):
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
    def get_descendant_literal_clauses(self):
        pass

    @abstractmethod
    def is_forbidden(self, instantiated_hyperparameters):
        pass


class AbstractForbiddenClause(AbstractForbiddenComponent):
    def get_descendant_literal_clauses(self):
        return [self]


class SingleValueForbiddenClause(AbstractForbiddenClause):
    def __init__(self, hyperparameter, value):
        super(SingleValueForbiddenClause, self).__init__()
        if not isinstance(hyperparameter, Hyperparameter):
            raise TypeError("'%s' is not of type %s." %
                            (str(hyperparameter), Hyperparameter))
        self.hyperparameter = hyperparameter
        if not hyperparameter.is_legal(value):
            raise ValueError("Forbidden clause must be instantiated with a "
                             "legal hyperparameter value for '%s', but got "
                             "'%s'" % (hyperparameter, str(value)))
        self.value = value

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif self.hyperparameter != other.hyperparameter:
            return False
        elif self.value != other.value:
            return False
        else:
            return True

    def is_forbidden(self, instantiated_hyperparameters):
        target_ihp = None
        for ihp in instantiated_hyperparameters:
            if not isinstance(ihp, InstantiatedHyperparameter):
                raise TypeError("Is_forbidden() must be called with an "
                                "instance of %s, you provided an instance of "
                                "%s." % (InstantiatedHyperparameter, type(ihp)))
            if ihp.hyperparameter.name == self.hyperparameter.name:
                target_ihp = ihp

        if target_ihp is None:
            raise ValueError("Is_forbidden must be called with the "
                             "instanstatiated hyperparameter in the "
                             "forbidden clause; you are missing "
                             "'%s'" % self.hyperparameter.name)

        return self._is_forbidden(target_ihp)

    @abstractmethod
    def _is_forbidden(self, target_instantiated_hyperparameter):
        pass



class MultipleValueForbiddenClause(AbstractForbiddenClause):
    def __init__(self, hyperparameter, values):
        super(MultipleValueForbiddenClause, self).__init__()
        if not isinstance(hyperparameter, Hyperparameter):
            raise TypeError("Argument 'hyperparameter' is not of type %s." %
                            Hyperparameter)
        self.hyperparameter = hyperparameter
        for value in values:
            if not hyperparameter.is_legal(value):
                raise ValueError("Forbidden clause must be instantiated with a "
                                 "legal hyperparameter value for '%s', but got "
                                 "'%s'" % (hyperparameter, str(value)))
        self.values = values

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif self.hyperparameter != other.hyperparameter:
            return False
        elif self.values != other.values:
            return False
        else:
            return True

    def is_forbidden(self, instantiated_hyperparameters):
        target_ihp = None
        for ihp in instantiated_hyperparameters:
            if not isinstance(ihp, InstantiatedHyperparameter):
                raise TypeError("Is_forbidden() must be called with an "
                                "instance of %s, you provided an instance of "
                                "%s." % (InstantiatedHyperparameter, type(ihp)))
            if ihp.hyperparameter.name == self.hyperparameter.name:
                target_ihp = ihp

        if target_ihp is None:
            raise ValueError("Is_forbidden must be called with the "
                             "instanstatiated hyperparameter in the "
                             "forbidden clause; you are missing "
                             "'%s'." % self.hyperparameter.name)

        return self._is_forbidden(target_ihp)

    @abstractmethod
    def _is_forbidden(self, target_instantiated_hyperparameter):
        pass


class ForbiddenEqualsClause(SingleValueForbiddenClause):
    def __repr__(self):
        return "Forbidden: %s == %s" % (self.hyperparameter.name,
                                        str(self.value))

    def _is_forbidden(self, target_instantiated_hyperparameter):
        return target_instantiated_hyperparameter.value == self.value


class ForbiddenInClause(MultipleValueForbiddenClause):
    def __init__(self, hyperparameter, values):
        super(ForbiddenInClause, self).__init__(hyperparameter, values)
        self.values = set(self.values)

    def __repr__(self):
        return "Forbidden: %s in %s" % (self.hyperparameter.name,
            "{" + ", ".join((str(value) for value in sorted(self.values))) + "}")

    def _is_forbidden(self, target_instantiated_hyperparameter):
        return target_instantiated_hyperparameter.value in self.values


class AbstractForbiddenConjunction(AbstractForbiddenComponent):
    def __init__(self, *args):
        super(AbstractForbiddenConjunction, self).__init__()
        # Test the classes
        for idx, component in enumerate(args):
            if not isinstance(component, AbstractForbiddenComponent):
                raise TypeError("Argument #%d is not an instance of %s, "
                                "but %s" % (
                    idx, AbstractForbiddenComponent, type(component)))

        self.components = args

    @abstractmethod
    def __repr__(self):
        pass

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif len(self.components) != len(other.components):
            return False
        else:
            for comp1, comp2 in izip(sorted(self.components,
                                            key=lambda t: t.__repr__()),
                                     sorted(other.components,
                                            key=lambda t: t.__repr__())):
                if comp1 != comp2:
                    return False
        return True

    def get_descendant_literal_clauses(self):
        children = []
        for component in self.components:
            if isinstance(component, AbstractForbiddenConjunction):
                children.extend(component.get_descendant_literal_clauses())
            else:
                children.append(component)
        return children

    def is_forbidden(self, instantiated_hyperparameters):
        ihp_names = []
        for ihp in instantiated_hyperparameters:
            if not isinstance(ihp, InstantiatedHyperparameter):
                raise TypeError("Is_forbidden() must be called with "
                                "instances of %s, you provided an instance of "
                                "%s" % (InstantiatedHyperparameter, type(ihp)))
            ihp_names.append(ihp.hyperparameter.name)

        dlcs = self.get_descendant_literal_clauses()
        for dlc in dlcs:
            if dlc.hyperparameter.name not in ihp_names:
                raise ValueError("Is_forbidden must be called with all "
                                 "instanstatiated hyperparameters in the and conjunction of "
                                 "forbidden clauses; you are (at least) missing "
                                 "'%s'" % dlc.hyperparameter.name)

        # Finally, call is_forbidden for all direct descendents and combine the
        # outcomes
        evaluations = []
        for component in self.components:
            # If it's a condition, we must only pass the parent hyperparameter
            e = component.is_forbidden(instantiated_hyperparameters)
            evaluations.append(e)

        return self._is_forbidden(evaluations)

    @abstractmethod
    def _is_forbidden(self, evaluations):
        pass


class ForbiddenAndConjunction(AbstractForbiddenConjunction):
    def __repr__(self):
        retval = StringIO.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    def _is_forbidden(self, evaluations):
        return reduce(operator.and_, evaluations)



