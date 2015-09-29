from abc import ABCMeta, abstractmethod
import operator

import six

from HPOlibConfigSpace.hyperparameters import Hyperparameter


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

    def is_forbidden(self, instantiated_hyperparameters, strict=True):
        value = None
        for hp_name in instantiated_hyperparameters:
            if hp_name == self.hyperparameter.name:
                value = instantiated_hyperparameters[hp_name]

        if value is None:
            if strict:
                raise ValueError("Is_forbidden must be called with the "
                                 "instanstatiated hyperparameter in the "
                                 "forbidden clause; you are missing "
                                 "'%s'" % self.hyperparameter.name)
            else:
                return False

        return self._is_forbidden(value)

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

    def is_forbidden(self, instantiated_hyperparameters, strict=True):
        value = None
        for hp_name in instantiated_hyperparameters:
            if hp_name == self.hyperparameter.name:
                value = instantiated_hyperparameters[hp_name]

        if value is None:
            if strict:
                raise ValueError("Is_forbidden must be called with the "
                                 "instanstatiated hyperparameter in the "
                                 "forbidden clause; you are missing "
                                 "'%s'." % self.hyperparameter.name)
            else:
                return False

        return self._is_forbidden(value)

    @abstractmethod
    def _is_forbidden(self, target_instantiated_hyperparameter):
        pass


class ForbiddenEqualsClause(SingleValueForbiddenClause):
    def __repr__(self):
        return "Forbidden: %s == %s" % (self.hyperparameter.name,
                                        str(self.value))

    def _is_forbidden(self, value):
        return value == self.value


class ForbiddenInClause(MultipleValueForbiddenClause):
    def __init__(self, hyperparameter, values):
        super(ForbiddenInClause, self).__init__(hyperparameter, values)
        self.values = set(self.values)

    def __repr__(self):
        return "Forbidden: %s in %s" % (
            self.hyperparameter.name,
            "{" + ", ".join((str(value)
                             for value in sorted(self.values))) + "}")

    def _is_forbidden(self, value):
        return value in self.values


class AbstractForbiddenConjunction(AbstractForbiddenComponent):
    def __init__(self, *args):
        super(AbstractForbiddenConjunction, self).__init__()
        # Test the classes
        for idx, component in enumerate(args):
            if not isinstance(component, AbstractForbiddenComponent):
                raise TypeError("Argument #%d is not an instance of %s, "
                                "but %s" % (
                                    idx, AbstractForbiddenComponent,
                                    type(component)))

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
            for comp1, comp2 in six.moves.zip(
                    sorted(self.components, key=lambda t: t.__repr__()),
                    sorted(other.components, key=lambda t: t.__repr__())):
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

    def is_forbidden(self, instantiated_hyperparameters, strict=True):
        ihp_names = []
        for ihp in instantiated_hyperparameters:
            ihp_names.append(ihp)

        dlcs = self.get_descendant_literal_clauses()
        for dlc in dlcs:
            if dlc.hyperparameter.name not in ihp_names:
                if strict:
                    raise ValueError("Is_forbidden must be called with all "
                                     "instanstatiated hyperparameters in the "
                                     "and conjunction of forbidden clauses; "
                                     "you are (at least) missing "
                                     "'%s'" % dlc.hyperparameter.name)
                else:
                    return False

        # Finally, call is_forbidden for all direct descendents and combine the
        # outcomes
        evaluations = []
        for component in self.components:
            e = component.is_forbidden(instantiated_hyperparameters,
                                       strict=strict)
            evaluations.append(e)
        return self._is_forbidden(evaluations)

    @abstractmethod
    def _is_forbidden(self, evaluations):
        pass


class ForbiddenAndConjunction(AbstractForbiddenConjunction):
    def __repr__(self):
        retval = six.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    def _is_forbidden(self, evaluations):
        return six.moves.reduce(operator.and_, evaluations)
