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
import operator

import io
from functools import reduce

from ConfigSpace.hyperparameters import Hyperparameter


class AbstractForbiddenComponent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
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

    def is_forbidden(self, instantiated_hyperparameters, strict=True):
        value = instantiated_hyperparameters.get(self.hyperparameter.name)

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

    def is_forbidden(self, instantiated_hyperparameters, strict=True):
        value = instantiated_hyperparameters.get(self.hyperparameter.name)

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
                                        repr(self.value))

    def _is_forbidden(self, value):
        return value == self.value


class ForbiddenInClause(MultipleValueForbiddenClause):
    def __init__(self, hyperparameter, values):
        super(ForbiddenInClause, self).__init__(hyperparameter, values)
        self.values = set(self.values)

    def __repr__(self):
        return "Forbidden: %s in %s" % (
            self.hyperparameter.name,
            "{" + ", ".join((repr(value)
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

    def get_descendant_literal_clauses(self):
        children = []
        for component in self.components:
            if isinstance(component, AbstractForbiddenConjunction):
                children.extend(component.get_descendant_literal_clauses())
            else:
                children.append(component)
        return children

    def is_forbidden(self, instantiated_hyperparameters, strict=True):
        ihp_names = list(instantiated_hyperparameters.keys())

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
        retval = io.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    def _is_forbidden(self, evaluations):
        return reduce(operator.and_, evaluations)
