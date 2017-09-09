def say_hello_to(name):
    print("Hello  %s!" % name)


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

import numpy as np
cimport numpy as np
import io.io as io
from ConfigSpace.hyperparameters import Hyperparameter
from typing import List, Dict, Any, Union

ctypedef np.int_t DTYPE_t


cdef class AbstractForbiddenComponent(object):
    # __metaclass__ = ABCMeta

    cdef public hyperparameter # type: Hyperparameter
    cdef public vector_id
    cdef public value
    cdef public float vector_value
    cdef dict __dict__

 #   @abstractmethod
    def __init__(self):
        pass

 #   @abstractmethod
    def __repr__(self):
        pass


    def __richcmp__(self, other: Any, int op):
        """Override the default Equals behavior
         There are no separate methods for the individual rich comparison operations (__eq__(), __le__(), etc.).
          Instead there is a single method __richcmp__() which takes an integer indicating which operation is to be performed, as follows:
         < 	0
         == 2
         > 	4
         <=	1
         !=	3
         >=	5
         """
        if self.value is None:
            self.value = self.values
            other.value = other.values

        if isinstance(other, self.__class__):
            if op == 2:
                return (self.value == other.value
                         and self.hyperparameter.name == other.hyperparameter.name)

            elif op == 3:
                return False == (self.value == other.value
                         and self.hyperparameter.name == other.hyperparameter.name)


        return NotImplemented

    def __hash__(self) -> int:
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

    cpdef get_descendant_literal_clauses(self):
        pass

    cpdef set_vector_idx(self, hyperparameter_to_idx):
        pass

    cpdef is_forbidden(self, instantiated_hyperparameters, strict=True):
        pass

    def is_forbidden_vector(self, instantiated_hyperparameters, strict=True):
        pass


cdef class AbstractForbiddenClause(AbstractForbiddenComponent):

    def __init__(self, hyperparameter: Hyperparameter):
        if not isinstance(hyperparameter, Hyperparameter):
            raise TypeError("Argument 'hyperparameter' is not of type %s." %
                            Hyperparameter)
        self.hyperparameter = hyperparameter
        self.vector_id = None

    cpdef get_descendant_literal_clauses(self):
        return [self]

    cpdef set_vector_idx(self, hyperparameter_to_idx):
        self.vector_id = hyperparameter_to_idx[self.hyperparameter.name]


cdef class SingleValueForbiddenClause(AbstractForbiddenClause):
    def __init__(self, hyperparameter: Hyperparameter, value: Any) -> None:
        super(SingleValueForbiddenClause, self).__init__(hyperparameter)
        if not self.hyperparameter.is_legal(value):
            raise ValueError("Forbidden clause must be instantiated with a "
                             "legal hyperparameter value for '%s', but got "
                             "'%s'" % (self.hyperparameter, str(value)))
        self.value = value
        self.vector_value = self.hyperparameter._inverse_transform(self.value)

    cpdef is_forbidden(self, instantiated_hyperparameters, strict = True):
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

    def is_forbidden_vector(self, instantiated_vector, strict = True):
        value = instantiated_vector[self.vector_id]
        if value != value:
            if strict:
                raise ValueError("Is_forbidden must be called with the "
                                 "instanstatiated vector id in the "
                                 "forbidden clause; you are missing "
                                 "'%s'" % self.vector_id)
            else:
                return False

        return self._is_forbidden_vector(value)

    cpdef _is_forbidden(self, target_instantiated_hyperparameter):
        pass

    def _is_forbidden_vector(self, target_instantiated_vector):
        pass


cdef class MultipleValueForbiddenClause(AbstractForbiddenClause):
    def __init__(self, hyperparameter: Hyperparameter, values: Any) -> None:
        super(MultipleValueForbiddenClause, self).__init__(hyperparameter)

        for value in values:
            if not self.hyperparameter.is_legal(value):
                raise ValueError("Forbidden clause must be instantiated with a "
                                 "legal hyperparameter value for '%s', but got "
                                 "'%s'" % (self.hyperparameter, str(value)))
        self.values = values
        self.vector_values = [self.hyperparameter._inverse_transform(value) for value in self.values]

    cpdef is_forbidden(self, instantiated_hyperparameters, strict=True):
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

    def is_forbidden_vector(self, instantiated_vector, strict=True):
        value = instantiated_vector[self.vector_id]

        if value is np.NaN:
            if strict:
                raise ValueError("Is_forbidden must be called with the "
                                 "instanstatiated vector id in the "
                                 "forbidden clause; you are missing "
                                 "'%s'" % self.vector_id)
            else:
                return False

        return self._is_forbidden_vector(value)

    cpdef _is_forbidden(self, target_instantiated_hyperparameter):
        pass

    def _is_forbidden_vector(self, target_instantiated_vector):
        pass


cdef class ForbiddenEqualsClause(SingleValueForbiddenClause):
    def __repr__(self):
        return "Forbidden: %s == %s" % (self.hyperparameter.name,
                                        repr(self.value))

    cpdef _is_forbidden(self, value: Any):
        return value == self.value

    def _is_forbidden_vector(self, value: Any):
        return value == self.vector_value


cdef class ForbiddenInClause(MultipleValueForbiddenClause):
    def __init__(self, hyperparameter: Dict[str, Union[None, str, float, int]], values: Any) -> None:
        super(ForbiddenInClause, self).__init__(hyperparameter, values)
        self.values = set(self.values)
        self.vector_values = set(self.vector_values)

    def __repr__(self) -> str:
        return "Forbidden: %s in %s" % (
            self.hyperparameter.name,
            "{" + ", ".join((repr(value)
                             for value in sorted(self.values))) + "}")

    cpdef _is_forbidden(self, value):
        return value in self.values

    def _is_forbidden_vector(self, value):
        return value in self.vector_values


cdef class AbstractForbiddenConjunction(AbstractForbiddenComponent):
    def __init__(self, *args: AbstractForbiddenComponent) -> None:
        super(AbstractForbiddenConjunction, self).__init__()
        # Test the classes
        for idx, component in enumerate(args):
            if not isinstance(component, AbstractForbiddenComponent):
                raise TypeError("Argument #%d is not an instance of %s, "
                                "but %s" % (
                                    idx, AbstractForbiddenComponent,
                                    type(component)))

        self.components = args

    def __repr__(self):
        pass

    cpdef  set_vector_idx(self, hyperparameter_to_idx):
        for component in self.components:
            component.set_vector_idx(hyperparameter_to_idx)

    # todo:recheck is return type should be AbstractForbiddenComponent or AbstractForbiddenConjunction or Hyperparameter
    cpdef get_descendant_literal_clauses(self):
        children = []
        for component in self.components:
            if isinstance(component, AbstractForbiddenConjunction):
                children.extend(component.get_descendant_literal_clauses())
            else:
                children.append(component)
        return children

    cpdef is_forbidden(self, instantiated_hyperparameters, strict: bool=True):
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
        cdef np.ndarray np_evaluations = np.zeros(len(self.components), dtype=bool)
        np_index = 0
        for component in self.components:
            e = component.is_forbidden(instantiated_hyperparameters,
                                       strict=strict)
            np_evaluations[np_index] = e
            np_index += 1

        return self._is_forbidden(np_evaluations)

    def is_forbidden_vector(self, instantiated_vector: np.ndarray, strict: bool = True):
        dlcs = self.get_descendant_literal_clauses()
        for dlc in dlcs:
            if dlc.vector_id not in range(len(instantiated_vector)):
                if strict:
                    raise ValueError("Is_forbidden must be called with all "
                                     "instanstatiated hyperparameters in the "
                                     "and conjunction of forbidden clauses; "
                                     "you are (at least) missing "
                                     "'%s'" % dlc.vector_id)
                else:
                    return False

        # Finally, call is_forbidden for all direct descendents and combine the
        # outcomes. Check only as many forbidden clauses as the actual
        # evaluation function queries for (e.g. and conditions are False
        # if only one of the components evaluates to False).

        cdef np.ndarray np_evaluations = np.zeros(len(self.components), dtype=bool)
        np_index = 0
        for component in self.components:
            e = component.is_forbidden_vector(instantiated_vector,
                                       strict=strict)
            np_evaluations[np_index] = e
            np_index += 1

        return self._is_forbidden(np_evaluations)

 #   @abstractmethod
    cpdef _is_forbidden(self, np.ndarray evaluations):
        pass


cdef class ForbiddenAndConjunction(AbstractForbiddenConjunction):
    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    cpdef _is_forbidden(self, np.ndarray evaluations):
        # Return False if one of the components evaluates to False

        cdef int I = evaluations.shape[0]

        for i in range(I):
            if evaluations[i] == False:
                return False
        return True
