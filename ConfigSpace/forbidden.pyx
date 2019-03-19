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

# cython: language_level=3

import copy
import numpy as np
cimport numpy as np
import io
from ConfigSpace.hyperparameters import Hyperparameter
from ConfigSpace.hyperparameters cimport Hyperparameter
from typing import List, Dict, Any, Union

from libc.stdlib cimport malloc, free

from ConfigSpace.forbidden cimport AbstractForbiddenComponent

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
#DTYPE = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
#ctypedef np.float_t DTYPE_t


cdef class AbstractForbiddenComponent(object):

    def __init__(self):
        pass

    def __repr__(self):
        pass


    def __richcmp__(self, other: Any, int op):
        """
        Todo: With Cython 2.7 this function can be replaced by using the six
              Python special methods (__eq__(), __lt__(),...).
        (--> https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#rich-comparisons)

        Before 2.7 there were no separate methods for the individual rich
        comparison operations. Instead there is a single method __richcmp__()
        which takes an integer indicating which operation is to be performed,
        as follows:
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

    def __copy__(self):
        raise NotImplementedError()

    cpdef get_descendant_literal_clauses(self):
        pass

    cpdef set_vector_idx(self, hyperparameter_to_idx):
        pass

    cpdef is_forbidden(self, instantiated_hyperparameters, strict):
        pass

    def is_forbidden_vector(self, instantiated_hyperparameters, strict):
        return bool(self.c_is_forbidden_vector(instantiated_hyperparameters, strict))

    cdef int c_is_forbidden_vector(self, np.ndarray instantiated_hyperparameters, int strict):
        pass


cdef class AbstractForbiddenClause(AbstractForbiddenComponent):

    def __init__(self, hyperparameter: Hyperparameter):
        if not isinstance(hyperparameter, Hyperparameter):
            raise TypeError("Argument 'hyperparameter' is not of type %s." %
                            Hyperparameter)
        self.hyperparameter = hyperparameter
        self.vector_id = -1

    cpdef get_descendant_literal_clauses(self):
        return (self, )

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

    def __copy__(self):
        return self.__class__(
            hyperparameter=copy.copy(self.hyperparameter),
            value=self.value
        )

    cpdef is_forbidden(self, instantiated_hyperparameters, strict):
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

    cdef int c_is_forbidden_vector(self, np.ndarray instantiated_vector, int strict):
        cdef DTYPE_t value = instantiated_vector[self.vector_id]
        if value != value:
            if strict:
                raise ValueError("Is_forbidden must be called with the "
                                 "instanstatiated vector id in the "
                                 "forbidden clause; you are missing "
                                 "'%s'" % self.vector_id)
            else:
                return False

        return self._is_forbidden_vector(value)

    cdef int _is_forbidden(self, float value):
        pass

    cdef int _is_forbidden_vector(self, DTYPE_t value):
        pass


cdef class MultipleValueForbiddenClause(AbstractForbiddenClause):
    cdef public values
    cdef public vector_values

    def __init__(self, hyperparameter: Hyperparameter, values: Any) -> None:
        super(MultipleValueForbiddenClause, self).__init__(hyperparameter)

        for value in values:
            if not self.hyperparameter.is_legal(value):
                raise ValueError("Forbidden clause must be instantiated with a "
                                 "legal hyperparameter value for '%s', but got "
                                 "'%s'" % (self.hyperparameter, str(value)))
        self.values = values
        self.vector_values = [self.hyperparameter._inverse_transform(value) for value in self.values]

    def __copy__(self):
        return self.__class__(
            hyperparameter=copy.copy(self.hyperparameter),
            values=copy.deepcopy(self.values)
        )

    cpdef is_forbidden(self, instantiated_hyperparameters, strict):
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

    cdef int c_is_forbidden_vector(self, np.ndarray instantiated_vector, int strict):
        cdef DTYPE_t value = instantiated_vector[self.vector_id]

        if value != value:
            if strict:
                raise ValueError("Is_forbidden must be called with the "
                                 "instanstatiated vector id in the "
                                 "forbidden clause; you are missing "
                                 "'%s'" % self.vector_id)
            else:
                return False

        return self._is_forbidden_vector(value)

    cdef int _is_forbidden(self, float value):
        pass

    cdef int _is_forbidden_vector(self, DTYPE_t value):
        pass


cdef class ForbiddenEqualsClause(SingleValueForbiddenClause):
    """
    A ForbiddenEqualsClause

    It forbids a value from the value range of a hyperparameter to be
    *equal to* ``value``.

    Example
    -------

    >>> cs = CS.ConfigurationSpace()
    >>> a = CSH.CategoricalHyperparameter('a', [1,2,3])
    >>> cs.add_hyperparameters([a])
    # It forbids the value 2 for the hyperparameter a
    >>> forbidden_clause_a = CS.ForbiddenEqualsClause(a, 2)
    >>> cs.add_forbidden_clause(forbidden_clause_a)

    Parameters
    ----------
    hyperparameter : :ref:`Hyperparameters`
        Methods on which a restriction will be made
    value : Any
        forbidden value
    """

    def __repr__(self):
        return "Forbidden: %s == %s" % (self.hyperparameter.name,
                                        repr(self.value))

    cdef int _is_forbidden(self, float value):
        return value == self.value

    cdef int _is_forbidden_vector(self, DTYPE_t value):
        return value == self.vector_value


cdef class ForbiddenInClause(MultipleValueForbiddenClause):
    def __init__(self, hyperparameter: Dict[str, Union[None, str, float, int]], values: Any) -> None:
        """
        A ForbiddenInClause.

        It forbids a value from the value range of a hyperparameter to be
        *in* a collection of ``values``.

        Note
        ----

        The forbidden values have to be a subset of the hyperparameter's values.

        Example
        -------

        >>> cs = CS.ConfigurationSpace()
        >>> a = CSH.CategoricalHyperparameter('a', [1,2,3])
        >>> cs.add_hyperparameters([a])
        # It forbids the values 2, 3, 4 for the hyperparameter 'a'
        >>> forbidden_clause_a = CS.ForbiddenInClause(a, [2, 3])
        >>> cs.add_forbidden_clause(forbidden_clause_a)

        Parameters
        ----------
        hyperparameter : (:ref:`Hyperparameters`, dict)
            Hyperparameter on which a restriction will be made
        values : Any
            Collection of forbidden values
        """

        super(ForbiddenInClause, self).__init__(hyperparameter, values)
        self.values = set(self.values)
        self.vector_values = set(self.vector_values)

    def __repr__(self) -> str:
        return "Forbidden: %s in %s" % (
            self.hyperparameter.name,
            "{" + ", ".join((repr(value)
                             for value in sorted(self.values))) + "}")

    cdef int _is_forbidden(self, float value):
        return value in self.values

    cdef int _is_forbidden_vector(self, DTYPE_t value):
        return value in self.vector_values


cdef class AbstractForbiddenConjunction(AbstractForbiddenComponent):
    cdef public tuple components
    cdef tuple dlcs
    cdef public int n_components

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
        self.n_components = len(self.components)
        self.dlcs = self.get_descendant_literal_clauses()

    def __repr__(self):
        pass

    def __copy__(self):
        return self.__class__([copy(comp) for comp in self.components])

    def __richcmp__(self, other: Any, int op):
        """
        Todo: With Cython 2.7 this function can be replaced by using the six
              Python special methods (__eq__(), __lt__(),...).
        (--> https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#rich-comparisons)

        Before 2.7 there were no separate methods for the individual rich
        comparison operations. Instead there is a single method __richcmp__()
        which takes an integer indicating which operation is to be performed,
        as follows:
        < 	0
        == 2
        > 	4
        <=	1
        !=	3
        >=	5
        """

        if isinstance(other, self.__class__):
            if op == 2:
                if self.n_components != other.n_components:
                    return False
                return all([self.components[i] == other.components[i]
                            for i in range(self.n_components)])

            elif op == 3:
                if self.n_copmonents == other.n_components:
                    return False
                return any([self.components[i] != other.components[i]
                            for i in range(self.n_components)])

        return NotImplemented

    cpdef set_vector_idx(self, hyperparameter_to_idx):
        for component in self.components:
            component.set_vector_idx(hyperparameter_to_idx)

    cpdef get_descendant_literal_clauses(self):
        children = []
        for component in self.components:
            if isinstance(component, AbstractForbiddenConjunction):
                children.extend(component.get_descendant_literal_clauses())
            else:
                children.append(component)
        return tuple(children)

    cpdef is_forbidden(self, instantiated_hyperparameters, strict):
        ihp_names = list(instantiated_hyperparameters.keys())

        for dlc in self.dlcs:
            if dlc.hyperparameter.name not in ihp_names:
                if strict:
                    raise ValueError("Is_forbidden must be called with all "
                                     "instanstatiated hyperparameters in the "
                                     "and conjunction of forbidden clauses; "
                                     "you are (at least) missing "
                                     "'%s'" % dlc.hyperparameter.name)
                else:
                    return False

        cdef int* arrptr
        arrptr = <int*> malloc(sizeof(int) * self.n_components)

        # Finally, call is_forbidden for all direct descendents and combine the
        # outcomes
        np_index = 0
        for component in self.components:
            e = component.is_forbidden(instantiated_hyperparameters,
                                       strict=strict)
            arrptr[np_index] = e
            np_index += 1

        rval = self._is_forbidden(self.n_components, arrptr)
        free(arrptr)
        return rval

    cdef int c_is_forbidden_vector(self, np.ndarray instantiated_vector, int strict):
        cdef int e = 0
        cdef int rval
        cdef AbstractForbiddenComponent component

        cdef int* arrptr
        arrptr = <int*> malloc(sizeof(int) * self.n_components)

        # Finally, call is_forbidden for all direct descendents and combine the
        # outcomes. Check only as many forbidden clauses as the actual
        # evaluation function queries for (e.g. and conditions are False
        # if only one of the components evaluates to False).

        for i in range(self.n_components):
            component = self.components[i]
            e = component.c_is_forbidden_vector(instantiated_vector, strict)
            arrptr[i] = e

        rval = self._is_forbidden(self.n_components, arrptr)
        free(arrptr)
        return rval

    cdef int _is_forbidden(self, int I, int* evaluations):
        pass


cdef class ForbiddenAndConjunction(AbstractForbiddenConjunction):
    """
    A ForbiddenAndConjunction.

    The ForbiddenAndConjunction combines forbidden-clauses, which allows to
    build powerful constraints.

    Example
    -------

    >>> cs = CS.ConfigurationSpace()
    >>> a = CSH.CategoricalHyperparameter('a', [1,2,3])
    >>> b = CSH.CategoricalHyperparameter('b', [2,5,6])
    >>> cs.add_hyperparameters([a, b])
    >>> forbidden_clause_a = CS.ForbiddenEqualsClause(a, 2)
    >>> forbidden_clause_b = CS.ForbiddenInClause(b, [2])
    >>> forbidden_clause = CS.ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_b)
    >>> cs.add_forbidden_clause(forbidden_clause)

    Parameters
    ----------
    *args : list([:ref:`Forbidden clauses`])
        forbidden clauses, which should be combined
    """

    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("(")
        for idx, component in enumerate(self.components):
            retval.write(str(component))
            if idx < len(self.components) - 1:
                retval.write(" && ")
        retval.write(")")
        return retval.getvalue()

    cdef int _is_forbidden(self, int I, int* evaluations):
        # Return False if one of the components evaluates to False

        for i in range(I):
            if evaluations[i] == 0:
                return 0
        return 1

    cdef int c_is_forbidden_vector(self, np.ndarray instantiated_vector, int strict):
        # Copy from above to have early stopping of the evaluation of clauses -
        # gave only very modest improvements of ~5%; should probably be reworked
        # if adding more conjunctions in order to use better software design to
        # avoid code duplication.
        cdef int e = 0
        cdef AbstractForbiddenComponent component

        # Finally, call is_forbidden for all direct descendents and combine the
        # outcomes. Check only as many forbidden clauses as the actual
        # evaluation function queries for (e.g. and conditions are False
        # if only one of the components evaluates to False).

        for i in range(self.n_components):
            component = self.components[i]
            e = component.c_is_forbidden_vector(instantiated_vector, strict)
            if e == 0:
                return 0

        return 1
