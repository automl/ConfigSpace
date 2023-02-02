from collections import OrderedDict
import copy
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
cimport numpy as np
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

from ConfigSpace.hyperparameters.hyperparameter cimport Hyperparameter


cdef class OrdinalHyperparameter(Hyperparameter):
    cdef public tuple sequence
    cdef public int num_elements
    cdef sequence_vector
    cdef value_dict

    def __init__(
        self,
        name: str,
        sequence: Union[List[Union[float, int, str]], Tuple[Union[float, int, str]]],
        default_value: Union[str, int, float, None] = None,
        meta: Optional[Dict] = None
    ) -> None:
        """
        An ordinal hyperparameter.

        Its values are sampled form a ``sequence`` of values.
        The sequence of values from a ordinal hyperparameter is ordered.

        ``None`` is a forbidden value, please use a string constant instead and parse
        it in your own code, see `here <https://github.com/automl/ConfigSpace/issues/159>`_
        for further details.

        >>> from ConfigSpace import OrdinalHyperparameter
        >>>
        >>> OrdinalHyperparameter('o', sequence=['10', '20', '30'])
        o, Type: Ordinal, Sequence: {10, 20, 30}, Default: 10

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed.
        sequence : list or tuple with (str, float, int)
            ordered collection of values to sample hyperparameter from.
        default_value : int, float, str, optional
            Sets the default value of a hyperparameter to a given value.
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """

        # Remark
        # Since the sequence can consist of elements from different types,
        # they are stored into a dictionary in order to handle them as a
        # numeric sequence according to their order/position.
        super(OrdinalHyperparameter, self).__init__(name, meta)
        if len(sequence) > len(set(sequence)):
            raise ValueError(
                "Ordinal Hyperparameter Sequence %s contain duplicate values." % sequence)
        self.sequence = tuple(sequence)
        self.num_elements = len(sequence)
        self.sequence_vector = list(range(self.num_elements))
        self.default_value = self.check_default(default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)
        self.value_dict = OrderedDict()  # type: OrderedDict[Union[int, float, str], int]
        counter = 0
        for element in self.sequence:
            self.value_dict[element] = counter
            counter += 1

    def __hash__(self):
        return hash((self.name, self.sequence))

    def __repr__(self) -> str:
        """
        write out the parameter definition
        """
        repr_str = io.StringIO()
        repr_str.write("%s, Type: Ordinal, Sequence: {" % (self.name))
        for idx, seq in enumerate(self.sequence):
            repr_str.write(str(seq))
            if idx < len(self.sequence) - 1:
                repr_str.write(", ")
        repr_str.write("}")
        repr_str.write(", Default: ")
        repr_str.write(str(self.default_value))
        repr_str.seek(0)
        return repr_str.getvalue()

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

        return (
            self.name == other.name and
            self.sequence == other.sequence and
            self.default_value == other.default_value
        )

    def __copy__(self):
        return OrdinalHyperparameter(
                name=self.name,
                sequence=copy.deepcopy(self.sequence),
                default_value=self.default_value,
                meta=self.meta
            )

    cpdef int compare(self, value: Union[int, float, str], value2: Union[int, float, str]):
        if self.value_dict[value] < self.value_dict[value2]:
            return -1
        elif self.value_dict[value] > self.value_dict[value2]:
            return 1
        elif self.value_dict[value] == self.value_dict[value2]:
            return 0

    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2):
        if value < value2:
            return -1
        elif value > value2:
            return 1
        elif value == value2:
            return 0

    def is_legal(self, value: Union[int, float, str]) -> bool:
        """
        check if a certain value is represented in the sequence
        """
        return value in self.sequence

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return value in self.sequence_vector

    def check_default(self, default_value: Optional[Union[int, float, str]]
                      ) -> Union[int, float, str]:
        """
        check if given default value is represented in the sequence.
        If there's no default value we simply choose the
        first element in our sequence as default.
        """
        if default_value is None:
            return self.sequence[0]
        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        if np.isnan(vector).any():
            raise ValueError('Vector %s contains NaN\'s' % vector)

        if np.equal(np.mod(vector, 1), 0):
            return self.sequence[vector.astype(int)]

        raise ValueError('Can only index the choices of the ordinal '
                         'hyperparameter %s with an integer, but provided '
                         'the following float: %f' % (self, vector))

    def _transform_scalar(self, scalar: Union[float, int]) -> Union[float, int, str]:
        if scalar != scalar:
            raise ValueError('Number %s is NaN' % scalar)

        if scalar % 1 == 0:
            return self.sequence[int(scalar)]

        raise ValueError('Can only index the choices of the ordinal '
                         'hyperparameter %s with an integer, but provided '
                         'the following float: %f' % (self, scalar))

    def _transform(self, vector: Union[np.ndarray, float, int]
                   ) -> Optional[Union[np.ndarray, float, int]]:
        try:
            if isinstance(vector, np.ndarray):
                return self._transform_vector(vector)
            return self._transform_scalar(vector)
        except ValueError:
            return None

    def _inverse_transform(self, vector: Optional[Union[np.ndarray, List, int, str, float]]
                           ) -> Union[float, List[int], List[str], List[float]]:
        if vector is None:
            return np.NaN
        return self.sequence.index(vector)

    def get_seq_order(self) -> np.ndarray:
        """
        return the ordinal sequence as numeric sequence
        (according to the the ordering) from 1 to length of our sequence.
        """
        return np.arange(0, self.num_elements)

    def get_order(self, value: Optional[Union[int, str, float]]) -> int:
        """
        return the seuence position/order of a certain value from the sequence
        """
        return self.value_dict[value]

    def get_value(self, idx: int) -> Union[int, str, float]:
        """
        return the sequence value of a given order/position
        """
        return list(self.value_dict.keys())[list(self.value_dict.values()).index(idx)]

    def check_order(self, val1: Union[int, str, float], val2: Union[int, str, float]) -> bool:
        """
        check whether value1 is smaller than value2.
        """
        idx1 = self.get_order(val1)
        idx2 = self.get_order(val2)
        if idx1 < idx2:
            return True
        else:
            return False

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None) -> int:
        """
        return a random sample from our sequence as order/position index
        """
        return rs.randint(0, self.num_elements, size=size)

    def has_neighbors(self) -> bool:
        """
        check if there are neighbors or we're only dealing with an
        one-element sequence
        """
        return len(self.sequence) > 1

    def get_num_neighbors(self, value: Union[int, float, str]) -> int:
        """
        return the number of existing neighbors in the sequence
        """
        max_idx = len(self.sequence) - 1
        # check if there is only one value
        if value == self.sequence[0] and value == self.sequence[max_idx]:
            return 0
        elif value == self.sequence[0] or value == self.sequence[max_idx]:
            return 1
        else:
            return 2

    def get_neighbors(self, value: Union[int, str, float], rs: None, number: int = 0,
                      transform: bool = False) -> List[Union[str, float, int]]:
        """
        Return the neighbors of a given value.
        Value must be in vector form. Ordinal name will not work.
        """
        neighbors = []
        if transform:
            if self.get_num_neighbors(value) < len(self.sequence):
                index = self.get_order(value)
                neighbor_idx1 = index - 1
                neighbor_idx2 = index + 1
                seq = self.get_seq_order()

                if neighbor_idx1 >= seq[0]:
                    candidate1 = self.get_value(neighbor_idx1)
                    if self.check_order(candidate1, value):
                        neighbors.append(candidate1)
                if neighbor_idx2 < self.num_elements:
                    candidate2 = self.get_value(neighbor_idx2)
                    if self.check_order(value, candidate2):
                        neighbors.append(candidate2)

        else:
            if self.get_num_neighbors(self.get_value(value)) < len(self.sequence):
                index = value
                neighbor_idx1 = index - 1
                neighbor_idx2 = index + 1
                seq = self.get_seq_order()

                if neighbor_idx1 < index and neighbor_idx1 >= seq[0]:
                    neighbors.append(neighbor_idx1)
                if neighbor_idx2 > index and neighbor_idx2 < self.num_elements:
                    neighbors.append(neighbor_idx2)

        return neighbors

    def allow_greater_less_comparison(self) -> bool:
        return True

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the hyperparameter in
        the original hyperparameter space (the one specified by the user).
        For each parameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) hyperparameter
        space. Only legal values return a positive probability density,
        otherwise zero. The OrdinalHyperparameter is treated
        as a UniformHyperparameter with regard to its probability density.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        if vector.ndim != 1:
            raise ValueError("Method pdf expects a one-dimensional numpy array")
        return self._pdf(vector)

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the hyperparameter in
        the transformed (and possibly normalized, depends on the hyperparameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these). The OrdinalHyperparameter is treated
        as a UniformHyperparameter with regard to its probability density.

        Parameters
        ----------
        vector: np.ndarray
            the (N, ) vector of inputs for which the probability density
            function is to be computed.

        Returns
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        if not np.all(np.isin(vector, self.sequence)):
            raise ValueError(f'Some element in the vector {vector} is not in the sequence {self.sequence}.')
        return np.ones_like(vector, dtype=np.float64) / self.num_elements

    def get_max_density(self) -> float:
        return 1 / self.num_elements

    def get_size(self) -> float:
        return len(self.sequence)
