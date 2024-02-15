from __future__ import annotations

import copy
import io
from typing import Any

import numpy as np

from ConfigSpace.hyperparameters.hyperparameter import Comparison, Hyperparameter


class OrdinalHyperparameter(Hyperparameter):
    def __init__(
        self,
        name: str,
        sequence: list[float | int | str] | tuple[float | int | str],
        default_value: str | int | float | None = None,
        meta: dict | None = None,
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
        super().__init__(name, meta)
        if len(sequence) > len(set(sequence)):
            raise ValueError(
                "Ordinal Hyperparameter Sequence %s contain duplicate values." % sequence,
            )
        self.sequence = tuple(sequence)
        self.num_elements = len(sequence)
        self.sequence_vector = list(range(self.num_elements))
        self.default_value = self.check_default(default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)
        self.value_dict = {e: i for i, e in enumerate(self.sequence)}

    def __hash__(self):
        return hash((self.name, self.sequence))

    def __repr__(self) -> str:
        """Write out the parameter definition."""
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
        """Comparison between self and another object.

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
            self.name == other.name
            and self.sequence == other.sequence
            and self.default_value == other.default_value
        )

    def __copy__(self):
        return OrdinalHyperparameter(
            name=self.name,
            sequence=copy.deepcopy(self.sequence),
            default_value=self.default_value,
            meta=self.meta,
        )

    def compare(self, value: int | float | str, value2: int | float | str) -> Comparison:
        if self.value_dict[value] < self.value_dict[value2]:
            return Comparison.LESS_THAN

        if self.value_dict[value] > self.value_dict[value2]:
            return Comparison.GREATER_THAN

        return Comparison.EQUAL

    def compare_vector(self, value, value2) -> Comparison:
        if value < value2:
            return Comparison.LESS_THAN
        if value > value2:
            return Comparison.GREATER_THAN

        return Comparison.EQUAL

    def is_legal(self, value: int | float | str) -> bool:
        """Check if a certain value is represented in the sequence."""
        return value in self.sequence

    def is_legal_vector(self, value) -> int:
        return value in self.sequence_vector

    def check_default(
        self,
        default_value: int | float | str | None,
    ) -> int | float | str:
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

    def _transform_vector(self, vector: np.ndarray) -> np.ndarray:
        if np.isnan(vector).any():
            raise ValueError("Vector %s contains NaN's" % vector)

        if np.equal(np.mod(vector, 1), 0):
            return self.sequence[vector.astype(int)]

        raise ValueError(
            "Can only index the choices of the ordinal "
            f"hyperparameter {self} with an integer, but provided "
            f"the following float: {vector:f}",
        )

    def _transform_scalar(self, scalar: float | int) -> float | int | str:
        if scalar != scalar:
            raise ValueError("Number %s is NaN" % scalar)

        if scalar % 1 == 0:
            return self.sequence[int(scalar)]

        raise ValueError(
            "Can only index the choices of the ordinal "
            f"hyperparameter {self} with an integer, but provided "
            f"the following float: {scalar:f}",
        )

    def _transform(
        self,
        vector: np.ndarray | float | int,
    ) -> np.ndarray | float | int | None:
        try:
            if isinstance(vector, np.ndarray):
                return self._transform_vector(vector)
            return self._transform_scalar(vector)
        except ValueError:
            return None

    def _inverse_transform(
        self,
        vector: np.ndarray | list | int | str | float | None,
    ) -> float | list[int] | list[str] | list[float]:
        if vector is None:
            return np.NaN
        return self.sequence.index(vector)

    def get_seq_order(self) -> np.ndarray:
        """
        return the ordinal sequence as numeric sequence
        (according to the the ordering) from 1 to length of our sequence.
        """
        return np.arange(0, self.num_elements)

    def get_order(self, value: int | str | float | None) -> int:
        """Return the seuence position/order of a certain value from the sequence."""
        return self.value_dict[value]

    def get_value(self, idx: int) -> int | str | float:
        """Return the sequence value of a given order/position."""
        return list(self.value_dict.keys())[list(self.value_dict.values()).index(idx)]

    def check_order(self, val1: int | str | float, val2: int | str | float) -> bool:
        """Check whether value1 is smaller than value2."""
        idx1 = self.get_order(val1)
        idx2 = self.get_order(val2)
        return idx1 < idx2

    def _sample(self, rs: np.random.RandomState, size: int | None = None) -> int:
        """Return a random sample from our sequence as order/position index."""
        return rs.randint(0, self.num_elements, size=size)

    def has_neighbors(self) -> bool:
        """
        check if there are neighbors or we're only dealing with an
        one-element sequence.
        """
        return len(self.sequence) > 1

    def get_num_neighbors(self, value: int | float | str) -> int:
        """Return the number of existing neighbors in the sequence."""
        max_idx = len(self.sequence) - 1
        # check if there is only one value
        if value == self.sequence[0] and value == self.sequence[max_idx]:
            return 0
        elif value in (self.sequence[0], self.sequence[max_idx]):
            return 1
        else:
            return 2

    def get_neighbors(
        self,
        value: int | str | float,
        rs: None,
        number: int = 0,
        transform: bool = False,
    ) -> list[str | float | int]:
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
        -------
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
        -------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        if not np.all(np.isin(vector, self.sequence)):
            raise ValueError(
                f"Some element in the vector {vector} is not in the sequence {self.sequence}.",
            )
        return np.ones_like(vector, dtype=np.float64) / self.num_elements

    def get_max_density(self) -> float:
        return 1 / self.num_elements

    def get_size(self) -> float:
        return len(self.sequence)
