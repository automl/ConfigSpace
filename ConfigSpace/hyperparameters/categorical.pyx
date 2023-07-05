from collections import Counter
import copy
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


cdef class CategoricalHyperparameter(Hyperparameter):
    cdef public tuple choices
    cdef public tuple weights
    cdef public int num_choices
    cdef public tuple probabilities
    cdef list choices_vector
    cdef set _choices_set

    # TODO add more magic for automated type recognition
    # TODO move from list to tuple for choices argument
    def __init__(
        self,
        name: str,
        choices: Union[List[Union[str, float, int]], Tuple[Union[float, int, str]]],
        default_value: Union[int, float, str, None] = None,
        meta: Optional[Dict] = None,
        weights: Optional[Sequence[Union[int, float]]] = None
    ) -> None:
        """
        A categorical hyperparameter.

        Its values are sampled from a set of ``values``.

        ``None`` is a forbidden value, please use a string constant instead and parse
        it in your own code, see `here <https://github.com/automl/ConfigSpace/issues/159>_`
        for further details.

        >>> from ConfigSpace import CategoricalHyperparameter
        >>>
        >>> CategoricalHyperparameter('c', choices=['red', 'green', 'blue'])
        c, Type: Categorical, Choices: {red, green, blue}, Default: red

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        choices : list or tuple with str, float, int
            Collection of values to sample hyperparameter from
        default_value : int, float, str, optional
            Sets the default value of the hyperparameter to a given value
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        weights: Sequence[int | float] | None = None
            List of weights for the choices to be used (after normalization) as
            probabilities during sampling, no negative values allowed
        """

        super(CategoricalHyperparameter, self).__init__(name, meta)
        # TODO check that there is no bullshit in the choices!
        counter = Counter(choices)
        for choice in choices:
            if counter[choice] > 1:
                raise ValueError(
                    "Choices for categorical hyperparameters %s contain choice '%s' %d "
                    "times, while only a single oocurence is allowed."
                    % (name, choice, counter[choice])
                )
            if choice is None:
                raise TypeError("Choice 'None' is not supported")
        if isinstance(choices, set):
            raise TypeError("Using a set of choices is prohibited as it can result in "
                            "non-deterministic behavior. Please use a list or a tuple.")
        if isinstance(weights, set):
            raise TypeError("Using a set of weights is prohibited as it can result in "
                            "non-deterministic behavior. Please use a list or a tuple.")
        self.choices = tuple(choices)
        if weights is not None:
            self.weights = tuple(weights)
        else:
            self.weights = None
        self.probabilities = self._get_probabilities(choices=self.choices, weights=weights)
        self.num_choices = len(choices)
        self.choices_vector = list(range(self.num_choices))
        self._choices_set = set(self.choices_vector)
        self.default_value = self.check_default(default_value)
        self.normalized_default_value = self._inverse_transform(self.default_value)

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: Categorical, Choices: {" % (self.name))
        for idx, choice in enumerate(self.choices):
            repr_str.write(str(choice))
            if idx < len(self.choices) - 1:
                repr_str.write(", ")
        repr_str.write("}")
        repr_str.write(", Default: ")
        repr_str.write(str(self.default_value))
        # if the probability distribution is not uniform, write out the probabilities
        if not np.all(self.probabilities == self.probabilities[0]):
            repr_str.write(", Probabilities: %s" % str(self.probabilities))
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

        if self.probabilities is not None:
            ordered_probabilities_self = {
                choice: self.probabilities[i] for i, choice in enumerate(self.choices)
            }
        else:
            ordered_probabilities_self = None
        if other.probabilities is not None:
            ordered_probabilities_other = {
                choice: (
                    other.probabilities[other.choices.index(choice)]
                    if choice in other.choices else
                    None
                )
                for choice in self.choices
            }
        else:
            ordered_probabilities_other = None

        return (
            self.name == other.name and
            set(self.choices) == set(other.choices) and
            self.default_value == other.default_value and
            (
                (ordered_probabilities_self is None and ordered_probabilities_other is None) or
                ordered_probabilities_self == ordered_probabilities_other or
                (
                    ordered_probabilities_self is None
                    and len(np.unique(list(ordered_probabilities_other.values()))) == 1
                ) or
                (
                    ordered_probabilities_other is None
                    and len(np.unique(list(ordered_probabilities_self.values()))) == 1
                )
             )
        )

    def __hash__(self):
        return hash((self.name, self.choices))

    def __copy__(self):
        return CategoricalHyperparameter(
            name=self.name,
            choices=copy.deepcopy(self.choices),
            default_value=self.default_value,
            weights=copy.deepcopy(self.weights),
            meta=self.meta
        )

    def to_uniform(self) -> "CategoricalHyperparameter":
        """
        Creates a categorical parameter with equal weights for all choices
        This is used for the uniform configspace when sampling configurations in the local search
        in PiBO: https://openreview.net/forum?id=MMAeCXIa89

        Returns
        ----------
        CategoricalHyperparameter
            An identical parameter as the original, except that all weights are uniform.
        """
        return CategoricalHyperparameter(
            name=self.name,
            choices=copy.deepcopy(self.choices),
            default_value=self.default_value,
            meta=self.meta
        )

    cpdef int compare(self, value: Union[int, float, str], value2: Union[int, float, str]):
        if value == value2:
            return 0
        else:
            return 1

    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2):
        if value == value2:
            return 0
        else:
            return 1

    def is_legal(self, value: Union[None, str, float, int]) -> bool:
        if value in self.choices:
            return True
        else:
            return False

    cpdef bint is_legal_vector(self, DTYPE_t value):
        return value in self._choices_set

    def _get_probabilities(self, choices: Tuple[Union[None, str, float, int]],
                           weights: Union[None, List[float]]) -> Union[None, List[float]]:
        if weights is None:
            return tuple(np.ones(len(choices)) / len(choices))

        if len(weights) != len(choices):
            raise ValueError(
                "The list of weights and the list of choices are required to be of same length.")

        weights = np.array(weights)

        if np.all(weights == 0):
            raise ValueError("At least one weight has to be strictly positive.")

        if np.any(weights < 0):
            raise ValueError("Negative weights are not allowed.")

        return tuple(weights / np.sum(weights))

    def check_default(self, default_value: Union[None, str, float, int]
                      ) -> Union[str, float, int]:
        if default_value is None:
            return self.choices[np.argmax(self.weights) if self.weights is not None else 0]
        elif self.is_legal(default_value):
            return default_value
        else:
            raise ValueError("Illegal default value %s" % str(default_value))

    def _sample(self, rs: np.random.RandomState, size: Optional[int] = None
                ) -> Union[int, np.ndarray]:
        return rs.choice(a=self.num_choices, size=size, replace=True, p=self.probabilities)

    cpdef np.ndarray _transform_vector(self, np.ndarray vector):
        if np.isnan(vector).any():
            raise ValueError('Vector %s contains NaN\'s' % vector)

        if np.equal(np.mod(vector, 1), 0):
            return self.choices[vector.astype(int)]

        raise ValueError("Can only index the choices of the ordinal "
                         "hyperparameter %s with an integer, but provided "
                         "the following float: %f" % (self, vector))

    def _transform_scalar(self, scalar: Union[float, int]) -> Union[float, int, str]:
        if scalar != scalar:
            raise ValueError("Number %s is NaN" % scalar)

        if scalar % 1 == 0:
            return self.choices[int(scalar)]

        raise ValueError("Can only index the choices of the ordinal "
                         "hyperparameter %s with an integer, but provided "
                         "the following float: %f" % (self, scalar))

    def _transform(self, vector: Union[np.ndarray, float, int, str]
                   ) -> Optional[Union[np.ndarray, float, int]]:
        try:
            if isinstance(vector, np.ndarray):
                return self._transform_vector(vector)
            return self._transform_scalar(vector)
        except ValueError:
            return None

    def _inverse_transform(self, vector: Union[None, str, float, int]) -> Union[int, float]:
        if vector is None:
            return np.NaN
        return self.choices.index(vector)

    def has_neighbors(self) -> bool:
        return len(self.choices) > 1

    def get_num_neighbors(self, value = None) -> int:
        return len(self.choices) - 1

    def get_neighbors(self, value: int, rs: np.random.RandomState,
                      number: Union[int, float] = np.inf, transform: bool = False
                      ) -> List[Union[float, int, str]]:
        neighbors = []  # type: List[Union[float, int, str]]
        if number < len(self.choices):
            while len(neighbors) < number:
                rejected = True
                index = int(value)
                while rejected:
                    neighbor_idx = rs.randint(0, self.num_choices)
                    if neighbor_idx != index:
                        rejected = False

                if transform:
                    candidate = self._transform(neighbor_idx)
                else:
                    candidate = float(neighbor_idx)

                if candidate in neighbors:
                    continue
                else:
                    neighbors.append(candidate)
        else:
            for candidate_idx, candidate_value in enumerate(self.choices):
                if int(value) == candidate_idx:
                    continue
                else:
                    if transform:
                        candidate = self._transform(candidate_idx)
                    else:
                        candidate = float(candidate_idx)

                    neighbors.append(candidate)

        return neighbors

    def allow_greater_less_comparison(self) -> bool:
        raise ValueError("Parent hyperparameter in a > or < "
                         "condition must be a subclass of "
                         "NumericalHyperparameter or "
                         "OrdinalHyperparameter, but is "
                         "<cdef class 'ConfigSpace.hyperparameters.CategoricalHyperparameter'>")

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the original parameter space (the one specified by the user).
        For each parameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) parameter
        space. Only legal values return a positive probability density,
        otherwise zero.

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
        # this check is to ensure shape is right (and np.shape does not work in cython)
        if vector.ndim != 1:
            raise ValueError("Method pdf expects a one-dimensional numpy array")
        vector = np.array(self._inverse_transform(vector))
        return self._pdf(vector)

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these). For categoricals, each vector gets
        transformed to its corresponding index (but in float form). To be
        able to retrieve the element corresponding to the index, the float
        must be cast to int.

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
        probs = np.array(self.probabilities)
        nan = np.isnan(vector)
        if np.any(nan):
            # Temporarily pick any valid index to use `vector` as an index for `probs`
            vector[nan] = 0
        res = np.array(probs[vector.astype(int)])
        if np.any(nan):
            res[nan] = 0
        if res.ndim == 0:
            return res.reshape(-1)
        return res

    def get_max_density(self) -> float:
        return np.max(self.probabilities)

    def get_size(self) -> float:
        return len(self.choices)
