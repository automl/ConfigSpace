from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


class Constant(Hyperparameter):
    def __init__(
        self,
        name: str,
        value: Union[str, int, float],
        meta: Optional[dict] = None,
    ) -> None:
        """
        Representing a constant hyperparameter in the configuration space.

        By sampling from the configuration space each time only a single,
        constant ``value`` will be drawn from this hyperparameter.

        Parameters
        ----------
        name : str
            Name of the hyperparameter, with which it can be accessed
        value : str, int, float
            value to sample hyperparameter from
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        super(Constant, self).__init__(name, meta)
        allowed_types = (int, float, str)

        if not isinstance(value, allowed_types) or isinstance(value, bool):
            raise TypeError(
                "Constant value is of type %s, but only the "
                "following types are allowed: %s" % (type(value), allowed_types),
            )  # type: ignore

        self.value = value
        self.value_vector = 0.0
        self.default_value = value
        self.normalized_default_value = 0.0

    def __repr__(self) -> str:
        repr_str = ["%s" % self.name, "Type: Constant", "Value: %s" % self.value]
        return ", ".join(repr_str)

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
            self.value == other.value
            and self.name == other.name
            and self.default_value == other.default_value
        )

    def __copy__(self):
        return Constant(self.name, self.value, meta=self.meta)

    def __hash__(self):
        return hash((self.name, self.value))

    def is_legal(self, value: Union[str, int, float]) -> bool:
        return value == self.value

    def is_legal_vector(self, value) -> int:
        return value == self.value_vector

    def _sample(self, rs: None, size: Optional[int] = None) -> Union[int, np.ndarray]:
        return 0 if size == 1 else np.zeros((size,))

    def _transform(
        self, vector: Optional[Union[np.ndarray, float, int]],
    ) -> Optional[Union[np.ndarray, float, int]]:
        return self.value

    def _transform_vector(
        self, vector: Optional[np.ndarray],
    ) -> Optional[Union[np.ndarray, float, int]]:
        return self.value

    def _transform_scalar(
        self, vector: Optional[Union[float, int]],
    ) -> Optional[Union[np.ndarray, float, int]]:
        return self.value

    def _inverse_transform(
        self,
        vector: Union[np.ndarray, float, int],
    ) -> Union[np.ndarray, int, float]:
        if vector != self.value:
            return np.NaN
        return 0

    def has_neighbors(self) -> bool:
        return False

    def get_num_neighbors(self, value=None) -> int:
        return 0

    def get_neighbors(
        self, value: Any, rs: np.random.RandomState, number: int, transform: bool = False,
    ) -> list:
        return []

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the parameter in
        the original parameter space (the one specified by the user).
        For each hyperparameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) parameter
        space. Only legal values return a positive probability density,
        otherwise zero.

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
        Computes the probability density function of the parameter in
        the transformed (and possibly normalized, depends on the parameter
        type) space. As such, one never has to worry about log-normal
        distributions, only normal distributions (as the inverse_transform
        in the pdf method handles these).

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
        return (vector == self.value).astype(float)

    def get_max_density(self):
        return 1.0

    def get_size(self) -> float:
        return 1.0


class UnParametrizedHyperparameter(Constant):
    pass
