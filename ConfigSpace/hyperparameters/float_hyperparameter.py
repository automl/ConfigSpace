from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ConfigSpace.hyperparameters.numerical import NumericalHyperparameter


class FloatHyperparameter(NumericalHyperparameter):
    def __init__(
        self,
        name: str,
        default_value: Union[int, float],
        meta: Optional[dict] = None,
    ) -> None:
        super(FloatHyperparameter, self).__init__(name, default_value, meta)

    def is_legal(self, value: Union[int, float]) -> bool:
        raise NotImplementedError()

    def is_legal_vector(self, value) -> int:
        raise NotImplementedError()

    def check_default(self, default_value: Union[int, float]) -> float:
        raise NotImplementedError()

    def _transform(
        self,
        vector: Union[np.ndarray, float, int],
    ) -> Optional[Union[np.ndarray, float, int]]:
        try:
            if isinstance(vector, np.ndarray):
                return self._transform_vector(vector)
            return self._transform_scalar(vector)
        except ValueError:
            return None

    def _transform_scalar(self, scalar: float) -> float:
        raise NotImplementedError()

    def _transform_vector(self, vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

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
        -------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        if vector.ndim != 1:
            raise ValueError("Method pdf expects a one-dimensional numpy array")
        vector = self._inverse_transform(vector)
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
        raise NotImplementedError()

    def get_max_density(self) -> float:
        """
        Returns the maximal density on the pdf for the parameter (so not
        the mode, but the value of the pdf on the mode).
        """
        raise NotImplementedError()
