from typing import Dict, Optional, Union

import numpy as np
cimport numpy as np
np.import_array()


cdef class Hyperparameter(object):

    def __init__(self, name: str, meta: Optional[Dict]) -> None:
        if not isinstance(name, str):
            raise TypeError(
                "The name of a hyperparameter must be an instance of"
                " %s, but is %s." % (str(str), type(name)))
        self.name: str = name
        self.meta = meta

    def __repr__(self):
        raise NotImplementedError()

    def is_legal(self, value):
        raise NotImplementedError()

    cpdef bint is_legal_vector(self, DTYPE_t value):
        """
        Check whether the given value is a legal value for the vector
        representation of this hyperparameter.

        Parameters
        ----------
        value
            the vector value to check

        Returns
        -------
        bool
            True if the given value is a legal vector value, otherwise False

        """
        raise NotImplementedError()

    def sample(self, rs):
        vector = self._sample(rs)
        return self._transform(vector)

    def rvs(
        self,
        size: Optional[int] = None,
        random_state: Optional[Union[int, np.random, np.random.RandomState]] = None
    ) -> Union[float, np.ndarray]:
        """
        scipy compatibility wrapper for ``_sample``,
        allowing the hyperparameter to be used in sklearn API
        hyperparameter searchers, eg. GridSearchCV.

        """

        # copy-pasted from scikit-learn utils/validation.py
        def check_random_state(seed):
            """
            Turn seed into a np.random.RandomState instance
            If seed is None (or np.random), return the RandomState singleton used
            by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            If seed is a new-style np.random.Generator, return it.
            Otherwise, raise ValueError.

            """
            if seed is None or seed is np.random:
                return np.random.mtrand._rand
            if isinstance(seed, (int, np.integer)):
                return np.random.RandomState(seed)
            if isinstance(seed, np.random.RandomState):
                return seed
            try:
                # Generator is only available in numpy >= 1.17
                if isinstance(seed, np.random.Generator):
                    return seed
            except AttributeError:
                pass
            raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                             " instance" % seed)

        # if size=None, return a value, but if size=1, return a 1-element array

        vector = self._sample(
            rs=check_random_state(random_state),
            size=size if size is not None else 1
        )
        if size is None:
            vector = vector[0]

        return self._transform(vector)

    def _sample(self, rs, size):
        raise NotImplementedError()

    def _transform(
        self,
        vector: Union[np.ndarray, float, int]
    ) -> Optional[Union[np.ndarray, float, int]]:
        raise NotImplementedError()

    def _inverse_transform(self, vector):
        raise NotImplementedError()

    def has_neighbors(self):
        raise NotImplementedError()

    def get_neighbors(self, value, rs, number, transform = False):
        raise NotImplementedError()

    def get_num_neighbors(self, value):
        raise NotImplementedError()

    cpdef int compare_vector(self, DTYPE_t value, DTYPE_t value2):
        raise NotImplementedError()

    def pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the hyperparameter in
        the hyperparameter space (the one specified by the user).
        For each hyperparameter type, there is also a method _pdf which
        operates on the transformed (and possibly normalized) hyperparameter
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
        raise NotImplementedError()

    def _pdf(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function of the hyperparameter in
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
        ----------
        np.ndarray(N, )
            Probability density values of the input vector
        """
        raise NotImplementedError()

    def get_size(self) -> float:
        raise NotImplementedError()
