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
import copy
import io
import math
import warnings
from collections import OrderedDict, Counter
from itertools import count
from more_itertools import roundrobin, duplicates_everseen
from typing import List, Any, Dict, Union, Set, Tuple, Optional, Sequence

from scipy.stats import truncnorm, beta as spbeta, norm
import numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
cimport numpy as np
np.import_array()

from ConfigSpace.functional import center_range, arange_chunked

# OPTIM: Some operations generate an arange which could blowup memory if
# done over the entire space of integers (int32/64).
# To combat this, `arange_chunked` is used in scenarios where reducion
# operations over all the elments could be done in partial steps independantly.
# For example, a sum over the pdf values could be done in chunks.
# This may add some small overhead for smaller ranges but is unlikely to
# be noticable.
ARANGE_CHUNKSIZE = 10_000_000


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
            raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                             ' instance' % seed)

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
