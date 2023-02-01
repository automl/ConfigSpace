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
from ConfigSpace.hyperparameters_.hyperparameter cimport Hyperparameter
from ConfigSpace.hyperparameters_.numerical cimport NumericalHyperparameter
from ConfigSpace.hyperparameters_.integer_hyperparameter cimport IntegerHyperparameter
from ConfigSpace.hyperparameters_.float_hyperparameter cimport FloatHyperparameter
from ConfigSpace.hyperparameters_.uniform_float cimport UniformFloatHyperparameter
from ConfigSpace.hyperparameters_.uniform_integer cimport UniformIntegerHyperparameter
from ConfigSpace.hyperparameters_.normal_float cimport NormalFloatHyperparameter
from ConfigSpace.hyperparameters_ import (
    Constant,
    UnParametrizedHyperparameter,
    OrdinalHyperparameter,
    CategoricalHyperparameter,
)

# OPTIM: Some operations generate an arange which could blowup memory if
# done over the entire space of integers (int32/64).
# To combat this, `arange_chunked` is used in scenarios where reducion
# operations over all the elments could be done in partial steps independantly.
# For example, a sum over the pdf values could be done in chunks.
# This may add some small overhead for smaller ranges but is unlikely to
# be noticable.
ARANGE_CHUNKSIZE = 10_000_000
