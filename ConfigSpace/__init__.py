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

from ConfigSpace.__version__ import __version__
from ConfigSpace.__authors__ import __authors__

from ConfigSpace.api import (Beta, Categorical, Distribution, Float, Integer,
                             Normal, Uniform)
from ConfigSpace.conditions import (AndConjunction, EqualsCondition,
                                    GreaterThanCondition, InCondition,
                                    LessThanCondition, NotEqualsCondition,
                                    OrConjunction)
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.forbidden import (ForbiddenAndConjunction,
                                   ForbiddenEqualsClause,
                                   ForbiddenEqualsRelation,
                                   ForbiddenGreaterThanRelation,
                                   ForbiddenInClause,
                                   ForbiddenLessThanRelation)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UnParametrizedHyperparameter,
)
from ConfigSpace.hyperparameters_ import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    NormalIntegerHyperparameter,
    NormalFloatHyperparameter,
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
)
import ConfigSpace.api.distributions as distributions
import ConfigSpace.api.types as types

__all__ = [
    "__authors__",
    "__version__",
    "Configuration",
    "ConfigurationSpace",
    "CategoricalHyperparameter",
    "UniformFloatHyperparameter",
    "UniformIntegerHyperparameter",
    "BetaFloatHyperparameter",
    "BetaIntegerHyperparameter",
    "NormalFloatHyperparameter",
    "NormalIntegerHyperparameter",
    "Constant",
    "UnParametrizedHyperparameter",
    "OrdinalHyperparameter",
    "AndConjunction",
    "OrConjunction",
    "EqualsCondition",
    "NotEqualsCondition",
    "InCondition",
    "GreaterThanCondition",
    "LessThanCondition",
    "ForbiddenAndConjunction",
    "ForbiddenEqualsClause",
    "ForbiddenInClause",
    "ForbiddenLessThanRelation",
    "ForbiddenEqualsRelation",
    "ForbiddenGreaterThanRelation",
    "Beta",
    "Categorical",
    "Distribution",
    "Float",
    "Integer",
    "Normal",
    "Uniform",
    "distributions",
    "types",
]
