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
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from ConfigSpace import OrdinalHyperparameter
from ConfigSpace.forbidden import (
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenEqualsRelation,
    ForbiddenGreaterThanRelation,
    ForbiddenInClause,
    ForbiddenLessThanRelation,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


def test_forbidden_equals_clause():
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    hp3 = CategoricalHyperparameter("grandchild", ["hot", "cold"])

    with pytest.raises(ValueError):
        ForbiddenEqualsClause(hp1, 2)

    forb1 = ForbiddenEqualsClause(hp1, 1)
    forb1_ = ForbiddenEqualsClause(hp1, 1)
    forb1__ = ForbiddenEqualsClause(hp1, 0)
    forb2 = ForbiddenEqualsClause(hp2, 10)
    forb3 = ForbiddenEqualsClause(hp3, "hot")
    forb3_ = ForbiddenEqualsClause(hp3, "hot")

    assert forb3 == forb3_
    assert forb1 == forb1_
    assert forb1 != "forb1"
    assert forb1 != forb2
    assert forb1__ != forb1
    assert str(forb1) == "Forbidden: parent == 1"

    assert not forb1.is_forbidden_value({"child": 1})
    assert forb1.is_forbidden_value({"parent": 1})

    assert forb3.is_forbidden_value({"grandchild": "hot"})
    assert not forb3.is_forbidden_value({"grandchild": "cold"})

    # Test forbidden on vector values
    hyperparameter_idx = {hp1.name: 0, hp2.name: 1}
    forb1.set_vector_idx(hyperparameter_idx)
    assert not forb1.is_forbidden_vector(np.array([np.nan, np.nan]))
    assert not forb1.is_forbidden_vector(np.array([0.0, np.nan]))
    assert forb1.is_forbidden_vector(np.array([1.0, np.nan]))


def test_in_condition():
    hp1 = CategoricalHyperparameter("parent", [0, 1, 2, 3, 4])
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    hp3 = UniformIntegerHyperparameter("child2", 0, 10)
    hp4 = CategoricalHyperparameter("grandchild", ["hot", "cold", "warm"])

    with pytest.raises(ValueError):
        ForbiddenInClause(hp1, [5])

    forb1 = ForbiddenInClause(hp2, [5, 6, 7, 8, 9])
    forb1_ = ForbiddenInClause(hp2, [9, 8, 7, 6, 5])
    forb2 = ForbiddenInClause(hp2, [5, 6, 7, 8])
    forb3 = ForbiddenInClause(hp3, [5, 6, 7, 8, 9])
    forb4 = ForbiddenInClause(hp4, ["hot", "cold"])
    forb4_ = ForbiddenInClause(hp4, ["hot", "cold"])
    forb5 = ForbiddenInClause(hp1, [3, 4])
    forb5_ = ForbiddenInClause(hp1, [3, 4])

    assert forb5 == forb5_
    assert forb4 == forb4_

    assert forb1 == forb1_
    assert forb1 != forb2
    assert forb1 != forb3
    assert str(forb1) == "Forbidden: child in {5, 6, 7, 8, 9}"
    assert not forb1.is_forbidden_value({"parent": 1})

    for i in range(5):
        assert not forb1.is_forbidden_value({"child": i})
    for i in range(5, 10):
        assert forb1.is_forbidden_value({"child": i})

    assert forb4.is_forbidden_value({"grandchild": "hot"})
    assert forb4.is_forbidden_value({"grandchild": "cold"})
    assert not forb4.is_forbidden_value({"grandchild": "warm"})

    # Test forbidden on vector values
    hyperparameter_idx = {hp1.name: 0, hp2.name: 1}
    forb1.set_vector_idx(hyperparameter_idx)
    assert not forb1.is_forbidden_vector(np.array([np.nan, np.nan]))
    assert not forb1.is_forbidden_vector(np.array([np.nan, 0]))
    correct_vector_value = hp2.to_vector(np.int64(6))
    assert forb1.is_forbidden_vector(np.array([np.nan, correct_vector_value]))


def test_and_conjunction():
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    hp2 = UniformIntegerHyperparameter("child", 0, 2)
    hp3 = UniformIntegerHyperparameter("child2", 0, 2)
    hp4 = UniformIntegerHyperparameter("child3", 0, 2)

    forb2 = ForbiddenEqualsClause(hp1, 1)
    forb3 = ForbiddenInClause(hp2, range(2, 3))
    forb4 = ForbiddenInClause(hp3, range(2, 3))
    forb5 = ForbiddenInClause(hp4, range(2, 3))

    and1 = ForbiddenAndConjunction(forb2, forb3)
    and2 = ForbiddenAndConjunction(forb2, forb4)
    and3 = ForbiddenAndConjunction(forb2, forb5)

    total_and = ForbiddenAndConjunction(and1, and2, and3)
    assert (
        str(total_and)
        == "((Forbidden: parent == 1 && Forbidden: child in {2}) && (Forbidden: parent == 1 && Forbidden: child2 in {2}) && (Forbidden: parent == 1 && Forbidden: child3 in {2}))"
    )

    results = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]

    for i, values in enumerate(product(range(2), range(3), range(3), range(3))):
        is_forbidden = total_and.is_forbidden_value(
            {
                "parent": values[0],
                "child": values[1],
                "child2": values[2],
                "child3": values[3],
            },
        )

        assert results[i] == is_forbidden

        assert not total_and.is_forbidden_value({})


def test_relation():
    hp1 = CategoricalHyperparameter("cat_int", [0, 1])
    hp2 = OrdinalHyperparameter("ord_int", [0, 1])
    hp3 = UniformIntegerHyperparameter("int", 0, 2)
    hp4 = UniformIntegerHyperparameter("int2", 0, 2)
    hp5 = UniformFloatHyperparameter("float", 0, 2)
    hp6 = CategoricalHyperparameter("str", ["a", "b"])
    hp7 = CategoricalHyperparameter("str2", ["b", "c"])

    forb = ForbiddenEqualsRelation(hp1, hp2)
    assert forb.is_forbidden_value({"cat_int": 1, "ord_int": 1})
    assert not forb.is_forbidden_value({"cat_int": 0, "ord_int": 1})

    forb = ForbiddenEqualsRelation(hp1, hp3)
    assert forb.is_forbidden_value({"cat_int": 1, "int": 1})
    assert not forb.is_forbidden_value({"cat_int": 0, "int": 1})

    forb = ForbiddenEqualsRelation(hp3, hp4)
    assert forb.is_forbidden_value({"int": 1, "int2": 1})
    assert not forb.is_forbidden_value({"int": 1, "int2": 0})

    forb = ForbiddenLessThanRelation(hp3, hp4)
    assert forb.is_forbidden_value({"int": 0, "int2": 1})
    assert not forb.is_forbidden_value({"int": 1, "int2": 1})
    assert not forb.is_forbidden_value({"int": 1, "int2": 0})

    forb = ForbiddenGreaterThanRelation(hp3, hp4)
    assert forb.is_forbidden_value({"int": 1, "int2": 0})
    assert not forb.is_forbidden_value({"int": 1, "int2": 1})
    assert not forb.is_forbidden_value({"int": 0, "int2": 1})

    forb = ForbiddenGreaterThanRelation(hp4, hp5)
    assert forb.is_forbidden_value({"int2": 1, "float": 0})
    assert not forb.is_forbidden_value({"int2": 1, "float": 1})
    assert not forb.is_forbidden_value({"int2": 0, "float": 1})

    forb = ForbiddenGreaterThanRelation(hp5, hp6)
    with pytest.raises(TypeError):
        forb.is_forbidden_value({"float": 1, "str": "b"})

    forb = ForbiddenGreaterThanRelation(hp5, hp7)
    with pytest.raises(TypeError):
        forb.is_forbidden_value({"float": 1, "str2": "b"})

    forb = ForbiddenGreaterThanRelation(hp6, hp7)
    assert forb.is_forbidden_value({"str": "b", "str2": "a"})
    assert forb.is_forbidden_value({"str": "c", "str2": "a"})

    forb1 = ForbiddenEqualsRelation(hp2, hp3)
    forb2 = ForbiddenEqualsRelation(hp2, hp3)
    forb3 = ForbiddenEqualsRelation(hp3, hp4)
    assert forb1 == forb2
    assert forb2 != forb3

    hp1 = OrdinalHyperparameter(
        "water_temperature",
        ["cold", "luke-warm", "hot", "boiling"],
    )
    hp2 = OrdinalHyperparameter(
        "water_temperature2",
        ["cold", "luke-warm", "hot", "boiling"],
    )
    forb = ForbiddenGreaterThanRelation(hp1, hp2)
    assert not forb.is_forbidden_value(
        {"water_temperature": "boiling", "water_temperature2": "cold"},
    )
    assert forb.is_forbidden_value(
        {"water_temperature": "hot", "water_temperature2": "cold"},
    )


def test_relation_conditioned():
    from ConfigSpace import ConfigurationSpace, EqualsCondition

    a = OrdinalHyperparameter("a", [2, 5, 10])
    enable_a = CategoricalHyperparameter("enable_a", [False, True], weights=[99999, 1])
    cond_a = EqualsCondition(a, enable_a, True)

    b = OrdinalHyperparameter("b", [5, 10, 15])
    for forbid in (
        ForbiddenEqualsRelation,
        ForbiddenGreaterThanRelation,
        ForbiddenLessThanRelation,
    ):
        forbid_a_b = forbid(a, b)

        cs = ConfigurationSpace()
        cs.add([a, enable_a, cond_a, b, forbid_a_b])
        cs.sample_configuration(100)
