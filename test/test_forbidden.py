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

import unittest
from itertools import product

import numpy as np

from ConfigSpace import OrdinalHyperparameter

#     ForbiddenInClause, ForbiddenAndConjunction
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


class TestForbidden(unittest.TestCase):
    # TODO: return only copies of the objects!

    def test_forbidden_equals_clause(self):
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        hp3 = CategoricalHyperparameter("grandchild", ["hot", "cold"])

        self.assertRaisesRegex(
            ValueError,
            r"Forbidden clause must be instantiated with a legal hyperparameter value for "
            r"'parent, Type: Categorical, Choices: \{0, 1\}, Default: 0', but got '2'",
            ForbiddenEqualsClause,
            hp1,
            2,
        )

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

        self.assertRaisesRegex(
            ValueError,
            "Is_forbidden must be called with the "
            "instantiated hyperparameter in the "
            "forbidden clause; you are missing "
            "'parent'",
            forb1.is_forbidden,
            {1: hp2},
            True,
        )
        assert not forb1.is_forbidden({"child": 1}, strict=False)
        assert not forb1.is_forbidden({"parent": 0}, True)
        assert forb1.is_forbidden({"parent": 1}, True)

        assert forb3.is_forbidden({"grandchild": "hot"}, True)
        assert not forb3.is_forbidden({"grandchild": "cold"}, True)

        # Test forbidden on vector values
        hyperparameter_idx = {hp1.name: 0, hp2.name: 1}
        forb1.set_vector_idx(hyperparameter_idx)
        assert not forb1.is_forbidden_vector(np.array([np.NaN, np.NaN]), strict=False)
        assert not forb1.is_forbidden_vector(np.array([0.0, np.NaN]), strict=False)
        assert forb1.is_forbidden_vector(np.array([1.0, np.NaN]), strict=False)

    def test_in_condition(self):
        hp1 = CategoricalHyperparameter("parent", [0, 1, 2, 3, 4])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        hp3 = UniformIntegerHyperparameter("child2", 0, 10)
        hp4 = CategoricalHyperparameter("grandchild", ["hot", "cold", "warm"])

        self.assertRaisesRegex(
            ValueError,
            "Forbidden clause must be instantiated with a "
            "legal hyperparameter value for "
            "'parent, Type: Categorical, Choices: {0, 1, 2, 3, 4}, "
            "Default: 0', but got '5'",
            ForbiddenInClause,
            hp1,
            [5],
        )

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
        self.assertRaisesRegex(
            ValueError,
            "Is_forbidden must be called with the "
            "instantiated hyperparameter in the "
            "forbidden clause; you are missing "
            "'child'",
            forb1.is_forbidden,
            {"parent": 1},
            True,
        )
        assert not forb1.is_forbidden({"parent": 1}, strict=False)
        for i in range(0, 5):
            assert not forb1.is_forbidden({"child": i}, True)
        for i in range(5, 10):
            assert forb1.is_forbidden({"child": i}, True)

        assert forb4.is_forbidden({"grandchild": "hot"}, True)
        assert forb4.is_forbidden({"grandchild": "cold"}, True)
        assert not forb4.is_forbidden({"grandchild": "warm"}, True)

        # Test forbidden on vector values
        hyperparameter_idx = {hp1.name: 0, hp2.name: 1}
        forb1.set_vector_idx(hyperparameter_idx)
        assert not forb1.is_forbidden_vector(np.array([np.NaN, np.NaN]), strict=False)
        assert not forb1.is_forbidden_vector(np.array([np.NaN, 0]), strict=False)
        correct_vector_value = hp2._inverse_transform(6)
        assert forb1.is_forbidden_vector(np.array([np.NaN, correct_vector_value]), strict=False)

    def test_and_conjunction(self):
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
            is_forbidden = total_and.is_forbidden(
                {"parent": values[0], "child": values[1], "child2": values[2], "child3": values[3]},
                True,
            )

            assert results[i] == is_forbidden

            assert not total_and.is_forbidden({}, strict=False)

    def test_relation(self):
        hp1 = CategoricalHyperparameter("cat_int", [0, 1])
        hp2 = OrdinalHyperparameter("ord_int", [0, 1])
        hp3 = UniformIntegerHyperparameter("int", 0, 2)
        hp4 = UniformIntegerHyperparameter("int2", 0, 2)
        hp5 = UniformFloatHyperparameter("float", 0, 2)
        hp6 = CategoricalHyperparameter("str", ["a", "b"])
        hp7 = CategoricalHyperparameter("str2", ["b", "c"])

        forb = ForbiddenEqualsRelation(hp1, hp2)
        assert forb.is_forbidden({"cat_int": 1, "ord_int": 1}, True)
        assert not forb.is_forbidden({"cat_int": 0, "ord_int": 1}, True)

        forb = ForbiddenEqualsRelation(hp1, hp3)
        assert forb.is_forbidden({"cat_int": 1, "int": 1}, True)
        assert not forb.is_forbidden({"cat_int": 0, "int": 1}, True)

        forb = ForbiddenEqualsRelation(hp3, hp4)
        assert forb.is_forbidden({"int": 1, "int2": 1}, True)
        assert not forb.is_forbidden({"int": 1, "int2": 0}, True)

        forb = ForbiddenLessThanRelation(hp3, hp4)
        assert forb.is_forbidden({"int": 0, "int2": 1}, True)
        assert not forb.is_forbidden({"int": 1, "int2": 1}, True)
        assert not forb.is_forbidden({"int": 1, "int2": 0}, True)

        forb = ForbiddenGreaterThanRelation(hp3, hp4)
        assert forb.is_forbidden({"int": 1, "int2": 0}, True)
        assert not forb.is_forbidden({"int": 1, "int2": 1}, True)
        assert not forb.is_forbidden({"int": 0, "int2": 1}, True)

        forb = ForbiddenGreaterThanRelation(hp4, hp5)
        assert forb.is_forbidden({"int2": 1, "float": 0}, True)
        assert not forb.is_forbidden({"int2": 1, "float": 1}, True)
        assert not forb.is_forbidden({"int2": 0, "float": 1}, True)

        forb = ForbiddenGreaterThanRelation(hp5, hp6)
        self.assertRaises(TypeError, forb.is_forbidden, {"float": 1, "str": "b"}, True)

        forb = ForbiddenGreaterThanRelation(hp5, hp7)
        self.assertRaises(TypeError, forb.is_forbidden, {"float": 1, "str2": "b"}, True)

        forb = ForbiddenGreaterThanRelation(hp6, hp7)
        assert forb.is_forbidden({"str": "b", "str2": "a"}, True)
        assert forb.is_forbidden({"str": "c", "str2": "a"}, True)

        forb1 = ForbiddenEqualsRelation(hp2, hp3)
        forb2 = ForbiddenEqualsRelation(hp2, hp3)
        forb3 = ForbiddenEqualsRelation(hp3, hp4)
        assert forb1 == forb2
        assert forb2 != forb3

        hp1 = OrdinalHyperparameter("water_temperature", ["cold", "luke-warm", "hot", "boiling"])
        hp2 = OrdinalHyperparameter("water_temperature2", ["cold", "luke-warm", "hot", "boiling"])
        forb = ForbiddenGreaterThanRelation(hp1, hp2)
        assert not forb.is_forbidden(
            {"water_temperature": "boiling", "water_temperature2": "cold"},
            True,
        )
        assert forb.is_forbidden({"water_temperature": "hot", "water_temperature2": "cold"}, True)
