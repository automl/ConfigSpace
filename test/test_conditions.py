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

import numpy as np

from ConfigSpace.conditions import (
    AndConjunction,
    EqualsCondition,
    GreaterThanCondition,
    InCondition,
    LessThanCondition,
    NotEqualsCondition,
    OrConjunction,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


class TestConditions(unittest.TestCase):
    # TODO: return only copies of the objects!
    def test_equals_condition(self):
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cond = EqualsCondition(hp2, hp1, 0)
        cond_ = EqualsCondition(hp2, hp1, 0)

        # Test vector value:
        assert cond.vector_value == hp1._inverse_transform(0)
        assert cond.vector_value == cond_.vector_value

        # Test invalid conditions:
        self.assertRaises(TypeError, EqualsCondition, hp2, "parent", 0)
        self.assertRaises(TypeError, EqualsCondition, "child", hp1, 0)
        self.assertRaises(ValueError, EqualsCondition, hp1, hp1, 0)

        assert cond == cond_

        cond_reverse = EqualsCondition(hp1, hp2, 0)
        assert cond != cond_reverse

        assert cond != {}

        assert str(cond) == "child | parent == 0"

    def test_equals_condition_illegal_value(self):
        epsilon = UniformFloatHyperparameter("epsilon", 1e-5, 1e-1, default_value=1e-4, log=True)
        loss = CategoricalHyperparameter(
            "loss",
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            default_value="hinge",
        )
        self.assertRaisesRegex(
            ValueError,
            "Hyperparameter 'epsilon' is "
            "conditional on the illegal value 'huber' of "
            "its parent hyperparameter 'loss'",
            EqualsCondition,
            epsilon,
            loss,
            "huber",
        )

    def test_not_equals_condition(self):
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cond = NotEqualsCondition(hp2, hp1, 0)
        cond_ = NotEqualsCondition(hp2, hp1, 0)
        assert cond == cond_

        # Test vector value:
        assert cond.vector_value == hp1._inverse_transform(0)
        assert cond.vector_value == cond_.vector_value

        cond_reverse = NotEqualsCondition(hp1, hp2, 0)
        assert cond != cond_reverse

        assert cond != {}

        assert str(cond) == "child | parent != 0"

    def test_not_equals_condition_illegal_value(self):
        epsilon = UniformFloatHyperparameter("epsilon", 1e-5, 1e-1, default_value=1e-4, log=True)
        loss = CategoricalHyperparameter(
            "loss",
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            default_value="hinge",
        )
        self.assertRaisesRegex(
            ValueError,
            "Hyperparameter 'epsilon' is "
            "conditional on the illegal value 'huber' of "
            "its parent hyperparameter 'loss'",
            NotEqualsCondition,
            epsilon,
            loss,
            "huber",
        )

    def test_in_condition(self):
        hp1 = CategoricalHyperparameter("parent", list(range(0, 11)))
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cond = InCondition(hp2, hp1, [0, 1, 2, 3, 4, 5])
        cond_ = InCondition(hp2, hp1, [0, 1, 2, 3, 4, 5])
        assert cond == cond_

        # Test vector value:
        assert cond.vector_values == [hp1._inverse_transform(i) for i in [0, 1, 2, 3, 4, 5]]
        assert cond.vector_values == cond_.vector_values

        cond_reverse = InCondition(hp1, hp2, [0, 1, 2, 3, 4, 5])
        assert cond != cond_reverse

        assert cond != {}

        assert str(cond) == "child | parent in {0, 1, 2, 3, 4, 5}"

    def test_greater_and_less_condition(self):
        child = Constant("child", "child")
        hp1 = UniformFloatHyperparameter("float", 0, 5)
        hp2 = UniformIntegerHyperparameter("int", 0, 5)
        hp3 = OrdinalHyperparameter("ord", list(range(6)))

        for hp in [hp1, hp2, hp3]:
            hyperparameter_idx = {child.name: 0, hp.name: 1}

            gt = GreaterThanCondition(child, hp, 1)
            gt.set_vector_idx(hyperparameter_idx)
            assert not gt.evaluate({hp.name: 0})
            assert gt.evaluate({hp.name: 2})
            assert not gt.evaluate({hp.name: None})

            # Evaluate vector
            test_value = hp._inverse_transform(2)
            assert not gt.evaluate_vector(np.array([np.NaN, 0]))
            assert gt.evaluate_vector(np.array([np.NaN, test_value]))
            assert not gt.evaluate_vector(np.array([np.NaN, np.NaN]))

            lt = LessThanCondition(child, hp, 1)
            lt.set_vector_idx(hyperparameter_idx)
            assert lt.evaluate({hp.name: 0})
            assert not lt.evaluate({hp.name: 2})
            assert not lt.evaluate({hp.name: None})

            # Evaluate vector
            test_value = hp._inverse_transform(2)
            assert lt.evaluate_vector(np.array([np.NaN, 0, 0, 0]))
            assert not lt.evaluate_vector(np.array([np.NaN, test_value]))
            assert not lt.evaluate_vector(np.array([np.NaN, np.NaN]))

        hp4 = CategoricalHyperparameter("cat", list(range(6)))
        self.assertRaisesRegex(
            ValueError,
            "Parent hyperparameter in a > or < condition must be a subclass of "
            "NumericalHyperparameter or OrdinalHyperparameter, but is "
            "<cdef class 'ConfigSpace.hyperparameters.CategoricalHyperparameter'>",
            GreaterThanCondition,
            child,
            hp4,
            1,
        )
        self.assertRaisesRegex(
            ValueError,
            "Parent hyperparameter in a > or < condition must be a subclass of "
            "NumericalHyperparameter or OrdinalHyperparameter, but is "
            "<cdef class 'ConfigSpace.hyperparameters.CategoricalHyperparameter'>",
            LessThanCondition,
            child,
            hp4,
            1,
        )

        hp5 = OrdinalHyperparameter("ord", ["cold", "luke warm", "warm", "hot"])

        hyperparameter_idx = {child.name: 0, hp5.name: 1}
        gt = GreaterThanCondition(child, hp5, "warm")
        gt.set_vector_idx(hyperparameter_idx)
        assert gt.evaluate({hp5.name: "hot"})
        assert not gt.evaluate({hp5.name: "cold"})

        assert gt.evaluate_vector(np.array([np.NaN, 3]))
        assert not gt.evaluate_vector(np.array([np.NaN, 0]))

        lt = LessThanCondition(child, hp5, "warm")
        lt.set_vector_idx(hyperparameter_idx)
        assert lt.evaluate({hp5.name: "luke warm"})
        assert not lt.evaluate({hp5.name: "warm"})

        assert lt.evaluate_vector(np.array([np.NaN, 1]))
        assert not lt.evaluate_vector(np.array([np.NaN, 2]))

    def test_in_condition_illegal_value(self):
        epsilon = UniformFloatHyperparameter("epsilon", 1e-5, 1e-1, default_value=1e-4, log=True)
        loss = CategoricalHyperparameter(
            "loss",
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            default_value="hinge",
        )
        self.assertRaisesRegex(
            ValueError,
            "Hyperparameter 'epsilon' is "
            "conditional on the illegal value 'huber' of "
            "its parent hyperparameter 'loss'",
            InCondition,
            epsilon,
            loss,
            ["huber", "log"],
        )

    def test_and_conjunction(self):
        self.assertRaises(TypeError, AndConjunction, "String1", "String2")

        hp1 = CategoricalHyperparameter("input1", [0, 1])
        hp2 = CategoricalHyperparameter("input2", [0, 1])
        hp3 = CategoricalHyperparameter("input3", [0, 1])
        hp4 = Constant("And", "True")
        cond1 = EqualsCondition(hp4, hp1, 1)

        # Only one condition in an AndConjunction!
        self.assertRaises(ValueError, AndConjunction, cond1)

        cond2 = EqualsCondition(hp4, hp2, 1)
        cond3 = EqualsCondition(hp4, hp3, 1)

        andconj1 = AndConjunction(cond1, cond2)
        andconj1_ = AndConjunction(cond1, cond2)
        assert andconj1 == andconj1_

        # Test setting vector idx
        hyperparameter_idx = {hp1.name: 0, hp2.name: 1, hp3.name: 2, hp4.name: 3}
        andconj1.set_vector_idx(hyperparameter_idx)
        assert andconj1.get_parents_vector() == [0, 1]
        assert andconj1.get_children_vector() == [3, 3]

        andconj2 = AndConjunction(cond2, cond3)
        assert andconj1 != andconj2

        andconj3 = AndConjunction(cond1, cond2, cond3)
        assert str(andconj3) == "(And | input1 == 1 && And | input2 == 1 && And | input3 == 1)"

        # Test __eq__
        assert andconj1 != andconj3
        assert andconj1 != "String"

    def test_or_conjunction(self):
        self.assertRaises(TypeError, AndConjunction, "String1", "String2")

        hp1 = CategoricalHyperparameter("input1", [0, 1])
        hp2 = CategoricalHyperparameter("input2", [0, 1])
        hp3 = CategoricalHyperparameter("input3", [0, 1])
        hp4 = Constant("Or", "True")
        cond1 = EqualsCondition(hp4, hp1, 1)

        self.assertRaises(ValueError, OrConjunction, cond1)

        cond2 = EqualsCondition(hp4, hp2, 1)
        cond3 = EqualsCondition(hp4, hp3, 1)

        andconj1 = OrConjunction(cond1, cond2)
        andconj1_ = OrConjunction(cond1, cond2)
        assert andconj1 == andconj1_

        # Test setting vector idx
        hyperparameter_idx = {hp1.name: 0, hp2.name: 1, hp3.name: 2, hp4.name: 3}
        andconj1.set_vector_idx(hyperparameter_idx)
        assert andconj1.get_parents_vector() == [0, 1]
        assert andconj1.get_children_vector() == [3, 3]

        andconj2 = OrConjunction(cond2, cond3)
        assert andconj1 != andconj2

        andconj3 = OrConjunction(cond1, cond2, cond3)
        assert str(andconj3) == "(Or | input1 == 1 || Or | input2 == 1 || Or | input3 == 1)"

    def test_nested_conjunctions(self):
        hp1 = CategoricalHyperparameter("input1", [0, 1])
        hp2 = CategoricalHyperparameter("input2", [0, 1])
        hp3 = CategoricalHyperparameter("input3", [0, 1])
        hp4 = CategoricalHyperparameter("input4", [0, 1])
        hp5 = CategoricalHyperparameter("input5", [0, 1])
        hp6 = Constant("AND", "True")

        cond1 = EqualsCondition(hp6, hp1, 1)
        cond2 = EqualsCondition(hp6, hp2, 1)
        cond3 = EqualsCondition(hp6, hp3, 1)
        cond4 = EqualsCondition(hp6, hp4, 1)
        cond5 = EqualsCondition(hp6, hp5, 1)

        conj1 = AndConjunction(cond1, cond2)
        conj2 = OrConjunction(conj1, cond3)
        conj3 = AndConjunction(conj2, cond4, cond5)

        # TODO: this does not look nice, And should depend on a large
        # conjunction, there should not be many ANDs inside this string!
        assert (
            str(conj3)
            == "(((AND | input1 == 1 && AND | input2 == 1) || AND | input3 == 1) && AND | input4 == 1 && AND | input5 == 1)"
        )

    def test_all_components_have_the_same_child(self):
        hp1 = CategoricalHyperparameter("input1", [0, 1])
        hp2 = CategoricalHyperparameter("input2", [0, 1])
        hp3 = CategoricalHyperparameter("input3", [0, 1])
        hp4 = CategoricalHyperparameter("input4", [0, 1])
        hp5 = CategoricalHyperparameter("input5", [0, 1])
        hp6 = Constant("AND", "True")

        cond1 = EqualsCondition(hp1, hp2, 1)
        cond2 = EqualsCondition(hp1, hp3, 1)
        cond3 = EqualsCondition(hp1, hp4, 1)
        cond4 = EqualsCondition(hp6, hp4, 1)
        cond5 = EqualsCondition(hp6, hp5, 1)

        AndConjunction(cond1, cond2, cond3)
        AndConjunction(cond4, cond5)
        self.assertRaisesRegex(
            ValueError,
            "All Conjunctions and Conditions must have " "the same child.",
            AndConjunction,
            cond1,
            cond4,
        )

    def test_condition_from_cryptominisat(self):
        parent = CategoricalHyperparameter("blkrest", ["0", "1"], default_value="1")
        child = UniformIntegerHyperparameter("blkrestlen", 2000, 10000, log=True)
        condition = EqualsCondition(child, parent, "1")
        assert not condition.evaluate({"blkrest": "0"})
        assert condition.evaluate({"blkrest": "1"})

    def test_get_parents(self):
        # Necessary because we couldn't call cs.get_parents for
        # clasp-sat-params-nat.pcs
        counter = UniformIntegerHyperparameter("bump", 10, 4096, log=True)
        _1_S_countercond = CategoricalHyperparameter("cony", ["yes", "no"])
        _1_0_restarts = CategoricalHyperparameter(
            "restarts",
            ["F", "L", "D", "x", "+", "no"],
            default_value="x",
        )

        condition = EqualsCondition(counter, _1_S_countercond, "yes")
        # All conditions inherit get_parents from abstractcondition
        assert [_1_S_countercond] == condition.get_parents()
        condition2 = InCondition(counter, _1_0_restarts, ["F", "D", "L", "x", "+"])
        # All conjunctions inherit get_parents from abstractconjunction
        conjunction = AndConjunction(condition, condition2)
        assert [_1_S_countercond, _1_0_restarts] == conjunction.get_parents()
