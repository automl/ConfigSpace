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

import itertools

import numpy as np
import pytest

from ConfigSpace import Configuration, ConfigurationSpace
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
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter


# TODO: return only copies of the objects!
def test_equals_condition():
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cond = EqualsCondition(hp2, hp1, 0)
    cond_ = EqualsCondition(hp2, hp1, 0)

    # Test vector value:
    assert cond.vector_value == hp1.to_vector(0)
    assert cond.vector_value == cond_.vector_value

    # Test invalid conditions:
    with pytest.raises(ValueError):
        EqualsCondition(hp1, hp1, 0)

    assert cond == cond_

    cond_reverse = EqualsCondition(hp1, hp2, 0)
    assert cond != cond_reverse

    assert cond != {}

    assert str(cond) == "child | parent == 0"


def test_equals_condition_illegal_value():
    epsilon = UniformFloatHyperparameter(
        "epsilon",
        1e-5,
        1e-1,
        default_value=1e-4,
        log=True,
    )
    loss = CategoricalHyperparameter(
        "loss",
        ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        default_value="hinge",
    )
    with pytest.raises(ValueError):
        EqualsCondition(epsilon, loss, "huber")


def test_not_equals_condition():
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cond = NotEqualsCondition(hp2, hp1, 0)
    cond_ = NotEqualsCondition(hp2, hp1, 0)
    assert cond == cond_

    # Test vector value:
    assert cond.vector_value == hp1.to_vector(0)
    assert cond.vector_value == cond_.vector_value

    cond_reverse = NotEqualsCondition(hp1, hp2, 0)
    assert cond != cond_reverse

    assert cond != {}

    assert str(cond) == "child | parent != 0"


def test_not_equals_condition_illegal_value():
    epsilon = UniformFloatHyperparameter(
        "epsilon",
        1e-5,
        1e-1,
        default_value=1e-4,
        log=True,
    )
    loss = CategoricalHyperparameter(
        "loss",
        ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        default_value="hinge",
    )
    with pytest.raises(ValueError):
        NotEqualsCondition(epsilon, loss, "huber")


def test_in_condition():
    hp1 = CategoricalHyperparameter("parent", list(range(11)))
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cond = InCondition(hp2, hp1, [0, 1, 2, 3, 4, 5])
    cond_ = InCondition(hp2, hp1, [0, 1, 2, 3, 4, 5])
    assert cond == cond_

    # Test vector value:
    assert cond.vector_values == [hp1.to_vector(i) for i in [0, 1, 2, 3, 4, 5]]
    assert cond.vector_values == cond_.vector_values

    cond_reverse = InCondition(hp1, hp2, [0, 1, 2, 3, 4, 5])
    assert cond != cond_reverse

    assert cond != {}

    assert str(cond) == "child | parent in {0, 1, 2, 3, 4, 5}"


def test_greater_and_less_condition():
    child = Constant("child", "child")
    hp1 = UniformFloatHyperparameter("float", 0, 5)
    hp2 = UniformIntegerHyperparameter("int", 0, 5)
    hp3 = OrdinalHyperparameter("ord", list(range(6)))
    hps: list[Hyperparameter] = [hp1, hp2, hp3]

    for hp in hps:
        hyperparameter_idx = {child.name: 0, hp.name: 1}

        gt = GreaterThanCondition(child, hp, 1)
        gt.set_vector_idx(hyperparameter_idx)
        assert not gt.satisfied_by_value({hp.name: 0})
        assert gt.satisfied_by_value({hp.name: 2})
        with pytest.raises((KeyError, TypeError)):
            gt.satisfied_by_value({hp.name: None})

        # Evaluate vector
        test_value = hp.to_vector(2)

        assert not gt.satisfied_by_vector(np.array([np.nan, 0]))
        assert gt.satisfied_by_vector(np.array([np.nan, test_value]))
        assert not gt.satisfied_by_vector(np.array([np.nan, np.nan]))

        lt = LessThanCondition(child, hp, 1)
        lt.set_vector_idx(hyperparameter_idx)
        assert lt.satisfied_by_value({hp.name: 0})
        assert not lt.satisfied_by_value({hp.name: 2})

        with pytest.raises((KeyError, TypeError)):
            lt.satisfied_by_value({hp.name: None})

        # Evaluate vector
        test_value = hp.to_vector(2)
        assert lt.satisfied_by_vector(np.array([np.nan, 0, 0, 0]))
        assert not lt.satisfied_by_vector(np.array([np.nan, test_value]))
        assert not lt.satisfied_by_vector(np.array([np.nan, np.nan]))

    hp4 = CategoricalHyperparameter("cat", list(range(6)))
    with pytest.raises(
        ValueError,
        match=r"The parent hyperparameter must be orderable",
    ):
        GreaterThanCondition(child, hp4, 1)

    with pytest.raises(
        ValueError,
        match=r"The parent hyperparameter must be orderable",
    ):
        LessThanCondition(child, hp4, 1)

    hp5 = OrdinalHyperparameter("ord", ["cold", "luke warm", "warm", "hot"])

    hyperparameter_idx = {child.name: 0, hp5.name: 1}
    gt = GreaterThanCondition(child, hp5, "warm")
    gt.set_vector_idx(hyperparameter_idx)
    assert gt.satisfied_by_value({hp5.name: "hot"})
    assert not gt.satisfied_by_value({hp5.name: "cold"})

    assert gt.satisfied_by_vector(np.array([np.nan, 3]))
    assert not gt.satisfied_by_vector(np.array([np.nan, 0]))

    lt = LessThanCondition(child, hp5, "warm")
    lt.set_vector_idx(hyperparameter_idx)
    assert lt.satisfied_by_value({hp5.name: "luke warm"})
    assert not lt.satisfied_by_value({hp5.name: "warm"})

    assert lt.satisfied_by_vector(np.array([np.nan, 1]))
    assert not lt.satisfied_by_vector(np.array([np.nan, 2]))


def test_in_condition_illegal_value():
    epsilon = UniformFloatHyperparameter(
        "epsilon",
        1e-5,
        1e-1,
        default_value=1e-4,
        log=True,
    )
    loss = CategoricalHyperparameter(
        "loss",
        ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        default_value="hinge",
    )
    with pytest.raises(ValueError):
        InCondition(epsilon, loss, ["huber", "log"])


def test_and_conjunction():
    with pytest.raises(TypeError):
        AndConjunction("String1", "String2")

    hp1 = CategoricalHyperparameter("input1", [0, 1])
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    hp3 = CategoricalHyperparameter("input3", [0, 1])
    hp4 = Constant("And", "True")
    cond1 = EqualsCondition(hp4, hp1, 1)

    # Only one condition in an AndConjunction!
    with pytest.raises(ValueError):
        AndConjunction(cond1)

    cond2 = EqualsCondition(hp4, hp2, 1)
    cond3 = EqualsCondition(hp4, hp3, 1)

    andconj1 = AndConjunction(cond1, cond2)
    andconj1_ = AndConjunction(cond1, cond2)
    assert andconj1 == andconj1_

    # Test setting vector idx
    hyperparameter_idx = {hp1.name: 0, hp2.name: 1, hp3.name: 2, hp4.name: 3}
    andconj1.set_vector_idx(hyperparameter_idx)
    np.testing.assert_equal(andconj1.parent_vector_ids, [0, 1])
    assert andconj1.child_vector_id == 3

    andconj2 = AndConjunction(cond2, cond3)
    assert andconj1 != andconj2

    andconj3 = AndConjunction(cond1, cond2, cond3)
    assert (
        str(andconj3) == "(And | input1 == 1 && And | input2 == 1 && And | input3 == 1)"
    )

    # Test __eq__
    assert andconj1 != andconj3
    assert andconj1 != "String"


def test_or_conjunction():
    with pytest.raises(TypeError):
        OrConjunction("String1", "String2")

    hp1 = CategoricalHyperparameter("input1", [0, 1])
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    hp3 = CategoricalHyperparameter("input3", [0, 1])
    hp4 = Constant("Or", "True")
    cond1 = EqualsCondition(hp4, hp1, 1)

    with pytest.raises(ValueError):
        OrConjunction(cond1)

    cond2 = EqualsCondition(hp4, hp2, 1)
    cond3 = EqualsCondition(hp4, hp3, 1)

    andconj1 = OrConjunction(cond1, cond2)
    andconj1_ = OrConjunction(cond1, cond2)
    assert andconj1 == andconj1_

    # Test setting vector idx
    hyperparameter_idx = {hp1.name: 0, hp2.name: 1, hp3.name: 2, hp4.name: 3}
    andconj1.set_vector_idx(hyperparameter_idx)
    np.testing.assert_equal(andconj1.parent_vector_ids, [0, 1])
    assert andconj1.child_vector_id == 3

    andconj2 = OrConjunction(cond2, cond3)
    assert andconj1 != andconj2

    andconj3 = OrConjunction(cond1, cond2, cond3)
    assert str(andconj3) == "(Or | input1 == 1 || Or | input2 == 1 || Or | input3 == 1)"


def test_nested_conjunctions():
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


def test_all_components_have_the_same_child():
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
    with pytest.raises(ValueError):
        AndConjunction(cond1, cond4)


def test_condition_from_cryptominisat():
    parent = CategoricalHyperparameter("blkrest", ["0", "1"], default_value="1")
    child = UniformIntegerHyperparameter("blkrestlen", 2000, 10000, log=True)
    condition = EqualsCondition(child, parent, "1")
    assert not condition.satisfied_by_value({"blkrest": "0"})
    assert condition.satisfied_by_value({"blkrest": "1"})


def test_get_parents() -> None:
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
    assert _1_S_countercond == condition.parent

    condition2 = InCondition(counter, _1_0_restarts, ["F", "D", "L", "x", "+"])
    conjunction = AndConjunction(condition, condition2)
    assert [_1_S_countercond, _1_0_restarts] == conjunction.parents


def test_active_hyperparameter():
    cs = ConfigurationSpace(
        [
            UniformFloatHyperparameter("age_weight_ratio:log_ratio", -10.0, 3.0, 0.0),
            CategoricalHyperparameter(
                "saturation_algorithm",
                ["discount", "fmb", "inst_gen", "lrs", "otter", "z3"],
                "lrs",
            ),
            CategoricalHyperparameter("inst_gen_with_resolution", ["off", "on"], "off"),
        ],
    )
    cs.add(
        OrConjunction(
            NotEqualsCondition(
                cs["age_weight_ratio:log_ratio"],
                cs["saturation_algorithm"],
                "inst_gen",
            ),
            EqualsCondition(
                cs["age_weight_ratio:log_ratio"],
                cs["inst_gen_with_resolution"],
                "on",
            ),
        ),
    )
    cs.add(
        EqualsCondition(
            cs["inst_gen_with_resolution"],
            cs["saturation_algorithm"],
            "inst_gen",
        ),
    )

    # Check that parameter age_weight_ratio:log_ratio is active according to the default configuration
    # This should be the case, as saturation_algorithm is set to "lrs" (which is NOT "inst_gen") in default.
    default = cs.get_default_configuration()
    cs._check_configuration_rigorous(default)


def test_active_hyperparameter_nested():
    # Based on: https://github.com/automl/ConfigSpace/issues/253
    # Check that a nested condition does not incorrectly deactivate a parameter
    cs = ConfigurationSpace()
    x_top = CategoricalHyperparameter("x_top", [0, 1, 2, 3])

    x_m0 = CategoricalHyperparameter("x_m0", [0, 1])
    x_m1 = CategoricalHyperparameter("x_m1", [0, 1])
    x_m2 = CategoricalHyperparameter("x_m2", [0, 1])

    y = CategoricalHyperparameter("y", [0, 1])
    x_b = CategoricalHyperparameter("x_b", [0, 1])

    cm0 = EqualsCondition(x_m0, x_top, 0)
    cm1 = EqualsCondition(x_m1, x_top, 1)
    cm2 = EqualsCondition(x_m2, x_top, 2)

    cb0 = EqualsCondition(x_b, x_top, 0)
    cb1 = EqualsCondition(x_b, x_m1, 0)
    cb2 = EqualsCondition(x_b, x_m2, 0)

    # The resulting nested condition is:
    # ((x_b | x_top == 0 || x_b | x_m1 == 0 || x_b | x_m2 == 0) && x_b | y == 0
    # Meaning that, for x_b to be active we need:
    # either x_top, x_m1 or x_m2 to be 0
    # AND y to be 0
    #
    cor = OrConjunction(cb0, cb1, cb2)
    cand = AndConjunction(
        cor,
        EqualsCondition(x_b, y, 0),
    )

    cs.add([x_top, x_m0, x_m1, x_b, x_m2, y])
    cs.add([cm0, cm1, cm2])
    cs.add(cand)

    # Create an **illegal** configuration: x_top is equal to three so left side is false eventhough y is equal to 0 (True)
    from ConfigSpace import InactiveHyperparameterSetError

    cfg = {"y": 0, "x_top": 3, "x_b": 0}
    with pytest.raises(InactiveHyperparameterSetError):
        cfg = Configuration(cs, values=cfg)

    # Now left side is true because x_top is equal to 0 but right side is false because y is equal to 1. Now x_m0 is active because x_top is equal to 0.
    cfg = {"y": 1, "x_top": 0, "x_b": 0, "x_m0": 0}
    with pytest.raises(InactiveHyperparameterSetError):
        cfg = Configuration(cs, values=cfg)

    # And now one where x_b is actually active
    cfg = {"y": 0, "x_top": 0, "x_b": 0, "x_m0": 0}
    cfg = Configuration(cs, values=cfg)
    assert cfg.check_valid_configuration() is None

    # Second test
    # 3 categorical params a = (A, B), b = (C, D), c = (E, F)
    # b is active if a == A
    # c is active if b == C (and then of course inactive if b is inactive)
    # The second condition (for activation of c) can be implemented in two ways:
    # 1: Using an EqualsCondition on b == C
    # 2: Using an AndConjuction combining the above with the condition a == A
    cs = ConfigurationSpace(
        name="cs1",
        space={
            "a": CategoricalHyperparameter("a", ["A", "B"]),
            "b": CategoricalHyperparameter("b", ["C", "D"]),
            "c": CategoricalHyperparameter("c", ["E", "F"]),
        },
    )
    cs.add(
        [
            EqualsCondition(cs["b"], cs["a"], "A"),  # b is active if a == A
            EqualsCondition(
                cs["c"],
                cs["b"],
                "C",
            ),  # c is active if b == C (and b is active)
        ],
    )

    # Check that the active hyperparameters are correct
    for x in itertools.product([0, 1], [0, 1], [0, 1]):
        configuration = Configuration(
            cs,
            vector=np.array(x),
            allow_inactive_with_values=True,
        )
        x_active = cs.get_active_hyperparameters(configuration)
        x_active_should_be = (
            {"a"} if x[0] == 1 else ({"a", "b"} if x[1] == 1 else {"a", "b", "c"})
        )
        try:
            assert x_active == x_active_should_be
        except AssertionError:
            print(
                f"{x} ({cs.name}): x_active = {x_active}, whereas it should be {x_active_should_be}",
            )

    # Second way of specifying nested conditions:
    # Child conditions include all ancestors in their condition
    cs = ConfigurationSpace(
        name="cs2",
        space={
            "a": CategoricalHyperparameter("a", ["A", "B"]),
            "b": CategoricalHyperparameter("b", ["C", "D"]),
            "c": CategoricalHyperparameter("c", ["E", "F"]),
        },
    )
    cs.add(
        [
            EqualsCondition(cs["b"], cs["a"], "A"),  # b is active if a == A
            # c is active if b == C (and b is active)
            AndConjunction(
                EqualsCondition(cs["c"], cs["a"], "A"),
                EqualsCondition(cs["c"], cs["b"], "C"),
            ),
        ],
    )

    # Check that the active hyperparameters are correct
    for x in itertools.product([0, 1], [0, 1], [0, 1]):
        x_active = cs.get_active_hyperparameters(
            Configuration(cs, vector=np.array(x), allow_inactive_with_values=True),
        )
        x_active_should_be = (
            {"a"} if x[0] == 1 else ({"a", "b"} if x[1] == 1 else {"a", "b", "c"})
        )
        try:
            assert x_active == x_active_should_be
        except AssertionError:
            print(
                f"{x} ({cs.name}): x_active = {x_active}, whereas it should be {x_active_should_be}",
            )
