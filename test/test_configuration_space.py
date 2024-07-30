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

import json
from collections import OrderedDict
from itertools import product

import numpy as np
import pytest

from ConfigSpace import (
    AndConjunction,
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    InCondition,
    NotEqualsCondition,
    OrConjunction,
    UniformIntegerHyperparameter,
)
from ConfigSpace.api.types.float import Float
from ConfigSpace.api.types.integer import Integer
from ConfigSpace.exceptions import (
    AmbiguousConditionError,
    ChildNotFoundError,
    CyclicDependancyError,
    ForbiddenValueError,
    HyperparameterAlreadyExistsError,
    HyperparameterNotFoundError,
    IllegalValueError,
    InactiveHyperparameterSetError,
    ParentNotFoundError,
)
from ConfigSpace.forbidden import ForbiddenEqualsRelation, ForbiddenLessThanRelation
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
)


def byteify(input):
    return input


def test_add():
    cs = ConfigurationSpace()
    hp = UniformIntegerHyperparameter("name", 0, 10)
    cs.add(hp)


def test_add_non_hyperparameter():
    cs = ConfigurationSpace()
    with pytest.raises(TypeError):
        cs.add(object())  # type: ignore


def test_add_hyperparameters_with_equal_names():
    cs = ConfigurationSpace()
    hp = UniformIntegerHyperparameter("name", 0, 10)
    cs.add(hp)
    with pytest.raises(HyperparameterAlreadyExistsError):
        cs.add(hp)


def test_illegal_default_configuration():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("loss", ["l1", "l2"], default_value="l1")
    hp2 = CategoricalHyperparameter("penalty", ["l1", "l2"], default_value="l1")
    cs.add(hp1)
    cs.add(hp2)
    forb1 = ForbiddenEqualsClause(hp1, "l1")
    forb2 = ForbiddenEqualsClause(hp2, "l1")
    forb3 = ForbiddenAndConjunction(forb1, forb2)

    with pytest.raises(ForbiddenValueError):
        cs.add(forb3)


def test_meta_data_stored():
    meta_data = {
        "additional": "meta-data",
        "useful": "for integrations",
        "input_id": 42,
    }
    cs = ConfigurationSpace(meta=dict(meta_data))
    assert cs.meta == meta_data


def test_add_non_condition():
    cs = ConfigurationSpace()
    with pytest.raises(TypeError):
        cs.add(object())  # type: ignore


def test_hyperparameters_with_valid_condition():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add(hp2)
    cond = EqualsCondition(hp2, hp1, 0)
    cs.add(cond)

    assert len(cs) == 2


def test_condition_without_added_hyperparameters():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cond = EqualsCondition(hp2, hp1, 0)

    with pytest.raises(ChildNotFoundError):
        cs.add(cond)

    cs.add(hp1)

    with pytest.raises(ChildNotFoundError):
        cs.add(cond)

    # Test also the parent hyperparameter
    cs2 = ConfigurationSpace()
    cs2.add(hp2)

    with pytest.raises(ParentNotFoundError):
        cs2.add(cond)


def test_condition_with_cycles():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add(cond1)
    cond2 = EqualsCondition(hp1, hp2, 0)

    with pytest.raises(CyclicDependancyError):
        cs.add(cond2)


def test_add_conjunction():
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    hp3 = CategoricalHyperparameter("input3", [0, 1])
    hp4 = Constant("And", "True")

    cond1 = EqualsCondition(hp4, hp1, 1)
    cond2 = EqualsCondition(hp4, hp2, 1)
    cond3 = EqualsCondition(hp4, hp3, 1)

    andconj1 = AndConjunction(cond1, cond2, cond3)

    cs = ConfigurationSpace()
    cs.add(hp1)
    cs.add(hp2)
    cs.add(hp3)
    cs.add(hp4)

    cs.add(andconj1)
    assert hp4 not in cs.unconditional_hyperparameters


def test_add_second_condition_wo_conjunction():
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    hp3 = Constant("And", "True")

    cond1 = EqualsCondition(hp3, hp1, 1)
    cond2 = EqualsCondition(hp3, hp2, 1)

    cs = ConfigurationSpace()
    cs.add([hp1, hp2, hp3])
    cs.add(cond1)

    with pytest.raises(AmbiguousConditionError):
        cs.add(cond2)


def test_add_forbidden_clause():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    cs.add(hp1)
    forb = ForbiddenEqualsClause(hp1, 1)
    # TODO add checking whether a forbidden clause makes sense at all
    cs.add(forb)
    # TODO add something to properly retrieve the forbidden clauses
    assert (
        str(cs)
        == "Configuration space object:\n  Hyperparameters:\n    input1, Type: Categorical, Choices: {0, 1}, Default: 0\n  Forbidden Clauses:\n    Forbidden: input1 == 1\n"
    )


def test_add_forbidden_relation():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    hp2 = CategoricalHyperparameter("input2", [1, 0])
    cs.add([hp1, hp2])
    forb = ForbiddenEqualsRelation(hp1, hp2)
    # TODO add checking whether a forbidden clause makes sense at all
    cs.add(forb)
    # TODO add something to properly retrieve the forbidden clauses
    assert (
        str(cs)
        == "Configuration space object:\n  Hyperparameters:\n    input1, Type: Categorical, Choices: {0, 1}, Default: 0\n    input2, Type: Categorical, Choices: {1, 0}, Default: 1\n  Forbidden Clauses:\n    Forbidden: input1 == input2\n"
    )


def test_add_forbidden_relation_categorical():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("input1", ["a", "b"], default_value="b")
    hp2 = CategoricalHyperparameter("input2", ["b", "c"], default_value="b")
    cs.add([hp1, hp2])
    forb = ForbiddenEqualsRelation(hp1, hp2)
    with pytest.raises(ForbiddenValueError):
        cs.add(forb)


def test_add_forbidden_illegal():
    cs = ConfigurationSpace()
    hp = CategoricalHyperparameter("input1", [0, 1])
    forb = ForbiddenEqualsClause(hp, 1)

    with pytest.raises(HyperparameterNotFoundError):
        cs.add(forb)

    forb2 = ForbiddenEqualsClause(hp, 0)

    with pytest.raises(HyperparameterNotFoundError):
        cs.add([forb, forb2])


def test_add_configuration_space():
    cs = ConfigurationSpace({"input1": [0, 1], "child": (0, 10)})
    cs.add(ForbiddenEqualsClause(cs["input1"], 1))
    cs.add(EqualsCondition(cs["child"], cs["input1"], 0))

    cs2 = ConfigurationSpace()
    cs2.add_configuration_space("prefix", cs, delimiter="__")
    assert (
        str(cs2)
        == "Configuration space object:\n  Hyperparameters:\n    prefix__child, Type: UniformInteger, Range: [0, 10], Default: 5\n    prefix__input1, Type: Categorical, Choices: {0, 1}, Default: 0\n  Conditions:\n    prefix__child | prefix__input1 == 0\n  Forbidden Clauses:\n    Forbidden: prefix__input1 == 1\n"
    )


def test_add_configuration_space_conjunctions():
    cs1 = ConfigurationSpace(
        {
            "input1": [0, 1],
            "input2": [0, 1],
            "child1": (0, 10),
            "child2": (0, 10),
        },
    )
    cs1.add(
        EqualsCondition(cs1["input2"], cs1["child1"], 0),
        AndConjunction(
            EqualsCondition(cs1["input1"], cs1["child1"], 5),
            EqualsCondition(cs1["input1"], cs1["child2"], 1),
        ),
    )

    cs2 = ConfigurationSpace()
    cs2.add_configuration_space(prefix="test", configuration_space=cs1)

    assert str(cs2).count("test:") == 10
    # Check that they're equal except for the "test:" prefix
    assert str(cs1) == str(cs2).replace("test:", "")


def test_add_conditions():
    cs1 = ConfigurationSpace(
        {"input1": [0, 1], "input2": [0, 1], "child1": (0, 10), "child2": (0, 10)},
    )
    cs2 = ConfigurationSpace(cs1)

    cond1 = EqualsCondition(cs1["input2"], cs1["child1"], 0)
    cond2 = EqualsCondition(cs1["input1"], cs1["child1"], 5)
    cond3 = EqualsCondition(cs1["input1"], cs1["child2"], 1)
    andCond = AndConjunction(cond2, cond3)

    cs1.add([cond1, andCond])
    cs2.add(cond1)
    cs2.add(andCond)

    assert str(cs1) == str(cs2)


def test_get_hyperparamforbidden_clauseseters():
    cs = ConfigurationSpace()
    assert len(cs) == 0
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add(hp1)
    assert [hp1] == list(cs.values())
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add(hp2)
    cond1 = EqualsCondition(hp2, hp1, 1)
    cs.add(cond1)
    assert [hp1, hp2] == list(cs.values())
    # TODO: I need more tests for the topological sort!
    assert [hp1, hp2] == list(cs.values())


def test_get_hyperparameters_topological_sort_simple():
    for _ in range(10):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs.add(hp1)
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cs.add(hp2)
        cond1 = EqualsCondition(hp2, hp1, 0)
        cs.add(cond1)
        # This automatically checks the configuration!
        Configuration(cs, {"parent": 0, "child": 5})


def test_get_hyperparameters_topological_sort():
    # and now for something more complicated
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    hp3 = CategoricalHyperparameter("input3", [0, 1])
    hp4 = CategoricalHyperparameter("input4", [0, 1])
    hp5 = CategoricalHyperparameter("input5", [0, 1])
    hp6 = Constant("AND", "True")
    # More top-level hyperparameters
    hp7 = CategoricalHyperparameter("input7", [0, 1])
    # Somewhat shuffled
    hyperparameters = [hp1, hp2, hp3, hp4, hp5, hp6, hp7]

    for hp in hyperparameters:
        cs.add(hp)

    cond1 = EqualsCondition(hp6, hp1, 1)
    cond2 = NotEqualsCondition(hp6, hp2, 1)
    cond3 = InCondition(hp6, hp3, [1])
    cond4 = EqualsCondition(hp5, hp3, 1)
    cond5 = EqualsCondition(hp4, hp5, 1)
    cond6 = EqualsCondition(hp6, hp4, 1)
    cond7 = EqualsCondition(hp6, hp5, 1)

    conj1 = AndConjunction(cond1, cond2)
    conj2 = OrConjunction(conj1, cond3)
    conj3 = AndConjunction(conj2, cond6, cond7)

    cs.add(cond4)
    hps = list(cs.values())
    # AND is moved to the front because of alphabetical sorting
    for hp, idx in zip(hyperparameters, [1, 2, 3, 4, 6, 0, 5]):
        assert hps.index(hp) == idx
        assert cs.index_of[hp.name] == idx
        assert cs.at[idx] == hp.name

    cs.add(cond5)
    hps = list(cs.values())
    for hp, idx in zip(hyperparameters, [1, 2, 3, 6, 5, 0, 4]):
        assert hps.index(hp) == idx
        assert cs.index_of[hp.name] == idx
        assert cs.at[idx] == hp.name

    cs.add(conj3)
    hps = list(cs.values())
    for hp, idx in zip(hyperparameters, [0, 1, 2, 5, 4, 6, 3]):
        assert hps.index(hp) == idx
        assert cs.index_of[hp.name] == idx
        assert cs.at[idx] == hp.name


def test_get_hyperparameter():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add(hp2)

    retval = cs["parent"]
    assert hp1 == retval
    retval = cs["child"]
    assert hp2 == retval

    with pytest.raises(KeyError):
        cs["grandfather"]


def test_get_conditions():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add(hp2)
    assert [] == cs.conditions
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add(cond1)
    assert [cond1] == cs.conditions


def test_get_parent_and_chil_conditions_of():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add(cond1)

    assert [cond1] == cs.parent_conditions_of[hp2.name]
    assert [cond1] == cs.parent_conditions_of[hp2.name]
    assert [cond1] == cs.child_conditions_of[hp1.name]
    assert [cond1] == cs.child_conditions_of[hp1.name]

    with pytest.raises(KeyError):
        cs.parent_conditions_of["Foo"]

    with pytest.raises(KeyError):
        cs.child_conditions_of["Foo"]


def test_get_parent_and_children_of():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add(cond1)

    assert [hp1] == cs.parents_of[hp2.name]
    assert [hp1] == cs.parents_of[hp2.name]
    assert [hp2] == cs.children_of[hp1.name]
    assert [hp2] == cs.children_of[hp1.name]

    with pytest.raises(KeyError):
        cs.parents_of["Foo"]

    with pytest.raises(KeyError):
        cs.children_of["Foo"]


def test_check_configuration():
    # TODO this is only a smoke test
    # TODO actually, this rather tests the evaluate methods in the
    # conditions module!
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add(cond1)
    # This automatically checks the configuration!
    Configuration(cs, {"parent": 0, "child": 5})

    # and now for something more complicated
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    cs.add(hp1)
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    cs.add(hp2)
    hp3 = CategoricalHyperparameter("input3", [0, 1])
    cs.add(hp3)
    hp4 = CategoricalHyperparameter("input4", [0, 1])
    cs.add(hp4)
    hp5 = CategoricalHyperparameter("input5", [0, 1])
    cs.add(hp5)
    hp6 = Constant("AND", "True")
    cs.add(hp6)

    cond1 = EqualsCondition(hp6, hp1, 1)
    cond2 = NotEqualsCondition(hp6, hp2, 1)
    cond3 = InCondition(hp6, hp3, [1])
    cond4 = EqualsCondition(hp6, hp4, 1)
    cond5 = EqualsCondition(hp6, hp5, 1)

    conj1 = AndConjunction(cond1, cond2)
    conj2 = OrConjunction(conj1, cond3)
    conj3 = AndConjunction(conj2, cond4, cond5)
    cs.add(conj3)

    expected_outcomes = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]

    for idx, values in enumerate(product([0, 1], repeat=5)):
        # The hyperparameters aren't sorted, but the test assumes them to
        #  be sorted.
        hyperparameters = sorted(cs.values(), key=lambda t: t.name)
        instantiations = {
            hyperparameters[jdx + 1].name: values[jdx] for jdx in range(len(values))
        }

        evaluation = conj3.satisfied_by_value(instantiations)
        assert expected_outcomes[idx] == evaluation

        if not evaluation:
            with pytest.raises(InactiveHyperparameterSetError):
                Configuration(
                    cs,
                    values={
                        "input1": values[0],
                        "input2": values[1],
                        "input3": values[2],
                        "input4": values[3],
                        "input5": values[4],
                        "AND": "True",
                    },
                )
        else:
            Configuration(
                cs,
                values={
                    "input1": values[0],
                    "input2": values[1],
                    "input3": values[2],
                    "input4": values[3],
                    "input5": values[4],
                    "AND": "True",
                },
            )


def test_check_configuration2():
    # Test that hyperparameters which are not active must not be set and
    # that evaluating forbidden clauses does not choke on missing
    # hyperparameters
    cs = ConfigurationSpace()
    classifier = CategoricalHyperparameter(
        "classifier",
        ["k_nearest_neighbors", "extra_trees"],
    )
    metric = CategoricalHyperparameter("metric", ["minkowski", "other"])
    p = CategoricalHyperparameter("k_nearest_neighbors:p", [1, 2])
    metric_depends_on_classifier = EqualsCondition(
        metric,
        classifier,
        "k_nearest_neighbors",
    )
    p_depends_on_metric = EqualsCondition(p, metric, "minkowski")
    cs.add(metric)
    cs.add(p)
    cs.add(classifier)
    cs.add(metric_depends_on_classifier)
    cs.add(p_depends_on_metric)

    forbidden = ForbiddenEqualsClause(metric, "other")
    cs.add(forbidden)

    configuration = Configuration(cs, {"classifier": "extra_trees"})

    # check backward compatibility with checking configurations instead of vectors
    configuration.check_valid_configuration()


def test_check_forbidden_with_sampled_vector_configuration():
    cs = ConfigurationSpace()
    metric = CategoricalHyperparameter("metric", ["minkowski", "other"])
    cs.add(metric)

    forbidden = ForbiddenEqualsClause(metric, "other")
    cs.add(forbidden)
    configuration = Configuration(cs, vector=np.ones(1, dtype=float))

    with pytest.raises(ValueError, match="violates forbidden clause"):
        cs._check_forbidden(configuration.get_array())


def test_eq():
    # Compare empty configuration spaces
    cs1 = ConfigurationSpace()
    cs2 = ConfigurationSpace()
    assert cs1 == cs2

    # Compare to something which isn't a configuration space
    assert cs1 != "ConfigurationSpace"

    # Compare to equal configuration spaces
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    hp3 = UniformIntegerHyperparameter("friend", 0, 5)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs1.add(hp1)
    cs1.add(hp2)
    cs1.add(cond1)
    cs2.add(hp1)
    cs2.add(hp2)
    cs2.add(cond1)
    assert cs1 == cs2
    cs1.add(hp3)
    assert cs1 != cs2


def test_neq():
    cs1 = ConfigurationSpace()
    assert cs1 != "ConfigurationSpace"


def test_repr():
    cs1 = ConfigurationSpace()
    retval = cs1.__str__()
    assert retval == "Configuration space object:\n  Hyperparameters:\n"

    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs1.add(hp1)
    retval = cs1.__str__()
    assert f"Configuration space object:\n  Hyperparameters:\n    {hp1!s}\n" == retval

    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs1.add(hp2)
    cs1.add(cond1)
    retval = cs1.__str__()
    assert (
        f"Configuration space object:\n  Hyperparameters:\n    {hp2}\n    {hp1}\n  Conditions:\n    {cond1}\n"
        == retval
    )


def test_sample_configuration():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add(cond1)
    # This automatically checks the configuration!
    Configuration(cs, {"parent": 0, "child": 5})

    # and now for something more complicated
    cs = ConfigurationSpace(seed=1)
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    cs.add(hp1)
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    cs.add(hp2)
    hp3 = CategoricalHyperparameter("input3", [0, 1])
    cs.add(hp3)
    hp4 = CategoricalHyperparameter("input4", [0, 1])
    cs.add(hp4)
    hp5 = CategoricalHyperparameter("input5", [0, 1])
    cs.add(hp5)
    hp6 = Constant("AND", "True")
    cs.add(hp6)

    cond1 = EqualsCondition(hp6, hp1, 1)
    cond2 = NotEqualsCondition(hp6, hp2, 1)
    cond3 = InCondition(hp6, hp3, [1])
    cond4 = EqualsCondition(hp5, hp3, 1)
    cond5 = EqualsCondition(hp4, hp5, 1)
    cond6 = EqualsCondition(hp6, hp4, 1)
    cond7 = EqualsCondition(hp6, hp5, 1)

    conj1 = AndConjunction(cond1, cond2)
    conj2 = OrConjunction(conj1, cond3)
    conj3 = AndConjunction(conj2, cond6, cond7)
    cs.add(cond4)
    cs.add(cond5)
    cs.add(conj3)

    samples: list[list[Configuration]] = []
    for i in range(5):
        cs.seed(1)
        samples.append([])
        for _ in range(100):
            sample = cs.sample_configuration()
            samples[-1].append(sample)

        if i > 0:
            for j in range(100):
                assert samples[-1][j] == samples[-2][j]


def test_sample_configuration_with_or_conjunction():
    cs = ConfigurationSpace(seed=1)

    hp5 = CategoricalHyperparameter("hp5", ["0", "1", "2"])
    hp7 = CategoricalHyperparameter("hp7", ["0", "1", "2"])
    hp8 = CategoricalHyperparameter("hp8", ["0", "1", "2"])
    cond1 = InCondition(hp5, hp8, ["0"])
    cond2 = OrConjunction(
        InCondition(hp7, hp8, ["1"]),
        InCondition(hp7, hp5, ["1"]),
    )
    cs.add(hp5, hp7, hp8, cond1, cond2)

    # == print(cs.at)
    # hp8, hp5, hp7
    # hp8 (first index) always active
    # hp5 (second index) only active if hp8 (second index) is 0
    # hp7 (last index) only active if either hp8 (first) or hp5 (second) is 1

    configs = cs.sample_configuration(6)
    sampled_arrays = np.asarray([config.get_array() for config in configs])
    expected_arrays = np.asarray(
        [
            [1, np.nan, 0],
            [0, 0, np.nan],
            [0, 1, 1],
            [1, np.nan, 2],
            [1, np.nan, 0],
            [2, np.nan, np.nan],
        ],
    )

    np.testing.assert_equal(sampled_arrays, expected_arrays)


def test_sample_wrong_argument():
    cs = ConfigurationSpace()
    with pytest.raises(TypeError):
        cs.sample_configuration(1.2)  # type: ignore


def test_sample_no_configuration():
    cs = ConfigurationSpace()
    rval = cs.sample_configuration(size=0)
    assert len(rval) == 0


def test_subspace_switches():
    # create a switch to select one of two algorithms
    algo_switch = CategoricalHyperparameter(
        name="switch",
        choices=["algo1", "algo2"],
        weights=[0.25, 0.75],
        default_value="algo1",
    )

    # create sub-configuration space for algorithm 1
    algo1_cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter(
        name="algo1_param1",
        choices=["A", "B"],
        weights=[0.3, 0.7],
        default_value="B",
    )
    algo1_cs.add(hp1)

    # create sub-configuration space for algorithm 2
    algo2_cs = ConfigurationSpace()
    hp2 = CategoricalHyperparameter(
        name="algo2_param1",
        choices=["X", "Y"],
        default_value="Y",
    )
    algo2_cs.add(hp2)

    # create a configuration space and populate it with both the switch
    # and the two sub-configuration spaces
    cs = ConfigurationSpace()
    cs.add(algo_switch)
    cs.add_configuration_space(
        prefix="algo1_subspace",
        configuration_space=algo1_cs,
        parent_hyperparameter={"parent": algo_switch, "value": "algo1"},
    )
    cs.add_configuration_space(
        prefix="algo2_subspace",
        configuration_space=algo2_cs,
        parent_hyperparameter={"parent": algo_switch, "value": "algo2"},
    )

    # check choices in the final configuration space
    assert cs["switch"].choices == ("algo1", "algo2")  # type: ignore
    assert cs["algo1_subspace:algo1_param1"].choices == ("A", "B")  # type: ignore
    assert cs["algo2_subspace:algo2_param1"].choices == ("X", "Y")  # type: ignore

    # check probabilities in the final configuration space
    np.testing.assert_equal(cs["switch"].probabilities, (0.25, 0.75))  # type: ignore
    np.testing.assert_equal(cs["algo1_subspace:algo1_param1"].probabilities, (0.3, 0.7))  # type: ignore
    np.testing.assert_equal(cs["algo2_subspace:algo2_param1"].probabilities, (0.5, 0.5))  # type: ignore

    # check default values in the final configuration space
    assert cs["switch"].default_value == "algo1"
    assert cs["algo1_subspace:algo1_param1"].default_value == "B"
    assert cs["algo2_subspace:algo2_param1"].default_value == "Y"


def test_configuration_space_acts_as_mapping():
    """Test that ConfigurationSpace can act as a mapping with iteration,
    indexing and items, values, keys.
    """
    cs = ConfigurationSpace()
    names = [f"name{i}" for i in range(5)]
    hyperparameters = [UniformIntegerHyperparameter(name, 0, 10) for name in names]
    cs.add(hyperparameters)

    # Test indexing
    assert cs["name3"] == hyperparameters[3]

    # Test dict methods
    assert list(cs.keys()) == names
    assert list(cs.values()) == hyperparameters
    assert list(cs.items()) == list(zip(names, hyperparameters))
    assert len(cs) == 5

    # Test __iter__
    assert list(iter(cs)) == names

    # Test unpacking
    d = {**cs}
    assert list(d.keys()) == names
    assert list(d.values()) == hyperparameters
    assert list(d.items()) == list(zip(names, hyperparameters))
    assert len(d) == 5


def test_remove_hyperparameter_priors():
    cs = ConfigurationSpace()
    integer = UniformIntegerHyperparameter("integer", 1, 5, log=True)
    cat = CategoricalHyperparameter("cat", [0, 1, 2], weights=[1, 2, 3])
    beta = BetaFloatHyperparameter("beta", alpha=8, beta=2, lower=-1, upper=11)
    norm = NormalIntegerHyperparameter("norm", mu=5, sigma=4, lower=1, upper=15)
    cs.add([integer, cat, beta, norm])

    cat_default = cat.default_value
    norm_default = norm.default_value
    beta_default = beta.default_value

    # add some conditions, to test that remove_parameter_priors keeps the forbiddensdef test_remove_hyp
    cond_2 = OrConjunction(EqualsCondition(beta, cat, 0), EqualsCondition(beta, cat, 1))
    cond_3 = OrConjunction(
        EqualsCondition(norm, cat, 2),
        EqualsCondition(norm, integer, 1),
        EqualsCondition(norm, integer, 3),
        EqualsCondition(norm, integer, 5),
    )
    cs.add([cond_2, cond_3])

    # add some forbidden clauses too, to test that remove_parameter_priors keeps the forbiddens
    forbidden_clause_a = ForbiddenEqualsClause(cat, 0)
    forbidden_clause_c = ForbiddenEqualsClause(integer, 3)
    forbidden_clause_d = ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_c)
    cs.add([forbidden_clause_c, forbidden_clause_d])

    uniform_cs = cs.remove_hyperparameter_priors()

    expected_cs = ConfigurationSpace()
    unif_integer = UniformIntegerHyperparameter("integer", 1, 5, log=True)
    unif_cat = CategoricalHyperparameter("cat", [0, 1, 2], default_value=cat_default)

    unif_beta = UniformFloatHyperparameter(
        "beta",
        lower=-1,
        upper=11,
        default_value=beta_default,
    )
    unif_norm = UniformIntegerHyperparameter(
        "norm",
        lower=1,
        upper=15,
        default_value=norm_default,
    )
    expected_cs.add([unif_integer, unif_cat, unif_beta, unif_norm])

    # add some conditions, to test that remove_parameter_priors keeps the forbiddens
    cond_2 = OrConjunction(
        EqualsCondition(unif_beta, unif_cat, 0),
        EqualsCondition(unif_beta, unif_cat, 1),
    )
    cond_3 = OrConjunction(
        EqualsCondition(unif_norm, unif_cat, 2),
        EqualsCondition(unif_norm, unif_integer, 1),
        EqualsCondition(unif_norm, unif_integer, 3),
        EqualsCondition(unif_norm, unif_integer, 5),
    )
    expected_cs.add([cond_2, cond_3])

    # add some forbidden clauses too, to test that remove_parameter_priors keeps the forbiddens
    forbidden_clause_a = ForbiddenEqualsClause(unif_cat, 0)
    forbidden_clause_c = ForbiddenEqualsClause(unif_integer, 3)
    forbidden_clause_d = ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_c)
    expected_cs.add([forbidden_clause_c, forbidden_clause_d])

    assert uniform_cs == expected_cs


def test_substitute_hyperparameters_in_conditions():
    cs1 = ConfigurationSpace()
    orig_hp1 = CategoricalHyperparameter("input1", [0, 1])
    orig_hp2 = CategoricalHyperparameter("input2", [0, 1])
    orig_hp3 = UniformIntegerHyperparameter("child1", 0, 10)
    orig_hp4 = UniformIntegerHyperparameter("child2", 0, 10)
    cs1.add([orig_hp1, orig_hp2, orig_hp3, orig_hp4])
    cond1 = EqualsCondition(orig_hp2, orig_hp3, 0)
    cond2 = EqualsCondition(orig_hp1, orig_hp3, 5)
    cond3 = EqualsCondition(orig_hp1, orig_hp4, 1)
    andCond = AndConjunction(cond2, cond3)
    cs1.add([cond1, andCond])

    cs2 = ConfigurationSpace()
    sub_hp1 = CategoricalHyperparameter("input1", [0, 1, 2])
    sub_hp2 = CategoricalHyperparameter("input2", [0, 1, 3])
    sub_hp3 = NormalIntegerHyperparameter("child1", lower=0, upper=10, mu=5, sigma=2)
    sub_hp4 = BetaIntegerHyperparameter("child2", lower=0, upper=10, alpha=3, beta=5)
    cs2.add([sub_hp1, sub_hp2, sub_hp3, sub_hp4])
    new_conditions = cs1.substitute_hyperparameters_in_conditions(
        cs1.conditions,
        cs2,
    )

    test_cond1 = EqualsCondition(sub_hp2, sub_hp3, 0)
    test_cond2 = EqualsCondition(sub_hp1, sub_hp3, 5)
    test_cond3 = EqualsCondition(sub_hp1, sub_hp4, 1)
    test_andCond = AndConjunction(test_cond2, test_cond3)
    cs2.add([test_cond1, test_andCond])
    test_conditions = cs2.conditions

    assert new_conditions[0] == test_conditions[0]
    assert new_conditions[1] == test_conditions[1]


def test_substitute_hyperparameters_in_inconditions():
    cs1 = ConfigurationSpace()
    a = UniformIntegerHyperparameter("a", lower=0, upper=10)
    b = UniformFloatHyperparameter("b", lower=1.0, upper=8.0, log=False)
    cs1.add([a, b])

    cond = InCondition(b, a, [1, 2, 3, 4])
    cs1.add([cond])

    cs2 = ConfigurationSpace()
    sub_a = UniformIntegerHyperparameter("a", lower=0, upper=10)
    sub_b = UniformFloatHyperparameter("b", lower=1.0, upper=8.0, log=False)
    cs2.add([sub_a, sub_b])
    new_conditions = cs1.substitute_hyperparameters_in_conditions(
        cs1.conditions,
        cs2,
    )

    test_cond = InCondition(b, a, [1, 2, 3, 4])
    cs2.add([test_cond])
    test_conditions = cs2.conditions

    new_c = new_conditions[0]
    test_c = test_conditions[0]

    assert isinstance(new_c, InCondition)
    assert isinstance(test_c, InCondition)
    assert new_c == test_c
    assert new_c is not test_c

    assert new_c.parent == test_c.parent
    assert new_c.parent is not test_c.parent

    assert new_c.child == test_c.child
    assert new_c.child is not test_c.child


def test_substitute_hyperparameters_in_forbiddens():
    cs1 = ConfigurationSpace()
    orig_hp1 = CategoricalHyperparameter("input1", [0, 1])
    orig_hp2 = CategoricalHyperparameter("input2", [0, 1])
    orig_hp3 = UniformIntegerHyperparameter("input3", 0, 10)
    orig_hp4 = UniformIntegerHyperparameter("input4", 0, 10)
    cs1.add([orig_hp1, orig_hp2, orig_hp3, orig_hp4])
    forb_1 = ForbiddenEqualsClause(orig_hp1, 0)
    forb_2 = ForbiddenEqualsClause(orig_hp2, 1)
    forb_3 = ForbiddenEqualsClause(orig_hp3, 10)
    forb_4 = ForbiddenAndConjunction(forb_1, forb_2)
    forb_5 = ForbiddenLessThanRelation(orig_hp1, orig_hp2)
    cs1.add([forb_3, forb_4, forb_5])

    cs2 = ConfigurationSpace()
    sub_hp1 = CategoricalHyperparameter("input1", [0, 1, 2])
    sub_hp2 = CategoricalHyperparameter("input2", [0, 1, 3])
    sub_hp3 = NormalIntegerHyperparameter("input3", lower=0, upper=10, mu=5, sigma=2)
    sub_hp4 = BetaIntegerHyperparameter("input4", lower=0, upper=10, alpha=3, beta=5)
    cs2.add([sub_hp1, sub_hp2, sub_hp3, sub_hp4])

    new_forbiddens = cs1.substitute_hyperparameters_in_forbiddens(
        cs1.forbidden_clauses,
        cs2,
    )

    test_forb_1 = ForbiddenEqualsClause(sub_hp1, 0)
    test_forb_2 = ForbiddenEqualsClause(sub_hp2, 1)
    test_forb_3 = ForbiddenEqualsClause(sub_hp3, 10)
    test_forb_4 = ForbiddenAndConjunction(test_forb_1, test_forb_2)
    test_forb_5 = ForbiddenLessThanRelation(sub_hp1, sub_hp2)
    cs2.add([test_forb_3, test_forb_4, test_forb_5])
    test_forbiddens = cs2.forbidden_clauses

    assert new_forbiddens[2] == test_forbiddens[2]
    assert new_forbiddens[1] == test_forbiddens[1]
    assert new_forbiddens[0] == test_forbiddens[0]


def test_estimate_size():
    cs = ConfigurationSpace()
    assert cs.estimate_size() == 0
    cs.add(Constant("constant", 0))
    assert cs.estimate_size() == 1
    cs.add(UniformIntegerHyperparameter("integer", 0, 5))
    assert cs.estimate_size() == 6
    cs.add(CategoricalHyperparameter("cat", [0, 1, 2]))
    assert cs.estimate_size() == 18
    cs.add(UniformFloatHyperparameter("float", 0, 1))
    assert np.isinf(cs.estimate_size())


@pytest.fixture
def simple_cs():
    return ConfigurationSpace({"parent": [0, 1], "child": (0, 10), "friend": (0, 5)})


def test_wrong_init(simple_cs: ConfigurationSpace):
    with pytest.raises(ValueError):
        Configuration(simple_cs)

    with pytest.raises(ValueError):
        Configuration(simple_cs, values={}, vector=np.zeros((3,)))


def test_init_with_values(simple_cs: ConfigurationSpace):
    c1 = Configuration(simple_cs, values={"parent": 1, "child": 2, "friend": 3})
    for i in range(5 + 1):
        Configuration(simple_cs, values={"parent": 1, "child": 2, "friend": i})
    # Pay attention that the vector does not necessarily has an intuitive
    #  sorting!
    vector_values = {
        "parent": 1,
        "child": 0.22727223140405708,
        "friend": 0.583333611112037,
    }
    vector = np.zeros(3)
    for name in simple_cs.index_of:
        vector[simple_cs.index_of[name]] = vector_values[name]
    c2 = Configuration(simple_cs, vector=vector)
    # This tests
    # a) that the vector representation of both are the same
    # b) that the dictionary representation of both are the same
    assert c1 == c2


def test_uniformfloat_transform():
    """This checks whether a value sampled through the configuration
    space (it does not happend when the variable is sampled alone) stays
    equal when it is serialized via JSON and the deserialized again.
    """
    cs = ConfigurationSpace()
    cs.add(
        UniformFloatHyperparameter("a", -5, 10),
        NormalFloatHyperparameter("b", 1, 2, log=True, lower=0.1, upper=5),
    )
    a = cs["a"]
    b = cs["b"]
    for _i in range(100):
        config = cs.sample_configuration()
        value = OrderedDict(sorted(config.items()))
        string = json.dumps(value)
        saved_value = json.loads(string)
        saved_value = OrderedDict(sorted(byteify(saved_value).items()))
        assert repr(value) == repr(saved_value)

    # Next, test whether the truncation also works when initializing the
    # Configuration with a dictionary
    for _i in range(100):
        rs = np.random.RandomState(1)
        value_a = a.sample_value(seed=rs)
        value_b = b.sample_value(seed=rs)
        values_dict = {"a": value_a, "b": value_b}
        config = Configuration(cs, values=values_dict)
        values = dict(config)
        string = json.dumps(values)
        saved_value = json.loads(string)
        saved_value = byteify(saved_value)
        assert values == saved_value


def test_setitem():
    """Checks overriding a sampled configuration."""
    pcs = ConfigurationSpace()
    pcs.add(UniformIntegerHyperparameter("x0", 1, 5, default_value=1))
    pcs.add(
        CategoricalHyperparameter("x1", ["ab", "bc", "cd", "de"], default_value="ab"),
    )

    # Condition
    pcs.add(CategoricalHyperparameter("x2", [1, 2]))
    pcs.add(EqualsCondition(pcs["x2"], pcs["x1"], "ab"))

    # Forbidden
    pcs.add(CategoricalHyperparameter("x3", [1, 2]))
    pcs.add(ForbiddenEqualsClause(pcs["x3"], 2))

    conf = pcs.get_default_configuration()

    # failed because it's a invalid configuration
    with pytest.raises(IllegalValueError):
        conf["x0"] = 0

    # failed because the variable didn't exists
    with pytest.raises(KeyError):
        conf["x_0"] = 1

    # failed because forbidden clause is violated
    with pytest.raises(ForbiddenValueError):
        conf["x3"] = 2

    assert conf["x3"] == 1

    # successful operation 1
    x0_old = conf["x0"]
    if x0_old == 1:
        conf["x0"] = 2
    else:
        conf["x0"] = 1
    x0_new = conf["x0"]
    assert x0_old != x0_new
    pcs._check_configuration_rigorous(conf)
    assert conf["x2"] == 1

    # successful operation 2
    x1_old = conf["x1"]
    if x1_old == "ab":
        conf["x1"] = "cd"
    else:
        conf["x1"] = "ab"
    x1_new = conf["x1"]
    assert x1_old != x1_new
    pcs._check_configuration_rigorous(conf)

    with pytest.raises(KeyError):
        conf["x2"]


def test_setting_illegal_value():
    cs = ConfigurationSpace()
    cs.add(UniformFloatHyperparameter("x", 0, 1))
    configuration = {"x": 2}
    with pytest.raises(ValueError):
        Configuration(cs, values=configuration)


def test_keys():
    # A regression test to make sure issue #49 does no longer pop up. By
    # iterating over the configuration in the for loop, it should not raise
    # a KeyError if the child hyperparameter is inactive.
    cs = ConfigurationSpace()
    shrinkage = CategoricalHyperparameter(
        "shrinkage",
        ["None", "auto", "manual"],
        default_value="None",
    )
    shrinkage_factor = UniformFloatHyperparameter(
        "shrinkage_factor",
        0.0,
        1.0,
        0.5,
    )
    cs.add([shrinkage, shrinkage_factor])

    cs.add(EqualsCondition(shrinkage_factor, shrinkage, "manual"))

    for _ in range(10):
        config = cs.sample_configuration()
        {hp_name: config[hp_name] for hp_name in config if config[hp_name] is not None}


def test_configuration_acts_as_mapping(simple_cs: ConfigurationSpace):
    """This tests checks that a Configuration can be used as a a dictionary by
    checking indexing[], iteration ..., items, keys.
    """
    names = ["parent", "child", "friend"]
    values = [1, 2, 3]
    values_dict = dict(zip(names, values))

    config = Configuration(simple_cs, values=values_dict)

    # Test indexing
    assert config["parent"] == values_dict["parent"]
    assert config["child"] == values_dict["child"]
    for name in names:
        assert name in config
    assert "mouse" not in config

    # Test dict methods
    assert set(config.keys()) == set(names)
    assert set(config.values()) == set(values)
    assert set(config.items()) == set(values_dict.items())
    assert len(config) == 3

    # Test __iter__
    assert set(iter(config)) == set(names)

    # Test unpacking
    d = {**config}
    assert d == values_dict


def test_order_of_hyperparameters_is_same_as_config_space(
    simple_cs: ConfigurationSpace,
):
    """Test the keys respect the contract that they follow the same order that
    is present in the ConfigurationSpace.
    """
    # Deliberatily different values
    config = Configuration(simple_cs, values={"child": 2, "parent": 1, "friend": 3})
    assert config.keys() == simple_cs.keys()


def test_meta_field():
    cs = ConfigurationSpace()
    cs.add(
        UniformIntegerHyperparameter("uihp", lower=1, upper=10, meta={"uihp": True}),
    )
    cs.add(
        NormalIntegerHyperparameter(
            "nihp",
            mu=5,
            sigma=1,
            meta={"nihp": True},
            lower=1,
            upper=10,
        ),
    )
    cs.add(
        UniformFloatHyperparameter("ufhp", lower=1, upper=10, meta={"ufhp": True}),
    )
    cs.add(
        NormalFloatHyperparameter(
            "nfhp",
            mu=5,
            sigma=1,
            meta={"nfhp": True},
            lower=1,
            upper=10,
        ),
    )
    cs.add(
        CategoricalHyperparameter("chp", choices=["1", "2", "3"], meta={"chp": True}),
    )
    cs.add(
        OrdinalHyperparameter("ohp", sequence=["1", "2", "3"], meta={"ohp": True}),
    )
    cs.add(Constant("const", value=1, meta={"const": True}))
    parent = ConfigurationSpace()
    parent.add_configuration_space("sub", cs, delimiter=":")
    assert parent["sub:uihp"].meta == {"uihp": True}
    assert parent["sub:nihp"].meta == {"nihp": True}
    assert parent["sub:ufhp"].meta == {"ufhp": True}
    assert parent["sub:nfhp"].meta == {"nfhp": True}
    assert parent["sub:chp"].meta == {"chp": True}
    assert parent["sub:ohp"].meta == {"ohp": True}
    assert parent["sub:const"].meta == {"const": True}


def test_repr_roundtrip():
    cs = ConfigurationSpace()
    cs.add(UniformIntegerHyperparameter("uihp", lower=1, upper=10))
    cs.add(
        NormalIntegerHyperparameter("nihp", mu=1, sigma=1, lower=0.1, upper=1),
    )
    cs.add(UniformFloatHyperparameter("ufhp", lower=1, upper=10))
    cs.add(
        NormalFloatHyperparameter("nfhp", mu=1, sigma=1, lower=0.1, upper=1),
    )
    cs.add(CategoricalHyperparameter("chp", choices=["1", "2", "3"]))
    cs.add(OrdinalHyperparameter("ohp", sequence=["1", "2", "3"]))
    cs.add(Constant("const", value=1))
    default = cs.get_default_configuration()
    repr = default.__repr__()
    repr = repr.replace("})", "}, configuration_space=cs)")
    config = eval(repr)  # noqa: S307
    assert default == config


def test_configuration_space_can_be_made_with_sequence_of_hyperparameters() -> None:
    cs = ConfigurationSpace(
        name="myspace",
        space=[Float("a", (1.0, 10.0)), Integer("b", (1, 10))],
    )
    assert len(cs) == 2
    assert "a" in cs
    assert "b" in cs
