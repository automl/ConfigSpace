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


def test_add_hyperparameter():
    cs = ConfigurationSpace()
    hp = UniformIntegerHyperparameter("name", 0, 10)
    cs.add_hyperparameter(hp)


def test_add_non_hyperparameter():
    cs = ConfigurationSpace()
    with pytest.raises(TypeError):
        cs.add_hyperparameter(object())  # type: ignore


def test_add_hyperparameters_with_equal_names():
    cs = ConfigurationSpace()
    hp = UniformIntegerHyperparameter("name", 0, 10)
    cs.add_hyperparameter(hp)
    with pytest.raises(HyperparameterAlreadyExistsError):
        cs.add_hyperparameter(hp)


def test_illegal_default_configuration():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("loss", ["l1", "l2"], default_value="l1")
    hp2 = CategoricalHyperparameter("penalty", ["l1", "l2"], default_value="l1")
    cs.add_hyperparameter(hp1)
    cs.add_hyperparameter(hp2)
    forb1 = ForbiddenEqualsClause(hp1, "l1")
    forb2 = ForbiddenEqualsClause(hp2, "l1")
    forb3 = ForbiddenAndConjunction(forb1, forb2)

    with pytest.raises(ForbiddenValueError):
        cs.add_forbidden_clause(forb3)


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
        cs.add_condition(object())  # type: ignore


def test_hyperparameters_with_valid_condition():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add_hyperparameter(hp2)
    cond = EqualsCondition(hp2, hp1, 0)
    cs.add_condition(cond)
    assert len(cs._hyperparameters) == 2


def test_condition_without_added_hyperparameters():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cond = EqualsCondition(hp2, hp1, 0)

    with pytest.raises(ChildNotFoundError):
        cs.add_condition(cond)

    cs.add_hyperparameter(hp1)

    with pytest.raises(ChildNotFoundError):
        cs.add_condition(cond)

    # Test also the parent hyperparameter
    cs2 = ConfigurationSpace()
    cs2.add_hyperparameter(hp2)

    with pytest.raises(ParentNotFoundError):
        cs2.add_condition(cond)


def test_condition_with_cycles():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add_hyperparameter(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add_condition(cond1)
    cond2 = EqualsCondition(hp1, hp2, 0)

    with pytest.raises(CyclicDependancyError):
        cs.add_condition(cond2)


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
    cs.add_hyperparameter(hp1)
    cs.add_hyperparameter(hp2)
    cs.add_hyperparameter(hp3)
    cs.add_hyperparameter(hp4)

    cs.add_condition(andconj1)
    assert hp4 not in cs.get_all_unconditional_hyperparameters()


def test_add_second_condition_wo_conjunction():
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    hp3 = Constant("And", "True")

    cond1 = EqualsCondition(hp3, hp1, 1)
    cond2 = EqualsCondition(hp3, hp2, 1)

    cs = ConfigurationSpace()
    cs.add_hyperparameter(hp1)
    cs.add_hyperparameter(hp2)
    cs.add_hyperparameter(hp3)

    cs.add_condition(cond1)

    with pytest.raises(AmbiguousConditionError):
        cs.add_condition(cond2)


def test_add_forbidden_clause():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    cs.add_hyperparameter(hp1)
    forb = ForbiddenEqualsClause(hp1, 1)
    # TODO add checking whether a forbidden clause makes sense at all
    cs.add_forbidden_clause(forb)
    # TODO add something to properly retrieve the forbidden clauses
    assert (
        str(cs)
        == "Configuration space object:\n  Hyperparameters:\n    input1, Type: Categorical, Choices: {0, 1}, Default: 0\n  Forbidden Clauses:\n    Forbidden: input1 == 1\n"
    )


def test_add_forbidden_relation():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    hp2 = CategoricalHyperparameter("input2", [1, 0])
    cs.add_hyperparameters([hp1, hp2])
    forb = ForbiddenEqualsRelation(hp1, hp2)
    # TODO add checking whether a forbidden clause makes sense at all
    cs.add_forbidden_clause(forb)
    # TODO add something to properly retrieve the forbidden clauses
    assert (
        str(cs)
        == "Configuration space object:\n  Hyperparameters:\n    input1, Type: Categorical, Choices: {0, 1}, Default: 0\n    input2, Type: Categorical, Choices: {1, 0}, Default: 1\n  Forbidden Clauses:\n    Forbidden: input1 == input2\n"
    )


def test_add_forbidden_relation_categorical():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("input1", ["a", "b"], default_value="b")
    hp2 = CategoricalHyperparameter("input2", ["b", "c"], default_value="b")
    cs.add_hyperparameters([hp1, hp2])
    forb = ForbiddenEqualsRelation(hp1, hp2)
    with pytest.raises(ForbiddenValueError):
        cs.add_forbidden_clause(forb)


def test_add_forbidden_illegal():
    cs = ConfigurationSpace()
    hp = CategoricalHyperparameter("input1", [0, 1])
    forb = ForbiddenEqualsClause(hp, 1)

    with pytest.raises(HyperparameterNotFoundError):
        cs.add_forbidden_clause(forb)

    forb2 = ForbiddenEqualsClause(hp, 0)

    with pytest.raises(HyperparameterNotFoundError):
        cs.add_forbidden_clauses([forb, forb2])


def test_add_configuration_space():
    cs = ConfigurationSpace()
    hp1 = cs.add_hyperparameter(CategoricalHyperparameter("input1", [0, 1]))
    cs.add_forbidden_clause(ForbiddenEqualsClause(hp1, 1))
    hp2 = cs.add_hyperparameter(UniformIntegerHyperparameter("child", 0, 10))
    cs.add_condition(EqualsCondition(hp2, hp1, 0))
    cs2 = ConfigurationSpace()
    cs2.add_configuration_space("prefix", cs, delimiter="__")
    assert (
        str(cs2)
        == "Configuration space object:\n  Hyperparameters:\n    prefix__child, Type: UniformInteger, Range: [0, 10], Default: 5\n    prefix__input1, Type: Categorical, Choices: {0, 1}, Default: 0\n  Conditions:\n    prefix__child | prefix__input1 == 0\n  Forbidden Clauses:\n    Forbidden: prefix__input1 == 1\n"
    )


def test_add_configuration_space_conjunctions():
    cs1 = ConfigurationSpace()
    cs2 = ConfigurationSpace()

    hp1 = cs1.add_hyperparameter(CategoricalHyperparameter("input1", [0, 1]))
    hp2 = cs1.add_hyperparameter(CategoricalHyperparameter("input2", [0, 1]))
    hp3 = cs1.add_hyperparameter(UniformIntegerHyperparameter("child1", 0, 10))
    hp4 = cs1.add_hyperparameter(UniformIntegerHyperparameter("child2", 0, 10))

    cond1 = EqualsCondition(hp2, hp3, 0)
    cond2 = EqualsCondition(hp1, hp3, 5)
    cond3 = EqualsCondition(hp1, hp4, 1)
    andCond = AndConjunction(cond2, cond3)

    cs1.add_conditions([cond1, andCond])
    cs2.add_configuration_space(prefix="test", configuration_space=cs1)

    assert str(cs2).count("test:") == 10
    # Check that they're equal except for the "test:" prefix
    assert str(cs1) == str(cs2).replace("test:", "")


def test_add_conditions():
    cs1 = ConfigurationSpace()
    cs2 = ConfigurationSpace()

    hp1 = cs1.add_hyperparameter(CategoricalHyperparameter("input1", [0, 1]))
    cs2.add_hyperparameter(hp1)
    hp2 = cs1.add_hyperparameter(CategoricalHyperparameter("input2", [0, 1]))
    cs2.add_hyperparameter(hp2)
    hp3 = cs1.add_hyperparameter(UniformIntegerHyperparameter("child1", 0, 10))
    cs2.add_hyperparameter(hp3)
    hp4 = cs1.add_hyperparameter(UniformIntegerHyperparameter("child2", 0, 10))
    cs2.add_hyperparameter(hp4)

    cond1 = EqualsCondition(hp2, hp3, 0)
    cond2 = EqualsCondition(hp1, hp3, 5)
    cond3 = EqualsCondition(hp1, hp4, 1)
    andCond = AndConjunction(cond2, cond3)

    cs1.add_conditions([cond1, andCond])
    cs2.add_condition(cond1)
    cs2.add_condition(andCond)

    assert str(cs1) == str(cs2)


def test_get_hyperparamforbidden_clauseseters():
    cs = ConfigurationSpace()
    assert len(cs) == 0
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add_hyperparameter(hp1)
    assert [hp1] == list(cs.values())
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add_hyperparameter(hp2)
    cond1 = EqualsCondition(hp2, hp1, 1)
    cs.add_condition(cond1)
    assert [hp1, hp2] == list(cs.values())
    # TODO: I need more tests for the topological sort!
    assert [hp1, hp2] == list(cs.values())


def test_get_hyperparameters_topological_sort_simple():
    for _ in range(10):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs.add_hyperparameter(hp1)
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cs.add_hyperparameter(hp2)
        cond1 = EqualsCondition(hp2, hp1, 0)
        cs.add_condition(cond1)
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
        cs.add_hyperparameter(hp)

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

    cs.add_condition(cond4)
    hps = list(cs.values())
    # AND is moved to the front because of alphabetical sorting
    for hp, idx in zip(hyperparameters, [1, 2, 3, 4, 6, 0, 5]):
        assert hps.index(hp) == idx
        assert cs._hyperparameter_idx[hp.name] == idx
        assert cs._idx_to_hyperparameter[idx] == hp.name

    cs.add_condition(cond5)
    hps = list(cs.values())
    for hp, idx in zip(hyperparameters, [1, 2, 3, 6, 5, 0, 4]):
        assert hps.index(hp) == idx
        assert cs._hyperparameter_idx[hp.name] == idx
        assert cs._idx_to_hyperparameter[idx] == hp.name

    cs.add_condition(conj3)
    hps = list(cs.values())
    for hp, idx in zip(hyperparameters, [0, 1, 2, 5, 4, 6, 3]):
        assert hps.index(hp) == idx
        assert cs._hyperparameter_idx[hp.name] == idx
        assert cs._idx_to_hyperparameter[idx] == hp.name


def test_get_hyperparameter():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add_hyperparameter(hp2)

    retval = cs["parent"]
    assert hp1 == retval
    retval = cs["child"]
    assert hp2 == retval

    with pytest.raises(HyperparameterNotFoundError):
        cs["grandfather"]


def test_get_conditions():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add_hyperparameter(hp2)
    assert [] == cs.get_conditions()
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add_condition(cond1)
    assert [cond1] == cs.get_conditions()


def test_get_parent_and_chil_conditions_of():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add_hyperparameter(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add_condition(cond1)

    assert [cond1] == cs.get_parent_conditions_of(hp2.name)
    assert [cond1] == cs.get_parent_conditions_of(hp2)
    assert [cond1] == cs.get_child_conditions_of(hp1.name)
    assert [cond1] == cs.get_child_conditions_of(hp1)

    with pytest.raises(HyperparameterNotFoundError):
        cs.get_parents_of("Foo")

    with pytest.raises(HyperparameterNotFoundError):
        cs.get_children_of("Foo")


def test_get_parent_and_children_of():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add_hyperparameter(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add_condition(cond1)

    assert [hp1] == cs.get_parents_of(hp2.name)
    assert [hp1] == cs.get_parents_of(hp2)
    assert [hp2] == cs.get_children_of(hp1.name)
    assert [hp2] == cs.get_children_of(hp1)

    with pytest.raises(HyperparameterNotFoundError):
        cs.get_parents_of("Foo")

    with pytest.raises(HyperparameterNotFoundError):
        cs.get_children_of("Foo")


def test_check_configuration_input_checking():
    cs = ConfigurationSpace()
    with pytest.raises(TypeError):
        cs.check_configuration("String")  # type: ignore

    with pytest.raises(TypeError):
        cs.check_configuration_vector_representation("String")  # type: ignore


def test_check_configuration():
    # TODO this is only a smoke test
    # TODO actually, this rather tests the evaluate methods in the
    # conditions module!
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add_hyperparameter(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add_condition(cond1)
    # This automatically checks the configuration!
    Configuration(cs, {"parent": 0, "child": 5})

    # and now for something more complicated
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    cs.add_hyperparameter(hp2)
    hp3 = CategoricalHyperparameter("input3", [0, 1])
    cs.add_hyperparameter(hp3)
    hp4 = CategoricalHyperparameter("input4", [0, 1])
    cs.add_hyperparameter(hp4)
    hp5 = CategoricalHyperparameter("input5", [0, 1])
    cs.add_hyperparameter(hp5)
    hp6 = Constant("AND", "True")
    cs.add_hyperparameter(hp6)

    cond1 = EqualsCondition(hp6, hp1, 1)
    cond2 = NotEqualsCondition(hp6, hp2, 1)
    cond3 = InCondition(hp6, hp3, [1])
    cond4 = EqualsCondition(hp6, hp4, 1)
    cond5 = EqualsCondition(hp6, hp5, 1)

    conj1 = AndConjunction(cond1, cond2)
    conj2 = OrConjunction(conj1, cond3)
    conj3 = AndConjunction(conj2, cond4, cond5)
    cs.add_condition(conj3)

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
    cs.add_hyperparameter(metric)
    cs.add_hyperparameter(p)
    cs.add_hyperparameter(classifier)
    cs.add_condition(metric_depends_on_classifier)
    cs.add_condition(p_depends_on_metric)

    forbidden = ForbiddenEqualsClause(metric, "other")
    cs.add_forbidden_clause(forbidden)

    configuration = Configuration(cs, {"classifier": "extra_trees"})

    # check backward compatibility with checking configurations instead of vectors
    cs.check_configuration(configuration)


def test_check_forbidden_with_sampled_vector_configuration():
    cs = ConfigurationSpace()
    metric = CategoricalHyperparameter("metric", ["minkowski", "other"])
    cs.add_hyperparameter(metric)

    forbidden = ForbiddenEqualsClause(metric, "other")
    cs.add_forbidden_clause(forbidden)
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
    cs1.add_hyperparameter(hp1)
    cs1.add_hyperparameter(hp2)
    cs1.add_condition(cond1)
    cs2.add_hyperparameter(hp1)
    cs2.add_hyperparameter(hp2)
    cs2.add_condition(cond1)
    assert cs1 == cs2
    cs1.add_hyperparameter(hp3)
    assert cs1 != cs2


def test_neq():
    cs1 = ConfigurationSpace()
    assert cs1 != "ConfigurationSpace"


def test_repr():
    cs1 = ConfigurationSpace()
    retval = cs1.__str__()
    assert retval == "Configuration space object:\n  Hyperparameters:\n"

    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs1.add_hyperparameter(hp1)
    retval = cs1.__str__()
    assert (
        "Configuration space object:\n  Hyperparameters:\n    %s\n" % str(hp1) == retval
    )

    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs1.add_hyperparameter(hp2)
    cs1.add_condition(cond1)
    retval = cs1.__str__()
    assert (
        f"Configuration space object:\n  Hyperparameters:\n    {hp2}\n    {hp1}\n  Conditions:\n    {cond1}\n"
        == retval
    )


def test_sample_configuration():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = UniformIntegerHyperparameter("child", 0, 10)
    cs.add_hyperparameter(hp2)
    cond1 = EqualsCondition(hp2, hp1, 0)
    cs.add_condition(cond1)
    # This automatically checks the configuration!
    Configuration(cs, {"parent": 0, "child": 5})

    # and now for something more complicated
    cs = ConfigurationSpace(seed=1)
    hp1 = CategoricalHyperparameter("input1", [0, 1])
    cs.add_hyperparameter(hp1)
    hp2 = CategoricalHyperparameter("input2", [0, 1])
    cs.add_hyperparameter(hp2)
    hp3 = CategoricalHyperparameter("input3", [0, 1])
    cs.add_hyperparameter(hp3)
    hp4 = CategoricalHyperparameter("input4", [0, 1])
    cs.add_hyperparameter(hp4)
    hp5 = CategoricalHyperparameter("input5", [0, 1])
    cs.add_hyperparameter(hp5)
    hp6 = Constant("AND", "True")
    cs.add_hyperparameter(hp6)

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
    cs.add_condition(cond4)
    cs.add_condition(cond5)
    cs.add_condition(conj3)

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

    hyper_params = {}
    hyper_params["hp5"] = CategoricalHyperparameter("hp5", ["0", "1", "2"])
    hyper_params["hp7"] = CategoricalHyperparameter("hp7", ["3", "4", "5"])
    hyper_params["hp8"] = CategoricalHyperparameter("hp8", ["6", "7", "8"])
    for key in hyper_params:
        cs.add_hyperparameter(hyper_params[key])

    cs.add_condition(InCondition(hyper_params["hp5"], hyper_params["hp8"], ["6"]))

    cs.add_condition(
        OrConjunction(
            InCondition(hyper_params["hp7"], hyper_params["hp8"], ["7"]),
            InCondition(hyper_params["hp7"], hyper_params["hp5"], ["1"]),
        ),
    )

    for cfg, fixture in zip(
        cs.sample_configuration(6),
        [
            [1, np.nan, 1],
            [0, 1, 2],
            [0, 0, np.nan],
            [1, np.nan, 2],
            [1, np.nan, 1],
            [0, 2, np.nan],
            [0, 1, 1],
        ],
    ):
        np.testing.assert_array_almost_equal(cfg.get_array(), fixture)


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
    algo1_cs.add_hyperparameter(hp1)

    # create sub-configuration space for algorithm 2
    algo2_cs = ConfigurationSpace()
    hp2 = CategoricalHyperparameter(
        name="algo2_param1",
        choices=["X", "Y"],
        default_value="Y",
    )
    algo2_cs.add_hyperparameter(hp2)

    # create a configuration space and populate it with both the switch
    # and the two sub-configuration spaces
    cs = ConfigurationSpace()
    cs.add_hyperparameter(algo_switch)
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
    assert cs["switch"].choices == ("algo1", "algo2")
    assert cs["algo1_subspace:algo1_param1"].choices == ("A", "B")
    assert cs["algo2_subspace:algo2_param1"].choices == ("X", "Y")

    # check probabilities in the final configuration space
    assert cs["switch"].probabilities == (0.25, 0.75)
    assert cs["algo1_subspace:algo1_param1"].probabilities == (0.3, 0.7)
    assert cs["algo2_subspace:algo2_param1"].probabilities == (0.5, 0.5)

    # check default values in the final configuration space
    assert cs["switch"].default_value == "algo1"
    assert cs["algo1_subspace:algo1_param1"].default_value == "B"
    assert cs["algo2_subspace:algo2_param1"].default_value == "Y"


def test_acts_as_mapping_2():
    """Test that ConfigurationSpace can act as a mapping with iteration,
    indexing and items, values, keys.
    """
    cs = ConfigurationSpace()
    names = [f"name{i}" for i in range(5)]
    hyperparameters = [UniformIntegerHyperparameter(name, 0, 10) for name in names]
    cs.add_hyperparameters(hyperparameters)

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
    cs.add_hyperparameters([integer, cat, beta, norm])

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
    cs.add_conditions([cond_2, cond_3])

    # add some forbidden clauses too, to test that remove_parameter_priors keeps the forbiddens
    forbidden_clause_a = ForbiddenEqualsClause(cat, 0)
    forbidden_clause_c = ForbiddenEqualsClause(integer, 3)
    forbidden_clause_d = ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_c)
    cs.add_forbidden_clauses([forbidden_clause_c, forbidden_clause_d])

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
    expected_cs.add_hyperparameters([unif_integer, unif_cat, unif_beta, unif_norm])

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
    expected_cs.add_conditions([cond_2, cond_3])

    # add some forbidden clauses too, to test that remove_parameter_priors keeps the forbiddens
    forbidden_clause_a = ForbiddenEqualsClause(unif_cat, 0)
    forbidden_clause_c = ForbiddenEqualsClause(unif_integer, 3)
    forbidden_clause_d = ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_c)
    expected_cs.add_forbidden_clauses([forbidden_clause_c, forbidden_clause_d])

    # __eq__ not implemented, so this is the next best thing
    assert repr(uniform_cs) == repr(expected_cs)


def test_substitute_hyperparameters_in_conditions():
    cs1 = ConfigurationSpace()
    orig_hp1 = CategoricalHyperparameter("input1", [0, 1])
    orig_hp2 = CategoricalHyperparameter("input2", [0, 1])
    orig_hp3 = UniformIntegerHyperparameter("child1", 0, 10)
    orig_hp4 = UniformIntegerHyperparameter("child2", 0, 10)
    cs1.add_hyperparameters([orig_hp1, orig_hp2, orig_hp3, orig_hp4])
    cond1 = EqualsCondition(orig_hp2, orig_hp3, 0)
    cond2 = EqualsCondition(orig_hp1, orig_hp3, 5)
    cond3 = EqualsCondition(orig_hp1, orig_hp4, 1)
    andCond = AndConjunction(cond2, cond3)
    cs1.add_conditions([cond1, andCond])

    cs2 = ConfigurationSpace()
    sub_hp1 = CategoricalHyperparameter("input1", [0, 1, 2])
    sub_hp2 = CategoricalHyperparameter("input2", [0, 1, 3])
    sub_hp3 = NormalIntegerHyperparameter("child1", lower=0, upper=10, mu=5, sigma=2)
    sub_hp4 = BetaIntegerHyperparameter("child2", lower=0, upper=10, alpha=3, beta=5)
    cs2.add_hyperparameters([sub_hp1, sub_hp2, sub_hp3, sub_hp4])
    new_conditions = cs1.substitute_hyperparameters_in_conditions(
        cs1.get_conditions(),
        cs2,
    )

    test_cond1 = EqualsCondition(sub_hp2, sub_hp3, 0)
    test_cond2 = EqualsCondition(sub_hp1, sub_hp3, 5)
    test_cond3 = EqualsCondition(sub_hp1, sub_hp4, 1)
    test_andCond = AndConjunction(test_cond2, test_cond3)
    cs2.add_conditions([test_cond1, test_andCond])
    test_conditions = cs2.get_conditions()

    assert new_conditions[0] == test_conditions[0]
    assert new_conditions[1] == test_conditions[1]


def test_substitute_hyperparameters_in_inconditions():
    cs1 = ConfigurationSpace()
    a = UniformIntegerHyperparameter("a", lower=0, upper=10)
    b = UniformFloatHyperparameter("b", lower=1.0, upper=8.0, log=False)
    cs1.add_hyperparameters([a, b])

    cond = InCondition(b, a, [1, 2, 3, 4])
    cs1.add_conditions([cond])

    cs2 = ConfigurationSpace()
    sub_a = UniformIntegerHyperparameter("a", lower=0, upper=10)
    sub_b = UniformFloatHyperparameter("b", lower=1.0, upper=8.0, log=False)
    cs2.add_hyperparameters([sub_a, sub_b])
    new_conditions = cs1.substitute_hyperparameters_in_conditions(
        cs1.get_conditions(),
        cs2,
    )

    test_cond = InCondition(b, a, [1, 2, 3, 4])
    cs2.add_conditions([test_cond])
    test_conditions = cs2.get_conditions()

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
    cs1.add_hyperparameters([orig_hp1, orig_hp2, orig_hp3, orig_hp4])
    forb_1 = ForbiddenEqualsClause(orig_hp1, 0)
    forb_2 = ForbiddenEqualsClause(orig_hp2, 1)
    forb_3 = ForbiddenEqualsClause(orig_hp3, 10)
    forb_4 = ForbiddenAndConjunction(forb_1, forb_2)
    forb_5 = ForbiddenLessThanRelation(orig_hp1, orig_hp2)
    cs1.add_forbidden_clauses([forb_3, forb_4, forb_5])

    cs2 = ConfigurationSpace()
    sub_hp1 = CategoricalHyperparameter("input1", [0, 1, 2])
    sub_hp2 = CategoricalHyperparameter("input2", [0, 1, 3])
    sub_hp3 = NormalIntegerHyperparameter("input3", lower=0, upper=10, mu=5, sigma=2)
    sub_hp4 = BetaIntegerHyperparameter("input4", lower=0, upper=10, alpha=3, beta=5)
    cs2.add_hyperparameters([sub_hp1, sub_hp2, sub_hp3, sub_hp4])
    new_forbiddens = cs1.substitute_hyperparameters_in_forbiddens(
        cs1.get_forbiddens(),
        cs2,
    )

    test_forb_1 = ForbiddenEqualsClause(sub_hp1, 0)
    test_forb_2 = ForbiddenEqualsClause(sub_hp2, 1)
    test_forb_3 = ForbiddenEqualsClause(sub_hp3, 10)
    test_forb_4 = ForbiddenAndConjunction(test_forb_1, test_forb_2)
    test_forb_5 = ForbiddenLessThanRelation(sub_hp1, sub_hp2)
    cs2.add_forbidden_clauses([test_forb_3, test_forb_4, test_forb_5])
    test_forbiddens = cs2.get_forbiddens()

    assert new_forbiddens[2] == test_forbiddens[2]
    assert new_forbiddens[1] == test_forbiddens[1]
    assert new_forbiddens[0] == test_forbiddens[0]


def test_estimate_size():
    cs = ConfigurationSpace()
    assert cs.estimate_size() == 0
    cs.add_hyperparameter(Constant("constant", 0))
    assert cs.estimate_size() == 1
    cs.add_hyperparameter(UniformIntegerHyperparameter("integer", 0, 5))
    assert cs.estimate_size() == 6
    cs.add_hyperparameter(CategoricalHyperparameter("cat", [0, 1, 2]))
    assert cs.estimate_size() == 18
    cs.add_hyperparameter(UniformFloatHyperparameter("float", 0, 1))
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
    # Values are a little bit higher than one would expect because,
    # an integer range of [0,10] is transformed to [-0.499,10.499].
    vector_values = {
        "parent": 1,
        "child": 0.22727223140405708,
        "friend": 0.583333611112037,
    }
    vector = [0.0] * 3
    for name in simple_cs._hyperparameter_idx:
        vector[simple_cs._hyperparameter_idx[name]] = vector_values[name]
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
    a = cs.add_hyperparameter(UniformFloatHyperparameter("a", -5, 10))
    b = cs.add_hyperparameter(
        NormalFloatHyperparameter("b", 1, 2, log=True, lower=0.1, upper=5),
    )
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
        value_a = a.sample(rs)
        value_b = b.sample(rs)
        values_dict = {"a": value_a, "b": value_b}
        config = Configuration(cs, values=values_dict)
        string = json.dumps(dict(config))
        saved_value = json.loads(string)
        saved_value = byteify(saved_value)
        assert values_dict == saved_value


def test_setitem():
    """Checks overriding a sampled configuration."""
    pcs = ConfigurationSpace()
    pcs.add_hyperparameter(UniformIntegerHyperparameter("x0", 1, 5, default_value=1))
    x1 = pcs.add_hyperparameter(
        CategoricalHyperparameter("x1", ["ab", "bc", "cd", "de"], default_value="ab"),
    )

    # Condition
    x2 = pcs.add_hyperparameter(CategoricalHyperparameter("x2", [1, 2]))
    pcs.add_condition(EqualsCondition(x2, x1, "ab"))

    # Forbidden
    x3 = pcs.add_hyperparameter(CategoricalHyperparameter("x3", [1, 2]))
    pcs.add_forbidden_clause(ForbiddenEqualsClause(x3, 2))

    conf = pcs.get_default_configuration()

    # failed because it's a invalid configuration
    with pytest.raises(IllegalValueError):
        conf["x0"] = 0

    # failed because the variable didn't exists
    with pytest.raises(HyperparameterNotFoundError):
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
    cs.add_hyperparameter(UniformFloatHyperparameter("x", 0, 1))
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
    cs.add_hyperparameters([shrinkage, shrinkage_factor])

    cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))

    for _ in range(10):
        config = cs.sample_configuration()
        {hp_name: config[hp_name] for hp_name in config if config[hp_name] is not None}


def test_acts_as_mapping(simple_cs: ConfigurationSpace):
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
    cs.add_hyperparameter(
        UniformIntegerHyperparameter("uihp", lower=1, upper=10, meta={"uihp": True}),
    )
    cs.add_hyperparameter(
        NormalIntegerHyperparameter(
            "nihp",
            mu=5,
            sigma=1,
            meta={"nihp": True},
            lower=1,
            upper=10,
        ),
    )
    cs.add_hyperparameter(
        UniformFloatHyperparameter("ufhp", lower=1, upper=10, meta={"ufhp": True}),
    )
    cs.add_hyperparameter(
        NormalFloatHyperparameter(
            "nfhp",
            mu=5,
            sigma=1,
            meta={"nfhp": True},
            lower=1,
            upper=10,
        ),
    )
    cs.add_hyperparameter(
        CategoricalHyperparameter("chp", choices=["1", "2", "3"], meta={"chp": True}),
    )
    cs.add_hyperparameter(
        OrdinalHyperparameter("ohp", sequence=["1", "2", "3"], meta={"ohp": True}),
    )
    cs.add_hyperparameter(Constant("const", value=1, meta={"const": True}))
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
    cs.add_hyperparameter(UniformIntegerHyperparameter("uihp", lower=1, upper=10))
    cs.add_hyperparameter(
        NormalIntegerHyperparameter("nihp", mu=1, sigma=1, lower=0.1, upper=1),
    )
    cs.add_hyperparameter(UniformFloatHyperparameter("ufhp", lower=1, upper=10))
    cs.add_hyperparameter(
        NormalFloatHyperparameter("nfhp", mu=1, sigma=1, lower=0.1, upper=1),
    )
    cs.add_hyperparameter(CategoricalHyperparameter("chp", choices=["1", "2", "3"]))
    cs.add_hyperparameter(OrdinalHyperparameter("ohp", sequence=["1", "2", "3"]))
    cs.add_hyperparameter(Constant("const", value=1))
    default = cs.get_default_configuration()
    repr = default.__repr__()
    repr = repr.replace("})", "}, configuration_space=cs)")
    config = eval(repr)
    assert default == config
