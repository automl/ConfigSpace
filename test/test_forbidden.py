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
    ForbiddenGreaterThanClause,
    ForbiddenGreaterThanEqualsClause,
    ForbiddenGreaterThanEqualsRelation,
    ForbiddenGreaterThanRelation,
    ForbiddenInClause,
    ForbiddenLessThanClause,
    ForbiddenLessThanEqualsClause,
    ForbiddenLessThanEqualsRelation,
    ForbiddenLessThanRelation,
    ForbiddenOrConjunction,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Hyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


def test_forbidden_invalid_values():
    hp_int = UniformIntegerHyperparameter("child", 0, 10)
    hp_cat = CategoricalHyperparameter("grandchild", ["hot", "lukewarm", "cold"])

    # Test ForbiddenEquals
    with pytest.raises(ValueError):  # Cannot create clause with invalid value
        ForbiddenEqualsClause(hp_int, 12)

    with pytest.raises(ValueError):  # Cannot create clause with invalid value
        ForbiddenEqualsClause(hp_cat, "heat")

    # Test ForbiddenLessThan
    with pytest.raises(ValueError):  # Cannot create clause with invalid value
        ForbiddenLessThanClause(hp_int, 12)

    with pytest.raises(ValueError):  # Cannot create clause for categorical
        ForbiddenLessThanClause(hp_cat, "hot")

    with pytest.raises(ValueError):  # Cannot create clause with invalid value
        ForbiddenLessThanEqualsClause(hp_int, 12)

    with pytest.raises(ValueError):  # Cannot create clause for categorical
        ForbiddenLessThanEqualsClause(hp_cat, "hot")

    # Test ForbiddenGreaterThan
    with pytest.raises(ValueError):  # Cannot create clause with invalid value
        ForbiddenGreaterThanClause(hp_int, 12)

    with pytest.raises(ValueError):  # Cannot create clause for categorical
        ForbiddenGreaterThanClause(hp_cat, "hot")

    # Test ForbiddenGreaterThanEquals
    with pytest.raises(ValueError):  # Cannot create clause with invalid value
        ForbiddenGreaterThanEqualsClause(hp_int, 12)

    with pytest.raises(ValueError):  # Cannot create clause for categorical
        ForbiddenGreaterThanEqualsClause(hp_cat, "hot")

    # Test ForbiddenIn
    with pytest.raises(ValueError):  # Cannot create clause with invalid value
        ForbiddenInClause(hp_int, [12])

    with pytest.raises(ValueError):  # Cannot create clause with invalid value
        ForbiddenInClause(hp_cat, ["heat"])


@pytest.mark.parametrize(
    "hyperparameter,value,valid_values,invalid_values",
    [
        (
            UniformFloatHyperparameter("parent", 0.0, 1.0),
            0.99,
            [0.98, 0.98999999999, 0.99000009, 1.0],
            [
                0.99,
            ],
        ),
        (
            UniformIntegerHyperparameter("child", 0, 10),
            9,
            [*list(range(9)), 10],
            [9],
        ),
        (
            OrdinalHyperparameter("grandchild", ["hot", "lukewarm", "cold"]),
            "lukewarm",
            ["hot", "cold"],
            ["lukewarm"],
        ),
    ],
)
def test_forbidden_equals_clause(
    hyperparameter: Hyperparameter,
    value: float | int | str,
    valid_values: list[float | int | str],
    invalid_values: list[float | int | str],
):
    forb = ForbiddenEqualsClause(hyperparameter, value)
    forb_ = ForbiddenEqualsClause(hyperparameter, value)

    # Basic properties
    assert forb == forb_
    str_value = "'" + value + "'" if isinstance(value, str) else value
    assert str(forb) == f"Forbidden: {hyperparameter.name} == {str_value}"

    # Test values
    for valid_value in valid_values:
        assert not forb.is_forbidden_value({hyperparameter.name: valid_value})
    for invalid_value in invalid_values:
        assert forb.is_forbidden_value({hyperparameter.name: invalid_value})

    # Test vectors
    hyperparameter_idx = {hyperparameter.name: 0, "dummy_var": 1}
    forb.set_vector_idx(hyperparameter_idx)
    for valid_value in valid_values:
        valid_value_vector = hyperparameter.to_vector(valid_value)
        assert not forb.is_forbidden_vector(np.array([valid_value_vector, np.nan]))
    for invalid_value in invalid_values:
        invalid_value_vector = hyperparameter.to_vector(invalid_value)
        assert forb.is_forbidden_vector(np.array([invalid_value_vector, np.nan]))


@pytest.mark.parametrize(
    "hyperparameter,value,valid_values,invalid_values",
    [
        (
            UniformFloatHyperparameter("parent", 0.0, 1.0),
            0.99,
            [0.98, 0.98999999999, 0.99],
            [0.99000009, 1.0],
        ),
        (UniformIntegerHyperparameter("child", 0, 10), 9, [6, 7, 8, 9], [10]),
        (
            OrdinalHyperparameter("grandchild", ["hot", "lukewarm", "cold"]),
            "lukewarm",
            ["hot", "lukewarm"],
            ["cold"],
        ),
    ],
)
def test_forbidden_greater_than_clause(
    hyperparameter: Hyperparameter,
    value: float | int | str,
    valid_values: list[float | int | str],
    invalid_values: list[float | int | str],
):
    forb = ForbiddenGreaterThanClause(hyperparameter, value)
    forb_ = ForbiddenGreaterThanClause(hyperparameter, value)

    # Basic properties
    assert forb == forb_
    str_value = "'" + value + "'" if isinstance(value, str) else value
    assert str(forb) == f"Forbidden: {hyperparameter.name} > {str_value}"

    # Test values
    for valid_value in valid_values:
        assert not forb.is_forbidden_value({hyperparameter.name: valid_value})
    for invalid_value in invalid_values:
        assert forb.is_forbidden_value({hyperparameter.name: invalid_value})

    # Test vectors
    hyperparameter_idx = {hyperparameter.name: 0, "dummy_var": 1}
    forb.set_vector_idx(hyperparameter_idx)
    for valid_value in valid_values:
        valid_value_vector = hyperparameter.to_vector(valid_value)
        assert not forb.is_forbidden_vector(np.array([valid_value_vector, np.nan]))
    for invalid_value in invalid_values:
        invalid_value_vector = hyperparameter.to_vector(invalid_value)
        assert forb.is_forbidden_vector(np.array([invalid_value_vector, np.nan]))


@pytest.mark.parametrize(
    "hyperparameter,value,valid_values,invalid_values",
    [
        (
            UniformFloatHyperparameter("parent", 0.0, 1.0),
            0.99,
            [0.98, 0.98999999999],
            [0.99, 1.0],
        ),
        (UniformIntegerHyperparameter("child", 0, 10), 9, [6, 7, 8], [9, 10]),
        (
            OrdinalHyperparameter("grandchild", ["hot", "lukewarm", "cold"]),
            "lukewarm",
            ["hot"],
            ["lukewarm", "cold"],
        ),
    ],
)
def test_forbidden_greater_than_equals_clause(
    hyperparameter: Hyperparameter,
    value: float | int | str,
    valid_values: list[float | int | str],
    invalid_values: list[float | int | str],
):
    forb = ForbiddenGreaterThanEqualsClause(hyperparameter, value)
    forb_ = ForbiddenGreaterThanEqualsClause(hyperparameter, value)

    # Basic properties
    assert forb == forb_
    str_value = "'" + value + "'" if isinstance(value, str) else value
    assert str(forb) == f"Forbidden: {hyperparameter.name} >= {str_value}"

    # Test values
    for valid_value in valid_values:
        assert not forb.is_forbidden_value({hyperparameter.name: valid_value})
    for invalid_value in invalid_values:
        assert forb.is_forbidden_value({hyperparameter.name: invalid_value})

    # Test vectors
    hyperparameter_idx = {hyperparameter.name: 0, "dummy_var": 1}
    forb.set_vector_idx(hyperparameter_idx)
    for valid_value in valid_values:
        valid_value_vector = hyperparameter.to_vector(valid_value)
        assert not forb.is_forbidden_vector(np.array([valid_value_vector, np.nan]))
    for invalid_value in invalid_values:
        invalid_value_vector = hyperparameter.to_vector(invalid_value)
        assert forb.is_forbidden_vector(np.array([invalid_value_vector, np.nan]))


@pytest.mark.parametrize(
    "hyperparameter,value,valid_values,invalid_values",
    [
        (
            UniformFloatHyperparameter("parent", 0.0, 1.0),
            0.33,
            [0.3300001, 0.45, 0.85],
            [0.01, 0.15, 0.3, 0.32999999],
        ),
        (
            UniformIntegerHyperparameter("child", 0, 10),
            4,
            [4, 5, 6, 7, 8, 9, 10],
            [0, 1, 2, 3],
        ),
        (
            OrdinalHyperparameter("grandchild", ["hot", "lukewarm", "cold"]),
            "lukewarm",
            ["lukewarm", "cold"],
            ["hot"],
        ),
    ],
)
def test_forbidden_less_than_clause(
    hyperparameter: Hyperparameter,
    value: float | int | str,
    valid_values: list[float | int | str],
    invalid_values: list[float | int | str],
):
    forb = ForbiddenLessThanClause(hyperparameter, value)
    forb_ = ForbiddenLessThanClause(hyperparameter, value)

    # Basic properties
    assert forb == forb_
    str_value = "'" + value + "'" if isinstance(value, str) else value
    assert str(forb) == f"Forbidden: {hyperparameter.name} < {str_value}"

    # Test values
    for valid_value in valid_values:
        assert not forb.is_forbidden_value({hyperparameter.name: valid_value})
    for invalid_value in invalid_values:
        assert forb.is_forbidden_value({hyperparameter.name: invalid_value})

    # Test vectors
    hyperparameter_idx = {hyperparameter.name: 0, "dummy_var": 1}
    forb.set_vector_idx(hyperparameter_idx)
    for valid_value in valid_values:
        valid_value_vector = hyperparameter.to_vector(valid_value)
        assert not forb.is_forbidden_vector(np.array([valid_value_vector, np.nan]))
    for invalid_value in invalid_values:
        invalid_value_vector = hyperparameter.to_vector(invalid_value)
        assert forb.is_forbidden_vector(np.array([invalid_value_vector, np.nan]))


@pytest.mark.parametrize(
    "hyperparameter,value,valid_values,invalid_values",
    [
        (
            UniformFloatHyperparameter("parent", 0.0, 1.0),
            0.33,
            [0.3300001, 0.45, 0.85],
            [0.01, 0.15, 0.3, 0.32999999, 0.33],
        ),
        (
            UniformIntegerHyperparameter("child", 0, 10),
            4,
            [5, 6, 7, 8, 9, 10],
            [0, 1, 2, 3, 4],
        ),
        (
            OrdinalHyperparameter("grandchild", ["hot", "lukewarm", "cold"]),
            "lukewarm",
            ["cold"],
            ["hot", "lukewarm"],
        ),
    ],
)
def test_forbidden_less_equals_clause(
    hyperparameter: Hyperparameter,
    value: float | int | str,
    valid_values: list[float | int | str],
    invalid_values: list[float | int | str],
):
    forb = ForbiddenLessThanEqualsClause(hyperparameter, value)
    forb_ = ForbiddenLessThanEqualsClause(hyperparameter, value)

    # Basic properties
    assert forb == forb_
    str_value = "'" + value + "'" if isinstance(value, str) else value
    assert str(forb) == f"Forbidden: {hyperparameter.name} <= {str_value}"

    # Test values
    for valid_value in valid_values:
        assert not forb.is_forbidden_value({hyperparameter.name: valid_value})
    for invalid_value in invalid_values:
        assert forb.is_forbidden_value({hyperparameter.name: invalid_value})

    # Test vectors
    hyperparameter_idx = {hyperparameter.name: 0, "dummy_var": 1}
    forb.set_vector_idx(hyperparameter_idx)
    for valid_value in valid_values:
        valid_value_vector = hyperparameter.to_vector(valid_value)
        assert not forb.is_forbidden_vector(np.array([valid_value_vector, np.nan]))
    for invalid_value in invalid_values:
        invalid_value_vector = hyperparameter.to_vector(invalid_value)
        assert forb.is_forbidden_vector(np.array([invalid_value_vector, np.nan]))


@pytest.mark.parametrize(
    "hyperparameter,value,valid_values,invalid_values",
    [
        (
            CategoricalHyperparameter("parent", [0, 1, 2, 3, 4]),
            [3, 4],
            [0, 1, 2],
            [3, 4],
        ),
        (
            UniformFloatHyperparameter("child", 0.0, 1.0),
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.3300001, 0.45, 0.85],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ),
        (
            UniformIntegerHyperparameter("child2", 0, 10),
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9, 10],
            [0, 1, 2, 3, 4],
        ),
        (
            OrdinalHyperparameter("grandchild", ["hot", "lukewarm", "cold"]),
            ["cold", "hot"],
            ["lukewarm"],
            ["cold", "hot"],
        ),
    ],
)
def test_in_condition(
    hyperparameter: Hyperparameter,
    value: list[float | int | str],
    valid_values: list[float | int | str],
    invalid_values: list[float | int | str],
):
    forb = ForbiddenInClause(hyperparameter, value)
    forb_ = ForbiddenInClause(hyperparameter, value)

    # Basic properties
    assert forb == forb_
    str_value = ", ".join(
        [str(v) if not isinstance(v, str) else "'" + v + "'" for v in value],
    )
    assert str(forb) == f"Forbidden: {hyperparameter.name} in {{{str_value}}}"

    # Test values
    for valid_value in valid_values:
        assert not forb.is_forbidden_value({hyperparameter.name: valid_value})
    for invalid_value in invalid_values:
        assert forb.is_forbidden_value({hyperparameter.name: invalid_value})

    # Test vectors
    hyperparameter_idx = {hyperparameter.name: 0, "dummy_var": 1}
    forb.set_vector_idx(hyperparameter_idx)
    for valid_value in valid_values:
        valid_value_vector = hyperparameter.to_vector(valid_value)
        assert not forb.is_forbidden_vector(np.array([valid_value_vector, np.nan]))
    for invalid_value in invalid_values:
        invalid_value_vector = hyperparameter.to_vector(invalid_value)
        assert forb.is_forbidden_vector(np.array([invalid_value_vector, np.nan]))


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

    results = [False] * 53 + [True]

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


def test_or_conjunction():
    hp1 = CategoricalHyperparameter("parent", [0, 1])
    hp2 = UniformIntegerHyperparameter("child", 0, 2)
    hp3 = UniformIntegerHyperparameter("child2", 0, 2)
    hp4 = UniformIntegerHyperparameter("child3", 0, 2)

    forb2 = ForbiddenEqualsClause(hp1, 1)
    forb3 = ForbiddenInClause(hp2, range(2, 3))
    forb4 = ForbiddenInClause(hp3, range(2, 3))
    forb5 = ForbiddenInClause(hp4, range(2, 3))

    or1 = ForbiddenOrConjunction(forb2, forb3)
    or2 = ForbiddenOrConjunction(forb2, forb4)
    or3 = ForbiddenOrConjunction(forb2, forb5)

    total_or = ForbiddenOrConjunction(or1, or2, or3)
    assert (
        str(total_or)
        == "((Forbidden: parent == 1 || Forbidden: child in {2}) || (Forbidden: parent == 1 || Forbidden: child2 in {2}) || (Forbidden: parent == 1 || Forbidden: child3 in {2}))"
    )

    results = (
        [False] * 2
        + [True]
        + [False] * 2
        + [True] * 4
        + [False] * 2
        + [True]
        + [False] * 2
        + [True] * 40
    )
    for i, values in enumerate(product(range(2), range(3), range(3), range(3))):
        is_forbidden = total_or.is_forbidden_value(
            {
                "parent": values[0],
                "child": values[1],
                "child2": values[2],
                "child3": values[3],
            },
        )
        assert results[i] == is_forbidden

        assert not total_or.is_forbidden_value({})


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


def test_forbidden_serialisation_deserialisation():
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.read_and_write.dictionary import (
        FORBIDDEN_DECODERS,
        FORBIDDEN_ENCODERS,
    )

    hp1 = UniformIntegerHyperparameter("a", 0, 10, default_value=1)
    hp2 = UniformIntegerHyperparameter("b", 0, 10, default_value=1)
    cs = ConfigurationSpace()
    cs.add([hp1, hp2])

    for forbidden_type in (
        ForbiddenInClause,
        ForbiddenLessThanClause,
        ForbiddenLessThanEqualsClause,
        ForbiddenGreaterThanClause,
        ForbiddenGreaterThanEqualsClause,
    ):
        forbidden_value = 5 if forbidden_type != ForbiddenInClause else [4, 5]
        forbidden = forbidden_type(hp1, forbidden_value)
        decoder_id, encoder = FORBIDDEN_ENCODERS[forbidden_type]
        decoder = FORBIDDEN_DECODERS[decoder_id]
        encoded = encoder(forbidden, {})
        assert decoder(encoded, cs, {}) == forbidden

    for forbidden_type in (
        ForbiddenEqualsRelation,
        ForbiddenGreaterThanRelation,
        ForbiddenGreaterThanEqualsRelation,
        ForbiddenLessThanRelation,
        ForbiddenLessThanEqualsRelation,
    ):
        forbidden = forbidden_type(hp1, hp2)
        decoder_id, encoder = FORBIDDEN_ENCODERS[forbidden_type]
        decoder = FORBIDDEN_DECODERS[decoder_id]
        encoded = encoder(forbidden, {})
        assert decoder(encoded, cs, {}) == forbidden
