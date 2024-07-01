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

import os

import numpy as np
import pytest
from pyparsing import warnings
from pytest import approx

from ConfigSpace import (
    AndConjunction,
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    GreaterThanCondition,
    LessThanCondition,
    OrConjunction,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.exceptions import NoPossibleNeighborsError

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ConfigSpace.read_and_write.pcs import read

from ConfigSpace.util import (
    change_hp_value,
    deactivate_inactive_hyperparameters,
    fix_types,
    generate_grid,
    get_one_exchange_neighbourhood,
    get_random_neighbor,
    impute_inactive_values,
)


def _test_random_neigbor(hp):
    cs = ConfigurationSpace()
    if not isinstance(hp, list):
        hp = [hp]
    for hp_ in hp:
        cs.add(hp_)
    cs.seed(1)
    config = cs.sample_configuration()
    for i in range(100):
        new_config = get_random_neighbor(config, i)
        assert config != new_config, cs


def _test_get_one_exchange_neighbourhood(hp):
    cs = ConfigurationSpace()
    num_neighbors = 0
    if not isinstance(hp, list):
        hp = [hp]

    for hp_ in hp:
        cs.add(hp_)
        if np.isinf(hp_.get_num_neighbors()):
            num_neighbors += 4
        else:
            num_neighbors += hp_.get_num_neighbors()

    cs.seed(1)
    config = cs.get_default_configuration()
    all_neighbors = []
    for i in range(100):
        neighborhood = get_one_exchange_neighbourhood(
            config,
            i,
            num_neighbors=4,
        )
        ns = list(neighborhood)
        for new_config in ns:
            assert config != new_config
            assert dict(config) != dict(new_config)
            all_neighbors.append(new_config)

    return all_neighbors


def test_impute_inactive_values():
    mini_autosklearn_config_space_path = os.path.join(
        os.path.dirname(__file__),
        "test_searchspaces",
        "mini_autosklearn_original.pcs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(mini_autosklearn_config_space_path) as fh:
            cs = read(fh)

    cs.seed(2)
    configuration = cs.sample_configuration()
    new_configuration = impute_inactive_values(configuration)
    assert id(configuration) != id(new_configuration)
    assert len(new_configuration) == 11
    for key in new_configuration:
        assert new_configuration[key] is not None
    assert new_configuration["random_forest:max_features"] == 10


def test_random_neighbor_float():
    hp = UniformFloatHyperparameter("a", 1, 10)
    _test_random_neigbor(hp)
    hp = UniformFloatHyperparameter("a", 1, 10, log=True)
    _test_random_neigbor(hp)


def test_random_neighborhood_float():
    hp = UniformFloatHyperparameter("a", 1, 10)
    all_neighbors = _test_get_one_exchange_neighbourhood(hp)
    all_neighbors = [neighbor["a"] for neighbor in all_neighbors]
    assert np.mean(all_neighbors) == pytest.approx(5.65, abs=1e-1)
    assert np.var(all_neighbors) == pytest.approx(2.85, abs=1e-2)
    hp = UniformFloatHyperparameter("a", 1, 10, log=True)
    all_neighbors = _test_get_one_exchange_neighbourhood(hp)
    all_neighbors = [neighbor["a"] for neighbor in all_neighbors]
    # Default value is 3.16
    assert np.mean(all_neighbors) == pytest.approx(3.61, abs=1e-2)
    assert np.var(all_neighbors) == pytest.approx(2.50, abs=1e-2)


def test_random_neighbor_int():
    hp = UniformIntegerHyperparameter("a", 1, 10)
    _test_random_neigbor(hp)
    hp = UniformIntegerHyperparameter("a", 1, 10, log=True)
    _test_random_neigbor(hp)


def test_random_neighborhood_int():
    hp = UniformIntegerHyperparameter("a", 1, 10)
    all_neighbors = _test_get_one_exchange_neighbourhood(hp)
    all_neighbors = [neighbor["a"] for neighbor in all_neighbors]
    assert pytest.approx(np.mean(all_neighbors), abs=1e-2) == 4.64
    assert pytest.approx(np.var(all_neighbors), abs=1e-2) == 3.57

    hp = UniformIntegerHyperparameter("a", 1, 10, log=True)
    all_neighbors = _test_get_one_exchange_neighbourhood(hp)
    all_neighbors = [neighbor["a"] for neighbor in all_neighbors]
    assert hp.default_value == 3

    assert pytest.approx(np.mean(all_neighbors), abs=1e-2) == 3.155
    assert pytest.approx(np.var(all_neighbors), abs=1e-2) == 3.09


def test_random_neighbor_cat():
    hp = CategoricalHyperparameter("a", [5, 6, 7, 8])
    all_neighbors = _test_get_one_exchange_neighbourhood(hp)
    all_neighbors = list(all_neighbors)
    assert len(all_neighbors) == 300  # 3 (neighbors) * 100 (samples)


def test_random_neighborhood_cat():
    hp = CategoricalHyperparameter("a", [5, 6, 7, 8])
    _test_random_neigbor(hp)


def test_random_neighbor_failing():
    hp = Constant("a", "b")
    with pytest.raises(NoPossibleNeighborsError):
        _test_random_neigbor(hp)

    hp = CategoricalHyperparameter("a", ["a"])
    with pytest.raises(NoPossibleNeighborsError):
        _test_random_neigbor(hp)


def test_random_neigbor_conditional():
    mini_autosklearn_config_space_path = os.path.join(
        os.path.dirname(__file__),
        "test_searchspaces",
        "mini_autosklearn_original.pcs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(mini_autosklearn_config_space_path) as fh:
            cs = read(fh)

    cs.seed(1)
    configuration = cs.get_default_configuration()
    for i in range(100):
        new_config = get_random_neighbor(configuration, i)
        assert configuration != new_config


def test_random_neigborhood_conditional():
    mini_autosklearn_config_space_path = os.path.join(
        os.path.dirname(__file__),
        "test_searchspaces",
        "mini_autosklearn_original.pcs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(mini_autosklearn_config_space_path) as fh:
            cs = read(fh)

    cs.seed(1)
    configuration = cs.get_default_configuration()
    for i in range(100):
        neighborhood = get_one_exchange_neighbourhood(configuration, i)
        for new_config in neighborhood:
            assert configuration != new_config


def test_deactivate_inactive_hyperparameters():
    diamond = ConfigurationSpace()
    head = CategoricalHyperparameter("head", [0, 1])
    left = CategoricalHyperparameter("left", [0, 1])
    right = CategoricalHyperparameter("right", [0, 1])
    bottom = CategoricalHyperparameter("bottom", [0, 1])
    diamond.add([head, left, right, bottom])
    diamond.add(EqualsCondition(left, head, 0))
    diamond.add(EqualsCondition(right, head, 0))
    diamond.add(
        AndConjunction(
            EqualsCondition(bottom, left, 0),
            EqualsCondition(bottom, right, 0),
        ),
    )

    c = deactivate_inactive_hyperparameters(
        {"head": 0, "left": 0, "right": 0, "bottom": 0},
        diamond,
    )
    diamond._check_configuration_rigorous(c)

    c = deactivate_inactive_hyperparameters(
        {"head": 1, "left": 0, "right": 0, "bottom": 0},
        diamond,
    )
    diamond._check_configuration_rigorous(c)

    c = deactivate_inactive_hyperparameters(
        {"head": 0, "left": 1, "right": 0, "bottom": 0},
        diamond,
    )
    diamond._check_configuration_rigorous(c)

    diamond = ConfigurationSpace()
    head = CategoricalHyperparameter("head", [0, 1])
    left = CategoricalHyperparameter("left", [0, 1])
    right = CategoricalHyperparameter("right", [0, 1])
    bottom = CategoricalHyperparameter("bottom", [0, 1])
    diamond.add([head, left, right, bottom])
    diamond.add(EqualsCondition(left, head, 0))
    diamond.add(EqualsCondition(right, head, 0))
    diamond.add(
        OrConjunction(
            EqualsCondition(bottom, left, 0),
            EqualsCondition(bottom, right, 0),
        ),
    )

    c = deactivate_inactive_hyperparameters(
        {"head": 0, "left": 0, "right": 0, "bottom": 0},
        diamond,
    )
    diamond._check_configuration_rigorous(c)

    c = deactivate_inactive_hyperparameters(
        {"head": 1, "left": 1, "right": 0, "bottom": 0},
        diamond,
    )
    diamond._check_configuration_rigorous(c)

    c = deactivate_inactive_hyperparameters(
        {"head": 0, "left": 1, "right": 0, "bottom": 0},
        diamond,
    )
    diamond._check_configuration_rigorous(c)

    plain = ConfigurationSpace()
    a = UniformIntegerHyperparameter("a", 0, 10)
    b = UniformIntegerHyperparameter("b", 0, 10)
    plain.add([a, b])
    c = deactivate_inactive_hyperparameters({"a": 5, "b": 6}, plain)
    c.check_valid_configuration()


def test_check_neighbouring_config_diamond():
    diamond = ConfigurationSpace()
    head = CategoricalHyperparameter("head", [0, 1])
    left = CategoricalHyperparameter("left", [0, 1])
    right = CategoricalHyperparameter("right", [0, 1, 2, 3])
    bottom = CategoricalHyperparameter("bottom", [0, 1])
    diamond.add([head, left, right, bottom])
    diamond.add(EqualsCondition(left, head, 0))
    diamond.add(EqualsCondition(right, head, 0))
    diamond.add(
        AndConjunction(
            EqualsCondition(bottom, left, 1),
            EqualsCondition(bottom, right, 1),
        ),
    )

    config = Configuration(diamond, {"bottom": 0, "head": 0, "left": 1, "right": 1})
    hp_name = "head"
    index = diamond.index_of[hp_name]
    neighbor_value = 1

    new_array = change_hp_value(
        diamond,
        config.get_array(),
        hp_name,
        neighbor_value,
        index,
    )
    expected_array = np.array([1, np.nan, np.nan, np.nan])

    np.testing.assert_almost_equal(new_array, expected_array)


def test_check_neighbouring_config_diamond_or_conjunction():
    diamond = ConfigurationSpace()
    top = CategoricalHyperparameter("top", [0, 1], 0)
    middle = CategoricalHyperparameter("middle", [0, 1], 1)
    bottom_left = CategoricalHyperparameter("bottom_left", [0, 1], 1)
    bottom_right = CategoricalHyperparameter("bottom_right", [0, 1, 2, 3], 1)

    diamond.add([top, bottom_left, bottom_right, middle])
    diamond.add(EqualsCondition(middle, top, 0))
    diamond.add(EqualsCondition(bottom_left, middle, 0))
    diamond.add(
        OrConjunction(
            EqualsCondition(bottom_right, middle, 1),
            EqualsCondition(bottom_right, top, 1),
        ),
    )

    config = Configuration(diamond, {"top": 0, "middle": 1, "bottom_right": 1})
    hp_name = "top"
    index = diamond.index_of[hp_name]
    neighbor_value = 1

    new_array = change_hp_value(
        diamond,
        config.get_array(),
        hp_name,
        neighbor_value,
        index,
    )
    expected_array = np.array([1, np.nan, np.nan, 1])

    np.testing.assert_almost_equal(new_array, expected_array)


def test_check_neighbouring_config_diamond_str():
    diamond = ConfigurationSpace()
    head = CategoricalHyperparameter("head", ["red", "green"])
    left = CategoricalHyperparameter("left", ["red", "green"])
    right = CategoricalHyperparameter("right", ["red", "green", "blue", "yellow"])
    bottom = CategoricalHyperparameter("bottom", ["red", "green"])
    diamond.add([head, left, right, bottom])
    diamond.add(EqualsCondition(left, head, "red"))
    diamond.add(EqualsCondition(right, head, "red"))
    diamond.add(
        AndConjunction(
            EqualsCondition(bottom, left, "green"),
            EqualsCondition(bottom, right, "green"),
        ),
    )

    config = Configuration(
        diamond,
        {"bottom": "red", "head": "red", "left": "green", "right": "green"},
    )
    hp_name = "head"
    index = diamond.index_of[hp_name]
    neighbor_value = 1

    new_array = change_hp_value(
        diamond,
        config.get_array(),
        hp_name,
        neighbor_value,
        index,
    )
    expected_array = np.array([1, np.nan, np.nan, np.nan])

    np.testing.assert_almost_equal(new_array, expected_array)


def test_fix_types():
    # Test categorical and ordinal
    for hyperparameter_type in [CategoricalHyperparameter, OrdinalHyperparameter]:
        cs = ConfigurationSpace()
        cs.add(
            [
                hyperparameter_type("bools", [True, False]),
                hyperparameter_type("ints", [1, 2, 3, 4, 5]),
                hyperparameter_type("floats", [1.5, 2.5, 3.5, 4.5, 5.5]),
                hyperparameter_type("str", ["string", "ding", "dong"]),
                hyperparameter_type("mixed", [2, True, 1.5, "string", False, "False"]),
            ],
        )
        c = dict(cs.get_default_configuration())
        # Check bools
        for b in [False, True]:
            c["bools"] = b
            c_str = {k: str(v) for k, v in c.items()}
            assert fix_types(c_str, cs) == c
        # Check legal mixed values
        for m in [2, True, 1.5, "string"]:
            c["mixed"] = m
            c_str = {k: str(v) for k, v in c.items()}
            assert fix_types(c_str, cs) == c
        # Check error on cornercase that cannot be caught
        for m in [False, "False"]:
            c["mixed"] = m
            c_str = {k: str(v) for k, v in c.items()}
            with pytest.raises(ValueError):
                fix_types(c_str, cs)
    # Test constant
    for m in [2, 1.5, "string"]:
        cs = ConfigurationSpace()
        cs.add(Constant("constant", m))
        c = dict(cs.get_default_configuration())
        c_str = {k: str(v) for k, v in c.items()}
        assert fix_types(c_str, cs) == c


def test_generate_grid():
    """Test grid generation."""
    # Sub-test 1
    cs = ConfigurationSpace(seed=1234)

    cat1 = CategoricalHyperparameter(name="cat1", choices=["T", "F"])
    const1 = Constant(name="const1", value=4)
    float1 = UniformFloatHyperparameter(name="float1", lower=-1, upper=1, log=False)
    int1 = UniformIntegerHyperparameter(name="int1", lower=10, upper=100, log=True)
    ord1 = OrdinalHyperparameter(name="ord1", sequence=["1", "2", "3"])

    cs.add([float1, int1, cat1, ord1, const1])

    num_steps_dict = {"float1": 11, "int1": 6}
    generated_grid = generate_grid(cs, num_steps_dict)

    # Check randomly pre-selected values in the generated_grid
    # 2 * 1 * 11 * 6 * 3 total diff. possible configurations
    assert len(generated_grid) == 396
    # Check 1st and last generated configurations completely:
    first_expected_dict = {
        "cat1": "T",
        "const1": 4,
        "float1": -1.0,
        "int1": 10,
        "ord1": "1",
    }
    last_expected_dict = {
        "cat1": "F",
        "const1": 4,
        "float1": 1.0,
        "int1": 100,
        "ord1": "3",
    }
    assert dict(generated_grid[0]) == first_expected_dict
    assert dict(generated_grid[-1]) == last_expected_dict
    assert generated_grid[198]["cat1"] == "F"
    assert generated_grid[45]["const1"] == 4
    # The 2 most frequently changing HPs (int1 and ord1) have 3 * 6 = 18 different values for
    # each value of float1, so the 4th value of float1 of -0.4 is reached after
    # 3 * 18 = 54 values in the generated_grid (and remains the same for the next 18 values):
    for i in range(18):
        assert generated_grid[54 + i]["float1"] == pytest.approx(-0.4, abs=1e-2)
    # 5th diff. value for int1 after 4 * 3 = 12 values. Reasoning as above.
    assert generated_grid[12]["int1"] == 63
    assert generated_grid[3]["ord1"] == "1"
    assert generated_grid[4]["ord1"] == "2"
    assert generated_grid[5]["ord1"] == "3"

    # Sub-test 2
    # Test for extreme cases: only numerical
    cs = ConfigurationSpace(seed=1234)
    cs.add([float1, int1])

    num_steps_dict = {"float1": 11, "int1": 6}
    generated_grid = generate_grid(cs, num_steps_dict)

    assert len(generated_grid) == 66
    # Check 1st and last generated configurations completely:
    first_expected_dict = {"float1": -1.0, "int1": 10}
    last_expected_dict = {"float1": 1.0, "int1": 100}
    assert dict(generated_grid[0]) == first_expected_dict
    assert dict(generated_grid[-1]) == last_expected_dict

    # Test: only categorical
    cs = ConfigurationSpace(seed=1234)
    cs.add([cat1])

    generated_grid = generate_grid(cs)

    assert len(generated_grid) == 2
    # Check 1st and last generated configurations completely:
    assert generated_grid[0]["cat1"] == "T"
    assert generated_grid[-1]["cat1"] == "F"

    # Test: only constant
    cs = ConfigurationSpace(seed=1234)
    cs.add([const1])

    generated_grid = generate_grid(cs)

    assert len(generated_grid) == 1
    # Check 1st and only generated configuration completely:
    assert generated_grid[0]["const1"] == 4

    # Test: no hyperparameters yet
    cs = ConfigurationSpace(seed=1234)

    generated_grid = generate_grid(cs, num_steps_dict)

    # For the case of no hyperparameters, in get_cartesian_product, itertools.product() returns
    # a single empty tuple element which leads to a single empty Configuration.
    assert len(generated_grid) == 0

    # Sub-test 3
    # The conditional space tested has 2 levels of conditions.
    cs2 = ConfigurationSpace(seed=123)
    float1 = UniformFloatHyperparameter(name="float1", lower=-1, upper=1, log=False)
    int1 = UniformIntegerHyperparameter(name="int1", lower=0, upper=1000, log=False)
    cs2.add([float1, int1])

    int2_cond = UniformIntegerHyperparameter(
        name="int2_cond",
        lower=10,
        upper=100,
        log=True,
    )
    cs2.add([int2_cond])
    cond_1 = AndConjunction(
        LessThanCondition(int2_cond, float1, -0.5),
        GreaterThanCondition(int2_cond, int1, 600),
    )
    cs2.add([cond_1])
    cat1_cond = CategoricalHyperparameter(name="cat1_cond", choices=["apple", "orange"])
    cs2.add([cat1_cond])
    cond_2 = AndConjunction(
        GreaterThanCondition(cat1_cond, int1, 300),
        LessThanCondition(cat1_cond, int1, 700),
        GreaterThanCondition(cat1_cond, float1, -0.5),
        LessThanCondition(cat1_cond, float1, 0.5),
    )
    cs2.add([cond_2])
    float2_cond = UniformFloatHyperparameter(
        name="float2_cond",
        lower=10.0,
        upper=100.0,
        log=True,
    )
    # 2nd level dependency in ConfigurationSpace tree being tested
    cs2.add([float2_cond])
    cond_3 = GreaterThanCondition(float2_cond, int2_cond, 50)
    cs2.add([cond_3])
    num_steps_dict1 = {"float1": 4, "int2_cond": 3, "float2_cond": 3, "int1": 3}
    generated_grid = generate_grid(cs2, num_steps_dict1)
    assert len(generated_grid) == 18

    # RR: I manually generated the grid and verified the values were correct.
    # Check 1st and last generated configurations completely:
    first_expected_dict = {"float1": -1.0, "int1": 0}
    last_expected_dict = {
        "float1": -1.0,
        "int1": 1000,
        "int2_cond": 100,
        "float2_cond": 100.0,
    }

    assert dict(generated_grid[0]) == first_expected_dict

    # This was having slight numerical instability (99.99999999999994 vs 100.0) and so
    # we manually do a pass over each value
    last_config = generated_grid[-1]
    for k, expected_value in last_config.items():
        generated_value = last_config[k]
        if isinstance(generated_value, float):
            assert generated_value == approx(expected_value)
        else:
            assert generated_value == expected_value
    # Here, we test that a few randomly chosen values in the generated grid
    # correspond to the ones I checked.
    assert generated_grid[3]["int1"] == 1000
    assert generated_grid[12]["cat1_cond"] == "orange"
    assert generated_grid[-2]["float2_cond"] == pytest.approx(
        31.622776601683803,
        abs=1e-3,
    )

    # Sub-test 4
    # Test: only a single hyperparameter and num_steps_dict is None
    cs = ConfigurationSpace(seed=1234)
    cs.add([float1])

    num_steps_dict = {"float1": 11}
    try:
        generated_grid = generate_grid(cs)
    except ValueError as e:
        assert (
            str(e) == "num_steps_dict is None or doesn't contain "
            "the number of points to divide float1 into. And its quantization "
            "factor is None. Please provide/set one of these values."
        )

    generated_grid = generate_grid(cs, num_steps_dict)

    assert len(generated_grid) == 11
    # Check 1st and last generated configurations completely:
    assert generated_grid[0]["float1"] == -1.0
    assert generated_grid[-1]["float1"] == 1.0

    # Test forbidden clause
    cs = ConfigurationSpace(seed=1234)
    cs.add([cat1, ord1, int1])
    cs.add(EqualsCondition(int1, cat1, "T"))  # int1 only active if cat1 == T
    cs.add(
        ForbiddenAndConjunction(  # Forbid ord1 == 3 if cat1 == F
            ForbiddenEqualsClause(cat1, "F"),
            ForbiddenEqualsClause(ord1, "3"),
        ),
    )

    generated_grid = generate_grid(cs, {"int1": 2})

    assert len(generated_grid) == 8
    assert dict(generated_grid[0]) == {"cat1": "F", "ord1": "1"}
    assert dict(generated_grid[1]) == {"cat1": "F", "ord1": "2"}
    assert dict(generated_grid[2]) == {"cat1": "T", "ord1": "1", "int1": 0}
    assert dict(generated_grid[-1]) == {"cat1": "T", "ord1": "3", "int1": 1000}
