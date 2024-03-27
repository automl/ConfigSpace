from __future__ import annotations

import os

import pytest

from ConfigSpace import (
    Beta,
    CategoricalHyperparameter,
    ConfigurationSpace,
    Float,
    ForbiddenInClause,
    Integer,
    Normal,
    Uniform,
)
from ConfigSpace.forbidden import ForbiddenLessThanRelation
from ConfigSpace.read_and_write.json import read, write
from ConfigSpace.read_and_write.pcs import read as read_pcs
from ConfigSpace.read_and_write.pcs_new import read as read_pcs_new


def test_serialize_forbidden_in_clause():
    cs = ConfigurationSpace()
    a = cs.add_hyperparameter(CategoricalHyperparameter("a", [0, 1, 2]))
    cs.add_forbidden_clause(ForbiddenInClause(a, [1, 2]))
    write(cs)


def test_serialize_forbidden_relation():
    cs = ConfigurationSpace()
    a = cs.add_hyperparameter(CategoricalHyperparameter("a", [0, 1, 2]))
    b = cs.add_hyperparameter(CategoricalHyperparameter("b", [0, 1, 2]))
    cs.add_forbidden_clause(ForbiddenLessThanRelation(a, b))
    write(cs)


def test_configspace_with_probabilities():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(
        CategoricalHyperparameter("a", [0, 1, 2], weights=[0.2, 0.2, 0.6]),
    )
    string = write(cs)
    new_cs = read(string)
    assert new_cs["a"].probabilities == (0.2, 0.2, 0.6)


this_file = os.path.abspath(__file__)
this_directory = os.path.dirname(this_file)
configuration_space_path = os.path.join(this_directory, "..", "test_searchspaces")
configuration_space_path = os.path.abspath(configuration_space_path)
pcs_files = sorted(
    os.path.join(configuration_space_path, filename)
    for filename in os.listdir(configuration_space_path)
    if ".pcs" in filename
)


@pytest.mark.parametrize("pcs_file", pcs_files)
def test_round_trip(pcs_file: str):
    with open(pcs_file) as fh:
        cs_string = fh.read().split("\n")
    try:
        cs = read_pcs(cs_string)
    except Exception:
        cs = read_pcs_new(cs_string)

    cs.name = pcs_file

    json_string = write(cs)
    new_cs = read(json_string)

    assert new_cs == cs


def test_beta_hyperparameter_serialization():
    # Test for BetaFloatHyperparameter
    cs = ConfigurationSpace(
        space={
            "p": Float("p", bounds=(0.0, 2.0), distribution=Beta(1.0, 2.0)),
        },
    )
    json_string = write(cs)
    new_cs = read(json_string)
    assert new_cs == cs

    # Test for BetaIntegerHyperparameter
    cs = ConfigurationSpace(
        space={
            "p": Integer("p", bounds=(0, 2), distribution=Beta(1.0, 2.0)),
        },
    )
    json_string = write(cs)
    new_cs = read(json_string)
    assert new_cs == cs


def test_float_hyperparameter_json_serialization():
    # Test for NormalFloatHyperparameter
    p = Float(
        "p",
        bounds=(1.0, 9.0),
        default=05.0,
        log=True,
        distribution=Normal(1.0, 0.6),
    )
    cs1 = ConfigurationSpace(space={"p": p})
    cs2 = read(write(cs1))
    assert cs1 == cs2

    # Test for UniformFloatHyperparameter
    p = Float(
        "p",
        bounds=(1.0, 9.0),
        default=2.0,
        log=True,
        distribution=Uniform(),
    )
    cs1 = ConfigurationSpace(space={"p": p})
    cs2 = read(write(cs1))
    assert cs1 == cs2


def test_integer_hyperparameter_json_serialization():
    # Test for NormalIntegerHyperparameter
    p = Integer(
        "p",
        bounds=(1, 17),
        default=2,
        log=True,
        distribution=Normal(1.0, 0.6),
    )
    cs1 = ConfigurationSpace(space={"p": p})
    cs2 = read(write(cs1))
    assert cs1 == cs2

    # Test for UniformIntegerHyperparameter
    p = Integer("p", bounds=(1, 17), default=2, log=True, distribution=Uniform())
    cs1 = ConfigurationSpace(space={"p": p})
    cs2 = read(write(cs1))
    assert cs1 == cs2
