from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ConfigSpace.read_and_write.pcs import read as read_pcs
    from ConfigSpace.read_and_write.pcs_new import read as read_pcs_new


def test_serialize_forbidden_in_clause(tmp_path: Path) -> None:
    cs = ConfigurationSpace({"a": [0, 1, 2]})
    cs.add(ForbiddenInClause(cs["a"], [1, 2]))
    cs.to_json(tmp_path / "cs.json")


def test_serialize_forbidden_relation(tmp_path: Path) -> None:
    cs = ConfigurationSpace({"a": [0, 1, 2], "b": [0, 1, 2]})
    cs.add(ForbiddenLessThanRelation(cs["a"], cs["b"]))
    cs.to_json(tmp_path / "cs.json")


def test_configspace_with_probabilities(tmp_path: Path) -> None:
    cs = ConfigurationSpace()
    cs.add(CategoricalHyperparameter("a", [0, 1, 2], weights=[0.2, 0.2, 0.6]))
    path = tmp_path / "cs.json"
    cs.to_json(path)
    new_cs = ConfigurationSpace.from_json(path)
    np.testing.assert_equal(new_cs["a"].probabilities, (0.2, 0.2, 0.6))  # type: ignore


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
def test_round_trip(pcs_file: str, tmp_path: Path) -> None:
    with open(pcs_file) as fh:
        cs_string = fh.read().split("\n")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            cs = read_pcs(cs_string)
        except Exception:
            cs = read_pcs_new(cs_string)

    cs.name = pcs_file

    tmp_file = tmp_path / "cs.json"
    cs.to_json(tmp_file)
    new_cs = ConfigurationSpace.from_json(tmp_file)
    assert new_cs == cs


def test_beta_hyperparameter_serialization(tmp_path: Path) -> None:
    # Test for BetaFloatHyperparameter
    cs = ConfigurationSpace(
        space={
            "p": Float("p", bounds=(0.0, 2.0), distribution=Beta(1.0, 2.0)),
        },
    )
    cs.to_json(tmp_path / "cs.json")
    new_cs = ConfigurationSpace.from_json(tmp_path / "cs.json")
    assert new_cs == cs

    # Test for BetaIntegerHyperparameter
    cs = ConfigurationSpace(
        space={
            "p": Integer("p", bounds=(0, 2), distribution=Beta(1.0, 2.0)),
        },
    )
    cs.to_json(tmp_path / "cs.json")
    new_cs = ConfigurationSpace.from_json(tmp_path / "cs.json")
    assert new_cs == cs


def test_float_hyperparameter_json_serialization(tmp_path: Path) -> None:
    # Test for NormalFloatHyperparameter
    p = Float(
        "p",
        bounds=(1.0, 9.0),
        default=05.0,
        log=True,
        distribution=Normal(1.0, 0.6),
    )
    cs1 = ConfigurationSpace(space={"p": p})

    cs1.to_json(tmp_path / "cs.json")
    cs2 = ConfigurationSpace.from_json(tmp_path / "cs.json")

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
    cs1.to_json(tmp_path / "cs.json")
    cs2 = ConfigurationSpace.from_json(tmp_path / "cs.json")
    assert cs1 == cs2


def test_integer_hyperparameter_json_serialization(tmp_path: Path) -> None:
    # Test for NormalIntegerHyperparameter
    p = Integer(
        "p",
        bounds=(1, 17),
        default=2,
        log=True,
        distribution=Normal(1.0, 0.6),
    )
    cs1 = ConfigurationSpace(space={"p": p})
    cs1.to_json(tmp_path / "cs.json")
    cs2 = ConfigurationSpace.from_json(tmp_path / "cs.json")
    assert cs1 == cs2

    # Test for UniformIntegerHyperparameter
    p = Integer("p", bounds=(1, 17), default=2, log=True, distribution=Uniform())
    cs1 = ConfigurationSpace(space={"p": p})
    cs1.to_json(tmp_path / "cs.json")
    cs2 = ConfigurationSpace.from_json(tmp_path / "cs.json")
    assert cs1 == cs2
