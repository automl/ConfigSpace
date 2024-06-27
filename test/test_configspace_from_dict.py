"""This file tests the easy api to create configspaces
ConfigurationSpace({
    "constant": "hello",
    "depth": (1, 10),
    "lr": (0.1, 1.0),
    "categorical": ["a", "b"],
}).
"""

from __future__ import annotations

from typing import Any

import pytest

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    NormalFloatHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Constant is just a value
        (
            "a",
            Constant("hp", "a"),
        ),
        (
            1337,
            Constant("hp", 1337),
        ),
        (
            1,
            Constant("hp", 1),
        ),
        (
            1.0,
            Constant("hp", 1.0),
        ),
        # Boundaries are tuples of length 2, int for Integer
        (
            (1, 10),
            UniformIntegerHyperparameter("hp", 1, 10),
        ),
        (
            (-5, 5),
            UniformIntegerHyperparameter("hp", -5, 5),
        ),
        # Boundaries are tuples of length 2, float for Float
        (
            (1.0, 10.0),
            UniformFloatHyperparameter("hp", 1.0, 10.0),
        ),
        (
            (-5.5, 5.5),
            UniformFloatHyperparameter("hp", -5.5, 5.5),
        ),
        # Lists are categorical
        (
            ["a"],
            CategoricalHyperparameter("hp", ["a"]),
        ),
        (
            ["a", "b"],
            CategoricalHyperparameter("hp", ["a", "b"]),
        ),
        # Something that is already a hyperparameter will stay a hyperparameter
        (
            NormalFloatHyperparameter("hp", mu=1, sigma=10, lower=-10, upper=10),
            NormalFloatHyperparameter("hp", mu=1, sigma=10, lower=-10, upper=10),
        ),
        # We can't use {} for categoricals as it becomes undeterministic
        # Hence we give Categorical the tuple() syntax and not support
        # Ordinal
    ],
)
def test_individual_hyperparameters(value: Any, expected: Hyperparameter) -> None:
    cs = ConfigurationSpace({"hp": value})
    assert cs["hp"] == expected


@pytest.mark.parametrize("value", [(1, 10, 999), (10,), (1.0, 10.0, 999.0), (1.0,), ()])
def test_bad_tuple_in_dict(value: tuple[int, ...]) -> None:
    """Expects.
    -------
    * Using a tuple that doesn't have 2 values will raise an error.
    """
    with pytest.raises(ValueError):
        ConfigurationSpace({"hp": value})


@pytest.mark.parametrize("value", [[]])
def test_bad_categorical(value: list) -> None:
    """Expects.
    -------
    * Using an empty list will raise an error.
    """
    with pytest.raises(ValueError):
        ConfigurationSpace({"hp": value})
