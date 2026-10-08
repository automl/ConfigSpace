from __future__ import annotations

import numpy as np
import pytest

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer


@pytest.fixture
def cs() -> ConfigurationSpace:
    """A space with one integer, one float and one categorical hyperparameter.

    Its defaults (n=5, x=0.5, c="a") are what the tests compare sampled and
    numpy-typed configurations against.
    """
    cs = ConfigurationSpace(seed=0)
    cs.add(
        [
            Integer("n", (1, 10), default=5),
            Float("x", (0, 1), default=0.5),
            Categorical("c", ["a", "b"], default="a"),
        ],
    )
    return cs


def test_numpy_and_builtin_values_hash_equal(cs: ConfigurationSpace) -> None:
    """Configurations with equal values hash equal whether the values are numpy or builtin.

    Setup: build the same configuration once from builtin values and once from numpy
    scalars (np.int64, np.float64, np.str_).
    Expect: the two compare equal and therefore hash equal.
    """
    builtin = Configuration(cs, values={"n": 5, "x": 0.5, "c": "a"})
    numpy = Configuration(
        cs,
        values={"n": np.int64(5), "x": np.float64(0.5), "c": np.str_("a")},
    )

    assert builtin == numpy
    assert hash(builtin) == hash(numpy)


def test_vector_constructed_configuration_hashes_like_default(
    cs: ConfigurationSpace,
) -> None:
    """A configuration rebuilt from the default's vector hashes like the default.

    Constructing from a vector yields np.str_ for categoricals, while the default
    configuration holds a builtin str, so this covers the path sampling takes.
    Expect: the two compare equal, hash equal, and collapse to one entry in a set.
    """
    default = cs.get_default_configuration()
    from_vector = Configuration(cs, vector=default.get_array())

    assert from_vector == default
    assert hash(from_vector) == hash(default)
    assert len({default, from_vector}) == 1


def test_sampled_configurations_hash_consistently_with_equality(
    cs: ConfigurationSpace,
) -> None:
    """Every sampled configuration hashes like its builtin-valued copy.

    Setup: sample 50 configurations and rebuild each from its values converted to
    builtin Python types. 50 samples cover every categorical choice many times over.
    Expect: each pair compares equal and hashes equal.
    """
    for sampled in cs.sample_configuration(50):
        values = {
            k: v.item() if isinstance(v, np.generic) else v for k, v in sampled.items()
        }
        builtin = Configuration(cs, values=values)

        assert sampled == builtin
        assert hash(sampled) == hash(builtin)


def test_unequal_configurations_hash_differently(cs: ConfigurationSpace) -> None:
    """Configurations that differ in one value get different hashes.

    Different hashes are not guaranteed in general, but for these small values a
    collision would mean the value is being ignored by the hash.
    """
    a = Configuration(cs, values={"n": 5, "x": 0.5, "c": "a"})
    b = Configuration(cs, values={"n": 5, "x": 0.5, "c": "b"})

    assert a != b
    assert hash(a) != hash(b)


def test_unhashable_categorical_choices_are_hashable_configurations() -> None:
    """Configurations holding unhashable categorical choices (lists) can still be hashed.

    Setup: a categorical whose choices are lists, which ConfigSpace accepts.
    Expect: hashing works, equal configurations hash equal, and both fit in one set entry.
    """
    cs = ConfigurationSpace(seed=0)
    cs.add([Categorical("c", [[1, 2], [3, 4]]), Integer("n", (1, 10))])

    a = Configuration(cs, values={"c": [1, 2], "n": 3})
    b = Configuration(cs, values={"c": [1, 2], "n": 3})

    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
