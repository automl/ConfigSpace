"""Testing the API for creating the different hyperparameters avialable.

These are intentionally verbose and using all parameters to ensure they maintain equality.
"""
from __future__ import annotations

from ConfigSpace import Beta, Categorical, Float, Int, Normal, Uniform
from ConfigSpace.hyperparameters import (BetaFloatHyperparameter,
                                         BetaIntegerHyperparameter,
                                         CategoricalHyperparameter,
                                         NormalFloatHyperparameter,
                                         NormalIntegerHyperparameter,
                                         OrdinalHyperparameter,
                                         UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter)


def test_uniform_int() -> None:
    """
    Expects
    -------
    * Should create an identical UniformIntegerHyperparameter
    """
    expected = UniformIntegerHyperparameter(
        "hp",
        lower=2,
        upper=10,
        default_value=5,
        q=2,
        log=True,
        meta={"a": "b"},
    )

    a = Int(
        "hp",
        bounds=(2, 10),
        default=5,
        distribution=Uniform(),
        q=2,
        log=True,
        meta={"a": "b"},
    )
    assert a == expected
    assert a.meta == expected.meta


def test_normal_int() -> None:
    """
    Expects
    -------
    * Should create an identical NormalIntegerHyperparameter with Normal distribution
    """
    expected = NormalIntegerHyperparameter(
        "hp",
        lower=2,
        upper=10,
        default_value=5,
        q=2,
        mu=5,
        sigma=1,
        log=True,
        meta={"a": "b"},
    )

    a = Int(
        "hp",
        bounds=(2, 10),
        distribution=Normal(mu=5, sigma=1),
        default=5,
        q=2,
        log=True,
        meta={"a": "b"},
    )

    assert a == expected
    assert a.meta == expected.meta


def test_beta_int() -> None:
    """
    Expects
    -------
    * Should create an identical BetaIntegerHyperparameter with a BetaDistribution
    """
    expected = BetaIntegerHyperparameter(
        "hp",
        lower=2,
        upper=10,
        alpha=1,
        beta=2,
        default_value=5,
        q=2,
        log=True,
        meta={"a": "b"},
    )

    a = Int(
        "hp",
        bounds=(2, 10),
        distribution=Beta(alpha=1, beta=2),
        default=5,
        q=2,
        log=True,
        meta={"a": "b"},
    )

    assert a == expected
    assert a.meta == expected.meta


def test_uniform_float() -> None:
    """
    Expects
    -------
    * Should create an identical UniformFloatHyperparameter with a UniformDistribution
    """
    expected = UniformFloatHyperparameter(
        "hp",
        lower=2,
        upper=10,
        default_value=5,
        q=2,
        log=True,
        meta={"a": "b"},
    )

    a = Float(
        "hp",
        bounds=(2, 10),
        default=5,
        distribution=Uniform(),
        q=2,
        log=True,
        meta={"a": "b"},
    )

    assert a == expected
    assert a.meta == expected.meta


def test_normal_float() -> None:
    """
    Expects
    -------
    * Should create an identical NormalFloatHyperparameter with a Normal distribution
    """
    expected = NormalFloatHyperparameter(
        "hp",
        lower=2,
        upper=10,
        mu=5,
        sigma=2,
        default_value=5,
        q=2,
        log=True,
        meta={"a": "b"},
    )

    a = Float(
        "hp",
        bounds=(2, 10),
        default=5,
        distribution=Normal(mu=5, sigma=2),
        q=2,
        log=True,
        meta={"a": "b"},
    )

    assert a == expected
    assert a.meta == expected.meta


def test_beta_float() -> None:
    """
    Expects
    -------
    * Should create an identical BetaFloatHyperparameter with a BetaDistribution
    """
    expected = BetaFloatHyperparameter(
        "hp",
        lower=2,
        upper=10,
        default_value=5,
        alpha=1,
        beta=2,
        log=True,
        meta={"a": "b"},
    )

    a = Float(
        "hp",
        bounds=(2, 10),
        default=5,
        distribution=Beta(alpha=1, beta=2),
        log=True,
        meta={"a": "b"},
    )

    assert a == expected
    assert a.meta == expected.meta


def test_categorical() -> None:
    """
    Expects
    -------
    * Should create an identical CategoricalHyperparameter
    """
    expected = CategoricalHyperparameter(
        "hp",
        choices=["a", "b", "c"],
        default_value="a",
        weights=[1, 2, 3],
        meta={"hello": "world"},
    )

    a = Categorical(
        "hp",
        items=["a", "b", "c"],
        default="a",
        weights=[1, 2, 3],
        ordered=False,
        meta={"hello": "world"},
    )

    assert a == expected
    assert a.meta == expected.meta


def test_ordinal() -> None:
    """
    Expects
    -------
    * Should create an identical CategoricalHyperparameter
    """
    expected = OrdinalHyperparameter(
        "hp",
        sequence=["a", "b", "c"],
        default_value="a",
        meta={"hello": "world"},
    )

    a = Categorical(
        "hp",
        items=["a", "b", "c"],
        default="a",
        ordered=True,
        meta={"hello": "world"},
    )

    assert a == expected
    assert a.meta == expected.meta
