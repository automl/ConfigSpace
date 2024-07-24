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

import copy
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import pytest

from ConfigSpace.conditions import AndConjunction, EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenInClause
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.util import get_one_exchange_neighbourhood

META_DATA: Mapping[Hashable, Any] = {
    "additional": "meta-data",
    "useful": "for integrations",
    "input_id": 42,
}


def f() -> None:
    pass


def test_constant():
    # Test construction
    c1 = Constant("value", 1)
    c2 = Constant("value", 1)
    c3 = Constant("value", 2)
    c4 = Constant("valuee", 1)
    c5 = Constant("valueee", 2)
    c6 = Constant("valueee", [1, 2])
    _c6 = Constant("valueee", [1, 2])
    c7 = Constant("valueee", f)
    c8 = Constant("valueee", Path("path"))

    # Test attributes are accessible
    assert c5.name == "valueee"
    assert c5.value == 2
    assert c7.value == f

    # Test the string representation
    assert str(c1) == "value, Type: Constant, Value: 1"

    # Test the equals operator (and the ne operator in the last line)
    assert c1 != 1
    assert c1 == c2
    assert c1 != c3
    assert c1 != c4
    assert c1 != c5
    assert c6 == _c6

    # test that meta-data is stored correctly
    c1_meta = Constant("value", 1, meta=dict(META_DATA))
    assert c1_meta.meta == META_DATA

    # Test getting the size
    for constant in (c1, c2, c3, c4, c5, c1_meta, c6, c7, c8):
        assert constant.size == 1
        _ = str(constant)  # Ensure str repr works

    with pytest.raises(ValueError):
        _ = Constant("value", np.array([1, 2]))


def test_constant_pdf():
    c1 = Constant("valuee", 1)
    c2 = Constant("valueee", -2)

    # TODO - change this once the is_legal support is there - should then be zero
    # but does not have an actual impact of now
    point_1 = np.array([1])
    point_2 = np.array([-2])
    array_1 = np.array([1, 1])
    array_2 = np.array([-2, -2])
    array_3 = np.array([1, -2])

    wrong_shape_1 = np.array([[1]])
    wrong_shape_2 = np.array([1, 2, 3]).reshape(1, -1)
    wrong_shape_3 = np.array([1, 2, 3]).reshape(-1, 1)

    assert c1.pdf_values(point_1) == np.array([1.0])
    assert c2.pdf_values(point_2) == np.array([1.0])
    assert c1.pdf_values(point_2) == np.array([0.0])
    assert c2.pdf_values(point_1) == np.array([0.0])

    assert tuple(c1.pdf_values(array_1)) == tuple(np.array([1.0, 1.0]))
    assert tuple(c2.pdf_values(array_2)) == tuple(np.array([1.0, 1.0]))
    assert tuple(c1.pdf_values(array_2)) == tuple(np.array([0.0, 0.0]))
    assert tuple(c1.pdf_values(array_3)) == tuple(np.array([1.0, 0.0]))

    # it must be one-dimensional
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_1)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_2)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_3)

    c3 = Constant("valueee", [1, 2])
    c4 = Constant("valueee", f)
    c5 = Constant("valueee", Path("path"))

    assert c3.pdf_values([[1, 2]]) == np.array(1.0)
    assert c4.pdf_values([f]) == np.array(1.0)
    assert c5.pdf_values([Path("path")]) == np.array(1.0)


def test_constant__pdf():
    c1 = Constant("valuee", 1)
    c2 = Constant("valueee", -2)

    point_1 = np.array([1])
    point_2 = np.array([0])
    array_1 = np.array([1, 1])
    array_2 = np.array([0, 0])
    array_3 = np.array([1, 0])

    # These shapes are allowed in _pdf
    accepted_shape_1 = np.array([[1]])
    accepted_shape_2 = np.array([1, 2, 3]).reshape(1, -1)
    accepted_shape_3 = np.array([3, 2, 1]).reshape(-1, 1)

    assert c1.pdf_vector(point_1) == np.array([1.0])
    assert c1.pdf_vector(point_2) == np.array([0.0])

    assert c2.pdf_vector(point_1) == np.array([1.0])
    assert c2.pdf_vector(point_2) == np.array([0.0])

    assert tuple(c1.pdf_vector(array_1)) == tuple(np.array([1.0, 1.0]))
    assert tuple(c1.pdf_vector(array_2)) == tuple(np.array([0.0, 0.0]))
    assert tuple(c1.pdf_vector(array_3)) == tuple(np.array([1.0, 0.0]))

    assert tuple(c2.pdf_vector(array_1)) == tuple(np.array([1.0, 1.0]))
    assert tuple(c2.pdf_vector(array_2)) == tuple(np.array([0.0, 0.0]))
    assert tuple(c2.pdf_vector(array_3)) == tuple(np.array([1.0, 0.0]))

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1.pdf_vector(accepted_shape_1)
    c1.pdf_vector(accepted_shape_2)
    c1.pdf_vector(accepted_shape_3)


def test_constant_get_max_density():
    c1 = Constant("valuee", 1)
    c2 = Constant("valueee", -2)
    assert c1.get_max_density() == 1.0
    assert c2.get_max_density() == 1.0


def test_uniformfloat():
    # TODO test non-equality
    # TODO test sampling from a log-distribution which has a negative
    # lower value!
    f1 = UniformFloatHyperparameter("param", 0, 10)
    f1_ = UniformFloatHyperparameter("param", 0, 10)
    assert f1 == f1_
    assert str(f1) == "param, Type: UniformFloat, Range: [0.0, 10.0], Default: 5.0"

    # Test attributes are accessible
    assert f1.name == "param"
    assert f1.lower == pytest.approx(0.0)
    assert f1.upper == pytest.approx(10.0)
    assert f1.log is False
    assert f1.default_value == pytest.approx(5.0)
    assert f1._normalized_default_value == pytest.approx(0.5)

    f3 = UniformFloatHyperparameter("param", 0.00001, 10, log=True)
    f3_ = UniformFloatHyperparameter("param", 0.00001, 10, log=True)
    assert f3 == f3_
    assert (
        str(f3)
        == "param, Type: UniformFloat, Range: [1e-05, 10.0], Default: 0.01, on log-scale"
    )

    f4 = UniformFloatHyperparameter("param", 0, 10, default_value=1.0)
    f4_ = UniformFloatHyperparameter("param", 0, 10, default_value=1.0)
    # Test that a int default is converted to float
    f4__ = UniformFloatHyperparameter("param", 0, 10, default_value=1)
    assert f4 == f4_
    assert isinstance(f4.default_value, type(f4__.default_value))
    assert str(f4) == "param, Type: UniformFloat, Range: [0.0, 10.0], Default: 1.0"

    f5 = UniformFloatHyperparameter(
        "param",
        0.1,
        10,
        log=True,
        default_value=1.0,
    )
    f5_ = UniformFloatHyperparameter(
        "param",
        0.1,
        10,
        log=True,
        default_value=1.0,
    )
    assert f5 == f5_
    assert (
        str(f5)
        == "param, Type: UniformFloat, Range: [0.1, 10.0], Default: 1.0, on log-scale"
    )

    assert f1 != f4
    assert f1 != "UniformFloat"

    # test that meta-data is stored correctly
    f_meta = UniformFloatHyperparameter(
        "param",
        0.1,
        10,
        log=True,
        default_value=1.0,
        meta=META_DATA,
    )
    assert f_meta.meta == META_DATA

    # Test get_size
    for float_hp in (f1, f3, f4, f5):
        assert np.isinf(float_hp.size)
        _ = str(float_hp)  # Ensure str repr works


def test_uniformfloat_to_integer():
    f1 = UniformFloatHyperparameter("param", 1, 10, log=True)
    f2 = f1.to_integer()
    # TODO is this a useful rounding?
    # TODO should there be any rounding, if e.g. lower=0.1
    assert (
        str(f2)
        == "param, Type: UniformInteger, Range: [1, 10], Default: 3, on log-scale"
    )


def test_uniformfloat_illegal_bounds():
    with pytest.raises(
        ValueError,
        match="Hyperparameter 'param' has illegal settings",
    ) as e:
        _ = UniformFloatHyperparameter("param", 0, 10, log=True)

    with pytest.raises(
        ValueError,
        match=r"Negative lower bound 0.000000 for log-scale is not possible",
    ):
        raise e.value.__cause__  # type: ignore

    with pytest.raises(
        ValueError,
        match="Hyperparameter 'param' has illegal settings",
    ) as e:
        _ = UniformFloatHyperparameter("param", 1, 0)

    with pytest.raises(
        ValueError,
        match=r"Upper bound 0.000000 must be larger than lower bound 1.000000",
    ):
        raise e.value.__cause__  # type: ignore


def test_uniformfloat_is_legal():
    lower = 0.1
    upper = 10
    f1 = UniformFloatHyperparameter("param", lower, upper, log=True)

    assert f1.legal_value(3.0)
    assert f1.legal_value(3)
    assert not f1.legal_value(-0.1)
    assert not f1.legal_value(10.1)
    assert not f1.legal_value("AAA")  # type: ignore
    assert not f1.legal_value({})  # type: ignore

    # Test legal vector values
    assert f1.legal_vector(1.0)
    assert f1.legal_vector(0.0)
    assert f1.legal_vector(0)
    assert f1.legal_vector(0.3)
    assert not f1.legal_vector(-0.1)
    assert not f1.legal_vector(1.1)
    assert not f1.legal_vector("Hahaha")  # type: ignore


def test_uniformfloat_pdf():
    c1 = UniformFloatHyperparameter("param", lower=0, upper=10)
    c2 = UniformFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        log=True,
    )
    c3 = UniformFloatHyperparameter("param", lower=0, upper=0.5)

    point_1 = np.array([3])
    point_2 = np.array([7])
    point_3 = np.array([0.3])
    array_1 = np.array([3, 7, 5])
    point_outside_range = np.array([-1])
    point_outside_range_log = np.array([0.1])
    wrong_shape_1 = np.array([[3]])
    wrong_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    wrong_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    assert c1.pdf_values(point_1)[0] == pytest.approx(0.1)
    assert c2.pdf_values(point_2)[0] == pytest.approx(4.539992976248485e-05, abs=1e-3)
    assert c1.pdf_values(point_1)[0] == pytest.approx(0.1)
    assert c2.pdf_values(point_2)[0] == pytest.approx(4.539992976248485e-05, abs=1e-3)
    assert c3.pdf_values(point_3)[0] == pytest.approx(2.0)

    assert c1.pdf_values(point_outside_range)[0] == 0.0
    assert c2.pdf_values(point_outside_range_log)[0] == 0.0

    # this, however, is a negative value on a log param, which cannot be pulled into range
    with pytest.warns(RuntimeWarning, match="invalid value encountered in log"):
        assert c2.pdf_values(point_outside_range)[0] == 0.0

    array_results = c1.pdf_values(array_1)
    array_results_log = c2.pdf_values(array_1)
    expected_results = np.array([0.1, 0.1, 0.1])
    expected_log_results = np.array(
        [4.539992976248485e-05, 4.539992976248485e-05, 4.539992976248485e-05],
    )
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_log_results.shape
    for res, log_res, exp_res, exp_log_res in zip(
        array_results,
        array_results_log,
        expected_results,
        expected_log_results,
    ):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_log_res, abs=1e-5)

    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_1)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_2)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_3)


def test_uniformfloat__pdf():
    c1 = UniformFloatHyperparameter("param", lower=0, upper=10)
    c2 = UniformFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        log=True,
    )
    c3 = UniformFloatHyperparameter("param", lower=0, upper=0.5)

    point_1 = np.array([0.3])
    point_2 = np.array([1])
    point_3 = np.array([0.0])
    array_1 = np.array([0.3, 0.7, 1.01])
    point_outside_range_1 = np.array([-1])
    point_outside_range_2 = np.array([1.1])
    accepted_shape_1 = np.array([[0.3]])
    accepted_shape_2 = np.array([0.3, 0.5, 1.1]).reshape(1, -1)
    accepted_shape_3 = np.array([1.1, 0.5, 0.3]).reshape(-1, 1)

    assert c1.pdf_vector(point_1)[0] == pytest.approx(0.1, abs=1e-3)
    assert c2.pdf_vector(point_2)[0] == pytest.approx(4.539992976248485e-05, abs=1e-3)
    assert c1.pdf_vector(point_1)[0] == pytest.approx(0.1)
    assert c2.pdf_vector(point_2)[0] == pytest.approx(4.539992976248485e-05, abs=1e-3)
    assert c3.pdf_vector(point_3)[0] == pytest.approx(2.0)

    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    # since inverse_transform pulls everything into range,
    # even points outside get evaluated in range
    assert c1.pdf_vector(point_outside_range_1)[0] == pytest.approx(0.0)
    assert c2.pdf_vector(point_outside_range_2)[0] == pytest.approx(0.0)
    assert c1.pdf_vector(point_outside_range_2)[0] == pytest.approx(0.0)
    assert c2.pdf_vector(point_outside_range_1)[0] == pytest.approx(0.0)

    array_results = c1.pdf_vector(array_1)
    array_results_log = c2.pdf_vector(array_1)
    expected_results = np.array([0.1, 0.1, 0])
    expected_log_results = np.array([4.539992976248485e-05, 4.539992976248485e-05, 0.0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_log_results.shape
    for res, log_res, exp_res, exp_log_res in zip(
        array_results,
        array_results_log,
        expected_results,
        expected_log_results,
    ):
        assert res == pytest.approx(exp_res, abs=1e-5)
        assert log_res == pytest.approx(exp_log_res, abs=1e-5)

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1.pdf_vector(accepted_shape_1)
    c1.pdf_vector(accepted_shape_2)
    c1.pdf_vector(accepted_shape_3)


def test_uniformfloat_get_max_density():
    c1 = UniformFloatHyperparameter("param", lower=0, upper=10)
    c2 = UniformFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        log=True,
    )
    c3 = UniformFloatHyperparameter("param", lower=0, upper=0.5)
    assert c1.get_max_density() == 0.1
    assert c2.get_max_density() == pytest.approx(4.5401991009687765e-05)
    assert c3.get_max_density() == 2


def test_normalfloat():
    # TODO test non-equality
    f1 = NormalFloatHyperparameter("param", 0.5, 10.5, lower=-20, upper=20)
    f1_ = NormalFloatHyperparameter("param", 0.5, 10.5, lower=-20, upper=20)
    assert f1 == f1_
    assert (
        str(f1)
        == "param, Type: NormalFloat, Mu: 0.5, Sigma: 10.5, Range: [-20.0, 20.0], Default: 0.5"
    )

    expected = [0.59934283060225, 0.47234713976576, 0.62953770762014, 0.80460597128161]
    np.testing.assert_almost_equal(
        f1.neighbors_vectorized(0.5, n=4, seed=np.random.RandomState(42)),
        expected,
        decimal=14,
    )

    # Test attributes are accessible
    assert f1.name == "param"
    assert f1.mu == pytest.approx(0.5)
    assert f1.sigma == pytest.approx(10.5)
    assert f1.log is False
    assert f1.default_value == pytest.approx(0.5)
    assert f1._normalized_default_value == pytest.approx(0.5125)

    # Test copy
    copy_f1 = copy.copy(f1)

    assert copy_f1.name == f1.name
    assert copy_f1.mu == f1.mu
    assert copy_f1.sigma == f1.sigma
    assert copy_f1.default_value == f1.default_value

    f2 = NormalFloatHyperparameter("param", 0, 10, lower=-10, upper=10)
    f2_ = NormalFloatHyperparameter("param", 0, 10, lower=-10, upper=10)
    assert f2 == f2_
    assert (
        str(f2)
        == "param, Type: NormalFloat, Mu: 0.0, Sigma: 10.0, Range: [-10.0, 10.0], Default: 0.0"
    )

    assert f1 != f2
    assert f1 != "UniformFloat"

    with pytest.raises(ValueError, match=r"Illegal default value 5.0"):
        NormalFloatHyperparameter(
            "param",
            5,
            10,
            lower=0.1,
            upper=0.11,
            default_value=5.0,
            log=True,
        )

    f6 = NormalFloatHyperparameter(
        "param",
        5,
        10,
        lower=0.1,
        upper=10,
        default_value=5.0,
        log=True,
    )
    f6_ = NormalFloatHyperparameter(
        "param",
        5,
        10,
        lower=0.1,
        upper=10,
        default_value=5.0,
        log=True,
    )
    assert f6 == f6_
    assert (
        str(f6)
        == "param, Type: NormalFloat, Mu: 5.0, Sigma: 10.0, Range: [0.1, 10.0], "
        + "Default: 5.0, on log-scale"
    )

    # Due to seemingly different numbers with x86_64 and i686 architectures
    # we got these numbers, where the first one is slightly different
    # They are equal up to 14 decimal places
    expected = [0.59934283060225, 0.47234713976576, 0.62953770762014, 0.80460597128161]
    np.testing.assert_almost_equal(
        f6.neighbors_vectorized(0.5, n=4, seed=np.random.RandomState(42)),
        expected,
        decimal=14,
    )

    assert f1 != f2
    assert f1 != "UniformFloat"

    # test that meta-data is stored correctly
    f_meta = NormalFloatHyperparameter(
        "param",
        0.1,
        10,
        lower=0.1,
        upper=10,
        log=True,
        default_value=1.0,
        meta=dict(META_DATA),
    )
    assert f_meta.meta == META_DATA

    # Test get_size
    for float_hp in (f1, f2, f6):
        assert np.isinf(float_hp.size)
        _ = str(float_hp)  # Ensure str repr works

    with pytest.raises(ValueError):
        _ = NormalFloatHyperparameter(
            "param",
            5,
            10,
            lower=0.1,
            upper=10,
            default_value=10.01,
        )

    with pytest.raises(ValueError):
        _ = NormalFloatHyperparameter(
            "param",
            5,
            10,
            lower=0.1,
            upper=10,
            default_value=0.09,
        )


def test_normalfloat_to_uniformfloat():
    f1 = NormalFloatHyperparameter("param", 0, 10, lower=-30, upper=30)
    f1_expected = UniformFloatHyperparameter("param", -30, 30)
    f1_actual = f1.to_uniform()
    assert f1_expected == f1_actual

    f2 = NormalFloatHyperparameter("param", 0, 10, lower=-20, upper=20)
    f2_expected = UniformFloatHyperparameter("param", -20, 20)
    f2_actual = f2.to_uniform()
    assert f2_expected == f2_actual


def test_normalfloat_is_legal():
    f1 = NormalFloatHyperparameter("param", 0, 10, lower=-20, upper=20)
    assert f1.legal_value(3.0)
    assert f1.legal_value(2)
    assert not f1.legal_value("Hahaha")  # type: ignore

    # Test legal vector values
    assert f1.legal_vector(1.0)
    assert f1.legal_vector(0.0)
    assert f1.legal_vector(0)
    assert f1.legal_vector(0.3)
    assert not f1.legal_vector(-0.1)
    assert not f1.legal_vector(1.1)
    assert not f1.legal_vector("Hahaha")  # type: ignore

    f2 = NormalFloatHyperparameter(
        "param",
        5,
        10,
        lower=0.1,
        upper=10,
        default_value=5.0,
    )
    assert f2.legal_vector(1.0)
    assert f2.legal_vector(0.0)
    assert f2.legal_vector(0)
    assert f2.legal_vector(0.3)
    assert f2.legal_value(5.0)
    assert not f2.legal_value(10.01)
    assert not f2.legal_value(0.009)


def test_normalfloat_to_integer():
    f1 = NormalFloatHyperparameter("param", 0, 10, lower=-20, upper=20)
    f2_expected = NormalIntegerHyperparameter("param", 0, 10, lower=-20, upper=20)
    f2_actual = f1.to_integer()
    assert f2_expected == f2_actual


def test_normalfloat_pdf():
    c1 = NormalFloatHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        mu=np.exp(3),
        sigma=np.exp(2),
        log=True,
    )
    c3 = NormalFloatHyperparameter("param", lower=0, upper=0.5, mu=-1, sigma=0.2)

    point_1 = np.array([3])
    point_1_log = np.array([np.exp(3)])
    point_2 = np.array([10])
    point_2_log = np.array([np.exp(10)])
    point_3 = np.array([0])
    point_outside_range_1 = np.array([-0.01])
    point_outside_range_2 = np.array([10.01])
    point_outside_range_1_log = np.array([np.exp(-0.01)])
    point_outside_range_2_log = np.array([np.exp(10.01)])
    wrong_shape_1 = np.array([[3]])
    wrong_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    wrong_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    assert c1.pdf_values(point_1)[0] == pytest.approx(2.138045617479014)
    assert c2.pdf_values(point_1_log)[0] == pytest.approx(2.038104873599176)
    assert c1.pdf_values(point_2)[0] == pytest.approx(0.00467695579850518)
    assert c2.pdf_values(point_2_log)[0] == pytest.approx(0.009061204414610455)
    assert c3.pdf_values(point_3)[0] == c3.get_max_density()
    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1.pdf_values(point_outside_range_1)[0] == 0.0
    assert c1.pdf_values(point_outside_range_2)[0] == 0.0
    assert c2.pdf_values(point_outside_range_1_log)[0] == 0.0
    assert c2.pdf_values(point_outside_range_2_log)[0] == 0.0

    array_1 = np.array([3, 10, 10.01])
    array_1_log = np.array([np.exp(3), np.exp(10), np.exp(10.01)])
    array_results = c1.pdf_values(array_1)
    array_results_log = c2.pdf_values(array_1_log)
    np.testing.assert_almost_equal(
        array_results,
        np.array([2.13804561747901, 0.00467695579851, 0.0]),
        decimal=14,
    )
    np.testing.assert_almost_equal(
        array_results_log,
        np.array([2.03810487359918, 0.00906120441461, 0.0]),
        decimal=14,
    )

    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_1)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_2)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_3)


def test_normalfloat_get_max_density():
    c1 = NormalFloatHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        mu=np.exp(3),
        sigma=np.exp(2),
        log=True,
    )
    c3 = NormalFloatHyperparameter("param", lower=0, upper=0.5, mu=-1, sigma=0.2)
    assert c1.get_max_density() == pytest.approx(2.138045617479014, abs=1e-9)
    assert c2.get_max_density() == pytest.approx(2.038104873599176, abs=1e-9)
    assert c3.get_max_density() == pytest.approx(12.966261361167449, abs=1e-9)


def test_betafloat():
    # TODO test non-equality
    f1 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.0)
    f1_ = BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=3, beta=1)
    assert f1 == f1_
    assert (
        str(f1_)
        == "param, Type: BetaFloat, Alpha: 3.0, Beta: 1.0, Range: [-2.0, 2.0], Default: 2.0"
    )

    u1 = UniformFloatHyperparameter("param", lower=0.0, upper=1.0)
    b1 = BetaFloatHyperparameter("param", lower=0.0, upper=1.0, alpha=3.0, beta=1.0)

    # with identical domains, beta and uniform should sample the same points
    np.testing.assert_equal(
        u1.neighbors_vectorized(0.5, n=4, seed=np.random.RandomState(42)),
        b1.neighbors_vectorized(0.5, n=4, seed=np.random.RandomState(42)),
    )
    # Test copy
    copy_f1 = copy.copy(f1)
    assert copy_f1.name == f1.name

    f2 = BetaFloatHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        alpha=3.0,
        beta=1.0,
    )
    f2_ = BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=3, beta=1)
    assert f2 == f2_

    assert (
        str(f2)
        == "param, Type: BetaFloat, Alpha: 3.0, Beta: 1.0, Range: [-2.0, 2.0], Default: 2.0"
    )

    f3 = BetaFloatHyperparameter(
        "param",
        lower=10 ** (-5),
        upper=10.0,
        alpha=6.0,
        beta=2.0,
        log=True,
    )
    f3_ = BetaFloatHyperparameter(
        "param",
        lower=10 ** (-5),
        upper=10.0,
        alpha=6.0,
        beta=2.0,
        log=True,
    )
    assert f3 == f3_
    assert (
        str(f3)
        == "param, Type: BetaFloat, Alpha: 6.0, Beta: 2.0, Range: [1e-05, 10.0], Default: 1.0, on log-scale"
    )

    f4 = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=1000.0,
        default_value=32.0,
        alpha=2.0,
        beta=2.0,
        log=True,
    )
    f4_ = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=1000.0,
        default_value=32.0,
        alpha=2.0,
        beta=2.0,
        log=True,
    )

    assert f4 == f4_
    assert (
        str(f4)
        == "param, Type: BetaFloat, Alpha: 2.0, Beta: 2.0, Range: [1.0, 1000.0], Default: 32.0, on log-scale"
    )

    # test that meta-data is stored correctly
    f_meta = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=10.0,
        alpha=3.0,
        beta=2.0,
        log=False,
        meta=dict(META_DATA),
    )
    assert f_meta.meta == META_DATA


def test_betafloat_dist_parameters():
    # This one should just be created without raising an error - corresponds to uniform dist.
    BetaFloatHyperparameter("param", lower=0, upper=10.0, alpha=1, beta=1)

    # This one is not permitted as the co-domain is not finite
    with pytest.raises(ValueError):
        BetaFloatHyperparameter("param", lower=0, upper=100, alpha=0.99, beta=0.99)
    # And these parameters do not define a proper beta distribution whatsoever
    with pytest.raises(ValueError):
        BetaFloatHyperparameter("param", lower=0, upper=100, alpha=-0.1, beta=-0.1)

    # test parameters that do not create a legit beta distribution, one at a time
    with pytest.raises(ValueError):
        BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=-11, beta=5)
    with pytest.raises(ValueError):
        BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=5, beta=-11)

    # test parameters that do not yield a finite co-domain, one at a time
    with pytest.raises(ValueError):
        BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=0.5, beta=11)
    with pytest.raises(ValueError):
        BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=11, beta=0.5)


def test_betafloat_default_value():
    # should default to the maximal value in the search space
    f_max = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.0)
    assert f_max.default_value == pytest.approx(2.0)
    assert f_max._normalized_default_value == pytest.approx(1.0)

    f_max_log = BetaFloatHyperparameter(
        "param",
        lower=1.0,
        upper=10.0,
        alpha=3.0,
        beta=1.0,
        log=True,
    )
    assert f_max_log.default_value == pytest.approx(10.0)
    assert f_max_log._normalized_default_value == pytest.approx(1.0)

    # should default to the minimal value in the search space
    f_min = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=1.0, beta=1.5)
    assert f_min.default_value == pytest.approx(-2.0)
    assert f_min._normalized_default_value == pytest.approx(0.0)

    f_min_log = BetaFloatHyperparameter(
        "param",
        lower=1.0,
        upper=10.0,
        alpha=1.0,
        beta=1.5,
        log=True,
    )
    assert f_min_log.default_value == pytest.approx(1.0)
    assert f_min_log._normalized_default_value == pytest.approx(0.0)

    # Symmeric, should default to the middle
    f_symm = BetaFloatHyperparameter("param", lower=5, upper=9, alpha=4.6, beta=4.6)
    assert f_symm.default_value == pytest.approx(7)
    assert f_symm._normalized_default_value == pytest.approx(0.5)

    # This should yield a value that's halfway towards the max in logspace
    f_symm_log = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=np.exp(10),
        alpha=4.6,
        beta=4.6,
        log=True,
    )
    assert f_symm_log.default_value == pytest.approx(np.exp(5))
    assert f_symm_log._normalized_default_value == pytest.approx(0.5)

    # Uniform, should also default to the middle
    f_unif = BetaFloatHyperparameter("param", lower=2.2, upper=3.2, alpha=1.0, beta=1.0)
    assert f_unif.default_value == pytest.approx(2.7)
    assert f_unif._normalized_default_value == pytest.approx(0.5)

    # This should yield a value that's halfway towards the max in logspace
    f_unif_log = BetaFloatHyperparameter(
        "param",
        lower=np.exp(2.2),
        upper=np.exp(3.2),
        alpha=1.0,
        beta=1.0,
        log=True,
    )
    assert f_unif_log.default_value == pytest.approx(np.exp(2.7))
    assert f_unif_log._normalized_default_value == pytest.approx(0.5)

    # Then, test a case where the default value is the mode of the beta dist
    f_max = BetaFloatHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        alpha=4.7,
        beta=2.12,
    )
    assert f_max.default_value == pytest.approx(1.0705394190871367)
    assert f_max._normalized_default_value == pytest.approx(0.7676348547717842)

    f_max_log = BetaFloatHyperparameter(
        "param",
        lower=np.exp(-2.0),
        upper=np.exp(2.0),
        alpha=4.7,
        beta=2.12,
        log=True,
    )
    assert f_max_log.default_value == pytest.approx(np.exp(1.0705394190871367))
    assert f_max_log._normalized_default_value == pytest.approx(0.7676348547717842)

    # TODO log and quantization together does not yield a correct default for the beta
    # hyperparameter, but it is relatively close to being correct. However, it is not
    # being

    # The default value is independent of whether you log the parameter or not
    f_legal_nolog = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=10.0,
        alpha=3.0,
        beta=2.0,
        default_value=1,
        log=True,
    )
    f_legal_log = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=10.0,
        alpha=3.0,
        beta=2.0,
        default_value=1,
        log=False,
    )

    assert f_legal_nolog.default_value == pytest.approx(1)
    assert f_legal_log.default_value == pytest.approx(1)

    # These are necessary, as we bypass the same check in the UniformFloatHP by design
    with pytest.raises(ValueError, match="Illegal default value 0"):
        BetaFloatHyperparameter(
            "param",
            lower=1,
            upper=10.0,
            alpha=3.0,
            beta=2.0,
            default_value=0,
            log=False,
        )
    with pytest.raises(ValueError, match="Illegal default value 0"):
        BetaFloatHyperparameter(
            "param",
            lower=1,
            upper=1000.0,
            alpha=3.0,
            beta=2.0,
            default_value=0,
            log=True,
        )


def test_betafloat_to_uniformfloat():
    f1 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=4, beta=2)
    f1_expected = UniformFloatHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        default_value=1,
    )
    f1_actual = f1.to_uniform()
    assert f1_expected == f1_actual

    f2 = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=1000,
        alpha=3,
        beta=2,
        log=True,
    )
    f2_expected = UniformFloatHyperparameter(
        "param",
        lower=1,
        upper=1000,
        log=True,
        default_value=100,
    )
    f2_actual = f2.to_uniform()
    assert f2_expected == f2_actual


def test_betafloat_to_integer():
    f1 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=4, beta=2)
    f2_expected = BetaIntegerHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        alpha=4,
        beta=2,
    )
    f2_actual = f1.to_integer()
    assert f2_expected == f2_actual


def test_betafloat_pdf():
    c1 = BetaFloatHyperparameter("param", lower=0, upper=10, alpha=3, beta=2)
    c2 = BetaFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        alpha=3,
        beta=2,
        log=True,
    )
    c3 = BetaFloatHyperparameter("param", lower=0, upper=0.5, alpha=1.1, beta=25)

    point_1 = np.array([3])
    point_1_log = np.array([np.exp(3)])
    point_2 = np.array([9.9])
    point_2_log = np.array([np.exp(9.9)])
    point_3 = np.array([0.01])
    array_1 = np.array([3, 9.9, 10.01])
    array_1_log = np.array([np.exp(3), np.exp(9.9), np.exp(10.01)])
    point_outside_range_1 = np.array([-0.01])
    point_outside_range_2 = np.array([10.01])
    point_outside_range_1_log = np.array([np.exp(-0.01)])
    point_outside_range_2_log = np.array([np.exp(10.01)])
    wrong_shape_1 = np.array([[3]])
    wrong_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    wrong_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    assert c1.pdf_values(point_1)[0] == pytest.approx(0.7559999999999997)
    assert c2.pdf_values(point_1_log)[0] == pytest.approx(0.7559999999999997)
    assert c1.pdf_values(point_2)[0] == pytest.approx(0.11761200000000013)
    assert c2.pdf_values(point_2_log)[0] == pytest.approx(0.11761200000000013)
    assert c3.pdf_values(point_3)[0] == pytest.approx(15.131082000930544)
    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1.pdf_values(point_outside_range_1)[0] == 0.0
    assert c1.pdf_values(point_outside_range_2)[0] == 0.0
    assert c2.pdf_values(point_outside_range_1_log)[0] == 0.0
    assert c2.pdf_values(point_outside_range_2_log)[0] == 0.0

    array_results = c1.pdf_values(array_1)
    array_results_log = c2.pdf_values(array_1_log)
    expected_results = np.array([0.7559999999999997, 0.11761200000000013, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res in zip(
        array_results,
        array_results_log,
        expected_results,
    ):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_res)

    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_1)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_2)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_3)


def test_betafloat__pdf():
    c1 = BetaFloatHyperparameter("param", lower=0, upper=10, alpha=3, beta=2)
    c2 = BetaFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        alpha=3,
        beta=2,
        log=True,
    )
    c3 = BetaFloatHyperparameter("param", lower=0, upper=0.5, alpha=1.1, beta=25)

    point_1 = np.array([0.3])
    point_2 = np.array([0.99])
    point_3 = np.array([0.02])
    array_1 = np.array([0.3, 0.99, 1.01])
    point_outside_range_1 = np.array([-0.01])
    point_outside_range_2 = np.array([1.01])
    accepted_shape_1 = np.array([[0.3]])
    accepted_shape_2 = np.array([0.3, 0.5, 0.7]).reshape(1, -1)
    accepted_shape_3 = np.array([0.7, 0.5, 0.3]).reshape(-1, 1)

    assert c1.pdf_vector(point_1)[0] == pytest.approx(0.7559999999999997)
    assert c2.pdf_vector(point_1)[0] == pytest.approx(0.7559999999999997)
    assert c1.pdf_vector(point_2)[0] == pytest.approx(0.11761200000000013)
    assert c2.pdf_vector(point_2)[0] == pytest.approx(0.11761200000000013)
    assert c3.pdf_vector(point_3)[0] == pytest.approx(15.131082000930544)

    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1.pdf_vector(point_outside_range_1)[0] == 0.0
    assert c1.pdf_vector(point_outside_range_2)[0] == 0.0
    assert c2.pdf_vector(point_outside_range_1)[0] == 0.0
    assert c2.pdf_vector(point_outside_range_2)[0] == 0.0

    array_results = c1.pdf_vector(array_1)
    array_results_log = c2.pdf_vector(array_1)
    expected_results = np.array([0.7559999999999997, 0.11761200000000013, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res in zip(
        array_results,
        array_results_log,
        expected_results,
    ):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_res)

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1.pdf_vector(accepted_shape_1)
    c1.pdf_vector(accepted_shape_2)
    c1.pdf_vector(accepted_shape_3)


def test_betafloat_get_max_density():
    c1 = BetaFloatHyperparameter("param", lower=0, upper=10, alpha=3, beta=2)
    c2 = BetaFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        alpha=3,
        beta=2,
        log=True,
    )
    c3 = BetaFloatHyperparameter("param", lower=0, upper=0.5, alpha=1.1, beta=25)
    assert c1.get_max_density() == pytest.approx(1.7777777777777776)
    assert c2.get_max_density() == pytest.approx(1.7777777777777776)
    assert c3.get_max_density() == pytest.approx(19.002040689325636)


def test_uniforminteger():
    # TODO: rounding or converting or error message?

    f1 = UniformIntegerHyperparameter("param", 0.0, 5.0)
    f1_ = UniformIntegerHyperparameter("param", 0, 5)
    assert f1 == f1_
    assert str(f1) == "param, Type: UniformInteger, Range: [0, 5], Default: 2"

    # Test name is accessible
    assert f1.name == "param"
    assert f1.lower == 0
    assert f1.upper == 5
    assert f1.default_value == 2
    assert f1.log is False
    assert f1._normalized_default_value == 0.4

    f3 = UniformIntegerHyperparameter("param", 1, 10, log=True)
    f3_ = UniformIntegerHyperparameter("param", 1, 10, log=True)
    assert f3 == f3_
    assert (
        str(f3)
        == "param, Type: UniformInteger, Range: [1, 10], Default: 3, on log-scale"
    )

    f4 = UniformIntegerHyperparameter("param", 1, 10, default_value=1, log=True)
    f4_ = UniformIntegerHyperparameter("param", 1, 10, default_value=1, log=True)
    assert f4 == f4_
    assert (
        str(f4)
        == "param, Type: UniformInteger, Range: [1, 10], Default: 1, on log-scale"
    )

    assert f1 != "UniformFloat"

    # test that meta-data is stored correctly
    f_meta = UniformIntegerHyperparameter(
        "param",
        1,
        10,
        log=True,
        default_value=1,
        meta=dict(META_DATA),
    )
    assert f_meta.meta == META_DATA

    assert f1.size == 6
    assert f3.size == 10
    assert f4.size == 10


def test_uniformint_legal_float_values():
    n_iter = UniformIntegerHyperparameter("n_iter", 5.0, 1000.0, default_value=20.0)

    assert isinstance(n_iter.default_value, int)
    with pytest.raises(
        TypeError,
        match=(
            r"`default_value` for hyperparameter 'n_iter' must be an"
            r" integer. Got 'float' for default_value=20.5."
        ),
    ):
        _ = UniformIntegerHyperparameter("n_iter", 5.0, 1000.0, default_value=20.5)


def test_uniformint_illegal_bounds():
    with pytest.raises(
        ValueError,
        match="Hyperparameter 'param' has illegal settings",
    ) as e:
        UniformIntegerHyperparameter("param", 0, 10, log=True)

    with pytest.raises(
        ValueError,
        match=r"Negative lower bound 0.000000 for log-scale is not possible",
    ):
        raise e.value.__cause__  # type: ignore

    with pytest.raises(
        ValueError,
        match="Hyperparameter 'param' has illegal settings",
    ) as e:
        _ = UniformIntegerHyperparameter("param", 1, 0)

    with pytest.raises(
        ValueError,
        match=r"Upper bound 0.000000 must be larger than lower bound 1.000000",
    ):
        raise e.value.__cause__  # type: ignore


def test_uniformint_pdf():
    c1 = UniformIntegerHyperparameter("param", lower=0, upper=4)
    c2 = UniformIntegerHyperparameter("logparam", lower=1, upper=10000, log=True)
    c3 = UniformIntegerHyperparameter("param", lower=-1, upper=12)
    point_1 = np.array([0])
    point_1_log = np.array([1])
    point_2 = np.array([3.0])
    point_2_log = np.array([3.0])
    non_integer_point = np.array([3.7])
    array_1 = np.array([1, 3, 3.7])
    point_outside_range = np.array([-1])
    wrong_shape_1 = np.array([[3]])
    wrong_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    wrong_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    assert c1.pdf_values(point_1)[0] == pytest.approx(0.2, abs=1e-5)
    assert c1.pdf_values(point_2)[0] == pytest.approx(0.2, abs=1e-5)
    assert c1.pdf_values(non_integer_point)[0] == pytest.approx(0.0, abs=1e-5)

    assert c2.pdf_values(point_1_log)[0] == pytest.approx(0.0001, abs=1e-5)
    assert c2.pdf_values(np.array([10_000]))[0] == pytest.approx(0.0001, abs=1e-5)
    assert c2.pdf_values(np.array([1]))[0] == pytest.approx(0.0001, abs=1e-5)
    assert c2.pdf_values(np.array([2]))[0] == pytest.approx(0.0001, abs=1e-5)
    assert c2.pdf_values(point_2_log)[0] == pytest.approx(0.0001, abs=1e-5)
    assert c2.pdf_values(non_integer_point)[0] == pytest.approx(0.0, abs=1e-5)

    assert c3.pdf_values(point_1)[0] == pytest.approx(0.07142857142857142, abs=1e-5)

    # this, however, is a negative value on a log param, which cannot be pulled into range
    with pytest.warns(RuntimeWarning, match="invalid value encountered in log"):
        assert c2.pdf_values(point_outside_range)[0] == 0.0

    array_results = c1.pdf_values(array_1)
    array_results_log = c2.pdf_values(array_1)
    expected_results = np.array([0.2, 0.2, 0])
    expected_results_log = np.array([0.0001, 0.0001, 0])
    assert array_results.shape == pytest.approx(expected_results.shape)
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res, _ in zip(
        array_results,
        array_results,
        expected_results,
        expected_results_log,
    ):
        assert res == pytest.approx(exp_res, abs=1e-5)
        assert log_res == pytest.approx(exp_res, abs=1e-5)

    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_1)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_2)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_3)


def test_uniformint__pdf():
    c1 = UniformIntegerHyperparameter("param", lower=0, upper=4)
    c2 = UniformIntegerHyperparameter("logparam", lower=1, upper=10000, log=True)

    point_1 = np.array([0])
    point_2 = np.array([0.75])
    array_1 = np.array([0, 0.75, 1.1])
    point_outside_range = np.array([-0.1])
    accepted_shape_1 = np.array([[0.75]])
    accepted_shape_2 = np.array([0, 0.75, 1.1]).reshape(1, -1)
    accepted_shape_3 = np.array([1.1, 0.75, 0]).reshape(-1, 1)

    # need to lower the amount of places since the bounds
    # are inexact (._lower=-0.49999, ._upper=4.49999)
    assert c1.pdf_vector(point_1)[0] == pytest.approx(0.2, abs=1e-5)
    assert c2.pdf_vector(point_1)[0] == pytest.approx(0.0001, abs=1e-5)
    assert c1.pdf_vector(point_2)[0] == pytest.approx(0.2, abs=1e-5)
    assert c2.pdf_vector(point_2)[0] == pytest.approx(0.0001, abs=1e-5)

    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    # since inverse_transform pulls everything into range,
    # even points outside get evaluated in range
    assert c1.pdf_vector(point_outside_range)[0] == pytest.approx(0.0, abs=1e-5)
    assert c2.pdf_vector(point_outside_range)[0] == pytest.approx(0.0, abs=1e-5)

    array_results = c1.pdf_vector(array_1)
    array_results_log = c2.pdf_vector(array_1)
    expected_results = np.array([0.2, 0.2, 0])
    expected_results_log = np.array([0.0001, 0.0001, 0])
    assert array_results.shape == pytest.approx(expected_results.shape)
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res, exp_log_res in zip(
        array_results,
        array_results_log,
        expected_results,
        expected_results_log,
    ):
        assert res == pytest.approx(exp_res, abs=1e-5)
        assert log_res == pytest.approx(exp_log_res, abs=1e-5)

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1.pdf_vector(accepted_shape_1)
    c1.pdf_vector(accepted_shape_2)
    c1.pdf_vector(accepted_shape_3)


def test_uniformint_get_max_density():
    c1 = UniformIntegerHyperparameter("param", lower=0, upper=4)
    c2 = UniformIntegerHyperparameter("logparam", lower=1, upper=10000, log=True)
    c3 = UniformIntegerHyperparameter("param", lower=-1, upper=12)
    assert c1.get_max_density() == pytest.approx(0.2)
    assert c2.get_max_density() == pytest.approx(0.0001)
    assert c3.get_max_density() == pytest.approx(0.07142857142857142)


def test_uniformint_get_neighbors():
    rs = np.random.RandomState(seed=1)
    for i_upper in range(1, 10):
        c1 = UniformIntegerHyperparameter("param", lower=0, upper=i_upper)
        for i_value in range(i_upper + 1):
            neighbors = c1.neighbors_values(i_value, n=i_upper, seed=rs)
            expected = set(range(i_upper + 1)) - {i_value}
            assert set(neighbors) == expected, f"{i_value=}"


def test_normalint():
    # TODO test for unequal!
    f1 = NormalIntegerHyperparameter("param", 0.5, 5.5, lower=-10, upper=10)
    f1_ = NormalIntegerHyperparameter("param", 0.5, 5.5, lower=-10, upper=10)
    assert f1 == f1_
    assert (
        str(f1)
        == "param, Type: NormalInteger, Mu: 0.5, Sigma: 5.5, Range: [-10, 10], Default: 0"
    )

    # Test attributes are accessible
    assert f1.name == "param"
    assert f1.mu == 0.5
    assert f1.sigma == 5.5
    assert f1.log is False
    assert f1.default_value == 0.0
    assert f1._normalized_default_value == pytest.approx(0.5)
    assert f1.size == 21

    f3 = NormalIntegerHyperparameter("param", 1, 10, log=True, lower=1, upper=10)
    f3_ = NormalIntegerHyperparameter("param", 1, 10, log=True, lower=1, upper=10)
    assert f3 == f3_
    assert f3.size == 10
    assert (
        str(f3)
        == "param, Type: NormalInteger, Mu: 1.0, Sigma: 10.0, Range: [1, 10], Default: 1, on log-scale"
    )

    f4 = NormalIntegerHyperparameter(
        "param",
        1,
        10,
        lower=1,
        upper=10,
        default_value=3,
        log=True,
    )
    f4_ = NormalIntegerHyperparameter(
        "param",
        1,
        10,
        lower=1,
        upper=10,
        default_value=3,
        log=True,
    )
    assert f4 == f4_
    assert f4.size == 10
    assert (
        str(f4)
        == "param, Type: NormalInteger, Mu: 1.0, Sigma: 10.0, Range: [1, 10], Default: 3, on log-scale"
    )

    assert f1 != f4
    assert f1 != "UniformFloat"

    # test that meta-data is stored correctly
    f_meta = NormalIntegerHyperparameter(
        "param",
        1,
        10,
        lower=1,
        upper=10,
        default_value=1,
        log=True,
        meta=dict(META_DATA),
    )
    assert f_meta.meta == META_DATA

    # Bounded case
    f1 = NormalIntegerHyperparameter("param", 0, 10, lower=-100, upper=100)
    np.testing.assert_equal(
        f1.neighbors_vectorized(0.1, seed=np.random.RandomState(9001), n=1),
        np.array([0.14]),
    )
    np.testing.assert_equal(
        f1.neighbors_vectorized(0.1, seed=np.random.RandomState(9001), n=5),
        np.array([0.06, 0.065, 0.09, 0.14, 0.175]),
    )

    # Bounded case with default value out of bounds
    with pytest.raises(ValueError, match=r"Illegal default value 11"):
        _ = NormalIntegerHyperparameter(
            "param",
            5,
            10,
            lower=1,
            upper=10,
            default_value=11,
        )

    with pytest.raises(ValueError, match=r"Illegal default value 0"):
        _ = NormalIntegerHyperparameter(
            "param",
            5,
            10,
            lower=1,
            upper=10,
            default_value=0,
        )

    with pytest.raises(ValueError, match=r"must be positive for log-scale"):
        _ = NormalIntegerHyperparameter(
            "param",
            0,  # mu must be positive for log-scale
            10,
            lower=1,
            upper=10,
            default_value=0,
            log=True,
        )


def test_normalint_legal_float_values():
    n_iter = NormalIntegerHyperparameter(
        "n_iter",
        0,
        1.0,
        default_value=2.0,
        lower=0,
        upper=10,
    )
    assert isinstance(n_iter.default_value, int)
    with pytest.raises(
        TypeError,
        match=(
            r"`default_value` for hyperparameter 'n_iter' must be an integer. "
            r"Got 'float' for default_value=0.5."
        ),
    ):
        _ = UniformIntegerHyperparameter("n_iter", 0, 1.0, default_value=0.5)


def test_normalint_to_uniform():
    f1 = NormalIntegerHyperparameter("param", 0, 10, lower=-30, upper=30)
    f1_expected = UniformIntegerHyperparameter("param", -30, 30)
    f1_actual = f1.to_uniform()
    assert f1_expected == f1_actual


def test_normalint_is_legal():
    f1 = NormalIntegerHyperparameter("param", 1, 10, log=True, lower=1, upper=30)
    assert f1.legal_value(3.0)  # 3.0 behaves like an Integer  # type: ignore
    assert f1.legal_value(2)
    assert not f1.legal_value(3.1)  # type: ignore
    assert not f1.legal_value("BlaBlaBla")  # type: ignore
    assert not f1.legal_value(-15)

    # Test is legal vector
    assert f1.legal_vector(1.0)
    assert f1.legal_vector(0.0)
    assert f1.legal_vector(0)
    assert f1.legal_vector(f1.to_vector(10))
    assert not f1.legal_vector(-0.1)
    assert not f1.legal_vector(1.1)
    assert not f1.legal_vector("Hahaha")  # type: ignore

    f2 = NormalIntegerHyperparameter("param", 5, 10, lower=1, upper=10, default_value=5)
    assert f2.legal_value(5)
    assert not f2.legal_value(0)
    assert not f2.legal_value(11)


def test_normalint_pdf():
    c1 = NormalIntegerHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalIntegerHyperparameter(
        "logparam",
        lower=1,
        upper=1000,
        mu=np.exp(3),
        sigma=np.exp(2),
        log=True,
    )
    c3 = NormalIntegerHyperparameter("param", lower=0, upper=2, mu=-1.2, sigma=0.5)

    point_1 = np.array([3])
    point_1_log = np.array([10])
    point_2 = np.array([10])
    point_2_log = np.array([1000])
    point_3 = np.array([0])
    array_1 = np.array([3, 10, 11])
    array_1_log = np.array([10, 1000, 1001])
    point_outside_range_1 = np.array([-1])
    point_outside_range_2 = np.array([11])
    point_outside_range_1_log = np.array([0])
    point_outside_range_2_log = np.array([1001])
    non_integer_point = np.array([5.7])
    wrong_shape_1 = np.array([[3]])
    wrong_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    wrong_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    assert c1.pdf_values(point_1)[0] == pytest.approx(0.18913087287205807)
    assert c2.pdf_values(point_1_log)[0] == pytest.approx(0.0013829743550526114)
    assert c1.pdf_values(point_2)[0] == pytest.approx(0.00041372210458180426)
    assert c2.pdf_values(point_2_log)[0] == pytest.approx(0.0002698746029856508)
    assert c3.pdf_values(point_3)[0] == pytest.approx(0.9834724443747417)
    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1.pdf_values(point_outside_range_1)[0] == 0.0
    assert c1.pdf_values(point_outside_range_2)[0] == 0.0
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
        assert c2.pdf_values(point_outside_range_1_log)[0] == 0.0
    assert c2.pdf_values(point_outside_range_2_log)[0] == 0.0

    assert c1.pdf_values(non_integer_point)[0] == 0.0
    assert c2.pdf_values(non_integer_point)[0] == 0.0

    array_results = c1.pdf_values(array_1)
    array_results_log = c2.pdf_values(array_1_log)
    np.testing.assert_allclose(
        array_results,
        np.array([0.18913087287205807, 0.00041372210458180426, 0]),
    )
    np.testing.assert_allclose(
        array_results_log,
        np.array([0.0013829743550526114, 0.0002698746029856508, 0.0]),
    )

    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_1)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_2)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_3)


def test_normalint_get_max_density():
    c1 = NormalIntegerHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalIntegerHyperparameter(
        "logparam",
        lower=1,
        upper=1000,
        mu=np.exp(3),
        sigma=np.exp(2),
        log=True,
    )
    c3 = NormalIntegerHyperparameter("param", lower=0, upper=2, mu=-1.2, sigma=0.5)
    assert c1.get_max_density() == pytest.approx(2.118259218934877)
    assert c2.get_max_density() == pytest.approx(1.4595513607866044)
    assert c3.get_max_density() == pytest.approx(10.927444887375877)


############################################################
def test_betaint():
    # TODO test non-equality
    f1 = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.1)
    f1_ = BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=3, beta=1.1)
    assert f1 == f1_
    assert (
        str(f1)
        == "param, Type: BetaInteger, Alpha: 3.0, Beta: 1.1, Range: [-2, 2], Default: 2"
    )

    assert f1.alpha == pytest.approx(3.0)
    assert f1.beta == pytest.approx(1.1)

    # Test copy
    copy_f1 = copy.copy(f1)
    assert copy_f1.name == f1.name
    assert copy_f1.alpha == f1.alpha
    assert copy_f1.beta == f1.beta
    assert copy_f1.default_value == f1.default_value

    f2 = BetaIntegerHyperparameter(
        "param",
        lower=-2.0,
        upper=4.0,
        alpha=3.0,
        beta=1.1,
    )
    f2_ = BetaIntegerHyperparameter("param", lower=-2, upper=4, alpha=3, beta=1.1)
    assert f2 == f2_

    assert (
        str(f2)
        == "param, Type: BetaInteger, Alpha: 3.0, Beta: 1.1, Range: [-2, 4], Default: 4"
    )

    f3 = BetaIntegerHyperparameter(
        "param",
        lower=1,
        upper=1000,
        alpha=3.0,
        beta=2.0,
        log=True,
    )
    f3_ = BetaIntegerHyperparameter(
        "param",
        lower=1,
        upper=1000,
        alpha=3.0,
        beta=2.0,
        log=True,
    )
    assert f3 == f3_
    assert (
        str(f3)
        == "param, Type: BetaInteger, Alpha: 3.0, Beta: 2.0, Range: [1, 1000], Default: 100, on log-scale"
    )

    with pytest.raises(ValueError):
        BetaIntegerHyperparameter(
            "param",
            lower=-1,
            upper=10.0,
            alpha=6.0,
            beta=2.0,
            log=True,
        )

    # test that meta-data is stored correctly
    f_meta = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=10.0,
        alpha=3.0,
        beta=2.0,
        log=False,
        meta=dict(META_DATA),
    )
    assert f_meta.meta == META_DATA


def test_betaint_default_value():
    # should default to the maximal value in the search space
    f_max = BetaIntegerHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        alpha=3.0,
        beta=1.0,
    )
    assert f_max.default_value == pytest.approx(2.0)
    assert f_max._normalized_default_value == pytest.approx(1.0, abs=1e-4)

    # The normalized log defaults should be the same as if one were to create a uniform
    # distribution with the same default value as is generated by the beta
    f_max_log = BetaIntegerHyperparameter(
        "param",
        lower=1.0,
        upper=10.0,
        alpha=3.0,
        beta=1.0,
        log=True,
    )
    assert f_max_log.default_value == pytest.approx(10.0)
    assert f_max_log._normalized_default_value == pytest.approx(1.0)

    # should default to the minimal value in the search space
    f_min = BetaIntegerHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        alpha=1.0,
        beta=1.5,
    )
    assert f_min.default_value == pytest.approx(-2.0)
    assert f_min._normalized_default_value == pytest.approx(0.0, abs=1e-4)

    f_min_log = BetaIntegerHyperparameter(
        "param",
        lower=1.0,
        upper=10.0,
        alpha=1.0,
        beta=1.5,
        log=True,
    )
    assert f_min_log.default_value == pytest.approx(1.0)
    assert f_min_log._normalized_default_value == pytest.approx(0.0)

    # Symmeric, should default to the middle
    f_symm = BetaIntegerHyperparameter("param", lower=5, upper=9, alpha=4.6, beta=4.6)
    assert f_symm.default_value == pytest.approx(7)
    assert f_symm._normalized_default_value == pytest.approx(0.5)

    # This should yield a value that's approximately halfway towards the max in logspace
    f_symm_log = BetaIntegerHyperparameter(
        "param",
        lower=1,
        upper=np.round(np.exp(10)),
        alpha=4.6,
        beta=4.6,
        log=True,
    )
    assert f_symm_log.default_value == pytest.approx(148)
    assert f_symm_log._normalized_default_value == pytest.approx(0.5, abs=1e-3)

    # Uniform, should also default to the middle
    f_unif = BetaIntegerHyperparameter("param", lower=2, upper=6, alpha=1.0, beta=1.0)
    assert f_unif.default_value == pytest.approx(4)
    assert f_unif._normalized_default_value == pytest.approx(0.5)

    # This should yield a value that's halfway towards the max in logspace
    f_unif_log = BetaIntegerHyperparameter(
        "param",
        lower=1,
        upper=np.round(np.exp(10)),
        alpha=1,
        beta=1,
        log=True,
    )
    assert f_unif_log.default_value == pytest.approx(148)
    assert f_unif_log._normalized_default_value == pytest.approx(0.5, abs=1e-3)

    # Then, test a case where the default value is the mode of the beta dist somewhere in
    # the interior of the search space - but not the center
    f_max = BetaIntegerHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        alpha=4.7,
        beta=2.12,
    )
    assert f_max.default_value == pytest.approx(1.0)
    assert f_max._normalized_default_value == pytest.approx(0.75)

    f_max_log = BetaIntegerHyperparameter(
        "param",
        lower=1,
        upper=np.round(np.exp(10)),
        alpha=4.7,
        beta=2.12,
        log=True,
    )
    assert f_max_log.default_value == pytest.approx(2157)
    assert f_max_log._normalized_default_value == pytest.approx(0.767648988)

    # These parameters yield a mode at approximately 1.1, so should thus yield default at 1
    f_quant = BetaIntegerHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        alpha=4.7,
        beta=2.12,
    )
    assert f_quant.default_value == pytest.approx(1.0)
    assert f_quant._normalized_default_value == pytest.approx(0.75)

    # TODO log and quantization together does not yield a correct default for the beta
    # hyperparameter, but it is relatively close to being correct.

    # The default value is independent of whether you log the parameter or not
    f_legal_nolog = BetaIntegerHyperparameter(
        "param",
        lower=1,
        upper=10.0,
        alpha=3.0,
        beta=2.0,
        default_value=1,
        log=True,
    )
    f_legal_log = BetaIntegerHyperparameter(
        "param",
        lower=1,
        upper=10.0,
        alpha=3.0,
        beta=2.0,
        default_value=1,
        log=False,
    )

    assert f_legal_nolog.default_value == pytest.approx(1)
    assert f_legal_log.default_value == pytest.approx(1)

    # These are necessary, as we bypass the same check in the UniformFloatHP by design
    with pytest.raises(ValueError, match="Illegal default value 0"):
        BetaFloatHyperparameter(
            "param",
            lower=1,
            upper=10.0,
            alpha=3.0,
            beta=2.0,
            default_value=0,
            log=False,
        )
    with pytest.raises(ValueError, match="Illegal default value 0"):
        BetaFloatHyperparameter(
            "param",
            lower=1,
            upper=1000.0,
            alpha=3.0,
            beta=2.0,
            default_value=0,
            log=True,
        )


def test_betaint_dist_parameters():
    # This one should just be created without raising an error - corresponds to uniform dist.
    BetaIntegerHyperparameter("param", lower=0, upper=10.0, alpha=1, beta=1)

    # This one is not permitted as the co-domain is not finite
    with pytest.raises(ValueError):
        BetaIntegerHyperparameter("param", lower=0, upper=100, alpha=0.99, beta=0.99)
    # And these parameters do not define a proper beta distribution whatsoever
    with pytest.raises(ValueError):
        BetaIntegerHyperparameter("param", lower=0, upper=100, alpha=-0.1, beta=-0.1)

    # test parameters that do not create a legit beta distribution, one at a time
    with pytest.raises(ValueError):
        BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=-11, beta=5)
    with pytest.raises(ValueError):
        BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=5, beta=-11)

    # test parameters that do not yield a finite co-domain, one at a time
    with pytest.raises(ValueError):
        BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=0.5, beta=11)
    with pytest.raises(ValueError):
        BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=11, beta=0.5)


def test_betaint_legal_float_values():
    f1 = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.1)
    assert isinstance(f1.default_value, np.integer)
    with pytest.raises(
        TypeError,
        match="`default_value` for hyperparameter 'param' must be an integer",
    ):
        _ = BetaIntegerHyperparameter(
            "param",
            lower=-2.0,
            upper=2.0,
            alpha=3.0,
            beta=1.1,
            default_value=0.5,
        )


def test_betaint_to_uniform():
    f1 = BetaIntegerHyperparameter("param", lower=-30, upper=30, alpha=6.0, beta=2)
    f1_expected = UniformIntegerHyperparameter("param", -30, 30, default_value=20)
    f1_actual = f1.to_uniform()
    assert f1_expected == f1_actual


def test_betaint_pdf():
    c1 = BetaIntegerHyperparameter("param", alpha=3, beta=2, lower=0, upper=10)
    c2 = BetaIntegerHyperparameter(
        "logparam",
        alpha=3,
        beta=2,
        lower=1,
        upper=1000,
        log=True,
    )
    c3 = BetaIntegerHyperparameter("param", alpha=1.1, beta=10, lower=0, upper=3)

    point_1 = np.array([3])
    point_1_log = np.array([9])
    point_2 = np.array([9])
    point_2_log = np.array([570])
    point_3 = np.array([1])
    array_1 = np.array([3, 9, 11])
    array_1_log = np.array([9, 570, 1001])
    point_outside_range_1 = np.array([-1])
    point_outside_range_2 = np.array([11])
    point_outside_range_1_log = np.array([0])
    point_outside_range_2_log = np.array([1001])
    non_integer_point = np.array([5.7])
    wrong_shape_1 = np.array([[3]])
    wrong_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    wrong_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    # The quantization constant (0.4999) dictates the accuracy of the integer beta pdf
    assert c1.pdf_values(point_1)[0] == pytest.approx(0.0692999999999999, abs=1e-3)
    assert c2.pdf_values(point_1_log)[0] == pytest.approx(
        0.0008724511426701984,
        abs=1e-3,
    )
    assert c1.pdf_values(point_2)[0] == pytest.approx(0.08909999999999998, abs=1e-3)
    assert c2.pdf_values(point_2_log)[0] == pytest.approx(
        0.0008683622684160343,
        abs=1e-3,
    )
    assert c3.pdf_values(point_3)[0] == pytest.approx(0.34686070212329767, abs=1e-3)

    assert c1.pdf_values(point_outside_range_1)[0] == 0.0
    assert c1.pdf_values(point_outside_range_2)[0] == 0.0
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
        assert c2.pdf_values(point_outside_range_1_log)[0] == 0.0
    assert c2.pdf_values(point_outside_range_2_log)[0] == 0.0

    assert c1.pdf_values(non_integer_point)[0] == 0.0
    assert c2.pdf_values(non_integer_point)[0] == 0.0

    array_results = c1.pdf_values(array_1)
    array_results_log = c2.pdf_values(array_1_log)
    expected_results = np.array([0.0692999999999999, 0.08909999999999998, 0])
    expected_results_log = np.array([0.0008724511426701984, 0.0008683622684160343, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res, exp_log_res in zip(
        array_results,
        array_results_log,
        expected_results,
        expected_results_log,
    ):
        assert res == pytest.approx(exp_res, abs=1e-3)
        assert log_res == pytest.approx(exp_log_res, abs=1e-3)

    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_1)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_2)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_3)


def test_betaint__pdf():
    c1 = BetaIntegerHyperparameter("param", alpha=3, beta=2, lower=0, upper=10)
    c2 = BetaIntegerHyperparameter(
        "logparam",
        alpha=3,
        beta=2,
        lower=1,
        upper=np.round(np.exp(10)),
        log=True,
    )

    # since the logged and unlogged parameters will have different active domains
    # in the unit range, they will not evaluate identically under _pdf
    point_1 = c1.to_vector([2])
    point_2 = c1.to_vector([8])
    point_outside_range_1 = np.array([-0.01])
    point_outside_range_2 = np.array([1.01])

    point_1_log = c2.to_vector([30])
    point_2_log = c2.to_vector([900])
    point_outside_range_1_log = np.array([-0.01])

    assert c1.legal_vector(point_1)
    assert c1.pdf_vector(point_1) == pytest.approx(0.03520000000000001)
    assert c1.pdf_vector(point_2) == pytest.approx(0.1408)
    assert c1.pdf_vector(point_outside_range_1) == 0.0
    assert c1.pdf_vector(point_outside_range_2) == 0.0

    assert c2.pdf_vector(point_1_log) == pytest.approx(4.158874218496887e-05)
    assert c2.pdf_vector(point_2_log) == pytest.approx(8.061094363679364e-05)
    assert c2.pdf_vector(point_outside_range_1_log) == 0.0

    c1_array = np.concatenate([c1.to_vector([2, 8]), np.array([-0.01])])
    array_results = c1.pdf_vector(c1_array)
    expected_results = np.array([0.0352, 0.1408, 0.0])

    c2_array = np.concatenate([c2.to_vector([30, 900]), np.array([-0.01])])
    array_results_log = c2.pdf_vector(c2_array)
    expected_results_log = np.array([4.158874e-05, 8.061094e-05, 0.0])

    np.testing.assert_allclose(array_results, expected_results)
    np.testing.assert_allclose(array_results_log, expected_results_log)
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results_log.shape

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    accepted_shape_1 = np.array([[3]])
    accepted_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    accepted_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)
    c1.pdf_vector(accepted_shape_1)
    c1.pdf_vector(accepted_shape_2)
    c1.pdf_vector(accepted_shape_3)


def test_betaint_get_max_density():
    c1 = BetaIntegerHyperparameter("param", alpha=3, beta=2, lower=0, upper=10)
    c2 = BetaIntegerHyperparameter(
        "logparam",
        alpha=3,
        beta=2,
        lower=1,
        upper=1000,
        log=True,
    )
    c3 = BetaIntegerHyperparameter("param", alpha=1.1, beta=10, lower=0, upper=3)
    assert c1.get_max_density() == pytest.approx(1.767092411720511)
    assert c2.get_max_density() == pytest.approx(1.7777777777777777)
    assert c3.get_max_density() == pytest.approx(0.8696530286559991)


def test_categorical():
    # TODO test for inequality
    f1 = CategoricalHyperparameter("param", [0, 1])
    f1_ = CategoricalHyperparameter("param", [0, 1])
    assert f1 == f1_
    assert str(f1) == "param, Type: Categorical, Choices: {0, 1}, Default: 0"

    # Test attributes are accessible
    assert f1.name == "param"
    assert f1.size == 2
    assert f1.default_value == 0
    assert f1._normalized_default_value == 0
    np.testing.assert_equal(f1.probabilities, (0.5, 0.5))

    f2 = CategoricalHyperparameter("param", list(range(1000)))
    f2_ = CategoricalHyperparameter("param", list(range(1000)))
    assert f2 == f2_
    assert "param, Type: Categorical, Choices: {{{}}}, Default: 0".format(
        ", ".join(
            [str(choice) for choice in range(1000)],
        ),
    ) == str(f2)

    f3 = CategoricalHyperparameter("param", list(range(999)))
    assert f2 != f3

    f4 = CategoricalHyperparameter("param_", list(range(1000)))
    assert f2 != f4

    f5 = CategoricalHyperparameter("param", [*list(range(999)), 1001])
    assert f2 != f5

    f6 = CategoricalHyperparameter("param", ["a", "b"], default_value="b")
    f6_ = CategoricalHyperparameter("param", ["a", "b"], default_value="b")
    assert f6 == f6_
    assert str(f6) == "param, Type: Categorical, Choices: {a, b}, Default: b"

    assert f1 != f2
    assert f1 != "UniformFloat"

    # test that meta-data is stored correctly
    f_meta = CategoricalHyperparameter(
        "param",
        ["a", "b"],
        default_value="a",
        meta=dict(META_DATA),
    )
    assert f_meta.meta == META_DATA

    assert f1.size == 2
    assert f2.size == 1000
    assert f3.size == 999
    assert f4.size == 1000
    assert f5.size == 1000
    assert f6.size == 2

    fn = CategoricalHyperparameter("param", [("a", "b"), [0, 1]])
    assert fn.default_value == ("a", "b")
    assert fn.legal_value(("a", "b"))
    _ = str(fn)


def test_cat_equal():
    # Test that weights are properly normalized and compared
    c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[2, 2])
    other = CategoricalHyperparameter("param", ["a", "b"])
    assert c1 == other

    c1 = CategoricalHyperparameter("param", ["a", "b"])
    other = CategoricalHyperparameter("param", ["a", "b"], weights=[2, 2])
    assert c1 == other

    c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2])
    other = CategoricalHyperparameter("param", ["a", "b"], weights=[10, 20])
    assert c1 == other

    # These result in different default values and are therefore different
    c1 = CategoricalHyperparameter("param", ["a", "b"])
    c2 = CategoricalHyperparameter("param", ["b", "a"])
    assert c1 != c2

    # Test that the order of the hyperparameter doesn't matter if the default is given
    c1 = CategoricalHyperparameter("param", ["a", "b"], default_value="a")
    c2 = CategoricalHyperparameter("param", ["b", "a"], default_value="a")
    assert c1 == c2

    # Test that the weights are ordered correctly
    c1 = CategoricalHyperparameter(
        "param",
        ["a", "b"],
        weights=[1, 2],
        default_value="a",
    )
    c2 = CategoricalHyperparameter(
        "param",
        ["b", "a"],
        weights=[2, 1],
        default_value="a",
    )
    assert c1 == c2

    c1 = CategoricalHyperparameter(
        "param",
        ["a", "b"],
        weights=[1, 2],
        default_value="a",
    )
    c2 = CategoricalHyperparameter(
        "param",
        ["b", "a"],
        weights=[1, 2],
        default_value="a",
    )
    assert c1 != c2

    c1 = CategoricalHyperparameter(
        "param",
        ["a", "b"],
        weights=[1, 2],
        default_value="a",
    )
    c2 = CategoricalHyperparameter("param", ["b", "a"], default_value="a")
    assert c1 != c2

    c1 = CategoricalHyperparameter("param", ["a", "b"], default_value="a")
    c2 = CategoricalHyperparameter(
        "param",
        ["b", "a"],
        weights=[1, 2],
        default_value="a",
    )
    assert c1 != c2

    # Test that the equals operator does not fail accessing the weight of choice "a" in c2
    c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2])
    c2 = CategoricalHyperparameter("param", ["b", "c"], weights=[1, 2])
    assert c1 != c2


def test_categorical_strings():
    f1 = CategoricalHyperparameter("param", ["a", "b"])
    f1_ = CategoricalHyperparameter("param", ["a", "b"])
    assert f1 == f1_
    assert str(f1) == "param, Type: Categorical, Choices: {a, b}, Default: a"


def test_categorical_is_legal():
    f1 = CategoricalHyperparameter("param", ["a", "b"])
    assert f1.legal_value("a")
    assert f1.legal_value("a")
    assert not f1.legal_value("c")
    assert not f1.legal_value(3)

    # Test is legal vector
    assert f1.legal_vector(1.0)
    assert f1.legal_vector(0.0)
    assert f1.legal_vector(0)
    assert not f1.legal_vector(0.3)
    assert not f1.legal_vector(-0.1)
    assert not f1.legal_vector("Hahaha")  # type: ignore


def test_categorical_choices():
    with pytest.raises(
        ValueError,
        match="Choices for categorical hyperparameters param contain choice `a` 2 times, "
        "while only a single occurence is allowed.",
    ):
        CategoricalHyperparameter("param", ["a", "a"])


def test_categorical_default():
    # Test that the default value is the most probable choice when weights are given
    f1 = CategoricalHyperparameter("param", ["a", "b"])
    f2 = CategoricalHyperparameter("param", ["a", "b"], weights=[0.3, 0.6])
    f3 = CategoricalHyperparameter("param", ["a", "b"], weights=[0.6, 0.3])
    assert f1.default_value != f2.default_value
    assert f1.default_value == f3.default_value


def test_sample_UniformFloatHyperparameter():
    # NOTE: Had to reconfigure this test to actually be fast

    # Uniform
    hp = UniformFloatHyperparameter("ufhp", 0.5, 2.5)
    rs = np.random.RandomState(1)
    samples = hp.sample_value(size=10_0000, seed=rs)
    counts_per_bin, _ = np.histogram(
        samples,
        bins=20,
        range=(float(hp.lower), float(hp.upper)),
    )

    # The 21st bin is only filled if exactly 2.5 is sampled...very rare...
    for bin in counts_per_bin:
        assert 5200 > bin > 4800, counts_per_bin

    rs1 = np.random.RandomState(1)
    rs2 = np.random.RandomState(1)
    np.testing.assert_array_equal(
        hp.sample_value(100, seed=rs1),
        hp.sample_value(100, seed=rs2),
    )

    # Log Uniform
    hp = UniformFloatHyperparameter("ufhp", 1.0, np.e**2, log=True)

    rs = np.random.RandomState(1)
    samples = hp.sample_value(size=10_0000, seed=rs)
    counts_per_bin, _ = np.histogram(
        samples,
        bins=20,
        range=(float(hp.lower), float(hp.upper)),
    )

    np.testing.assert_array_equal(
        counts_per_bin,
        [
            14012,
            10977,
            8809,
            7559,
            6424,
            5706,
            5276,
            4694,
            4328,
            3928,
            3655,
            3386,
            3253,
            2932,
            2816,
            2727,
            2530,
            2479,
            2280,
            2229,
        ],
    )


def test_categorical_pdf():
    c1 = CategoricalHyperparameter(
        "x1",
        choices=["one", "two", "three"],
        weights=[2, 1, 2],
    )
    c2 = CategoricalHyperparameter(
        "x1",
        choices=["one", "two", "three"],
        weights=[5, 0, 2],
    )
    c3 = CategoricalHyperparameter("x1", choices=["one", "two", "three", "four"])

    point_1 = np.array(["one"])
    point_2 = np.array(["two"])

    wrong_shape_1 = np.array([["one"]])
    wrong_shape_2 = np.array(["one", "two"]).reshape(1, -1)
    wrong_shape_3 = np.array(["one", "two"]).reshape(-1, 1)

    assert c1.pdf_values(point_1)[0] == 0.4
    assert c1.pdf_values(point_2)[0] == 0.2
    assert c2.pdf_values(point_1)[0] == pytest.approx(0.7142857142857143)
    assert c2.pdf_values(point_2)[0] == 0.0
    assert c3.pdf_values(point_1)[0] == 0.25

    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_1)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_2)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_3)


def test_categorical__pdf():
    c1 = CategoricalHyperparameter(
        "x1",
        choices=["one", "two", "three"],
        weights=[2, 1, 2],
    )
    c2 = CategoricalHyperparameter(
        "x1",
        choices=["one", "two", "three"],
        weights=[5, 0, 2],
    )

    point_1 = np.array([0])
    point_2 = np.array([1])
    array_1 = np.array([1, 0, 2])
    nan = np.array([0, np.nan])
    assert c1.pdf_vector(point_1)[0] == 0.4
    assert c1.pdf_vector(point_2)[0] == 0.2
    assert c2.pdf_vector(point_1)[0] == pytest.approx(0.7142857142857143)
    assert c2.pdf_vector(point_2)[0] == 0.0

    array_results = c1.pdf_vector(array_1)
    expected_results = np.array([0.2, 0.4, 0.4])
    assert array_results.shape == expected_results.shape
    for res, exp_res in zip(array_results, expected_results):
        assert res == exp_res

    nan_results = c1.pdf_vector(nan)
    expected_results = np.array([0.4, 0])
    assert nan_results.shape == expected_results.shape
    for res, exp_res in zip(nan_results, expected_results):
        assert res == exp_res


def test_categorical_get_max_density():
    c1 = CategoricalHyperparameter(
        "x1",
        choices=["one", "two", "three"],
        weights=[2, 1, 2],
    )
    c2 = CategoricalHyperparameter(
        "x1",
        choices=["one", "two", "three"],
        weights=[5, 0, 2],
    )
    c3 = CategoricalHyperparameter("x1", choices=["one", "two", "three"])
    assert c1.get_max_density() == 0.4
    assert c2.get_max_density() == 0.7142857142857143
    assert c3.get_max_density() == pytest.approx(0.33333333333333)


def test_sample_NormalFloatHyperparameter_with_bounds():
    hp = NormalFloatHyperparameter("nfhp", 0, 1, lower=-3, upper=3)

    rs1 = np.random.RandomState(1)
    rs2 = np.random.RandomState(1)

    samples1 = hp.sample_value(1000, seed=rs1)
    samples2 = hp.sample_value(1000, seed=rs2)

    np.testing.assert_array_equal(samples1, samples2)

    bin_counts, _ = np.histogram(samples1, bins=11, range=(-3, 3))

    # Approximatly normally distributed and approximately symmetric
    np.testing.assert_array_equal(
        bin_counts,
        [7, 30, 54, 116, 174, 227, 186, 122, 54, 27, 3],
    )


def test_sample_BetaFloatHyperparameter():
    hp = BetaFloatHyperparameter("bfhp", alpha=8, beta=1.5, lower=-1, upper=10)

    rs1 = np.random.RandomState(1)
    rs2 = np.random.RandomState(1)

    samples1 = hp.sample_value(1000, seed=rs1)
    samples2 = hp.sample_value(1000, seed=rs2)

    np.testing.assert_array_equal(samples1, samples2)

    bin_counts, _ = np.histogram(samples1, bins=12, range=(-1, 10))
    np.testing.assert_array_equal(
        bin_counts,
        [0, 0, 1, 2, 4, 5, 20, 50, 121, 197, 284, 316],
    )


def test_sample_UniformIntegerHyperparameter():
    hp = UniformIntegerHyperparameter("uihp", 1, 10)

    rs1 = np.random.RandomState(1)
    rs2 = np.random.RandomState(1)

    samples1 = hp.sample_value(1000, seed=rs1)
    samples2 = hp.sample_value(1000, seed=rs2)

    np.testing.assert_array_equal(samples1, samples2)

    bin_counts, _ = np.histogram(samples1, bins=10, range=(1, 10))
    for bin in bin_counts:
        assert 85 < bin < 115


def test_sample_BetaIntegerHyperparameter():
    hp = BetaIntegerHyperparameter("bihp", alpha=4, beta=4, lower=0, upper=10)

    rs1 = np.random.RandomState(1)
    rs2 = np.random.RandomState(1)

    samples1 = hp.sample_value(1000, seed=rs1)
    samples2 = hp.sample_value(1000, seed=rs2)

    np.testing.assert_array_equal(samples1, samples2)

    bin_counts, _ = np.histogram(samples1, bins=11, range=(0, 10))
    np.testing.assert_array_equal(
        bin_counts,
        [1, 23, 82, 121, 174, 197, 174, 115, 86, 27, 0],
    )


def test_sample_CategoricalHyperparameter():
    hp = CategoricalHyperparameter("chp", ["0", "2", "Bla", "Blub"])

    rs1 = np.random.RandomState(1)
    rs2 = np.random.RandomState(1)

    samples1 = hp.sample_value(1000, seed=rs1)
    samples2 = hp.sample_value(1000, seed=rs2)

    np.testing.assert_array_equal(samples1, samples2)

    _, bin_counts = np.unique(samples1, return_counts=True)
    for bin in bin_counts:
        assert 235 <= bin <= 265


def test_sample_CategoricalHyperparameter_with_weights():
    # check also that normalization works
    hp = CategoricalHyperparameter(
        "chp",
        [0, 2, "Bla", "Blub", "Blurp"],
        weights=[1, 2, 3, 4, 0],
    )
    np.testing.assert_almost_equal(
        actual=hp.probabilities,
        desired=[0.1, 0.2, 0.3, 0.4, 0],
        decimal=3,
    )

    def actual_test():
        rs = np.random.RandomState(1)
        counts_per_bin: dict[str | int, int] = defaultdict(int)
        for _ in range(10000):
            value = hp.sample_value(seed=rs)
            counts_per_bin[value] += 1

        assert {0: 1003, 2: 2061, "Bla": 2994, "Blub": 3942} == dict(
            counts_per_bin.items(),
        )
        return counts_per_bin

    assert actual_test() == actual_test()


def test_categorical_copy_with_weights():
    orig_hp = CategoricalHyperparameter(
        name="param",
        choices=[1, 2, 3],
        default_value=2,
        weights=[1, 3, 6],
    )
    copy_hp = copy.copy(orig_hp)

    assert copy_hp.name == orig_hp.name
    assert copy_hp.choices == orig_hp.choices
    assert copy_hp.default_value == orig_hp.default_value
    assert copy_hp.size == orig_hp.size
    np.testing.assert_equal(copy_hp.probabilities, orig_hp.probabilities)


def test_categorical_copy_without_weights():
    orig_hp = CategoricalHyperparameter(
        name="param",
        choices=[1, 2, 3],
        default_value=2,
    )
    copy_hp = copy.copy(orig_hp)

    assert copy_hp.name == orig_hp.name
    assert copy_hp.choices == orig_hp.choices
    assert copy_hp.default_value == orig_hp.default_value
    assert copy_hp.size == orig_hp.size
    np.testing.assert_equal(
        copy_hp.probabilities,
        (0.3333333333333333, 0.3333333333333333, 0.3333333333333333),
    )
    np.testing.assert_equal(
        orig_hp.probabilities,
        (0.3333333333333333, 0.3333333333333333, 0.3333333333333333),
    )


def test_categorical_with_weights():
    rs = np.random.RandomState()

    cat_hp_str = CategoricalHyperparameter(
        name="param",
        choices=["A", "B", "C"],
        default_value="A",
        weights=[0.1, 0.6, 0.3],
    )
    assert sorted(np.unique(cat_hp_str.sample_value(1000, seed=rs))) == ["A", "B", "C"]

    cat_hp_int = CategoricalHyperparameter(
        name="param",
        choices=[1, 2, 3],
        default_value=2,
        weights=[0.1, 0.3, 0.6],
    )
    assert sorted(np.unique(cat_hp_int.sample_value(1000, seed=rs))) == [1, 2, 3]

    cat_hp_float = CategoricalHyperparameter(
        name="param",
        choices=[-0.1, 0.0, 0.3],
        default_value=0.3,
        weights=[10, 60, 30],
    )
    assert sorted(np.unique(cat_hp_float.sample_value(1000, seed=rs))) == [
        -0.1,
        0.0,
        0.3,
    ]


def test_categorical_with_some_zero_weights():
    # zero weights are okay as long as there is at least one strictly positive weight

    rs = np.random.RandomState()

    cat_hp_str = CategoricalHyperparameter(
        name="param",
        choices=["A", "B", "C"],
        default_value="A",
        weights=[0.1, 0.0, 0.3],
    )
    assert sorted(np.unique(cat_hp_str.sample_value(1000, seed=rs))) == ["A", "C"]

    np.testing.assert_almost_equal(
        actual=cat_hp_str.probabilities,
        desired=[0.25, 0.0, 0.75],
        decimal=3,
    )

    cat_hp_int = CategoricalHyperparameter(
        name="param",
        choices=[1, 2, 3],
        default_value=2,
        weights=[0.1, 0.6, 0.0],
    )
    assert sorted(np.unique(cat_hp_int.sample_value(1000, seed=rs))) == [1, 2]

    np.testing.assert_almost_equal(
        actual=cat_hp_int.probabilities,
        desired=[0.1429, 0.8571, 0.0],
        decimal=3,
    )

    cat_hp_float = CategoricalHyperparameter(
        name="param",
        choices=[-0.1, 0.0, 0.3],
        default_value=0.3,
        weights=[0.0, 0.6, 0.3],
    )
    assert sorted(np.unique(cat_hp_float.sample_value(1000, seed=rs))) == [0.0, 0.3]

    np.testing.assert_almost_equal(
        actual=cat_hp_float.probabilities,
        desired=[0.00, 0.6667, 0.3333],
        decimal=3,
    )


def test_categorical_with_all_zero_weights():
    with pytest.raises(
        ValueError,
        match=r"All weights are zero, at least one weight has to be strictly positive.",
    ):
        CategoricalHyperparameter(
            name="param",
            choices=["A", "B", "C"],
            default_value="A",
            weights=[0.0, 0.0, 0.0],
        )


def test_categorical_with_wrong_length_weights():
    with pytest.raises(
        ValueError,
        match="The list of weights and the list of choices are required to be of same length.",
    ):
        CategoricalHyperparameter(
            name="param",
            choices=["A", "B", "C"],
            default_value="A",
            weights=[0.1, 0.3],
        )

    with pytest.raises(
        ValueError,
        match="The list of weights and the list of choices are required to be of same length.",
    ):
        CategoricalHyperparameter(
            name="param",
            choices=["A", "B", "C"],
            default_value="A",
            weights=[0.1, 0.0, 0.5, 0.3],
        )


def test_categorical_with_negative_weights():
    with pytest.raises(ValueError, match="Negative weights are not allowed."):
        CategoricalHyperparameter(
            name="param",
            choices=["A", "B", "C"],
            default_value="A",
            weights=[0.1, -0.1, 0.3],
        )


def test_categorical_with_set():
    with pytest.raises(TypeError, match="Using a set of choices is prohibited."):
        CategoricalHyperparameter(
            name="param",
            choices={"A", "B", "C"},  # type: ignore
            default_value="A",
        )

    with pytest.raises(TypeError, match="Using a set of weights is prohibited."):
        CategoricalHyperparameter(
            name="param",
            choices=["A", "B", "C"],
            default_value="A",
            weights={0.2, 0.6, 0.8},  # type: ignore
        )


def test_log_space_conversion():
    lower, upper = 1e-5, 1e5
    hyper = UniformFloatHyperparameter("test", lower=lower, upper=upper, log=True)
    assert hyper.legal_value(hyper.to_value(1.0))

    lower, upper = 1e-10, 1e10
    hyper = UniformFloatHyperparameter("test", lower=lower, upper=upper, log=True)
    assert hyper.legal_value(hyper.to_value(1.0))


def test_ordinal_attributes_accessible():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.name == "temp"
    assert f1.sequence == ("freezing", "cold", "warm", "hot")
    assert f1.size == 4
    assert f1.default_value == "freezing"
    assert f1._normalized_default_value == 0


def test_ordinal_is_legal():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.legal_value("warm")
    assert f1.legal_value("freezing")
    assert not f1.legal_value("chill")
    assert not f1.legal_value(2.5)
    assert not f1.legal_value("3")

    # Test is legal vector
    assert f1.legal_vector(1.0)
    assert f1.legal_vector(0.0)
    assert f1.legal_vector(0)
    assert f1.legal_vector(3)
    assert not f1.legal_vector(-0.1)
    assert not f1.legal_vector("Hahaha")  # type: ignore


def test_ordinal_nested_lists_prints_correctly():
    f1 = OrdinalHyperparameter(
        "temp",
        [["freezing", "cold", "warm", "hot"], ["a", "b"]],
    )
    _ = str(f1)


def test_ordinal_check_order():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.check_order("freezing", "cold")
    assert f1.check_order("freezing", "hot")
    assert not f1.check_order("hot", "cold")
    assert not f1.check_order("hot", "warm")


def test_ordinal_get_value():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.get_value(3) == "hot"
    assert f1.get_value(1) != "warm"


def test_ordinal_get_order():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.get_order("warm") == 2
    assert f1.get_order("freezing") != 3


def test_ordinal_get_seq_order():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert tuple(f1.get_seq_order()) == (0, 1, 2, 3)


def test_ordinal_get_neighbors():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])

    np.testing.assert_array_equal(f1.neighbors_vectorized(0, n=4, seed=None), [1])
    np.testing.assert_array_equal(f1.neighbors_vectorized(1, n=4, seed=None), [0, 2])
    np.testing.assert_array_equal(f1.neighbors_vectorized(3, n=4, seed=None), [2])

    np.testing.assert_array_equal(f1.neighbors_values("hot", n=1), ["warm"])
    np.testing.assert_array_equal(f1.neighbors_values("freezing", n=1), ["cold"])
    np.testing.assert_array_equal(
        f1.neighbors_values("cold", n=2),
        ["freezing", "warm"],
    )
    np.testing.assert_array_equal(
        f1.neighbors_values("warm", n=2),
        ["cold", "hot"],
    )


def test_get_num_neighbors():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.get_num_neighbors("freezing") == 1
    assert f1.get_num_neighbors("hot") == 1
    assert f1.get_num_neighbors("cold") == 2
    assert f1.get_num_neighbors("warm") == 2


def test_ordinal_get_size():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.size == 4


def test_ordinal_pdf():
    c1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    point_1 = np.array(["freezing"])
    point_2 = np.array(["warm"])
    array_1 = np.array(["freezing", "warm"])

    wrong_shape_1 = np.array([["freezing"]])
    wrong_shape_2 = np.array(["freezing", "warm"]).reshape(1, -1)
    wrong_shape_3 = np.array(["freezing", "warm"]).reshape(-1, 1)

    assert c1.pdf_values(point_1)[0] == 0.25
    assert c1.pdf_values(point_2)[0] == 0.25

    array_results = c1.pdf_values(array_1)
    expected_results = np.array([0.25, 0.25])
    assert array_results.shape == expected_results.shape
    for res, exp_res in zip(array_results, expected_results):
        assert res == exp_res

    with pytest.raises(KeyError):
        c1.pdf_values(np.array(["zero"]))

    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_1)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_2)
    with pytest.raises(
        ValueError,
        match="Method pdf expects a one-dimensional numpy array",
    ):
        c1.pdf_values(wrong_shape_3)


def test_ordinal_get_max_density():
    c1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    c2 = OrdinalHyperparameter("temp", ["freezing", "cold"])
    assert c1.get_max_density() == 0.25
    assert c2.get_max_density() == 0.5


def test_rvs():
    f1 = UniformFloatHyperparameter("param", 0, 10)

    # test that returned types are correct
    # if size=None, return a value, but if size=1, return a 1-element array
    assert isinstance(f1.rvs(), float)
    assert isinstance(f1.rvs(size=1), np.ndarray)
    assert isinstance(f1.rvs(size=2), np.ndarray)

    assert f1.rvs(random_state=100) == pytest.approx(f1.rvs(random_state=100))
    assert f1.rvs(random_state=100) == pytest.approx(
        f1.rvs(random_state=np.random.RandomState(100)),
    )
    f1.rvs(random_state=np.random.default_rng(1))


def test_hyperparam_representation():
    # Float
    f1 = UniformFloatHyperparameter("param", 1, 100, log=True)
    assert (
        str(f1)
        == "param, Type: UniformFloat, Range: [1.0, 100.0], Default: 10.0, on log-scale"
    )
    f3 = NormalFloatHyperparameter("param", 8, 99.1, log=False, lower=1, upper=16)
    assert (
        str(f3)
        == "param, Type: NormalFloat, Mu: 8.0, Sigma: 99.1, Range: [1.0, 16.0], Default: 8.0"
    )
    i1 = UniformIntegerHyperparameter("param", 0, 100)
    assert str(i1) == "param, Type: UniformInteger, Range: [0, 100], Default: 50"
    i3 = NormalIntegerHyperparameter("param", 5, 8, lower=1, upper=10)
    assert (
        str(i3)
        == "param, Type: NormalInteger, Mu: 5.0, Sigma: 8.0, Range: [1, 10], Default: 5"
    )
    o1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert (
        str(o1)
        == "temp, Type: Ordinal, Sequence: {freezing, cold, warm, hot}, Default: freezing"
    )
    c1 = CategoricalHyperparameter("param", [True, False])
    assert str(c1) == "param, Type: Categorical, Choices: {True, False}, Default: True"


@pytest.mark.parametrize(
    "hp, i",
    [
        (
            CategoricalHyperparameter("param", [True, False, None], default_value=None),
            2,
        ),
        (
            CategoricalHyperparameter("param", ["a", "b", None], default_value=None),
            2,
        ),
        (CategoricalHyperparameter("param", [None]), 0),
        (CategoricalHyperparameter("param", [None, 1, 2]), 0),
        (
            OrdinalHyperparameter(
                "param",
                [1, None, 2],
                default_value=None,
            ),  # Essential None is in the middle for tests
            1,
        ),
        (OrdinalHyperparameter("param", [None]), 0),
    ],
)
def test_none_allowed_in_categorical_ordinal(
    hp: CategoricalHyperparameter | OrdinalHyperparameter,
    i: int,
    tmp_path: Path,
) -> None:
    assert hp.legal_value(None)
    assert hp.to_value(np.float64(i)) is None
    assert hp.to_vector(None) == i

    if hp.size != 1:
        seq = hp.choices if isinstance(hp, CategoricalHyperparameter) else hp.sequence
        first_non_none = next(x for x in seq if x is not None)
        assert None in hp.neighbors_values(first_non_none, n=hp.size)
        assert 0 < hp.get_num_neighbors(None) < hp.size

    assert hp.pdf_values([None])[0] > 0

    space = ConfigurationSpace({"c": hp})

    _path = tmp_path / "space.json"
    with _path.open("w") as f:
        space.to_json(f)

    with _path.open("r") as f:
        loaded_space = ConfigurationSpace.from_json(f)

    assert space == loaded_space

    default_config = space.get_default_configuration()
    assert dict(default_config) == {"param": None}

    assert default_config._vector[0] == i

    assert None in default_config.values()
    default_config["param"] = None  # no raise
    assert default_config == default_config  # noqa: PLR0124
    default_config.check_valid_configuration()  # no raise

    _ = list(get_one_exchange_neighbourhood(default_config, seed=1))  # no raise


@dataclass
class _DummyClass:
    x: int


@pytest.mark.parametrize(
    "hp",
    [
        (
            CategoricalHyperparameter(
                "param",
                [{"hello": "world"}, _DummyClass(4), (1, 2), None],
                default_value=_DummyClass(4),
            )
        ),
        (
            OrdinalHyperparameter(
                "param",
                [(0, 0), (10, 20), _DummyClass(10), (30, 40), None],
            )
        ),
    ],
)
def test_arbitrary_object_allowed_in_categorical_ordinal(
    hp: CategoricalHyperparameter | OrdinalHyperparameter,
    tmp_path: Path,
) -> None:
    assert hp == hp  # noqa: PLR0124

    _seq = hp.choices if isinstance(hp, CategoricalHyperparameter) else hp.sequence
    for s in _seq:
        assert hp.legal_value(s)

        vector_value = hp.to_vector(s)
        assert hp.lower_vectorized <= vector_value <= hp.upper_vectorized

        value_value = hp.to_value(vector_value)
        assert value_value == s

        if isinstance(hp, CategoricalHyperparameter):
            neighbors = hp.neighbors_values(s, n=hp.size)
            for other in _seq:
                if other is s:
                    continue
                assert other in neighbors.tolist()

        assert 0 < hp.get_num_neighbors(s) < hp.size

        assert hp.pdf_values([s])[0] > 0

    space = ConfigurationSpace({"c": hp})
    assert space == space  # noqa: PLR0124

    with pytest.raises(TypeError):
        _path = tmp_path / "space.json"
        with _path.open("w") as f:
            space.to_json(f)

    default_config = space.get_default_configuration()
    default_config.check_valid_configuration()  # no raise

    for s in _seq:
        u = UniformFloatHyperparameter("u", 1, 10)
        space_with_cond = ConfigurationSpace()
        space_with_cond.add(
            hp,
            u,
            AndConjunction(
                EqualsCondition(child=u, parent=hp, value=s),
                InCondition(child=u, parent=hp, values=[s]),
            ),
        )
        samples = space_with_cond.sample_configuration(10)
        for sample in samples:
            ns = list(get_one_exchange_neighbourhood(sample, seed=1))  # no raise
            for n in ns:
                n.check_valid_configuration()  # no raise

    for s in _seq:
        # We can't put a forbidden on the default value unfortunatly...
        if s == hp.default_value:
            continue

        u = UniformFloatHyperparameter("u", 1, 10)
        space_with_forb = ConfigurationSpace()
        space_with_forb.add(
            hp,
            u,
            ForbiddenEqualsClause(hp, s),
            ForbiddenInClause(hp, [s]),
        )
        samples = space_with_cond.sample_configuration(10)
        for sample in samples:
            list(get_one_exchange_neighbourhood(sample, seed=1))  # no raise
            for n in ns:
                n.check_valid_configuration()  # no raise
