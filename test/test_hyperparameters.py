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
from typing import Any

import numpy as np
import pytest

from ConfigSpace.functional import arange_chunked
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

META_DATA = {"additional": "meta-data", "useful": "for integrations", "input_id": 42}


def test_constant():
    # Test construction
    c1 = Constant("value", 1)
    c2 = Constant("value", 1)
    c3 = Constant("value", 2)
    c4 = Constant("valuee", 1)
    c5 = Constant("valueee", 2)

    # Test attributes are accessible
    assert c5.name == "valueee"
    assert c5.value == 2

    # Test the representation
    assert c1.__repr__() == "value, Type: Constant, Value: 1"

    # Test the equals operator (and the ne operator in the last line)
    assert c1 != 1
    assert c1 == c2
    assert c1 != c3
    assert c1 != c4
    assert c1 != c5

    # Test that only string, integers and floats are allowed
    v: Any
    for v in [{}, None, True]:
        with pytest.raises(TypeError):
            Constant("value", v)

    # Test that only string names are allowed
    for name in [1, {}, None, True]:
        with pytest.raises(TypeError):
            Constant(name, "value")

    # test that meta-data is stored correctly
    c1_meta = Constant("value", 1, dict(META_DATA))
    assert c1_meta.meta == META_DATA

    # Test getting the size
    for constant in (c1, c2, c3, c4, c5, c1_meta):
        assert constant.get_size() == 1


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

    assert c1.pdf(point_1) == np.array([1.0])
    assert c2.pdf(point_2) == np.array([1.0])
    assert c1.pdf(point_2) == np.array([0.0])
    assert c2.pdf(point_1) == np.array([0.0])

    assert tuple(c1.pdf(array_1)) == tuple(np.array([1.0, 1.0]))
    assert tuple(c2.pdf(array_2)) == tuple(np.array([1.0, 1.0]))
    assert tuple(c1.pdf(array_2)) == tuple(np.array([0.0, 0.0]))
    assert tuple(c1.pdf(array_3)) == tuple(np.array([1.0, 0.0]))

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    # and it must be one-dimensional
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_1)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_2)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_3)


def test_constant__pdf():
    c1 = Constant("valuee", 1)
    c2 = Constant("valueee", -2)

    point_1 = np.array([1])
    point_2 = np.array([-2])
    array_1 = np.array([1, 1])
    array_2 = np.array([-2, -2])
    array_3 = np.array([1, -2])

    # These shapes are allowed in _pdf
    accepted_shape_1 = np.array([[1]])
    accepted_shape_2 = np.array([1, 2, 3]).reshape(1, -1)
    accepted_shape_3 = np.array([3, 2, 1]).reshape(-1, 1)

    assert c1._pdf(point_1) == np.array([1.0])
    assert c2._pdf(point_2) == np.array([1.0])
    assert c1._pdf(point_2) == np.array([0.0])
    assert c2._pdf(point_1) == np.array([0.0])

    # Only (N, ) numpy arrays are seamlessly converted to tuples
    # so the __eq__ method works as intended
    assert tuple(c1._pdf(array_1)) == tuple(np.array([1.0, 1.0]))
    assert tuple(c2._pdf(array_2)) == tuple(np.array([1.0, 1.0]))
    assert tuple(c1._pdf(array_2)) == tuple(np.array([0.0, 0.0]))
    assert tuple(c1._pdf(array_3)) == tuple(np.array([1.0, 0.0]))

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1._pdf(0.2)
    with pytest.raises(TypeError):
        c1._pdf("pdf")

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1._pdf(accepted_shape_1)
    c1._pdf(accepted_shape_2)
    c1._pdf(accepted_shape_3)


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
    assert f1.q is None
    assert f1.log is False
    assert f1.default_value == pytest.approx(5.0)
    assert f1.normalized_default_value == pytest.approx(0.5)

    f2 = UniformFloatHyperparameter("param", 0, 10, q=0.1)
    f2_ = UniformFloatHyperparameter("param", 0, 10, q=0.1)
    assert f2 == f2_
    assert str(f2) == "param, Type: UniformFloat, Range: [0.0, 10.0], Default: 5.0, Q: 0.1"

    f3 = UniformFloatHyperparameter("param", 0.00001, 10, log=True)
    f3_ = UniformFloatHyperparameter("param", 0.00001, 10, log=True)
    assert f3 == f3_
    assert str(f3) == "param, Type: UniformFloat, Range: [1e-05, 10.0], Default: 0.01, on log-scale"

    f4 = UniformFloatHyperparameter("param", 0, 10, default_value=1.0)
    f4_ = UniformFloatHyperparameter("param", 0, 10, default_value=1.0)
    # Test that a int default is converted to float
    f4__ = UniformFloatHyperparameter("param", 0, 10, default_value=1)
    assert f4 == f4_
    assert isinstance(f4.default_value, type(f4__.default_value))
    assert str(f4) == "param, Type: UniformFloat, Range: [0.0, 10.0], Default: 1.0"

    f5 = UniformFloatHyperparameter("param", 0.1, 10, q=0.1, log=True, default_value=1.0)
    f5_ = UniformFloatHyperparameter("param", 0.1, 10, q=0.1, log=True, default_value=1.0)
    assert f5 == f5_
    assert (
        str(f5)
        == "param, Type: UniformFloat, Range: [0.1, 10.0], Default: 1.0, on log-scale, Q: 0.1"
    )

    assert f1 != f2
    assert f1 != "UniformFloat"

    # test that meta-data is stored correctly
    f_meta = UniformFloatHyperparameter(
        "param",
        0.1,
        10,
        q=0.1,
        log=True,
        default_value=1.0,
        meta=dict(META_DATA),
    )
    assert f_meta.meta == META_DATA

    # Test get_size
    for float_hp in (f1, f3, f4):
        assert np.isinf(float_hp.get_size())
    assert f2.get_size() == 101
    assert f5.get_size() == 100


def test_uniformfloat_to_integer():
    f1 = UniformFloatHyperparameter("param", 1, 10, q=0.1, log=True)
    with pytest.warns(
        UserWarning,
        match="Setting quantization < 1 for Integer " "Hyperparameter 'param' has no effect",
    ):
        f2 = f1.to_integer()
    # TODO is this a useful rounding?
    # TODO should there be any rounding, if e.g. lower=0.1
    assert str(f2) == "param, Type: UniformInteger, Range: [1, 10], Default: 3, on log-scale"


def test_uniformfloat_is_legal():
    lower = 0.1
    upper = 10
    f1 = UniformFloatHyperparameter("param", lower, upper, q=0.1, log=True)

    assert f1.is_legal(3.0)
    assert f1.is_legal(3)
    assert not f1.is_legal(-0.1)
    assert not f1.is_legal(10.1)
    assert not f1.is_legal("AAA")
    assert not f1.is_legal({})

    # Test legal vector values
    assert f1.is_legal_vector(1.0)
    assert f1.is_legal_vector(0.0)
    assert f1.is_legal_vector(0)
    assert f1.is_legal_vector(0.3)
    assert not f1.is_legal_vector(-0.1)
    assert not f1.is_legal_vector(1.1)
    with pytest.raises(TypeError):
        f1.is_legal_vector("Hahaha")


def test_uniformfloat_illegal_bounds():
    with pytest.raises(
        ValueError,
        match=r"Negative lower bound \(0.000000\) for log-scale hyperparameter " r"param is forbidden.",
    ):
        _ = UniformFloatHyperparameter("param", 0, 10, q=0.1, log=True)

    with pytest.raises(
        ValueError,
        match="Upper bound 0.000000 must be larger than lower bound " "1.000000 for hyperparameter param",
    ):
        _ = UniformFloatHyperparameter("param", 1, 0)


def test_uniformfloat_pdf():
    c1 = UniformFloatHyperparameter("param", lower=0, upper=10)
    c2 = UniformFloatHyperparameter("logparam", lower=np.exp(0), upper=np.exp(10), log=True)
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

    assert c1.pdf(point_1)[0] == pytest.approx(0.1)
    assert c2.pdf(point_2)[0] == pytest.approx(4.539992976248485e-05)
    assert c1.pdf(point_1)[0] == pytest.approx(0.1)
    assert c2.pdf(point_2)[0] == pytest.approx(4.539992976248485e-05)
    assert c3.pdf(point_3)[0] == pytest.approx(2.0)

    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    # since inverse_transform pulls everything into range,
    # even points outside get evaluated in range
    assert c1.pdf(point_outside_range)[0] == pytest.approx(0.1)
    assert c2.pdf(point_outside_range_log)[0] == pytest.approx(4.539992976248485e-05)

    # this, however, is a negative value on a log param, which cannot be pulled into range
    with pytest.warns(RuntimeWarning, match="invalid value encountered in log"):
        assert c2.pdf(point_outside_range)[0] == 0.0

    array_results = c1.pdf(array_1)
    array_results_log = c2.pdf(array_1)
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
        assert log_res == pytest.approx(exp_log_res)

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_1)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_2)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_3)


def test_uniformfloat__pdf():
    c1 = UniformFloatHyperparameter("param", lower=0, upper=10)
    c2 = UniformFloatHyperparameter("logparam", lower=np.exp(0), upper=np.exp(10), log=True)
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

    assert c1._pdf(point_1)[0] == pytest.approx(0.1)
    assert c2._pdf(point_2)[0] == pytest.approx(4.539992976248485e-05)
    assert c1._pdf(point_1)[0] == pytest.approx(0.1)
    assert c2._pdf(point_2)[0] == pytest.approx(4.539992976248485e-05)
    assert c3._pdf(point_3)[0] == pytest.approx(2.0)

    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    # since inverse_transform pulls everything into range,
    # even points outside get evaluated in range
    assert c1._pdf(point_outside_range_1)[0] == pytest.approx(0.0)
    assert c2._pdf(point_outside_range_2)[0] == pytest.approx(0.0)
    assert c1._pdf(point_outside_range_2)[0] == pytest.approx(0.0)
    assert c2._pdf(point_outside_range_1)[0] == pytest.approx(0.0)

    array_results = c1._pdf(array_1)
    array_results_log = c2._pdf(array_1)
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
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_log_res)

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1._pdf(0.2)
    with pytest.raises(TypeError):
        c1._pdf("pdf")

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1._pdf(accepted_shape_1)
    c1._pdf(accepted_shape_2)
    c1._pdf(accepted_shape_3)


def test_uniformfloat_get_max_density():
    c1 = UniformFloatHyperparameter("param", lower=0, upper=10)
    c2 = UniformFloatHyperparameter("logparam", lower=np.exp(0), upper=np.exp(10), log=True)
    c3 = UniformFloatHyperparameter("param", lower=0, upper=0.5)
    assert c1.get_max_density() == 0.1
    assert c2.get_max_density() == pytest.approx(4.539992976248485e-05)
    assert c3.get_max_density() == 2


def test_normalfloat():
    # TODO test non-equality
    f1 = NormalFloatHyperparameter("param", 0.5, 10.5)
    f1_ = NormalFloatHyperparameter("param", 0.5, 10.5)
    assert f1 == f1_
    assert str(f1) == "param, Type: NormalFloat, Mu: 0.5 Sigma: 10.5, Default: 0.5"

    # Due to seemingly different numbers with x86_64 and i686 architectures
    # we got these numbers, where last two are slightly different
    #   5.715498606617943, -0.9517751622974389,
    #   7.3007296500572725, 16.49181349228427
    # They are equal up to 14 decimal places
    expected = [5.715498606617943, -0.9517751622974389, 7.300729650057271, 16.491813492284265]
    np.testing.assert_almost_equal(
        f1.get_neighbors(0.5, rs=np.random.RandomState(42)),
        expected,
        decimal=14,
    )

    # Test attributes are accessible
    assert f1.name == "param"
    assert f1.mu == pytest.approx(0.5)
    assert f1.sigma == pytest.approx(10.5)
    assert f1.q == pytest.approx(None)
    assert f1.log is False
    assert f1.default_value == pytest.approx(0.5)
    assert f1.normalized_default_value == pytest.approx(0.5)

    # Test copy
    copy_f1 = copy.copy(f1)

    assert copy_f1.name == f1.name
    assert copy_f1.mu == f1.mu
    assert copy_f1.sigma == f1.sigma
    assert copy_f1.default_value == f1.default_value

    f2 = NormalFloatHyperparameter("param", 0, 10, q=0.1)
    f2_ = NormalFloatHyperparameter("param", 0, 10, q=0.1)
    assert f2 == f2_
    assert str(f2) == "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 0.0, Q: 0.1"

    f3 = NormalFloatHyperparameter("param", 0, 10, log=True)
    f3_ = NormalFloatHyperparameter("param", 0, 10, log=True)
    assert f3 == f3_
    assert str(f3) == "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 1.0, on log-scale"

    f4 = NormalFloatHyperparameter("param", 0, 10, default_value=1.0)
    f4_ = NormalFloatHyperparameter("param", 0, 10, default_value=1.0)
    assert f4 == f4_
    assert str(f4) == "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 1.0"

    f5 = NormalFloatHyperparameter("param", 0, 10, default_value=3.0, q=0.1, log=True)
    f5_ = NormalFloatHyperparameter("param", 0, 10, default_value=3.0, q=0.1, log=True)
    assert f5 == f5_
    assert (
        str(f5)
        == "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 3.0, on log-scale, Q: 0.1"
    )

    assert f1 != f2
    assert f1 != "UniformFloat"

    with pytest.raises(ValueError):
        f6 = NormalFloatHyperparameter(
            "param",
            5,
            10,
            lower=0.1,
            upper=0.1,
            default_value=5.0,
            q=0.1,
            log=True,
        )

    with pytest.raises(ValueError):
        f6 = NormalFloatHyperparameter(
            "param",
            5,
            10,
            lower=0.1,
            default_value=5.0,
            q=0.1,
            log=True,
        )

    with pytest.raises(ValueError):
        f6 = NormalFloatHyperparameter(
            "param",
            5,
            10,
            upper=0.1,
            default_value=5.0,
            q=0.1,
            log=True,
        )

    f6 = NormalFloatHyperparameter(
        "param",
        5,
        10,
        lower=0.1,
        upper=10,
        default_value=5.0,
        q=0.1,
        log=True,
    )
    f6_ = NormalFloatHyperparameter(
        "param",
        5,
        10,
        lower=0.1,
        upper=10,
        default_value=5.0,
        q=0.1,
        log=True,
    )
    assert f6 == f6_
    assert (
        "param, Type: NormalFloat, Mu: 5.0 Sigma: 10.0, Range: [0.1, 10.0], "
        + "Default: 5.0, on log-scale, Q: 0.1"
        == str(f6)
    )

    # Due to seemingly different numbers with x86_64 and i686 architectures
    # we got these numbers, where the first one is slightly different
    # They are equal up to 14 decimal places
    expected = [9.967141530112327, 3.6173569882881536, 10.0, 10.0]
    np.testing.assert_almost_equal(
        f6.get_neighbors(5, rs=np.random.RandomState(42)),
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
        q=0.1,
        log=True,
        default_value=1.0,
        meta=dict(META_DATA),
    )
    assert f_meta.meta == META_DATA

    # Test get_size
    for float_hp in (f1, f2, f3, f4, f5):
        assert np.isinf(float_hp.get_size())
    assert f6.get_size() == 100

    with pytest.raises(ValueError):
        _ = NormalFloatHyperparameter("param", 5, 10, lower=0.1, upper=10, default_value=10.01)

    with pytest.raises(ValueError):
        _ = NormalFloatHyperparameter("param", 5, 10, lower=0.1, upper=10, default_value=0.09)


def test_normalfloat_to_uniformfloat():
    f1 = NormalFloatHyperparameter("param", 0, 10, q=0.1)
    f1_expected = UniformFloatHyperparameter("param", -30, 30, q=0.1)
    f1_actual = f1.to_uniform()
    assert f1_expected == f1_actual

    f2 = NormalFloatHyperparameter("param", 0, 10, lower=-20, upper=20, q=0.1)
    f2_expected = UniformFloatHyperparameter("param", -20, 20, q=0.1)
    f2_actual = f2.to_uniform()
    assert f2_expected == f2_actual


def test_normalfloat_is_legal():
    f1 = NormalFloatHyperparameter("param", 0, 10)
    assert f1.is_legal(3.0)
    assert f1.is_legal(2)
    assert not f1.is_legal("Hahaha")

    # Test legal vector values
    assert f1.is_legal_vector(1.0)
    assert f1.is_legal_vector(0.0)
    assert f1.is_legal_vector(0)
    assert f1.is_legal_vector(0.3)
    assert f1.is_legal_vector(-0.1)
    assert f1.is_legal_vector(1.1)
    with pytest.raises(TypeError):
        f1.is_legal_vector("Hahaha")

    f2 = NormalFloatHyperparameter("param", 5, 10, lower=0.1, upper=10, default_value=5.0)
    assert f2.is_legal(5.0)
    assert not f2.is_legal(10.01)
    assert not f2.is_legal(0.09)


def test_normalfloat_to_integer():
    f1 = NormalFloatHyperparameter("param", 0, 10)
    f2_expected = NormalIntegerHyperparameter("param", 0, 10)
    f2_actual = f1.to_integer()
    assert f2_expected == f2_actual


def test_normalfloat_pdf():
    c1 = NormalFloatHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        mu=3,
        sigma=2,
        log=True,
    )
    c3 = NormalFloatHyperparameter("param", lower=0, upper=0.5, mu=-1, sigma=0.2)

    point_1 = np.array([3])
    point_1_log = np.array([np.exp(3)])
    point_2 = np.array([10])
    point_2_log = np.array([np.exp(10)])
    point_3 = np.array([0])
    array_1 = np.array([3, 10, 10.01])
    array_1_log = np.array([np.exp(3), np.exp(10), np.exp(10.01)])
    point_outside_range_1 = np.array([-0.01])
    point_outside_range_2 = np.array([10.01])
    point_outside_range_1_log = np.array([np.exp(-0.01)])
    point_outside_range_2_log = np.array([np.exp(10.01)])
    wrong_shape_1 = np.array([[3]])
    wrong_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    wrong_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    assert c1.pdf(point_1)[0] == pytest.approx(0.2138045617479014)
    assert c2.pdf(point_1_log)[0] == pytest.approx(0.2138045617479014)
    assert c1.pdf(point_2)[0] == pytest.approx(0.000467695579850518)
    assert c2.pdf(point_2_log)[0] == pytest.approx(0.000467695579850518)
    assert c3.pdf(point_3)[0] == pytest.approx(25.932522722334905)
    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1.pdf(point_outside_range_1)[0] == 0.0
    assert c1.pdf(point_outside_range_2)[0] == 0.0
    assert c2.pdf(point_outside_range_1_log)[0] == 0.0
    assert c2.pdf(point_outside_range_2_log)[0] == 0.0

    array_results = c1.pdf(array_1)
    array_results_log = c2.pdf(array_1_log)
    expected_results = np.array([0.2138045617479014, 0.0004676955798505186, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res in zip(array_results, array_results, expected_results):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_res)

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    c_nobounds = NormalFloatHyperparameter("param", mu=3, sigma=2)
    assert c_nobounds.pdf(np.array([2]))[0] == pytest.approx(0.17603266338214976)

    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_1)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_2)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_3)


def test_normalfloat__pdf():
    c1 = NormalFloatHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        mu=3,
        sigma=2,
        log=True,
    )
    c3 = NormalFloatHyperparameter("param", lower=0, upper=0.5, mu=-1, sigma=0.2)

    # since there is no logtransformation, the logged and unlogged parameters
    # should output the same given the same input

    point_1 = np.array([3])
    point_2 = np.array([10])
    point_3 = np.array([0])
    array_1 = np.array([3, 10, 10.01])
    point_outside_range_1 = np.array([-0.01])
    point_outside_range_2 = np.array([10.01])
    accepted_shape_1 = np.array([[3]])
    accepted_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    accepted_shape_3 = np.array([7, 5, 3]).reshape(-1, 1)

    assert c1._pdf(point_1)[0] == pytest.approx(0.2138045617479014)
    assert c2._pdf(point_1)[0] == pytest.approx(0.2138045617479014)
    assert c1._pdf(point_2)[0] == pytest.approx(0.000467695579850518)
    assert c2._pdf(point_2)[0] == pytest.approx(0.000467695579850518)
    assert c3._pdf(point_3)[0] == pytest.approx(25.932522722334905)
    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1._pdf(point_outside_range_1)[0] == 0.0
    assert c1._pdf(point_outside_range_2)[0] == 0.0
    assert c2._pdf(point_outside_range_1)[0] == 0.0
    assert c2._pdf(point_outside_range_2)[0] == 0.0

    array_results = c1.pdf(array_1)
    array_results_log = c2.pdf(array_1)
    expected_results = np.array([0.2138045617479014, 0.0004676955798505186, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res in zip(array_results, array_results, expected_results):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_res)

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    c_nobounds = NormalFloatHyperparameter("param", mu=3, sigma=2)
    assert c_nobounds.pdf(np.array([2]))[0] == pytest.approx(0.17603266338214976)

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1._pdf(accepted_shape_1)
    c1._pdf(accepted_shape_2)
    c1._pdf(accepted_shape_3)


def test_normalfloat_get_max_density():
    c1 = NormalFloatHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalFloatHyperparameter(
        "logparam",
        lower=np.exp(0),
        upper=np.exp(10),
        mu=3,
        sigma=2,
        log=True,
    )
    c3 = NormalFloatHyperparameter("param", lower=0, upper=0.5, mu=-1, sigma=0.2)
    assert c1.get_max_density() == pytest.approx(0.2138045617479014, abs=1e-9)
    assert c2.get_max_density() == pytest.approx(0.2138045617479014, abs=1e-9)
    assert c3.get_max_density() == pytest.approx(25.932522722334905, abs=1e-9)


def test_betafloat():
    # TODO test non-equality
    f1 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.0)
    f1_ = BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=3, beta=1)
    assert f1 == f1_
    assert (
        str(f1_) == "param, Type: BetaFloat, Alpha: 3.0 Beta: 1.0, Range: [-2.0, 2.0], Default: 2.0"
    )

    u1 = UniformFloatHyperparameter("param", lower=0.0, upper=1.0)
    b1 = BetaFloatHyperparameter("param", lower=0.0, upper=1.0, alpha=3.0, beta=1.0)

    # with identical domains, beta and uniform should sample the same points
    assert u1.get_neighbors(0.5, rs=np.random.RandomState(42)) == b1.get_neighbors(
        0.5,
        rs=np.random.RandomState(42),
    )
    # Test copy
    copy_f1 = copy.copy(f1)
    assert copy_f1.name == f1.name

    f2 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.0, q=0.1)
    f2_ = BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=3, beta=1, q=0.1)
    assert f2 == f2_

    assert (
        str(f2)
        == "param, Type: BetaFloat, Alpha: 3.0 Beta: 1.0, Range: [-2.0, 2.0], Default: 2.0, Q: 0.1"
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
        == "param, Type: BetaFloat, Alpha: 6.0 Beta: 2.0, Range: [1e-05, 10.0], Default: 1.0, on log-scale"
    )

    f4 = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=1000.0,
        alpha=2.0,
        beta=2.0,
        log=True,
        q=1.0,
    )
    f4_ = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=1000.0,
        alpha=2.0,
        beta=2.0,
        log=True,
        q=1.0,
    )

    assert f4 == f4_
    assert (
        str(f4)
        == "param, Type: BetaFloat, Alpha: 2.0 Beta: 2.0, Range: [1.0, 1000.0], Default: 32.0, on log-scale, Q: 1.0"
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

    with pytest.raises(
        UserWarning,
        match=(
            "Logscale and quantization together results in "
            "incorrect default values. We recommend specifying a default "
            "value manually for this specific case."
        ),
    ):
        BetaFloatHyperparameter(
            "param",
            lower=1,
            upper=100.0,
            alpha=3.0,
            beta=2.0,
            log=True,
            q=1,
        )


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
    assert f_max.normalized_default_value == pytest.approx(1.0)

    f_max_log = BetaFloatHyperparameter(
        "param",
        lower=1.0,
        upper=10.0,
        alpha=3.0,
        beta=1.0,
        log=True,
    )
    assert f_max_log.default_value == pytest.approx(10.0)
    assert f_max_log.normalized_default_value == pytest.approx(1.0)

    # should default to the minimal value in the search space
    f_min = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=1.0, beta=1.5)
    assert f_min.default_value == pytest.approx(-2.0)
    assert f_min.normalized_default_value == pytest.approx(0.0)

    f_min_log = BetaFloatHyperparameter(
        "param",
        lower=1.0,
        upper=10.0,
        alpha=1.0,
        beta=1.5,
        log=True,
    )
    assert f_min_log.default_value == pytest.approx(1.0)
    assert f_min_log.normalized_default_value == pytest.approx(0.0)

    # Symmeric, should default to the middle
    f_symm = BetaFloatHyperparameter("param", lower=5, upper=9, alpha=4.6, beta=4.6)
    assert f_symm.default_value == pytest.approx(7)
    assert f_symm.normalized_default_value == pytest.approx(0.5)

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
    assert f_symm_log.normalized_default_value == pytest.approx(0.5)

    # Uniform, should also default to the middle
    f_unif = BetaFloatHyperparameter("param", lower=2.2, upper=3.2, alpha=1.0, beta=1.0)
    assert f_unif.default_value == pytest.approx(2.7)
    assert f_unif.normalized_default_value == pytest.approx(0.5)

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
    assert f_unif_log.normalized_default_value == pytest.approx(0.5)

    # Then, test a case where the default value is the mode of the beta dist
    f_max = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=4.7, beta=2.12)
    assert f_max.default_value == pytest.approx(1.0705394190871367)
    assert f_max.normalized_default_value == pytest.approx(0.7676348547717842)

    f_max_log = BetaFloatHyperparameter(
        "param",
        lower=np.exp(-2.0),
        upper=np.exp(2.0),
        alpha=4.7,
        beta=2.12,
        log=True,
    )
    assert f_max_log.default_value == pytest.approx(np.exp(1.0705394190871367))
    assert f_max_log.normalized_default_value == pytest.approx(0.7676348547717842)

    # These parameters do not yeild an integer default solution
    f_quant = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=4.7, beta=2.12, q=1)
    assert f_quant.default_value == pytest.approx(1.0)

    f_log_quant = BetaFloatHyperparameter(
        "param",
        lower=1,
        upper=100000,
        alpha=2,
        beta=2,
        q=1,
        log=True,
    )
    assert f_log_quant.default_value == pytest.approx(316)

    # since it's quantized, it gets distributed evenly among the search space
    # as such, the possible normalized defaults are 0.1, 0.3, 0.5, 0.7, 0.9
    assert f_quant.normalized_default_value == pytest.approx(0.7, abs=1e-4)

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
    f1 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=4, beta=2, q=0.1)
    f1_expected = UniformFloatHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        q=0.1,
        default_value=1,
    )
    f1_actual = f1.to_uniform()
    assert f1_expected == f1_actual

    f2 = BetaFloatHyperparameter("param", lower=1, upper=1000, alpha=3, beta=2, log=True)
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
    f2_expected = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=4, beta=2)
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

    assert c1.pdf(point_1)[0] == pytest.approx(0.07559999999999997)
    assert c2.pdf(point_1_log)[0] == pytest.approx(0.07559999999999997)
    assert c1.pdf(point_2)[0] == pytest.approx(0.011761200000000013)
    assert c2.pdf(point_2_log)[0] == pytest.approx(0.011761200000000013)
    assert c3.pdf(point_3)[0] == pytest.approx(30.262164001861198)
    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1.pdf(point_outside_range_1)[0] == 0.0
    assert c1.pdf(point_outside_range_2)[0] == 0.0
    assert c2.pdf(point_outside_range_1_log)[0] == 0.0
    assert c2.pdf(point_outside_range_2_log)[0] == 0.0

    array_results = c1.pdf(array_1)
    array_results_log = c2.pdf(array_1_log)
    expected_results = np.array([0.07559999999999997, 0.011761200000000013, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res in zip(array_results, array_results, expected_results):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_res)

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_1)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_2)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_3)


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

    assert c1._pdf(point_1)[0] == pytest.approx(0.07559999999999997)
    assert c2._pdf(point_1)[0] == pytest.approx(0.07559999999999997)
    assert c1._pdf(point_2)[0] == pytest.approx(0.011761200000000013)
    assert c2._pdf(point_2)[0] == pytest.approx(0.011761200000000013)
    assert c3._pdf(point_3)[0] == pytest.approx(30.262164001861198)

    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1._pdf(point_outside_range_1)[0] == 0.0
    assert c1._pdf(point_outside_range_2)[0] == 0.0
    assert c2._pdf(point_outside_range_1)[0] == 0.0
    assert c2._pdf(point_outside_range_2)[0] == 0.0

    array_results = c1._pdf(array_1)
    array_results_log = c2._pdf(array_1)
    expected_results = np.array([0.07559999999999997, 0.011761200000000013, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res in zip(array_results, array_results, expected_results):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_res)

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1._pdf(accepted_shape_1)
    c1._pdf(accepted_shape_2)
    c1._pdf(accepted_shape_3)


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
    assert c1.get_max_density() == pytest.approx(0.17777777777777776)
    assert c2.get_max_density() == pytest.approx(0.17777777777777776)
    assert c3.get_max_density() == pytest.approx(38.00408137865127)


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
    assert f1.q is None
    assert f1.default_value == 2
    assert f1.log is False
    assert f1.normalized_default_value == pytest.approx((2.0 + 0.49999) / (5.49999 + 0.49999))

    quantization_warning = (
        "Setting quantization < 1 for Integer Hyperparameter 'param' has no effect"
    )
    with pytest.warns(UserWarning, match=quantization_warning):
        f2 = UniformIntegerHyperparameter("param", 0, 10, q=0.1)
    with pytest.warns(UserWarning, match=quantization_warning):
        f2_ = UniformIntegerHyperparameter("param", 0, 10, q=0.1)
    assert f2 == f2_
    assert str(f2) == "param, Type: UniformInteger, Range: [0, 10], Default: 5"

    f2_large_q = UniformIntegerHyperparameter("param", 0, 10, q=2)
    f2_large_q_ = UniformIntegerHyperparameter("param", 0, 10, q=2)
    assert f2_large_q == f2_large_q_
    assert str(f2_large_q) == "param, Type: UniformInteger, Range: [0, 10], Default: 5, Q: 2"

    f3 = UniformIntegerHyperparameter("param", 1, 10, log=True)
    f3_ = UniformIntegerHyperparameter("param", 1, 10, log=True)
    assert f3 == f3_
    assert str(f3) == "param, Type: UniformInteger, Range: [1, 10], Default: 3, on log-scale"

    f4 = UniformIntegerHyperparameter("param", 1, 10, default_value=1, log=True)
    f4_ = UniformIntegerHyperparameter("param", 1, 10, default_value=1, log=True)
    assert f4 == f4_
    assert str(f4) == "param, Type: UniformInteger, Range: [1, 10], Default: 1, on log-scale"

    with pytest.warns(UserWarning, match=quantization_warning):
        f5 = UniformIntegerHyperparameter("param", 1, 10, default_value=1, q=0.1, log=True)
    with pytest.warns(UserWarning, match=quantization_warning):
        f5_ = UniformIntegerHyperparameter("param", 1, 10, default_value=1, q=0.1, log=True)
    assert f5 == f5_
    assert str(f5) == "param, Type: UniformInteger, Range: [1, 10], Default: 1, on log-scale"

    assert f1 != "UniformFloat"

    # test that meta-data is stored correctly
    with pytest.warns(UserWarning, match=quantization_warning):
        f_meta = UniformIntegerHyperparameter(
            "param",
            1,
            10,
            q=0.1,
            log=True,
            default_value=1,
            meta=dict(META_DATA),
        )
    assert f_meta.meta == META_DATA

    assert f1.get_size() == 6
    assert f2.get_size() == 11
    assert f2_large_q.get_size() == 6
    assert f3.get_size() == 10
    assert f4.get_size() == 10
    assert f5.get_size() == 10


def test_uniformint_legal_float_values():
    n_iter = UniformIntegerHyperparameter("n_iter", 5.0, 1000.0, default_value=20.0)

    assert isinstance(n_iter.default_value, int)
    with pytest.raises(ValueError,
        match=r"For the Integer parameter n_iter, "
        r"the value must be an Integer, too."
        r" Right now it is a <(type|class) "
        r"'float'>"
        r" with value 20.5.",
    ):
        _ = UniformIntegerHyperparameter("n_iter", 5.0, 1000.0, default_value=20.5)


def test_uniformint_illegal_bounds():
    with pytest.raises(
        ValueError,
        match=r"Negative lower bound \(0\) for log-scale hyperparameter " r"param is forbidden.",
    ):
        UniformIntegerHyperparameter("param", 0, 10, log=True)

    with pytest.raises(
        ValueError,
        match="Upper bound 1 must be larger than lower bound 0 for " "hyperparameter param",
    ):
        _ = UniformIntegerHyperparameter( "param", 1, 0)


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
    point_outside_range_log = np.array([10001])
    wrong_shape_1 = np.array([[3]])
    wrong_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    wrong_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    # need to lower the amount of places since the bounds
    # are inexact (._lower=-0.49999, ._upper=4.49999)
    assert c1.pdf(point_1)[0] == pytest.approx(0.2, abs=1e-5)
    assert c2.pdf(point_1_log)[0] == pytest.approx(0.0001, abs=1e-5)
    assert c1.pdf(point_2)[0] == pytest.approx(0.2, abs=1e-5)
    assert c2.pdf(point_2_log)[0] == pytest.approx(0.0001, abs=1e-5)
    assert c1.pdf(non_integer_point)[0] == pytest.approx(0.0, abs=1e-5)
    assert c2.pdf(non_integer_point)[0] == pytest.approx(0.0, abs=1e-5)
    assert c3.pdf(point_1)[0] == pytest.approx(0.07142857142857142, abs=1e-5)

    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    # since inverse_transform pulls everything into range,
    # even points outside get evaluated in range
    assert c1.pdf(point_outside_range)[0] == pytest.approx(0.2, abs=1e-5)
    assert c2.pdf(point_outside_range_log)[0] == pytest.approx(0.0001, abs=1e-5)

    # this, however, is a negative value on a log param, which cannot be pulled into range
    with pytest.warns(RuntimeWarning, match="invalid value encountered in log"):
        assert c2.pdf(point_outside_range)[0] == 0.0

    array_results = c1.pdf(array_1)
    array_results_log = c2.pdf(array_1)
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

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_1)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_2)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_3)


def test_uniformint__pdf():
    c1 = UniformIntegerHyperparameter("param", lower=0, upper=4)
    c2 = UniformIntegerHyperparameter("logparam", lower=1, upper=10000, log=True)

    point_1 = np.array([0])
    point_2 = np.array([0.7])
    array_1 = np.array([0, 0.7, 1.1])
    point_outside_range = np.array([-0.1])
    accepted_shape_1 = np.array([[0.7]])
    accepted_shape_2 = np.array([0, 0.7, 1.1]).reshape(1, -1)
    accepted_shape_3 = np.array([1.1, 0.7, 0]).reshape(-1, 1)

    # need to lower the amount of places since the bounds
    # are inexact (._lower=-0.49999, ._upper=4.49999)
    assert c1._pdf(point_1)[0] == pytest.approx(0.2, abs=1e-5)
    assert c2._pdf(point_1)[0] == pytest.approx(0.0001, abs=1e-5)
    assert c1._pdf(point_2)[0] == pytest.approx(0.2, abs=1e-5)
    assert c2._pdf(point_2)[0] == pytest.approx(0.0001, abs=1e-5)

    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    # since inverse_transform pulls everything into range,
    # even points outside get evaluated in range
    assert c1._pdf(point_outside_range)[0] == pytest.approx(0.0, abs=1e-5)
    assert c2._pdf(point_outside_range)[0] == pytest.approx(0.0, abs=1e-5)

    array_results = c1._pdf(array_1)
    array_results_log = c2._pdf(array_1)
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

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1._pdf(0.2)
    with pytest.raises(TypeError):
        c1._pdf("pdf")

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1._pdf(accepted_shape_1)
    c1._pdf(accepted_shape_2)
    c1._pdf(accepted_shape_3)


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
            float_value = c1._inverse_transform(np.array([i_value]))[0]
            neighbors = c1.get_neighbors(float_value, rs, number=i_upper, transform=True)
            assert set(neighbors) == set(range(i_upper + 1)) - {i_value}


def test_normalint():
    # TODO test for unequal!
    f1 = NormalIntegerHyperparameter("param", 0.5, 5.5)
    f1_ = NormalIntegerHyperparameter("param", 0.5, 5.5)
    assert f1 == f1_
    assert str(f1) == "param, Type: NormalInteger, Mu: 0.5 Sigma: 5.5, Default: 0.5"

    # Test attributes are accessible
    assert f1.name == "param"
    assert f1.mu == 0.5
    assert f1.sigma == 5.5
    assert f1.q is None
    assert f1.log is False
    assert f1.default_value == pytest.approx(0.5)
    assert f1.normalized_default_value == pytest.approx(0.5)

    with pytest.warns(
        UserWarning,
        match="Setting quantization < 1 for Integer " "Hyperparameter 'param' has no effect",
    ):
        f2 = NormalIntegerHyperparameter("param", 0, 10, q=0.1)
    with pytest.warns(
        UserWarning,
        match="Setting quantization < 1 for Integer " "Hyperparameter 'param' has no effect",
    ):
        f2_ = NormalIntegerHyperparameter("param", 0, 10, q=0.1)
    assert f2 == f2_
    assert str(f2) == "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 0"

    f2_large_q = NormalIntegerHyperparameter("param", 0, 10, q=2)
    f2_large_q_ = NormalIntegerHyperparameter("param", 0, 10, q=2)
    assert f2_large_q == f2_large_q_
    assert str(f2_large_q) == "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 0, Q: 2"

    f3 = NormalIntegerHyperparameter("param", 0, 10, log=True)
    f3_ = NormalIntegerHyperparameter("param", 0, 10, log=True)
    assert f3 == f3_
    assert str(f3) == "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 1, on log-scale"

    f4 = NormalIntegerHyperparameter("param", 0, 10, default_value=3, log=True)
    f4_ = NormalIntegerHyperparameter("param", 0, 10, default_value=3, log=True)
    assert f4 == f4_
    assert str(f4) == "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 3, on log-scale"

    with pytest.warns(
        UserWarning,
        match="Setting quantization < 1 for Integer " "Hyperparameter 'param' has no effect",
    ):
        f5 = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
        f5_ = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
    assert f5 == f5_
    assert str(f5) == "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 1, on log-scale"

    assert f1 != f2
    assert f1 != "UniformFloat"

    # test that meta-data is stored correctly
    f_meta = NormalIntegerHyperparameter(
        "param",
        0,
        10,
        default_value=1,
        log=True,
        meta=dict(META_DATA),
    )
    assert f_meta.meta == META_DATA

    # Test get_size
    for int_hp in (f1, f2, f3, f4, f5):
        assert np.isinf(int_hp.get_size())

    # Unbounded case
    f1 = NormalIntegerHyperparameter("param", 0, 10, q=1)
    assert f1.get_neighbors(2, np.random.RandomState(9001), number=1) == [1]
    assert f1.get_neighbors(2, np.random.RandomState(9001), number=5) == [0, 1, 9, 16, -1]

    # Bounded case
    f1 = NormalIntegerHyperparameter("param", 0, 10, q=1, lower=-100, upper=100)
    assert f1.get_neighbors(2, np.random.RandomState(9001), number=1) == [-11]
    assert f1.get_neighbors(2, np.random.RandomState(9001), number=5) == [4, 11, 12, 15, -11]

    # Bounded case with default value out of bounds
    with pytest.raises(ValueError):
        _ = NormalIntegerHyperparameter("param", 5, 10, lower=1, upper=10, default_value=11)

    with pytest.raises(ValueError):
        _ = NormalIntegerHyperparameter("param", 5, 10, lower=1, upper=10, default_value=0)


def test_normalint_legal_float_values():
    n_iter = NormalIntegerHyperparameter("n_iter", 0, 1.0, default_value=2.0)
    assert isinstance(n_iter.default_value, int)
    with pytest.raises(
        ValueError,
        match=r"For the Integer parameter n_iter, "
        r"the value must be an Integer, too."
        r" Right now it is a "
        r"<(type|class) 'float'>"
        r" with value 0.5.",
    ):
        _ = UniformIntegerHyperparameter("n_iter", 0, 1.0, default_value=0.5)


def test_normalint_to_uniform():
    with pytest.warns(
        UserWarning,
        match="Setting quantization < 1 for Integer " "Hyperparameter 'param' has no effect",
    ):
        f1 = NormalIntegerHyperparameter("param", 0, 10, q=0.1)
    f1_expected = UniformIntegerHyperparameter("param", -30, 30)
    f1_actual = f1.to_uniform()
    assert f1_expected == f1_actual


def test_normalint_is_legal():
    with pytest.warns(
        UserWarning,
        match="Setting quantization < 1 for Integer " "Hyperparameter 'param' has no effect",
    ):
        f1 = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
    assert not f1.is_legal(3.1)
    assert not f1.is_legal(3.0)  # 3.0 behaves like an Integer
    assert not f1.is_legal("BlaBlaBla")
    assert f1.is_legal(2)
    assert f1.is_legal(-15)

    # Test is legal vector
    assert f1.is_legal_vector(1.0)
    assert f1.is_legal_vector(0.0)
    assert f1.is_legal_vector(0)
    assert f1.is_legal_vector(0.3)
    assert f1.is_legal_vector(-0.1)
    assert f1.is_legal_vector(1.1)
    with pytest.raises(TypeError):
        f1.is_legal_vector("Hahaha")

    f2 = NormalIntegerHyperparameter("param", 5, 10, lower=1, upper=10, default_value=5)
    assert f2.is_legal(5)
    assert not f2.is_legal(0)
    assert not f2.is_legal(11)


def test_normalint_pdf():
    c1 = NormalIntegerHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalIntegerHyperparameter("logparam", lower=1, upper=1000, mu=3, sigma=2, log=True)
    c3 = NormalIntegerHyperparameter("param", lower=0, upper=2, mu=-1.2, sigma=0.5)

    point_1 = np.array([3])
    point_1_log = np.array([10])
    point_2 = np.array([10])
    point_2_log = np.array([1000])
    point_3 = np.array([0])
    array_1 = np.array([3, 10, 11])
    array_1_log = np.array([10, 570, 1001])
    point_outside_range_1 = np.array([-1])
    point_outside_range_2 = np.array([11])
    point_outside_range_1_log = np.array([0])
    point_outside_range_2_log = np.array([1001])
    non_integer_point = np.array([5.7])
    wrong_shape_1 = np.array([[3]])
    wrong_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    wrong_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    assert c1.pdf(point_1)[0] == pytest.approx(0.20747194595587332)
    assert c2.pdf(point_1_log)[0] == pytest.approx(0.002625781612612434)
    assert c1.pdf(point_2)[0] == pytest.approx(0.00045384303905059246)
    assert c2.pdf(point_2_log)[0] == pytest.approx(0.0004136885586376241)
    assert c3.pdf(point_3)[0] == pytest.approx(0.9988874412972069)
    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1.pdf(point_outside_range_1)[0] == 0.0
    assert c1.pdf(point_outside_range_2)[0] == 0.0
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
        assert c2.pdf(point_outside_range_1_log)[0] == 0.0
    assert c2.pdf(point_outside_range_2_log)[0] == 0.0

    assert c1.pdf(non_integer_point)[0] == 0.0
    assert c2.pdf(non_integer_point)[0] == 0.0

    array_results = c1.pdf(array_1)
    array_results_log = c2.pdf(array_1_log)
    expected_results = np.array([0.20747194595587332, 0.00045384303905059246, 0])
    expected_results_log = np.array([0.002625781612612434, 0.000688676747843256, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res, exp_log_res in zip(
        array_results,
        array_results_log,
        expected_results,
        expected_results_log,
    ):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_log_res)

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    c_nobounds = NormalFloatHyperparameter("param", mu=3, sigma=2)
    assert c_nobounds.pdf(np.array([2]))[0] == pytest.approx(0.17603266338214976)

    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_1)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_2)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_3)


def test_normalint__pdf():
    c1 = NormalIntegerHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalIntegerHyperparameter("logparam", lower=1, upper=1000, mu=3, sigma=2, log=True)

    point_1 = np.array([3])
    point_2 = np.array([5.2])
    array_1 = np.array([3, 5.2, 11])
    point_outside_range_1 = np.array([-1])
    point_outside_range_2 = np.array([11])

    assert c1._pdf(point_1)[0] == pytest.approx(0.20747194595587332)
    assert c2._pdf(point_1)[0] == pytest.approx(0.0027903779510164133)
    assert c1._pdf(point_2)[0] == pytest.approx(0.1132951239316783)
    assert c2._pdf(point_2)[0] == pytest.approx(0.001523754039709375)
    # TODO - change this once the is_legal support is there
    # but does not have an actual impact of now
    assert c1._pdf(point_outside_range_1)[0] == 0.0
    assert c1._pdf(point_outside_range_2)[0] == 0.0
    assert c2._pdf(point_outside_range_1)[0] == 0.0
    assert c2._pdf(point_outside_range_2)[0] == 0.0

    array_results = c1._pdf(array_1)
    array_results_log = c2._pdf(array_1)
    expected_results = np.array([0.20747194595587332, 0.1132951239316783, 0])
    expected_results_log = np.array([0.0027903779510164133, 0.001523754039709375, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results.shape
    for res, log_res, exp_res, exp_log_res in zip(
        array_results,
        array_results_log,
        expected_results,
        expected_results_log,
    ):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_log_res)

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    c_nobounds = NormalFloatHyperparameter("param", mu=3, sigma=2)
    assert c_nobounds.pdf(np.array([2]))[0] == pytest.approx(0.17603266338214976)


def test_normalint_get_max_density():
    c1 = NormalIntegerHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
    c2 = NormalIntegerHyperparameter("logparam", lower=1, upper=1000, mu=3, sigma=2, log=True)
    c3 = NormalIntegerHyperparameter("param", lower=0, upper=2, mu=-1.2, sigma=0.5)
    assert c1.get_max_density() == pytest.approx(0.20747194595587332)
    assert c2.get_max_density() == pytest.approx(0.002790371598208875)
    assert c3.get_max_density() == pytest.approx(0.9988874412972069)


def test_normalint_compute_normalization():
    ARANGE_CHUNKSIZE = 10_000_000
    lower, upper = 1, ARANGE_CHUNKSIZE * 2

    c = NormalIntegerHyperparameter("c", mu=10, sigma=500, lower=lower, upper=upper)
    chunks = arange_chunked(lower, upper, chunk_size=ARANGE_CHUNKSIZE)
    # exact computation over the complete range
    N = sum(c.nfhp.pdf(chunk).sum() for chunk in chunks)
    assert c.normalization_constant == pytest.approx(N, abs=1e-5)


############################################################
def test_betaint():
    # TODO test non-equality
    f1 = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.1)
    f1_ = BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=3, beta=1.1)
    assert f1 == f1_
    assert str(f1) == "param, Type: BetaInteger, Alpha: 3.0 Beta: 1.1, Range: [-2, 2], Default: 2"

    assert f1.alpha == pytest.approx(3.0)
    assert f1.beta == pytest.approx(1.1)

    # Test copy
    copy_f1 = copy.copy(f1)
    assert copy_f1.name == f1.name
    assert copy_f1.alpha == f1.alpha
    assert copy_f1.beta == f1.beta
    assert copy_f1.default_value == f1.default_value

    f2 = BetaIntegerHyperparameter("param", lower=-2.0, upper=4.0, alpha=3.0, beta=1.1, q=2)
    f2_ = BetaIntegerHyperparameter("param", lower=-2, upper=4, alpha=3, beta=1.1, q=2)
    assert f2 == f2_

    assert (
        str(f2)
        == "param, Type: BetaInteger, Alpha: 3.0 Beta: 1.1, Range: [-2, 4], Default: 4, Q: 2"
    )

    f3 = BetaIntegerHyperparameter("param", lower=1, upper=1000, alpha=3.0, beta=2.0, log=True)
    f3_ = BetaIntegerHyperparameter("param", lower=1, upper=1000, alpha=3.0, beta=2.0, log=True)
    assert f3 == f3_
    assert (
        str(f3)
        == "param, Type: BetaInteger, Alpha: 3.0 Beta: 2.0, Range: [1, 1000], Default: 100, on log-scale"
    )

    with pytest.raises(ValueError):
        BetaIntegerHyperparameter("param", lower=-1, upper=10.0, alpha=6.0, beta=2.0, log=True)

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
    f_max = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.0)
    assert f_max.default_value == pytest.approx(2.0)
    # since integer values are staggered over the normalized space
    assert f_max.normalized_default_value == pytest.approx(0.9, abs=1e-4)

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
    assert f_max_log.normalized_default_value == pytest.approx(0.983974646746037)

    # should default to the minimal value in the search space
    f_min = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=1.0, beta=1.5)
    assert f_min.default_value == pytest.approx(-2.0)
    assert f_min.normalized_default_value == pytest.approx(0.1, abs=1e-4)

    f_min_log = BetaIntegerHyperparameter(
        "param",
        lower=1.0,
        upper=10.0,
        alpha=1.0,
        beta=1.5,
        log=True,
    )
    assert f_min_log.default_value == pytest.approx(1.0)
    assert f_min_log.normalized_default_value == pytest.approx(0.22766524636349278)

    # Symmeric, should default to the middle
    f_symm = BetaIntegerHyperparameter("param", lower=5, upper=9, alpha=4.6, beta=4.6)
    assert f_symm.default_value == pytest.approx(7)
    assert f_symm.normalized_default_value == pytest.approx(0.5)

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
    assert f_symm_log.normalized_default_value == pytest.approx(0.5321491582577761)

    # Uniform, should also default to the middle
    f_unif = BetaIntegerHyperparameter("param", lower=2, upper=6, alpha=1.0, beta=1.0)
    assert f_unif.default_value == pytest.approx(4)
    assert f_unif.normalized_default_value == pytest.approx(0.5)

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
    assert f_unif_log.normalized_default_value == pytest.approx(0.5321491582577761)

    # Then, test a case where the default value is the mode of the beta dist somewhere in
    # the interior of the search space - but not the center
    f_max = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=4.7, beta=2.12)
    assert f_max.default_value == pytest.approx(1.0)
    assert f_max.normalized_default_value == pytest.approx(0.7, abs=1e-4)

    f_max_log = BetaIntegerHyperparameter(
        "param",
        lower=1,
        upper=np.round(np.exp(10)),
        alpha=4.7,
        beta=2.12,
        log=True,
    )
    assert f_max_log.default_value == pytest.approx(2157)
    assert f_max_log.normalized_default_value == pytest.approx(0.7827083200774537)

    # These parameters yield a mode at approximately 1.1, so should thus yield default at 2
    f_quant = BetaIntegerHyperparameter(
        "param",
        lower=-2.0,
        upper=2.0,
        alpha=4.7,
        beta=2.12,
        q=2,
    )
    assert f_quant.default_value == pytest.approx(2.0)

    # since it's quantized, it gets distributed evenly among the search space
    # as such, the possible normalized defaults are 0.1, 0.3, 0.5, 0.7, 0.9
    assert f_quant.normalized_default_value == pytest.approx(0.9, abs=1e-4)

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
    assert isinstance(f1.default_value, int)
    with pytest.raises(
        ValueError,
        match="Illegal default value 0.5",
    ):
        _ = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.1, default_value=0.5)


def test_betaint_to_uniform():
    with pytest.warns(
        UserWarning,
        match="Setting quantization < 1 for Integer " "Hyperparameter 'param' has no effect",
    ):
        f1 = BetaIntegerHyperparameter("param", lower=-30, upper=30, alpha=6.0, beta=2, q=0.1)

    f1_expected = UniformIntegerHyperparameter("param", -30, 30, default_value=20)
    f1_actual = f1.to_uniform()
    assert f1_expected == f1_actual


def test_betaint_pdf():
    c1 = BetaIntegerHyperparameter("param", alpha=3, beta=2, lower=0, upper=10)
    c2 = BetaIntegerHyperparameter("logparam", alpha=3, beta=2, lower=1, upper=1000, log=True)
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
    assert c1.pdf(point_1)[0] == pytest.approx(0.07636363636363634, abs=1e-3)
    assert c2.pdf(point_1_log)[0] == pytest.approx(0.0008724511426701984, abs=1e-3)
    assert c1.pdf(point_2)[0] == pytest.approx(0.09818181818181816, abs=1e-3)
    assert c2.pdf(point_2_log)[0] == pytest.approx(0.0008683622684160343, abs=1e-3)
    assert c3.pdf(point_3)[0] == pytest.approx(0.9979110652388783, abs=1e-3)

    assert c1.pdf(point_outside_range_1)[0] == 0.0
    assert c1.pdf(point_outside_range_2)[0] == 0.0
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
        assert c2.pdf(point_outside_range_1_log)[0] == 0.0
    assert c2.pdf(point_outside_range_2_log)[0] == 0.0

    assert c1.pdf(non_integer_point)[0] == 0.0
    assert c2.pdf(non_integer_point)[0] == 0.0

    array_results = c1.pdf(array_1)
    array_results_log = c2.pdf(array_1_log)
    expected_results = np.array([0.07636363636363634, 0.09818181818181816, 0])
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

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")

    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_1)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_2)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_3)


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
    point_1 = np.array([0.249995])
    point_1_log = np.array([0.345363])
    point_2 = np.array([0.850001])
    point_2_log = np.array([0.906480])
    array_1 = np.array([0.249995, 0.850001, 0.045])
    array_1_log = np.array([0.345363, 0.906480, 0.065])
    point_outside_range_1 = np.array([0.045])
    point_outside_range_1_log = np.array([0.06])
    point_outside_range_2 = np.array([0.96])

    accepted_shape_1 = np.array([[3]])
    accepted_shape_2 = np.array([3, 5, 7]).reshape(1, -1)
    accepted_shape_3 = np.array([3, 5, 7]).reshape(-1, 1)

    assert c1._pdf(point_1)[0] == pytest.approx(0.0475566)
    assert c2._pdf(point_1_log)[0] == pytest.approx(0.00004333811)
    assert c1._pdf(point_2)[0] == pytest.approx(0.1091810)
    assert c2._pdf(point_2_log)[0] == pytest.approx(0.00005571951)

    # test points that are actually outside of the _pdf range due to the skewing
    # of the unit hypercube space
    assert c1._pdf(point_outside_range_1)[0] == 0.0
    assert c1._pdf(point_outside_range_2)[0] == 0.0
    assert c2._pdf(point_outside_range_1_log)[0] == 0.0

    array_results = c1._pdf(array_1)
    array_results_log = c2._pdf(array_1_log)
    expected_results = np.array([0.0475566, 0.1091810, 0])
    expected_results_log = np.array([0.00004333811, 0.00005571951, 0])
    assert array_results.shape == expected_results.shape
    assert array_results_log.shape == expected_results_log.shape
    for res, log_res, exp_res, exp_log_res in zip(
        array_results,
        array_results_log,
        expected_results,
        expected_results_log,
    ):
        assert res == pytest.approx(exp_res)
        assert log_res == pytest.approx(exp_log_res)

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1._pdf(0.2)
    with pytest.raises(TypeError):
        c1._pdf("pdf")

    # Simply check that it runs, since _pdf does not restrict shape (only public method does)
    c1._pdf(accepted_shape_1)
    c1._pdf(accepted_shape_2)
    c1._pdf(accepted_shape_3)


def test_betaint_get_max_density():
    c1 = BetaIntegerHyperparameter("param", alpha=3, beta=2, lower=0, upper=10)
    c2 = BetaIntegerHyperparameter("logparam", alpha=3, beta=2, lower=1, upper=1000, log=True)
    c3 = BetaIntegerHyperparameter("param", alpha=1.1, beta=10, lower=0, upper=3)
    assert c1.get_max_density() == pytest.approx(0.1781818181818181)
    assert c2.get_max_density() == pytest.approx(0.0018733953504422762)
    assert c3.get_max_density() == pytest.approx(0.9979110652388783)


def test_betaint_compute_normalization():
    ARANGE_CHUNKSIZE = 10_000_000
    lower, upper = 0, ARANGE_CHUNKSIZE * 2

    c = BetaIntegerHyperparameter("c", alpha=3, beta=2, lower=lower, upper=upper)
    chunks = arange_chunked(lower, upper, chunk_size=ARANGE_CHUNKSIZE)
    # exact computation over the complete range
    N = sum(c.bfhp.pdf(chunk).sum() for chunk in chunks)
    assert c.normalization_constant == pytest.approx(N, abs=1e-5)


def test_categorical():
    # TODO test for inequality
    f1 = CategoricalHyperparameter("param", [0, 1])
    f1_ = CategoricalHyperparameter("param", [0, 1])
    assert f1 == f1_
    assert str(f1) == "param, Type: Categorical, Choices: {0, 1}, Default: 0"

    # Test attributes are accessible
    assert f1.name == "param"
    assert f1.num_choices == 2
    assert f1.default_value == 0
    assert f1.normalized_default_value == 0
    assert f1.probabilities == pytest.approx((0.5, 0.5))

    f2 = CategoricalHyperparameter("param", list(range(1000)))
    f2_ = CategoricalHyperparameter("param", list(range(1000)))
    assert f2 == f2_
    assert "param, Type: Categorical, Choices: {%s}, Default: 0" % ", ".join(
        [str(choice) for choice in range(1000)],
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

    assert f1.get_size() == 2
    assert f2.get_size() == 1000
    assert f3.get_size() == 999
    assert f4.get_size() == 1000
    assert f5.get_size() == 1000
    assert f6.get_size() == 2


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
    c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2], default_value="a")
    c2 = CategoricalHyperparameter("param", ["b", "a"], weights=[2, 1], default_value="a")
    assert c1 == c2

    c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2], default_value="a")
    c2 = CategoricalHyperparameter("param", ["b", "a"], weights=[1, 2], default_value="a")
    assert c1 != c2

    c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2], default_value="a")
    c2 = CategoricalHyperparameter("param", ["b", "a"], default_value="a")
    assert c1 != c2

    c1 = CategoricalHyperparameter("param", ["a", "b"], default_value="a")
    c2 = CategoricalHyperparameter("param", ["b", "a"], weights=[1, 2], default_value="a")
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
    assert f1.is_legal("a")
    assert f1.is_legal("a")
    assert not f1.is_legal("c")
    assert not f1.is_legal(3)

    # Test is legal vector
    assert f1.is_legal_vector(1.0)
    assert f1.is_legal_vector(0.0)
    assert f1.is_legal_vector(0)
    assert not f1.is_legal_vector(0.3)
    assert not f1.is_legal_vector(-0.1)
    with pytest.raises(TypeError):
        f1.is_legal_vector("Hahaha")


def test_categorical_choices():
    with pytest.raises(
        ValueError,
        match="Choices for categorical hyperparameters param contain choice 'a' 2 times, "
        "while only a single oocurence is allowed.",
    ):
        CategoricalHyperparameter("param", ["a", "a"])

    with pytest.raises(TypeError, match="Choice 'None' is not supported"):
        CategoricalHyperparameter("param", ["a", None])


def test_categorical_default():
    # Test that the default value is the most probable choice when weights are given
    f1 = CategoricalHyperparameter("param", ["a", "b"])
    f2 = CategoricalHyperparameter("param", ["a", "b"], weights=[0.3, 0.6])
    f3 = CategoricalHyperparameter("param", ["a", "b"], weights=[0.6, 0.3])
    assert f1.default_value != f2.default_value
    assert f1.default_value == f3.default_value


def test_sample_UniformFloatHyperparameter():
    # This can sample four distributions
    def sample(hp):
        rs = np.random.RandomState(1)
        counts_per_bin = [0 for _ in range(21)]
        value = None
        for _ in range(100000):
            value = hp.sample(rs)
            if hp.log:
                assert value <= np.exp(hp._upper)
                assert value >= np.exp(hp._lower)
            else:
                assert value <= hp._upper
                assert value >= hp._lower
            index = int((value - hp.lower) / (hp.upper - hp.lower) * 20)
            counts_per_bin[index] += 1

        assert isinstance(value, float)
        return counts_per_bin

    # Uniform
    hp = UniformFloatHyperparameter("ufhp", 0.5, 2.5)

    counts_per_bin = sample(hp)
    # The 21st bin is only filled if exactly 2.5 is sampled...very rare...
    for bin in counts_per_bin[:-1]:
        assert 5200 > bin > 4800
    assert sample(hp) == sample(hp)

    # Quantized Uniform
    hp = UniformFloatHyperparameter("ufhp", 0.0, 1.0, q=0.1)

    counts_per_bin = sample(hp)
    for bin in counts_per_bin[::2]:
        assert 9301 > bin > 8700
    for bin in counts_per_bin[1::2]:
        assert bin == 0
    assert sample(hp) == sample(hp)

    # Log Uniform
    hp = UniformFloatHyperparameter("ufhp", 1.0, np.e**2, log=True)

    counts_per_bin = sample(hp)
    assert counts_per_bin == [
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
        0,
    ]
    assert sample(hp) == sample(hp)

    # Quantized Log-Uniform
    # 7.2 ~ np.round(e * e, 1)
    hp = UniformFloatHyperparameter("ufhp", 1.2, 7.2, q=0.6, log=True)

    counts_per_bin = sample(hp)
    assert counts_per_bin == [
        24359,
        15781,
        0,
        11635,
        0,
        0,
        9506,
        7867,
        0,
        0,
        6763,
        0,
        5919,
        0,
        5114,
        4798,
        0,
        0,
        4339,
        0,
        3919,
    ]
    assert sample(hp) == sample(hp)

    # Issue #199
    hp = UniformFloatHyperparameter("uni_float_q", lower=1e-4, upper=1e-1, q=1e-5, log=True)
    assert np.isfinite(hp._lower)
    assert np.isfinite(hp._upper)
    sample(hp)


def test_categorical_pdf():
    c1 = CategoricalHyperparameter("x1", choices=["one", "two", "three"], weights=[2, 1, 2])
    c2 = CategoricalHyperparameter("x1", choices=["one", "two", "three"], weights=[5, 0, 2])
    c3 = CategoricalHyperparameter("x1", choices=["one", "two", "three", "four"])

    point_1 = np.array(["one"])
    point_2 = np.array(["two"])

    wrong_shape_1 = np.array([["one"]])
    wrong_shape_2 = np.array(["one", "two"]).reshape(1, -1)
    wrong_shape_3 = np.array(["one", "two"]).reshape(-1, 1)

    assert c1.pdf(point_1)[0] == 0.4
    assert c1.pdf(point_2)[0] == 0.2
    assert c2.pdf(point_1)[0] == pytest.approx(0.7142857142857143)
    assert c2.pdf(point_2)[0] == 0.0
    assert c3.pdf(point_1)[0] == 0.25

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")
    with pytest.raises(TypeError):
        c1.pdf("one")
    with pytest.raises(ValueError):
        c1.pdf(np.array(["zero"]))

    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_1)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_2)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_3)


def test_categorical__pdf():
    c1 = CategoricalHyperparameter("x1", choices=["one", "two", "three"], weights=[2, 1, 2])
    c2 = CategoricalHyperparameter("x1", choices=["one", "two", "three"], weights=[5, 0, 2])

    point_1 = np.array([0])
    point_2 = np.array([1])
    array_1 = np.array([1, 0, 2])
    nan = np.array([0, np.nan])
    assert c1._pdf(point_1)[0] == 0.4
    assert c1._pdf(point_2)[0] == 0.2
    assert c2._pdf(point_1)[0] == pytest.approx(0.7142857142857143)
    assert c2._pdf(point_2)[0] == 0.0

    array_results = c1._pdf(array_1)
    expected_results = np.array([0.2, 0.4, 0.4])
    assert array_results.shape == expected_results.shape
    for res, exp_res in zip(array_results, expected_results):
        assert res == exp_res

    nan_results = c1._pdf(nan)
    expected_results = np.array([0.4, 0])
    assert nan_results.shape == expected_results.shape
    for res, exp_res in zip(nan_results, expected_results):
        assert res == exp_res

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1._pdf(0.2)
    with pytest.raises(TypeError):
        c1._pdf("pdf")
    with pytest.raises(TypeError):
        c1._pdf("one")
    with pytest.raises(TypeError):
        c1._pdf(np.array(["zero"]))


def test_categorical_get_max_density():
    c1 = CategoricalHyperparameter("x1", choices=["one", "two", "three"], weights=[2, 1, 2])
    c2 = CategoricalHyperparameter("x1", choices=["one", "two", "three"], weights=[5, 0, 2])
    c3 = CategoricalHyperparameter("x1", choices=["one", "two", "three"])
    assert c1.get_max_density() == 0.4
    assert c2.get_max_density() == 0.7142857142857143
    assert c3.get_max_density() == pytest.approx(0.33333333333333)


def test_sample_NormalFloatHyperparameter():
    hp = NormalFloatHyperparameter("nfhp", 0, 1)

    def actual_test():
        rs = np.random.RandomState(1)
        counts_per_bin = [0 for _ in range(11)]
        value = None
        for _ in range(100000):
            value = hp.sample(rs)
            index = min(max(int((np.round(value + 0.5)) + 5), 0), 9)
            counts_per_bin[index] += 1

        assert [0, 4, 138, 2113, 13394, 34104, 34282, 13683, 2136, 146, 0] == counts_per_bin

        assert isinstance(value, float)
        return counts_per_bin

    assert actual_test() == actual_test()


def test_sample_NormalFloatHyperparameter_with_bounds():
    hp = NormalFloatHyperparameter("nfhp", 0, 1, lower=-3, upper=3)

    # TODO: This should probably be a smaller amount
    def actual_test():
        rs = np.random.RandomState(1)
        counts_per_bin = [0 for _ in range(11)]
        value = None
        for _ in range(100000):
            value = hp.sample(rs)
            index = min(max(int((np.round(value + 0.5)) + 5), 0), 9)
            counts_per_bin[index] += 1

        assert [0, 0, 0, 2184, 13752, 34078, 34139, 13669, 2178, 0, 0] == counts_per_bin

        assert isinstance(value, float)
        return counts_per_bin

    assert actual_test() == actual_test()


def test_sample_BetaFloatHyperparameter():
    hp = BetaFloatHyperparameter("bfhp", alpha=8, beta=1.5, lower=-1, upper=10)

    def actual_test():
        rs = np.random.RandomState(1)
        counts_per_bin = [0 for _ in range(11)]
        value = None
        for _ in range(1000):
            value = hp.sample(rs)
            index = np.floor(value).astype(int)
            counts_per_bin[index] += 1

        assert [0, 2, 2, 4, 15, 39, 101, 193, 289, 355, 0] == counts_per_bin

        assert isinstance(value, float)
        return counts_per_bin

    assert actual_test() == actual_test()


def test_sample_UniformIntegerHyperparameter():
    # TODO: disentangle, actually test _sample and test sample on the
    # base class
    def sample(hp):
        rs = np.random.RandomState(1)
        counts_per_bin = [0 for _ in range(21)]
        values = []
        value = None
        for _ in range(100000):
            value = hp.sample(rs)
            values.append(value)
            index = int(float(value - hp.lower) / (hp.upper - hp.lower) * 20)
            counts_per_bin[index] += 1

        assert isinstance(value, int)
        return counts_per_bin

    # Quantized Uniform
    hp = UniformIntegerHyperparameter("uihp", 0, 10)

    counts_per_bin = sample(hp)
    for bin in counts_per_bin[::2]:
        assert 9302 > bin > 8700
    for bin in counts_per_bin[1::2]:
        assert bin == 0
    assert sample(hp) == sample(hp)


def test__sample_UniformIntegerHyperparameter():
    hp = UniformIntegerHyperparameter("uihp", 0, 10)
    values = []
    rs = np.random.RandomState(1)
    for _ in range(100):
        values.append(hp._sample(rs))
    assert len(np.unique(values)) == 11

    hp = UniformIntegerHyperparameter("uihp", 2, 12)
    values = []
    rs = np.random.RandomState(1)
    for _ in range(100):
        values.append(hp._sample(rs))
        assert hp._transform(values[-1]) >= 2
        assert hp._transform(values[-1]) <= 12
    assert len(np.unique(values)) == 11


def test_quantization_UniformIntegerHyperparameter():
    hp = UniformIntegerHyperparameter("uihp", 1, 100, q=3)
    rs = np.random.RandomState()

    sample_one = hp._sample(rs=rs, size=1)
    assert isinstance(sample_one, np.ndarray)
    assert sample_one.size == 1
    assert (hp._transform(sample_one) - 1) % 3 == 0
    assert hp._transform(sample_one) >= 1
    assert hp._transform(sample_one) <= 100

    sample_hundred = hp._sample(rs=rs, size=100)
    assert isinstance(sample_hundred, np.ndarray)
    assert sample_hundred.size == 100
    np.testing.assert_array_equal(
        [(hp._transform(val) - 1) % 3 for val in sample_hundred],
        np.zeros((100,), dtype=int),
    )
    samples_in_original_space = hp._transform(sample_hundred)
    for i in range(100):
        assert samples_in_original_space[i] >= 1
        assert samples_in_original_space[i] <= 100


def test_quantization_UniformIntegerHyperparameter_negative():
    hp = UniformIntegerHyperparameter("uihp", -2, 100, q=3)
    rs = np.random.RandomState()

    sample_one = hp._sample(rs=rs, size=1)
    assert isinstance(sample_one, np.ndarray)
    assert sample_one.size == 1
    assert (hp._transform(sample_one) + 2) % 3 == 0
    assert hp._transform(sample_one) >= -2
    assert hp._transform(sample_one) <= 100

    sample_hundred = hp._sample(rs=rs, size=100)
    assert isinstance(sample_hundred, np.ndarray)
    assert sample_hundred.size == 100
    np.testing.assert_array_equal(
        [(hp._transform(val) + 2) % 3 for val in sample_hundred],
        np.zeros((100,), dtype=int),
    )
    samples_in_original_space = hp._transform(sample_hundred)
    for i in range(100):
        assert samples_in_original_space[i] >= -2
        assert samples_in_original_space[i] <= 100


def test_illegal_quantization_UniformIntegerHyperparameter():
    with pytest.raises(
        ValueError,
        match=r"Upper bound \(4\) - lower bound \(1\) must be a multiple of q \(2\)",
    ):
        UniformIntegerHyperparameter("uihp", 1, 4, q=2)


def test_quantization_UniformFloatHyperparameter():
    hp = UniformFloatHyperparameter("ufhp", 1, 100, q=3)
    rs = np.random.RandomState()

    sample_one = hp._sample(rs=rs, size=1)
    assert isinstance(sample_one, np.ndarray)
    assert sample_one.size == 1
    assert (hp._transform(sample_one) - 1) % 3 == 0
    assert hp._transform(sample_one) >= 1
    assert hp._transform(sample_one) <= 100

    sample_hundred = hp._sample(rs=rs, size=100)
    assert isinstance(sample_hundred, np.ndarray)
    assert sample_hundred.size == 100
    np.testing.assert_array_equal(
        [(hp._transform(val) - 1) % 3 for val in sample_hundred],
        np.zeros((100,), dtype=int),
    )
    samples_in_original_space = hp._transform(sample_hundred)
    for i in range(100):
        assert samples_in_original_space[i] >= 1
        assert samples_in_original_space[i] <= 100


def test_quantization_UniformFloatHyperparameter_decimal_numbers():
    hp = UniformFloatHyperparameter("ufhp", 1.2, 3.6, q=0.2)
    rs = np.random.RandomState()

    sample_one = hp._sample(rs=rs, size=1)
    assert isinstance(sample_one, np.ndarray)
    assert sample_one.size == 1
    try:
        assert float(hp._transform(sample_one) + 1.2) % 0.2 == pytest.approx(0.0)
    except Exception:
        assert float(hp._transform(sample_one) + 1.2) % 0.2 == pytest.approx(0.2)
    assert hp._transform(sample_one) >= 1
    assert hp._transform(sample_one) <= 100


def test_quantization_UniformFloatHyperparameter_decimal_numbers_negative():
    hp = UniformFloatHyperparameter("ufhp", -1.2, 1.2, q=0.2)
    rs = np.random.RandomState()

    sample_one = hp._sample(rs=rs, size=1)
    assert isinstance(sample_one, np.ndarray)
    assert sample_one.size == 1
    try:
        assert float(hp._transform(sample_one) + 1.2) % 0.2 == pytest.approx(0.0)
    except Exception:
        assert float(hp._transform(sample_one) + 1.2) % 0.2 == pytest.approx(0.2)
    assert hp._transform(sample_one) >= -1.2
    assert hp._transform(sample_one) <= 1.2


def test_sample_NormalIntegerHyperparameter():
    def sample(hp):
        lower = -30
        upper = 30
        rs = np.random.RandomState(1)
        counts_per_bin = [0 for _ in range(21)]
        value = None
        for _ in range(100000):
            value = hp.sample(rs)
            sample = float(value)
            if sample < lower:
                sample = lower
            if sample > upper:
                sample = upper
            index = int((sample - lower) / (upper - lower) * 20)
            counts_per_bin[index] += 1

        assert isinstance(value, int)
        return counts_per_bin

    hp = NormalIntegerHyperparameter("nihp", 0, 10)
    assert sample(hp) == [
        305,
        422,
        835,
        1596,
        2682,
        4531,
        6572,
        8670,
        10649,
        11510,
        11854,
        11223,
        9309,
        7244,
        5155,
        3406,
        2025,
        1079,
        514,
        249,
        170,
    ]
    assert sample(hp) == sample(hp)


def test__sample_NormalIntegerHyperparameter():
    # mean zero, std 1
    hp = NormalIntegerHyperparameter("uihp", 0, 1)
    values = []
    rs = np.random.RandomState(1)
    for _ in range(100):
        values.append(hp._sample(rs))
    assert len(np.unique(values)) == 5


def test_sample_BetaIntegerHyperparameter():
    hp = BetaIntegerHyperparameter("bihp", alpha=4, beta=4, lower=0, upper=10)

    def actual_test():
        rs = np.random.RandomState(1)
        counts_per_bin = [0 for _ in range(11)]
        for _ in range(1000):
            value = hp.sample(rs)
            counts_per_bin[value] += 1

        # The chosen distribution is symmetric, so we expect to see a symmetry in the bins
        assert [1, 23, 82, 121, 174, 197, 174, 115, 86, 27, 0] == counts_per_bin

        return counts_per_bin

    assert actual_test() == actual_test()


def test_sample_CategoricalHyperparameter():
    hp = CategoricalHyperparameter("chp", [0, 2, "Bla", "Blub"])

    def actual_test():
        rs = np.random.RandomState(1)
        counts_per_bin: dict[str, int] = defaultdict(int)
        for _ in range(10000):
            value = hp.sample(rs)
            counts_per_bin[value] += 1

        assert {0: 2539, 2: 2451, "Bla": 2549, "Blub": 2461} == dict(counts_per_bin.items())
        return counts_per_bin

    assert actual_test() == actual_test()


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
            value = hp.sample(rs)
            counts_per_bin[value] += 1

        assert {0: 1003, 2: 2061, "Bla": 2994, "Blub": 3942} == dict(counts_per_bin.items())
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
    assert copy_hp.num_choices == orig_hp.num_choices
    assert copy_hp.probabilities == orig_hp.probabilities


def test_categorical_copy_without_weights():
    orig_hp = CategoricalHyperparameter(name="param", choices=[1, 2, 3], default_value=2)
    copy_hp = copy.copy(orig_hp)

    assert copy_hp.name == orig_hp.name
    assert copy_hp.choices == orig_hp.choices
    assert copy_hp.default_value == orig_hp.default_value
    assert copy_hp.num_choices == orig_hp.num_choices
    assert copy_hp.probabilities == (0.3333333333333333, 0.3333333333333333, 0.3333333333333333)
    assert orig_hp.probabilities == (0.3333333333333333, 0.3333333333333333, 0.3333333333333333)


def test_categorical_with_weights():
    rs = np.random.RandomState()

    cat_hp_str = CategoricalHyperparameter(
        name="param",
        choices=["A", "B", "C"],
        default_value="A",
        weights=[0.1, 0.6, 0.3],
    )
    for _ in range(1000):
        assert cat_hp_str.sample(rs) in ["A", "B", "C"]

    cat_hp_int = CategoricalHyperparameter(
        name="param",
        choices=[1, 2, 3],
        default_value=2,
        weights=[0.1, 0.3, 0.6],
    )
    for _ in range(1000):
        assert cat_hp_int.sample(rs) in [1, 3, 2]

    cat_hp_float = CategoricalHyperparameter(
        name="param",
        choices=[-0.1, 0.0, 0.3],
        default_value=0.3,
        weights=[10, 60, 30],
    )
    for _ in range(1000):
        assert cat_hp_float.sample(rs) in [-0.1, 0.0, 0.3]


def test_categorical_with_some_zero_weights():
    # zero weights are okay as long as there is at least one strictly positive weight

    rs = np.random.RandomState()

    cat_hp_str = CategoricalHyperparameter(
        name="param",
        choices=["A", "B", "C"],
        default_value="A",
        weights=[0.1, 0.0, 0.3],
    )
    for _ in range(1000):
        assert cat_hp_str.sample(rs) in ["A", "C"]
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
    for _ in range(1000):
        assert cat_hp_int.sample(rs) in [1, 2]
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
    for _ in range(1000):
        assert cat_hp_float.sample(rs) in [0.0, 0.3]
    np.testing.assert_almost_equal(
        actual=cat_hp_float.probabilities,
        desired=[0.00, 0.6667, 0.3333],
        decimal=3,
    )


def test_categorical_with_all_zero_weights():
    with pytest.raises(ValueError, match="At least one weight has to be strictly positive."):
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
            choices={"A", "B", "C"},
            default_value="A",
        )

    with pytest.raises(TypeError, match="Using a set of weights is prohibited."):
        CategoricalHyperparameter(
            name="param",
            choices=["A", "B", "C"],
            default_value="A",
            weights={0.2, 0.6, 0.8},
        )


def test_log_space_conversion():
    lower, upper = 1e-5, 1e5
    hyper = UniformFloatHyperparameter("test", lower=lower, upper=upper, log=True)
    assert hyper.is_legal(hyper._transform(1.0))

    lower, upper = 1e-10, 1e10
    hyper = UniformFloatHyperparameter("test", lower=lower, upper=upper, log=True)
    assert hyper.is_legal(hyper._transform(1.0))


def test_ordinal_attributes_accessible():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.name == "temp"
    assert f1.sequence == ("freezing", "cold", "warm", "hot")
    assert f1.num_elements == 4
    assert f1.default_value == "freezing"
    assert f1.normalized_default_value == 0


def test_ordinal_is_legal():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.is_legal("warm")
    assert f1.is_legal("freezing")
    assert not f1.is_legal("chill")
    assert not f1.is_legal(2.5)
    assert not f1.is_legal("3")

    # Test is legal vector
    assert f1.is_legal_vector(1.0)
    assert f1.is_legal_vector(0.0)
    assert f1.is_legal_vector(0)
    assert f1.is_legal_vector(3)
    assert not f1.is_legal_vector(-0.1)
    with pytest.raises(TypeError):
        f1.is_legal_vector("Hahaha")


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
    assert f1.get_neighbors(0, rs=None) == [1]
    assert f1.get_neighbors(1, rs=None) == [0, 2]
    assert f1.get_neighbors(3, rs=None) == [2]
    assert f1.get_neighbors("hot", transform=True, rs=None) == ["warm"]
    assert f1.get_neighbors("cold", transform=True, rs=None) == ["freezing", "warm"]


def test_get_num_neighbors():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.get_num_neighbors("freezing") == 1
    assert f1.get_num_neighbors("hot") == 1
    assert f1.get_num_neighbors("cold") == 2


def test_ordinal_get_size():
    f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert f1.get_size() == 4


def test_ordinal_pdf():
    c1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    point_1 = np.array(["freezing"])
    point_2 = np.array(["warm"])
    array_1 = np.array(["freezing", "warm"])

    wrong_shape_1 = np.array([["freezing"]])
    wrong_shape_2 = np.array(["freezing", "warm"]).reshape(1, -1)
    wrong_shape_3 = np.array(["freezing", "warm"]).reshape(-1, 1)

    assert c1.pdf(point_1)[0] == 0.25
    assert c1.pdf(point_2)[0] == 0.25

    array_results = c1.pdf(array_1)
    expected_results = np.array([0.25, 0.25])
    assert array_results.shape == expected_results.shape
    for res, exp_res in zip(array_results, expected_results):
        assert res == exp_res

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1.pdf(0.2)
    with pytest.raises(TypeError):
        c1.pdf("pdf")
    with pytest.raises(TypeError):
        c1.pdf("one")
    with pytest.raises(ValueError):
        c1.pdf(np.array(["zero"]))

    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_1)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_2)
    with pytest.raises(ValueError, match="Method pdf expects a one-dimensional numpy array"):
        c1.pdf(wrong_shape_3)


def test_ordinal__pdf():
    c1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    point_1 = np.array(["freezing"])
    point_2 = np.array(["warm"])
    array_1 = np.array(["freezing", "warm"])
    assert c1._pdf(point_1)[0] == 0.25
    assert c1._pdf(point_2)[0] == 0.25

    array_results = c1._pdf(array_1)
    expected_results = np.array([0.25, 0.25])
    assert array_results.shape == expected_results.shape
    for res, exp_res in zip(array_results, expected_results):
        assert res == exp_res

    # pdf must take a numpy array
    with pytest.raises(TypeError):
        c1._pdf(0.2)
    with pytest.raises(TypeError):
        c1._pdf("pdf")
    with pytest.raises(TypeError):
        c1._pdf("one")
    with pytest.raises(ValueError):
        c1._pdf(np.array(["zero"]))


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
    assert f1.rvs(random_state=100) == pytest.approx(f1.rvs(random_state=np.random.RandomState(100)))
    f1.rvs(random_state=np.random)
    f1.rvs(random_state=np.random.default_rng(1))
    with pytest.raises(ValueError):
        f1.rvs(1, "a")


def test_hyperparam_representation():
    # Float
    f1 = UniformFloatHyperparameter("param", 1, 100, log=True)
    assert repr(f1) == "param, Type: UniformFloat, Range: [1.0, 100.0], Default: 10.0, on log-scale"
    f2 = NormalFloatHyperparameter("param", 8, 99.1, log=False)
    assert repr(f2) == "param, Type: NormalFloat, Mu: 8.0 Sigma: 99.1, Default: 8.0"
    f3 = NormalFloatHyperparameter("param", 8, 99.1, log=False, lower=1, upper=16)
    assert (
        repr(f3)
        == "param, Type: NormalFloat, Mu: 8.0 Sigma: 99.1, Range: [1.0, 16.0], Default: 8.0"
    )
    i1 = UniformIntegerHyperparameter("param", 0, 100)
    assert repr(i1) == "param, Type: UniformInteger, Range: [0, 100], Default: 50"
    i2 = NormalIntegerHyperparameter("param", 5, 8)
    assert repr(i2) == "param, Type: NormalInteger, Mu: 5 Sigma: 8, Default: 5"
    i3 = NormalIntegerHyperparameter("param", 5, 8, lower=1, upper=10)
    assert repr(i3) == "param, Type: NormalInteger, Mu: 5 Sigma: 8, Range: [1, 10], Default: 5"
    o1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
    assert (
        repr(o1) == "temp, Type: Ordinal, Sequence: {freezing, cold, warm, hot}, Default: freezing"
    )
    c1 = CategoricalHyperparameter("param", [True, False])
    assert repr(c1) == "param, Type: Categorical, Choices: {True, False}, Default: True"
