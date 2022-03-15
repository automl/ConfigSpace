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

from collections import defaultdict
import copy
import unittest
import pytest

import numpy as np

from ConfigSpace.hyperparameters import (Constant,
                                         UniformFloatHyperparameter, NormalFloatHyperparameter,
                                         BetaFloatHyperparameter, UniformIntegerHyperparameter,
                                         NormalIntegerHyperparameter, BetaIntegerHyperparameter,
                                         CategoricalHyperparameter, OrdinalHyperparameter)


class TestHyperparameters(unittest.TestCase):

    def setUp(self):
        self.meta_data = {'additional': 'meta-data',
                          'useful': 'for integrations',
                          'input_id': 42}

    def test_constant(self):
        # Test construction
        c1 = Constant("value", 1)
        c2 = Constant("value", 1)
        c3 = Constant("value", 2)
        c4 = Constant("valuee", 1)
        c5 = Constant("valueee", 2)

        # Test attributes are accessible
        self.assertEqual(c5.name, "valueee")
        self.assertEqual(c5.value, 2)

        # Test the representation
        self.assertEqual("value, Type: Constant, Value: 1", c1.__repr__())

        # Test the equals operator (and the ne operator in the last line)
        self.assertFalse(c1 == 1)
        self.assertEqual(c1, c2)
        self.assertFalse(c1 == c3)
        self.assertFalse(c1 == c4)
        self.assertTrue(c1 != c5)

        # Test that only string, integers and floats are allowed
        self.assertRaises(TypeError, Constant, "value", dict())
        self.assertRaises(TypeError, Constant, "value", None)
        self.assertRaises(TypeError, Constant, "value", True)

        # Test that only string names are allowed
        self.assertRaises(TypeError, Constant, 1, "value")
        self.assertRaises(TypeError, Constant, dict(), "value")
        self.assertRaises(TypeError, Constant, None, "value")
        self.assertRaises(TypeError, Constant, True, "value")

        # test that meta-data is stored correctly
        c1_meta = Constant("value", 1, dict(self.meta_data))
        self.assertEqual(c1_meta.meta, self.meta_data)
        # Test getting the size
        for constant in (c1, c2, c3, c4, c5, c1_meta):
            self.assertEqual(constant.get_size(), 1)

    def test_constant_pdf(self):
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

        self.assertEqual(c1.pdf(point_1), np.array([1.0]))
        self.assertEqual(c2.pdf(point_2), np.array([1.0]))
        self.assertEqual(c1.pdf(point_2), np.array([0.0]))
        self.assertEqual(c2.pdf(point_1), np.array([0.0]))

        self.assertEqual(tuple(c1.pdf(array_1)), tuple(np.array([1.0, 1.0])))
        self.assertEqual(tuple(c2.pdf(array_2)), tuple(np.array([1.0, 1.0])))
        self.assertEqual(tuple(c1.pdf(array_2)), tuple(np.array([0.0, 0.0])))
        self.assertEqual(tuple(c1.pdf(array_3)), tuple(np.array([1.0, 0.0])))

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        # and it must be one-dimensional
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_1)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_2)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_3)

    def test_constant__pdf(self):
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

        self.assertEqual(c1._pdf(point_1), np.array([1.0]))
        self.assertEqual(c2._pdf(point_2), np.array([1.0]))
        self.assertEqual(c1._pdf(point_2), np.array([0.0]))
        self.assertEqual(c2._pdf(point_1), np.array([0.0]))

        # Only (N, ) numpy arrays are seamlessly converted to tuples
        # so the __eq__ method works as intended
        self.assertEqual(tuple(c1._pdf(array_1)), tuple(np.array([1.0, 1.0])))
        self.assertEqual(tuple(c2._pdf(array_2)), tuple(np.array([1.0, 1.0])))
        self.assertEqual(tuple(c1._pdf(array_2)), tuple(np.array([0.0, 0.0])))
        self.assertEqual(tuple(c1._pdf(array_3)), tuple(np.array([1.0, 0.0])))

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1._pdf(0.2)
        with self.assertRaises(TypeError):
            c1._pdf('pdf')

        # Simply check that it runs, since _pdf does not restrict shape (only public method does)
        c1._pdf(accepted_shape_1)
        c1._pdf(accepted_shape_2)
        c1._pdf(accepted_shape_3)

    def test_constant_get_max_density(self):
        c1 = Constant("valuee", 1)
        c2 = Constant("valueee", -2)
        self.assertEqual(c1.get_max_density(), 1.0)
        self.assertEqual(c2.get_max_density(), 1.0)

    def test_uniformfloat(self):
        # TODO test non-equality
        # TODO test sampling from a log-distribution which has a negative
        # lower value!
        f1 = UniformFloatHyperparameter("param", 0, 10)
        f1_ = UniformFloatHyperparameter("param", 0, 10)
        self.assertEqual(f1, f1_)
        self.assertEqual("param, Type: UniformFloat, Range: [0.0, 10.0], "
                         "Default: 5.0",
                         str(f1))

        # Test attributes are accessible
        self.assertEqual(f1.name, "param")
        self.assertAlmostEqual(f1.lower, 0.0)
        self.assertAlmostEqual(f1.upper, 10.0)
        self.assertEqual(f1.q, None)
        self.assertEqual(f1.log, False)
        self.assertAlmostEqual(f1.default_value, 5.0)
        self.assertAlmostEqual(f1.normalized_default_value, 0.5)

        f2 = UniformFloatHyperparameter("param", 0, 10, q=0.1)
        f2_ = UniformFloatHyperparameter("param", 0, 10, q=0.1)
        self.assertEqual(f2, f2_)
        self.assertEqual("param, Type: UniformFloat, Range: [0.0, 10.0], "
                         "Default: 5.0, Q: 0.1", str(f2))

        f3 = UniformFloatHyperparameter("param", 0.00001, 10, log=True)
        f3_ = UniformFloatHyperparameter("param", 0.00001, 10, log=True)
        self.assertEqual(f3, f3_)
        self.assertEqual(
            "param, Type: UniformFloat, Range: [1e-05, 10.0], Default: 0.01, "
            "on log-scale", str(f3))

        f4 = UniformFloatHyperparameter("param", 0, 10, default_value=1.0)
        f4_ = UniformFloatHyperparameter("param", 0, 10, default_value=1.0)
        # Test that a int default is converted to float
        f4__ = UniformFloatHyperparameter("param", 0, 10, default_value=1)
        self.assertEqual(f4, f4_)
        self.assertEqual(type(f4.default_value), type(f4__.default_value))
        self.assertEqual(
            "param, Type: UniformFloat, Range: [0.0, 10.0], Default: 1.0",
            str(f4))

        f5 = UniformFloatHyperparameter("param", 0.1, 10, q=0.1, log=True,
                                        default_value=1.0)
        f5_ = UniformFloatHyperparameter("param", 0.1, 10, q=0.1, log=True,
                                         default_value=1.0)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: UniformFloat, Range: [0.1, 10.0], Default: 1.0, "
            "on log-scale, Q: 0.1", str(f5))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

        # test that meta-data is stored correctly
        f_meta = UniformFloatHyperparameter("param", 0.1, 10, q=0.1, log=True,
                                            default_value=1.0, meta=dict(self.meta_data))
        self.assertEqual(f_meta.meta, self.meta_data)

        # Test get_size
        for float_hp in (f1, f3, f4):
            self.assertTrue(np.isinf(float_hp.get_size()))
        self.assertEqual(f2.get_size(), 101)
        self.assertEqual(f5.get_size(), 100)

    def test_uniformfloat_to_integer(self):
        f1 = UniformFloatHyperparameter("param", 1, 10, q=0.1, log=True)
        with pytest.warns(UserWarning, match="Setting quantization < 1 for Integer "
                                             "Hyperparameter 'param' has no effect"):
            f2 = f1.to_integer()
        # TODO is this a useful rounding?
        # TODO should there be any rounding, if e.g. lower=0.1
        self.assertEqual("param, Type: UniformInteger, Range: [1, 10], "
                         "Default: 3, on log-scale", str(f2))

    def test_uniformfloat_is_legal(self):
        lower = 0.1
        upper = 10
        f1 = UniformFloatHyperparameter("param", lower, upper, q=0.1, log=True)

        self.assertTrue(f1.is_legal(3.0))
        self.assertTrue(f1.is_legal(3))
        self.assertFalse(f1.is_legal(-0.1))
        self.assertFalse(f1.is_legal(10.1))
        self.assertFalse(f1.is_legal("AAA"))
        self.assertFalse(f1.is_legal(dict()))

        # Test legal vector values
        self.assertTrue(f1.is_legal_vector(1.0))
        self.assertTrue(f1.is_legal_vector(0.0))
        self.assertTrue(f1.is_legal_vector(0))
        self.assertTrue(f1.is_legal_vector(0.3))
        self.assertFalse(f1.is_legal_vector(-0.1))
        self.assertFalse(f1.is_legal_vector(1.1))
        self.assertRaises(TypeError, f1.is_legal_vector, "Hahaha")

    def test_uniformfloat_illegal_bounds(self):
        self.assertRaisesRegex(
            ValueError,
            r"Negative lower bound \(0.000000\) for log-scale hyperparameter "
            r"param is forbidden.", UniformFloatHyperparameter, "param", 0, 10,
            q=0.1, log=True)

        self.assertRaisesRegex(
            ValueError, "Upper bound 0.000000 must be larger than lower bound "
            "1.000000 for hyperparameter param", UniformFloatHyperparameter,
            "param", 1, 0)

    def test_uniformfloat_pdf(self):
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

        self.assertAlmostEqual(c1.pdf(point_1)[0], 0.1)
        self.assertAlmostEqual(c2.pdf(point_2)[0], 4.539992976248485e-05)
        self.assertAlmostEqual(c1.pdf(point_1)[0], 0.1)
        self.assertAlmostEqual(c2.pdf(point_2)[0], 4.539992976248485e-05)
        self.assertAlmostEqual(c3.pdf(point_3)[0], 2.0)

        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        # since inverse_transform pulls everything into range,
        # even points outside get evaluated in range
        self.assertAlmostEqual(c1.pdf(point_outside_range)[0], 0.1)
        self.assertAlmostEqual(c2.pdf(point_outside_range_log)[0], 4.539992976248485e-05)

        # this, however, is a negative value on a log param, which cannot be pulled into range
        with pytest.warns(RuntimeWarning, match='invalid value encountered in log'):
            self.assertEqual(c2.pdf(point_outside_range)[0], 0.0)

        array_results = c1.pdf(array_1)
        array_results_log = c2.pdf(array_1)
        expected_results = np.array([0.1, 0.1, 0.1])
        expected_log_results = np.array(
            [4.539992976248485e-05, 4.539992976248485e-05, 4.539992976248485e-05])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_log_results.shape)
        for res, log_res, exp_res, exp_log_res in zip(array_results, array_results_log,
                                                      expected_results, expected_log_results):
            self.assertAlmostEqual(res, exp_res)
            self.assertAlmostEqual(log_res, exp_log_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_1)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_2)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_3)

    def test_uniformfloat__pdf(self):
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

        self.assertAlmostEqual(c1._pdf(point_1)[0], 0.1)
        self.assertAlmostEqual(c2._pdf(point_2)[0], 4.539992976248485e-05)
        self.assertAlmostEqual(c1._pdf(point_1)[0], 0.1)
        self.assertAlmostEqual(c2._pdf(point_2)[0], 4.539992976248485e-05)
        self.assertAlmostEqual(c3._pdf(point_3)[0], 2.0)

        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        # since inverse_transform pulls everything into range,
        # even points outside get evaluated in range
        self.assertAlmostEqual(c1._pdf(point_outside_range_1)[0], 0.0)
        self.assertAlmostEqual(c2._pdf(point_outside_range_2)[0], 0.0)
        self.assertAlmostEqual(c1._pdf(point_outside_range_2)[0], 0.0)
        self.assertAlmostEqual(c2._pdf(point_outside_range_1)[0], 0.0)

        array_results = c1._pdf(array_1)
        array_results_log = c2._pdf(array_1)
        expected_results = np.array([0.1, 0.1, 0])
        expected_log_results = np.array([4.539992976248485e-05, 4.539992976248485e-05, 0.0])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_log_results.shape)
        for res, log_res, exp_res, exp_log_res in zip(array_results, array_results_log,
                                                      expected_results, expected_log_results):
            self.assertAlmostEqual(res, exp_res)
            self.assertAlmostEqual(log_res, exp_log_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1._pdf(0.2)
        with self.assertRaises(TypeError):
            c1._pdf('pdf')

        # Simply check that it runs, since _pdf does not restrict shape (only public method does)
        c1._pdf(accepted_shape_1)
        c1._pdf(accepted_shape_2)
        c1._pdf(accepted_shape_3)

    def test_uniformfloat_get_max_density(self):
        c1 = UniformFloatHyperparameter("param", lower=0, upper=10)
        c2 = UniformFloatHyperparameter("logparam", lower=np.exp(0), upper=np.exp(10), log=True)
        c3 = UniformFloatHyperparameter("param", lower=0, upper=0.5)
        self.assertEqual(c1.get_max_density(), 0.1)
        self.assertAlmostEqual(c2.get_max_density(), 4.539992976248485e-05)
        self.assertEqual(c3.get_max_density(), 2)

    def test_normalfloat(self):
        # TODO test non-equality
        f1 = NormalFloatHyperparameter("param", 0.5, 10.5)
        f1_ = NormalFloatHyperparameter("param", 0.5, 10.5)
        self.assertEqual(f1, f1_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.5 Sigma: 10.5, Default: 0.5",
            str(f1))

        # Due to seemingly different numbers with x86_64 and i686 architectures
        # we got these numbers, where last two are slightly different
        # [
        #   5.715498606617943, -0.9517751622974389,
        #   7.3007296500572725, 16.49181349228427
        # ]
        # They are equal up to 14 decimal places
        expected = [
            5.715498606617943, -0.9517751622974389,
            7.300729650057271, 16.491813492284265
        ]
        np.testing.assert_almost_equal(
            f1.get_neighbors(0.5, rs=np.random.RandomState(42)),
            expected,
            decimal=14
        )

        # Test attributes are accessible
        self.assertEqual(f1.name, "param")
        self.assertAlmostEqual(f1.mu, 0.5)
        self.assertAlmostEqual(f1.sigma, 10.5)
        self.assertAlmostEqual(f1.q, None)
        self.assertEqual(f1.log, False)
        self.assertAlmostEqual(f1.default_value, 0.5)
        self.assertAlmostEqual(f1.normalized_default_value, 0.5)

        # Test copy
        copy_f1 = copy.copy(f1)

        self.assertEqual(copy_f1.name, f1.name)
        self.assertEqual(copy_f1.mu, f1.mu)
        self.assertEqual(copy_f1.sigma, f1.sigma)
        self.assertEqual(copy_f1.default_value, f1.default_value)

        f2 = NormalFloatHyperparameter("param", 0, 10, q=0.1)
        f2_ = NormalFloatHyperparameter("param", 0, 10, q=0.1)
        self.assertEqual(f2, f2_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 0.0, "
            "Q: 0.1", str(f2))

        f3 = NormalFloatHyperparameter("param", 0, 10, log=True)
        f3_ = NormalFloatHyperparameter("param", 0, 10, log=True)
        self.assertEqual(f3, f3_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 1.0, "
            "on log-scale", str(f3))

        f4 = NormalFloatHyperparameter("param", 0, 10, default_value=1.0)
        f4_ = NormalFloatHyperparameter("param", 0, 10, default_value=1.0)
        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 1.0",
            str(f4))

        f5 = NormalFloatHyperparameter("param", 0, 10, default_value=3.0,
                                       q=0.1, log=True)
        f5_ = NormalFloatHyperparameter("param", 0, 10, default_value=3.0,
                                        q=0.1, log=True)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 3.0, "
            "on log-scale, Q: 0.1", str(f5))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

        with pytest.raises(ValueError):
            f6 = NormalFloatHyperparameter("param", 5, 10, lower=0.1, upper=0.1,
                                           default_value=5.0, q=0.1, log=True)

        with pytest.raises(ValueError):
            f6 = NormalFloatHyperparameter("param", 5, 10, lower=0.1, default_value=5.0,
                                           q=0.1, log=True)

        with pytest.raises(ValueError):
            f6 = NormalFloatHyperparameter("param", 5, 10, upper=0.1, default_value=5.0,
                                           q=0.1, log=True)

        f6 = NormalFloatHyperparameter("param", 5, 10, lower=0.1, upper=10,
                                       default_value=5.0, q=0.1, log=True)
        f6_ = NormalFloatHyperparameter("param", 5, 10, lower=0.1, upper=10,
                                        default_value=5.0, q=0.1, log=True)
        self.assertEqual(f6, f6_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 5.0 Sigma: 10.0, Range: [0.1, 10.0], " +
            "Default: 5.0, on log-scale, Q: 0.1", str(f6))

        # Due to seemingly different numbers with x86_64 and i686 architectures
        # we got these numbers, where the first one is slightly different
        # [9.967141530112325, 3.6173569882881536, 10.0, 10.0]
        # They are equal up to 14 decimal places
        expected = [9.967141530112327, 3.6173569882881536, 10.0, 10.0]
        np.testing.assert_almost_equal(
            f6.get_neighbors(5, rs=np.random.RandomState(42)),
            expected,
            decimal=14
        )

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

        # test that meta-data is stored correctly
        f_meta = NormalFloatHyperparameter("param", 0.1, 10, q=0.1, log=True,
                                           default_value=1.0, meta=dict(self.meta_data))
        self.assertEqual(f_meta.meta, self.meta_data)

        # Test get_size
        for float_hp in (f1, f2, f3, f4, f5):
            self.assertTrue(np.isinf(float_hp.get_size()))
        self.assertEqual(f6.get_size(), 100)

    def test_normalfloat_to_uniformfloat(self):
        f1 = NormalFloatHyperparameter("param", 0, 10, q=0.1)
        f1_expected = UniformFloatHyperparameter("param", -30, 30, q=0.1)
        f1_actual = f1.to_uniform()
        self.assertEqual(f1_expected, f1_actual)

        f2 = NormalFloatHyperparameter("param", 0, 10, lower=-20, upper=20, q=0.1)
        f2_expected = UniformFloatHyperparameter("param", -20, 20, q=0.1)
        f2_actual = f2.to_uniform()
        self.assertEqual(f2_expected, f2_actual)

    def test_normalfloat_is_legal(self):
        f1 = NormalFloatHyperparameter("param", 0, 10)
        self.assertTrue(f1.is_legal(3.0))
        self.assertTrue(f1.is_legal(2))
        self.assertFalse(f1.is_legal("Hahaha"))

        # Test legal vector values
        self.assertTrue(f1.is_legal_vector(1.0))
        self.assertTrue(f1.is_legal_vector(0.0))
        self.assertTrue(f1.is_legal_vector(0))
        self.assertTrue(f1.is_legal_vector(0.3))
        self.assertTrue(f1.is_legal_vector(-0.1))
        self.assertTrue(f1.is_legal_vector(1.1))
        self.assertRaises(TypeError, f1.is_legal_vector, "Hahaha")

    def test_normalfloat_to_integer(self):
        f1 = NormalFloatHyperparameter("param", 0, 10)
        f2_expected = NormalIntegerHyperparameter("param", 0, 10)
        f2_actual = f1.to_integer()
        self.assertEqual(f2_expected, f2_actual)

    def test_normalfloat_pdf(self):
        c1 = NormalFloatHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
        c2 = NormalFloatHyperparameter("logparam", lower=np.exp(
            0), upper=np.exp(10), mu=3, sigma=2, log=True)
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

        self.assertAlmostEqual(c1.pdf(point_1)[0], 0.2138045617479014)
        self.assertAlmostEqual(c2.pdf(point_1_log)[0], 0.2138045617479014)
        self.assertAlmostEqual(c1.pdf(point_2)[0], 0.000467695579850518)
        self.assertAlmostEqual(c2.pdf(point_2_log)[0], 0.000467695579850518)
        self.assertAlmostEqual(c3.pdf(point_3)[0], 25.932522722334905)
        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        self.assertEqual(c1.pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c1.pdf(point_outside_range_2)[0], 0.0)
        self.assertEqual(c2.pdf(point_outside_range_1_log)[0], 0.0)
        self.assertEqual(c2.pdf(point_outside_range_2_log)[0], 0.0)

        array_results = c1.pdf(array_1)
        array_results_log = c2.pdf(array_1_log)
        expected_results = np.array([0.2138045617479014, 0.0004676955798505186, 0])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results.shape)
        for res, log_res, exp_res in zip(array_results, array_results, expected_results):
            self.assertAlmostEqual(res, exp_res)
            self.assertAlmostEqual(log_res, exp_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        c_nobounds = NormalFloatHyperparameter("param", mu=3, sigma=2)
        self.assertAlmostEqual(c_nobounds.pdf(np.array([2])), 0.17603266338214976)

        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_1)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_2)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_3)

    def test_normalfloat__pdf(self):
        c1 = NormalFloatHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
        c2 = NormalFloatHyperparameter("logparam", lower=np.exp(
            0), upper=np.exp(10), mu=3, sigma=2, log=True)
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

        self.assertAlmostEqual(c1._pdf(point_1)[0], 0.2138045617479014)
        self.assertAlmostEqual(c2._pdf(point_1)[0], 0.2138045617479014)
        self.assertAlmostEqual(c1._pdf(point_2)[0], 0.000467695579850518)
        self.assertAlmostEqual(c2._pdf(point_2)[0], 0.000467695579850518)
        self.assertAlmostEqual(c3._pdf(point_3)[0], 25.932522722334905)
        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        self.assertEqual(c1._pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c1._pdf(point_outside_range_2)[0], 0.0)
        self.assertEqual(c2._pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c2._pdf(point_outside_range_2)[0], 0.0)

        array_results = c1.pdf(array_1)
        array_results_log = c2.pdf(array_1)
        expected_results = np.array([0.2138045617479014, 0.0004676955798505186, 0])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results.shape)
        for res, log_res, exp_res in zip(array_results, array_results, expected_results):
            self.assertAlmostEqual(res, exp_res)
            self.assertAlmostEqual(log_res, exp_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        c_nobounds = NormalFloatHyperparameter("param", mu=3, sigma=2)
        self.assertAlmostEqual(c_nobounds.pdf(np.array([2])), 0.17603266338214976)

        # Simply check that it runs, since _pdf does not restrict shape (only public method does)
        c1._pdf(accepted_shape_1)
        c1._pdf(accepted_shape_2)
        c1._pdf(accepted_shape_3)

    def test_normalfloat_get_max_density(self):
        c1 = NormalFloatHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
        c2 = NormalFloatHyperparameter("logparam", lower=np.exp(
            0), upper=np.exp(10), mu=3, sigma=2, log=True)
        c3 = NormalFloatHyperparameter("param", lower=0, upper=0.5, mu=-1, sigma=0.2)
        self.assertEqual(c1.get_max_density(), 0.2138045617479014)
        self.assertAlmostEqual(c2.get_max_density(), 0.2138045617479014)
        self.assertAlmostEqual(c3.get_max_density(), 25.932522722334905)

    def test_betafloat(self):
        # TODO test non-equality
        f1 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.0)
        f1_ = BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=3, beta=1)
        self.assertEqual(f1, f1_)
        self.assertEqual(
            "param, Type: BetaFloat, Alpha: 3.0 Beta: 1.0, Range: [-2.0, 2.0], Default: 2.0",
            str(f1_))

        u1 = UniformFloatHyperparameter("param", lower=0.0, upper=1.0)
        b1 = BetaFloatHyperparameter("param", lower=0.0, upper=1.0, alpha=3.0, beta=1.0)

        # with identical domains, beta and uniform should sample the same points
        self.assertEqual(u1.get_neighbors(0.5, rs=np.random.RandomState(42)),
                         b1.get_neighbors(0.5, rs=np.random.RandomState(42)),
                         )
        # Test copy
        copy_f1 = copy.copy(f1)
        self.assertEqual(copy_f1.name, f1.name)

        f2 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.0, q=0.1)
        f2_ = BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=3, beta=1, q=0.1)
        self.assertEqual(f2, f2_)

        self.assertEqual(
            "param, Type: BetaFloat, Alpha: 3.0 Beta: 1.0, Range: [-2.0, 2.0], Default: 2.0, "
            "Q: 0.1", str(f2))

        f3 = BetaFloatHyperparameter("param", lower=10**(-5), upper=10.0,
                                     alpha=6.0, beta=2.0, log=True)
        f3_ = BetaFloatHyperparameter("param", lower=10**(-5),
                                      upper=10.0, alpha=6.0, beta=2.0, log=True)
        self.assertEqual(f3, f3_)
        self.assertEqual(
            "param, Type: BetaFloat, Alpha: 6.0 Beta: 2.0, Range: [1e-05, 10.0], Default: 1.0, "
            "on log-scale", str(f3))

        f4 = BetaFloatHyperparameter("param", lower=1, upper=1000.0,
                                     alpha=2.0, beta=2.0, log=True, q=1.0)
        f4_ = BetaFloatHyperparameter("param", lower=1, upper=1000.0,
                                      alpha=2.0, beta=2.0, log=True, q=1.0)

        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: BetaFloat, Alpha: 2.0 Beta: 2.0, Range: [1.0, 1000.0], Default: 32.0, "
            "on log-scale, Q: 1.0", str(f4))

        # test that meta-data is stored correctly
        f_meta = BetaFloatHyperparameter(
            "param", lower=1, upper=10.0, alpha=3.0, beta=2.0, log=False, meta=dict(self.meta_data))
        self.assertEqual(f_meta.meta, self.meta_data)

        with self.assertWarnsRegex(UserWarning, 'Logscale and quantization together results in '
                                   'incorrect default values. We recommend specifying a default '
                                   'value manually for this specific case.'):
            BetaFloatHyperparameter(
                "param", lower=1, upper=100.0, alpha=3.0, beta=2.0, log=True, q=1)

    def test_betafloat_dist_parameters(self):
        # This one should just be created without raising an error - corresponds to uniform dist.
        BetaFloatHyperparameter("param", lower=0, upper=10.0, alpha=1, beta=1)

        # This one is not permitted as the co-domain is not finite
        with self.assertRaises(ValueError):
            BetaFloatHyperparameter("param", lower=0, upper=100, alpha=0.99, beta=0.99)
        # And these parameters do not define a proper beta distribution whatsoever
        with self.assertRaises(ValueError):
            BetaFloatHyperparameter("param", lower=0, upper=100, alpha=-0.1, beta=-0.1)

        # test parameters that do not create a legit beta distribution, one at a time
        with self.assertRaises(ValueError):
            BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=-11, beta=5)
        with self.assertRaises(ValueError):
            BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=5, beta=-11)

        # test parameters that do not yield a finite co-domain, one at a time
        with self.assertRaises(ValueError):
            BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=0.5, beta=11)
        with self.assertRaises(ValueError):
            BetaFloatHyperparameter("param", lower=-2, upper=2, alpha=11, beta=0.5)

    def test_betafloat_default_value(self):
        # should default to the maximal value in the search space
        f_max = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.0)
        self.assertAlmostEqual(f_max.default_value, 2.0)
        self.assertAlmostEqual(f_max.normalized_default_value, 1.0)

        f_max_log = BetaFloatHyperparameter(
            "param", lower=1.0, upper=10.0, alpha=3.0, beta=1.0, log=True)
        self.assertAlmostEqual(f_max_log.default_value, 10.0)
        self.assertAlmostEqual(f_max_log.normalized_default_value, 1.0)

        # should default to the minimal value in the search space
        f_min = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=1.0, beta=1.5)
        self.assertAlmostEqual(f_min.default_value, -2.0)
        self.assertAlmostEqual(f_min.normalized_default_value, 0.0)

        f_min_log = BetaFloatHyperparameter(
            "param", lower=1.0, upper=10.0, alpha=1.0, beta=1.5, log=True)
        self.assertAlmostEqual(f_min_log.default_value, 1.0)
        self.assertAlmostEqual(f_min_log.normalized_default_value, 0.0)

        # Symmeric, should default to the middle
        f_symm = BetaFloatHyperparameter("param", lower=5, upper=9, alpha=4.6, beta=4.6)
        self.assertAlmostEqual(f_symm.default_value, 7)
        self.assertAlmostEqual(f_symm.normalized_default_value, 0.5)

        # This should yield a value that's halfway towards the max in logspace
        f_symm_log = BetaFloatHyperparameter(
            "param", lower=1, upper=np.exp(10), alpha=4.6, beta=4.6, log=True)
        self.assertAlmostEqual(f_symm_log.default_value, np.exp(5))
        self.assertAlmostEqual(f_symm_log.normalized_default_value, 0.5)

        # Uniform, should also default to the middle
        f_unif = BetaFloatHyperparameter("param", lower=2.2, upper=3.2, alpha=1.0, beta=1.0)
        self.assertAlmostEqual(f_unif.default_value, 2.7)
        self.assertAlmostEqual(f_unif.normalized_default_value, 0.5)

        # This should yield a value that's halfway towards the max in logspace
        f_unif_log = BetaFloatHyperparameter(
            "param", lower=np.exp(2.2), upper=np.exp(3.2), alpha=1.0, beta=1.0, log=True)
        self.assertAlmostEqual(f_unif_log.default_value, np.exp(2.7))
        self.assertAlmostEqual(f_unif_log.normalized_default_value, 0.5)

        # Then, test a case where the default value is the mode of the beta dist
        f_max = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=4.7, beta=2.12)
        self.assertAlmostEqual(f_max.default_value, 1.0705394190871367)
        self.assertAlmostEqual(f_max.normalized_default_value, 0.7676348547717842)

        f_max_log = BetaFloatHyperparameter(
            "param", lower=np.exp(-2.0), upper=np.exp(2.0), alpha=4.7, beta=2.12, log=True)
        self.assertAlmostEqual(f_max_log.default_value, np.exp(1.0705394190871367))
        self.assertAlmostEqual(f_max_log.normalized_default_value, 0.7676348547717842)

        # These parameters do not yeild an integer default solution
        f_quant = BetaFloatHyperparameter(
            "param", lower=-2.0, upper=2.0, alpha=4.7, beta=2.12, q=1)
        self.assertAlmostEqual(f_quant.default_value, 1.0)

        f_log_quant = BetaFloatHyperparameter(
            "param", lower=1, upper=100000, alpha=2, beta=2, q=1, log=True)
        self.assertAlmostEqual(f_log_quant.default_value, 316)

        # since it's quantized, it gets distributed evenly among the search space
        # as such, the possible normalized defaults are 0.1, 0.3, 0.5, 0.7, 0.9
        self.assertAlmostEqual(f_quant.normalized_default_value, 0.7, places=4)

        # TODO log and quantization together does not yield a correct default for the beta
        # hyperparameter, but it is relatively close to being correct. However, it is not
        # being

        # The default value is independent of whether you log the parameter or not
        f_legal_nolog = BetaFloatHyperparameter(
            "param", lower=1, upper=10.0, alpha=3.0, beta=2.0, default_value=1, log=True)
        f_legal_log = BetaFloatHyperparameter(
            "param", lower=1, upper=10.0, alpha=3.0, beta=2.0, default_value=1, log=False)

        self.assertAlmostEqual(f_legal_nolog.default_value, 1)
        self.assertAlmostEqual(f_legal_log.default_value, 1)

        # These are necessary, as we bypass the same check in the UniformFloatHP by design
        with self.assertRaisesRegex(ValueError, "Illegal default value 0"):
            BetaFloatHyperparameter("param", lower=1, upper=10.0, alpha=3.0,
                                    beta=2.0, default_value=0, log=False)
        with self.assertRaisesRegex(ValueError, "Illegal default value 0"):
            BetaFloatHyperparameter("param", lower=1, upper=1000.0, alpha=3.0,
                                    beta=2.0, default_value=0, log=True)

    def test_betafloat_to_uniformfloat(self):
        f1 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=4, beta=2, q=0.1)
        f1_expected = UniformFloatHyperparameter(
            "param", lower=-2.0, upper=2.0, q=0.1, default_value=1)
        f1_actual = f1.to_uniform()
        self.assertEqual(f1_expected, f1_actual)

        f2 = BetaFloatHyperparameter("param", lower=1, upper=1000, alpha=3, beta=2, log=True)
        f2_expected = UniformFloatHyperparameter(
            "param", lower=1, upper=1000, log=True, default_value=100)
        f2_actual = f2.to_uniform()
        self.assertEqual(f2_expected, f2_actual)

    def test_betafloat_to_integer(self):
        f1 = BetaFloatHyperparameter("param", lower=-2.0, upper=2.0, alpha=4, beta=2)
        f2_expected = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=4, beta=2)
        f2_actual = f1.to_integer()
        self.assertEqual(f2_expected, f2_actual)

    def test_betafloat_pdf(self):
        c1 = BetaFloatHyperparameter("param", lower=0, upper=10, alpha=3, beta=2)
        c2 = BetaFloatHyperparameter("logparam", lower=np.exp(
            0), upper=np.exp(10), alpha=3, beta=2, log=True)
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

        self.assertAlmostEqual(c1.pdf(point_1)[0], 0.07559999999999997)
        self.assertAlmostEqual(c2.pdf(point_1_log)[0], 0.07559999999999997)
        self.assertAlmostEqual(c1.pdf(point_2)[0], 0.011761200000000013)
        self.assertAlmostEqual(c2.pdf(point_2_log)[0], 0.011761200000000013)
        self.assertAlmostEqual(c3.pdf(point_3)[0], 30.262164001861198)
        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        self.assertEqual(c1.pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c1.pdf(point_outside_range_2)[0], 0.0)
        self.assertEqual(c2.pdf(point_outside_range_1_log)[0], 0.0)
        self.assertEqual(c2.pdf(point_outside_range_2_log)[0], 0.0)

        array_results = c1.pdf(array_1)
        array_results_log = c2.pdf(array_1_log)
        expected_results = np.array([0.07559999999999997, 0.011761200000000013, 0])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results.shape)
        for res, log_res, exp_res in zip(array_results, array_results, expected_results):
            self.assertAlmostEqual(res, exp_res)
            self.assertAlmostEqual(log_res, exp_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_1)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_2)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_3)

    def test_betafloat__pdf(self):
        c1 = BetaFloatHyperparameter("param", lower=0, upper=10, alpha=3, beta=2)
        c2 = BetaFloatHyperparameter("logparam", lower=np.exp(
            0), upper=np.exp(10), alpha=3, beta=2, log=True)
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

        self.assertAlmostEqual(c1._pdf(point_1)[0], 0.07559999999999997)
        self.assertAlmostEqual(c2._pdf(point_1)[0], 0.07559999999999997)
        self.assertAlmostEqual(c1._pdf(point_2)[0], 0.011761200000000013)
        self.assertAlmostEqual(c2._pdf(point_2)[0], 0.011761200000000013)
        self.assertAlmostEqual(c3._pdf(point_3)[0], 30.262164001861198)

        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        self.assertEqual(c1._pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c1._pdf(point_outside_range_2)[0], 0.0)
        self.assertEqual(c2._pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c2._pdf(point_outside_range_2)[0], 0.0)

        array_results = c1._pdf(array_1)
        array_results_log = c2._pdf(array_1)
        expected_results = np.array([0.07559999999999997, 0.011761200000000013, 0])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results.shape)
        for res, log_res, exp_res in zip(array_results, array_results, expected_results):
            self.assertAlmostEqual(res, exp_res)
            self.assertAlmostEqual(log_res, exp_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        # Simply check that it runs, since _pdf does not restrict shape (only public method does)
        c1._pdf(accepted_shape_1)
        c1._pdf(accepted_shape_2)
        c1._pdf(accepted_shape_3)

    def test_betafloat_get_max_density(self):
        c1 = BetaFloatHyperparameter("param", lower=0, upper=10, alpha=3, beta=2)
        c2 = BetaFloatHyperparameter("logparam", lower=np.exp(
            0), upper=np.exp(10), alpha=3, beta=2, log=True)
        c3 = BetaFloatHyperparameter("param", lower=0, upper=0.5, alpha=1.1, beta=25)
        self.assertAlmostEqual(c1.get_max_density(), 0.17777777777777776)
        self.assertAlmostEqual(c2.get_max_density(), 0.17777777777777776)
        self.assertAlmostEqual(c3.get_max_density(), 38.00408137865127)

    def test_uniforminteger(self):
        # TODO: rounding or converting or error message?

        f1 = UniformIntegerHyperparameter("param", 0.0, 5.0)
        f1_ = UniformIntegerHyperparameter("param", 0, 5)
        self.assertEqual(f1, f1_)
        self.assertEqual("param, Type: UniformInteger, Range: [0, 5], "
                         "Default: 2", str(f1))

        # Test name is accessible
        self.assertEqual(f1.name, "param")
        self.assertEqual(f1.lower, 0)
        self.assertEqual(f1.upper, 5)
        self.assertEqual(f1.q, None)
        self.assertEqual(f1.default_value, 2)
        self.assertEqual(f1.log, False)
        self.assertAlmostEqual(f1.normalized_default_value, (2.0 + 0.49999) / (5.49999 + 0.49999))

        quantization_warning = (
            "Setting quantization < 1 for Integer Hyperparameter 'param' has no effect"
        )
        with pytest.warns(UserWarning, match=quantization_warning):
            f2 = UniformIntegerHyperparameter("param", 0, 10, q=0.1)
        with pytest.warns(UserWarning, match=quantization_warning):
            f2_ = UniformIntegerHyperparameter("param", 0, 10, q=0.1)
        self.assertEqual(f2, f2_)
        self.assertEqual(
            "param, Type: UniformInteger, Range: [0, 10], Default: 5",
            str(f2))

        f2_large_q = UniformIntegerHyperparameter("param", 0, 10, q=2)
        f2_large_q_ = UniformIntegerHyperparameter("param", 0, 10, q=2)
        self.assertEqual(f2_large_q, f2_large_q_)
        self.assertEqual(
            "param, Type: UniformInteger, Range: [0, 10], Default: 5, Q: 2",
            str(f2_large_q))

        f3 = UniformIntegerHyperparameter("param", 1, 10, log=True)
        f3_ = UniformIntegerHyperparameter("param", 1, 10, log=True)
        self.assertEqual(f3, f3_)
        self.assertEqual(
            "param, Type: UniformInteger, Range: [1, 10], Default: 3, "
            "on log-scale", str(f3))

        f4 = UniformIntegerHyperparameter("param", 1, 10, default_value=1, log=True)
        f4_ = UniformIntegerHyperparameter("param", 1, 10, default_value=1, log=True)
        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: UniformInteger, Range: [1, 10], Default: 1, "
            "on log-scale", str(f4))

        with pytest.warns(UserWarning, match=quantization_warning):
            f5 = UniformIntegerHyperparameter("param", 1, 10, default_value=1, q=0.1, log=True)
        with pytest.warns(UserWarning, match=quantization_warning):
            f5_ = UniformIntegerHyperparameter("param", 1, 10, default_value=1, q=0.1, log=True)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: UniformInteger, Range: [1, 10], Default: 1, "
            "on log-scale", str(f5))

        # self.assertNotEqual(f2, f2_large_q)
        self.assertNotEqual(f1, "UniformFloat")

        # test that meta-data is stored correctly
        with pytest.warns(UserWarning, match=quantization_warning):
            f_meta = UniformIntegerHyperparameter("param", 1, 10, q=0.1, log=True,
                                                  default_value=1, meta=dict(self.meta_data))
        self.assertEqual(f_meta.meta, self.meta_data)

        self.assertEqual(f1.get_size(), 6)
        self.assertEqual(f2.get_size(), 11)
        self.assertEqual(f2_large_q.get_size(), 6)
        self.assertEqual(f3.get_size(), 10)
        self.assertEqual(f4.get_size(), 10)
        self.assertEqual(f5.get_size(), 10)

    def test_uniformint_legal_float_values(self):
        n_iter = UniformIntegerHyperparameter("n_iter", 5., 1000., default_value=20.0)

        self.assertIsInstance(n_iter.default_value, int)
        self.assertRaisesRegex(ValueError, r"For the Integer parameter n_iter, "
                                           r"the value must be an Integer, too."
                                           r" Right now it is a <(type|class) "
                                           r"'float'>"
                                           r" with value 20.5.",
                               UniformIntegerHyperparameter, "n_iter", 5.,
                               1000., default_value=20.5)

    def test_uniformint_illegal_bounds(self):
        self.assertRaisesRegex(
            ValueError,
            r"Negative lower bound \(0\) for log-scale hyperparameter "
            r"param is forbidden.", UniformIntegerHyperparameter, "param", 0, 10,
            log=True)

        self.assertRaisesRegex(
            ValueError,
            "Upper bound 1 must be larger than lower bound 0 for "
            "hyperparameter param", UniformIntegerHyperparameter, "param", 1, 0)

    def test_uniformint_pdf(self):
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
        self.assertAlmostEqual(c1.pdf(point_1)[0], 0.2, places=5)
        self.assertAlmostEqual(c2.pdf(point_1_log)[0], 0.0001, places=5)
        self.assertAlmostEqual(c1.pdf(point_2)[0], 0.2, places=5)
        self.assertAlmostEqual(c2.pdf(point_2_log)[0], 0.0001, places=5)
        self.assertAlmostEqual(c1.pdf(non_integer_point)[0], 0.0, places=5)
        self.assertAlmostEqual(c2.pdf(non_integer_point)[0], 0.0, places=5)
        self.assertAlmostEqual(c3.pdf(point_1)[0], 0.07142857142857142, places=5)

        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        # since inverse_transform pulls everything into range,
        # even points outside get evaluated in range
        self.assertAlmostEqual(c1.pdf(point_outside_range)[0], 0.2, places=5)
        self.assertAlmostEqual(c2.pdf(point_outside_range_log)[0], 0.0001, places=5)

        # this, however, is a negative value on a log param, which cannot be pulled into range
        with pytest.warns(RuntimeWarning, match='invalid value encountered in log'):
            self.assertEqual(c2.pdf(point_outside_range)[0], 0.0)

        array_results = c1.pdf(array_1)
        array_results_log = c2.pdf(array_1)
        expected_results = np.array([0.2, 0.2, 0])
        expected_results_log = np.array([0.0001, 0.0001, 0])
        self.assertAlmostEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results.shape)
        for res, log_res, exp_res, exp_log_res in zip(array_results, array_results,
                                                      expected_results, expected_results_log):
            self.assertAlmostEqual(res, exp_res, places=5)
            self.assertAlmostEqual(log_res, exp_res, places=5)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_1)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_2)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_3)

    def test_uniformint__pdf(self):
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
        self.assertAlmostEqual(c1._pdf(point_1)[0], 0.2, places=5)
        self.assertAlmostEqual(c2._pdf(point_1)[0], 0.0001, places=5)
        self.assertAlmostEqual(c1._pdf(point_2)[0], 0.2, places=5)
        self.assertAlmostEqual(c2._pdf(point_2)[0], 0.0001, places=5)

        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        # since inverse_transform pulls everything into range,
        # even points outside get evaluated in range
        self.assertAlmostEqual(c1._pdf(point_outside_range)[0], 0.0, places=5)
        self.assertAlmostEqual(c2._pdf(point_outside_range)[0], 0.0, places=5)

        array_results = c1._pdf(array_1)
        array_results_log = c2._pdf(array_1)
        expected_results = np.array([0.2, 0.2, 0])
        expected_results_log = np.array([0.0001, 0.0001, 0])
        self.assertAlmostEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results.shape)
        for res, log_res, exp_res, exp_log_res in zip(array_results, array_results_log,
                                                      expected_results, expected_results_log):
            self.assertAlmostEqual(res, exp_res, places=5)
            self.assertAlmostEqual(log_res, exp_log_res, places=5)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1._pdf(0.2)
        with self.assertRaises(TypeError):
            c1._pdf('pdf')

        # Simply check that it runs, since _pdf does not restrict shape (only public method does)
        c1._pdf(accepted_shape_1)
        c1._pdf(accepted_shape_2)
        c1._pdf(accepted_shape_3)

    def test_uniformint_get_max_density(self):
        c1 = UniformIntegerHyperparameter("param", lower=0, upper=4)
        c2 = UniformIntegerHyperparameter("logparam", lower=1, upper=10000, log=True)
        c3 = UniformIntegerHyperparameter("param", lower=-1, upper=12)
        self.assertAlmostEqual(c1.get_max_density(), 0.2)
        self.assertAlmostEqual(c2.get_max_density(), 0.0001)
        self.assertAlmostEqual(c3.get_max_density(), 0.07142857142857142)

    def test_normalint(self):
        # TODO test for unequal!
        f1 = NormalIntegerHyperparameter("param", 0.5, 5.5)
        f1_ = NormalIntegerHyperparameter("param", 0.5, 5.5)
        self.assertEqual(f1, f1_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0.5 Sigma: 5.5, Default: 0.5",
            str(f1))

        # Test attributes are accessible
        self.assertEqual(f1.name, "param")
        self.assertEqual(f1.mu, 0.5)
        self.assertEqual(f1.sigma, 5.5)
        self.assertEqual(f1.q, None)
        self.assertEqual(f1.log, False)
        self.assertAlmostEqual(f1.default_value, 0.5)
        self.assertAlmostEqual(f1.normalized_default_value, 0.5)

        with pytest.warns(UserWarning, match="Setting quantization < 1 for Integer "
                                             "Hyperparameter 'param' has no effect"):
            f2 = NormalIntegerHyperparameter("param", 0, 10, q=0.1)
        with pytest.warns(UserWarning, match="Setting quantization < 1 for Integer "
                                             "Hyperparameter 'param' has no effect"):
            f2_ = NormalIntegerHyperparameter("param", 0, 10, q=0.1)
        self.assertEqual(f2, f2_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 0",
            str(f2))

        f2_large_q = NormalIntegerHyperparameter("param", 0, 10, q=2)
        f2_large_q_ = NormalIntegerHyperparameter("param", 0, 10, q=2)
        self.assertEqual(f2_large_q, f2_large_q_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 0, Q: 2",
            str(f2_large_q))

        f3 = NormalIntegerHyperparameter("param", 0, 10, log=True)
        f3_ = NormalIntegerHyperparameter("param", 0, 10, log=True)
        self.assertEqual(f3, f3_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 1, "
            "on log-scale", str(f3))

        f4 = NormalIntegerHyperparameter("param", 0, 10, default_value=3, log=True)
        f4_ = NormalIntegerHyperparameter("param", 0, 10, default_value=3, log=True)
        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 3, "
            "on log-scale", str(f4))

        with pytest.warns(UserWarning, match="Setting quantization < 1 for Integer "
                                             "Hyperparameter 'param' has no effect"):
            f5 = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
            f5_ = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 1, "
            "on log-scale", str(f5))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

        # test that meta-data is stored correctly
        f_meta = NormalIntegerHyperparameter("param", 0, 10, default_value=1, log=True,
                                             meta=dict(self.meta_data))
        self.assertEqual(f_meta.meta, self.meta_data)

        # Test get_size
        for int_hp in (f1, f2, f3, f4, f5):
            self.assertTrue(np.isinf(int_hp.get_size()))

    def test_normalint_legal_float_values(self):
        n_iter = NormalIntegerHyperparameter("n_iter", 0, 1., default_value=2.0)
        self.assertIsInstance(n_iter.default_value, int)
        self.assertRaisesRegex(ValueError, r"For the Integer parameter n_iter, "
                                           r"the value must be an Integer, too."
                                           r" Right now it is a "
                                           r"<(type|class) 'float'>"
                                           r" with value 0.5.",
                               UniformIntegerHyperparameter, "n_iter", 0,
                               1., default_value=0.5)

    def test_normalint_to_uniform(self):
        with pytest.warns(UserWarning, match="Setting quantization < 1 for Integer "
                                             "Hyperparameter 'param' has no effect"):
            f1 = NormalIntegerHyperparameter("param", 0, 10, q=0.1)
        f1_expected = UniformIntegerHyperparameter("param", -30, 30)
        f1_actual = f1.to_uniform()
        self.assertEqual(f1_expected, f1_actual)

    def test_normalint_is_legal(self):
        with pytest.warns(UserWarning, match="Setting quantization < 1 for Integer "
                                             "Hyperparameter 'param' has no effect"):
            f1 = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
        self.assertFalse(f1.is_legal(3.1))
        self.assertFalse(f1.is_legal(3.0))   # 3.0 behaves like an Integer
        self.assertFalse(f1.is_legal("BlaBlaBla"))
        self.assertTrue(f1.is_legal(2))
        self.assertTrue(f1.is_legal(-15))

        # Test is legal vector
        self.assertTrue(f1.is_legal_vector(1.0))
        self.assertTrue(f1.is_legal_vector(0.0))
        self.assertTrue(f1.is_legal_vector(0))
        self.assertTrue(f1.is_legal_vector(0.3))
        self.assertTrue(f1.is_legal_vector(-0.1))
        self.assertTrue(f1.is_legal_vector(1.1))
        self.assertRaises(TypeError, f1.is_legal_vector, "Hahaha")

    def test_normalint_pdf(self):
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

        self.assertAlmostEqual(c1.pdf(point_1)[0], 0.20747194595587332)
        self.assertAlmostEqual(c2.pdf(point_1_log)[0], 0.002625781612612434)
        self.assertAlmostEqual(c1.pdf(point_2)[0], 0.00045384303905059246)
        self.assertAlmostEqual(c2.pdf(point_2_log)[0], 0.0004136885586376241)
        self.assertAlmostEqual(c3.pdf(point_3)[0], 0.9988874412972069)
        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        self.assertEqual(c1.pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c1.pdf(point_outside_range_2)[0], 0.0)
        with pytest.warns(RuntimeWarning, match='divide by zero encountered in log'):
            self.assertEqual(c2.pdf(point_outside_range_1_log)[0], 0.0)
        self.assertEqual(c2.pdf(point_outside_range_2_log)[0], 0.0)

        self.assertEqual(c1.pdf(non_integer_point)[0], 0.0)
        self.assertEqual(c2.pdf(non_integer_point)[0], 0.0)

        array_results = c1.pdf(array_1)
        array_results_log = c2.pdf(array_1_log)
        expected_results = np.array([0.20747194595587332, 0.00045384303905059246, 0])
        expected_results_log = np.array([0.002625781612612434, 0.000688676747843256, 0])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results.shape)
        for res, log_res, exp_res, exp_log_res in zip(array_results, array_results_log,
                                                      expected_results, expected_results_log):
            self.assertAlmostEqual(res, exp_res)
            self.assertAlmostEqual(log_res, exp_log_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        c_nobounds = NormalFloatHyperparameter("param", mu=3, sigma=2)
        self.assertAlmostEqual(c_nobounds.pdf(np.array([2])), 0.17603266338214976)

        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_1)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_2)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_3)

    def test_normalint__pdf(self):
        c1 = NormalIntegerHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
        c2 = NormalIntegerHyperparameter("logparam", lower=1, upper=1000, mu=3, sigma=2, log=True)

        point_1 = np.array([3])
        point_2 = np.array([5.2])
        array_1 = np.array([3, 5.2, 11])
        point_outside_range_1 = np.array([-1])
        point_outside_range_2 = np.array([11])

        self.assertAlmostEqual(c1._pdf(point_1)[0], 0.20747194595587332)
        self.assertAlmostEqual(c2._pdf(point_1)[0], 0.0027903779510164133)
        self.assertAlmostEqual(c1._pdf(point_2)[0], 0.1132951239316783)
        self.assertAlmostEqual(c2._pdf(point_2)[0], 0.001523754039709375)
        # TODO - change this once the is_legal support is there
        # but does not have an actual impact of now
        self.assertEqual(c1._pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c1._pdf(point_outside_range_2)[0], 0.0)
        self.assertEqual(c2._pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c2._pdf(point_outside_range_2)[0], 0.0)

        array_results = c1._pdf(array_1)
        array_results_log = c2._pdf(array_1)
        expected_results = np.array([0.20747194595587332, 0.1132951239316783, 0])
        expected_results_log = np.array([0.0027903779510164133, 0.001523754039709375, 0])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results.shape)
        for res, log_res, exp_res, exp_log_res in zip(array_results, array_results_log,
                                                      expected_results, expected_results_log):
            self.assertAlmostEqual(res, exp_res)
            self.assertAlmostEqual(log_res, exp_log_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        c_nobounds = NormalFloatHyperparameter("param", mu=3, sigma=2)
        self.assertAlmostEqual(c_nobounds.pdf(np.array([2])), 0.17603266338214976)

    def test_normalint_get_max_density(self):
        c1 = NormalIntegerHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
        c2 = NormalIntegerHyperparameter("logparam", lower=1, upper=1000, mu=3, sigma=2, log=True)
        c3 = NormalIntegerHyperparameter("param", lower=0, upper=2, mu=-1.2, sigma=0.5)
        self.assertAlmostEqual(c1.get_max_density(), 0.20747194595587332)
        self.assertAlmostEqual(c2.get_max_density(), 0.002790371598208875)
        self.assertAlmostEqual(c3.get_max_density(), 0.9988874412972069)

    ############################################################
    def test_betaint(self):
        # TODO test non-equality
        f1 = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.1)
        f1_ = BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=3, beta=1.1)
        self.assertEqual(f1, f1_)
        self.assertEqual(
            "param, Type: BetaInteger, Alpha: 3.0 Beta: 1.1, Range: [-2, 2], Default: 2",
            str(f1))

        self.assertAlmostEqual(f1.alpha, 3.0)
        self.assertAlmostEqual(f1.beta, 1.1)

        # Test copy
        copy_f1 = copy.copy(f1)
        self.assertEqual(copy_f1.name, f1.name)
        self.assertEqual(copy_f1.alpha, f1.alpha)
        self.assertEqual(copy_f1.beta, f1.beta)
        self.assertEqual(copy_f1.default_value, f1.default_value)

        f2 = BetaIntegerHyperparameter("param", lower=-2.0, upper=4.0, alpha=3.0, beta=1.1, q=2)
        f2_ = BetaIntegerHyperparameter("param", lower=-2, upper=4, alpha=3, beta=1.1, q=2)
        self.assertEqual(f2, f2_)

        self.assertEqual(
            "param, Type: BetaInteger, Alpha: 3.0 Beta: 1.1, Range: [-2, 4], Default: 4, "
            "Q: 2", str(f2))

        f3 = BetaIntegerHyperparameter("param", lower=1, upper=1000, alpha=3.0, beta=2.0, log=True)
        f3_ = BetaIntegerHyperparameter("param", lower=1, upper=1000, alpha=3.0, beta=2.0, log=True)
        self.assertEqual(f3, f3_)
        self.assertEqual(
            "param, Type: BetaInteger, Alpha: 3.0 Beta: 2.0, Range: [1, 1000], Default: 100, "
            "on log-scale", str(f3))

        with self.assertRaises(ValueError):
            BetaIntegerHyperparameter("param", lower=-1, upper=10.0, alpha=6.0, beta=2.0, log=True)

        # test that meta-data is stored correctly
        f_meta = BetaFloatHyperparameter("param", lower=1, upper=10.0, alpha=3.0, beta=2.0,
                                         log=False, meta=dict(self.meta_data))
        self.assertEqual(f_meta.meta, self.meta_data)

    def test_betaint_default_value(self):
        # should default to the maximal value in the search space
        f_max = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.0)
        self.assertAlmostEqual(f_max.default_value, 2.0)
        # since integer values are staggered over the normalized space
        self.assertAlmostEqual(f_max.normalized_default_value, 0.9, places=4)

        # The normalized log defaults should be the same as if one were to create a uniform
        # distribution with the same default value as is generated by the beta
        f_max_log = BetaIntegerHyperparameter(
            "param", lower=1.0, upper=10.0, alpha=3.0, beta=1.0, log=True)
        self.assertAlmostEqual(f_max_log.default_value, 10.0)
        self.assertAlmostEqual(f_max_log.normalized_default_value, 0.983974646746037)

        # should default to the minimal value in the search space
        f_min = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=1.0, beta=1.5)
        self.assertAlmostEqual(f_min.default_value, -2.0)
        self.assertAlmostEqual(f_min.normalized_default_value, 0.1, places=4)

        f_min_log = BetaIntegerHyperparameter(
            "param", lower=1.0, upper=10.0, alpha=1.0, beta=1.5, log=True)
        self.assertAlmostEqual(f_min_log.default_value, 1.0)
        self.assertAlmostEqual(f_min_log.normalized_default_value, 0.22766524636349278)

        # Symmeric, should default to the middle
        f_symm = BetaIntegerHyperparameter("param", lower=5, upper=9, alpha=4.6, beta=4.6)
        self.assertAlmostEqual(f_symm.default_value, 7)
        self.assertAlmostEqual(f_symm.normalized_default_value, 0.5)

        # This should yield a value that's approximately halfway towards the max in logspace
        f_symm_log = BetaIntegerHyperparameter(
            "param", lower=1, upper=round(np.exp(10)), alpha=4.6, beta=4.6, log=True)
        self.assertAlmostEqual(f_symm_log.default_value, 148)
        self.assertAlmostEqual(f_symm_log.normalized_default_value, 0.5321491582577761)

        # Uniform, should also default to the middle
        f_unif = BetaIntegerHyperparameter("param", lower=2, upper=6, alpha=1.0, beta=1.0)
        self.assertAlmostEqual(f_unif.default_value, 4)
        self.assertAlmostEqual(f_unif.normalized_default_value, 0.5)

        # This should yield a value that's halfway towards the max in logspace
        f_unif_log = BetaIntegerHyperparameter(
            "param", lower=1, upper=round(np.exp(10)), alpha=1, beta=1, log=True)
        self.assertAlmostEqual(f_unif_log.default_value, 148)
        self.assertAlmostEqual(f_unif_log.normalized_default_value, 0.5321491582577761)

        # Then, test a case where the default value is the mode of the beta dist somewhere in
        # the interior of the search space - but not the center
        f_max = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=4.7, beta=2.12)
        self.assertAlmostEqual(f_max.default_value, 1.0)
        self.assertAlmostEqual(f_max.normalized_default_value, 0.7, places=4)

        f_max_log = BetaIntegerHyperparameter(
            "param", lower=1, upper=round(np.exp(10)), alpha=4.7, beta=2.12, log=True)
        self.assertAlmostEqual(f_max_log.default_value, 2157)
        self.assertAlmostEqual(f_max_log.normalized_default_value, 0.7827083200774537)

        # These parameters yield a mode at approximately 1.1, so should thus yield default at 2
        f_quant = BetaIntegerHyperparameter(
            "param", lower=-2.0, upper=2.0, alpha=4.7, beta=2.12, q=2)
        self.assertAlmostEqual(f_quant.default_value, 2.0)

        # since it's quantized, it gets distributed evenly among the search space
        # as such, the possible normalized defaults are 0.1, 0.3, 0.5, 0.7, 0.9
        self.assertAlmostEqual(f_quant.normalized_default_value, 0.9, places=4)

        # TODO log and quantization together does not yield a correct default for the beta
        # hyperparameter, but it is relatively close to being correct.

        # The default value is independent of whether you log the parameter or not
        f_legal_nolog = BetaIntegerHyperparameter(
            "param", lower=1, upper=10.0, alpha=3.0, beta=2.0, default_value=1, log=True)
        f_legal_log = BetaIntegerHyperparameter(
            "param", lower=1, upper=10.0, alpha=3.0, beta=2.0, default_value=1, log=False)

        self.assertAlmostEqual(f_legal_nolog.default_value, 1)
        self.assertAlmostEqual(f_legal_log.default_value, 1)

        # These are necessary, as we bypass the same check in the UniformFloatHP by design
        with self.assertRaisesRegex(ValueError, "Illegal default value 0"):
            BetaFloatHyperparameter("param", lower=1, upper=10.0, alpha=3.0,
                                    beta=2.0, default_value=0, log=False)
        with self.assertRaisesRegex(ValueError, "Illegal default value 0"):
            BetaFloatHyperparameter("param", lower=1, upper=1000.0, alpha=3.0,
                                    beta=2.0, default_value=0, log=True)

    def test_betaint_dist_parameters(self):
        # This one should just be created without raising an error - corresponds to uniform dist.
        BetaIntegerHyperparameter("param", lower=0, upper=10.0, alpha=1, beta=1)

        # This one is not permitted as the co-domain is not finite
        with self.assertRaises(ValueError):
            BetaIntegerHyperparameter("param", lower=0, upper=100, alpha=0.99, beta=0.99)
        # And these parameters do not define a proper beta distribution whatsoever
        with self.assertRaises(ValueError):
            BetaIntegerHyperparameter("param", lower=0, upper=100, alpha=-0.1, beta=-0.1)

        # test parameters that do not create a legit beta distribution, one at a time
        with self.assertRaises(ValueError):
            BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=-11, beta=5)
        with self.assertRaises(ValueError):
            BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=5, beta=-11)

        # test parameters that do not yield a finite co-domain, one at a time
        with self.assertRaises(ValueError):
            BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=0.5, beta=11)
        with self.assertRaises(ValueError):
            BetaIntegerHyperparameter("param", lower=-2, upper=2, alpha=11, beta=0.5)

    def test_betaint_legal_float_values(self):
        f1 = BetaIntegerHyperparameter("param", lower=-2.0, upper=2.0, alpha=3.0, beta=1.1)
        self.assertIsInstance(f1.default_value, int)
        self.assertRaisesRegex(ValueError, "Illegal default value 0.5",
                                           BetaIntegerHyperparameter, "param",
                                           lower=-2.0,
                                           upper=2.0,
                                           alpha=3.0,
                                           beta=1.1,
                                           default_value=0.5)

    def test_betaint_to_uniform(self):
        with pytest.warns(UserWarning, match="Setting quantization < 1 for Integer "
                                             "Hyperparameter 'param' has no effect"):
            f1 = BetaIntegerHyperparameter("param", lower=-30, upper=30, alpha=6.0, beta=2, q=0.1)

        f1_expected = UniformIntegerHyperparameter("param", -30, 30, default_value=20)
        f1_actual = f1.to_uniform()
        self.assertEqual(f1_expected, f1_actual)

    def test_betaint_pdf(self):
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
        self.assertAlmostEqual(c1.pdf(point_1)[0], 0.07636363636363634, places=3)
        self.assertAlmostEqual(c2.pdf(point_1_log)[0], 0.0008724511426701984, places=3)
        self.assertAlmostEqual(c1.pdf(point_2)[0], 0.09818181818181816, places=3)
        self.assertAlmostEqual(c2.pdf(point_2_log)[0], 0.0008683622684160343, places=3)
        self.assertAlmostEqual(c3.pdf(point_3)[0], 0.9979110652388783, places=3)

        self.assertEqual(c1.pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c1.pdf(point_outside_range_2)[0], 0.0)
        with pytest.warns(RuntimeWarning, match='divide by zero encountered in log'):
            self.assertEqual(c2.pdf(point_outside_range_1_log)[0], 0.0)
        self.assertEqual(c2.pdf(point_outside_range_2_log)[0], 0.0)

        self.assertEqual(c1.pdf(non_integer_point)[0], 0.0)
        self.assertEqual(c2.pdf(non_integer_point)[0], 0.0)

        array_results = c1.pdf(array_1)
        array_results_log = c2.pdf(array_1_log)
        expected_results = np.array([0.07636363636363634, 0.09818181818181816, 0])
        expected_results_log = np.array([0.0008724511426701984, 0.0008683622684160343, 0])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results.shape)
        for res, log_res, exp_res, exp_log_res in zip(array_results, array_results_log,
                                                      expected_results, expected_results_log):
            self.assertAlmostEqual(res, exp_res, places=3)
            self.assertAlmostEqual(log_res, exp_log_res, places=3)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')

        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_1)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_2)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_3)

    def test_betaint__pdf(self):
        c1 = BetaIntegerHyperparameter("param", alpha=3, beta=2, lower=0, upper=10)
        c2 = BetaIntegerHyperparameter("logparam", alpha=3, beta=2,
                                       lower=1, upper=round(np.exp(10)), log=True)

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

        self.assertAlmostEqual(c1._pdf(point_1)[0], 0.0475566)
        self.assertAlmostEqual(c2._pdf(point_1_log)[0], 0.00004333811)
        self.assertAlmostEqual(c1._pdf(point_2)[0], 0.1091810)
        self.assertAlmostEqual(c2._pdf(point_2_log)[0], 0.00005571951)

        # test points that are actually outside of the _pdf range due to the skewing
        # of the unit hypercube space
        self.assertEqual(c1._pdf(point_outside_range_1)[0], 0.0)
        self.assertEqual(c1._pdf(point_outside_range_2)[0], 0.0)
        self.assertEqual(c2._pdf(point_outside_range_1_log)[0], 0.0)

        array_results = c1._pdf(array_1)
        array_results_log = c2._pdf(array_1_log)
        expected_results = np.array([0.0475566, 0.1091810, 0])
        expected_results_log = np.array([0.00004333811, 0.00005571951, 0])
        self.assertEqual(array_results.shape, expected_results.shape)
        self.assertEqual(array_results_log.shape, expected_results_log.shape)
        for res, log_res, exp_res, exp_log_res in zip(array_results, array_results_log,
                                                      expected_results, expected_results_log):
            self.assertAlmostEqual(res, exp_res)
            self.assertAlmostEqual(log_res, exp_log_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1._pdf(0.2)
        with self.assertRaises(TypeError):
            c1._pdf('pdf')

        # Simply check that it runs, since _pdf does not restrict shape (only public method does)
        c1._pdf(accepted_shape_1)
        c1._pdf(accepted_shape_2)
        c1._pdf(accepted_shape_3)

    def test_betaint_get_max_density(self):
        c1 = BetaIntegerHyperparameter("param", alpha=3, beta=2, lower=0, upper=10)
        c2 = BetaIntegerHyperparameter("logparam", alpha=3, beta=2, lower=1, upper=1000, log=True)
        c3 = BetaIntegerHyperparameter("param", alpha=1.1, beta=10, lower=0, upper=3)
        self.assertAlmostEqual(c1.get_max_density(), 0.1781818181818181)
        self.assertAlmostEqual(c2.get_max_density(), 0.0018733953504422762)
        self.assertAlmostEqual(c3.get_max_density(), 0.9979110652388783)

    def test_categorical(self):
        # TODO test for inequality
        f1 = CategoricalHyperparameter("param", [0, 1])
        f1_ = CategoricalHyperparameter("param", [0, 1])
        self.assertEqual(f1, f1_)
        self.assertEqual("param, Type: Categorical, Choices: {0, 1}, Default: 0",
                         str(f1))

        # Test attributes are accessible
        self.assertEqual(f1.name, "param")
        self.assertEqual(f1.num_choices, 2)
        self.assertEqual(f1.default_value, 0)
        self.assertEqual(f1.normalized_default_value, 0)
        self.assertTupleEqual(f1.probabilities, (0.5, 0.5))

        f2 = CategoricalHyperparameter("param", list(range(0, 1000)))
        f2_ = CategoricalHyperparameter("param", list(range(0, 1000)))
        self.assertEqual(f2, f2_)
        self.assertEqual(
            "param, Type: Categorical, Choices: {%s}, Default: 0" %
            ", ".join([str(choice) for choice in range(0, 1000)]),
            str(f2))

        f3 = CategoricalHyperparameter("param", list(range(0, 999)))
        self.assertNotEqual(f2, f3)

        f4 = CategoricalHyperparameter("param_", list(range(0, 1000)))
        self.assertNotEqual(f2, f4)

        f5 = CategoricalHyperparameter("param", list(range(0, 999)) + [1001])
        self.assertNotEqual(f2, f5)

        f6 = CategoricalHyperparameter("param", ["a", "b"], default_value="b")
        f6_ = CategoricalHyperparameter("param", ["a", "b"], default_value="b")
        self.assertEqual(f6, f6_)
        self.assertEqual("param, Type: Categorical, Choices: {a, b}, Default: b", str(f6))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

        # test that meta-data is stored correctly
        f_meta = CategoricalHyperparameter("param", ["a", "b"], default_value="a",
                                           meta=dict(self.meta_data))
        self.assertEqual(f_meta.meta, self.meta_data)

        self.assertEqual(f1.get_size(), 2)
        self.assertEqual(f2.get_size(), 1000)
        self.assertEqual(f3.get_size(), 999)
        self.assertEqual(f4.get_size(), 1000)
        self.assertEqual(f5.get_size(), 1000)
        self.assertEqual(f6.get_size(), 2)

    def test_cat_equal(self):
        # Test that weights are properly normalized and compared
        c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[2, 2])
        other = CategoricalHyperparameter("param", ["a", "b"])
        self.assertEqual(c1, other)

        c1 = CategoricalHyperparameter("param", ["a", "b"])
        other = CategoricalHyperparameter("param", ["a", "b"], weights=[2, 2])
        self.assertEqual(c1, other)

        c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2])
        other = CategoricalHyperparameter("param", ["a", "b"], weights=[10, 20])
        self.assertEqual(c1, other)

        # These result in different default values and are therefore different
        c1 = CategoricalHyperparameter("param", ["a", "b"])
        c2 = CategoricalHyperparameter("param", ["b", "a"])
        self.assertNotEqual(c1, c2)

        # Test that the order of the hyperparameter doesn't matter if the default is given
        c1 = CategoricalHyperparameter("param", ["a", "b"], default_value="a")
        c2 = CategoricalHyperparameter("param", ["b", "a"], default_value="a")
        self.assertEqual(c1, c2)

        # Test that the weights are ordered correctly
        c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2], default_value="a")
        c2 = CategoricalHyperparameter("param", ["b", "a"], weights=[2, 1], default_value="a")
        self.assertEqual(c1, c2)

        c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2], default_value="a")
        c2 = CategoricalHyperparameter("param", ["b", "a"], weights=[1, 2], default_value="a")
        self.assertNotEqual(c1, c2)

        c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2], default_value="a")
        c2 = CategoricalHyperparameter("param", ["b", "a"], default_value="a")
        self.assertNotEqual(c1, c2)

        c1 = CategoricalHyperparameter("param", ["a", "b"], default_value="a")
        c2 = CategoricalHyperparameter("param", ["b", "a"], weights=[1, 2], default_value="a")
        self.assertNotEqual(c1, c2)

        # Test that the equals operator does not fail accessing the weight of choice "a" in c2
        c1 = CategoricalHyperparameter("param", ["a", "b"], weights=[1, 2])
        c2 = CategoricalHyperparameter("param", ["b", "c"], weights=[1, 2])
        self.assertNotEqual(c1, c2)

    def test_categorical_strings(self):
        f1 = CategoricalHyperparameter("param", ["a", "b"])
        f1_ = CategoricalHyperparameter("param", ["a", "b"])
        self.assertEqual(f1, f1_)
        self.assertEqual("param, Type: Categorical, Choices: {a, b}, Default: a", str(f1))

    def test_categorical_is_legal(self):
        f1 = CategoricalHyperparameter("param", ["a", "b"])
        self.assertTrue(f1.is_legal("a"))
        self.assertTrue(f1.is_legal(u"a"))
        self.assertFalse(f1.is_legal("c"))
        self.assertFalse(f1.is_legal(3))

        # Test is legal vector
        self.assertTrue(f1.is_legal_vector(1.0))
        self.assertTrue(f1.is_legal_vector(0.0))
        self.assertTrue(f1.is_legal_vector(0))
        self.assertFalse(f1.is_legal_vector(0.3))
        self.assertFalse(f1.is_legal_vector(-0.1))
        self.assertRaises(TypeError, f1.is_legal_vector, "Hahaha")

    def test_categorical_choices(self):
        with self.assertRaisesRegex(
            ValueError,
            "Choices for categorical hyperparameters param contain choice 'a' 2 times, "
            "while only a single oocurence is allowed.",
        ):
            CategoricalHyperparameter('param', ['a', 'a'])

        with self.assertRaisesRegex(
            TypeError,
            "Choice 'None' is not supported",
        ):
            CategoricalHyperparameter('param', ['a', None])

    def test_sample_UniformFloatHyperparameter(self):
        # This can sample four distributions
        def sample(hp):
            rs = np.random.RandomState(1)
            counts_per_bin = [0 for i in range(21)]
            for i in range(100000):
                value = hp.sample(rs)
                if hp.log:
                    self.assertLessEqual(value, np.exp(hp._upper))
                    self.assertGreaterEqual(value, np.exp(hp._lower))
                else:
                    self.assertLessEqual(value, hp._upper)
                    self.assertGreaterEqual(value, hp._lower)
                index = int((value - hp.lower) / (hp.upper - hp.lower) * 20)
                counts_per_bin[index] += 1

            self.assertIsInstance(value, float)
            return counts_per_bin

        # Uniform
        hp = UniformFloatHyperparameter("ufhp", 0.5, 2.5)

        counts_per_bin = sample(hp)
        # The 21st bin is only filled if exactly 2.5 is sampled...very rare...
        for bin in counts_per_bin[:-1]:
            self.assertTrue(5200 > bin > 4800)
        self.assertEqual(sample(hp), sample(hp))

        # Quantized Uniform
        hp = UniformFloatHyperparameter("ufhp", 0.0, 1.0, q=0.1)

        counts_per_bin = sample(hp)
        for bin in counts_per_bin[::2]:
            self.assertTrue(9301 > bin > 8700)
        for bin in counts_per_bin[1::2]:
            self.assertEqual(bin, 0)
        self.assertEqual(sample(hp), sample(hp))

        # Log Uniform
        hp = UniformFloatHyperparameter("ufhp", 1.0, np.e ** 2, log=True)

        counts_per_bin = sample(hp)
        # print(counts_per_bin)
        self.assertEqual(counts_per_bin,
                         [14012, 10977, 8809, 7559, 6424, 5706, 5276, 4694,
                          4328, 3928, 3655, 3386, 3253, 2932, 2816, 2727, 2530,
                          2479, 2280, 2229, 0])
        self.assertEqual(sample(hp), sample(hp))

        # Quantized Log-Uniform
        # 7.2 ~ np.round(e * e, 1)
        hp = UniformFloatHyperparameter("ufhp", 1.2, 7.2, q=0.6, log=True)

        counts_per_bin = sample(hp)
        self.assertEqual(counts_per_bin,
                         [24359, 15781, 0, 11635, 0, 0, 9506, 7867, 0, 0, 6763,
                          0, 5919, 0, 5114, 4798, 0, 0, 4339, 0, 3919])
        self.assertEqual(sample(hp), sample(hp))

        # Issue #199
        hp = UniformFloatHyperparameter('uni_float_q', lower=1e-4, upper=1e-1, q=1e-5, log=True)
        self.assertTrue(np.isfinite(hp._lower))
        self.assertTrue(np.isfinite(hp._upper))
        sample(hp)

    def test_categorical_pdf(self):
        c1 = CategoricalHyperparameter('x1', choices=['one', 'two', 'three'], weights=[2, 1, 2])
        c2 = CategoricalHyperparameter('x1', choices=['one', 'two', 'three'], weights=[5, 0, 2])
        c3 = CategoricalHyperparameter('x1', choices=['one', 'two', 'three', 'four'])

        point_1 = np.array(['one'])
        point_2 = np.array(['two'])

        wrong_shape_1 = np.array([["one"]])
        wrong_shape_2 = np.array(["one", "two"]).reshape(1, -1)
        wrong_shape_3 = np.array(["one", "two"]).reshape(-1, 1)

        self.assertEqual(c1.pdf(point_1)[0], 0.4)
        self.assertEqual(c1.pdf(point_2)[0], 0.2)
        self.assertAlmostEqual(c2.pdf(point_1)[0], 0.7142857142857143)
        self.assertEqual(c2.pdf(point_2)[0], 0.0)
        self.assertEqual(c3.pdf(point_1)[0], 0.25)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')
        with self.assertRaises(TypeError):
            c1.pdf('one')
        with self.assertRaises(ValueError):
            c1.pdf(np.array(['zero']))

        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_1)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_2)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_3)

    def test_categorical__pdf(self):
        c1 = CategoricalHyperparameter('x1', choices=['one', 'two', 'three'], weights=[2, 1, 2])
        c2 = CategoricalHyperparameter('x1', choices=['one', 'two', 'three'], weights=[5, 0, 2])

        point_1 = np.array([0])
        point_2 = np.array([1])
        array_1 = np.array([1, 0, 2])
        self.assertEqual(c1._pdf(point_1)[0], 0.4)
        self.assertEqual(c1._pdf(point_2)[0], 0.2)
        self.assertAlmostEqual(c2._pdf(point_1)[0], 0.7142857142857143)
        self.assertEqual(c2._pdf(point_2)[0], 0.0)

        array_results = c1._pdf(array_1)
        expected_results = np.array([0.2, 0.4, 0.4])
        self.assertEqual(array_results.shape, expected_results.shape)
        for res, exp_res in zip(array_results, expected_results):
            self.assertEqual(res, exp_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1._pdf(0.2)
        with self.assertRaises(TypeError):
            c1._pdf('pdf')
        with self.assertRaises(TypeError):
            c1._pdf('one')
        with self.assertRaises(ValueError):
            c1._pdf(np.array(['zero']))

    def test_categorical_get_max_density(self):
        c1 = CategoricalHyperparameter('x1', choices=['one', 'two', 'three'], weights=[2, 1, 2])
        c2 = CategoricalHyperparameter('x1', choices=['one', 'two', 'three'], weights=[5, 0, 2])
        c3 = CategoricalHyperparameter('x1', choices=['one', 'two', 'three'])
        self.assertEqual(c1.get_max_density(), 0.4)
        self.assertEqual(c2.get_max_density(), 0.7142857142857143)
        self.assertAlmostEqual(c3.get_max_density(), 0.33333333333333)

    def test_sample_NormalFloatHyperparameter(self):
        hp = NormalFloatHyperparameter("nfhp", 0, 1)

        def actual_test():
            rs = np.random.RandomState(1)
            counts_per_bin = [0 for i in range(11)]
            for i in range(100000):
                value = hp.sample(rs)
                index = min(max(int((np.round(value + 0.5)) + 5), 0), 9)
                counts_per_bin[index] += 1

            self.assertEqual([0, 4, 138, 2113, 13394, 34104, 34282, 13683,
                              2136, 146, 0], counts_per_bin)

            self.assertIsInstance(value, float)
            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

    def test_sample_NormalFloatHyperparameter_with_bounds(self):
        hp = NormalFloatHyperparameter("nfhp", 0, 1, lower=-3, upper=3)

        def actual_test():
            rs = np.random.RandomState(1)
            counts_per_bin = [0 for i in range(11)]
            for i in range(100000):
                value = hp.sample(rs)
                index = min(max(int((np.round(value + 0.5)) + 5), 0), 9)
                counts_per_bin[index] += 1

            self.assertEqual([0, 0, 0, 2184, 13752, 34078, 34139, 13669,
                              2178, 0, 0], counts_per_bin)

            self.assertIsInstance(value, float)
            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

    def test_sample_BetaFloatHyperparameter(self):
        hp = BetaFloatHyperparameter("bfhp", alpha=8, beta=1.5, lower=-1, upper=10)

        def actual_test():
            rs = np.random.RandomState(1)
            counts_per_bin = [0 for i in range(11)]
            for i in range(1000):
                value = hp.sample(rs)
                index = np.floor(value).astype(int)
                counts_per_bin[index] += 1

            self.assertEqual([0, 2, 2, 4, 15, 39, 101, 193, 289, 355, 0], counts_per_bin)

            self.assertIsInstance(value, float)
            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

    def test_sample_UniformIntegerHyperparameter(self):
        # TODO: disentangle, actually test _sample and test sample on the
        # base class
        def sample(hp):
            rs = np.random.RandomState(1)
            counts_per_bin = [0 for i in range(21)]
            values = []
            for i in range(100000):
                value = hp.sample(rs)
                values.append(value)
                index = int(float(value - hp.lower) / (hp.upper - hp.lower) * 20)
                counts_per_bin[index] += 1

            self.assertIsInstance(value, int)
            return counts_per_bin

        # Quantized Uniform
        hp = UniformIntegerHyperparameter("uihp", 0, 10)

        counts_per_bin = sample(hp)
        # print(counts_per_bin)
        for bin in counts_per_bin[::2]:
            self.assertTrue(9302 > bin > 8700)
        for bin in counts_per_bin[1::2]:
            self.assertEqual(bin, 0)
        self.assertEqual(sample(hp), sample(hp))

    def test__sample_UniformIntegerHyperparameter(self):
        hp = UniformIntegerHyperparameter("uihp", 0, 10)
        values = []
        rs = np.random.RandomState(1)
        for i in range(100):
            values.append(hp._sample(rs))
        self.assertEqual(len(np.unique(values)), 11)

        hp = UniformIntegerHyperparameter("uihp", 2, 12)
        values = []
        rs = np.random.RandomState(1)
        for i in range(100):
            values.append(hp._sample(rs))
            self.assertGreaterEqual(hp._transform(values[-1]), 2)
            self.assertLessEqual(hp._transform(values[-1]), 12)
        self.assertEqual(len(np.unique(values)), 11)

    def test_quantization_UniformIntegerHyperparameter(self):
        hp = UniformIntegerHyperparameter("uihp", 1, 100, q=3)
        rs = np.random.RandomState()

        sample_one = hp._sample(rs=rs, size=1)
        self.assertIsInstance(obj=sample_one, cls=np.ndarray)
        self.assertEqual(1, sample_one.size)
        self.assertEqual((hp._transform(sample_one) - 1) % 3, 0)
        self.assertGreaterEqual(hp._transform(sample_one), 1)
        self.assertLessEqual(hp._transform(sample_one), 100)

        sample_hundred = hp._sample(rs=rs, size=100)
        self.assertIsInstance(obj=sample_hundred, cls=np.ndarray)
        self.assertEqual(100, sample_hundred.size)
        np.testing.assert_array_equal(
            [(hp._transform(val) - 1) % 3 for val in sample_hundred],
            np.zeros((100,), dtype=int),
        )
        samples_in_original_space = hp._transform(sample_hundred)
        for i in range(100):
            self.assertGreaterEqual(samples_in_original_space[i], 1)
            self.assertLessEqual(samples_in_original_space[i], 100)

    def test_quantization_UniformIntegerHyperparameter_negative(self):
        hp = UniformIntegerHyperparameter("uihp", -2, 100, q=3)
        rs = np.random.RandomState()

        sample_one = hp._sample(rs=rs, size=1)
        self.assertIsInstance(obj=sample_one, cls=np.ndarray)
        self.assertEqual(1, sample_one.size)
        self.assertEqual((hp._transform(sample_one) + 2) % 3, 0)
        self.assertGreaterEqual(hp._transform(sample_one), -2)
        self.assertLessEqual(hp._transform(sample_one), 100)

        sample_hundred = hp._sample(rs=rs, size=100)
        self.assertIsInstance(obj=sample_hundred, cls=np.ndarray)
        self.assertEqual(100, sample_hundred.size)
        np.testing.assert_array_equal(
            [(hp._transform(val) + 2) % 3 for val in sample_hundred],
            np.zeros((100, ), dtype=int),
        )
        samples_in_original_space = hp._transform(sample_hundred)
        for i in range(100):
            self.assertGreaterEqual(samples_in_original_space[i], -2)
            self.assertLessEqual(samples_in_original_space[i], 100)

    def test_illegal_quantization_UniformIntegerHyperparameter(self):
        with self.assertRaisesRegex(
            ValueError,
            r'Upper bound \(4\) - lower bound \(1\) must be a multiple of q \(2\)',
        ):
            UniformIntegerHyperparameter("uihp", 1, 4, q=2)

    def test_quantization_UniformFloatHyperparameter(self):
        hp = UniformFloatHyperparameter("ufhp", 1, 100, q=3)
        rs = np.random.RandomState()

        sample_one = hp._sample(rs=rs, size=1)
        self.assertIsInstance(obj=sample_one, cls=np.ndarray)
        self.assertEqual(1, sample_one.size)
        self.assertEqual((hp._transform(sample_one) - 1) % 3, 0)
        self.assertGreaterEqual(hp._transform(sample_one), 1)
        self.assertLessEqual(hp._transform(sample_one), 100)

        sample_hundred = hp._sample(rs=rs, size=100)
        self.assertIsInstance(obj=sample_hundred, cls=np.ndarray)
        self.assertEqual(100, sample_hundred.size)
        np.testing.assert_array_equal(
            [(hp._transform(val) - 1) % 3 for val in sample_hundred],
            np.zeros((100,), dtype=int),
        )
        samples_in_original_space = hp._transform(sample_hundred)
        for i in range(100):
            self.assertGreaterEqual(samples_in_original_space[i], 1)
            self.assertLessEqual(samples_in_original_space[i], 100)

    def test_quantization_UniformFloatHyperparameter_decimal_numbers(self):
        hp = UniformFloatHyperparameter("ufhp", 1.2, 3.6, q=0.2)
        rs = np.random.RandomState()

        sample_one = hp._sample(rs=rs, size=1)
        self.assertIsInstance(obj=sample_one, cls=np.ndarray)
        self.assertEqual(1, sample_one.size)
        try:
            self.assertAlmostEqual(float(hp._transform(sample_one) + 1.2) % 0.2, 0.0)
        except Exception:
            self.assertAlmostEqual(float(hp._transform(sample_one) + 1.2) % 0.2, 0.2)
        self.assertGreaterEqual(hp._transform(sample_one), 1)
        self.assertLessEqual(hp._transform(sample_one), 100)

    def test_quantization_UniformFloatHyperparameter_decimal_numbers_negative(self):
        hp = UniformFloatHyperparameter("ufhp", -1.2, 1.2, q=0.2)
        rs = np.random.RandomState()

        sample_one = hp._sample(rs=rs, size=1)
        self.assertIsInstance(obj=sample_one, cls=np.ndarray)
        self.assertEqual(1, sample_one.size)
        try:
            self.assertAlmostEqual(float(hp._transform(sample_one) + 1.2) % 0.2, 0.0)
        except Exception:
            self.assertAlmostEqual(float(hp._transform(sample_one) + 1.2) % 0.2, 0.2)
        self.assertGreaterEqual(hp._transform(sample_one), -1.2)
        self.assertLessEqual(hp._transform(sample_one), 1.2)

    def test_sample_NormalIntegerHyperparameter(self):
        def sample(hp):
            lower = -30
            upper = 30
            rs = np.random.RandomState(1)
            counts_per_bin = [0 for i in range(21)]
            for i in range(100000):
                value = hp.sample(rs)
                sample = float(value)
                if sample < lower:
                    sample = lower
                if sample > upper:
                    sample = upper
                index = int((sample - lower) / (upper - lower) * 20)
                counts_per_bin[index] += 1

            self.assertIsInstance(value, int)
            return counts_per_bin

        hp = NormalIntegerHyperparameter("nihp", 0, 10)
        self.assertEqual(sample(hp),
                         [305, 422, 835, 1596, 2682, 4531, 6572, 8670, 10649,
                          11510, 11854, 11223, 9309, 7244, 5155, 3406, 2025,
                          1079, 514, 249, 170])
        self.assertEqual(sample(hp), sample(hp))

    def test__sample_NormalIntegerHyperparameter(self):
        # mean zero, std 1
        hp = NormalIntegerHyperparameter("uihp", 0, 1)
        values = []
        rs = np.random.RandomState(1)
        for i in range(100):
            values.append(hp._sample(rs))
        self.assertEqual(len(np.unique(values)), 5)

    def test_sample_BetaIntegerHyperparameter(self):
        hp = BetaIntegerHyperparameter("bihp", alpha=4, beta=4, lower=0, upper=10)

        def actual_test():
            rs = np.random.RandomState(1)
            counts_per_bin = [0 for i in range(11)]
            for i in range(1000):
                value = hp.sample(rs)
                counts_per_bin[value] += 1

            # The chosen distribution is symmetric, so we expect to see a symmetry in the bins
            self.assertEqual([1, 23, 82, 121, 174, 197, 174, 115, 86, 27, 0], counts_per_bin)

            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

    def test_sample_CategoricalHyperparameter(self):
        hp = CategoricalHyperparameter("chp", [0, 2, "Bla", u"Blub"])

        def actual_test():
            rs = np.random.RandomState(1)
            counts_per_bin = defaultdict(int)
            for i in range(10000):
                value = hp.sample(rs)
                counts_per_bin[value] += 1

            self.assertEqual(
                {0: 2539, 2: 2451, 'Bla': 2549, 'Blub': 2461},
                dict(counts_per_bin.items()))
            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

    def test_sample_CategoricalHyperparameter_with_weights(self):
        # check also that normalization works
        hp = CategoricalHyperparameter("chp", [0, 2, "Bla", u"Blub", u"Blurp"],
                                       weights=[1, 2, 3, 4, 0])
        np.testing.assert_almost_equal(
            actual=hp.probabilities,
            desired=[0.1, 0.2, 0.3, 0.4, 0],
            decimal=3
        )

        def actual_test():
            rs = np.random.RandomState(1)
            counts_per_bin = defaultdict(int)
            for i in range(10000):
                value = hp.sample(rs)
                counts_per_bin[value] += 1

            self.assertEqual(
                {0: 1003, 2: 2061, 'Bla': 2994, u'Blub': 3942},
                dict(counts_per_bin.items()))
            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

    def test_categorical_copy_with_weights(self):
        orig_hp = CategoricalHyperparameter(
            name="param",
            choices=[1, 2, 3],
            default_value=2,
            weights=[1, 3, 6]
        )
        copy_hp = copy.copy(orig_hp)

        self.assertEqual(copy_hp.name, orig_hp.name)
        self.assertTupleEqual(copy_hp.choices, orig_hp.choices)
        self.assertEqual(copy_hp.default_value, orig_hp.default_value)
        self.assertEqual(copy_hp.num_choices, orig_hp.num_choices)
        self.assertTupleEqual(copy_hp.probabilities, orig_hp.probabilities)

    def test_categorical_copy_without_weights(self):
        orig_hp = CategoricalHyperparameter(
            name="param",
            choices=[1, 2, 3],
            default_value=2
        )
        copy_hp = copy.copy(orig_hp)

        self.assertEqual(copy_hp.name, orig_hp.name)
        self.assertTupleEqual(copy_hp.choices, orig_hp.choices)
        self.assertEqual(copy_hp.default_value, orig_hp.default_value)
        self.assertEqual(copy_hp.num_choices, orig_hp.num_choices)
        self.assertTupleEqual(copy_hp.probabilities, (
            0.3333333333333333, 0.3333333333333333, 0.3333333333333333))
        self.assertTupleEqual(orig_hp.probabilities, (
            0.3333333333333333, 0.3333333333333333, 0.3333333333333333))

    def test_categorical_with_weights(self):
        rs = np.random.RandomState()

        cat_hp_str = CategoricalHyperparameter(
            name="param",
            choices=["A", "B", "C"],
            default_value="A",
            weights=[0.1, 0.6, 0.3]
        )
        for _ in range(1000):
            self.assertIn(member=cat_hp_str.sample(rs), container=["A", "B", "C"])

        cat_hp_int = CategoricalHyperparameter(
            name="param",
            choices=[1, 2, 3],
            default_value=2,
            weights=[0.1, 0.3, 0.6]
        )
        for _ in range(1000):
            self.assertIn(member=cat_hp_int.sample(rs), container=[1, 3, 2])

        cat_hp_float = CategoricalHyperparameter(
            name="param",
            choices=[-0.1, 0.0, 0.3],
            default_value=0.3,
            weights=[10, 60, 30]
        )
        for _ in range(1000):
            self.assertIn(member=cat_hp_float.sample(rs), container=[-0.1, 0.0, 0.3])

    def test_categorical_with_some_zero_weights(self):
        # zero weights are okay as long as there is at least one strictly positive weight

        rs = np.random.RandomState()

        cat_hp_str = CategoricalHyperparameter(
            name="param",
            choices=["A", "B", "C"],
            default_value="A",
            weights=[0.1, 0.0, 0.3]
        )
        for _ in range(1000):
            self.assertIn(member=cat_hp_str.sample(rs), container=["A", "C"])
        np.testing.assert_almost_equal(
            actual=cat_hp_str.probabilities,
            desired=[0.25, 0., 0.75],
            decimal=3
        )

        cat_hp_int = CategoricalHyperparameter(
            name="param",
            choices=[1, 2, 3],
            default_value=2,
            weights=[0.1, 0.6, 0.0]
        )
        for _ in range(1000):
            self.assertIn(member=cat_hp_int.sample(rs), container=[1, 2])
        np.testing.assert_almost_equal(
            actual=cat_hp_int.probabilities,
            desired=[0.1429, 0.8571, 0.0],
            decimal=3
        )

        cat_hp_float = CategoricalHyperparameter(
            name="param",
            choices=[-0.1, 0.0, 0.3],
            default_value=0.3,
            weights=[0.0, 0.6, 0.3]
        )
        for _ in range(1000):
            self.assertIn(member=cat_hp_float.sample(rs), container=[0.0, 0.3])
        np.testing.assert_almost_equal(
            actual=cat_hp_float.probabilities,
            desired=[0.00, 0.6667, 0.3333],
            decimal=3
        )

    def test_categorical_with_all_zero_weights(self):
        with self.assertRaisesRegex(ValueError, 'At least one weight has to be strictly positive.'):
            CategoricalHyperparameter(
                name="param",
                choices=["A", "B", "C"],
                default_value="A",
                weights=[0.0, 0.0, 0.0]
            )

    def test_categorical_with_wrong_length_weights(self):
        with self.assertRaisesRegex(
                ValueError,
                'The list of weights and the list of choices are required to be of same length.'):
            CategoricalHyperparameter(
                name="param",
                choices=["A", "B", "C"],
                default_value="A",
                weights=[0.1, 0.3]
            )

        with self.assertRaisesRegex(
                ValueError,
                'The list of weights and the list of choices are required to be of same length.'):
            CategoricalHyperparameter(
                name="param",
                choices=["A", "B", "C"],
                default_value="A",
                weights=[0.1, 0.0, 0.5, 0.3]
            )

    def test_categorical_with_negative_weights(self):
        with self.assertRaisesRegex(ValueError, 'Negative weights are not allowed.'):
            CategoricalHyperparameter(
                name="param",
                choices=["A", "B", "C"],
                default_value="A",
                weights=[0.1, -0.1, 0.3]
            )

    def test_categorical_with_set(self):
        with self.assertRaisesRegex(TypeError, 'Using a set of choices is prohibited.'):
            CategoricalHyperparameter(
                name="param",
                choices={"A", "B", "C"},
                default_value="A",
            )

        with self.assertRaisesRegex(TypeError, 'Using a set of weights is prohibited.'):
            CategoricalHyperparameter(
                name="param",
                choices=["A", "B", "C"],
                default_value="A",
                weights={0.2, 0.6, 0.8},
            )

    def test_log_space_conversion(self):
        lower, upper = 1e-5, 1e5
        hyper = UniformFloatHyperparameter('test', lower=lower, upper=upper, log=True)
        self.assertTrue(hyper.is_legal(hyper._transform(1.)))

        lower, upper = 1e-10, 1e10
        hyper = UniformFloatHyperparameter('test', lower=lower, upper=upper, log=True)
        self.assertTrue(hyper.is_legal(hyper._transform(1.)))

    def test_ordinal_attributes_accessible(self):
        f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
        self.assertEqual(f1.name, "temp")
        self.assertTupleEqual(f1.sequence, ("freezing", "cold", "warm", "hot"))
        self.assertEqual(f1.num_elements, 4)
        self.assertEqual(f1.default_value, "freezing")
        self.assertEqual(f1.normalized_default_value, 0)

    def test_ordinal_is_legal(self):
        f1 = OrdinalHyperparameter("temp",
                                   ["freezing", "cold", "warm", "hot"])
        self.assertTrue(f1.is_legal("warm"))
        self.assertTrue(f1.is_legal(u"freezing"))
        self.assertFalse(f1.is_legal("chill"))
        self.assertFalse(f1.is_legal(2.5))
        self.assertFalse(f1.is_legal("3"))

        # Test is legal vector
        self.assertTrue(f1.is_legal_vector(1.0))
        self.assertTrue(f1.is_legal_vector(0.0))
        self.assertTrue(f1.is_legal_vector(0))
        self.assertTrue(f1.is_legal_vector(3))
        self.assertFalse(f1.is_legal_vector(-0.1))
        self.assertRaises(TypeError, f1.is_legal_vector, "Hahaha")

    def test_ordinal_check_order(self):
        f1 = OrdinalHyperparameter("temp",
                                   ["freezing", "cold", "warm", "hot"])
        self.assertTrue(f1.check_order("freezing", "cold"))
        self.assertTrue(f1.check_order("freezing", "hot"))
        self.assertFalse(f1.check_order("hot", "cold"))
        self.assertFalse(f1.check_order("hot", "warm"))

    def test_ordinal_get_value(self):
        f1 = OrdinalHyperparameter("temp",
                                   ["freezing", "cold", "warm", "hot"])
        self.assertEqual(f1.get_value(3), "hot")
        self.assertNotEqual(f1.get_value(1), "warm")

    def test_ordinal_get_order(self):
        f1 = OrdinalHyperparameter("temp",
                                   ["freezing", "cold", "warm", "hot"])
        self.assertEqual(f1.get_order("warm"), 2)
        self.assertNotEqual(f1.get_order("freezing"), 3)

    def test_ordinal_get_seq_order(self):
        f1 = OrdinalHyperparameter("temp",
                                   ["freezing", "cold", "warm", "hot"])
        self.assertEqual(tuple(f1.get_seq_order()), tuple([0, 1, 2, 3]))

    def test_ordinal_get_neighbors(self):
        f1 = OrdinalHyperparameter("temp",
                                   ["freezing", "cold", "warm", "hot"])
        self.assertEqual(f1.get_neighbors(0, rs=None), [1])
        self.assertEqual(f1.get_neighbors(1, rs=None), [0, 2])
        self.assertEqual(f1.get_neighbors(3, rs=None), [2])
        self.assertEqual(f1.get_neighbors("hot", transform=True, rs=None), ["warm"])
        self.assertEqual(f1.get_neighbors("cold", transform=True, rs=None), ["freezing", "warm"])

    def test_get_num_neighbors(self):
        f1 = OrdinalHyperparameter("temp",
                                   ["freezing", "cold", "warm", "hot"])
        self.assertEqual(f1.get_num_neighbors("freezing"), 1)
        self.assertEqual(f1.get_num_neighbors("hot"), 1)
        self.assertEqual(f1.get_num_neighbors("cold"), 2)

    def test_ordinal_get_size(self):
        f1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
        self.assertEqual(f1.get_size(), 4)

    def test_ordinal_pdf(self):
        c1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
        point_1 = np.array(["freezing"])
        point_2 = np.array(["warm"])
        array_1 = np.array(["freezing", "warm"])

        wrong_shape_1 = np.array([["freezing"]])
        wrong_shape_2 = np.array(["freezing", "warm"]).reshape(1, -1)
        wrong_shape_3 = np.array(["freezing", "warm"]).reshape(-1, 1)

        self.assertEqual(c1.pdf(point_1)[0], 0.25)
        self.assertEqual(c1.pdf(point_2)[0], 0.25)

        array_results = c1.pdf(array_1)
        expected_results = np.array([0.25, 0.25])
        self.assertEqual(array_results.shape, expected_results.shape)
        for res, exp_res in zip(array_results, expected_results):
            self.assertEqual(res, exp_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1.pdf(0.2)
        with self.assertRaises(TypeError):
            c1.pdf('pdf')
        with self.assertRaises(TypeError):
            c1.pdf('one')
        with self.assertRaises(ValueError):
            c1.pdf(np.array(['zero']))

        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_1)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_2)
        with self.assertRaisesRegex(ValueError, "Method pdf expects a one-dimensional numpy array"):
            c1.pdf(wrong_shape_3)

    def test_ordinal__pdf(self):
        c1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
        point_1 = np.array(["freezing"])
        point_2 = np.array(["warm"])
        array_1 = np.array(["freezing", "warm"])
        self.assertEqual(c1._pdf(point_1)[0], 0.25)
        self.assertEqual(c1._pdf(point_2)[0], 0.25)

        array_results = c1._pdf(array_1)
        expected_results = np.array([0.25, 0.25])
        self.assertEqual(array_results.shape, expected_results.shape)
        for res, exp_res in zip(array_results, expected_results):
            self.assertEqual(res, exp_res)

        # pdf must take a numpy array
        with self.assertRaises(TypeError):
            c1._pdf(0.2)
        with self.assertRaises(TypeError):
            c1._pdf('pdf')
        with self.assertRaises(TypeError):
            c1._pdf('one')
        with self.assertRaises(ValueError):
            c1._pdf(np.array(['zero']))

    def test_ordinal_get_max_density(self):
        c1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
        c2 = OrdinalHyperparameter("temp", ["freezing", "cold"])
        self.assertEqual(c1.get_max_density(), 0.25)
        self.assertEqual(c2.get_max_density(), 0.5)

    def test_rvs(self):
        f1 = UniformFloatHyperparameter("param", 0, 10)

        # test that returned types are correct
        # if size=None, return a value, but if size=1, return a 1-element array
        self.assertIsInstance(f1.rvs(), float)
        self.assertIsInstance(f1.rvs(size=1), np.ndarray)
        self.assertIsInstance(f1.rvs(size=2), np.ndarray)

        self.assertAlmostEqual(f1.rvs(random_state=100), f1.rvs(random_state=100))
        self.assertAlmostEqual(
            f1.rvs(random_state=100),
            f1.rvs(random_state=np.random.RandomState(100))
        )
        f1.rvs(random_state=np.random)
        f1.rvs(random_state=np.random.default_rng(1))
        self.assertRaises(ValueError, f1.rvs, 1, "a")

    def test_hyperparam_representation(self):
        # Float
        f1 = UniformFloatHyperparameter("param", 1, 100, log=True)
        self.assertEqual(
            "param, Type: UniformFloat, Range: [1.0, 100.0], Default: 10.0, on log-scale",
            repr(f1)
        )
        f2 = NormalFloatHyperparameter("param", 8, 99.1, log=False)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 8.0 Sigma: 99.1, Default: 8.0",
            repr(f2)
        )
        f3 = NormalFloatHyperparameter("param", 8, 99.1, log=False, lower=1, upper=16)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 8.0 Sigma: 99.1, Range: [1.0, 16.0], Default: 8.0",
            repr(f3)
        )
        i1 = UniformIntegerHyperparameter("param", 0, 100)
        self.assertEqual(
            "param, Type: UniformInteger, Range: [0, 100], Default: 50",
            repr(i1)
        )
        i2 = NormalIntegerHyperparameter("param", 5, 8)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 5 Sigma: 8, Default: 5",
            repr(i2)
        )
        i3 = NormalIntegerHyperparameter("param", 5, 8, lower=1, upper=10)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 5 Sigma: 8, Range: [1, 10], Default: 5",
            repr(i3)
        )
        o1 = OrdinalHyperparameter("temp", ["freezing", "cold", "warm", "hot"])
        self.assertEqual(
            "temp, Type: Ordinal, Sequence: {freezing, cold, warm, hot}, Default: freezing",
            repr(o1)
        )
        c1 = CategoricalHyperparameter("param", [True, False])
        self.assertEqual(
            "param, Type: Categorical, Choices: {True, False}, Default: True",
            repr(c1)
        )
