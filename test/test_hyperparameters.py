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
import unittest
import pytest

import numpy as np

from ConfigSpace.hyperparameters import Constant, \
    UniformFloatHyperparameter, NormalFloatHyperparameter, \
    UniformIntegerHyperparameter, NormalIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter


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
            ValueError, "Upper bound 1.000000 must be larger than lower bound "
            "0.000000 for hyperparameter param", UniformFloatHyperparameter,
            "param", 1, 0)

    def test_normalfloat(self):
        # TODO test non-equality
        f1 = NormalFloatHyperparameter("param", 0.5, 10.5)
        f1_ = NormalFloatHyperparameter("param", 0.5, 10.5)
        self.assertEqual(f1, f1_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.5 Sigma: 10.5, Default: 0.5",
            str(f1))

        # Test attributes are accessible
        self.assertEqual(f1.name, "param")
        self.assertAlmostEqual(f1.mu, 0.5)
        self.assertAlmostEqual(f1.sigma, 10.5)
        self.assertAlmostEqual(f1.q, None)
        self.assertEqual(f1.log, False)
        self.assertAlmostEqual(f1.default_value, 0.5)
        self.assertAlmostEqual(f1.normalized_default_value, 0.5)

        f2 = NormalFloatHyperparameter("param", 0, 10, q=0.1)
        f2_ = NormalFloatHyperparameter("param", 0, 10, q=0.1)
        self.assertEqual(f2, f2_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 0.0, "
            "Q: 0.1", str(f2))

        with pytest.warns(RuntimeWarning, match='divide by zero encountered in log'):
            f3 = NormalFloatHyperparameter("param", 0, 10, log=True)
        with pytest.warns(RuntimeWarning, match='divide by zero encountered in log'):
            f3_ = NormalFloatHyperparameter("param", 0, 10, log=True)
        self.assertEqual(f3, f3_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 0.0, "
            "on log-scale", str(f3))

        f4 = NormalFloatHyperparameter("param", 0, 10, default_value=1.0)
        f4_ = NormalFloatHyperparameter("param", 0, 10, default_value=1.0)
        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 1.0",
            str(f4))

        f5 = NormalFloatHyperparameter("param", 0, 10, default_value=1.0,
                                       q=0.1, log=True)
        f5_ = NormalFloatHyperparameter("param", 0, 10, default_value=1.0,
                                        q=0.1, log=True)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 1.0, "
            "on log-scale, Q: 0.1", str(f5))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

        # test that meta-data is stored correctly
        f_meta = NormalFloatHyperparameter("param", 0.1, 10, q=0.1, log=True,
                                           default_value=1.0, meta=dict(self.meta_data))
        self.assertEqual(f_meta.meta, self.meta_data)

    def test_normalfloat_to_uniformfloat(self):
        f1 = NormalFloatHyperparameter("param", 0, 10, q=0.1)
        f1_expected = UniformFloatHyperparameter("param", -30, 30, q=0.1)
        f1_actual = f1.to_uniform()
        self.assertEqual(f1_expected, f1_actual)

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
        self.assertAlmostEqual(f1.normalized_default_value, (2.0+0.49999)/(5.49999+0.49999))

        quantization_warning = "Setting quantization < 1 for Integer Hyperparameter 'param' has no effect"
        with pytest.warns(UserWarning, match=quantization_warning):
            f2 = UniformIntegerHyperparameter("param", 0, 10, q=0.1)
        with pytest.warns(UserWarning, match=quantization_warning):
            f2_ = UniformIntegerHyperparameter("param", 0, 10, q=0.1)
        self.assertEqual(f2, f2_)
        self.assertEqual(
            "param, Type: UniformInteger, Range: [0, 10], Default: 5",
            str(f2))

        # f2_large_q = UniformIntegerHyperparameter("param", 0, 10, q=2)
        # f2_large_q_ = UniformIntegerHyperparameter("param", 0, 10, q=2)
        # self.assertEqual(f2_large_q, f2_large_q_)
        # self.assertEqual(
        #    "param, Type: UniformInteger, Range: [0, 10], Default: 5, Q: 2",
        #    str(f2_large_q))

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

        with pytest.warns(RuntimeWarning, match='divide by zero encountered in log'):
            f3 = NormalIntegerHyperparameter("param", 0, 10, log=True)
        with pytest.warns(RuntimeWarning, match='divide by zero encountered in log'):
            f3_ = NormalIntegerHyperparameter("param", 0, 10, log=True)
        self.assertEqual(f3, f3_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 0, "
            "on log-scale", str(f3))

        f4 = NormalIntegerHyperparameter("param", 0, 10, default_value=1, log=True)
        f4_ = NormalIntegerHyperparameter("param", 0, 10, default_value=1, log=True)
        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 1, "
            "on log-scale", str(f4))

        with pytest.warns(RuntimeWarning, match='divide by zero encountered in log'):
            f5 = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
        with pytest.warns(RuntimeWarning, match='divide by zero encountered in log'):
            f5_ = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 0, "
            "on log-scale", str(f5))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

        # test that meta-data is stored correctly
        f_meta = NormalIntegerHyperparameter("param", 0, 10, default_value=1, log=True,
                                             meta=dict(self.meta_data))
        self.assertEqual(f_meta.meta, self.meta_data)

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
        with pytest.warns(RuntimeWarning, match='divide by zero encountered in log'):
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
                          0, 5919, 5114, 0, 4798, 0, 0, 4339, 3919, 0])
        self.assertEqual(sample(hp), sample(hp))

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
                index = int(float(value - hp.lower) /
                            (hp.upper - hp.lower) * 20)
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

    def test_sample_CategoricalHyperparameter(self):
        hp = CategoricalHyperparameter("chp", [0, 2, "Bla", u"Blub"])

        def actual_test():
            rs = np.random.RandomState(1)
            counts_per_bin = defaultdict(int)
            for i in range(10000):
                value = hp.sample(rs)
                counts_per_bin[value] += 1

            self.assertEqual(
                {0: 2456, 2: 2485, 'Bla': 2550, u'Blub': 2509},
                dict(counts_per_bin.items()))
            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

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
