import unittest
import warnings

from HPOlibConfigSpace.hyperparameters import Constant, \
    UniformFloatHyperparameter, NormalFloatHyperparameter, \
    UniformIntegerHyperparameter, NormalIntegerHyperparameter, \
    CategoricalHyperparameter


class TestHyperparameters(unittest.TestCase):
    def test_constant(self):
        # Test construction
        c1 = Constant("value", 1)
        c2 = Constant("value", 1)
        c3 = Constant("value", 2)
        c4 = Constant("valuee", 1)
        c5 = Constant("valueee", 2)

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

        f4 = UniformFloatHyperparameter("param", 0, 10, default=1.0)
        f4_ = UniformFloatHyperparameter("param", 0, 10, default=1.0)
        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: UniformFloat, Range: [0.0, 10.0], Default: 1.0",
            str(f4))

        f5 = UniformFloatHyperparameter("param", 0.1, 10, q=0.1, log=True,
                                        default=1.0)
        f5_ = UniformFloatHyperparameter("param", 0.1, 10, q=0.1, log=True,
                                         default=1.0)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: UniformFloat, Range: [0.1, 10.0], Default: 1.0, "
            "on log-scale, Q: 0.1", str(f5))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

    def test_uniformfloat_to_integer(self):
        f1 = UniformFloatHyperparameter("param", 1, 10, q=0.1, log=True)
        with warnings.catch_warnings():
            f2 = f1.to_integer()
            warnings.simplefilter("ignore")
            # TODO is this a useful rounding?
            # TODO should there be any rounding, if e.g. lower=0.1
            self.assertEqual("param, Type: UniformInteger, Range: [1, 10], "
                             "Default: 3, on log-scale", str(f2))

    def test_uniformfloat_is_legal(self):
        f1 = UniformFloatHyperparameter("param", 0.1, 10, q=0.1, log=True)
        self.assertTrue(f1.is_legal(3.0))
        self.assertTrue(f1.is_legal(3))
        self.assertFalse(f1.is_legal(-0.1))
        self.assertFalse(f1.is_legal(10.1))
        self.assertFalse(f1.is_legal("AAA"))
        self.assertFalse(f1.is_legal(dict()))

    def test_uniformfloat_illegal_bounds(self):
        self.assertRaisesRegexp(ValueError,
            "Negative lower bound \(0.000000\) for log-scale hyperparameter "
            "param is forbidden.", UniformFloatHyperparameter, "param", 0, 10,
            q=0.1, log=True)

        self.assertRaisesRegexp(ValueError,
            "Upper bound 1.000000 must be larger than lower bound "
            "0.000000 for hyperparameter param",
                                UniformFloatHyperparameter, "param", 1, 0)

        self.assertRaisesRegexp(ValueError,
            "If q is active, the upper bound 1.000000 must be a multiple of q "
            "0.300000!", UniformFloatHyperparameter, "param", 0, 1, q=0.3)

        self.assertRaisesRegexp(ValueError,
            "If q is active, the lower bound 0.100000 must be a multiple of q "
            "0.300000!", UniformFloatHyperparameter, "param", 0.1, 1, q=0.3)

    def test_normalfloat(self):
        # TODO test non-equality
        f1 = NormalFloatHyperparameter("param", 0.5, 10.5)
        f1_ = NormalFloatHyperparameter("param", 0.5, 10.5)
        self.assertEqual(f1, f1_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.5 Sigma: 10.5, Default: 0.5",
            str(f1))

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
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 0.0, "
            "on log-scale", str(f3))

        f4 = NormalFloatHyperparameter("param", 0, 10, default=1.0)
        f4_ = NormalFloatHyperparameter("param", 0, 10, default=1.0)
        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 1.0",
            str(f4))

        f5 = NormalFloatHyperparameter("param", 0, 10, default=1.0,
                                       q=0.1, log=True)
        f5_ = NormalFloatHyperparameter("param", 0, 10, default=1.0,
                                        q=0.1, log=True)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: NormalFloat, Mu: 0.0 Sigma: 10.0, Default: 1.0, "
            "on log-scale, Q: 0.1", str(f5))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

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

        f2 = UniformIntegerHyperparameter("param", 0, 10, q=0.1)
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

        f4 = UniformIntegerHyperparameter("param", 1, 10, default=1, log=True)
        f4_ = UniformIntegerHyperparameter("param", 1, 10, default=1, log=True)
        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: UniformInteger, Range: [1, 10], Default: 1, "
            "on log-scale", str(f4))

        f5 = UniformIntegerHyperparameter("param", 1, 10, default=1, q=0.1,\
                                                                     log=True)
        f5_ = UniformIntegerHyperparameter("param", 1, 10, default=1, q=0.1,\
                                                                      log=True)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: UniformInteger, Range: [1, 10], Default: 1, "
            "on log-scale", str(f5))

        self.assertNotEqual(f2, f2_large_q)
        self.assertNotEqual(f1, "UniformFloat")

    def test_uniformint_legal_float_values(self):
        n_iter = UniformIntegerHyperparameter("n_iter", 5., 1000., default=20.0)

        self.assertIsInstance(n_iter.default, int)
        self.assertRaisesRegexp(ValueError, "For the Integer parameter n_iter, "
                                            "the value must be an Integer, too."
                                            " Right now it is a <type 'float'>"
                                            " with value 20.5.",
                                UniformIntegerHyperparameter,"n_iter", 5.,
                                1000., default=20.5)

        self.assertRaisesRegexp(ValueError, "For the Integer parameter n_iter, "
                                            "the value must be an Integer, too. "
                                            "Right now it is a <type 'float'> "
                                            "with value 350.5.",
                                n_iter.instantiate, 350.5)
        self.assertIsInstance(n_iter.instantiate(20.0).value, int)

    def test_uniformint_illegal_bounds(self):
        self.assertRaisesRegexp(ValueError,
            "Negative lower bound \(0\) for log-scale hyperparameter "
            "param is forbidden.", UniformIntegerHyperparameter, "param", 0, 10,
            log=True)

        self.assertRaisesRegexp(ValueError,
            "Upper bound 1 must be larger than lower bound 0 for "
            "hyperparameter param", UniformIntegerHyperparameter, "param", 1, 0)

        self.assertRaisesRegexp(ValueError,
            "If q is active, the upper bound 1 must be a multiple of q "
            "2!", UniformIntegerHyperparameter, "param", 0, 1, q=2)

        self.assertRaisesRegexp(ValueError,
            "If q is active, the lower bound 1 must be a multiple of q "
            "2!", UniformIntegerHyperparameter, "param", 1, 3, q=2)


    def test_normalint(self):
        # TODO test for unequal!
        f1 = NormalIntegerHyperparameter("param", 0.5, 5.5)
        f1_ = NormalIntegerHyperparameter("param", 0.5, 5.5)
        self.assertEqual(f1, f1_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0.5 Sigma: 5.5, Default: 0.5",
            str(f1))

        f2 = NormalIntegerHyperparameter("param", 0, 10, q=0.1)
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
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 0, "
            "on log-scale", str(f3))

        f4 = NormalIntegerHyperparameter("param", 0, 10, default=1, log=True)
        f4_ = NormalIntegerHyperparameter("param", 0, 10, default=1, log=True)
        self.assertEqual(f4, f4_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 1, "
            "on log-scale", str(f4))

        f5 = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
        f5_ = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
        self.assertEqual(f5, f5_)
        self.assertEqual(
            "param, Type: NormalInteger, Mu: 0 Sigma: 10, Default: 0, "
            "on log-scale", str(f5))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

    def test_normalint_legal_float_values(self):
        n_iter = NormalIntegerHyperparameter("n_iter", 0, 1., default=2.0)
        self.assertIsInstance(n_iter.default, int)
        self.assertRaisesRegexp(ValueError, "For the Integer parameter n_iter, "
                                            "the value must be an Integer, too."
                                            " Right now it is a <type 'float'>"
                                            " with value 0.5.",
                                UniformIntegerHyperparameter, "n_iter", 0,
                                1., default=0.5)

        self.assertRaisesRegexp(ValueError, "For the Integer parameter n_iter, "
                                            "the value must be an Integer, too. "
                                            "Right now it is a <type 'float'> "
                                            "with value 350.5.",
                                n_iter.instantiate, 350.5)
        self.assertIsInstance(n_iter.instantiate(20.0).value, int)

    def test_normalint_to_uniform(self):
        f1 = NormalIntegerHyperparameter("param", 0, 10, q=0.1)
        f1_expected = UniformIntegerHyperparameter("param", -30, 30)
        f1_actual = f1.to_uniform()
        self.assertEqual(f1_expected, f1_actual)

    def test_normalint_is_legal(self):
        f1 = NormalIntegerHyperparameter("param", 0, 10, q=0.1, log=True)
        self.assertFalse(f1.is_legal(3.1))
        self.assertFalse(f1.is_legal(3.0))   # 3.0 behaves like an Integer
        self.assertFalse(f1.is_legal("BlaBlaBla"))
        self.assertTrue(f1.is_legal(2))
        self.assertTrue(f1.is_legal(-15))

    def test_categorical(self):
        # TODO test for inequality
        f1 = CategoricalHyperparameter("param", [0, 1])
        f1_ = CategoricalHyperparameter("param", [0, 1])
        self.assertEqual(f1, f1_)
        self.assertEqual("param, Type: Categorical, Choices: {0, 1}, Default: 0",
                         str(f1))

        f2 = CategoricalHyperparameter("param", range(0, 1000))
        f2_ = CategoricalHyperparameter("param", range(0, 1000))
        self.assertEqual(f2, f2_)
        self.assertEqual(
            "param, Type: Categorical, Choices: {%s}, Default: 0" %
            ", ".join([str(choice) for choice in range(0, 1000)]),
            str(f2))

        f3 = CategoricalHyperparameter("param", range(0, 999))
        self.assertNotEqual(f2, f3)

        f4 = CategoricalHyperparameter("param_", range(0, 1000))
        self.assertNotEqual(f2, f4)

        f5 = CategoricalHyperparameter("param", range(0, 999) + [1001])
        self.assertNotEqual(f2, f5)

        f6 = CategoricalHyperparameter("param", ["a", "b"], default="b")
        f6_ = CategoricalHyperparameter("param", ["a", "b"], default="b")
        self.assertEqual(f6, f6_)
        self.assertEqual("param, Type: Categorical, Choices: {a, b}, Default: b"
                         , str(f6))

        self.assertNotEqual(f1, f2)
        self.assertNotEqual(f1, "UniformFloat")

    def test_categorical_strings(self):
        f1 = CategoricalHyperparameter("param", ["a", "b"])
        f1_ = CategoricalHyperparameter("param", ["a", "b"])
        self.assertEqual(f1, f1_)
        self.assertEqual("param, Type: Categorical, Choices: {a, b}, Default: a"
                         , str(f1))

    def test_categorical_is_legal(self):
        f1 = CategoricalHyperparameter("param", ["a", "b"])
        self.assertTrue(f1.is_legal("a"))
        self.assertTrue(f1.is_legal(u"a"))
        self.assertFalse(f1.is_legal("c"))
        self.assertFalse(f1.is_legal(3))