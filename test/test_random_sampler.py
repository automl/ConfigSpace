from collections import defaultdict
import unittest

import numpy as np

from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, NormalFloatHyperparameter, \
    UniformIntegerHyperparameter, NormalIntegerHyperparameter, \
    InstantiatedCategoricalHyperparameter, \
    InstantiatedUniformFloatHyperparameter, \
    InstantiatedNormalFloatHyperparameter, \
    InstantiatedUniformIntegerHyperparameter, \
    InstantiatedNormalIntegerHyperparameter, Constant
from HPOlibConfigSpace.conditions import EqualsCondition, AndConjunction, \
    NotEqualsCondition, InCondition, OrConjunction

from HPOlibConfigSpace.random_sampler import RandomSampler


class TestRandomSampler(unittest.TestCase):
    def test_sample_configuration(self):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs.add_hyperparameter(hp1)
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cs.add_hyperparameter(hp2)
        cond1 = EqualsCondition(hp2, hp1, 0)
        cs.add_condition(cond1)
        # This automatically checks the configuration!
        Configuration(cs, parent=hp1.instantiate(0), child=hp2.instantiate(5))

        # and now for something more complicated
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("input1", [0, 1])
        cs.add_hyperparameter(hp1)
        hp2 = CategoricalHyperparameter("input2", [0, 1])
        cs.add_hyperparameter(hp2)
        hp3 = CategoricalHyperparameter("input3", [0, 1])
        cs.add_hyperparameter(hp3)
        hp4 = CategoricalHyperparameter("input4", [0, 1])
        cs.add_hyperparameter(hp4)
        hp5 = CategoricalHyperparameter("input5", [0, 1])
        cs.add_hyperparameter(hp5)
        hp6 = Constant("AND", "True")
        cs.add_hyperparameter(hp6)

        cond1 = EqualsCondition(hp6, hp1, 1)
        cond2 = NotEqualsCondition(hp6, hp2, 1)
        cond3 = InCondition(hp6, hp3, [1])
        cond4 = EqualsCondition(hp5, hp3, 1)
        cond5 = EqualsCondition(hp4, hp5, 1)
        cond6 = EqualsCondition(hp6, hp4, 1)
        cond7 = EqualsCondition(hp6, hp5, 1)

        conj1 = AndConjunction(cond1, cond2)
        conj2 = OrConjunction(conj1, cond3)
        conj3 = AndConjunction(conj2, cond6, cond7)
        cs.add_condition(cond4)
        cs.add_condition(cond5)
        cs.add_condition(conj3)

        rs = RandomSampler(cs, 1)
        print rs.sample_configuration()

    def test_sample_UniformFloatHyperparameter(self):
        # This can sample four distributions

        def sample(cs, hp):
            rs = RandomSampler(cs, 1)
            counts_per_bin = [0 for i in range(21)]
            for i in range(100000):
                ihp = rs._sample_UniformFloatHyperparameter(hp)
                sample = ihp.value
                index = int((sample - hp.lower) / (hp.upper - hp.lower) * 20)
                #print sample, index
                counts_per_bin[index] += 1

            self.assertIsInstance(ihp, InstantiatedUniformFloatHyperparameter)
            return counts_per_bin

        # Uniform
        cs = ConfigurationSpace()
        hp = UniformFloatHyperparameter("ufhp", 0.5, 2.5)
        cs.add_hyperparameter(hp)

        counts_per_bin = sample(cs, hp)
        # The 21st bin is only filled if exactly 2.5 is sampled...very rare...
        for bin in counts_per_bin[:-1]:
            self.assertTrue(5200 > bin > 4800)
        self.assertEqual(sample(cs, hp), sample(cs, hp))

        # Quantized Uniform
        cs = ConfigurationSpace()
        hp = UniformFloatHyperparameter("ufhp", 0.0, 1.0, q=0.1)
        cs.add_hyperparameter(hp)

        counts_per_bin = sample(cs, hp)
        for bin in counts_per_bin[::2]:
            self.assertTrue(9300 > bin > 8700)
        for bin in counts_per_bin[1::2]:
            self.assertEqual(bin, 0)
        self.assertEqual(sample(cs, hp), sample(cs, hp))

        # Log Uniform
        cs = ConfigurationSpace()
        hp = UniformFloatHyperparameter("ufhp", 1.0, np.e ** 2, log=True)
        cs.add_hyperparameter(hp)

        counts_per_bin = sample(cs, hp)
        self.assertEqual(counts_per_bin,
                         [13781, 11025, 8661, 7543, 6757, 5831, 5313, 4659,
                          4219, 3859, 3698, 3315, 3104, 3005, 2866, 2761, 2570,
                          2426, 2313, 2294, 0])
        self.assertEqual(sample(cs, hp), sample(cs, hp))

        # Quantized Log-Uniform
        cs = ConfigurationSpace()
        # 7.2 ~ np.round(e * e, 1)
        hp = UniformFloatHyperparameter("ufhp", 1.2, 7.2, q=0.6, log=True)
        cs.add_hyperparameter(hp)

        counts_per_bin = sample(cs, hp)
        self.assertEqual(counts_per_bin,
                         [24193, 15623, 0, 12043, 0, 0, 9634, 7688, 0, 0, 6698,
                          0, 5722, 5275, 0, 4806, 0, 0, 4294, 4024, 0])
        self.assertEqual(sample(cs, hp), sample(cs, hp))

    def test_sample_NormalFloatHyperparameter(self):
        cs = ConfigurationSpace()
        hp = NormalFloatHyperparameter("nfhp", 0, 1)
        cs.add_hyperparameter(hp)

        def actual_test():
            rs = RandomSampler(cs, 1)
            counts_per_bin = [0 for i in range(11)]
            for i in range(100000):
                ihp = rs._sample_NormalFloatHyperparameter(hp)
                sample = ihp.value
                index = min(max(int((round(sample + 0.5)) + 5), 0), 9)
                counts_per_bin[index] += 1

            self.assertEqual([0, 3, 143, 2121, 13613, 33952, 34152, 13776,
                              2112, 128, 0], counts_per_bin)
            self.assertIsInstance(ihp, InstantiatedNormalFloatHyperparameter)

            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

    def test_sample_UniformIntegerHyperparameter(self):
        def sample(cs, hp):
            rs = RandomSampler(cs, 1)
            counts_per_bin = [0 for i in range(21)]
            for i in range(100000):
                ihp = rs._sample_UniformIntegerHyperparameter(hp)
                sample = float(ihp.value)
                index = int((sample - hp.lower) / (hp.upper - hp.lower) * 20)
                # print sample, index
                counts_per_bin[index] += 1

            self.assertIsInstance(ihp, InstantiatedUniformIntegerHyperparameter)
            return counts_per_bin

        # Quantized Uniform
        cs = ConfigurationSpace()
        hp = UniformIntegerHyperparameter("uihp", 0, 10)
        cs.add_hyperparameter(hp)

        counts_per_bin = sample(cs, hp)
        for bin in counts_per_bin[::2]:
            self.assertTrue(9300 > bin > 8700)
        for bin in counts_per_bin[1::2]:
            self.assertEqual(bin, 0)
        self.assertEqual(sample(cs, hp), sample(cs, hp))

    def test_sample_NormalIntegerHyperparameter(self):
        def sample(cs, hp):
            lower = -30
            upper = 30
            rs = RandomSampler(cs, 1)
            counts_per_bin = [0 for i in range(21)]
            for i in range(100000):
                ihp = rs._sample_NormalIntegerHyperparameter(hp)
                sample = float(ihp.value)
                if sample < lower:
                    sample = lower
                if sample > upper:
                    sample = upper
                index = int((sample - lower) / (upper - lower) * 20)
                # print sample, index
                counts_per_bin[index] += 1

            self.assertIsInstance(ihp, InstantiatedNormalIntegerHyperparameter)
            return counts_per_bin

        cs = ConfigurationSpace()
        hp = NormalIntegerHyperparameter("nihp", 0, 10)
        cs.add_hyperparameter(hp)

        self.assertEqual(sample(cs, hp),
                         [290, 401, 899, 1629, 2814, 4481, 6642, 8456, 10552,
                          11592, 11863, 11079, 9398, 7369, 5142, 3382, 2035,
                          1040, 527, 256, 153])
        self.assertEqual(sample(cs, hp), sample(cs, hp))

    def test_sample_CategoricalHyperparameter(self):
        cs = ConfigurationSpace()
        hp = CategoricalHyperparameter("chp", [0, 2, "Bla", u"Blub"])

        def actual_test():
            rs = RandomSampler(cs, 1)
            counts_per_bin = defaultdict(int)
            for i in range(10000):
                ihp = rs._sample_CategoricalHyperparameter(hp)
                sample = ihp.value
                counts_per_bin[sample] += 1

            self.assertIsInstance(ihp, InstantiatedCategoricalHyperparameter)
            self.assertEqual({0: 2564, 2: 2459, 'Bla': 2468, u'Blub': 2509},
                             dict(counts_per_bin.items()))
            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())


