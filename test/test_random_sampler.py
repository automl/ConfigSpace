from collections import defaultdict
import unittest


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
        cs = ConfigurationSpace()
        hp = UniformFloatHyperparameter("ufhp", -0.5, 1.5)
        cs.add_hyperparameter(hp)

        def actual_test():
            rs = RandomSampler(cs, 1)
            counts_per_bin = [0 for i in range(10)]
            for i in range(100000):
                ihp = rs._sample_UniformFloatHyperparameter(hp)
                sample = ihp.value
                index = int((sample + 0.5) * 5)
                counts_per_bin[index] += 1

            self.assertIsInstance(ihp, InstantiatedUniformFloatHyperparameter)

            for bin in counts_per_bin:
                self.assertTrue(10500 > bin > 9500)
            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

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
        cs = ConfigurationSpace()
        hp = UniformIntegerHyperparameter("uihp", -2, 9)
        cs.add_hyperparameter(hp)

        def actual_test():
            rs = RandomSampler(cs, 1)
            counts_per_bin = [0 for i in range(12)]
            for i in range(120000):
                ihp = rs._sample_UniformIntegerHyperparameter(hp)
                sample = ihp.value
                index = sample + 2
                counts_per_bin[index] += 1

            for bin in counts_per_bin:
                self.assertTrue(10500 > bin > 9500)
            self.assertIsInstance(ihp, InstantiatedUniformIntegerHyperparameter)

            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

    def test_sample_NormalIntegerHyperparameter(self):
        cs = ConfigurationSpace()
        hp = NormalIntegerHyperparameter("nihp", 5, 1)
        cs.add_hyperparameter(hp)

        def actual_test():
            rs = RandomSampler(cs, 1)
            counts_per_bin = [0] * 10
            for i in range(100000):
                ihp = rs._sample_NormalIntegerHyperparameter(hp)
                sample = ihp.value
                index = min(max(sample, 0), 9)
                counts_per_bin[index] += 1

            self.assertIsInstance(ihp, InstantiatedNormalIntegerHyperparameter)
            self.assertEqual(
                [1, 24, 580, 6043, 24087, 38231, 24333, 6077, 598, 26], counts_per_bin)
            return counts_per_bin

        self.assertEqual(actual_test(), actual_test())

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


