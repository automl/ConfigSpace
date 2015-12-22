import os
import unittest

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter, Constant
from ConfigSpace.io.pcs import read
from ConfigSpace.util import impute_inactive_values, get_random_neighbor


class UtilTest(unittest.TestCase):
    def test_impute_inactive_values(self):
        mini_autosklearn_config_space_path = os.path.join(
            os.path.dirname(__file__), 'test_searchspaces',
            'mini_autosklearn_original.pcs')
        with open(mini_autosklearn_config_space_path) as fh:
            cs = read(fh)

        cs.seed(1)
        configuration = cs.sample_configuration()
        new_configuration = impute_inactive_values(configuration)
        self.assertNotEqual(id(configuration), id(new_configuration))
        self.assertEqual(len(new_configuration._values), 11)
        for key in new_configuration:
            self.assertIsNotNone(new_configuration[key])
        self.assertEqual(new_configuration['random_forest:max_features'], 10)

    def _test_random_neigbor(self, hp):
        cs = ConfigurationSpace()
        if not isinstance(hp, list):
            hp = [hp]
        for hp_ in hp:
            cs.add_hyperparameter(hp_)
        cs.seed(1)
        config = cs.sample_configuration()
        for i in range(100):
            new_config = get_random_neighbor(config, i)
            self.assertNotEqual(config, new_config)

    def test_random_neighbor_float(self):
        hp = UniformFloatHyperparameter('a', 1, 10)
        self._test_random_neigbor(hp)
        hp = UniformFloatHyperparameter('a', 1, 10, log=True)
        self._test_random_neigbor(hp)

    def test_random_neighbor_int(self):
        hp = UniformIntegerHyperparameter('a', 1, 10)
        self._test_random_neigbor(hp)
        hp = UniformIntegerHyperparameter('a', 1, 10, log=True)
        self._test_random_neigbor(hp)

    def test_random_neighbor_cat(self):
        hp = CategoricalHyperparameter('a', [5, 6, 7, 8])
        self._test_random_neigbor(hp)

    def test_random_neighbor_failing(self):
        hp = Constant('a', 'b')
        self.assertRaisesRegexp(ValueError, 'Probably caught in an infinite '
                                           'loop.',
                                self._test_random_neigbor, hp)

        hp = CategoricalHyperparameter('a', ['a'])
        self.assertRaisesRegexp(ValueError, 'Probably caught in an infinite '
                                           'loop.',
                                self._test_random_neigbor, hp)

    def test_random_neigbor_conditional(self):
        mini_autosklearn_config_space_path = os.path.join(
            os.path.dirname(__file__), 'test_searchspaces',
            'mini_autosklearn_original.pcs')
        with open(mini_autosklearn_config_space_path) as fh:
            cs = read(fh)

        cs.seed(1)
        configuration = cs.get_default_configuration()
        for i in range(100):
            new_config = get_random_neighbor(configuration, i)
            self.assertNotEqual(configuration, new_config)