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

import os
import unittest

import numpy as np

from ConfigSpace import Configuration, ConfigurationSpace, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter, Constant, \
    EqualsCondition, AndConjunction, OrConjunction
from ConfigSpace.read_and_write.pcs import read
from ConfigSpace.util import impute_inactive_values, get_random_neighbor, \
    get_one_exchange_neighbourhood, deactivate_inactive_hyperparameters, ConfigSpaceGrid
import ConfigSpace.c_util


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

    def _test_get_one_exchange_neighbourhood(self, hp):
        cs = ConfigurationSpace()
        num_neighbors = 0
        if not isinstance(hp, list):
            hp = [hp]
        for hp_ in hp:
            cs.add_hyperparameter(hp_)
            if np.isinf(hp_.get_num_neighbors()):
                num_neighbors += 4
            else:
                num_neighbors += hp_.get_num_neighbors()

        cs.seed(1)
        config = cs.get_default_configuration()
        all_neighbors = []
        for i in range(100):
            neighborhood = get_one_exchange_neighbourhood(config, i)
            for new_config in neighborhood:
                self.assertNotEqual(config, new_config)
                self.assertNotEqual(config.get_dictionary(), new_config.get_dictionary())
                all_neighbors.append(new_config)

        return all_neighbors

    def test_random_neighbor_float(self):
        hp = UniformFloatHyperparameter('a', 1, 10)
        self._test_random_neigbor(hp)
        hp = UniformFloatHyperparameter('a', 1, 10, log=True)
        self._test_random_neigbor(hp)

    def test_random_neighborhood_float(self):
        hp = UniformFloatHyperparameter('a', 1, 10)
        all_neighbors = self._test_get_one_exchange_neighbourhood(hp)
        all_neighbors = [neighbor['a'] for neighbor in all_neighbors]
        self.assertAlmostEqual(5.44, np.mean(all_neighbors), places=2)
        self.assertAlmostEqual(3.065, np.var(all_neighbors), places=2)
        hp = UniformFloatHyperparameter('a', 1, 10, log=True)
        all_neighbors = self._test_get_one_exchange_neighbourhood(hp)
        all_neighbors = [neighbor['a'] for neighbor in all_neighbors]
        # Default value is 3.16
        self.assertAlmostEqual(3.45, np.mean(all_neighbors), places=2)
        self.assertAlmostEqual(2.67, np.var(all_neighbors), places=2)

    def test_random_neighbor_int(self):
        hp = UniformIntegerHyperparameter('a', 1, 10)
        self._test_random_neigbor(hp)
        hp = UniformIntegerHyperparameter('a', 1, 10, log=True)
        self._test_random_neigbor(hp)

    def test_random_neighborhood_int(self):
        hp = UniformIntegerHyperparameter('a', 1, 10)
        all_neighbors = self._test_get_one_exchange_neighbourhood(hp)
        all_neighbors = [neighbor['a'] for neighbor in all_neighbors]
        self.assertAlmostEqual(5.8125, np.mean(all_neighbors), places=2)
        self.assertAlmostEqual(5.60234375, np.var(all_neighbors), places=2)
        hp = UniformIntegerHyperparameter('a', 1, 10, log=True)
        all_neighbors = self._test_get_one_exchange_neighbourhood(hp)
        all_neighbors = [neighbor['a'] for neighbor in all_neighbors]
        # Default value is 3.16
        self.assertAlmostEqual(3.9425, np.mean(all_neighbors), places=2)
        self.assertAlmostEqual(5.91, np.var(all_neighbors), places=2)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(hp)
        for val in range(1, 11):
            config = Configuration(cs, values={'a': val})
            for i in range(100):
                neighborhood = get_one_exchange_neighbourhood(config, 1)
                neighbors = [neighbor['a'] for neighbor in neighborhood]
                self.assertEqual(len(neighbors), len(np.unique(neighbors)))
                self.assertNotIn(val, neighbors)

    def test_random_neighbor_cat(self):
        hp = CategoricalHyperparameter('a', [5, 6, 7, 8])
        all_neighbors = self._test_get_one_exchange_neighbourhood(hp)
        all_neighbors = [neighbor for neighbor in all_neighbors]
        self.assertEqual(len(all_neighbors), 300)  # 3 (neighbors) * 100 (samples)

    def test_random_neighborhood_cat(self):
        hp = CategoricalHyperparameter('a', [5, 6, 7, 8])
        self._test_random_neigbor(hp)

    def test_random_neighbor_failing(self):
        hp = Constant('a', 'b')
        self.assertRaisesRegex(ValueError, 'Probably caught in an infinite '
                                           'loop.',
                               self._test_random_neigbor, hp)

        hp = CategoricalHyperparameter('a', ['a'])
        self.assertRaisesRegex(ValueError, 'Probably caught in an infinite '
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

    def test_random_neigborhood_conditional(self):
        mini_autosklearn_config_space_path = os.path.join(
            os.path.dirname(__file__), 'test_searchspaces',
            'mini_autosklearn_original.pcs')
        with open(mini_autosklearn_config_space_path) as fh:
            cs = read(fh)

        cs.seed(1)
        configuration = cs.get_default_configuration()
        for i in range(100):
            neighborhood = get_one_exchange_neighbourhood(configuration, i)
            for new_config in neighborhood:
                self.assertNotEqual(configuration, new_config)

    def test_deactivate_inactive_hyperparameters(self):
        diamond = ConfigurationSpace()
        head = CategoricalHyperparameter('head', [0, 1])
        left = CategoricalHyperparameter('left', [0, 1])
        right = CategoricalHyperparameter('right', [0, 1])
        bottom = CategoricalHyperparameter('bottom', [0, 1])
        diamond.add_hyperparameters([head, left, right, bottom])
        diamond.add_condition(EqualsCondition(left, head, 0))
        diamond.add_condition(EqualsCondition(right, head, 0))
        diamond.add_condition(AndConjunction(EqualsCondition(bottom, left, 0),
                                             EqualsCondition(bottom, right, 0)))

        c = deactivate_inactive_hyperparameters({'head': 0, 'left': 0,
                                                 'right': 0, 'bottom': 0},
                                                diamond)
        diamond._check_configuration_rigorous(c)

        c = deactivate_inactive_hyperparameters({'head': 1, 'left': 0,
                                                 'right': 0, 'bottom': 0},
                                                diamond)
        diamond._check_configuration_rigorous(c)

        c = deactivate_inactive_hyperparameters({'head': 0, 'left': 1,
                                                 'right': 0, 'bottom': 0},
                                                diamond)
        diamond._check_configuration_rigorous(c)

        diamond = ConfigurationSpace()
        head = CategoricalHyperparameter('head', [0, 1])
        left = CategoricalHyperparameter('left', [0, 1])
        right = CategoricalHyperparameter('right', [0, 1])
        bottom = CategoricalHyperparameter('bottom', [0, 1])
        diamond.add_hyperparameters([head, left, right, bottom])
        diamond.add_condition(EqualsCondition(left, head, 0))
        diamond.add_condition(EqualsCondition(right, head, 0))
        diamond.add_condition(OrConjunction(EqualsCondition(bottom, left, 0),
                                            EqualsCondition(bottom, right, 0)))

        c = deactivate_inactive_hyperparameters({'head': 0, 'left': 0,
                                                 'right': 0, 'bottom': 0},
                                                diamond)
        diamond._check_configuration_rigorous(c)

        c = deactivate_inactive_hyperparameters({'head': 1, 'left': 1,
                                                 'right': 0, 'bottom': 0},
                                                diamond)
        diamond._check_configuration_rigorous(c)

        c = deactivate_inactive_hyperparameters({'head': 0, 'left': 1,
                                                 'right': 0, 'bottom': 0},
                                                diamond)
        diamond._check_configuration_rigorous(c)

        plain = ConfigurationSpace()
        a = UniformIntegerHyperparameter('a', 0, 10)
        b = UniformIntegerHyperparameter('b', 0, 10)
        plain.add_hyperparameters([a, b])
        c = deactivate_inactive_hyperparameters({'a': 5, 'b': 6}, plain)
        plain.check_configuration(c)

    def test_check_neighbouring_config_diamond(self):
        diamond = ConfigurationSpace()
        head = CategoricalHyperparameter('head', [0, 1])
        left = CategoricalHyperparameter('left', [0, 1])
        right = CategoricalHyperparameter('right', [0, 1, 2, 3])
        bottom = CategoricalHyperparameter('bottom', [0, 1])
        diamond.add_hyperparameters([head, left, right, bottom])
        diamond.add_condition(EqualsCondition(left, head, 0))
        diamond.add_condition(EqualsCondition(right, head, 0))
        diamond.add_condition(AndConjunction(EqualsCondition(bottom, left, 1),
                                             EqualsCondition(bottom, right, 1)))

        config = Configuration(diamond, {'bottom': 0, 'head': 0, 'left': 1, 'right': 1})
        hp_name = "head"
        index = diamond.get_idx_by_hyperparameter_name(hp_name)
        neighbor_value = 1

        new_array = ConfigSpace.c_util.change_hp_value(
            diamond,
            config.get_array(),
            hp_name,
            neighbor_value,
            index
        )
        expected_array = np.array([1, np.nan, np.nan, np.nan])

        np.testing.assert_almost_equal(new_array, expected_array)

    def test_check_neighbouring_config_diamond_str(self):
        diamond = ConfigurationSpace()
        head = CategoricalHyperparameter('head', ['red', 'green'])
        left = CategoricalHyperparameter('left', ['red', 'green'])
        right = CategoricalHyperparameter('right', ['red', 'green', 'blue', 'yellow'])
        bottom = CategoricalHyperparameter('bottom', ['red', 'green'])
        diamond.add_hyperparameters([head, left, right, bottom])
        diamond.add_condition(EqualsCondition(left, head, 'red'))
        diamond.add_condition(EqualsCondition(right, head, 'red'))
        diamond.add_condition(AndConjunction(EqualsCondition(bottom, left, 'green'),
                                             EqualsCondition(bottom, right, 'green')))

        config = Configuration(
            diamond,
            {'bottom': 'red', 'head': 'red', 'left': 'green', 'right': 'green'},
        )
        hp_name = "head"
        index = diamond.get_idx_by_hyperparameter_name(hp_name)
        neighbor_value = 1

        new_array = ConfigSpace.c_util.change_hp_value(
            diamond,
            config.get_array(),
            hp_name,
            neighbor_value,
            index
        )
        expected_array = np.array([1, np.nan, np.nan, np.nan])

        np.testing.assert_almost_equal(new_array, expected_array)


    def test_generate_grid(self):
        cs = ConfigurationSpace(seed=1234)

        cat1 = CategoricalHyperparameter(name='cat1', choices=['T', 'F'])
        const1 = Constant(name='const1', value=4)
        float1 = UniformFloatHyperparameter(name='float1', lower=-1, upper=1, log=False)
        int1 = UniformIntegerHyperparameter(name='int1', lower=10, upper=100, log=True)
        ord1 = OrdinalHyperparameter(name='ord1', sequence=['1', '2', '3'])

        cs.add_hyperparameters([float1, int1, cat1, ord1, const1])

        num_steps_dict = {'float1': 11, 'int1': 6}
        #TODO uncomment below and add asserts for later test cases
        # generated_grid = generate_grid(cs, num_steps_dict)
        #
        # # Check randomly pre-selected values in the generated_grid
        # self.assertEqual(len(generated_grid), 396) # 2 * 1 * 11 * 6 * 3 total diff. possibilities for the Configuration (i.e., for each tuple of HPs)
        # self.assertEqual(generated_grid[0].get_dictionary()['cat1'], 'T')
        # self.assertEqual(generated_grid[198].get_dictionary()['cat1'], 'F') #
        # self.assertEqual(generated_grid[45].get_dictionary()['const1'], 4) #
        # self.assertAlmostEqual(generated_grid[55].get_dictionary()['float1'], -0.4, places=2) # The 2 most frequently changing HPs (int1 and ord1) have 3*6 = 18 different values for each value of float1, so the 4th value of float1 of -0.4 is reached after 3*18 = 54 values.
        # self.assertEqual(generated_grid[12].get_dictionary()['int1'], 63) # 5th diff. value for int1 after 4*3 = 12 values. Reasoning as above.
        # self.assertEqual(generated_grid[3].get_dictionary()['ord1'], '1') #
        # self.assertEqual(generated_grid[4].get_dictionary()['ord1'], '2') #
        # self.assertEqual(generated_grid[5].get_dictionary()['ord1'], '3') #

        #tests for quantization and conditional spaces: 1 basic one for 1 condition; 1 for tree of conditions; 1 for tree of conditions with OrConjunction and AndConjunction;
        cs2 = CS.ConfigurationSpace(seed=123)
        float1 = CSH.UniformFloatHyperparameter(name='float1', lower=-1, upper=1, log=False)
        int1 = CSH.UniformIntegerHyperparameter(name='int1', lower=0, upper=1000, log=False, q=500)
        cs2.add_hyperparameters([float1, int1])

        int2_cond = CSH.UniformIntegerHyperparameter(name='int2_cond', lower=10, upper=100, log=True)
        cs2.add_hyperparameters([int2_cond])
        cond_1 = CS.AndConjunction(CS.LessThanCondition(int2_cond, float1, -0.5),
                                  CS.GreaterThanCondition(int2_cond, int1, 600))
        cs2.add_conditions([cond_1])
        cat1_cond = CSH.CategoricalHyperparameter(name='cat1_cond', choices=['apple', 'orange'])
        cs2.add_hyperparameters([cat1_cond])
        cond_2 = CS.AndConjunction(CS.GreaterThanCondition(cat1_cond, int1, 300),
                                   CS.LessThanCondition(cat1_cond, int1, 700),
                                   CS.GreaterThanCondition(cat1_cond, float1, -0.5),
                                  CS.LessThanCondition(cat1_cond, float1, 0.5)
                                  )
        cs2.add_conditions([cond_2])
        float2_cond = CSH.UniformFloatHyperparameter(name='float2_cond', lower=10., upper=100., log=True)
        cs2.add_hyperparameters([float2_cond])
        cond_3 = CS.GreaterThanCondition(float2_cond, int2_cond, 50)
        cs2.add_conditions([cond_3])
        print(cs2)
        num_steps_dict1 = {'float1': 4, 'int2_cond': 3, 'float2_cond': 3 }
        configspace_grid = ConfigSpaceGrid(cs2)
        generated_grid = configspace_grid.generate_grid(num_steps_dict1)
