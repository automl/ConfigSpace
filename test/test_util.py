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

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter, Constant
from ConfigSpace.io.pcs import read
from ConfigSpace.util import impute_inactive_values, get_random_neighbor, \
    get_one_exchange_neighbourhood


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
            # self.assertEqual(len(neighborhood), num_neighbors)
            for new_config in neighborhood:
                self.assertEqual(len(new_config), num_neighbors)
                self.assertNotEqual(config, new_config)
                # all_neighbors.append(new_config)
                all_neighbors.extend(new_config)

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
        self.assertAlmostEqual(3.065,  np.var(all_neighbors), places=2)
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
        self.assertAlmostEqual(5.79, np.mean(all_neighbors), places=2)
        self.assertAlmostEqual(4.99, np.var(all_neighbors), places=2)
        hp = UniformIntegerHyperparameter('a', 1, 10, log=True)
        all_neighbors = self._test_get_one_exchange_neighbourhood(hp)
        all_neighbors = [neighbor['a'] for neighbor in all_neighbors]
        # Default value is 3.16
        self.assertAlmostEqual(3.55, np.mean(all_neighbors), places=2)
        self.assertAlmostEqual(5.91, np.var(all_neighbors), places=2)

    def test_random_neighbor_cat(self):
        hp = CategoricalHyperparameter('a', [5, 6, 7, 8])
        all_neighbors = self._test_get_one_exchange_neighbourhood(hp)
        all_neighbors = [neighbor['a'] for neighbor in all_neighbors]
        self.assertEqual(len(all_neighbors), 300) # 3 (neighbors) * 100 (samples)

    def test_random_neighborhood_cat(self):
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