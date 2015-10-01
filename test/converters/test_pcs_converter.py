#!/usr/bin/env python

##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

import unittest

import six

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
import HPOlibConfigSpace.converters.pcs_parser as pcs_parser
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from HPOlibConfigSpace.conditions import EqualsCondition, InCondition, \
    AndConjunction
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenInClause, ForbiddenAndConjunction

# More complex search space
classifier = CategoricalHyperparameter("classifier", ["svm", "nn"])
kernel = CategoricalHyperparameter("kernel", ["rbf", "poly", "sigmoid"])
kernel_condition = EqualsCondition(kernel, classifier, "svm")
C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True)
C_condition = EqualsCondition(C, classifier, "svm")
gamma = UniformFloatHyperparameter("gamma", 0.000030518, 8, log=True)
gamma_condition = EqualsCondition(gamma, kernel, "rbf")
degree = UniformIntegerHyperparameter("degree", 1, 5)
degree_condition = InCondition(degree, kernel, ["poly", "sigmoid"])
neurons = UniformIntegerHyperparameter("neurons", 16, 1024)
neurons_condition = EqualsCondition(neurons, classifier, "nn")
lr = UniformFloatHyperparameter("lr", 0.0001, 1.0)
lr_condition = EqualsCondition(lr, classifier, "nn")
preprocessing = CategoricalHyperparameter("preprocessing", ["None", "pca"])
conditional_space = ConfigurationSpace()
conditional_space.add_hyperparameter(classifier)
conditional_space.add_hyperparameter(kernel)
conditional_space.add_hyperparameter(C)
conditional_space.add_hyperparameter(gamma)
conditional_space.add_hyperparameter(degree)
conditional_space.add_hyperparameter(neurons)
conditional_space.add_hyperparameter(lr)
conditional_space.add_hyperparameter(preprocessing)
conditional_space.add_condition(kernel_condition)
conditional_space.add_condition(C_condition)
conditional_space.add_condition(gamma_condition)
conditional_space.add_condition(degree_condition)
conditional_space.add_condition(neurons_condition)
conditional_space.add_condition(lr_condition)

float_a = UniformFloatHyperparameter("float_a", -1.23, 6.45)
e_float_a = UniformFloatHyperparameter("e_float_a", .5E-2, 4.5e+06)
int_a = UniformIntegerHyperparameter("int_a", -1, 6)
log_a = UniformFloatHyperparameter("log_a", 4e-1, 6.45, log=True)
int_log_a = UniformIntegerHyperparameter("int_log_a", 1, 6, log=True)
cat_a = CategoricalHyperparameter("cat_a", ["a", "b", "c", "d"])
crazy = CategoricalHyperparameter("@.:;/\?!$%&_-<>*+1234567890", ["const"])
easy_space = ConfigurationSpace()
easy_space.add_hyperparameter(float_a)
easy_space.add_hyperparameter(e_float_a)
easy_space.add_hyperparameter(int_a)
easy_space.add_hyperparameter(log_a)
easy_space.add_hyperparameter(int_log_a)
easy_space.add_hyperparameter(cat_a)
easy_space.add_hyperparameter(crazy)


class TestPCSConverter(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_read_configuration_space_basic(self):
        # TODO: what does this test has to do with the PCS converter?
        float_a_copy = UniformFloatHyperparameter("float_a", -1.23, 6.45)
        a_copy = {"a": float_a_copy, "b": int_a}
        a_real = {"b": int_a, "a": float_a}
        self.assertDictEqual(a_real, a_copy)

    def test_read_configuration_space_easy(self):
        expected = six.StringIO()
        expected.write('# This is a \n')
        expected.write('   # This is a comment with a leading whitespace ### ffds \n')
        expected.write('\n')
        expected.write('float_a [-1.23, 6.45] [2.61] # bla\n')
        expected.write('e_float_a [.5E-2, 4.5e+06] [2250000.0025]\n')
        expected.write('int_a [-1, 6] [2]i\n')
        expected.write('log_a [4e-1, 6.45] [1.6062378404]l\n')
        expected.write('int_log_a [1, 6] [2]il\n')
        expected.write('cat_a {a,"b",c,d} [a]\n')
        expected.write('@.:;/\?!$%&_-<>*+1234567890 {"const"} ["const"]\n')
        expected.seek(0)
        cs = pcs_parser.read(expected)
        self.assertEqual(cs, easy_space)

    def test_read_configuration_space_conditional(self):
        # More complex search space as string array
        complex_cs = list()
        complex_cs.append("kernel {rbf, poly, sigmoid} [rbf]")
        complex_cs.append("C [0.03125, 32768] [32]l")
        complex_cs.append("gamma [0.000030518, 8] [0.0156251079996]l")
        complex_cs.append("degree [1, 5] [3]i")
        complex_cs.append("lr [0.0001, 1.0] [0.50005]")
        complex_cs.append("neurons [16, 1024] [520]i # Should be Q16")
        complex_cs.append("preprocessing {None, pca} [None]")

        complex_cs.append("classifier {svm, nn} [svm]")
        complex_cs.append("C | classifier in {svm}")
        complex_cs.append("kernel | classifier in {svm}")
        complex_cs.append("degree | kernel in {poly, sigmoid}")
        complex_cs.append("gamma | kernel in {rbf}")
        complex_cs.append("lr | classifier in {nn}")
        complex_cs.append("neurons | classifier in {nn}")

        cs = pcs_parser.read(complex_cs)
        self.assertEqual(cs, conditional_space)

    def test_read_configuration_space_conditional_with_two_parents(self):
        pcs = list()
        pcs.append("@1:0:restarts {F,L,D,x,+,no}[x]")
        pcs.append("@1:S:Luby:aryrestarts {1,2}[1]")
        pcs.append("@1:2:Luby:restarts [1,65535][1000]il")
        pcs.append("@1:2:Luby:restarts | @1:0:restarts in {L}")
        pcs.append("@1:2:Luby:restarts | @1:S:Luby:aryrestarts in {2}")
        cs = pcs_parser.read(pcs)
        self.assertEqual(len(cs.get_conditions()), 1)
        self.assertIsInstance(cs.get_conditions()[0], AndConjunction)

    def test_write_illegal_argument(self):
        sp = {"a": int_a}
        self.assertRaisesRegexp(TypeError, "pcs_parser.write expects an "
                                "instance of "
                                "<class 'HPOlibConfigSpace.configuration_"
                                "space.ConfigurationSpace'>, you provided "
                                "'<(type|class) 'dict'>'", pcs_parser.write, sp)

    def test_write_int(self):
        expected = "int_a [-1, 6] [2]i"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(int_a)
        value = pcs_parser.write(cs)
        self.assertEqual(expected, value)

    def test_write_log_int(self):
        expected = "int_log_a [1, 6] [2]il"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(int_log_a)
        value = pcs_parser.write(cs)
        self.assertEqual(expected, value)

    def test_write_q_int(self):
        expected = "Q16_int_a [16, 1024] [520]i"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("int_a", 16, 1024, q=16))
        value = pcs_parser.write(cs)
        self.assertEqual(expected, value)

    def test_write_q_float(self):
        expected = "Q16_float_a [16.0, 1024.0] [520.0]"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter("float_a", 16, 1024, q=16))
        value = pcs_parser.write(cs)
        self.assertEqual(expected, value)

    def test_write_log10(self):
        expected = "a [10.0, 1000.0] [100.0]l"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter("a", 10, 1000, log=True))
        value = pcs_parser.write(cs)
        self.assertEqual(expected, value)

    def test_build_forbidden(self):
        expected = "a {a, b, c} [a]\nb {a, b, c} [c]\n\n" \
                   "{a=a, b=a}\n{a=a, b=b}\n{a=b, b=a}\n{a=b, b=b}"
        cs = ConfigurationSpace()
        a = CategoricalHyperparameter("a", ["a", "b", "c"], "a")
        b = CategoricalHyperparameter("b", ["a", "b", "c"], "c")
        cs.add_hyperparameter(a)
        cs.add_hyperparameter(b)
        fb = ForbiddenAndConjunction(ForbiddenInClause(a, ["a", "b"]),
                                     ForbiddenInClause(b, ["a", "b"]))
        cs.add_forbidden_clause(fb)
        value = pcs_parser.write(cs)
        self.assertIn(expected, value)

