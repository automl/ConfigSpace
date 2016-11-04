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

import unittest

import six

from ConfigSpace.configuration_space import ConfigurationSpace
import ConfigSpace.io.pcs as pcs
import ConfigSpace.io.pcs_new as pcs_new
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition, \
    AndConjunction
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
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
conditional_space.add_hyperparameter(preprocessing)
conditional_space.add_hyperparameter(classifier)
conditional_space.add_hyperparameter(kernel)
conditional_space.add_hyperparameter(C)
conditional_space.add_hyperparameter(neurons)
conditional_space.add_hyperparameter(lr)
conditional_space.add_hyperparameter(degree)
conditional_space.add_hyperparameter(gamma)

conditional_space.add_condition(C_condition)
conditional_space.add_condition(kernel_condition)
conditional_space.add_condition(lr_condition)
conditional_space.add_condition(neurons_condition)
conditional_space.add_condition(degree_condition)
conditional_space.add_condition(gamma_condition)

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
    
    '''
    Tests for the "older pcs" version
    
    '''
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
        cs = pcs.read(expected)
        self.assertEqual(cs, easy_space)
    
    def test_read_configuration_space_conditional(self):
        # More complex search space as string array
        complex_cs = list()
        complex_cs.append("preprocessing {None, pca} [None]")
        complex_cs.append("classifier {svm, nn} [svm]")
        complex_cs.append("kernel {rbf, poly, sigmoid} [rbf]")
        complex_cs.append("C [0.03125, 32768] [32]l")
        complex_cs.append("neurons [16, 1024] [520]i # Should be Q16")
        complex_cs.append("lr [0.0001, 1.0] [0.50005]")
        complex_cs.append("degree [1, 5] [3]i")
        complex_cs.append("gamma [0.000030518, 8] [0.0156251079996]l")

        complex_cs.append("C | classifier in {svm}")
        complex_cs.append("kernel | classifier in {svm}")
        complex_cs.append("lr | classifier in {nn}")
        complex_cs.append("neurons | classifier in {nn}")
        complex_cs.append("degree | kernel in {poly, sigmoid}")
        complex_cs.append("gamma | kernel in {rbf}")

        cs = pcs.read(complex_cs)
        self.assertEqual(cs, conditional_space)

    def test_read_configuration_space_conditional_with_two_parents(self):
        config_space = list()
        config_space.append("@1:0:restarts {F,L,D,x,+,no}[x]")
        config_space.append("@1:S:Luby:aryrestarts {1,2}[1]")
        config_space.append("@1:2:Luby:restarts [1,65535][1000]il")
        config_space.append("@1:2:Luby:restarts | @1:0:restarts in {L}")
        config_space.append("@1:2:Luby:restarts | @1:S:Luby:aryrestarts in {2}")
        cs = pcs.read(config_space)
        self.assertEqual(len(cs.get_conditions()), 1)
        self.assertIsInstance(cs.get_conditions()[0], AndConjunction)

    def test_write_illegal_argument(self):
        sp = {"a": int_a}
        self.assertRaisesRegexp(TypeError, "pcs_parser.write expects an "
                                "instance of "
                                "<class "
                                "'ConfigSpace.configuration_"
                                "space.ConfigurationSpace'>, you provided "
                                "'<(type|class) 'dict'>'", pcs.write, sp)

    def test_write_int(self):
        expected = "int_a [-1, 6] [2]i"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(int_a)
        value = pcs.write(cs)
        self.assertEqual(expected, value)

    def test_write_log_int(self):
        expected = "int_log_a [1, 6] [2]il"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(int_log_a)
        value = pcs.write(cs)
        self.assertEqual(expected, value)

    def test_write_q_int(self):
        expected = "Q16_int_a [16, 1024] [520]i"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("int_a", 16, 1024, q=16))
        value = pcs.write(cs)
        self.assertEqual(expected, value)

    def test_write_q_float(self):
        expected = "Q16_float_a [16.0, 1024.0] [520.0]"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter("float_a", 16, 1024, q=16))
        value = pcs.write(cs)
        self.assertEqual(expected, value)

    def test_write_log10(self):
        expected = "a [10.0, 1000.0] [100.0]l"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter("a", 10, 1000, log=True))
        value = pcs.write(cs)
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
        value = pcs.write(cs)
        self.assertIn(expected, value)
    
    """
    Tests for the "newer pcs" version in order to check 
    if both deliver the same results
    """
    def test_read_new_configuration_space_easy(self):
        expected = six.StringIO()
        expected.write('# This is a \n')
        expected.write('   # This is a comment with a leading whitespace ### ffds \n')
        expected.write('\n')
        expected.write('float_a real [-1.23, 6.45] [2.61] # bla\n')
        expected.write('e_float_a real [.5E-2, 4.5e+06] [2250000.0025]\n')
        expected.write('int_a integer [-1, 6] [2]\n')
        expected.write('log_a real [4e-1, 6.45] [1.6062378404]log\n')
        expected.write('int_log_a integer [1, 6] [2]log\n')
        expected.write('cat_a categorical {a,"b",c,d} [a]\n')
        expected.write('@.:;/\?!$%&_-<>*+1234567890 categorical {"const"} ["const"]\n')
        expected.seek(0)
        cs = pcs_new.read(expected)
        self.assertEqual(cs, easy_space)
        
    def test_read_new_configuration_space_conditional(self):
        # More complex search space as string array
        complex_cs = list()
        complex_cs.append("preprocessing categorical {None, pca} [None]")
        complex_cs.append("classifier categorical {svm, nn} [svm]")
        complex_cs.append("kernel categorical {rbf, poly, sigmoid} [rbf]")
        complex_cs.append("C real [0.03125, 32768] [32]log")
        complex_cs.append("neurons integer [16, 1024] [520] # Should be Q16")
        complex_cs.append("lr real [0.0001, 1.0] [0.50005]")
        complex_cs.append("degree integer [1, 5] [3]")
        complex_cs.append("gamma real [0.000030518, 8] [0.0156251079996]log")

        complex_cs.append("C | classifier in {svm}")
        complex_cs.append("kernel | classifier in {svm}")
        complex_cs.append("lr | classifier in {nn}")
        complex_cs.append("neurons | classifier in {nn}")
        complex_cs.append("degree | kernel in {poly, sigmoid}")
        complex_cs.append("gamma | kernel in {rbf}")

        cs_new = pcs_new.read(complex_cs)
        self.assertEqual(cs_new, conditional_space)
        
        # same in older version
        complex_cs_old = list()
        complex_cs_old.append("preprocessing {None, pca} [None]")
        complex_cs_old.append("classifier {svm, nn} [svm]")
        complex_cs_old.append("kernel {rbf, poly, sigmoid} [rbf]")
        complex_cs_old.append("C [0.03125, 32768] [32]l")
        complex_cs_old.append("neurons [16, 1024] [520]i # Should be Q16")
        complex_cs_old.append("lr [0.0001, 1.0] [0.50005]")
        complex_cs_old.append("degree [1, 5] [3]i")
        complex_cs_old.append("gamma [0.000030518, 8] [0.0156251079996]l")

        complex_cs_old.append("C | classifier in {svm}")
        complex_cs_old.append("kernel | classifier in {svm}")
        complex_cs_old.append("lr | classifier in {nn}")
        complex_cs_old.append("neurons | classifier in {nn}")
        complex_cs_old.append("degree | kernel in {poly, sigmoid}")
        complex_cs_old.append("gamma | kernel in {rbf}")

        cs_old = pcs.read(complex_cs_old)
        self.assertEqual(cs_old, cs_new)
        
    def test_write_new_illegal_argument(self):
        sp = {"a": int_a}
        self.assertRaisesRegexp(TypeError, "pcs_parser.write expects an "
                                "instance of "
                                "<class "
                                "'ConfigSpace.configuration_"
                                "space.ConfigurationSpace'>, you provided "
                                "'<(type|class) 'dict'>'", pcs_new.write, sp)
                                
    def test_write_new_int(self):
        expected = "int_a integer [-1, 6] [2]"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(int_a)
        value = pcs_new.write(cs)
        self.assertEqual(expected, value)

    def test_write_new_log_int(self):
        expected = "int_log_a integer [1, 6] [2]log"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(int_log_a)
        value = pcs_new.write(cs)
        self.assertEqual(expected, value)

    def test_write_new_q_int(self):
        expected = "Q16_int_a integer [16, 1024] [520]"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("int_a", 16, 1024, q=16))
        value = pcs_new.write(cs)
        self.assertEqual(expected, value)

    def test_write_new_q_float(self):
        expected = "Q16_float_a real [16.0, 1024.0] [520.0]"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter("float_a", 16, 1024, q=16))
        value = pcs_new.write(cs)
        self.assertEqual(expected, value)

    def test_write_new_log10(self):
        expected = "a real [10.0, 1000.0] [100.0]log"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter("a", 10, 1000, log=True))
        value = pcs_new.write(cs)
        self.assertEqual(expected, value)

    def test_build_new_forbidden(self):
        expected = "a categorical {a, b, c} [a]\nb categorical {a, b, c} [c]\n\n" \
                   "{a=a, b=a}\n{a=a, b=b}\n{a=b, b=a}\n{a=b, b=b}"
        cs = ConfigurationSpace()
        a = CategoricalHyperparameter("a", ["a", "b", "c"], "a")
        b = CategoricalHyperparameter("b", ["a", "b", "c"], "c")
        cs.add_hyperparameter(a)
        cs.add_hyperparameter(b)
        fb = ForbiddenAndConjunction(ForbiddenInClause(a, ["a", "b"]),
                                     ForbiddenInClause(b, ["a", "b"]))
        cs.add_forbidden_clause(fb)
        value = pcs_new.write(cs)
        self.assertIn(expected, value)
        