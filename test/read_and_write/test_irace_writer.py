# Copyright (c) 2014-2017, ConfigSpace developers
# Matthias Feurer
# Katharina Eggensperger
# Mohsin Ali
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
import shutil
import tempfile
import os

from ConfigSpace.configuration_space import ConfigurationSpace
import ConfigSpace.read_and_write.irace as irace

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    OrdinalHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition, \
    AndConjunction, OrConjunction
from ConfigSpace.forbidden import ForbiddenInClause, ForbiddenEqualsClause, \
    ForbiddenAndConjunction


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


int_a = UniformIntegerHyperparameter("int_a", -1, 6)


class TestIraceWriter(unittest.TestCase):
    '''
    Test IRACE writer
    '''
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.test_dir)

    def test_write_illegal_argument(self):
        sp = {"a": int_a}
        self.assertRaisesRegex(
            TypeError, r"irace.write expects an "
            r"instance of "
            r"<class "
            r"'ConfigSpace.configuration_"
            r"space.ConfigurationSpace'>, you provided "
            r"'<(type|class) 'dict'>'", irace.write, sp)

    def test_write_int(self):
        expected = "int_a '--int_a ' i (-1, 6)\n"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(int_a)
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_float(self):
        expected = "float_a '--float_a ' r (16.000000, 1024.000000)\n"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformFloatHyperparameter("float_a", 16, 1024))
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_categorical(self):
        expected = "cat_a '--cat_a ' c {a,b,c}\n"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            CategoricalHyperparameter("cat_a", ["a", "b", "c"]))
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_ordinal(self):
        expected = "ord_a '--ord_a ' o {a,b,3}\n"
        cs = ConfigurationSpace()
        cs.add_hyperparameter(OrdinalHyperparameter("ord_a", ["a", "b", 3]))
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_equals_condition_categorical(self):
        expected = "ls '--ls ' c {sa,ca,ny}\ntemp '--temp ' r (0.500000, 1.000000)|  ls==sa\n"

        temp = UniformFloatHyperparameter("temp", 0.5, 1)
        ls = CategoricalHyperparameter("ls", ["sa", "ca", "ny"], "sa")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(temp)
        cs.add_hyperparameter(ls)
        c1 = EqualsCondition(temp, ls, 'sa')
        cs.add_condition(c1)
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_equals_condition_numerical(self):
        expected = "temp '--temp ' i (1, 2)\nls '--ls ' c {sa,ca,ny}|  temp==2\n"

        temp = UniformIntegerHyperparameter("temp", 1, 2)
        ls = CategoricalHyperparameter("ls", ["sa", "ca", "ny"], "sa")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(temp)
        cs.add_hyperparameter(ls)
        c1 = EqualsCondition(ls, temp, 2)
        cs.add_condition(c1)
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_in_condition(self):
        expected = "ls '--ls ' c {sa,ca,ny}\ntemp '--temp ' r (0.500000, 1.000000)|  ls  %in%  c(sa,ca)\n"

        temp = UniformFloatHyperparameter("temp", 0.5, 1)
        ls = CategoricalHyperparameter("ls", ["sa", "ca", "ny"], "sa")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(temp)
        cs.add_hyperparameter(ls)
        c1 = InCondition(temp, ls, ['sa', 'ca'])
        cs.add_condition(c1)
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_AndConjunction_condition(self):
        expected = (
            "lp '--lp ' c {mi,bo}\nls '--ls ' c {sa,ca,ny}\ntemp '--temp ' "
            "r (0.500000, 1.000000)|  ls  %in%  c(sa,ca)  &&  lp  %in%  c(bo)\n"
        )

        temp = UniformFloatHyperparameter("temp", 0.5, 1)
        ls = CategoricalHyperparameter("ls", ["sa", "ca", "ny"], "sa")
        lp = CategoricalHyperparameter("lp", ["mi", "bo"], "bo")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(temp)
        cs.add_hyperparameter(lp)
        cs.add_hyperparameter(ls)

        c1 = InCondition(temp, ls, ['sa', 'ca'])
        c2 = InCondition(temp, lp, ['bo'])
        c3 = AndConjunction(c1, c2)
        cs.add_condition(c3)
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_OrConjunction_condition(self):
        import numpy as np
        expected = (
            "lp '--lp ' c {mi,bo}\ntemp '--temp ' r (2.000000, 5.000000)\nls "
            "'--ls ' c {sa,ca,ny}|  temp==3.0  ||  lp  %in%  c(bo)\n")

        temp = UniformFloatHyperparameter(
            "temp", np.exp(2), np.exp(5), log=True)
        ls = CategoricalHyperparameter("ls", ["sa", "ca", "ny"], "sa")
        lp = CategoricalHyperparameter("lp", ["mi", "bo"], "bo")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(temp)
        cs.add_hyperparameter(lp)
        cs.add_hyperparameter(ls)

        c1 = EqualsCondition(ls, temp, np.exp(3))
        c2 = InCondition(ls, lp, ['bo'])
        c3 = OrConjunction(c1, c2)
        cs.add_condition(c3)
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_forbidden(self):
        cs = ConfigurationSpace()

        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 2)
        hp3 = UniformIntegerHyperparameter("child2", 0, 2)
        hp4 = UniformIntegerHyperparameter("child3", 0, 2)
        hp5 = CategoricalHyperparameter("child4", [4, 5, 6, 7])

        cs.add_hyperparameters([hp1, hp2, hp3, hp4, hp5])

        forb2 = ForbiddenEqualsClause(hp1, 1)
        forb3 = ForbiddenInClause(hp2, range(2, 3))
        forb4 = ForbiddenInClause(hp3, range(2, 3))
        forb5 = ForbiddenInClause(hp4, range(2, 3))
        forb6 = ForbiddenInClause(hp5, [6, 7])

        and1 = ForbiddenAndConjunction(forb2, forb3)
        and2 = ForbiddenAndConjunction(forb2, forb4)
        and3 = ForbiddenAndConjunction(forb2, forb5)

        cs.add_forbidden_clauses(
            [forb2, forb3, forb4, forb5, forb6, and1, and2, and3])

        irace.write(cs)  # generates file called forbidden.txt

    def test_write_log_int(self):
        expected = "int_log '--int_log ' i (2, 4)\n"
        int_log = UniformIntegerHyperparameter("int_log", 10, 100, log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(int_log)
        value = irace.write(cs)
        self.assertEqual(expected, value)

    def test_write_log_float(self):
        import numpy as np
        expected = "float_log '--float_log ' r (2.000000, 5.000000)\n"
        float_log = UniformFloatHyperparameter(
            "float_log", np.exp(2), np.exp(5), log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(float_log)
        value = irace.write(cs)
        self.assertEqual(expected, value)
