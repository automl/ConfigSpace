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

from itertools import product
import unittest
import warnings

import numpy as np

from ConfigSpace.hyperparameters import \
    UniformIntegerHyperparameter, CategoricalHyperparameter

# from ConfigSpace.forbidden import ForbiddenEqualsClause, \
#     ForbiddenInClause, ForbiddenAndConjunction
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenInClause, ForbiddenAndConjunction


class TestForbidden(unittest.TestCase):
    # TODO: return only copies of the objects!

    def test_forbidden_equals_clause(self):
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        hp3 = CategoricalHyperparameter("grandchild", ["hot", "cold"])

        self.assertRaisesRegexp(
            TypeError,
            "Argument 'hyperparameter' has incorrect type \(expected ConfigSpace.hyperparameters.Hyperparameter, got str\)",
            ForbiddenEqualsClause, "HP1", 1,
        )

        self.assertRaisesRegexp(
            ValueError,
            "Forbidden clause must be instantiated with a legal hyperparameter value for "
            "'parent, Type: Categorical, Choices: \{0, 1\}, Default: 0', but got '2'",
            ForbiddenEqualsClause, hp1, 2,
        )

        forb1 = ForbiddenEqualsClause(hp1, 1)
        forb1_ = ForbiddenEqualsClause(hp1, 1)
        forb1__ = ForbiddenEqualsClause(hp1, 0)
        forb2 = ForbiddenEqualsClause(hp2, 10)
        forb3 = ForbiddenEqualsClause(hp3, "hot")
        forb3_ = ForbiddenEqualsClause(hp3, "hot")

        self.assertEqual(forb3, forb3_)
        # print("\eq0:", 1, 1)
        # self.assertEqual(1, 1)
        # print("\neq1:", forb1, forb1_)
        self.assertEqual(forb1, forb1_)
        # print("\nneq2:", forb1, "forb1")
        self.assertNotEqual(forb1, "forb1")
        # print("\nneq3:", forb1, forb2)
        self.assertNotEqual(forb1, forb2)
        # print("\nneq4:", forb1_, forb1)
        self.assertNotEqual(forb1__, forb1)
        # print("\neq5:", "Forbidden: parent == 1", str(forb1))
        self.assertEqual("Forbidden: parent == 1", str(forb1))

        # print("\nraisereg6:")
        self.assertRaisesRegexp(ValueError,
                                "Is_forbidden must be called with the "
                                "instanstatiated hyperparameter in the "
                                "forbidden clause; you are missing "
                                "'parent'", forb1.is_forbidden,
                                {1: hp2}, True)
        # print("\nneq7:")
        self.assertFalse(forb1.is_forbidden({'child': 1}, strict=False))
        # print("\nneq8:")
        self.assertFalse(forb1.is_forbidden({'parent': 0}, True))
        # print("\nneq9:")
        self.assertTrue(forb1.is_forbidden({'parent': 1}, True))

        # Test forbidden on vector values
        hyperparameter_idx = {
            hp1.name: 0,
            hp2.name: 1
        }
        forb1.set_vector_idx(hyperparameter_idx)
        # print("\nneq10:")
        self.assertFalse(forb1.is_forbidden_vector(np.array([np.NaN, np.NaN]), strict=False))
        # print("\nneq11:")
        self.assertFalse(forb1.is_forbidden_vector(np.array([0., np.NaN]), strict=False))
        # print("\nneq12:")
        self.assertTrue(forb1.is_forbidden_vector(np.array([1., np.NaN]), strict=False))


    def test_in_condition(self):
        hp1 = CategoricalHyperparameter("parent", [0, 1, 2, 3, 4])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        hp3 = UniformIntegerHyperparameter("child2", 0, 10)
        hp4 = CategoricalHyperparameter("grandchild", ["hot", "cold", "warm"])

        self.assertRaisesRegexp(
            TypeError,
            "Argument 'hyperparameter' has incorrect type \(expected ConfigSpace.hyperparameters.Hyperparameter, got str\)",
            ForbiddenInClause, "HP1", 1,
        )

        self.assertRaisesRegexp(
            ValueError,
            "Forbidden clause must be instantiated with a "
            "legal hyperparameter value for "
            "'parent, Type: Categorical, Choices: {0, 1, 2, 3, 4}, "
            "Default: 0', but got '5'",
            ForbiddenInClause, hp1, [5],
        )

        forb1 = ForbiddenInClause(hp2, [5, 6, 7, 8, 9])
        forb1_ = ForbiddenInClause(hp2, [9, 8, 7, 6, 5])
        forb2 = ForbiddenInClause(hp2, [5, 6, 7, 8])
        forb3 = ForbiddenInClause(hp3, [5, 6, 7, 8, 9])
        forb4 = ForbiddenInClause(hp4, ["hot", "cold"])
        forb4_ = ForbiddenInClause(hp4, ["hot", "cold"])
        forb5 = ForbiddenInClause(hp1, [3, 4])
        forb5_ = ForbiddenInClause(hp1, [3, 4])

        self.assertEqual(forb5, forb5_)
        self.assertEqual(forb4, forb4_)

        # print("\nTest1:")
        self.assertEqual(forb1, forb1_)
        # print("\nTest2:")
        self.assertNotEqual(forb1, forb2)
        # print("\nTest3:")
        self.assertNotEqual(forb1, forb3)
        # print("\nTest4:")
        self.assertEqual("Forbidden: child in {5, 6, 7, 8, 9}", str(forb1))
        # print("\nTest5:")
        self.assertRaisesRegexp(ValueError,
                                "Is_forbidden must be called with the "
                                "instanstatiated hyperparameter in the "
                                "forbidden clause; you are missing "
                                "'child'", forb1.is_forbidden,
                                {'parent': 1}, True)
        # print("\nTest6:")
        self.assertFalse(forb1.is_forbidden({'parent': 1}, strict=False))
        # print("\nTest7:")
        for i in range(0, 5):
            self.assertFalse(forb1.is_forbidden({'child': i}, True))
        # print("\nTest8:")
        for i in range(5, 10):
            self.assertTrue(forb1.is_forbidden({'child': i}, True))

        # Test forbidden on vector values
        hyperparameter_idx = {
            hp1.name: 0,
            hp2.name: 1
        }
        forb1.set_vector_idx(hyperparameter_idx)
        # print("\nTest9:")
        self.assertFalse(forb1.is_forbidden_vector(np.array([np.NaN, np.NaN]), strict=False))
        # print("\nTest10:")
        self.assertFalse(forb1.is_forbidden_vector(np.array([np.NaN, 0]), strict=False))
        correct_vector_value = hp2._inverse_transform(6)
        # print("\nTest11:")
        print(correct_vector_value, np.array([np.NaN, correct_vector_value]))
        self.assertTrue(forb1.is_forbidden_vector(np.array([np.NaN, correct_vector_value]), strict=False))
        # print("\nfinished:")

    def test_and_conjunction(self):
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 2)
        hp3 = UniformIntegerHyperparameter("child2", 0, 2)
        hp4 = UniformIntegerHyperparameter("child3", 0, 2)

        forb2 = ForbiddenEqualsClause(hp1, 1)
        forb3 = ForbiddenInClause(hp2, range(2, 3))
        forb4 = ForbiddenInClause(hp3, range(2, 3))
        forb5 = ForbiddenInClause(hp4, range(2, 3))

        and1 = ForbiddenAndConjunction(forb2, forb3)
        and2 = ForbiddenAndConjunction(forb2, forb4)
        and3 = ForbiddenAndConjunction(forb2, forb5)

        total_and = ForbiddenAndConjunction(and1, and2, and3)
        self.assertEqual("((Forbidden: parent == 1 && Forbidden: child in {2}) "
                         "&& (Forbidden: parent == 1 && Forbidden: child2 in {2}) "
                         "&& (Forbidden: parent == 1 && Forbidden: child3 in "
                         "{2}))", str(total_and))

        results = [False, False, False, False, False,
                   False, False, False, False, False,
                   False, False, False, False, False,
                   False, False, False, False, False,
                   False, False, False, False, False,
                   False, False, False, False, False,
                   False, False, False, False, False,
                   False, False, False, False, False,
                   False, False, False, False, False,
                   False, False, False, False, False,
                   False, False, False, True]

        for i, values in enumerate(product(range(2), range(3), range(3),
                                           range(3))):
            is_forbidden = total_and.is_forbidden(
                {"parent": values[0],
                 "child": values[1],
                 "child2": values[2],
                 "child3": values[3]},
                True,
            )

            self.assertEqual(results[i], is_forbidden)

            self.assertFalse(total_and.is_forbidden({}, strict=False))
