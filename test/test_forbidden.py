from itertools import product
import unittest
import warnings

from HPOlibConfigSpace.hyperparameters import \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenInClause, ForbiddenAndConjunction


class TestForbidden(unittest.TestCase):
    # TODO: return only copies of the objects!
    def test_forbidden_equals_clause(self):
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)

        self.assertRaisesRegexp(TypeError, "HP1' is not of"
            " type <class 'HPOlibConfigSpace.hyperparameters.Hyperparameter'>.",
                                ForbiddenEqualsClause, "HP1", 1)

        self.assertRaisesRegexp(ValueError,
                                "Forbidden clause must be instantiated with a "
                                "legal hyperparameter value for "
                                "'parent, Type: Categorical, Choices: \{0, "
                                "1\}, Default: 0', but got '2'",
                                ForbiddenEqualsClause, hp1, 2)

        forb1 = ForbiddenEqualsClause(hp1, 1)
        forb1_ = ForbiddenEqualsClause(hp1, 1)
        forb1__ = ForbiddenEqualsClause(hp1, 0)
        forb2 = ForbiddenEqualsClause(hp2, 10)

        self.assertEqual(forb1, forb1_)
        self.assertNotEqual(forb1, "forb1")
        self.assertNotEqual(forb1, forb2)
        self.assertNotEqual(forb1__, forb1)
        self.assertEqual("Forbidden: parent == 1", str(forb1))

        self.assertRaisesRegexp(ValueError,
                                "Is_forbidden must be called with the "
                                "instanstatiated hyperparameter in the "
                                "forbidden clause; you are missing "
                                "'parent'", forb1.is_forbidden,
                                [{1: hp2}])
        self.assertFalse(forb1.is_forbidden({'child': 1}, strict=False))
        self.assertFalse(forb1.is_forbidden({'parent': 0}))
        self.assertTrue(forb1.is_forbidden({'parent': 1}))

    def test_in_condition(self):
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        hp3 = UniformIntegerHyperparameter("child2", 0, 10)

        self.assertRaisesRegexp(TypeError, "Argument 'hyperparameter' is not of"
                                " type <class 'HPOlibConfigSpace.hyperparameters.Hyperparameter'>.",
                                ForbiddenInClause, "HP1", 1)

        self.assertRaisesRegexp(ValueError,
                                "Forbidden clause must be instantiated with a "
                                "legal hyperparameter value for "
                                "'parent, Type: Categorical, Choices: {0, 1}, "
                                "Default: 0', but got '2'",
                                ForbiddenInClause, hp1, [2])

        forb1 = ForbiddenInClause(hp2, [5, 6, 7, 8, 9])
        forb1_ = ForbiddenInClause(hp2, [9, 8, 7, 6, 5])
        forb2 = ForbiddenInClause(hp2, [5, 6, 7, 8])
        forb3 = ForbiddenInClause(hp3, [5, 6, 7, 8, 9])

        self.assertEqual(forb1, forb1_)
        self.assertNotEqual(forb1, forb2)
        self.assertNotEqual(forb1, forb3)
        self.assertEqual("Forbidden: child in {5, 6, 7, 8, 9}", str(forb1))

        self.assertRaisesRegexp(ValueError,
                                "Is_forbidden must be called with the "
                                "instanstatiated hyperparameter in the "
                                "forbidden clause; you are missing "
                                "'child'", forb1.is_forbidden,
                                {'parent': 1})
        self.assertFalse(forb1.is_forbidden({'parent': 1}, strict=False))

        for i in range(0, 5):
            self.assertFalse(forb1.is_forbidden({'child': i}))

        for i in range(5, 10):
            self.assertTrue(forb1.is_forbidden({'child': i}))


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
                 "child3": values[3]})

            self.assertEqual(results[i], is_forbidden)

            self.assertFalse(total_and.is_forbidden([], strict=False))
