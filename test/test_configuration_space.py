from itertools import product
import unittest

from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, Constant
from HPOlibConfigSpace.conditions import EqualsCondition, NotEqualsCondition,\
    InCondition, AndConjunction, OrConjunction
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction


class TestConfigurationSpace(unittest.TestCase):
    # TODO generalize a few simple configuration spaces which are used over
    # and over again throughout this test suite
    # TODO make sure that every function here tests one aspect of the
    # configuration space object!
    def test_add_hyperparameter(self):
        cs = ConfigurationSpace()
        hp = UniformIntegerHyperparameter("name", 0, 10)
        cs.add_hyperparameter(hp)

    def test_add_non_hyperparameter(self):
        cs = ConfigurationSpace()
        non_hp = unittest.TestSuite()
        self.assertRaises(TypeError, cs.add_hyperparameter, non_hp)

    def test_add_hyperparameters_with_equal_names(self):
        cs = ConfigurationSpace()
        hp = UniformIntegerHyperparameter("name", 0, 10)
        cs.add_hyperparameter(hp)
        self.assertRaisesRegexp(ValueError,
                                "Hyperparameter 'name' is already in the "
                                "configuration space.",
                                cs.add_hyperparameter, hp)

    def test_illegal_default_configuration(self):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("loss", ["l1", "l2"])
        hp2 = CategoricalHyperparameter("penalty", ["l1", "l2"])
        cs.add_hyperparameter(hp1)
        cs.add_hyperparameter(hp2)
        forb1 = ForbiddenEqualsClause(hp1, "l1")
        forb2 = ForbiddenEqualsClause(hp2, "l1")
        forb3 = ForbiddenAndConjunction(forb1, forb2)
        #cs.add_forbidden_clause(forb3)
        self.assertRaisesRegexp(ValueError, "Configuration:\n"
            "  loss, Value: l1\n  penalty, Value: l1\n"
            "violates forbidden clause \(Forbidden: loss == l1 && Forbidden: "
            "penalty == l1\)", cs.add_forbidden_clause, forb3)

    def test_add_non_condition(self):
        cs = ConfigurationSpace()
        non_cond = unittest.TestSuite()
        self.assertRaises(TypeError, cs.add_condition, non_cond)

    def test_hyperparameters_with_valid_condition(self):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs.add_hyperparameter(hp1)
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cs.add_hyperparameter(hp2)
        cond = EqualsCondition(hp2, hp1, 0)
        cs.add_condition(cond)
        self.assertEqual(len(cs._dg.nodes()), 3)

    def test_condition_without_added_hyperparameters(self):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cond = EqualsCondition(hp2, hp1, 0)
        self.assertRaisesRegexp(ValueError, "Child hyperparameter 'child' not "
                                "in configuration space.", cs.add_condition,
                                cond)
        cs.add_hyperparameter(hp1)
        self.assertRaisesRegexp(ValueError, "Child hyperparameter 'child' not "
                                "in configuration space.", cs.add_condition,
                                cond)

        # Test also the parent hyperparameter
        cs2 = ConfigurationSpace()
        cs2.add_hyperparameter(hp2)
        self.assertRaisesRegexp(ValueError, "Parent hyperparameter 'parent' "
                                "not in configuration space.",
                                cs2.add_condition, cond)


    def test_condition_with_cycles(self):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs.add_hyperparameter(hp1)
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cs.add_hyperparameter(hp2)
        cond1 = EqualsCondition(hp2, hp1, 0)
        cs.add_condition(cond1)
        cond2 = EqualsCondition(hp1, hp2, 0)
        self.assertRaisesRegexp(ValueError, "Hyperparameter configuration "
                                "contains a cycle \[\['child', 'parent'\]\]",
                                cs.add_condition, cond2)

    def test_add_conjunction(self):
        hp1 = CategoricalHyperparameter("input1", [0, 1])
        hp2 = CategoricalHyperparameter("input2", [0, 1])
        hp3 = CategoricalHyperparameter("input3", [0, 1])
        hp4 = Constant("And", "True")

        cond1 = EqualsCondition(hp4, hp1, 1)
        cond2 = EqualsCondition(hp4, hp2, 1)
        cond3 = EqualsCondition(hp4, hp3, 1)

        andconj1 = AndConjunction(cond1, cond2, cond3)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(hp1)
        cs.add_hyperparameter(hp2)
        cs.add_hyperparameter(hp3)
        cs.add_hyperparameter(hp4)

        cs.add_condition(andconj1)
        self.assertNotIn(hp4, cs.get_all_uncoditional_hyperparameters())

    def test_add_second_condition_wo_conjunction(self):
        hp1 = CategoricalHyperparameter("input1", [0, 1])
        hp2 = CategoricalHyperparameter("input2", [0, 1])
        hp3 = Constant("And", "True")

        cond1 = EqualsCondition(hp3, hp1, 1)
        cond2 = EqualsCondition(hp3, hp2, 1)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(hp1)
        cs.add_hyperparameter(hp2)
        cs.add_hyperparameter(hp3)

        cs.add_condition(cond1)
        self.assertRaisesRegexp(ValueError,
                                "Adding a second condition \(different\) for a "
                                "hyperparameter is ambigouos and "
                                "therefore forbidden. Add a conjunction "
                                "instead!",
                                cs.add_condition, cond2)

    def test_add_forbidden_clause(self):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("input1", [0, 1])
        cs.add_hyperparameter(hp1)
        forb = ForbiddenEqualsClause(hp1, 1)
        # TODO add checking whether a forbidden clause makes sense at all
        cs.add_forbidden_clause(forb)
        # TODO add something to properly retrieve the forbidden clauses
        self.assertEqual(str(cs), "Configuration space object:\n  "
                                  "Hyperparameters:\n    input1, "
                                  "Type: Categorical, Choices: {0, 1}, "
                                  "Default: 0\n"
                                  "  Forbidden Clauses:\n"
                                  "    Forbidden: input1 == 1\n")

    def test_get_hyperparameters(self):
        cs = ConfigurationSpace()
        self.assertEqual([], cs.get_hyperparameters())
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs.add_hyperparameter(hp1)
        self.assertEqual([hp1], cs.get_hyperparameters())
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cs.add_hyperparameter(hp2)
        cond1 = EqualsCondition(hp2, hp1, 1)
        cs.add_condition(cond1)
        self.assertEqual([hp1, hp2], cs.get_hyperparameters())
        self.assertEqual([hp2, hp1], cs.get_hyperparameters(
            order='dfs_postorder'))
        # TODO: I need more tests for the topological sort!
        self.assertEqual([hp1, hp2], cs.get_hyperparameters(
            order='topological'))

    def test_get_hyperparameters_topological_sort(self):
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

        self.assertEqual([hp1, hp2, hp3, hp5, hp4, hp6],
                         cs.get_hyperparameters(order='topological'))

    def test_get_hyperparameter(self):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs.add_hyperparameter(hp1)
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cs.add_hyperparameter(hp2)

        retval = cs.get_hyperparameter("parent")
        self.assertEqual(hp1, retval)
        retval = cs.get_hyperparameter("child")
        self.assertEqual(hp2, retval)
        self.assertRaises(KeyError, cs.get_hyperparameter, "grandfather")

    def test_get_conditions(self):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs.add_hyperparameter(hp1)
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cs.add_hyperparameter(hp2)
        self.assertEqual([], cs.get_conditions())
        cond1 = EqualsCondition(hp2, hp1, 0)
        cs.add_condition(cond1)
        self.assertEqual([cond1], cs.get_conditions())

    def test_get_parents_and_children_of(self):
        cs = ConfigurationSpace()
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs.add_hyperparameter(hp1)
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cs.add_hyperparameter(hp2)
        cond1 = EqualsCondition(hp2, hp1, 0)
        cs.add_condition(cond1)

        self.assertEqual([cond1], cs.get_parents_of(hp2.name))
        self.assertEqual([cond1], cs.get_children_of(hp1.name))

    def test_check_configuration_input_checking(self):
        cs = ConfigurationSpace()
        self.assertRaisesRegexp(TypeError, "The method check_configuration must"
                                           " be called with an instance of "
                                           "%s." % Configuration,
                                cs.check_configuration, "String")

    def test_check_configuration(self):
        # TODO this is only a smoke test
        # TODO actually, this rather tests the evaluate methods in the
        # conditions module!
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
        cond4 = EqualsCondition(hp6, hp4, 1)
        cond5 = EqualsCondition(hp6, hp5, 1)

        conj1 = AndConjunction(cond1, cond2)
        conj2 = OrConjunction(conj1, cond3)
        conj3 = AndConjunction(conj2, cond4, cond5)
        cs.add_condition(conj3)

        expected_outcomes = [False, False, False, False, False,
                             False, False, True, False, False,
                             False, False, False, False, False,
                             True, False, False, False, True,
                             False, False, False, True, False,
                             False, False, False, False, False,
                             False, True]

        for idx, values in enumerate(product([0, 1], repeat=5)):
            # The first hyperparameter returned is the Constant!
            instantiations = [cs.get_hyperparameters(order='topological')
                              [jdx].instantiate(values[jdx])
                              for jdx in range(len(values))]

            evaluation = conj3.evaluate(instantiations)
            self.assertEqual(expected_outcomes[idx], evaluation)

            if evaluation == False:
                self.assertRaisesRegexp(ValueError,
                                        "Inactive hyperparameter 'AND' must "
                                        "not be specified, but is: "
                                        "'AND, Constant: True'.",
                                        Configuration, cs,
                                        input1=cs.get_hyperparameter(
                                            "input1").instantiate(values[0]),
                                        input2=cs.get_hyperparameter(
                                            "input2").instantiate(values[1]),
                                        input3=cs.get_hyperparameter(
                                            "input3").instantiate(values[2]),
                                        input4=cs.get_hyperparameter(
                                            "input4").instantiate(values[3]),
                                        input5=cs.get_hyperparameter(
                                            "input5").instantiate(values[4]),
                                        AND=cs.get_hyperparameter(
                                            "AND").instantiate("True"))
            else:
                Configuration(cs, input1 = cs.get_hyperparameter(
                                   "input1").instantiate(values[0]),
                              input2 = cs.get_hyperparameter(
                                  "input2").instantiate(values[1]),
                              input3 = cs.get_hyperparameter(
                                   "input3").instantiate(values[2]),
                              input4 = cs.get_hyperparameter(
                                   "input4").instantiate(values[3]),
                              input5 = cs.get_hyperparameter(
                                    "input5").instantiate(values[4]),
                              AND = cs.get_hyperparameter(
                                    "AND").instantiate("True"))


    def test_eq(self):
        # Compare empty configuration spaces
        cs1 = ConfigurationSpace()
        cs2 = ConfigurationSpace()
        self.assertEqual(cs1, cs2)

        # Compare to something which isn't a configuration space
        self.assertTrue(not (cs1 == "ConfigurationSpace"))

        # Compare to equal configuration spaces
        hp1 = CategoricalHyperparameter("parent", [0, 1])
        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        hp3 = UniformIntegerHyperparameter("friend", 0, 5)
        cond1 = EqualsCondition(hp2, hp1, 0)
        cs1.add_hyperparameter(hp1)
        cs1.add_hyperparameter(hp2)
        cs1.add_condition(cond1)
        cs2.add_hyperparameter(hp1)
        cs2.add_hyperparameter(hp2)
        cs2.add_condition(cond1)
        self.assertEqual(cs1, cs2)
        cs1.add_hyperparameter(hp3)
        self.assertFalse(cs1 == cs2)

    def test_neq(self):
        cs1 = ConfigurationSpace()
        self.assertNotEqual(cs1, "ConfigurationSpace")

    def test_repr(self):
        cs1 = ConfigurationSpace()
        retval = cs1.__str__()
        self.assertEqual("Configuration space object:\n  Hyperparameters:\n",
                         retval)

        hp1 = CategoricalHyperparameter("parent", [0, 1])
        cs1.add_hyperparameter(hp1)
        retval = cs1.__str__()
        self.assertEqual("Configuration space object:\n  Hyperparameters:\n"
                         "    %s\n" % str(hp1), retval)

        hp2 = UniformIntegerHyperparameter("child", 0, 10)
        cond1 = EqualsCondition(hp2, hp1, 0)
        cs1.add_hyperparameter(hp2)
        cs1.add_condition(cond1)
        retval = cs1.__str__()
        self.assertEqual("Configuration space object:\n  Hyperparameters:\n"
                         "    %s\n    %s\n  Conditions:\n    %s\n" %
                         (str(hp1), str(hp2), str(cond1)), retval)