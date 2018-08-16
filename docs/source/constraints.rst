Constraints and Forbidden clauses
=================================

Constraints
-----------

ConfigSpace is able to realize constraints in the *configuration space*.
This is often necessary, because some hyperparameters necessitate some more hyperparameters.

ConfigSpace contains the following conditions:

1) EqualsCondition
2) NotEqualsCondition
3) LessThanCondition
4) GreaterThanCondition
5) InCondition

For more powerful conditions, it is possible to use the conjunction "AND" and "OR":
6) AndConjunction
7) OrConjunction

For demonstratrion purpose we create a ConfigSpace with the following hyperparameters:

+------------------------+---------------+----------+---------------------------+
| Parameter              | Type          | values   |  condition                |
+========================+===============+==========+===========================+
| a                      | categorical   | 1, 2, 3  |  None                     |
+------------------------+---------------+----------+---------------------------+
| b                      | uniform float | 1.-8.    |  a == 1                   |
+------------------------+---------------+----------+---------------------------+
| c                      | uniform float | 10-100   |  a != 2                   |
+------------------------+---------------+----------+---------------------------+
| d                      | uniform int   | 10-100   |  b < 5 && b > 2           |
+------------------------+---------------+----------+---------------------------+
| e                      | uniform int   | 10-100   | c in {25,26,27} || a == 2 |
+------------------------+---------------+----------+---------------------------+

First lets create a ConfigSpace and add the hyperparameters a, b, c, then add the constraints::

   import ConfigSpace as CS
   import ConfigSpace.hyperparameters as CSH

   cs = CS.ConfigurationSpace()
   a = CSH.CategoricalHyperparameter('a', choices=[1, 2, 3])
   b = CSH.UniformFloatHyperparameter('b', lower=1., upper=8., log=False)
   c = CSH.UniformIntegerHyperparameter('c', lower=10, upper=100, log=False)
   d = CSH.UniformIntegerHyperparameter('d', lower=10, upper=100, log=False)
   e = CSH.UniformIntegerHyperparameter('e', lower=10, upper=100, log=False)

   cs.add_hyperparameters([a, b, c, d, e])

1) EqualsCondition
++++++++++++++++++

To realize the equal-condition on hyperparameter b, we can use the **ConfigSpace.EqualsCondition** function::

    cond = CS.EqualsCondition(b, a, 1)
    cs.add_condition(cond)

2) NotEqualsCondition
+++++++++++++++++++++

::

    cond = CS.NotEqualsCondition(c, a, 2)
    cs.add_condition(cond)

3) LessThanCondition
++++++++++++++++++++

::

    less_cond = CS.LessThanCondition(d, b, 5)
    cs.add_condition(less_cond)


4) GreaterThanCondition
+++++++++++++++++++++++

::

    greater_cond = CS.GreaterThanCondition(d, b, 2)
    cs.add_condition(greater_cond)


5) InCondition
++++++++++++++

::

    in_cond = CS.InCondition(e, c, [25, 26, 27])
    cs.add_condition(in_cond)

6) AndConjunction
+++++++++++++++++

We can instead of adding the conditions *less_cond* and *greater_cond*
one after the other to the configspace, use the **ConfigSpace.AndConjunction**::

    less_cond = CS.LessThanCondition(d, b, 5)
    greater_cond = CS.GreaterThanCondition(d, b, 2)
    cs.add_condition(CS.AndConjunction(less_cond, greater_cond))

7) OrConjunction
++++++++++++++++

::

    in_cond = CS.InCondition(e, c, [25, 26, 27])
    equals_cond = CS.EqualsCondition(e, a, 2)
    cs.add_condition(CS.OrConjunction(in_cond, equals_cond))


Forbidden Clauses
-----------------



