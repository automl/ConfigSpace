Advanced Example - Constraints and forbidden clauses
====================================================

Constraints
-----------

| ConfigSpace is able to realize constraints in the *configuration space*.
| This is often necessary, because some hyperparameters necessitate some other hyperparameters.
| We will explain you the conditions by showing you a simple example.

ConfigSpace contains the following conditions:

1) EqualsCondition
2) NotEqualsCondition
3) LessThanCondition
4) GreaterThanCondition
5) InCondition

To build even more powerful conditions, it is possible to combine conditions by using conjunction "AND" and "OR":

6) AndConjunction
7) OrConjunction

For demonstration purpose we create a ConfigSpace with the following hyperparameters:

+------------------------+---------------+----------+---------------------------+
| Parameter              | Type          | values   |  condition                |
+========================+===============+==========+===========================+
| a                      | categorical   | 1, 2, 3  |  None                     |
+------------------------+---------------+----------+---------------------------+
| b                      | uniform float | 1.-8.    |  a == 1                   |
+------------------------+---------------+----------+---------------------------+
| c                      | uniform float | 10-100   |  a != 2                   |
+------------------------+---------------+----------+---------------------------+
| d                      | uniform int   | 10-100   |  b < 5 AND b > 2          |
+------------------------+---------------+----------+---------------------------+
| e                      | uniform int   | 10-100   | c in {25,26,27} OR a == 2 |
+------------------------+---------------+----------+---------------------------+

.. note::

    The code of this example can be found here: :math:`\rightarrow` :doc:`auto_examples/AdvancedExample`

First, let's create a ConfigSpace and add the hyperparameters a, b, c::

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

Now, we allow c only to be active, if a is not equal 2.
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

In addition to the conditions, it's also possible to add forbidden clauses to the configuration space.
They allow us to make some more restrictions to the configuration space.

Mainly they are realised by using:

1) ConfigSpace.ForbiddenAndConjunction
2) ConfigSpace.ForbiddenEqualsClause
3) ConfigSpace.ForbiddenInClause

Their usage is shown in the following example.

Our configuration space is defined as:

+------------------------+---------------+----------+---------------------------+
| Parameter              | Type          | values   |  condition                |
+========================+===============+==========+===========================+
| f                      | categorical   | 1, 2, 3  |  None                     |
+------------------------+---------------+----------+---------------------------+
| g                      | categorical   | 2, 5, 6  |  None                     |
+------------------------+---------------+----------+---------------------------+

We have two hyperparameter *f* and *g* and we want to forbid the case, where *f and g is 2 at the same time*::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    f = CSH.CategoricalHyperparameter('f', [1,2,3])
    g = CSH.CategoricalHyperparameter('g', [2,5,6])
    cs.add_hyperparameters([f, g])

    forbidden_clause_f = CS.ForbiddenEqualsClause(f, 2)
    forbidden_clause_g = CS.ForbiddenInClause(g, [2])

    forbidden_clause = CS.ForbiddenAndConjunction(forbidden_clause_f, forbidden_clause_g)

    cs.add_forbidden_clause(forbidden_clause)


