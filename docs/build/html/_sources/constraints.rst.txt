Conditions and Forbidden clauses [API]
======================================

Constraints
-----------

| It is often necessary to make some constraint on hyperparameters. For example we have one hyperparameter, which decides whether
| we use methodA or methodB, and each method has additional unique hyperparameter.
| Those depending hyperparameters are called *child* hyperparameter.
| It is desired, that they should only be "active", if their *parent* hyperparameter
| has the right value.
| This can be accomplished, using **conditions**.

To see an example of how to use conditions, please take a look at the :doc:`advanced example <AdvancedExample>`

1) EqualsCondition
++++++++++++++++++

.. py:class:: ConfigSpace.conditions.EqualsCondition(child: Hyperparameter, parent: Hyperparameter, value: Union[str,float,int]) -> None:

    The equal-condition adds the *child*-hyperparameter to the configuration space if and only if the *parent*-hyperparameter's value is equal to *value*

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the equal-condition is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the equal-condition
    :param str,float,int value: value, which the parent is compared to

Example::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    a = CSH.CategoricalHyperparameter('a', choices=[1, 2, 3])
    b = CSH.UniformFloatHyperparameter('b', lower=1., upper=8., log=False)
    cs.add_hyperparameters([a, b])

    # makes 'b' an active hyperparameter if 'a' has the value 1
    cond = CS.EqualsCondition(b, a, 1)
    cs.add_condition(cond)

2) NotEqualsCondition
+++++++++++++++++++++

.. py:class:: ConfigSpace.conditions.NotEqualsCondition(child: Hyperparameter, parent: Hyperparameter, value: Union[str, float, int]) -> None:

    The not-equals-condition adds the *child*-hyperparameter to the configuration space if and only if the *parent*-hyperparameter's value is not equal to *value*

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the not-equals-condition is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the not-equal-condition
    :param str,float,int value: value, which the parent is compared to

Example::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    a = CSH.CategoricalHyperparameter('a', choices=[1, 2, 3])
    b = CSH.UniformFloatHyperparameter('b', lower=1., upper=8., log=False)
    cs.add_hyperparameters([a, b])

    # makes 'b' an active hyperparameter if 'a' has **not** the value 1
    cond = CS.NotEqualsCondition(b, a, 1)
    cs.add_condition(cond)

3) LessThanCondition
++++++++++++++++++++

.. py:class:: ConfigSpace.conditions.LessThanCondition(child: Hyperparameter, parent: Hyperparameter, value: Union[str, float, int]) -> None:

    The less-than-condition adds the *child*-hyperparameter to the configuration space if and only if the *parent*-hyperparameter's value is less than *value*

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the less-than-condition is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the less-than-condition
    :param str,float,int value: value, which the parent is compared to

Example::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    a = CSH.UniformFloatHyperparameter('a', lower=0., upper=10.)
    b = CSH.UniformFloatHyperparameter('b', lower=1., upper=8., log=False)
    cs.add_hyperparameters([a, b])

    # makes 'b' an active hyperparameter if 'a' is less than 5
    cond = CS.LessThanCondition(b, a, 5.)
    cs.add_condition(cond)


4) GreaterThanCondition
+++++++++++++++++++++++

.. py:class:: ConfigSpace.conditions.GreaterThanCondition(child: Hyperparameter, parent: Hyperparameter, value: Union[str, float, int]) -> None:

    The greater-than-condition adds the *child*-hyperparameter to the configuration space if and only if the *parent*-hyperparameter's value is greater than *value*

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the greater-than-condition is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the greater-than-condition
    :param str,float,int value: value, which the parent is compared to

Example::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    a = CSH.UniformFloatHyperparameter('a', lower=0., upper=10.)
    b = CSH.UniformFloatHyperparameter('b', lower=1., upper=8., log=False)
    cs.add_hyperparameters([a, b])

    # makes 'b' an active hyperparameter if 'a' is greater than 5
    cond = CS.GreaterThanCondition(b, a, 5.)
    cs.add_condition(cond)

5) InCondition
++++++++++++++

.. py:class:: ConfigSpace.conditions.InCondition(child: Hyperparameter, parent: Hyperparameter, values: List[Union[str, float, int]]) -> None:

    The in-condition adds the *child*-hyperparameter to the configuration space if and only if the *parent*-hyperparameter's value is in the subset *values*

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the in-condition is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the in-condition
    :param List[str,float,int] value: subset of values, which the parent is compared to

Example::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    a = CSH.UniformIntegerHyperparameter('a', lower=0, upper=10)
    b = CSH.UniformFloatHyperparameter('b', lower=1., upper=8., log=False)
    cs.add_hyperparameters([a, b])

    # makes 'b' an active hyperparameter if 'a' is in the set [1, 2, 3, 4]
    cond = CS.InCondition(b, a, [1, 2, 3, 4])
    cs.add_condition(cond)


6) AndConjunction
+++++++++++++++++

By using the *and*-conjunction, we can easily connect constraints.

.. py:class:: ConfigSpace.conditions.AndConjunction(*args: AbstractCondition) -> None:

    :param AbstractCondition args: conditions, which will be combined with an and-conjunction

The following example shows how we can combine two constraints with an *and*-conjunction.
Example::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()

    a = CSH.UniformIntegerHyperparameter('a', lower=5, upper=15)
    b = CSH.UniformIntegerHyperparameter('b', lower=0, upper=10)
    c = CSH.UniformFloatHyperparameter('c', lower=0., upper=1.)
    cs.add_hyperparameters([a, b, c])

    less_cond = CS.LessThanCondition(c, a, 10)
    greater_cond = CS.GreaterThanCondition(c, b, 5)

    cs.add_condition(CS.AndConjunction(less_cond, greater_cond))

7) OrConjunction
++++++++++++++++


Similar to the *and*-conjunction, new constraints can be combined by using the *or*-conjunction.

.. py:class:: ConfigSpace.conditions.OrConjunction(*args: AbstractCondition) -> None:

    :param AbstractCondition args: conditions, which will be combined with an or-conjunction

The following example shows how we can combine two constraints with an *or*-conjunction.
Example::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()

    a = CSH.UniformIntegerHyperparameter('a', lower=5, upper=15)
    b = CSH.UniformIntegerHyperparameter('b', lower=0, upper=10)
    c = CSH.UniformFloatHyperparameter('c', lower=0., upper=1.)
    cs.add_hyperparameters([a, b, c])

    less_cond = CS.LessThanCondition(c, a, 10)
    greater_cond = CS.GreaterThanCondition(c, b, 5)

    cs.add_condition(CS.OrConjunction(less_cond, greater_cond))

Forbidden Clauses
-----------------

In addition to the conditions, it's also possible to add forbidden clauses to the configuration space.
They allow us to make some more restrictions to the configuration space.

The forbidden clauses are also captured in the examples. Please take a look at the :doc:`advanced example <AdvancedExample>`

1) ForbiddenEqualsClause
++++++++++++++++++++++++
.. py:class:: ConfigSpace.ForbiddenEqualsClause(hyperparameter: Hyperparameter, value: Any) -> None:

    :param Hyperparameter hyperparameter: hyperparameter on which a restriction will be made
    :param Any value: This value will be forbidden

Example::

    cs = CS.ConfigurationSpace()
    a = CSH.CategoricalHyperparameter('a', [1,2,3])
    cs.add_hyperparameters([a])

    # It forbids the value 2 for the hyperparameter a
    forbidden_clause_a = CS.ForbiddenEqualsClause(a, 2)
    cs.add_forbidden_clause(forbidden_clause_a)

2) ForbiddenInClause
++++++++++++++++++++
.. py:class:: ConfigSpace.ForbiddenInClause(hyperparameter: Dict[str, Union[None, str, float, int]], values: Any) -> None:

    :param Hyperparameter hyperparameter: hyperparameter on which a restriction will be made
    :param Any value: This value will be forbidden

.. note::

    The forbidden values have to be a subset of the hyperparameter's values.

Example::

    cs = CS.ConfigurationSpace()
    a = CSH.CategoricalHyperparameter('a', [1,2,3])
    cs.add_hyperparameters([a])

    # It forbids the values 2, 3, 4 for the hyperparameter 'a'
    forbidden_clause_a = CS.ForbiddenInClause(a, [2, 3])

    cs.add_forbidden_clause(forbidden_clause_a)

3) ForbiddenAndConjunction
++++++++++++++++++++++++++
.. py:class:: ConfigSpace.ForbiddenAndConjunction(hyperparameter: Hyperparameter, value: Any) -> None:

    The *ForbiddenAndConjunction* combines forbidden-clauses, which allows to build powerful constraints.

    :param Hyperparameter hyperparameter: hyperparameter on which a restriction will be made
    :param Any value: This value will be forbidden

Example::

    cs = CS.ConfigurationSpace()
    a = CSH.CategoricalHyperparameter('a', [1,2,3])
    b = CSH.CategoricalHyperparameter('b', [2,5,6])
    cs.add_hyperparameters([a, b])

    forbidden_clause_a = CS.ForbiddenEqualsClause(a, 2)
    forbidden_clause_b = CS.ForbiddenInClause(b, [2])

    forbidden_clause = CS.ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_b)

    cs.add_forbidden_clause(forbidden_clause)
