ConfigurationSpace [API]
========================

A configuration space organizes all hyperparameters and its conditions as well as its forbidden clauses.
Configurations can be sampled from this configuration space.

.. autoclass:: ConfigSpace.configuration_space.ConfigurationSpace

   .. method:: add_hyperparameter(hyperparameter: Hyperparameter)

        :param Hyperparameter hyperparameter: hyperparameter to add

   .. method:: add_hyperparameters(hyperparameters: list[Hyperparameter])

        :param list hyperparameters: collection of hyperparameters to add

   .. method:: add_condition(condition: ConditionComponent)

        :param ConditionComponent condition: condition to add

   .. method:: add_conditions(conditions: list[ConditionComponent])

        :param list conditions: collection of conditions to add

   .. method:: add_forbidden_clause(clause: AbstractForbiddenComponent)

        :param AbstractForbiddenComponent clause: Forbidden clause to add

   .. method:: add_forbidden_clauses(clauses: list[AbstractForbiddenComponent])

        :param list clauses: collection of forbidden clauses to add

   .. method:: add_configuration_space(prefix: str, configuration_space: 'ConfigurationSpace', delimiter: str=":", parent_hyperparameter: Hyperparameter=None)

        This function adds a configuration space to another one. The added entries will receive the prefix *prefix*.

        :param str prefix: new hyperparameters will be renamed to prefix + 'old name'
        :param ConfigurationSpace configuration_space: the configuration space which should be added
        :param str delimiter: default ':'
        :param Hyperparameter parent_hyperparamter: adds for each new hyperparameter the condition, that *parent_hyperparameter* is active
        :return: ConfigurationSpace

   .. method:: get_hyperparameters()

       :return: all hyperparameters of the configuration space

   .. method:: get_hyperparameter_names()

       :return: all names of the contained hyperparameters

   .. method:: get_hyperparameter(name: str)

       :param str name: Name of the searched hyperparameter
       :return Hyperparameter: the hyperparameter with the name *name*

   .. method:: get_hyperparameter_by_idx(idx: int))

       :param int idx: id of a hyperparameter
       :return str: name of the hyperparameter with the id *idx*

   .. method:: get_idx_by_hyperparameter_name(name: str))

       :param str name: name of a hyperparameter
       :return int: id of the hyperparameter with name *name*

   .. method:: get_conditions()

       :return: List[AbstractConditions] All conditions from the configuration space

   .. method:: get_forbiddens()

       :return: List[AbstractForbiddenComponent] All forbiddens from the configuration space

   .. method:: get_children_of(name: Union[str, Hyperparameter])

       Returns a list with the children of a given hyperparameter

       :param str,Hyperparameter name:
       :return List[Hyperparameter]:

   .. method:: get_child_conditions_of(name: Union[str, Hyperparameter])

       Returns a List with conditions of all children of a given hyperparameter

       :return List[AbstractCondition]:

   .. method:: get_parents_of(name: Union[str, Hyperparameter])

       Returns a list with the parents of a given hyperparameter

       :param str,Hyperparameter name:
       :return List[Hyperparameter]:

   .. method:: get_parent_conditions_of(name: Union[str, Hyperparameter])

       Returns a List with conditions of all parents of a given hyperparameter

       :return List[AbstractCondition]:


   .. method:: get_all_unconditional_hyperparameters()

       Returns a list with names of unconditional hyperparameters

       :return List[Hyperparamter]:

   .. method:: get_all_conditional_hyperparameters()

       Returns a list with names of all conditional hyperparameters

       :return List[Hyperparamter]:

   .. method:: get_default_configuration()

       Returns a configuration containing hyperparameters with default values

       :return Configuration:

   .. method:: check_configuration(configuration: 'Configuration')

       :param Configuration configuration:


   .. method:: check_configuration_vector_representation(np.ndarray: vector)

       :param np.ndarray vector:

   .. method:: sample_configuration(int : size=1)

       Samples *size*-times a configuration

       :param int size:
       :return Configuration,List[Configuration]:

   .. method:: seed(int: seed)

       Sets the random seed.

       :param int seed:

Configuration [API]
===================

.. autoclass:: ConfigSpace.configuration_space.Configuration

Hyperparameters [API]
=====================

In this section, the different types of hyperparameters are introduced. ConfigSpace is able to handle integer, float, categorical as well as ordinal hyperparameters.
Float and integer hyperparameters are available as **uniform** or **normal distributed** ones.
This means, that when a hyperparameter is sampled from the configuration space, its value is distributed according to the specified type.

Example usages are shown in the :doc:`quickstart <quickstart>`

1) Integer hyperparameters
--------------------------

.. py:class:: ConfigSpace.hyperparameters.UniformIntegerHyperparameter(name: str, lower: int, upper: int, default_value: Union[int, None]=None, q: Union[int, None]=None, log: bool=False) -> None:

    Creates a integer-hyperparameter with values sampled from a uniform distribution with values from *lower* to *upper*

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int lower: lower bound of a range of values from which the hyperparameter will be sampled.
    :param int upper: upper bound.
    :param int,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int q:
    :param bool log: If *bool* is true, the values of the hyperparameter will be sampled on a logarithmic scale.

**Example**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    uniform_integer_hp = CSH.UniformIntegerHyperparameter('uni_int', lower=10, upper=100, log=False)

    cs.add_hyperparameter(uniform_integer_hp)

.. py:class:: ConfigSpace.hyperparameters.NormalIntegerHyperparameter(name: str, mu: int, sigma: Union[int, float], default_value: Union[int, None]=None, q: Union[None, int]=None, log: bool=False) -> None:

    Creates a integer-hyperparameter with values sampled from a normal distribution :math:`\mathcal{N}(\mu, \sigma^2)`

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int mu: Mean of the distribution.
    :param int,float sigma: Standard deviation of the distribution.
    :param int,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int q:
    :param bool log: If *bool* is true, the values of the hyperparameter will be sampled on a logarithmic scale.

**Example**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    normal_int_hp = CSH.NormalIntegerHyperparameter('uni_int', mu=0, sigma=1, log=False)
    cs.add_hyperparameter(normal_int_hp)



2) Float hyperparameters
------------------------

.. py:class:: ConfigSpace.hyperparameters.UniformFloatHyperparameter(name: str, lower: Union[int, float], upper: Union[int, float], default_value: Union[int, float, None]=None, q: Union[int, float, None]=None, log: bool=False) -> None:

    Creates a float-hyperparameter with values sampled from a uniform distribution with values from *lower* to *upper*

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int,float lower: lower bound of a range of values from which the hyperparameter will be sampled.
    :param int,float upper: upper bound.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int,float q:
    :param bool log: If *bool* is true, the values of the hyperparameter will be sampled on a logarithmic scale.

**Example:**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    uniform_float_hp = CSH.UniformFloatHyperparameter('uni_float', lower=10, upper=100, log=False)

    cs.add_hyperparameter(uniform_float_hp)

.. py:class:: ConfigSpace.hyperparameters.NormalFloatHyperparameter(name: str, mu: Union[int, float], sigma: Union[int, float], default_value: Union[float, None]=None, q: Union[None, int, float]=None, log: bool=False) -> None:

    Creates a float-hyperparameter with values sampled from a normal distribution :math:`\mathcal{N}(\mu, \sigma^2)`

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int,float mu: Mean of the distribution.
    :param int,float sigma: Standard deviation of the distribution.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int,float q:
    :param bool log: If *bool* is true, the values of the hyperparameter will be sampled on a logarithmic scale.

**Example:**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    normal_float_hp = CSH.NormalFloatHyperparameter('normal_float', mu=0, sigma=1, log=False)

    cs.add_hyperparameter(normal_float_hp)



3) Categorical hyperparameters
------------------------------

.. py:class:: ConfigSpace.hyperparameters.CategoricalHyperparameter(name: str, choices: Union[List[Union[str, float, int]], Tuple[Union[float, int, str]]], default_value: Union[int, float, str, None]=None) -> None:

    Creates a categorical-hyperparameter

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param List[Union[str,float,int]],Tuple[Union[float,int,str]]] choices: set of values to sample hyperparameter from.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.

**Example:**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    cat_hp = CSH.CategoricalHyperparameter('cat_hp', choices=['red', 'green', 'blue'])

    cs.add_hyperparameter(cat_hp)



4) OrdinalHyperparameters
-------------------------

.. py:class:: ConfigSpace.hyperparameters.CategoricalHyperparameter(name: str, choices: Union[List[Union[str, float, int]], Tuple[Union[float, int, str]]], default_value: Union[int, float, str, None]=None) -> None:

    Creates a ordinal-hyperparameter

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param List[Union[str,float,int]],Tuple[Union[float,int,str]]] sequence: set of values to sample hyperparameter from.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.

**Example:**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    ord_hp = CSH.OrdinalHyperparameter('ordinal_hp', sequence=['10', '20', '30'])

    cs.add_hyperparameter(ord_hp)



Conditions [API]
================

| It is often necessary to make some constraint on hyperparameters. For example we have one hyperparameter, which decides whether
| we use methodA or methodB, and each method has additional unique hyperparameter.
| Those depending hyperparameters are called *child* hyperparameter.
| It is desired, that they should only be "active", if their *parent* hyperparameter
| has the right value.
| This can be accomplished, using **conditions**.

To see an example of how to use conditions, please take a look at the :doc:`advanced example <AdvancedExample>`

1) EqualsCondition
------------------

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
---------------------

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
--------------------

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
-----------------------

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
--------------

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
-----------------

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
----------------


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
=================

In addition to the conditions, it's also possible to add forbidden clauses to the configuration space.
They allow us to make some more restrictions to the configuration space.

The forbidden clauses are also captured in the examples. Please take a look at the :doc:`advanced example <AdvancedExample>`

1) ForbiddenEqualsClause
------------------------
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
--------------------
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
--------------------------
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

Serialization
=============

| Sometimes, it can be useful to serialize the *configuration space*.
| This can be achieved by using the classes **ConfigSpace.read_and_write.pcs**,
  **ConfigSpace.read_and_write.pcs_new** or **ConfigSpace.read_and_write.json**.

1) Serialization to JSON
------------------------

.. automodule:: ConfigSpace.read_and_write.json
   :members:
   :undoc-members:

This example shows how to write and read a configuration space to *json* - file::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    from ConfigSpace.read_and_write import json

    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))

    # Store the configuration space to file as json-file
    with open('configspace.json', 'w') as fh:
        fh.write(json.write(cs))

    # Read the configuration space from file
    with open('configspace.json', 'r') as fh:
        json_string = fh.read()
        restored_conf = json.read(json_string)

2) Serialization with pcs-new
-----------------------------

.. automodule:: ConfigSpace.read_and_write.pcs_new
   :members: read, write
   :undoc-members:


To write to pcs is similar to the example above.::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    from ConfigSpace.read_and_write import pcs_new

    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))

    # Store the configuration space to file configspace as pcs-file
    with open('configspace.pcs', 'w') as fh:
        fh.write(pcs_new.write(cs))

    # Read the configuration space from file
    with open('configspace.pcs', 'r') as fh:
        restored_conf = pcs_new.read(fh)


3) Serialization with pcs
-------------------------

.. automodule:: ConfigSpace.read_and_write.pcs
   :members: read, write
   :undoc-members:

::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    from ConfigSpace.read_and_write import pcs

    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))

    # Store the configuration space to file configspace as pcs-file
    with open('configspace.pcs', 'w') as fh:
        fh.write(pcs.write(cs))

    # Read the configuration space from file
    with open('configspace.pcs', 'r') as fh:
        restored_conf = pcs.read(fh)



