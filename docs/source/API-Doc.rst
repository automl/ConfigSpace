API-Documentation
+++++++++++++++++

ConfigurationSpace
==================

A configuration space organizes all hyperparameters and its conditions as well as its forbidden clauses.
Configurations can be sampled from this configuration space.

.. autoclass:: ConfigSpace.configuration_space.ConfigurationSpace

   .. method:: add_hyperparameter(hyperparameter)

        :param Hyperparameter hyperparameter: hyperparameter to add

   .. method:: add_hyperparameters(hyperparameters)

        :param list[Hyperparameter] hyperparameters: collection of hyperparameters to add

   .. method:: add_condition(condition)

        :param ConditionComponent condition: condition to add

   .. method:: add_conditions(conditions)

        :param list[ConditionComponent] conditions: collection of conditions to add

   .. method:: add_forbidden_clause(clause)

        :param AbstractForbiddenComponent clause: Forbidden clause to add

   .. method:: add_forbidden_clauses(clauses)

        :param list[AbstractForbiddenComponent] clauses: collection of forbidden clauses to add

   .. method:: add_configuration_space(prefix, configuration_space, delimiter=":", parent_hyperparameter=None)

        This function adds a configuration space to another one. The added entries will receive the prefix ``prefix``.

        :param str prefix: new hyperparameters will be renamed to ``prefix`` + 'old name'
        :param ConfigurationSpace configuration_space: the configuration space which should be added
        :param str(optional) delimiter: default ':'
        :param Hyperparameter(optional) parent_hyperparamter: adds for each new hyperparameter the condition, that ``parent_hyperparameter`` is active
        :return: ConfigurationSpace

   .. method:: get_hyperparameters()

       :return: list[Hyperparameter] -- list with all the hyperparameter, which were added to the configurationsspace-object earlier.

   .. method:: get_hyperparameter_names()

       :return: list[str] -- list with names of all hyperparameter, which were added to the configurationsspace-object earlier.

   .. method:: get_hyperparameter(name)

       :param str name: Name of the searched hyperparameter
       :return: Hyperparameter -- the hyperparameter with the name ``name``

   .. method:: get_hyperparameter_by_idx(idx))

       :param int idx: id of a hyperparameter
       :return: str -- name of the hyperparameter with the id ``idx``

   .. method:: get_idx_by_hyperparameter_name(name))

       :param str name: name of a hyperparameter
       :return:  int -- id of the hyperparameter with name ``name``

   .. method:: get_conditions()

       :return: list[AbstractConditions] -- All conditions from the configuration space

   .. method:: get_forbiddens()

       :return: list[AbstractForbiddenComponent] -- All forbiddens from the configuration space

   .. method:: get_children_of(name)

       Returns a list with the children of a given hyperparameter

       :param str,Hyperparameter name:
       :return: list[Hyperparameter]

   .. method:: get_child_conditions_of(name)

       Returns a List with conditions of all children of a given hyperparameter

       :param str,Hyperparameter name:
       :return: list[AbstractCondition]

   .. method:: get_parents_of(name)

       Returns a list with the parents of a given hyperparameter

       :param str,Hyperparameter name:
       :return: list[Hyperparameter]

   .. method:: get_parent_conditions_of(name)

       Returns a List with conditions of all parents of a given hyperparameter

       :param str,Hyperparameter name:
       :return: list[AbstractCondition]


   .. method:: get_all_unconditional_hyperparameters()

       Returns a list with names of unconditional hyperparameters

       :return: list[Hyperparamter]

   .. method:: get_all_conditional_hyperparameters()

       Returns a list with names of all conditional hyperparameters

       :return: list[Hyperparamter]

   .. method:: get_default_configuration()

       Returns a configuration containing hyperparameters with default values

       :return: Configuration

   .. method:: check_configuration(configuration)

       :param Configuration configuration:


   .. method:: check_configuration_vector_representation(vector)

       :param np.ndarray vector:

   .. method:: sample_configuration(size=1)

       sample ``size`` configurations

       :param int size:
       :return: Configuration,list[Configuration]

   .. method:: seed(seed)

       Sets the random seed.

       :param int seed:

Configuration
=============

.. autoclass:: ConfigSpace.configuration_space.Configuration

.. _Hyperparameters:

Hyperparameters
===============

In this section, the different types of hyperparameters are introduced. ConfigSpace is able to handle integer, float, categorical as well as ordinal hyperparameters.
Float and integer hyperparameters are available as **uniform** or **normal distributed** ones.
So when a hyperparameter is sampled from the configuration space, its value is distributed according to the specified type.

Example usages are shown in the :doc:`quickstart <quickstart>`

3.1 Integer hyperparameters
---------------------------

.. py:class:: ConfigSpace.hyperparameters.UniformIntegerHyperparameter(name, lower, upper, default_value=None, q=None, log=False) -> None:

    Creates an integer hyperparameter with values sampled from a uniform distribution with values from ``lower`` to ``upper``

    :param str name: Name of the hyperparameter with which it can be accessed.
    :param int lower: lower bound of a range of values from which the hyperparameter will be sampled.
    :param int upper: upper bound.
    :param int,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int q:
    :param bool log: If ``True``, the values of the hyperparameter will be sampled on a logarithmic scale.

**Example**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    uniform_integer_hp = CSH.UniformIntegerHyperparameter('uni_int', lower=10, upper=100, log=False)

    cs.add_hyperparameter(uniform_integer_hp)

.. py:class:: ConfigSpace.hyperparameters.NormalIntegerHyperparameter(name, mu, sigma, default_value=None, q=None, log=False) -> None:

    Creates an integer hyperparameter with values sampled from a normal distribution :math:`\mathcal{N}(\mu, \sigma^2)`

    :param str name: Name of the hyperparameter with which it can be accessed.
    :param int mu: Mean of the distribution.
    :param int,float sigma: Standard deviation of the distribution.
    :param int,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int q:
    :param bool log: If ``True``, the values of the hyperparameter will be sampled on a logarithmic scale.

**Example**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    normal_int_hp = CSH.NormalIntegerHyperparameter('uni_int', mu=0, sigma=1, log=False)
    cs.add_hyperparameter(normal_int_hp)



3.2 Float hyperparameters
-------------------------

.. py:class:: ConfigSpace.hyperparameters.UniformFloatHyperparameter(name, lower, upper, default_value=None, q=None, log=False) -> None:

    Creates a float hyperparameter with values sampled from a uniform distribution with values from ``lower`` to ``upper``

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int,float lower: lower bound of a range of values from which the hyperparameter will be sampled.
    :param int,float upper: upper bound.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int,float q:
    :param bool log: If ``True``, the values of the hyperparameter will be sampled on a logarithmic scale.

**Example:**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    uniform_float_hp = CSH.UniformFloatHyperparameter('uni_float', lower=10, upper=100, log=False)

    cs.add_hyperparameter(uniform_float_hp)

.. py:class:: ConfigSpace.hyperparameters.NormalFloatHyperparameter(name, mu, sigma, default_value=None, q=None, log=False) -> None:

    Creates a float hyperparameter with values sampled from a normal distribution :math:`\mathcal{N}(\mu, \sigma^2)`

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int,float mu: Mean of the distribution.
    :param int,float sigma: Standard deviation of the distribution.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int,float q:
    :param bool log: If ``True``, the values of the hyperparameter will be sampled on a logarithmic scale.

**Example:**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    normal_float_hp = CSH.NormalFloatHyperparameter('normal_float', mu=0, sigma=1, log=False)

    cs.add_hyperparameter(normal_float_hp)


.. _Categorical hyperparameters:

3.3 Categorical hyperparameters
-------------------------------

.. py:class:: ConfigSpace.hyperparameters.CategoricalHyperparameter(name, choices, default_value=None) -> None:

    Creates a categorical hyperparameter

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param list[Union[str,float,int]],Tuple[Union[float,int,str]]] choices: set of values to sample hyperparameter from.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.

**Example:**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    cat_hp = CSH.CategoricalHyperparameter('cat_hp', choices=['red', 'green', 'blue'])

    cs.add_hyperparameter(cat_hp)



3.4 OrdinalHyperparameters
--------------------------

.. py:class:: ConfigSpace.hyperparameters.CategoricalHyperparameter(name, choices, default_value=None) -> None:

    Creates an ordinal hyperparameter

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param list[Union[str,float,int]],Tuple[Union[float,int,str]]] sequence: set of values to sample hyperparameter from.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.

**Example:**::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    ord_hp = CSH.OrdinalHyperparameter('ordinal_hp', sequence=['10', '20', '30'])

    cs.add_hyperparameter(ord_hp)


.. _Conditions:

Conditions
==========

It is often necessary to make some constraint on hyperparameters. For example we have one hyperparameter, which decides whether
we use method b or method b, and each method has additional unique hyperparameter.
Those depending hyperparameters are called *child* hyperparameter.
It is desired, that they should only be "active", if their *parent* hyperparameter
has the right value.
This can be accomplished, using **conditions**.

To see an example of how to use conditions, please take a look at the :doc:`Guide`

4.1 EqualsCondition
-------------------

.. py:class:: ConfigSpace.conditions.EqualsCondition(child, parent, value) -> None:

    Adds on the ``child`` hyperparameter the condition, that the ``parent`` hyperparameter has to be equal to ``value``

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the ``equal condition`` is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the ``equal condition``
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

4.2 NotEqualsCondition
----------------------

.. py:class:: ConfigSpace.conditions.NotEqualsCondition(child, parent, value) -> None:

    Adds on the ``child`` hyperparameter the condition, that the ``parent`` hyperparameter's value is not equal to ``value``

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the not-equals condition is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the ``not equal condition``
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

4.3 LessThanCondition
---------------------

.. py:class:: ConfigSpace.conditions.LessThanCondition(child, parent, value) -> None:

    Adds on the ``child`` hyperparameter the condition, that the ``parent`` hyperparameter's value has to be less than ``value``

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the ``LessThanCondition`` is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the ``LessThanCondition``
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


4.4 GreaterThanCondition
------------------------

.. py:class:: ConfigSpace.conditions.GreaterThanCondition(child, parent, value) -> None:

    Adds on the ``child`` hyperparameter the condition, that the ``parent`` hyperparameter's value has to be greater than ``value``

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the ``GreaterThanCondition`` is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the ``GreaterThanCondition``
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

4.5 InCondition
---------------

.. py:class:: ConfigSpace.conditions.InCondition(child, parent, values) -> None:

    Adds on the ``child`` hyperparameter the condition, that the ``parent`` hyperparameter's value has to be in the set ``values``

    :param Hyperparameter child: This hyperparameter will be sampled in the configspace, if the ``InCondition`` is satisfied
    :param Hyperparameter parent: The hyperparameter, which has to satisfy the ``InCondition``
    :param list[str,float,int] value: subset of values, which the parent is compared to

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


4.6 AndConjunction
------------------

By using the and conjunction, we can easily connect constraints.

.. py:class:: ConfigSpace.conditions.AndConjunction(*args) -> None:

    :param AbstractCondition args: conditions, which will be combined with an ``AndConjunction``

The following example shows how we can combine two constraints with an ``AndConjunction``.
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

4.7 OrConjunction
-----------------


Similar to the ``AndConjunction``, new constraints can be combined by using the ``OrConjunction``.

.. py:class:: ConfigSpace.conditions.OrConjunction(*args) -> None:

    :param AbstractCondition args: conditions, which will be combined with an ``OrConjunction``

The following example shows how we can combine two constraints with an ``OrConjunction``.
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

.. _Forbidden clauses:

Forbidden Clauses
=================

In addition to the conditions, it's also possible to add forbidden clauses to the configuration space.
They allow us to make some more restrictions to the configuration space.

In the following example the usage and utility of forbidden clauses is shown.
It describes the configuration space for the linear support vector machine implementation from auto-sklearn. The full code can be found
`here <https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/liblinear_svc.py>`_.::

    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        penalty = cs.add_hyperparameter(CategoricalHyperparameter(
            "penalty", ["l1", "l2"], default="l2"))
        loss = cs.add_hyperparameter(CategoricalHyperparameter(
            "loss", ["hinge", "squared_hinge"], default="squared_hinge"))
        dual = cs.add_hyperparameter(Constant("dual", "False"))
        # This is set ad-hoc
        tol = cs.add_hyperparameter(UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default=1e-4, log=True))
        C = cs.add_hyperparameter(UniformFloatHyperparameter(
            "C", 0.03125, 32768, log=True, default=1.0))
        multi_class = cs.add_hyperparameter(Constant("multi_class", "ovr"))
        # These are set ad-hoc
        fit_intercept = cs.add_hyperparameter(Constant("fit_intercept", "True"))
        intercept_scaling = cs.add_hyperparameter(Constant(
            "intercept_scaling", 1))

        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"),
            ForbiddenEqualsClause(loss, "hinge")
        )
        constant_penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(penalty, "l2"),
            ForbiddenEqualsClause(loss, "hinge")
        )
        penalty_and_dual = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(penalty, "l1")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        cs.add_forbidden_clause(constant_penalty_and_loss)
        cs.add_forbidden_clause(penalty_and_dual)
        return cs


For a further example, please take a look in the :doc:`guide <Guide>`

5.1 ForbiddenEqualsClause
-------------------------
.. py:class:: ConfigSpace.ForbiddenEqualsClause(hyperparameter, value) -> None:

    :param Hyperparameter hyperparameter: hyperparameter on which a restriction will be made
    :param Any value: forbidden value

Example::

    cs = CS.ConfigurationSpace()
    a = CSH.CategoricalHyperparameter('a', [1,2,3])
    cs.add_hyperparameters([a])

    # It forbids the value 2 for the hyperparameter a
    forbidden_clause_a = CS.ForbiddenEqualsClause(a, 2)
    cs.add_forbidden_clause(forbidden_clause_a)

5.2 ForbiddenInClause
---------------------
.. py:class:: ConfigSpace.ForbiddenInClause(hyperparameter, values) -> None:

    :param Hyperparameter hyperparameter: hyperparameter on which a restriction will be made
    :param Any values: forbidden values

.. note::

    The forbidden values have to be a subset of the hyperparameter's values.

Example::

    cs = CS.ConfigurationSpace()
    a = CSH.CategoricalHyperparameter('a', [1,2,3])
    cs.add_hyperparameters([a])

    # It forbids the values 2, 3, 4 for the hyperparameter 'a'
    forbidden_clause_a = CS.ForbiddenInClause(a, [2, 3])

    cs.add_forbidden_clause(forbidden_clause_a)

5.3 ForbiddenAndConjunction
---------------------------
.. py:class:: ConfigSpace.ForbiddenAndConjunction(*args) -> None:

    The *ForbiddenAndConjunction* combines forbidden-clauses, which allows to build powerful constraints.

    :param AbstractForbiddenComponent : forbidden clauses, which should be combined

Example::

    cs = CS.ConfigurationSpace()
    a = CSH.CategoricalHyperparameter('a', [1,2,3])
    b = CSH.CategoricalHyperparameter('b', [2,5,6])
    cs.add_hyperparameters([a, b])

    forbidden_clause_a = CS.ForbiddenEqualsClause(a, 2)
    forbidden_clause_b = CS.ForbiddenInClause(b, [2])

    forbidden_clause = CS.ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_b)

    cs.add_forbidden_clause(forbidden_clause)


.. _Serialization:

Serialization
=============

| ConfigSpace offers also the functionality to serialize the configuration space.
  This can be useful to share configuration spaces across experiments,
  or use them in other tools, for example to analyze hyperparameter importance with `CAVE <https://github.com/automl/CAVE>`_,
| This can be achieved by using the classes **ConfigSpace.read_and_write.pcs**,
  **ConfigSpace.read_and_write.pcs_new** or **ConfigSpace.read_and_write.json**.

.. _json:

6.1 Serialization to JSON
-------------------------

.. automodule:: ConfigSpace.read_and_write.json
   :members:
   :undoc-members:

This example shows how to write and read a configuration space in *json* file format::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH


    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))

    # Store the configuration space to file as json-file
    with open('configspace.json', 'w') as fh:
        fh.write(json.write(cs))

    # Read the configuration space from file
    with open('configspace.json', 'r') as fh:
        json_string = fh.read()
        restored_conf = json.read(json_string)

.. _pcs_new:

6.2 Serialization with pcs-new
------------------------------

Pcs is a simple, human-readable file format for the description of an algorithm's configurable parameters, their possible values, as well as any
parameter dependencies.

It is part of the `Algorithm Configuration Library <http://aclib.net/#>`_. A detailed explanation of the pcs format can be
found `here. <http://aclib.net/cssc2014/pcs-format.pdf>`_ or a short summary is also given in the
`SMAC Documentation <https://automl.github.io/SMAC3/dev/options.html#paramcs>`_

.. note::

    The pcs format definition has changed in the year 2016 and is supported by AClib 2.0 as well as SMAC.
    To write or read the old version of pcs, please use the ``ConfigSpace.read_and_write.pcs`` (see below).

.. automodule:: ConfigSpace.read_and_write.pcs_new
   :members: read, write
   :undoc-members:

To write to a pcs file is similar to the example above.::

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


6.3 Serialization with pcs
--------------------------

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


Utils
=====

Until now, all examples were designed to define a configuration space.
However, there is also the application that you want to develop your own tool that creates configurations from
a given configuration space. Or that you want to modify a given configuration space.

The functionalities from the file util.py can be very useful.

.. automodule:: ConfigSpace.util
    :members:
    :undoc-members:
