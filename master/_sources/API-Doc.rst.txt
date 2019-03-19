API-Documentation
+++++++++++++++++

ConfigurationSpace
==================

First, we start by introducing the main object, the *configuration space* and
all of its functions.

.. autoclass:: ConfigSpace.configuration_space.ConfigurationSpace
   :members:


Configuration
=============

.. autoclass:: ConfigSpace.configuration_space.Configuration
    :members:

.. _Hyperparameters:

Hyperparameters
===============

In this section, the different types of hyperparameters are introduced.
ConfigSpace is able to handle integer, float, categorical as well as ordinal hyperparameters.
Float and integer hyperparameters are available as **uniform** or **normal distributed** ones.
So when a hyperparameter is sampled from the configuration space,
its value is distributed according to the specified type.

Example usages are shown in the :doc:`quickstart <quickstart>`

3.1 Integer hyperparameters
---------------------------

.. autoclass:: ConfigSpace.hyperparameters.UniformIntegerHyperparameter


.. autoclass:: ConfigSpace.hyperparameters.NormalIntegerHyperparameter



3.2 Float hyperparameters
-------------------------

.. autoclass:: ConfigSpace.hyperparameters.UniformFloatHyperparameter


.. autoclass:: ConfigSpace.hyperparameters.NormalFloatHyperparameter



.. _Categorical hyperparameters:

3.3 Categorical hyperparameters
-------------------------------

.. autoclass:: ConfigSpace.hyperparameters.CategoricalHyperparameter


3.4 OrdinalHyperparameters
--------------------------

.. autoclass:: ConfigSpace.hyperparameters.OrdinalHyperparameter


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

.. autoclass:: ConfigSpace.conditions.EqualsCondition

.. _NotEqualsCondition:

4.2 NotEqualsCondition
----------------------

.. autoclass:: ConfigSpace.conditions.NotEqualsCondition

.. _LessThanCondition:

4.3 LessThanCondition
---------------------

.. autoclass:: ConfigSpace.conditions.LessThanCondition



4.4 GreaterThanCondition
------------------------

.. autoclass:: ConfigSpace.conditions.GreaterThanCondition


4.5 InCondition
---------------

.. autoclass:: ConfigSpace.conditions.InCondition


4.6 AndConjunction
------------------

.. autoclass:: ConfigSpace.conditions.AndConjunction


4.7 OrConjunction
-----------------

.. autoclass:: ConfigSpace.conditions.OrConjunction


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
.. autoclass:: ConfigSpace.ForbiddenEqualsClause(hyperparameter, value)


5.2 ForbiddenInClause
---------------------
.. autoclass:: ConfigSpace.ForbiddenInClause(hyperparameter, values)


5.3 ForbiddenAndConjunction
---------------------------
.. autoclass:: ConfigSpace.ForbiddenAndConjunction(*args)


.. _Serialization:

Serialization
=============

| ConfigSpace offers also the functionality to serialize the configuration space.
  This can be useful to share configuration spaces across experiments,
  or use them in other tools, for example to analyze hyperparameter importance with
  `CAVE <https://github.com/automl/CAVE>`_,
| This can be achieved by using the classes **ConfigSpace.read_and_write.pcs**,
  **ConfigSpace.read_and_write.pcs_new** or **ConfigSpace.read_and_write.json**.

.. _json:

6.1 Serialization to JSON
-------------------------

.. automodule:: ConfigSpace.read_and_write.json
   :members: read, write
   :undoc-members:

.. _pcs_new:

6.2 Serialization with pcs-new
------------------------------

Pcs is a simple, human-readable file format for the description of an algorithm's configurable parameters, their possible values, as well as any
parameter dependencies.

It is part of the `Algorithm Configuration Library <http://aclib.net/#>`_. A detailed explanation of the pcs format can be
found `here. <http://aclib.net/cssc2014/pcs-format.pdf>`_ or a short summary is also given in the
`SMAC Documentation <https://automl.github.io/SMAC3/dev/options.html#paramcs>`_.
Examples are also provieded in the `pysmac documentation <https://pysmac.readthedocs.io/en/latest/pcs.html>`_

.. note::

    The pcs format definition has changed in the year 2016 and is supported by AClib 2.0 as well as SMAC.
    To write or read the old version of pcs, please use the :class:`~ConfigSpace.read_and_write.pcs` module.

.. automodule:: ConfigSpace.read_and_write.pcs_new
   :members: read, write
   :undoc-members:

6.3 Serialization with pcs
--------------------------

.. automodule:: ConfigSpace.read_and_write.pcs
   :members: read, write
   :undoc-members:


Utils
=====

Until now, all examples were designed to define a configuration space.
However, there is also the application that you want to develop your own tool that creates configurations from
a given configuration space. Or that you want to modify a given configuration space.

The functionalities from the file util.py can be very useful.

.. automodule:: ConfigSpace.util
    :members:
    :undoc-members:
