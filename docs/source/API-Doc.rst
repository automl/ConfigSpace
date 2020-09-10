API-Documentation
+++++++++++++++++

ConfigurationSpace
==================

.. autoclass:: ConfigSpace.configuration_space.ConfigurationSpace
    :members:

Configuration
=============

.. autoclass:: ConfigSpace.configuration_space.Configuration
    :members:

.. _Hyperparameters:

Hyperparameters
===============

ConfigSpace contains integer, float, categorical, as well as ordinal
hyperparameters. Integer and float hyperparameter can be sampled from a uniform
or normal distribution. Example usages are shown in the
:doc:`quickstart <quickstart>`.

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

.. _Other hyperparameters:

3.5 Constant
------------

.. autoclass:: ConfigSpace.hyperparameters.Constant


.. _Conditions:

Conditions
==========

ConfigSpace can realize *equal*, *not equal*, *less than*, *greater than* and
*in conditions*. Conditions can be combined by using the conjunctions *and* and
*or*. To see how to use conditions, please take a look at the
:doc:`user guide <User-Guide>`.

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

ConfigSpace contains *forbidden equal* and *forbidden in clauses*.
The *ForbiddenEqualsClause* and the *ForbiddenInClause* can forbid values to be
sampled from a configuration space if a certain condition is met. The
*ForbiddenAndConjunction* can be used to combine *ForbiddenEqualsClauses* and
the *ForbiddenInClauses*.

For a further example, please take a look in the :doc:`user guide <User-Guide>`.

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

ConfigSpace offers *json*, *pcs* and *pcs_new* writers/readers.
These classes can serialize and deserialize configuration spaces.
Serializing configuration spaces is useful to share configuration spaces across
experiments, or use them in other tools, for example, to analyze hyperparameter
importance with `CAVE <https://github.com/automl/CAVE>`_.

.. _json:

6.1 Serialization to JSON
-------------------------

.. automodule:: ConfigSpace.read_and_write.json
   :members: read, write
   :undoc-members:

.. _pcs_new:

6.2 Serialization with pcs-new
------------------------------

Pcs is a simple, human-readable file format for the description of an
algorithm's configurable parameters, their possible values, as well as any
parameter dependencies.

Pcs is part of the `Algorithm Configuration Library <http://aclib.net/#>`_.
A detailed explanation of the pcs format can be found
`here. <http://aclib.net/cssc2014/pcs-format.pdf>`_ A short summary is also
given in the
`SMAC Documentation <https://automl.github.io/SMAC3/dev/options.html#paramcs>`_.
Further examples are provided in the
`pysmac documentation <https://pysmac.readthedocs.io/en/latest/pcs.html>`_

.. note::

    The pcs format definition has changed in the year 2016 and is supported by
    AClib 2.0, as well as SMAC. To write or to read the old version of pcs, please
    use the :class:`~ConfigSpace.read_and_write.pcs` module.

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

Functions defined in the utils module can be helpful to
develop custom tools that create configurations from a given configuration
space or modify a given configuration space.

.. automodule:: ConfigSpace.util
    :members:
    :undoc-members:
