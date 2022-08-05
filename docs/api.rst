API
+++

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
ConfigSpace contains
:func:`~ConfigSpace.api.types.float.Float`,
:func:`~ConfigSpace.api.types.int.Int`
and :func:`~ConfigSpace.api.types.categorical.Categorical` hyperparamters, each with their own customizability.

For :func:`~ConfigSpace.api.types.float.Float` and :func:`~ConfigSpace.api.types.int.Int`, you will find their
interface much the same, being able to take the same :ref:`distributions <Distributions>` while :func:`~ConfigSpace.api.types.categorical.Categorical` can take weights or be ordered.

These are all convenience functions that construct the more complex :ref:`hyperparameter classes <Advanced_Hyperparameters>` which make up the backbone of what's possible.

Example usages are shown below each.

Simple Types
------------

Float
^^^^^

.. automodule:: ConfigSpace.api.types.float

Int
^^^

.. automodule:: ConfigSpace.api.types.int

Categorical
^^^^^^^^^^^

.. automodule:: ConfigSpace.api.types.categorical


.. _Distributions:

Distributions
-------------
These can be used as part of the ``distribution`` parameter for the basic
:func:`~ConfigSpace.api.types.int.Int` and :func:`~ConfigSpace.api.types.float.Float` functions.

.. automodule:: ConfigSpace.api.distributions
    :exclude-members: Distribution

.. _Advanced_Hyperparameters:

Advanced Types
--------------
The full hyperparameters are exposed through the following API points.

Integer hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^
These can all be constructed with the simple :func:`~ConfigSpace.api.types.int` function and
passing the corresponding :ref:`distribution <Distributions>`.

.. autoclass:: ConfigSpace.hyperparameters.UniformIntegerHyperparameter

.. autoclass:: ConfigSpace.hyperparameters.NormalIntegerHyperparameter

.. autoclass:: ConfigSpace.hyperparameters.BetaIntegerHyperparameter



.. _advanced_float:
Float hyperparameters
^^^^^^^^^^^^^^^^^^^^^
These can all be constructed with the simple :func:`~ConfigSpace.api.types.float` function and
passing the corresponding :ref:`distribution <Distributions>`.

.. autoclass:: ConfigSpace.hyperparameters.UniformFloatHyperparameter

.. autoclass:: ConfigSpace.hyperparameters.NormalFloatHyperparameter

.. autoclass:: ConfigSpace.hyperparameters.BetaFloatHyperparameter



.. _advanced_categorical:
Categorical Hyperparameter
^^^^^^^^^^^^^^^^^^^^^^^^^^
This can be constructed with the simple form :func:`~ConfigSpace.api.types.categorical` and setting
``ordered=False`` which is the default.

.. autoclass:: ConfigSpace.hyperparameters.CategoricalHyperparameter


Ordinal Hyperparameter
^^^^^^^^^^^^^^^^^^^^^^
This can be constructed with the simple form :func:`~ConfigSpace.api.types.categorical` and setting
``ordered=True``.

.. autoclass:: ConfigSpace.hyperparameters.OrdinalHyperparameter

.. _Other hyperparameters:

Constant
^^^^^^^^

.. autoclass:: ConfigSpace.hyperparameters.Constant


.. _Conditions:

Conditions
==========

ConfigSpace can realize *equal*, *not equal*, *less than*, *greater than* and
*in conditions*. Conditions can be combined by using the conjunctions *and* and
*or*. To see how to use conditions, please take a look at the
:doc:`user guide <guide>`.

EqualsCondition
-------------------

.. autoclass:: ConfigSpace.conditions.EqualsCondition

.. _NotEqualsCondition:

NotEqualsCondition
------------------

.. autoclass:: ConfigSpace.conditions.NotEqualsCondition

.. _LessThanCondition:

LessThanCondition
-----------------

.. autoclass:: ConfigSpace.conditions.LessThanCondition



GreaterThanCondition
--------------------

.. autoclass:: ConfigSpace.conditions.GreaterThanCondition


InCondition
-----------

.. autoclass:: ConfigSpace.conditions.InCondition


AndConjunction
--------------

.. autoclass:: ConfigSpace.conditions.AndConjunction


OrConjunction
-------------

.. autoclass:: ConfigSpace.conditions.OrConjunction


.. _Forbidden clauses:

Forbidden Clauses
=================

ConfigSpace contains *forbidden equal* and *forbidden in clauses*.
The *ForbiddenEqualsClause* and the *ForbiddenInClause* can forbid values to be
sampled from a configuration space if a certain condition is met. The
*ForbiddenAndConjunction* can be used to combine *ForbiddenEqualsClauses* and
the *ForbiddenInClauses*.

For a further example, please take a look in the :doc:`user guide <guide>`.

ForbiddenEqualsClause
---------------------
.. autoclass:: ConfigSpace.ForbiddenEqualsClause(hyperparameter, value)


ForbiddenInClause
-----------------
.. autoclass:: ConfigSpace.ForbiddenInClause(hyperparameter, values)


ForbiddenAndConjunction
-----------------------
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

Serialization to JSON
---------------------

.. automodule:: ConfigSpace.read_and_write.json
   :members: read, write

.. _pcs_new:

6.2 Serialization with pcs-new (new format)
-------------------------------------------

.. automodule:: ConfigSpace.read_and_write.pcs_new
   :members: read, write

Serialization with pcs (old format)
-----------------------------------

.. automodule:: ConfigSpace.read_and_write.pcs
   :members: read, write

Utils
=====

Functions defined in the utils module can be helpful to
develop custom tools that create configurations from a given configuration
space or modify a given configuration space.

.. automodule:: ConfigSpace.util
    :members:
    :undoc-members:
