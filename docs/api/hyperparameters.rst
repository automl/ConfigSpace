.. _Hyperparameters:

Hyperparameters
===============
ConfigSpace contains
:func:`~ConfigSpace.api.types.float.Float`,
:func:`~ConfigSpace.api.types.integer.Integer`
and :func:`~ConfigSpace.api.types.categorical.Categorical` hyperparameters, each with their own customizability.

For :func:`~ConfigSpace.api.types.float.Float` and :func:`~ConfigSpace.api.types.integer.Integer`, you will find their
interface much the same, being able to take the same :ref:`distributions <Distributions>` and parameters.

A :func:`~ConfigSpace.api.types.categorical.Categorical` can optionally take weights to define
your own custom distribution over the discrete **un-ordered** choices.
One can also pass ``ordered=True`` to make it an :class:`~ConfigSpace.hyperparameters.OrdinalHyperparameter`.

These are all **convenience** functions that construct the more complex :ref:`hyperparameter classes <Advanced_Hyperparameters>`, *e.g.* :class:`~ConfigSpace.hyperparameters.UniformIntegerHyperparameter`,
which are the underlying complex types which make up the backbone of what's possible.
You may still use these complex classes without any functional difference.

.. note::

   The Simple types, `Integer`, `Float` and `Categorical` are just simple functions that construct the more complex underlying types.

Example usages are shown below each.

Simple Types
------------

Float
^^^^^

.. automodule:: ConfigSpace.api.types.float

Integer
^^^^^^^

.. automodule:: ConfigSpace.api.types.integer

Categorical
^^^^^^^^^^^

.. automodule:: ConfigSpace.api.types.categorical


.. _Distributions:

Distributions
-------------
These can be used as part of the ``distribution`` parameter for the basic
:func:`~ConfigSpace.api.types.integer.Integer` and :func:`~ConfigSpace.api.types.float.Float` functions.

.. automodule:: ConfigSpace.api.distributions
    :exclude-members: Distribution

.. _Advanced_Hyperparameters:

Advanced Types
--------------
The full hyperparameters are exposed through the following API points.


Integer hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^

These can all be constructed with the simple :func:`~ConfigSpace.api.types.integer.Integer` function and
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
