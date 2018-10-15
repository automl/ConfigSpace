Quickstart
==========

A :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
is a data structure to describe the configuration space of an algorithm to tune.
It contains numerical, categorical hyperparameters and condtional hyperparameters.
As well as ordinal hyperparameters.

Our tools `SMAC3`_ and `BOHB`_ are using the configuration space module to sample hyperparameter configurations.
Also, `auto-sklearn`_, an automated machine learning toolkit, which frees the machine learning user from
algorithm selection and hyperparameter tuning, works with defined a ``ConfigurationSpace``.

This simple quickstart tutorial will show you, how to set up you own ``ConfigurationSpace``, and demonstrate what you can realize with it.
To accomplish this task, we need to:

- Create a ``ConfigurationSpace``
- Define hyperparameters and their value ranges
- Add the hyperparameters to the ``ConfigurationSpace``
- (Optional) Add constraints to the ``ConfigurationSpace``
- (Optional) Add :ref:`Forbidden clauses`
- (Optional) Serialize the ``ConfigurationSpace``

We will show those steps in an exemplary way, by creating a ``ConfigurationSpace`` for rigde regression.
Note that the topics adding constraints, adding forbidden clauses and serialization is explained in the :doc:`Guide`.


Basic Usage
-----------

To see the basic usage of the ConfigSpace tool, we take a look at a simple
`ridge regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_,
which has only one floating hyperparameter :math:`\alpha`.

The first step is always to create a ``ConfigurationSpace`` object. All the hyperparameters and constraints will be added to this
object.
::

   import ConfigSpace as CS
   cs = CS.ConfigurationSpace()

The hyperparameter :math:`\alpha` is choosen in this example to have values from 0 to 1. ::

   import ConfigSpace.hyperparameters as CSH
   alpha = CSH.UniformFloatHyperparameter(name='alpha', lower=0, upper=1)

We need to add it to the ``ConfigurationSpace`` object. ::

   cs.add_hyperparameter(alpha)

For demonstration purpose we want to sample a configuration from it. ::

   cs.sample_configuration()

And that's it.

To continue reading, visit the :doc:`Guide` section. There are more information about hyperparameters, as well as the powerful concepts
:ref:`Conditions` and :ref:`Forbidden clauses`.

.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn