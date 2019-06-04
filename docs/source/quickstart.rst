Quickstart
==========

A :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
is a data structure to describe the configuration space of an algorithm to tune.
Possible hyperparameter types are numerical, categorical, conditional and ordinal hyperparameters.

AutoML tools, such as `SMAC3`_ and `BOHB`_ are using the configuration space
module to sample hyperparameter configurations.
Also, `auto-sklearn`_, an automated machine learning toolkit, which frees the
machine learning user from algorithm selection and hyperparameter tuning,
makes heavy use of the ConfigSpace package.

This simple quickstart tutorial will show you, how to set up your own
:class:`~ConfigSpace.configuration_space.ConfigurationSpace`, and will demonstrate
what you can realize with it. This tutorial will include the following steps:

- Create a :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
- Define :ref:`hyperparameters <Hyperparameters>` and their value ranges
- Add the :ref:`hyperparameters <Hyperparameters>` to the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
- (Optional) Add :ref:`Conditions` to the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
- (Optional) Add :ref:`Forbidden clauses`
- (Optional) :ref:`Serialize <Serialization>` the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`

We will show those steps in an exemplary way by creating a
:class:`~ConfigSpace.configuration_space.ConfigurationSpace` for ridge regression.
Note that the topics adding constraints, adding forbidden clauses and
serialization are explained in the :doc:`user guide <User-Guide>`.


Basic Usage
-----------

We take a look at a simple
`ridge regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_,
which has only one floating hyperparameter :math:`\alpha`.

The first step is always to create a
:class:`~ConfigSpace.configuration_space.ConfigurationSpace` object. All the
hyperparameters and constraints will be added to this object.

>>> import ConfigSpace as CS
>>> cs = CS.ConfigurationSpace(seed=1234)

The hyperparameter :math:`\alpha` is chosen to have floating point values from 0 to 1.

>>> import ConfigSpace.hyperparameters as CSH
>>> alpha = CSH.UniformFloatHyperparameter(name='alpha', lower=0, upper=1)

We add it to the :class:`~ConfigSpace.configuration_space.ConfigurationSpace` object.

>>> cs.add_hyperparameter(alpha)
alpha, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5

For demonstration purpose, we sample a configuration from the :class:`~ConfigSpace.configuration_space.ConfigurationSpace` object.

.. doctest::

    >>> cs.sample_configuration()
    Configuration:
      alpha, Value: 0.1915194503788923
    <BLANKLINE>

And that's it.

To continue reading, visit the :doc:`user guide <User-Guide>` section. There are
more information about hyperparameters, as well as an introduction to the
powerful concepts of :ref:`Conditions` and :ref:`Forbidden clauses`.

.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn
