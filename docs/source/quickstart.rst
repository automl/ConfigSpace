Quickstart
==========

A :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
is a data structure to describe the configuration space of an algorithm to tune.
Possible hyperparameter types are numerical, categorical, conditional and ordinal hyperparameters.

Our tools `SMAC3`_ and `BOHB`_ are using the configuration space module to sample hyperparameter configurations.
Also, `auto-sklearn`_, an automated machine learning toolkit, which frees the machine learning user from
algorithm selection and hyperparameter tuning, works with defined a :class:`~ConfigSpace.configuration_space.ConfigurationSpace`.

This simple quickstart tutorial will show you, how to set up you own :class:`~ConfigSpace.configuration_space.ConfigurationSpace`, and demonstrate what you can realize with it.
To accomplish this task, we need to:

- Create a :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
- Define :ref:`hyperparameters <Hyperparameters>` and their value ranges
- Add the :ref:`hyperparameters <Hyperparameters>` to the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
- (Optional) Add :ref:`Conditions` to the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
- (Optional) Add :ref:`Forbidden clauses`
- (Optional) :ref:`Serialize <Serialization>` the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`

We will show those steps in an exemplary way, by creating a :class:`~ConfigSpace.configuration_space.ConfigurationSpace` for rigde regression.
Note that the topics adding constraints, adding forbidden clauses and serialization is explained in the :doc:`Guide`.


Basic Usage
-----------

To see the basic usage of the ConfigSpace tool, we take a look at a simple
`ridge regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_,
which has only one floating hyperparameter :math:`\alpha`.

The first step is always to create a :class:`~ConfigSpace.configuration_space.ConfigurationSpace` object. All the hyperparameters and constraints will be added to this
object.

>>> import ConfigSpace as CS
>>> cs = CS.ConfigurationSpace(seed=1234)

The hyperparameter :math:`\alpha` is choosen to have floating point values from 0 to 1.

>>> import ConfigSpace.hyperparameters as CSH
>>> alpha = CSH.UniformFloatHyperparameter(name='alpha', lower=0, upper=1)

We need to add it to the :class:`~ConfigSpace.configuration_space.ConfigurationSpace` object.

>>> cs.add_hyperparameter(alpha)
alpha, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5

For demonstration purpose we want to sample a configuration from it.

.. doctest::

    >>> cs.sample_configuration()
    Configuration:
      alpha, Value: 0.1915194503788923
    <BLANKLINE>

And that's it.

To continue reading, visit the :doc:`Guide` section. There are more information about hyperparameters, as well as the powerful concepts
:ref:`Conditions` and :ref:`Forbidden clauses`.

.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn
