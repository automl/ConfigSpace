Quickstart
==========

A ConfigSpace is equal to a container holding all different kinds of hyperparameters.
For example numerical, categorical hyperparameters and condtional hyperparameters.
As well as ordinal hyperparameters.

Tools like `SMAC3`_, `BOHB`_ or `auto-sklearn`_ can sample hyperparameter-configurations from this ConfigSpace.

This simple quickstart tutorial will show you, how to set up you own ConfigSpace, and what you can realize with it.
To accomplish this task, we need to:

- Create a ConfigSpace
- Define hyperparameters and their value ranges
- Add the hyperparameters to the ConfigSpace
- (Optional) Add constraints to the ConfigSpace

We will show those steps in an exemplary way, by creating a ConfigSpace for a soft-margin SVM classifier and a neural network.

1. Example: Basic Usage and Conditional Hyperparameter
------------------------------------------------------

Assume, we want to train a soft-margin SVM classifier with a RBF-kernel. This classifier has at least two hyperparametrs.
A regularization constant :math:`\mathcal{C}` and a kernel hyperparameter  :math:`\gamma` with

- :math:`\mathcal{C} \in` {10, 100, 1000}
- :math:`\gamma \in` {0.1, 0.2, 0.5, 1.0}

| How this classifier is implemented is not necessary in this example and thus not shown.

The first step is always to create a ConfigSpace-object. All the hyperparameters and constraints will be added to this
object.
::

   import ConfigSpace as CS
   cs = CS.ConfigurationSpace()

Now, we have to define the hyperparameters :math:`\mathcal{C}` and :math:`\gamma`. Since each hyperparameter only can
take on values from a given subset, we use categorical-hyperparameters in this case.
:: 

   import ConfigSpace.hyperparameters as CSH
   c = CSH.CategoricalHyperparameter(name='C', choices=[10, 100, 1000])
   gamma = CSH.CategoricalHyperparameter(name='gamma', choices=[0.1, 0.2, 0.5, 1.0])

As last step in this first example, we only need to add them to the ConfigSpace::

   cs.add_hyperparameter(c)
   cs.add_hyperparameter(gamma)

And that's it.
The ConfigSpace *cs* stores the hyperparameters :math:`\mathcal{C}` and :math:`\gamma` with their defined value-ranges.

2. Example: IntegerHyperparameters and FloatHyperparameters:
------------------------------------------------------------

| As already mentioned, ConfigSpace is also able to distinguish between some other common types like integer- and float-hyperparameter.
| In the next example, we will create a ConfigSpace for a simple neural network.
| Assume this neural network model has a float-hyperparameter *learning_rate* and a integer-hyperparameter
  *number of units in the hidden layer*.
| Since the learning rate should be sampled from a value range from 0.000001 to 0.1, we choose this hyperparameter to be
  a float-hyperparameter.
| Note, that the parameter *log* is set to True: This causes that the values of this hyperparameter
  is sampled from a logarithmic scale.
  In hyperparameter-optimization this is sometimes useful, because a logarithmic scale enables us to search a bigger space quickly.
  It could be also preferred, if a hyperparameter is not very sensitive. In this example here, it makes more sense to compare result from runs with
  learning rates on a logarithmic scale with values 0.000001, 0.00001,..., 0.1 instead of 0.000001, 0.000002, ... .

::

   lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
   cs.add_hyperparameter(lr)
   
   num_hidden_units = CSH.UniformIntegerHyperparameter('num_hidden_units', lower=10, upper=150, default_value=100)
   cs.add_hyperparameter(num_hidden_units)
 
3. Example: ConditionalHyperparameters:
---------------------------------------

Sometimes it is necessary to integrate conditions in our ConfigSpace.
We extend the neural network example from above by another hyperparameter, the *number of hidden layers*. Since
each layer should have a independent *number of units*, we have to add for each layer a own hyperparameter: *number of hidden units 1*
and *number of hidden units 2*.

+--------------------------+---------------+----------+---------------------------+
| Parameter                | Type          | values   |  condition                |
+==========================+===============+==========+===========================+
| num hidden layers        | uniform int   | 1  - 2   |  None                     |
+--------------------------+---------------+----------+---------------------------+
| num hidden units layer 1 | uniform int   | 64 - 128 |  None                     |
+--------------------------+---------------+----------+---------------------------+
| num hidden units layer 2 | uniform int   | 64 - 128 |  num hidden layers == 2   |
+--------------------------+---------------+----------+---------------------------+

| So we need to create a condition, that a value for *num hidden units layer 2* is sampled only if we have two hidden layers.
| Or intuitively spoken, the hyperparameter *num hidden units layer 2* should only be *active*, if we have two hidden layers.
| ConfigSpace is capable of handling this situation.

First, add the new hyperparameters to our ConfigSpace ::

   num_hidden_layers = CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=1, upper=2)
   cs.add_hyperparameter(num_hidden_layers)
   
   num_units_1 = CSH.UniformIntegerHyperparameter('num_units_1', lower=64, upper=128, default=64)
   cs.add_hyperparameter(num_units_1)
   
   num_units_2 = CSH.UniformIntegerHyperparameter('num_units_2', lower=64, upper=128, default=64)
   cs.add_hyperparameter(num_units_2)
   

And now, let's create the condition, that *num_units_2* is only active if *num_hidden_layers* is greater than one::

   cond = CS.GreaterThanCondition(num_units_2, num_hidden_layers, 1)
   cs.add_condition(cond)


| In this example, we used a greater-than-condition. It remains to say, that
  ConfigSpace is able to realize more kinds of conditions, like not-equal- or less-than-conditions.
| To read more about constraints, please take a look at the `constraints-documentation [API] <constraints>`_  or the :doc:`auto_examples/AdvancedExample`
| For more information about the different hyperparameter types, visit the `hyperparameter-documentation [API] <hyperparameter>`_.


.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn