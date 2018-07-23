How to use ConfigSpace
======================

A ConfigSpace-Object is equal to a container holding all different kinds of hyperparameters.
For example it is able to distinguish between numerical, and categorical hyperparameters.
As well as conditional hyperparameters.

The usage of this will be shown in the following short examples.

First lets create a ConfigSpace::

   import ConfigSpace as CS
   cs = CS.ConfigSpace()
   
Later, we can add all our hyperparameters and conditions to this object.

A first simple example - CategoricalHyperparameters:
----------------------------------------------------

Assume, we want to train an RBF-kernel based soft-margin SVM classifier. Which has at least two hyperparametrs.
A regularization constant :math:`\mathcal{C}` and a kernel hyperparameter  :math:`\gamma` with

- :math:`\mathcal{C} \in` {10, 100, 1000}
- :math:`\gamma \in` {0.1, 0.2, 0.5, 1.0}

:: 

   import ConfigSpace.hyperparameters as CSH
   c = CSH.CategoricalHyperparameter('C', [10, 100, 1000])
   gamma = CSH.CategoricalHyperparameter('gamma', [0.1, 0.2, 0.5, 1.0])

And add them to the ConfigSpace::

   cs.add_hyperparameter(c)
   cs.add_hyperparameter(gamma)

IntegerHyperparameters and FloatHyperparameters:
------------------------------------------------

As already mentioned, ConfigSpace is also able to distinguish between some other common types like intger and float hyperparameter.
Assume we have another model with a hyperparameter *learning_rate* and a integer hyperparameter - for example the batch size.

In the case of the learning rate, we'd like to sample a value on a logarithmic scale from 0.1 to 0.000001::
   
   lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
   cs.add_hyperparameter(lr)
   
   bs = CSH.UniformIntegerHyperparameter('bs', lower=64, upper=128, default_value=64)
   cs.add_hyperparameter(bs)
 
ConditionalHyperparameters:
---------------------------

Sometimes it happens, that it is necessary to integrate conditions in our configSpace. 
For example, if you like to optimize a simple MLP neural network with either one or two hidden layers. 
But each layer should have a independent number of units. 

ConfigSpace is again capable of handling this situation::

   num_hidden_layers = CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=1, upper=2)
   cs.add_hyperparameter(num_hidden_layers)
   
   num_units_1 = CSH.UniformIntegerHyperparameter('num_units_1', lower=64, upper=128, default=64)
   cs.add_hyperparameter(num_units_1)
   
   num_units_2 = CSH.UniformIntegerHyperparameter('num_units_2', lower=64, upper=128, default=64)
   cs.add_hyperparameter(num_units_2)
   
Now, lets create the condition, that *num_units_2* is only active if *num_hidden_layers* is greater than one::

   cond = CS.GreaterThanCondition(num_units_2, num_hidden_layers, 1)
   cs.add_condition(cond)
	

   
   
   