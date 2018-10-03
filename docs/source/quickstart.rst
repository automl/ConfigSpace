Quickstart
==========

A ``ConfigurationSpace`` is a data structure to describe the configuration space of an algorithm to tune.
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

We will show those steps in an exemplary way, by creating a ``ConfigurationSpace`` for a support vector machine (=SVM) classifier
and a neural network.

1st Example: Basic Usage and Conditional Hyperparameter
-------------------------------------------------------

Assume that we want to train a support vector machines (=SVM) classifier, but we are not sure, which kernel is the best for our dataset.
Besides the hyperparameter kernel type, we are interested in results from varying a regularization constant :math:`\mathcal{C}`:

- regularization constant :math:`\mathcal{C} \in \mathbb{R}_{\geq 0}` with 0 :math:`\geq \mathcal{C} \geq` 10
- kernel type :math:`\in` {'linear', 'poly', 'rbf', 'sigmoid'}

The implementation of the classifier is out of scope for this example and thus not shown. But for further reading about
support vector machines and the meaning of its hyperparameter, take a look `here <https://en.wikipedia.org/wiki/Support_vector_machine>`_.
Or directly in the implementation of the SVM in
`scikit-learn  <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`_.

The first step is always to create a ``ConfigurationSpace`` object. All the hyperparameters and constraints will be added to this
object.
::

   import ConfigSpace as CS
   cs = CS.ConfigurationSpace()

Now, we have to define the hyperparameters :math:`\mathcal{C}` and kernel type. Since the kernel hyperparameter only can
take on values from a given subset, we use for this a categorical hyperparameter. For the regularization constant
we choose a floating hyperparameter.
:: 

   import ConfigSpace.hyperparameters as CSH
   c = CSH.UniformFloatHyperparameter(name='C', lower=0, upper=100)
   kernel = CSH.CategoricalHyperparameter(name='kernel', choices=['linear', 'poly', 'rbf', 'sigmoid'])

As last step in this example, we need to add them to the ``ConfigurationSpace`` object and sample a configuration from it::

   cs.add_hyperparameter(c)
   cs.add_hyperparameter(kernel)
   cs.sample_configuration()

And that's it.
The ``ConfigurationSpace`` object *cs* stores the hyperparameters :math:`\mathcal{C}` and kernel with their defined value-ranges.

2nd Example: Integer hyperparameters and float hyperparameters:
---------------------------------------------------------------

| As already mentioned, ConfigSpace is also able to distinguish between some other common types like integer- and float hyperparameter.
| In the next example, we will create a ``ConfigurationSpace`` object for a simple neural network.
| Assume this neural network model has a float hyperparameter *learning_rate* and an integer hyperparameter
  *number of units in the hidden layer*.
| Since the learning rate should be sampled from a value range from 0.000001 to 0.1, we choose this hyperparameter to be
  a float hyperparameter.
| Note, that the parameter *log* is set to True: This causes that the values of this hyperparameter
  is sampled from a logarithmic scale.
  In hyperparameter-optimization this is useful, because a logarithmic scale enables us to search a larger space quickly.
  It basically states that a change from 0.01 to 0.1 is as important as a change from 0.001 to 0.01.

::

   lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

   num_hidden_units = CSH.UniformIntegerHyperparameter('num_hidden_units', lower=10, upper=150, default_value=100)
   cs.add_hyperparameters([lr, num_hidden_units])
 
3rd Example: Conditional Hyperparameters:
-----------------------------------------

In real world applications, hyperparameters are often dependent on the values of other hyperparameters.
This means, that a hyperparameter, which is dependent on a parent hyperparameter should only be sampled if the parent satisfies a defined condition.

To create a example, which has conditions,
we extend the neural network example from above by another hyperparameter, the *number of hidden layers*. Since
each layer should have an independent *number of units*, we have to add for each layer an own hyperparameter: *number of hidden units 1*
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

First, add the new hyperparameters to our ``ConfigurationSpace`` object ::

   num_hidden_layers = CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=1, upper=2)
   cs.add_hyperparameter(num_hidden_layers)
   
   num_units_1 = CSH.UniformIntegerHyperparameter('num_units_1', lower=64, upper=128, default=64)

   num_units_2 = CSH.UniformIntegerHyperparameter('num_units_2', lower=64, upper=128, default=64)
   # you can also add them with one function call
   cs.add_hyperparameters([num_units_1, num_units_2])
   

And now, let's create the condition, that *num_units_2* is only active if *num_hidden_layers* is greater than one::

   cond = CS.GreaterThanCondition(num_units_2, num_hidden_layers, 1)
   cs.add_condition(cond)


| In this example, we used a ``GreaterThanCondition``. It remains to say, that
  ConfigSpace is able to realize more kinds of conditions, like ``NotEqualCondition`` or ``LessThanCondition``.
| To read more about conditions, please take a look at the :ref:`Conditions` or the :doc:`auto_examples/AdvancedExample`
| For more information about the different hyperparameter types, visit the :ref:`hyperparameters`.
  In the :doc:`Guide`, you will learn another powerful kind of restriction to the configuration space, the :ref:`Forbidden clauses`.


.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn