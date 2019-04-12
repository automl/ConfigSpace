Guide
=====

In this guide the concepts of using different hyperparameters, applying conditions and forbidden clauses to
a configuration space are captured.

They will be shown by explaining a more detailed example, a configuration space for a support vector machine.

1st Example: Integer hyperparameters and float hyperparameters
--------------------------------------------------------------

Assume that we want to train a support vector machine (=SVM) for classification tasks with the hyperparameters:

- :math:`\mathcal{C}`: regularization constant  with :math:`\mathcal{C} \in \mathbb{R}_{\geq 0}` and 0 :math:`\geq \mathcal{C} \geq` 1
- ``max_iter``: the maximum number of iterations within the solver

The implementation of the classifier is out of scope for this example and thus not shown.
But for further reading about
support vector machines and the meaning of its hyperparameter, take a look
`here <https://en.wikipedia.org/wiki/Support_vector_machine>`_.
Or directly in the implementation of the SVM in
`scikit-learn  <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`_.

The first step is always to create a
:class:`~ConfigSpace.configuration_space.ConfigurationSpace` object. All the
hyperparameters and constraints will be added to this object.

>>> import ConfigSpace as CS
>>> cs = CS.ConfigurationSpace(seed=1234)

Now, we have to define the hyperparameters :math:`\mathcal{C}` and ``max_iter``.
We choose :math:`\mathcal{C}` to be a float and
``max_iter`` to be an :class:`~ConfigSpace.hyperparameters.UniformIntegerHyperparameter` .

>>> import ConfigSpace.hyperparameters as CSH
>>> c = CSH.UniformFloatHyperparameter(name='C', lower=0, upper=1)
>>> max_iter = CSH.UniformIntegerHyperparameter(name='max_iter', lower=10, upper=100)

As last step in this example, we need to add them to the
:class:`~ConfigSpace.configuration_space.ConfigurationSpace` and sample a
configuration from it

.. doctest::

    >>> cs.add_hyperparameters([c, max_iter])
    [C, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5, max_iter, Type: ...]
    >>> cs.sample_configuration()
    Configuration:
      C, Value: 0.1915194503788923
      max_iter, Value: 66
    <BLANKLINE>


The :class:`~ConfigSpace.configuration_space.ConfigurationSpace` object *cs*
stores now the hyperparameters :math:`\mathcal{C}` and ``max_iter`` with their
defined value-ranges.

.. _1st_Example:

In a :class:`~ConfigSpace.configuration_space.Configuration` object, the value
of a parameter can be accessed or modified similar to a python dictionary.

>>> conf = cs.sample_configuration()
>>> conf['max_iter'] = 42
>>> conf['max_iter']
42

2nd Example: Categorical hyperparameters and conditions
-------------------------------------------------------

The support vector machine from the sklearn implementation supports different kinds of kernels, such as a an RBF or a polynomial kernel.
We want to include them in our configuration space.
This is a new hyperparameter with a finite number of values.
For this scenario, ConfigSpace offers the :class:`~ConfigSpace.hyperparameters.CategoricalHyperparameter`.

- ``kernel_type``: with values 'linear', 'poly', 'rbf', 'sigmoid'.

If we take a closer look at the possible values of ``kernel_type``, we observe, that the kernel type 'poly' has an additional,
unique hyperparameter, the degree of the polynomial kernel function. There is also a hyperparameter ``coef0`` for the kernel
types 'poly' and 'sigmoid', as well as ``gamma`` for 'rbf', 'poly' and 'sigmoid'

- ``degree``: degree of polynomial kernel function, being :math:`\in \mathbb{N}`
- ``coef0``: Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
- ``gamma``: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

To realize this example, we use :ref:`Conditions`.

Even in simple examples the configuration space grows easily very fast and
with it the number of possible configurations.
It makes sense to limit the search space for hyperparameter optimizations in order to quickly find good configurations.
For conditional hyperparameters (= hyperparameters which only take a value if some condition is met), ConfigSpace achieves this by sampling those
hyperparameters from the defined configuration space only if their condition is met.

To add conditions on hyperparameters to the configuration space, we first have to insert the new hyperparameters in the ``ConfigSpace``
and in a second step the conditions on them.

>>> kernel_type = CSH.CategoricalHyperparameter(
...         name='kernel_type', choices=['linear', 'poly', 'rbf', 'sigmoid'])
>>> degree = CSH.UniformIntegerHyperparameter(
...         'degree', lower=2, upper=4, default_value=2)
>>> coef0 = CSH.UniformFloatHyperparameter(
...         name='coef0', lower=0, upper=1, default_value=0.0)
>>> gamma = CSH.UniformFloatHyperparameter(
...         name='gamma', lower=1e-5, upper=1e2, default_value=1, log=True)

>>> cs.add_hyperparameters([kernel_type, degree, coef0, gamma])
[kernel_type, Type: Categorical, Choices: {linear, poly, rbf, sigmoid}, ...]

Now, we define the conditions. Conditions work by constraining a child hyperparameter (the first argument) on its parent hyperparameter (the second argument) 
being in a certain relation to a value (the third argument).
So ``CS.EqualsCondition(degree, kernel_type, 'poly')`` expresses that ``degree`` is constrained on ``kernel_type`` being equal to the value ``'poly'``. 
To express constraints involving multiple parameters or values we can use conjunctions. For example ``cond_2`` in the
following example expresses the fact, that ``coef0`` is a valid hyperparameter, if ``kernel_type`` either has the value
``'poly'`` or ``'sigmoid'``

>>> cond_1 = CS.EqualsCondition(degree, kernel_type, 'poly')

>>> cond_2 = CS.OrConjunction(CS.EqualsCondition(coef0, kernel_type, 'poly'),
...                           CS.EqualsCondition(coef0, kernel_type, 'sigmoid'))

>>> cond_3 = CS.OrConjunction(CS.EqualsCondition(gamma, kernel_type, 'rbf'),
...                           CS.EqualsCondition(gamma, kernel_type, 'poly'),
...                           CS.EqualsCondition(gamma, kernel_type, 'sigmoid'))

Add it to the configuration space

>>> cs.add_conditions([cond_1, cond_2, cond_3])
[degree | kernel_type == 'poly', (coef0 | kernel_type == 'poly' || coef0 | ...), ...]

.. note::
    ConfigSpace offers a lot of different condition types. For example the :class:`~ConfigSpace.conditions.NotEqualsCondition` ,
    :class:`~ConfigSpace.conditions.LessThanCondition`, or :class:`~ConfigSpace.conditions.GreaterThanCondition`.
    To read more about conditions, please take a look at the :ref:`Conditions` or the :doc:`auto_examples/AdvancedExample`

.. note::
    Don't use either the :class:`~ConfigSpace.conditions.EqualsCondition` or the :class:`~ConfigSpace.conditions.InCondition`
    on float hyperparameters.
    Due to floating-point inaccuracy, it is very unlikely that, for example, the
    :class:`~ConfigSpace.conditions.EqualsCondition` is evaluated to True.



3rd Example: Forbidden clauses
------------------------------

It may occur, that some states in the configuration space are not allowed.
ConfigSpace supports this functionality by offering :ref:`Forbidden clauses`.

To demonstrate the usage of :ref:`Forbidden clauses`, we assume that if ``kernel_type`` in the example above is 'linear', we use the
`LinearSVC  <http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_
sklearn implementation, which has more flexibility in the choice of penalties and loss functions.
This gives us three new hyperparameters:

- ``penalty``: Specifies the norm used in the penalization with values 'l1' or 'l2'
- ``loss``: Specifies the loss function with values 'hinge' or 'squared_hinge'
- ``dual``: Solve the optimization problem either in dual or simple form with values True or False

However, some combinations of the hyperparameters ``penalty``, ``loss`` and ``dual`` are forbidden, because they don't work together.
Firstly, add these 3 new hyperparameters to the configuration space.

>>> penalty = CSH.CategoricalHyperparameter(
...         name="penalty", choices=["l1", "l2"], default_value="l2")
>>> loss = CSH.CategoricalHyperparameter(
...         name="loss", choices=["hinge", "squared_hinge"], default_value="squared_hinge")
>>> dual = CSH.Constant("dual", "False")

>>> cs.add_hyperparameters([penalty, loss, dual])
[penalty, Type: Categorical, Choices: {l1, l2}, Default: l2, ...]

And define the conditions and add them to the configuration space.

>>> cond_1 = CS.EqualsCondition(penalty, kernel_type, 'linear')
>>> cond_2 = CS.EqualsCondition(loss, kernel_type, 'linear')
>>> cond_3 = CS.EqualsCondition(dual, kernel_type, 'linear')
>>> cs.add_conditions([cond_1, cond_2, cond_3])
[penalty | kernel_type == 'linear', ... dual | kernel_type == 'linear']

Now we want to forbid some combinations in the configurations.

- ``penalty`` is 'l1' and ``loss`` is 'hinge'
- ``dual`` is False and ``penalty`` is 'l2' and ``loss`` is 'hinge'
- ``dual`` is False and ``penalty`` is 'l1'

>>> penalty_and_loss = CS.ForbiddenAndConjunction(
...         CS.ForbiddenEqualsClause(penalty, "l1"),
...         CS.ForbiddenEqualsClause(loss, "hinge")
...     )
>>> constant_penalty_and_loss = CS.ForbiddenAndConjunction(
...         CS.ForbiddenEqualsClause(dual, "False"),
...         CS.ForbiddenEqualsClause(penalty, "l2"),
...         CS.ForbiddenEqualsClause(loss, "hinge")
...     )
>>> penalty_and_dual = CS.ForbiddenAndConjunction(
...         CS.ForbiddenEqualsClause(dual, "False"),
...         CS.ForbiddenEqualsClause(penalty, "l1")
...     )

Add them to the configuration space

>>> cs.add_forbidden_clauses([penalty_and_loss,
...                           constant_penalty_and_loss,
...                           penalty_and_dual])
[(Forbidden: penalty == 'l1' && Forbidden: loss == 'hinge'), ...]

4th Example Serialization
-------------------------

If you want to use the configuration space in another tool, such as `CAVE <https://github.com/automl/CAVE>`_, it is useful to store it to file.
To serialize the defined :class:`~ConfigSpace.configuration_space.ConfigurationSpace`, we can choose between different output formats, such as
:ref:`json <json>` or :ref:`pcs <pcs_new>`.

In this case, we want to store the :class:`~ConfigSpace.configuration_space.ConfigurationSpace` object as json file
.. testcode::

    >>> from ConfigSpace.read_and_write import json
    >>> with open('configspace.json', 'w') as fh:
    ...     fh.write(json.write(cs))

.. testoutput::
    :hide:

    3713

To read it again

.. testsetup:: json_block

    from ConfigSpace.read_and_write import json

.. doctest:: json_block

    >>> with open('configspace.json', 'r') as fh:
    ...     json_string = fh.read()
    ...     restored_conf = json.read(json_string)
