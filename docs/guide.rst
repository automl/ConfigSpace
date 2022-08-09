User Guide
==========

In this user guide, the concepts of using different hyperparameters, applying
conditions and forbidden clauses to
a configuration space are explained.

These concepts will be introduced by defining a more complex configuration space
for a support vector machine.

1st Example: Integer hyperparameters and float hyperparameters
--------------------------------------------------------------

Assume that we want to use a support vector machine (=SVM) for classification
tasks and therefore, we want to optimize its hyperparameters:

- :math:`\mathcal{C}`: regularization constant  with :math:`\mathcal{C} \in \mathbb{R}`
- ``max_iter``: the maximum number of iterations within the solver with :math:`max\_iter \in \mathbb{N}`

The implementation of the classifier is out of scope and thus not shown.
But for further reading about
support vector machines and the meaning of its hyperparameter, you can continue
reading `here <https://en.wikipedia.org/wiki/Support_vector_machine>`_ or
in the `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`_.

The first step is always to create a
:class:`~ConfigSpace.configuration_space.ConfigurationSpace` with the
hyperparameters :math:`\mathcal{C}` and ``max_iter``.

To restrict the search space, we choose :math:`\mathcal{C}` to be a
:class:`~ConfigSpace.api.types.float` between -1 and 1.
Furthermore, we choose ``max_iter`` to be an :class:`~ConfigSpace.api.types.integer.Integer` .

.. code:: python

    from ConfigSpace import ConfigurationSpace

    cs = ConfigurationSpace(
        seed=1234,
        space={
            "C": (-1.0, 1.0),  # Note the decimal to make it a float
            "max_iter": (10, 100),
        }
    )

For demonstration  purpose, we sample a configuration from it.

.. code:: python

    cs.sample_configuration()

    # Configuration(values={
    #   'C': -0.6169610992422154,
    #   'max_iter': 66,
    # })


Now, the :class:`~ConfigSpace.configuration_space.ConfigurationSpace` object *cs*
contains definitions of the hyperparameters :math:`\mathcal{C}` and ``max_iter`` with their
value-ranges.

.. _1st_Example:

Sampled instances from a :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
are called :class:`~ConfigSpace.configuration_space.Configuration`.
In a :class:`~ConfigSpace.configuration_space.Configuration` object, the value
of a parameter can be accessed or modified similar to a python dictionary.

.. code:: python

    conf = cs.sample_configuration()
    conf['max_iter'] = 42
    print(conf['max_iter'])

    # 42


2nd Example: Categorical hyperparameters and conditions
-------------------------------------------------------

The scikit-learn SVM supports different kernels, such as an RBF, a sigmoid,
a linear or a polynomial kernel. We want to include them in the configuration space.
Since this new hyperparameter has a finite number of values, we use a
:class:`~ConfigSpace.api.types.categorical`.


- ``kernel_type``: with values 'linear', 'poly', 'rbf', 'sigmoid'.

Taking a look at the SVM documentation, we observe that if the kernel type is
chosen to be 'poly', another hyperparameter ``degree`` must be specified.
Also, for the kernel types 'poly' and 'sigmoid', there is an additional hyperparameter ``coef0``.
As well as the hyperparameter ``gamma`` for the kernel types 'rbf', 'poly' and 'sigmoid'.

- ``degree``: the degree of a polynomial kernel function, being :math:`\in \mathbb{N}`
- ``coef0``: Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
- ``gamma``: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

To realize the different hyperparameter for the kernels, we use :ref:`Conditions`.

Even in simple examples, the configuration space grows easily very fast and
with it the number of possible configurations.
It makes sense to limit the search space for hyperparameter optimizations in
order to quickly find good configurations. For conditional hyperparameters
(= hyperparameters which only take a value if some condition is met), ConfigSpace
achieves this by sampling those hyperparameters from the configuration
space only if their condition is met.

To add conditions on hyperparameters to the configuration space, we first have
to insert the new hyperparameters in the ``ConfigSpace`` and in a second step, the
conditions on them.

.. code:: python

    from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer

    kernel_type = Categorical('kernel_type', ['linear', 'poly', 'rbf', 'sigmoid'])
    degree = Integer('degree', bounds=(2, 4), default=2)
    coef0 = Float('coef0', bounds=(0, 1), default=0.0)
    gamma = Float('gamma', bounds=(1e-5, 1e2), default_value=1, log=True)

    cs = ConfigurationSpace()
    cs.add_hyperparameters([kernel_type, degree, coef0, gamma])

    # [kernel_type, Type: Categorical, Choices: {linear, poly, rbf, sigmoid}, ...]

First, we define the conditions. Conditions work by constraining a child
hyperparameter (the first argument) on its parent hyperparameter (the second argument)
being in a certain relation to a value (the third argument).
``EqualsCondition(degree, kernel_type, 'poly')`` expresses that ``degree`` is
constrained on ``kernel_type`` being equal to the value 'poly'.  To express
constraints involving multiple parameters or values, we can use conjunctions.
In the following example, ``cond_2`` describes that ``coef0``
is a valid hyperparameter, if the ``kernel_type`` has either the value
'poly' or 'sigmoid'.

.. code:: python

    from ConfigSpace import EqualsCondition, OrConjunction

    cond_1 = EqualsCondition(degree, kernel_type, 'poly')

    cond_2 = OrConjunction(
        EqualsCondition(coef0, kernel_type, 'poly'),
        EqualsCondition(coef0, kernel_type, 'sigmoid')
    )

    cond_3 = OrConjunction(
        EqualsCondition(gamma, kernel_type, 'rbf'),
        EqualsCondition(gamma, kernel_type, 'poly'),
        EqualsCondition(gamma, kernel_type, 'sigmoid')
    )

In this specific example, you may wish to use the :class:`~ConfigSpace.conditions.InCondition` to express
that ``gamma`` is valid if ``kernel_type in ["rbf", "poly", "sigmoid"]`` which we show for completness

.. code:: python

   from ConfigSpace import InCondition

   cond_3 = InCondition(gamma, kernel_type, ["rbf", "poly", "sigmoid"])

Finally, we add the conditions to the configuration space

.. code:: python

    cs.add_conditions([cond_1, cond_2, cond_3])

    # [degree | kernel_type == 'poly', (coef0 | kernel_type == 'poly' || coef0 | ...), ...]

.. note::

    ConfigSpace offers a lot of different condition types. For example the
    :class:`~ConfigSpace.conditions.NotEqualsCondition`,
    :class:`~ConfigSpace.conditions.LessThanCondition`,
    or :class:`~ConfigSpace.conditions.GreaterThanCondition`.
    To read more about conditions, please take a look at the :ref:`Conditions`.

.. note::
    Don't use either the :class:`~ConfigSpace.conditions.EqualsCondition` or the
    :class:`~ConfigSpace.conditions.InCondition` on float hyperparameters.
    Due to floating-point inaccuracy, it is very unlikely that the
    :class:`~ConfigSpace.conditions.EqualsCondition` is evaluated to True.


3rd Example: Forbidden clauses
------------------------------

It may occur that some states in the configuration space are not allowed.
ConfigSpace supports this functionality by offering :ref:`Forbidden clauses`.

We demonstrate the usage of :ref:`Forbidden clauses` by defining the
configuration space for the
`linear SVM  <http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_.
Again, we use the sklearn implementation. This implementation has three
hyperparameters to tune:

- ``penalty``: Specifies the norm used in the penalization with values 'l1' or 'l2'
- ``loss``: Specifies the loss function with values 'hinge' or 'squared_hinge'
- ``dual``: Solves the optimization problem either in the dual or simple form with values True or False

Because some combinations of ``penalty``, ``loss`` and ``dual`` just don't work
together, we want to make sure that these combinations are not sampled from the
configuration space.

First, we add these three new hyperparameters to the configuration space.

.. code:: python

    from ConfigSpace import ConfigurationSpace, Categorical, Constant

    penalty = Categorical("penalty", ["l1", "l2"], default="l2")
    loss = Categorical("loss", ["hinge", "squared_hinge"], default="squared_hinge")
    dual = Constant("dual", "False")
    cs.add_hyperparameters([penalty, loss, dual])

    # [penalty, Type: Categorical, Choices: {l1, l2}, Default: l2, ...]

Now, we want to forbid the following hyperparameter combinations:

- ``penalty`` is 'l1' and ``loss`` is 'hinge'
- ``dual`` is False and ``penalty`` is 'l2' and ``loss`` is 'hinge'
- ``dual`` is False and ``penalty`` is 'l1'

.. code:: python

    from ConfigSpace import ForbiddenEqualsClause, ForbiddenAndConjunction

    penalty_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(penalty, "l1"),
        ForbiddenEqualsClause(loss, "hinge")
    )
    constant_penalty_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(dual, "False"),
        ForbiddenEqualsClause(penalty, "l2"),
        ForbiddenEqualsClause(loss, "hinge")
    )
    penalty_and_dual = ForbiddenAndConjunction(
        ForbiddenEqualsClause(dual, "False"),
        ForbiddenEqualsClause(penalty, "l1")
    )

In the last step, we add them to the configuration space object:

.. code:: python

    cs.add_forbidden_clauses([penalty_and_loss, constant_penalty_and_loss, penalty_and_dual])

    # [(Forbidden: penalty == 'l1' && Forbidden: loss == 'hinge'), ...]


4th Example Serialization
-------------------------

If you want to use the configuration space in another tool, such as
`CAVE <https://github.com/automl/CAVE>`_, it is useful to store it to file.
To serialize the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`,
we can choose between different output formats, such as
:ref:`json <json>` or :ref:`pcs <pcs_new>`.

In this example, we want to store the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
object as json file

.. code:: python

    from ConfigSpace.read_and_write import json
    with open('configspace.json', 'w') as fh:
        fh.write(json.write(cs))

To read it from file

.. code:: python

    with open('configspace.json', 'r') as fh:
        json_string = fh.read()
        restored_conf = json.read(json_string)



5th Example: Placing priors on the hyperparameters
--------------------------------------------------

If you want to conduct black-box optimization in SMAC (https://arxiv.org/abs/2109.09831), and you have prior knowledge about the which regions of the search space are more likely to contain the optimum, you may include this knowledge when designing the configuration space. More specifically, you place prior distributions over the optimum on the parameters, either by a (log)-normal or (log)-Beta distribution. SMAC then considers the given priors through the optimization by using PiBO (https://openreview.net/forum?id=MMAeCXIa89).

Consider the case of optimizing the accuracy of an MLP with three hyperparameters: learning rate [1e-5, 1e-1], dropout [0, 0.99] and activation {Tanh, ReLU}. From prior experience, you believe the optimal learning rate to be around 1e-3, a good dropout to be around 0.25, and the optimal activation function to be ReLU about 80% of the time. This can be represented accordingly:

.. code-block:: python

    import numpy as np
    from ConfigSpace.configuration_space import ConfigurationSpace

    # convert 10 log to natural log for learning rate, mean 1e-3
    # with two standard deviations on either side of the mean to cover the search space
    logmean = np.log(1e-3)
    logstd = np.log(10.0)

    cs = ConfigurationSpace({
        "lr": Float('lr', bounds=(1e-5, 1e-1), default=1e-3, log=True, disitribution=Normal(logmean, logstd)),
        "dropout": Float('dropout', bounds=(0, 0.99), default=0.25, distribution=Beta(alpha=2, beta=4)),
        "activation": Categorical('activation', ['tanh', 'relu'], weights=[0.2, 0.8]),
    })
    # [lr, Type: NormalFloat, Mu: -6.907755278982137 Sigma: 2.302585092994046, Range: [1e-05, 0.1], Default: 0.001, on log-scale, dropout, Type: BetaFloat, Alpha: 2.0 Beta: 4.0, Range: [0.0, 0.99], Default: 0.25, activation, Type: Categorical, Choices: {tanh, relu}, Default: tanh, Probabilities: (0.2, 0.8)]

To check that your prior makes sense for each hyperparameter, you can easily do so with the ``__pdf__`` method. There, you will see that the probability of the optimal learning rate peaks at 10^-3, and decays as we go further away from it:

.. code-block:: python

    test_points = np.logspace(-5, -1, 5)
    print(test_points)

    # array([1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01])

The pdf function accepts an (N, ) numpy array as input.

.. code-block:: python

    test_points_pdf = lr.pdf(test_points)
    print(test_points_pdf)

    # array([0.02456573, 0.11009594, 0.18151753, 0.11009594, 0.02456573])




