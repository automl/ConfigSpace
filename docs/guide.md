## User Guide
In this user guide, the concepts of using different hyperparameters,
applying conditions and forbidden clauses to a configuration space are explained.

These concepts will be introduced by defining a more complex configuration space
for a support vector machine.

### 1st Example: Integer hyperparameters and float hyperparameters
Assume that we want to use a support vector machine (=SVM) for classification
tasks and therefore, we want to optimize its hyperparameters:

- `C`: regularization constant  with `C` being a float value.
- `max_iter`: the maximum number of iterations within the solver with `max_iter` being a positive integer.

The implementation of the classifier is out of scope and thus not shown.
But for further reading about support vector machines and the meaning of its hyperparameter,
you can continue reading [here](https://en.wikipedia.org/wiki/Support_vector_machine) or
in the [scikit-learn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC).

The first step is always to create a
[`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace] with the
hyperparameters `C` and `max_iter`.

To restrict the search space, we choose `C` to be a
[`Float`][ConfigSpace.api.types.float.Float] between -1 and 1.
Furthermore, we choose `max_iter` to be an [`Integer`][ConfigSpace.api.types.integer.Integer].

```python exec="True" source="material-block" result="python" session="example_one"
from ConfigSpace import ConfigurationSpace

cs = ConfigurationSpace(
    space={
        "C": (-1.0, 1.0),  # Note the decimal to make it a float
        "max_iter": (10, 100),
    },
    seed=1234,
)
print(cs)
```

Now, the [`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace] object *cs*
contains definitions of the hyperparameters `C` and `max_iter` with their
value-ranges.

For demonstration purpose, we sample a configuration from it.

```python exec="True" source="material-block" result="python" session="example_one"
config = cs.sample_configuration()
print(config)
```

Sampled instances from a [`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace]
are called a [`Configuration`][ConfigSpace.configuration.Configuration].
In a [`Configuration`][ConfigSpace.configuration.Configuration],
a parameter can be accessed or modified similar to a python dictionary.

```python exec="True" source="material-block" result="python" session="example_one"
for key, value in config.items():
    print(f"{key}: {value}")

print(config["C"])
```


### 2nd Example: Categorical hyperparameters and conditions
The scikit-learn SVM supports different kernels, such as an RBF, a sigmoid,
a linear or a polynomial kernel. We want to include them in the configuration space.
Since this new hyperparameter has a finite number of values, we use a
[`Categorical`][`ConfigSpace.api.types.categorical.Categorical`].

- `kernel_type` in `#!python ['linear', 'poly', 'rbf', 'sigmoid']`.

Taking a look at the SVM documentation, we observe that if the kernel type is
chosen to be `'poly'`, another hyperparameter `degree` must be specified.
Also, for the kernel types `'poly'` and `'sigmoid'`, there is an additional hyperparameter `coef0`.
As well as the hyperparameter `gamma` for the kernel types `'rbf'`, `'poly'` and `'sigmoid'`.

- `degree`: the integer degree of a polynomial kernel function.
- `coef0`: Independent term in kernel function. It is only needed for `'poly'` and `'sigmoid'` kernel.
- `gamma`: Kernel coefficient for `'rbf'`, `'poly'` and `'sigmoid'`.

To realize the different hyperparameter for the kernels, we use **Conditionals**.
Please refer to their [reference page](./reference/conditions.md) for more.

Even in simple examples, the configuration space grows easily very fast and
with it the number of possible configurations.
It makes sense to limit the search space for hyperparameter optimizations in
order to quickly find good configurations. For conditional hyperparameters
_(hyperparameters which only take a value if some condition is met)_, ConfigSpace
achieves this by sampling those hyperparameters from the configuration
space only if their condition is met.

To add conditions on hyperparameters to the configuration space, we first have
to insert the new hyperparameters in the `ConfigSpace` and in a second step, the
conditions on them.

```python exec="True" source="material-block" result="python" session="example_two"
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer

kernel_type = Categorical('kernel_type', ['linear', 'poly', 'rbf', 'sigmoid'])
degree = Integer('degree', bounds=(2, 4), default=2)
coef0 = Float('coef0', bounds=(0, 1), default=0.0)
gamma = Float('gamma', bounds=(1e-5, 1e2), default=1, log=True)

cs = ConfigurationSpace()
cs.add([kernel_type, degree, coef0, gamma])
print(cs)
```

First, we define the conditions. Conditions work by constraining a child
hyperparameter (the first argument) on its parent hyperparameter (the second argument)
being in a certain relation to a value (the third argument).
`EqualsCondition(degree, kernel_type, 'poly')` expresses that `degree` is
constrained on `kernel_type` being equal to the value `'poly'`.
To express constraints involving multiple parameters or values, we can use conjunctions.
In the following example, `cond_2` describes that `coef0`
is a valid hyperparameter, if the `kernel_type` has either the value `'poly'` or `'sigmoid'`.

```python exec="True" source="material-block" result="python" session="example_two"
from ConfigSpace import EqualsCondition, InCondition, OrConjunction

# read as: "degree is active if kernel_type == 'poly'"
cond_1 = EqualsCondition(degree, kernel_type, 'poly')

# read as: "coef0 is active if (kernel_type == 'poly' or kernel_type == 'sigmoid')"
# You could also define this using an InCondition as shown below
cond_2 = OrConjunction(
    EqualsCondition(coef0, kernel_type, 'poly'),
    EqualsCondition(coef0, kernel_type, 'sigmoid')
)

# read as: "gamma is active if kernel_type in ['rbf', 'poly', 'sigmoid']"
cond_3 = InCondition(gamma, kernel_type, ['rbf', 'poly','sigmoid'])
```

Finally, we add the conditions to the configuration space

```python exec="True" source="material-block" result="python" session="example_two"
cs.add([cond_1, cond_2, cond_3])
print(cs)
```

!!! note

    ConfigSpace offers a lot of different condition types.
    Please check out the [conditions reference page](./reference/conditions.md) for more.

!!! warning
 
    We advise not  using the `EqualsCondition` or the `InCondition` on float hyperparameters.
    Due to numerical rounding that can occur, it can be the case that these conditions evaluate to
    `False` even if they should evaluate to `True`.


### 3rd Example: Forbidden clauses
It may occur that some states in the configuration space are not allowed.
ConfigSpace supports this functionality by offering **Forbidden clauses**.

We demonstrate the usage of Forbidden clauses by defining the configuration space for the
[linear SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC).
Again, we use the sklearn implementation. This implementation has three
hyperparameters to tune:

- `penalty`: Specifies the norm used in the penalization with values `'l1'` or `'l2j'`.
- `loss`: Specifies the loss function with values `'hinge'` or `'squared_hinge'`.
- `dual`: Solves the optimization problem either in the dual or simple form with values `True` or `False`.

Because some combinations of `penalty`, `loss` and `dual` just don't work
together, we want to make sure that these combinations are not sampled from the
configuration space.
It is possible to represent these as conditionals, however sometimes it is easier to
express them as forbidden clauses.

First, we add these three new hyperparameters to the configuration space.

```python exec="True" source="material-block" result="python" session="example_three"
from ConfigSpace import ConfigurationSpace, Categorical, Constant

cs = ConfigurationSpace()

penalty = Categorical("penalty", ["l1", "l2"], default="l2")
loss = Categorical("loss", ["hinge", "squared_hinge"], default="squared_hinge")
dual = Constant("dual", "False")
cs.add([penalty, loss, dual])
print(cs)
```

Now, we want to forbid the following hyperparameter combinations:

- `penalty` is `'l1'` and `loss` is `'hinge'`.
- `dual` is False and `penalty` is `'l2'` and `loss` is `'hinge'`
- `dual` is False and `penalty` is `'l1'`

```python exec="True" source="material-block" result="python" session="example_three"
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
```

In the last step, we add them to the configuration space object:

```python exec="True" source="material-block" result="python" session="example_three"
cs.add([penalty_and_loss, constant_penalty_and_loss, penalty_and_dual])
print(cs)
```


### 4th Example Serialization
To serialize the `ConfigurationSpace` object, we can choose between different output formats, such as
as plain-type dictionary, directly to `.yaml` or `.json` and if required for backwards compatiblity `pcs`.
Plese see the [serialization reference page](./reference/serialization.md) for more.

In this example, we want to store the [`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace]
object as a `.yaml` file.

```python exec="True" source="material-block" result="yaml" session="example_four"
from pathlib import Path
from ConfigSpace import ConfigurationSpace

path = Path("configspace.yaml")
cs = ConfigurationSpace(
    space={
        "C": (-1.0, 1.0),  # Note the decimal to make it a float
        "max_iter": (10, 100),
    },
    seed=1234,
)
cs.to_yaml(path)
loaded_cs = ConfigurationSpace.from_yaml(path)

with path.open() as f:
    print(f.read())
path.unlink()  # markdown-exec: hide
```

If you require custom encoding or decoding or parameters, please refer to the
[serialization reference page](./reference/serialization.md) for more.

### 5th Example: Placing priors on the hyperparameters
If you want to conduct black-box optimization in [SMAC](https://arxiv.org/abs/2109.09831),
and you have prior knowledge about the which regions of the search space are more likely to contain the optimum,
you may include this knowledge when designing the configuration space.
More specifically, you place prior distributions over the optimum on the parameters,
either by a (log)-normal or (log)-Beta distribution.
SMAC then considers the given priors through the optimization by using
[PiBO](https://openreview.net/forum?id=MMAeCXIa89).

Consider the case of optimizing the accuracy of an MLP with three hyperparameters:

* learning rate in  `(1e-5, 1e-1)`
* dropout in `(0, 0.99)`
* activation in `["Tanh", "ReLU"]`.

From prior experience, you believe the optimal learning rate to be around `1e-3`,
a good dropout to be around `0.25`,
and the optimal activation function to be ReLU about 80% of the time.

This can be represented accordingly:

```python exec="True" source="material-block" result="python" session="example_five"
import numpy as np
from ConfigSpace import ConfigurationSpace, Float, Categorical, Beta, Normal

cs = ConfigurationSpace(
    space={
        "lr": Float(
            'lr',
            bounds=(1e-5, 1e-1),
            default=1e-3,
            log=True,
            distribution=Normal(1e-3, 1e-1)
        ),
        "dropout": Float(
            'dropout',
            bounds=(0, 0.99),
            default=0.25,
            distribution=Beta(alpha=2, beta=4)
        ),
        "activation": Categorical(
            'activation',
            items=['tanh', 'relu'],
            weights=[0.2, 0.8]
        ),
    },
    seed=1234,
)
print(cs)
```

To check that your prior makes sense for each hyperparameter,
you can easily do so with the [`pdf_values()`][ConfigSpace.hyperparameters.Hyperparameter.pdf_values] method.
There, you will see that the probability of the optimal learning rate peaks at
10^-3, and decays as we go further away from it:

```python exec="True" source="material-block" result="python" session="example_five"
test_points = np.logspace(-5, -1, 5)
print(test_points)
print(cs['lr'].pdf_values(test_points))
```
