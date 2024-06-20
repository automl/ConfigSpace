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
what you can realize with it. This :ref:`Basic Usage` will include the following:

- Create a :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
- Define a simple :ref:`hyperparameter <Hyperparameters>` and its range
- Change its :ref:`distributions <Distributions>`.

The :ref:`Advanced Usage` will cover:

- Creating two sets of possible model configs, using :ref:`Conditions`
- Create two subspaces from these and add them to a parent :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
- Turn these configs into actual models!

These will not show the following and you should refer to the :doc:`user guide <guide>` for more:

- Add :ref:`Forbidden clauses`
- Add :ref:`Conditions` to the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
- :ref:`Serialize <Serialization>` the :class:`~ConfigSpace.configuration_space.ConfigurationSpace`


.. _Basic Usage:

Basic Usage
-----------

We take a look at a simple
`ridge regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_,
which has only one floating hyperparameter :math:`\alpha`.

The first step is always to create a
:class:`~ConfigSpace.configuration_space.ConfigurationSpace` object. All the
hyperparameters and constraints will be added to this object.

>>> from ConfigSpace import ConfigurationSpace, Float
>>>
>>> cs = ConfigurationSpace(
...     seed=1234,
...     space={ "alpha": (0.0, 1.0) }
... )

The hyperparameter :math:`\alpha` is chosen to have floating point values from 0 to 1.
For demonstration purpose, we sample a configuration from the :class:`~ConfigSpace.configuration_space.ConfigurationSpace` object.

>>> config = cs.sample_configuration()
>>> print(config)
Configuration(values={
  'alpha': 0.1915194503788923,
})
<BLANKLINE>

You can use this configuration just like you would a regular old python dictionary!

>>> for key, value in config.items():
...     print(key, value)
alpha 0.1915194503788923

And that's it!


.. _Advanced Usage:

Advanced Usage
--------------
Lets create a more complex example where we have two models, model `A` and model `B`.
Model `B` is some kernel based algorithm and `A` just needs a simple float hyperparamter.


We're going to create a config space that will let us correctly build a randomly selected model.


```python
class ModelA:

    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha: float
            Some value between 0 and 1
        """
        ...

class ModelB:

   def __init__(self, kernel: str, kernel_floops: int | None = None):
       """
       Parameters
       ----------
       kernel: "rbf" or "flooper"
           If the kernel is set to "flooper", kernel_floops must be set.

       kernel_floops: int | None = None
           Floop factor of the kernel
       """
       ...
```


First, lets start with building the two individual subspaces where for `A`, we want to sample alpha from a normal distribution and for `B` we have the conditioned parameter and we slightly weight one kernel over another.

```python
from ConfigSpace import ConfigSpace, Categorical, Integer, Float, Normal

class ModelA:

    def __init__(self, alpha: float):
        ...

    @staticmethod
    def space(self) -> ConfigSpace:
        return ConfigurationSpace({
            "alpha": Float("alpha", bounds=(0, 1), distribution=Normal(mu=0.5, sigma=0.2)
        })

class ModelB:

    def __init__(self, kernel: str, kernel_floops: int | None = None):
        ...

    @staticmethod
    def space(self) -> ConfigSpace:
        cs = ConfigurationSpace(
            {
                "kernel": Categorical("kernel", ["rbf", "flooper"], default="rbf", weights=[.75, .25]),
                "kernel_floops": Integer("kernel_floops", bounds=(1, 10)),
            }
        )

        # We have to make sure "kernel_floops" is only active when the kernel is "floops"
        cs.add(EqualsCondition(cs_B["kernel_floops"], cs_B["kernel"], "flooper"))

        return cs
```


Finally, we need add these two a parent space where we condition each subspace to only be active depending on a **parent**.
We'll have the default configuration be `A` but we put more emphasis when sampling on `B`

```python
cs = ConfigurationSpace(
    seed=1234,
    space={
        "model": Categorical("model", ["A", "B"], default="A", weights=[1, 2]),
    }
)

# We set the prefix and delimiter to be empty string "" so that we don't have to do
# any extra parsing once sampling
cs.add_configuration_space(
    prefix="",
    delimiter="",
    configuration_space=ModelA.space(),
    parent_hyperparameter={"parent": cs["model"], "value": "A"},
)

cs.add_configuration_space(
    prefix="",
    delimiter="",
    configuration_space=modelB.space(),
    parent_hyperparameter={"parent": cs["model"], "value": "B"}
)
```

And that's it!

However for completness, lets examine how this works by first sampling from our config space.

```python
configs = cs.sample_configuration(4)
print(configs)

# [Configuration(values={
#  'model': 'A',
#  'alpha': 0.7799758081188035,
# })
# , Configuration(values={
#   'model': 'B',
#   'kernel': 'flooper',
#   'kernel_floops': 8,
# })
# , Configuration(values={
#   'model': 'B',
#   'kernel': 'rbf',
# })
# , Configuration(values={
#   'model': 'B',
#   'kernel': 'rbf',
# })
# ]
```

We can see the three different kinds of models we have, our basic `A` model as well as our `B` model
with the two kernels.

Next, we do some processing of these configs to generate valid params to pass to these models


```python
models = []

for config in configs:
  model_type = config.pop("model")

  model = ModelA(**config) if model_type == "A" else ModelB(**config)

  models.append(model)
```


To continue reading, visit the :doc:`user guide <guide>` section. There are
more information about hyperparameters, as well as an introduction to the
powerful concepts of :ref:`Conditions` and :ref:`Forbidden clauses`.

.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn
