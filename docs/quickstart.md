## Quickstart
A [ConfigurationSpace][ConfigSpace.configuration_space.ConfigurationSpace]
is a data structure to describe the configuration space of an algorithm to tune.
Possible hyperparameter types are numerical, categorical, conditional and ordinal hyperparameters.

AutoML tools, such as [`SMAC3`](https://github.com/automl/SMAC3) and [`BOHB`](https://github.com/automl/HpBandSter) are using the configuration space
module to sample hyperparameter configurations.
Also, [`auto-sklearn`](https://github.com/automl/auto-sklearn), an automated machine learning toolkit, which frees the
machine learning user from algorithm selection and hyperparameter tuning,
makes heavy use of the ConfigSpace package.

This simple quickstart tutorial will show you, how to set up your own
[ConfigurationSpace][ConfigSpace.configuration_space.ConfigurationSpace], and will demonstrate
what you can realize with it. This [Basic Usage](#basic-usage) will include the following:

- Create a [ConfigurationSpace][ConfigSpace.configuration_space.ConfigurationSpace]
- Define a simple [hyperparameter](./reference/hyperparameters.md) with a float value

The [Advanced Usage](#advanced-usage) will cover:

- Creating two sets of possible model configs, using [Conditions](./reference/conditions.md).
- Use a different distirbution for one of the hyperparameters.
- Create two subspaces from these and add them to a parent [ConfigurationSpace][ConfigSpace.configuration_space.ConfigurationSpace]
- Turn these configs into actual models!

These will not show the following and you should refer to the [user guide](./guide.md) for more:

- Add [Forbidden clauses](./reference/forbiddens.md)
- Add [Conditions](./reference/conditions.md)
- [Serialize](./reference/configuration.md)


### Basic Usage

We take a look at a simple
[ridge regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
which has only one floating hyperparameter `alpha`.

The first step is always to create a
[ConfigurationSpace][ConfigSpace.configuration_space.ConfigurationSpace] object. All the
hyperparameters and constraints will be added to this object.

```python exec="true", source="material-block" result="python" session="quickstart-basic"
from ConfigSpace import ConfigurationSpace, Float

cs = ConfigurationSpace(space={"alpha": (0.0, 1.0)}, seed=1234)
print(cs)
```

The hyperparameter `alpha` is chosen to have floating point values from `0` to `1`.
For demonstration purpose, we sample a configuration from the [ConfigurationSpace][ConfigSpace.configuration_space.ConfigurationSpace] object.

```python exec="true", source="material-block" result="python" session="quickstart-basic"
config = cs.sample_configuration()
print(config)
```

You can use this configuration just like you would a regular old python dictionary!

```python exec="true", source="material-block" result="python" session="quickstart-basic"
for key, value in config.items():
    print(key, value)
```

And that's it!


### Advanced Usage
Lets create a more complex example where we have two models, model `A` and model `B`.
Model `B` is some kernel based algorithm and `A` just needs a simple float hyperparamter.


We're going to create a config space that will let us correctly build a randomly selected model.


```python exec="true", source="material-block" result="python" session="quickstart-advanced"
from typing import Literal
from dataclasses import dataclass

@dataclass
class ModelA:
    alpha: float
    """Some value between 0 and 1"""

@dataclass
class ModelB:
    kernel: Literal["rbf", "flooper"]
    """Kernel type."""

    kernel_floops: int | None = None
    """Number of floops for the flooper kernel, only used if kernel == "flooper"."""
```


First, lets start with building the two individual subspaces where for `A`, we want to sample alpha from a normal distribution and for `B` we have the conditioned parameter and we slightly weight one kernel over another.

```python exec="true", source="material-block" result="python" session="quickstart-advanced"
from typing import Literal
from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float, Normal, EqualsCondition

@dataclass
class ModelA:
    alpha: float
    """Some value between 0 and 1"""

    @staticmethod
    def space() -> ConfigurationSpace:
        return ConfigurationSpace({
            "alpha": Float("alpha", bounds=(0, 1), distribution=Normal(mu=0.5, sigma=0.2))
        })

@dataclass
class ModelB:
    kernel: Literal["rbf", "flooper"]
    """Kernel type."""

    kernel_floops: int | None = None
    """Number of floops for the flooper kernel, only used if kernel == "flooper"."""

    @staticmethod
    def space() -> ConfigurationSpace:
        cs = ConfigurationSpace(
            {
                "kernel": Categorical("kernel", ["rbf", "flooper"], default="rbf", weights=[.75, .25]),
                "kernel_floops": Integer("kernel_floops", bounds=(1, 10)),
            }
        )

        # We have to make sure "kernel_floops" is only active when the kernel is "floops"
        cs.add(EqualsCondition(cs["kernel_floops"], cs["kernel"], "flooper"))

        return cs
```


Finally, we need add these two a parent space where we condition each subspace to only be active depending on a **parent**.
We'll have the default configuration be `A` but we put more emphasis when sampling on `B`

```python exec="true", source="material-block" result="python" session="quickstart-advanced"
from ConfigSpace import ConfigurationSpace, Categorical

cs = ConfigurationSpace(
    seed=123456,
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
    configuration_space=ModelB.space(),
    parent_hyperparameter={"parent": cs["model"], "value": "B"}
)
print(cs)
```

And that's it!

However for completness, lets examine how this works by first sampling from our config space.

```python exec="true", source="material-block" result="python" session="quickstart-advanced"
configs = cs.sample_configuration(4)
print(configs)
```

We can see the three different kinds of models we have, our basic `A` model as well as our `B` model
with the two kernels.

Next, we do some processing of these configs to generate valid params to pass to these models


```python exec="true", source="material-block" result="python" session="quickstart-advanced"
models = []

for config in configs:
    config_as_dict = dict(config)
    model_type = config_as_dict.pop("model")

    model = ModelA(**config_as_dict) if model_type == "A" else ModelB(**config_as_dict)

    models.append(model)

print(models)
```


To continue reading, visit the [user guide](./guide.md) section. There are
more information about hyperparameters, as well as an introduction to the
powerful concepts of [Conditions](./reference/conditions.md) and [Forbidden clauses](./reference/forbiddens.md).
