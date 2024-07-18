## Welcome to ConfigSpace's documentation!
ConfigSpace is a simple python package to manage configuration spaces for
[algorithm configuration](https://ml.informatik.uni-freiburg.de/papers/09-JAIR-ParamILS.pdf) and
[hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) tasks.
It includes various modules to translate between different text formats for configuration space descriptions.

ConfigSpace is often used in AutoML tools such as
[SMAC3](https://github.com/automl/SMAC3),
[BOHB](https://github.com/automl/HpBandSter),
[auto-sklearn](https://github.com/automl/auto-sklearn).
To read more about our group and projects, visit our homepage [AutoML.org](https://www.automl.org).

This documentation explains how to use ConfigSpace and demonstrates its features.
In the [quickstart](./quickstart.md), you will see how to set up a
[`ConfiguratonSpace`][ConfigSpace.configuration_space.ConfigurationSpace]
and add hyperparameters of different types to it.
Besides containing hyperparameters, `ConfigurationSpace` can contain constraints such as conditions and forbidden clauses.
Those are introduced in the [user guide](./guide.md)

!!! tip "New in 1.0!"

    In ConfigSpace 1.0, we have removed the dependancy on `Cython` while even improving
    the performance!

    * Should now install anywhere.
    * You can now use your editor to jump to definition and see the source code.
    * Contribute more easily!

    There is no also better support in Categorical, Ordinal and Constant hyperparameters,
    for arbitrary values, for example:

    ```python
    from dataclasses import dataclass
    from ConfigSpace import ConfigurationSpace, Constant

    @dataclass
    class A:
        a: int

    def f() -> None:
        return None

    cs = ConfigurationSpace({
        "cat": [True, False, None],
        "othercat": [A(1), f],
        "constant": Constant("constant": (24, 25)),
    })
    ```


    With this, we have also deprecated many of the previous functions, simplifying the API
    where possible or improving it's clarity. We have tried hard to keep everything backwards
    compatible, and also recommend the new functionality to use!

    We've also made some strides towards extensibilty of ConfigSpace, making it simpler to
    define you own hyperparamter types. Please see the
    [hyperparameter reference](./reference/hyperparameters.md) page for more.

    !!! warning

        One notable hard removal is the use of the `"q"` parameter to numerical parameters.
        We recommend using an `Ordinal` distribution where possible. Please let us know if this
        effects you and we can help migrate where possible.

### Getting Started
Create a simple [`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace]
and then sample a [`Configuration`][ConfigSpace.configuration.Configuration] from it.

```python exec="True" result="python" source="material-block"
from ConfigSpace import ConfigurationSpace

cs = ConfigurationSpace({
    "myfloat": (0.1, 1.5),                # Uniform Float
    "myint": (2, 10),                     # Uniform Integer
    "species": ["mouse", "cat", "dog"],   # Categorical
})
configs = cs.sample_configuration(2)
print(configs)
```


Use [`Float`][ConfigSpace.api.types.float.Float],
[`Integer`][ConfigSpace.api.types.integer.Integer],
and [`Categorical`][ConfigSpace.api.types.categorical.Categorical] to define hyperparameters
and define how sampling is done.

```python exec="True" result="python" source="material-block"
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal

cs = ConfigurationSpace(
    name="myspace",
    seed=1234,
    space={
        "a": Float("a", bounds=(0.1, 1.5), distribution=Normal(1, 0.5)),
        "b": Integer("b", bounds=(1, 10_00), log=True, default=100),
        "c": Categorical("c", ["mouse", "cat", "dog"], weights=[2, 1, 1]),
    },
)
configs = cs.sample_configuration(2)
print(configs)
```

Maximum flexibility with conditionals, see the [user guide](./guide.md) for more information.

```python exec="True" result="python" source="material-block"
from ConfigSpace import Categorical, ConfigurationSpace, EqualsCondition, Float

cs = ConfigurationSpace(seed=1234)

c = Categorical("c1", items=["a", "b"])
f = Float("f1", bounds=(1.0, 10.0))

# A condition where `f` is only active if `c` is equal to `a` when sampled
cond = EqualsCondition(f, c, "a")

# Add them explicitly to the configuration space
cs.add([c, f])
cs.add(cond)

print(cs)
```


### Installation
ConfigSpace requires **Python 3.8** or higher
and can be installed directly from the Python Package Index (PyPI) using `pip`.

```bash
pip install ConfigSpace
```

### Citing ConfigSpace
```bibtex
 @article{
     title   = {BOAH: A Tool Suite for Multi-Fidelity Bayesian Optimization & Analysis of Hyperparameters},
     author  = {M. Lindauer and K. Eggensperger and M. Feurer and A. Biedenkapp and J. Marben and P. Müller and F. Hutter},
     journal = {arXiv:1908.06756 {[cs.LG]}},
     date    = {2019},
 }
```
