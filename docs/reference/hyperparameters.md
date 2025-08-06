## Hyperparameters
ConfigSpace contains three ways to define hyperparameters, each offering more customizabilty than the last.
We first demonstrate the three different ways to define hyperparameters, **inferred**, **simple**, and **direct**.

Later, we will show how to directly use the hyperparameters if required, however this is mostly for library developers
using ConfigSpace as a dependency.

---

* Directly when constructing the [`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace] object,
we call these **inferred** hyperparameters. **Use these if you have a simple search space or are doing rapid prototyping.**
```python exec="True" result="python" source="tabbed-left"
from ConfigSpace import ConfigurationSpace

cs = ConfigurationSpace(
    {
        "a": (0, 10),    # Integer from 0 to 10
        "b": ["cat", "dog"],  # Categorical with choices "cat" and "dog"
        "c": (0.0, 1.0),  # Float from 0.0 to 1.0
    }
)
print(cs)
```
* Using functions to create them for you. We call these **simple** hyperparameters and they should
satisfy most use cases. **Use these if you just want to create a searchspace required by another library.**
```python exec="True" result="python" source="tabbed-left"
from ConfigSpace import ConfigurationSpace, Integer, Categorical, Float, Normal

cs = ConfigurationSpace(
    {
        "a": Integer("a", (0, 10), log=False),    # Integer from 0 to 10
        "b": Categorical("b", ["cat", "dog"], ordered=True),  # Ordered categorical with choices "cat" and "dog"
        "c": Float("c", (1e-5, 1e2), log=True),  # Float from 0.0 to 1.0, log scaled
        "d": Float("d", (10, 20), distribution=Normal(15, 2)),  # Float from 10 to 20, normal distribution
    }
)
print(cs)
```
* Using the types directly. We call these **direct** hyperparameters. These are the real types used
throughout ConfigSpace and offer the most customizability.
**Use these if you are building a library the utilizes ConfigSpace.**
```python exec="True" result="python" source="tabbed-left"
from ConfigSpace import (
    ConfigurationSpace,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    NormalFloatHyperparameter,
    OrdinalHyperparameter
)

cs = ConfigurationSpace(
    {
        "a": UniformIntegerHyperparameter("a", lower=0, upper=10, log=False),    # Integer from 0 to 10
        "b": CategoricalHyperparameter("b", choices=["cat", "dog"], default_value="dog"),  # Ordered categorical with choices "cat" and "dog"
        "c": UniformFloatHyperparameter("c", lower=1e-5, upper=1e2, log=True),  # Float from 0.0 to 1.0, log scaled
        "d": NormalFloatHyperparameter("d", lower=10, upper=20, mu=15, sigma=2),  # Float from 10 to 20, normal distribution
        "e": OrdinalHyperparameter("e", sequence=["s", "m", "l"], default_value="s"),  # Ordered categorical
    }
)
print(cs)
```

## Inferred Hyperparameters
When creating hyperparameters directly in the [`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace] object,
you can create three different kinds of hyperparameters. This can be useful for simple testing or quick prototyping.

```python exec="True" result="python" source="material-block"
from ConfigSpace import ConfigurationSpace

cs = ConfigurationSpace(
    {
        "a": (0, 10),    # Integer from 0 to 10
        "b": ["cat", "dog"],  # Categorical with choices "cat" and "dog"
        "c": (0.0, 1.0)  # Float from 0.0 to 1.0
    }
)
print(cs)
```

The rules are as follows:

* If the value is a tuple, with `int`s, then it is considered an integer hyperparameter with a **uniform** distribution.
* If the value is a tuple, with `float`s, then it is considered a float hyperparameter with a **uniform** distribution.
* If the value is a list, then each element is considered a choice for a categorical hyperparameter, with no inherit
order.

!!! warning "Mixed types in a tuple"

    If you use an `int` and a `float` in the same tuple, it will infer the type using the **first** element.
    For example, `(0, 1.0)` will be inferred as an integer hyperparameter, while `(1.0, 10)` will
    be inferred as a float hyperparameter.

## Simple Hyperparameters
Most of the time, you just require the ability to create hyperparameters and pass them to some other library.
To make this is as possible, we parametrize building the various **direct** hyperparameters that exist.

### Integer
The [`Integer()`][ConfigSpace.api.types.integer.Integer] **function** samples an `int` uniformly
from the range `(lower, upper)`, with options to define them as being on a `log=` scale or
that you prefer the sampling to be done under a different `distribution=`.
```python exec="True" result="python" source="material-block"
from ConfigSpace import Integer, ConfigurationSpace, Uniform, Normal

cs = ConfigurationSpace()

cs.add(
    Integer("a", (0, 10), log=False),
    Integer("b", (0, 10), log=False, distribution=Uniform(), default=5),
    Integer("c", (1, 1000), log=True, distribution=Normal(mu=200, sigma=200)),
)
print(cs)
print(cs["a"].sample_value(size=5))
```

Please check out the [distributions API][ConfigSpace.api.distributions.Distribution] for more information on the available
distributions.

!!! warning "Not a type"
    Please be aware that `Integer` is a convenience **function** that returns
    one of the **direct** hyperparameter classes. Please see the [direct hyperparameters](#direct-hyperparameters) if you need to
    access the underlying classes.

### Float
The [`Float()`][ConfigSpace.api.types.float.Float] **function** samples a `float` uniformly from the range `(lower, upper)`,
with options to define them as being on a `log=` scale or
that you prefer the sampling to be done under a different `distribution=`.
```python exec="True" result="python" source="material-block"
from ConfigSpace import Float, ConfigurationSpace, Uniform, Normal

cs = ConfigurationSpace()

cs.add(
    Float("a", (0, 10), log=False),
    Float("b", (0, 10), log=False, distribution=Uniform(), default=5),
    Float("c", (1, 1000), log=True, distribution=Normal(mu=200, sigma=200)),
)
print(cs)
print(cs["a"].sample_value(size=5))
```

Please check out the [distributions API][ConfigSpace.api.distributions.Distribution] for more information on the available
distributions.

!!! warning "Not a type"
    Please be aware that `Float` is a convenience **function** that returns
    one of the **direct** hyperparameter classes. Please see the [direct hyperparameters](#direct-hyperparameters) if you need to
    access the underlying classes.

### Categorical
The [`Categorical()`][ConfigSpace.api.types.categorical.Categorical] **function** samples a value from the `choices=` provided.
optionally giving them `weights=`, influencing the distribution of the sampling. You may also define them
as `ordered=` if there is an inherent order to the choices.
```python exec="True" result="python" source="material-block"
from ConfigSpace import Categorical, ConfigurationSpace

cs = ConfigurationSpace()

cs.add(
    Categorical("a", ["cat", "dog", "mouse"], default="dog"),
    Categorical("b", ["small", "medium", "large"], ordered=True, default="medium"),
    Categorical("c", [True, False], weights=[0.2, 0.8]),
)
print(cs)
print(cs["c"].sample_value(size=5))
```

!!! warning "Not a type"
    Please be aware that `Categorical` is a convenience **function** that returns
    one of the **direct** hyperparameter classes. Please see the [direct hyperparameters](#direct-hyperparameters) if you need to
    access the underlying classes.

## Direct Hyperparameters
All of the methods for constructing hyperparameters above will result in one of the following types.

**Integer Hyperparameter**

* [`UniformIntegerHyperparameter`][ConfigSpace.hyperparameters.UniformIntegerHyperparameter]
* [`NormalIntegerHyperparameter`][ConfigSpace.hyperparameters.NormalIntegerHyperparameter]
* [`BetaIntegerHyperparameter`][ConfigSpace.hyperparameters.BetaIntegerHyperparameter]

**Float Hyperparameter**

* [`UniformFloatHyperparameter`][ConfigSpace.hyperparameters.UniformFloatHyperparameter]
* [`NormalFloatHyperparameter`][ConfigSpace.hyperparameters.NormalFloatHyperparameter]
* [`BetaFloatHyperparameter`][ConfigSpace.hyperparameters.BetaFloatHyperparameter]

**Categorical Hyperparameter**

* [`CategoricalHyperparameter`][ConfigSpace.hyperparameters.CategoricalHyperparameter]
* [`OrdinalHyperparameter`][ConfigSpace.hyperparameters.OrdinalHyperparameter]

You can utilize these types in your code as required for `isinstance` checks or allow your own code to create
them as required. If developing a library, please see below to understand a bit more about the [structure of a Hyperparameter](#structure-of-a-hyperparameter).

## Structure of a Hyperparameter
All hyperparameters inherit from the [`Hyperparameter`][ConfigSpace.hyperparameters.Hyperparameter] base class, with
two important components to consider:

1. **vectorized space**: This defines some underlying numeric range along with a procedure to sample from it.
2. **value space**: These are the values that are given back to the user, *e.g.* `["cat", "dog"]`.

What makes a hyperparameter the hyperparameter it is then:

1. How we sample from the vectorized space, defined by a [`Distribution`][ConfigSpace.hyperparameters.distributions.Distribution].
2. How we map to and from the vectorized space to the value space, defined by a [`Transformer`][ConfigSpace.hyperparameters.hp_components.Transformer].

??? tip "Why a vectorized space?"

    Most optimizers requires some kind of bounds and a pure numeric space from which to optimize over, i.e.
    it would be hard to optimize over a hyperparameter space of `["cat", "dog"]` directly.

    This also lets use share implementation details and optimization across various kinds of hyperparameters
    if they share the same underlying vectorized space.


!!! example "CategoricalHyperparameter"

    Inside of the `__init__` method of a `CategoricalHyperparameter`, you will find something along the lines
    of the following:

    ```python
    class CategoricalHyperparameter(Hyperparameter):
        def __init__(...):

            # ...
            super().__init__(
                vector_dist=UniformIntegerDistribution(size=len(choices)),
                transformer=TransformerSeq(seq=choices),
                ...
            )
    ```

    What this is showing is that we will use
    [`UniformIntegerDistribution`][ConfigSpace.hyperparameters.distributions.UniformIntegerDistribution], which
    samples integers uniformly from `0` to `len(choices) - 1`, and then we use a
    [`TransformerSeq`][ConfigSpace.hyperparameters.hp_components.TransformerSeq] to map these integers to the
    corresponding choices provided by the users.

    Internally in `ConfigSpace`, we will primarily work with the vectorized space for efficiency purposes,
    but when providing values back to the user, either from the
    [`Configuration`][ConfigSpace.configuration.Configuration] or other means, we will use the `transformer=` to
    map the vectorized space back to the value space.

Using just these two components alone, we can provide the following functionality from the [`Hyperparameter`][ConfigSpace.hyperparameters.Hyperparameter] base class:

* [`sample_vector()`][ConfigSpace.hyperparameters.Hyperparameter.sample_vector]: Samples a vectorized value
* [`sample_value()`][ConfigSpace.hyperparameters.Hyperparameter.sample_value]:
Samples a vectorized value and transforms it back to the value space.
* [`to_value()`][ConfigSpace.hyperparameters.Hyperparameter.to_value]: Transforms a vectorized value to the value space.
* [`to_vector()`][ConfigSpace.hyperparameters.Hyperparameter.to_vector]: Transforms a value space value to the vectorized space.
* [`pdf_vector()`][ConfigSpace.hyperparameters.Hyperparameter.pdf_vector]: The probability density function of a vectorized value.
* [`pdf_values()`][ConfigSpace.hyperparameters.Hyperparameter.pdf_values]: The probability density function of a value,
  by transforming it to the vectorized space and then calculating the pdf.
* [`legal_value()`][ConfigSpace.hyperparameters.Hyperparameter.legal_value]: Check if a value is legal.
* [`legal_vector()`][ConfigSpace.hyperparameters.Hyperparameter.legal_vector]: Check if a vectorized value is legal.
* [`.lower_vectorized`][ConfigSpace.hyperparameters.Hyperparameter.lower_vectorized]: The lower bound in vectorized space.
* [`.upper_vectorized`][ConfigSpace.hyperparameters.Hyperparameter.upper_vectorized]: The upper bound in vectorized space.


Please note that most of these methods support individual values or numpy arrays of values, either as input or output.
Refer to the [API documentation][ConfigSpace.hyperparameters.Hyperparameter] for more information on the available methods.

### Neighborhoods
One utility `ConfigSpace` provides to library developers is the ability to define a neighbourhood around a value.
This is often important for optimizers who require a neighbourhood to explore around a particular configuration or value.

A class inheriting from [`Hyperparameter`][ConfigSpace.hyperparameters.Hyperparameter] must also provide
a [`Neighborhood`][ConfigSpace.hyperparameters.hp_components.Neighborhood], which is something that can be called
with a vectorized value and provide values around that point.

The expected signature is rather straight forward, given a `vector` value and a `n` number of samples to return,
it should return a numpy array of **up to** `n` **unique** samples.

```python
def __call__(
    self,
    vector: np.float64,
    n: int,
    *,
    std: float | None = None,
    seed: RandomState | None = None,
) -> npt.NDArray[np.float64]: ...
```

They must also provide a `_neighbourhood_size`, either `np.inf` if
unbounded or a method that returns the maximum possible neighbors that are possible around a given value.

By subclasses providing this through the `__init__` method, we can then provide the following functionality:

* [`get_num_neighbors()`][ConfigSpace.hyperparameters.Hyperparameter.get_num_neighbors]: Get the number of neighbours around a value.
* [`neighbors_vectorized()`][ConfigSpace.hyperparameters.Hyperparameter.neighbors_vectorized]: Get neighbors around a
point in vectorized space.
* [`neighbors_values()`][ConfigSpace.hyperparameters.Hyperparameter.neighbors_values]: Get neighbors around a
point in value space.

Please refer to the source code definition of existing hyperparameters for more information on how to implement this.
Most of this is defined in the `__init__` method of the hyperparameter.

### Example: Implementing the BetaIntegerHyperparameter
For implementing your own hyperparameter type, it's useful to look at a case study of implementing an existing
hyperparameter and to see what functionality can be re-used in the library.
Please refer to this article on Wikipedia for more information on the [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
for more on the distribution.

!!! tip

    Be aware the `ConfigSpace` is heavily optimized towards performance using profiling, and where possible, it
    would be good to use pre-existing components to build your hyperparameter. You'd be surprised how much milliseconds
    add up when sampling thousands of configurations both globally and using neighborhoods.

---

#### Defining the BetaIntegerHyperparameter
First, we need to define the `__init__` method of the `BetaIntegerHyperparameter`, where we're going to
need the following for a [`BetaIntegerHyperparameter`][ConfigSpace.hyperparameters.integer_hyperparameter.IntegerHyperparameter]:

* `name=`: The name of the hyperparameter, required for all kinds of hyperparameters
* `lower=`, `upper=`: The bounds the user would like in value space, i.e. `(1, 5)`
* `default_value=`: The default value of the hyperparameter.
* `alpha=`, `beta=`: The parameters of the beta distribution itself.

#### Vectorized Space
For our purposes, we will mostly rely on scipys `beta` distribution to sample from a **vectorized space**.
Here is how you would sample from it in `scipy:`

```python exec="True" source="material-block" result="python"
from scipy.stats import beta as spbeta

alpha, beta = 3, 2
beta_rv = spbeta(alpha, beta)
samples = beta_rv.rvs(size=5)
print(samples)
```

The problem however is that scipy only offers a contiuous version of this distribution, however we
need to sample integers. To solve this, we will use the
[`DiscretizedContinuousScipyDistribution`][ConfigSpace.hyperparameters.distributions.DiscretizedContinuousScipyDistribution]

```python exec="True" source="material-block" result="python"
import numpy as np
from scipy.stats import beta as spbeta
from ConfigSpace.hyperparameters.distributions import DiscretizedContinuousScipyDistribution

# As before
alpha, beta = 3, 2
beta_rv = spbeta(alpha, beta)


# Declare our value space bounds and how many discrete steps there
# are between them.
value_bounds = (1, 5)
discrete_steps = value_bounds[1] - value_bounds[0] + 1

# Creates a distribution which can discretize the continuous range
# into `size` number of steps, such that we can map the discretized
# vector values into integers in the range that was requested.

# Where possible, it is usually preferable to have vectorized bounds from (0, 1)
# We also require all vectorized values to be np.float64, even if they represent integers
vector_distribution = DiscretizedContinuousScipyDistribution(
    rv=beta_rv,
    steps=discrete_steps,
    lower_vectorized=np.float64(0),
    upper_vectorized=np.float64(1),
)
print(vector_distribution.sample_vector(n=5))
```

!!! tip

    To support `scipy` distributions we implement various optimized [`Distribution`][ConfigSpace.hyperparameters.distributions.Distribution]s

    * [`ScipyContinuousDistribution`][ConfigSpace.hyperparameters.distributions.ScipyContinuousDistribution]:
    Samples from a continuous scipy distribution.
    * [`ScipyDiscreteDistribution`][ConfigSpace.hyperparameters.distributions.ScipyDiscreteDistribution]:
    Samples from a discrete scipy distribution.
    * [`DiscretizedContinuousScipyDistribution`][ConfigSpace.hyperparameters.distributions.DiscretizedContinuousScipyDistribution]:
    Samples from a continuous scipy distribution, but discretizes the output efficiently.

    The also often provide a `neighborhood` method to sample around a point that can be used, as well as a
    `pdf` method, which can do so efficiently in both memory and time.
    Please refer to their individual API documentation for more information on how to create and use them.

### Transforming from Vectorized Space to Value Space
To convert from the vectorized space to the value space, we will need to implement a
[`Transformer`][ConfigSpace.hyperparameters.hp_components.Transformer] that can map the vectorized space to the
value space, e.g. `(0.0, 1.0)` to `(1, 5)`.

To do this, we provide a convenience class called [`UnitScaler`][ConfigSpace.hyperparameters.hp_components.UnitScaler],
which also allows for a `log=` scale transformation.

```python exec="True" source="material-block" result="python"
import numpy as np
from scipy.stats import beta as spbeta
from ConfigSpace.hyperparameters.distributions import DiscretizedContinuousScipyDistribution
from ConfigSpace.hyperparameters.hp_components import UnitScaler

# Define the distribution sampler
alpha, beta = 3, 2
vector_distribution = DiscretizedContinuousScipyDistribution(
    rv=spbeta(alpha, beta),
    steps=5,
    lower_vectorized=np.float64(0),
    upper_vectorized=np.float64(1),
)
vector_samples = vector_distribution.sample_vector(n=5)
print(vector_samples)

# Define the transformer from the samplers range to the range we care about
transformer = UnitScaler(
    lower_value=np.int64(1),
    upper_value=np.int64(5),
    dtype=np.int64,  # We want integers in value space
    log=False,
)
integer_values = transformer.to_value(vector_samples)
print(integer_values)

back_to_vector = transformer.to_vector(integer_values)
print(back_to_vector)
```

You are of course free to implement your own [`Transformer`][ConfigSpace.hyperparameters.hp_components.Transformer]
if you require a more complex transformation, however where possible, the
[`UnitScaler`][ConfigSpace.hyperparameters.hp_components.UnitScaler] is preffered as it handles some edge cases
and performs some optimized routines while remaining fully within the expected API.

### Creating the BetaIntegerHyperparameter class
Below we provide what is essentially the entire `BetaIntegerHyperparameter` in `ConfigSpace`.
Nothing else is required and you can hotswap this out with other kinds of distributions if you require
new kinds of `Hyperparameters`. Most libraries using `ConfigSpace` who do not require explicit kinds
of hyperparameters should be able to utilize these.


!!! note

    We use dataclasses in ConfigSpace, which means that inherting classes should also be a
    dataclass. This is not a strict requirement, but it is recommended to keep the API consistent.

```python
from typing import TypeAlias, Union, Mapping, Hashable, Any
import numpy as np
from scipy.stats import beta as spbeta

from ConfigSpace.hyperparameters import IntegerHyperparameter
from ConfigSpace.hyperparameters.distributions import DiscretizedContinuousScipyDistribution
from ConfigSpace.hyperparameters.hp_components import UnitScaler
from ConfigSpace.functional import is_close_to_integer_single

i64 = np.int64
f64 = np.float64

# We allow any kind of number to be used, we will cast as required
Number: TypeAlias = Union[int, float, np.number]

@dataclass(init=False)  # We provide our own init
class BetaIntegerHyperparamter(IntegerHyperparameter):
    ORDERABLE: ClassVar[bool] = True  # Let ConfigSpace know there is an order to the values

    alpha: float
    """Some docstring decsription of this attribute."""

    beta: float
    lower: float
    upper: float
    log: bool
    name: str
    default_value: float
    meta: Mapping[Hashable, Any] | None

    size: float = field(init=False)  # This will be calculated

    def __init__(
        self,
        name: str,
        alpha: Number,
        beta: Number,
        lower: Number,
        upper: Number,
        default_value: Number | None = None,
        log: bool = False,
        meta: Mapping[Hashable, Any] | None = None,
    ) -> None:
        if (alpha < 1) or (beta < 1):
            raise ValueError(
                "Please provide values of alpha and beta larger than or equal to"
                "1 so that the probability density is finite.",
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lower = int(np.rint(lower))
        self.upper = int(np.rint(upper))
        self.log = bool(log)

        # Create the transformer
        try:
            scaler = UnitScaler(i64(self.lower), i64(self.upper), log=log, dtype=i64)
        except ValueError as e:
            raise ValueError(f"Hyperparameter '{name}' has illegal settings") from e


        if default_value is None:
            # Get the mode of the distribution for setting a default
            if (self.alpha > 1) or (self.beta > 1):
                vectorized_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)
            else:
                # If both alpha and beta are 1, we have a uniform distribution.
                vectorized_mode = 0.5

            _default_value = np.rint(
                scaler.to_value(np.array([vectorized_mode]))[0],
            ).astype(i64)
        else:
            if not is_close_to_integer_single(default_value):
                raise TypeError(
                    f"`default_value` for hyperparameter '{name}' must be an integer."
                    f" Got '{type(default_value).__name__}' for {default_value=}.",
                )

            _default_value = np.rint(default_value).astype(i64)

        size = int(self.upper - self.lower + 1)
        vector_dist = DiscretizedContinuousScipyDistribution(
            rv=spbeta(self.alpha, self.beta),  # type: ignore
            steps=size,
            lower_vectorized=f64(0.0),
            upper_vectorized=f64(1.0),
        )

        super().__init__(
            name=name,
            size=size,
            default_value=_default_value,
            meta=meta,
            transformer=scaler,
            vector_dist=vector_dist,
            neighborhood=vector_dist.neighborhood,
            # Tell ConfigSpace we expect an `int` when giving back a single value
            # For a np.ndarray of values, this will be `np.int64`
            value_cast=int,
            # This method comes from the IntegerHyperparameter
            # you can implement this you self if you'd like
            neighborhood_size=self._integer_neighborhood_size,
        )
```
