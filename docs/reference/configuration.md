## Configuration
A [`Configuration`][ConfigSpace.configuration.Configuration] is an dict-like object,
going from the name of selected hyperparameters to the values.

```python exec="True" result="python"
cs = ConfigurationSpace(
    {
        "a": (0, 10),
        "b": ["cat", "dog"],
    }
)
configuration = cs.sample_configuration()

for name, value in configuration.items():
    print(f"{name}: {value}")

print(configuration["a"])
```

Underneath the hood, there is some **vectorized** representation of the configuration,
which in this case may look like `np.array([0.32, 1])` which stands for `{"a": 3.2, "b": "dog"}`.
This vectorized representation can be useful for optimizer numerical optimization algorithms.
You can access it with [`configuration.get_array()`][ConfigSpace.configuration.Configuration].

!!! tip

    All `Configuration` have a reference to the underlying
    [`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace]
    which can be access with [`Configuration.config_space`][ConfigSpace.configuration.Configuration.config_space].


For more, please check out the API documentation for [`Configuration`][ConfigSpace.configuration.Configuration].
