## Serialization

ConfigSpaces overs two primary methods of serialization, namely `json` and `yaml`.
Serializing is straight forward and can be done using the methods
[`configspace.to_json()`][ConfigSpace.configuration_space.ConfigurationSpace.to_json]
and [`configspace.to_yaml()`][ConfigSpace.configuration_space.ConfigurationSpace.to_yaml].
To deserialize, you can call the corresponding classmethods
[`ConfigurationSpace.from_json()`][ConfigSpace.configuration_space.ConfigurationSpace.from_json]
and [`ConfigurationSpace.from_yaml()`][ConfigSpace.configuration_space.ConfigurationSpace.from_yaml].

```python
from ConfigSpace import ConfigurationSpace
cs = ConfigurationSpace({"a": (0, 10), "b": ["cat", "dog"]})
cs.to_json("configspace.json")
cs = ConfigurationSpace.from_json("configspace.json")

cs.to_yaml("configspace.yaml")
cs = ConfigurationSpace.from_yaml("configspace.yaml")
```

### Plain type dict
We also support exporting the configuration space as a dictionary with plain simple python types.
This allows for easy serialization to other formats the support dictionary formats, for example, `toml`.

This is provided through [`to_serialized_dict()`][ConfigSpace.configuration_space.ConfigurationSpace.to_serialized_dict]
and [`from_serialized_dict()`][ConfigSpace.configuration_space.ConfigurationSpace.from_serialized_dict].

### Custom Encoding and Decoding
To support custom hyperparameters or various other purposes, we allow you to include custom methods
for encoding and decoding, based on the type encountered.

#### Encoding
For example, all serializing methods accept an `encoders=` parameter, which is a dictionary of
`type: (type_name_as_str, encoder)` pairs.

For example:
```python exec="True" source="material-block" result="python"
from typing import Any, Callable
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter

cs = ConfigurationSpace({"a": ["cat", "dog"]})

def my_custom_encoder(
    hp: CategoricalHyperparameter,
    encoder: Callable[[Any], dict],
) -> dict:
    return {
        "name": hp.name,
        "choices": [f"!{c}!" for c in hp.choices],
    }

without_custom_encoder = cs.to_serialized_dict()
with_custom_encoder = cs.to_serialized_dict(
    # Overrides the default encoder for CategoricalHyperparameters
    encoders={
        CategoricalHyperparameter: ("my_category", my_custom_encoder),
    }
)
print(without_custom_encoder)
print("--------")
print(with_custom_encoder)
```

The second argument to the encoder is a callable that can be used to encode any nested types,
deferring to the encoder for that type. This is useful for types such as conditionals or forbidden clauses,
which often contain hyperparameters within them.

#### Decoding
Decoding is quite similar with a few minor differences to specification.

```python
def my_decoder(
    # The dictionary that needs to be decoded into a type
    d: dict[str, Any],
    # The current state of the ConfigurationSpace being decoded
    space: ConfigurationSpace,
    # A callable to offload decoding of nested types
    decoder: Callable
) -> Any:
    ...
```

As things such as conditions and forbidden clauses rely on hyperparmeters to be decoded first,
you need to specify what _kind_ of thing your decoder will operate on,
namely `"hyperparameters"`, `"conditions"` or `"forbiddens"`.

```python
my_configspace = ConfigurationSpace.from_serialized_dict(
    my_serialized_dict,
    # Overrides the default decoder for CategoricalHyperparameters
    decoders={
        "hyperparameters": {
            "my_category": my_decoder,
        },
        "conditions": {},  # No need to specify, just here for completeness
        "forbiddens": {},  # No need to specify, just here for completeness
    }
)
```


### PCS
A common format for serialization of configuration spaces used to be the `PCS` format.
For those familiar with this, we still provide this using
[`ConfigSpace.read_and_write.pcs_new.read()`][ConfigSpace.read_and_write.pcs_new.read]
and [`ConfigSpace.read_and_write.pcs_new.write()`][ConfigSpace.read_and_write.pcs_new.write].

However this format is no longer directly supported and will issue deprecation warnings.
Going forward, we recommend using `json` or `yaml` where possible, as newer version of
ConfigSpace may include features not supported by the `PCS` format.
