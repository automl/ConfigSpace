## Utilities

### Expression to Configspace

In some cases we may have (highly) complex conditions or forbidden expressions that are already denoted as a regular expression. In that case, `ConfigSpace` can automatically convert them into a `ConfigSpace` expression using the [`parse_expression_from_string`][ConfigSpace.util.parse_expression_from_string]`parse_expression_from_string`. This function interprets the expression using the Python `Abstract Syntax Tree` parser and recursively converts it into the appropriate structure.

!!! note
    The converted expression is not added to ConfigSpace, only returned to the user.

!!! note
    If the expression contains illegal values, errors, or requires functionalities not available in `ConfigSpace`, appriopriate exceptions will be raised.

!!! note
    Expressions differentiate variables (Hyperparameter names) from constants (Categorical values) based on quotation marks; "a != b" implies hyperparameter a does not equal hyperparameter b, "a != 'b'" implies hyperparameter a does not equal categorical/ordinal value b.

#### Adding a condition

In this code example we show how you can add a hyperparameter condition to ConfigSpace from a string. Note that the conditional hyperparameter is specified as a seperate argument and is not part of the expression string!

```python exec="True" result="python" source="tabbed-left"
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import parse_expression_from_string

cs = ConfigurationSpace(
    {
        "a": (0, 10),    # Integer from 0 to 10
        "b": ["cat", "dog"],  # Categorical with choices "cat" and "dog"
        "c": (0.0, 1.0),  # Float from 0.0 to 1.0
    }
)
print(cs)

# Now we add a condition and forbidden using regular expressions
condition = "b != 'cat' && c > 0.001"
condition = parse_expression_from_string(condition, cs, conditional_hyperparameter=cs["a"])  # We have to specify the conditional HP seperately here as the final argument

print(condition)

cs.add(condition)

print(cs)
```

#### Adding a forbidden expression

In this example we add a forbidden expression to ConfigSpace from string. Note that the conditional hyperparameter remains unspecified; this leads to ConfigSpace interpreting the expression as a forbidden expression.

```python exec="True" result="python" source="tabbed-left"
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import parse_expression_from_string

cs = ConfigurationSpace(
    {
        "a": (0, 10),    # Integer from 0 to 10
        "b": ["cat", "dog"],  # Categorical with choices "cat" and "dog"
        "c": (0.0, 1.0),  # Float from 0.0 to 1.0
    }
)
print(cs)
forbidden = "a > 5 && c >= 0.94"
forbidden = parse_expression_from_string(forbidden, cs)

print(forbidden)

cs.add(forbidden)

print(cs)
```