## Utilities

### Expression to Configspace

In some cases we may have (highly) complex conditions or forbidden expressions that are already denoted as a regular expression. In that case, `ConfigSpace` can automatically convert them into a `ConfigSpace` expression using the [`expression_to_configspace`][ConfigSpace.util.expression_to_configspace]`expression_to_configspace`. This function interprets the expression using the Python `Abstract Syntax Tree` parser and recursively converts it into the appropriate structure.

!!! note
    The converted expression is not added to ConfigSpace, only returned to the user.

!!! note
    If the expression contains illegal values, errors, or requires functionalities not available in `ConfigSpace`, appriopriate exceptions will be raised.

```python exec="True" result="python" source="tabbed-left"
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import expression_to_configspace

cs = ConfigurationSpace(
    {
        "a": (0, 10),    # Integer from 0 to 10
        "b": ["cat", "dog"],  # Categorical with choices "cat" and "dog"
        "c": (0.0, 1.0),  # Float from 0.0 to 1.0
    }
)
print(cs)

# Now we add a condition and forbidden using regular expressions
condition = "b != cat && c > 0.001"
condition = expression_to_configspace(condition, cs, target_hyperparameter=cs["a"])  # We have to specify the conditional HP seperately here as the final argument

print(condition)

forbidden = "a > 5 && c >= 0.94"
forbidden = expression_to_configspace(forbidden, cs)

print(forbidden)

cs.add(condition)
cs.add(forbidden)

print(cs)
```
