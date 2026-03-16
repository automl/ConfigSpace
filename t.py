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