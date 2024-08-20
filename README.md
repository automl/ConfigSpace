# ConfigSpace

A simple Python module implementing a domain specific language to manage
configuration spaces for algorithm configuration and hyperparameter optimization tasks.  
Distributed under BSD 3-clause, see LICENSE except all files in the directory
ConfigSpace.nx, which are copied from the networkx package and licensed
under a BSD license.

The documentation can be found
at [https://automl.github.io/ConfigSpace/latest/](https://automl.github.io/ConfigSpace/latest/).
Further examples can be found in the [SMAC documentation](https://automl.github.io/SMAC3/main/examples/index.html).

## Minimum Example

```python
from ConfigSpace import ConfigurationSpace

cs = ConfigurationSpace(
    name="myspace",
    space={
        "a": (0.1, 1.5),  # UniformFloat
        "b": (2, 10),  # UniformInt
        "c": ["mouse", "cat", "dog"],  # Categorical
    },
)

configs = cs.sample_configuration(2)
```

## Citing the ConfigSpace

```bibtex
@article{
    title   = {BOAH: A Tool Suite for Multi-Fidelity Bayesian Optimization & Analysis of Hyperparameters},
    author  = {M. Lindauer and K. Eggensperger and M. Feurer and A. Biedenkapp and J. Marben and P. Müller and F. Hutter},
    journal = {arXiv:1908.06756 {[cs.LG]}},
    date    = {2019},
}
```
