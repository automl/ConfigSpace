# Design principles
ConfigSpace relies on the vectorization of search spaces to efficiently sample from
them. A few things are worth knowing before changing the internals:

* **Every hyperparameter has a value space and a vectorized space.** The value space
  holds what users see (e.g. `"adam"`, `0.01`), while the vectorized space is a numeric
  `float64` representation. A hyperparameter combines a
  [`Distribution`][ConfigSpace.hyperparameters.distributions.Distribution], which samples
  in the vectorized space, with a
  [`Transformer`][ConfigSpace.hyperparameters.hp_components.Transformer], which maps
  between the two spaces (`to_vector()` / `to_value()`). Internally, ConfigSpace works
  almost exclusively with vectors, and a `Configuration` is stored as one vector.
  See [Hyperparameters](reference/hyperparameters.md) for details.
* **The `ConfigurationSpace` is backed by a DAG** (`ConfigSpace/_condition_tree.py`).
  The DAG orders hyperparameters by their conditions and caches derived state, such as
  normalized default values, vectorized condition checks and merged forbidden checks.
