# Contributing guidelines
Contributions are welcome! For bugs, questions and feature requests, please
[open an issue](https://github.com/automl/ConfigSpace/issues). If you already have a fix
or a feature in mind, feel free to open a pull request directly, or open an issue first
to discuss larger changes.

## Installing for contributors
Install ConfigSpace in editable mode with the `dev` extra, 
which includes everything needed for testing, linting and the docs:

```bash
cd ConfigSpace
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### pre-commit hooks
The repository uses [pre-commit](https://pre-commit.com) to run `ruff` (linting and
formatting) and a few general checks.
Install the git hook, so the checks run automatically on every `git commit`:

```bash
pre-commit install
```

To run the hooks by hand:

```bash
pre-commit run --all-files                 # whole repo [use sparingly]
pre-commit run --files path/a.py path/b.py # specific files
pre-commit run ruff-format --all-files     # a single hook
```

Perhaps more useful is to run them only on the files changed by your current branch:

```bash
pre-commit run --files $(git diff --name-only main...HEAD)
```

If a hook reports `Failed` with "files were modified by this hook", it has already fixed
the files for you. Review the changes, `git add` them and commit again.

You can also run `ruff` directly, e.g. on the source files:

```bash
ruff check --fix src/ConfigSpace
ruff format src/ConfigSpace
```

## Tests
Tests live in the `test/` directory and use `pytest`. Run them from the repository root:

```bash
pytest                                   # all tests (or: python -m pytest)
pytest -x                                # stop at the first failure
pytest test/test_configuration_space.py  # a single file
```

**New features and bug fixes should come with tests**. For a bug fix, you can add a test that fails
without the fix.

## Documentation
The documentation is built with [mkdocs](https://www.mkdocs.org) from the `docs/`
directory, and the API reference is generated from the docstrings. Please document new
public functions and classes with docstrings, and update the relevant pages in `docs/`
when behaviour changes.

Build the docs locally the same way the CI does:

```bash
mkdocs build --clean --strict
```

or preview them with live reloading at `http://127.0.0.1:8000`:

```bash
mkdocs serve
```

## Pull requests
Pull requests are expected to pass all checks **before they are reviewed**. The CI runs:

* the pre-commit hooks (`pre-commit run --all-files`),
* the test suite on Python 3.9 to 3.14, on Linux, macOS and Windows,
* the strict docs build (`mkdocs build --clean --strict`).

When opening a pull request, please:

1. Make sure the tests, the pre-commit hooks and the docs build pass locally.
2. Keep the pull request focused on one change, and avoid unrelated reformatting.
3. Describe what the change does and why, and link the related issue if there is one.

## Tips on ConfigSpace architecture
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
