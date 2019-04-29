# Version 0.4.10

* DOC: improved documentation and plenty of new docstrings.
* FIX #114: Checking categorical hyperparameters for uniqueness.
* MAINT #110: temporarily disable categorical value `None`
* MAINT #112: improved unit tests and compilation; new pep8 checks


# Version 0.4.9

* Fixes an issue where adding a new forbidden for an unknown hyperparameter 
  did not result in an immediate exception.
* Add a new argument `vector` to `util.deactivate_inactive_hyperparameters`
* Make the number of categories a public variable for categorical and
  ordinal hyperparameters

# Version 0.4.8

* Fixes an issue which made serialization of `ForbiddenInCondition` to json 
  fail.
* MAINT #101: Improved error message on setting illegal value in a 
  configuration.
* DOC #91: Added a documentation to automl.github.io/ConfigSpace

# Version 0.4.7

* Tests Python3.7.
* Fixes #87: better handling of Conjunctions when adding them to the 
  configuration space.
* MAINT: Improved type annotation in `util.py` which results in improved
  performance (due to better Cython optimization).
* MAINT: `util.get_one_exchange_neighborhood` now accepts two arguments 
  `num_neighbors` and `stdev` which govern the neighborhood creation behaviour 
  of several continuous hyperparameters.
* NEW #85: Add function to obtain active hyperparameters
* NEW #84: Add field for meta-data to the configuration space object.
* MAINT: json serialization now has an argument to control indentation

# Version 0.4.6

* Fixes a bug which caused a `KeyError` on the usage of tuples in `InCondition`.

# Version 0.4.5

* Stricter typechecking by using a new Cython features which automatically
  transfers type annotations into type checks at compilation time.
* New attribute of the ConfigSpace object: name
* Added JSON as a new, experimental serialization format for configuration
  spaces.
* Fixed a bug writing AND-conjunctions in the pcs writer.
* Fixed a lot of hash functions which broke during the conversion to Cython

# Version 0.4.4

* Fixes issue #49. The `Configuration` object no longer iterates over inactive
  hyperparameters which resulted in an unintuitive API.

# Version 0.4.3

* Fix a memory leak when repeatedly sampling a large amount of configurations.

# Version 0.4.2

* Fix a bug which caused a segfault when trying to sample zero configurations.

# Version 0.4.1

* Rewrite of major functions in Cython
* Attribute `default` of class Hyperparameter is renamed to `default_value`.
* Package `io` is renamed to `read_and_write`.

# Version 0.3.10
* Fix issue #56. The writer for the new pcs format can now write correct
  conjunctions of forbidden parameters.
* The class `Configuration` now raises an exception if trying to instantiate it
  with an illegal value for a hyperparameter

# Version 0.3.9

* Fix issue #53. Functionality for retrieving a one exchange neighborhood does
  no longer create illegal configurations when hyperparameters have extreme
  ranges.
* New functionality `__setitem__` for `Configuration`. Allows dictionary syntax
  to change the value of one hyperparameter in a configuration.

# Version 0.3.8

* Fix issue #25. Parents and children are now sorted topologically in the
  underlying datastructure.
* The reader and writer for the new pcs format can now handle Constants.
* Speed improvements for one exchange neighborhood

# Version 0.3.7

* Faster checking of valid configurations
* Fixes a bug in sampling configurations for SatenStein
* Fixes a bug getting neighbors from OrdinalHyperparameter
* Fixes a bug getting neighbors from a UniformIntegerHyperparameter
* Utility function to deactivate inactive, but specified hyperparameters
* Fixes a bug in retrieving the one exchange neighborhood, i.e. it could not
  change the value of parent parameters

# Version 0.3.6

* Minor speed improvements when checking forbidden clauses

# Version 0.3.5

* Even more speed improvements for highly conditional configurations
* Speed improvements for retrieving neighbor configurations

# Version 0.3.4

* Further improve sampling speed for highly conditional configuration spaces

# Version 0.3.3

* Improve sampling speed for highly conditional configuration spaces

# Version 0.3.2

* FIX: do not store `None` when calling `populate_values` on a Configuration
* FIX #26: raise Exception if argument has the wrong type.
* FIX #25: a bug related to sorting hyperparameters when reading them from a
  PCS files.

# Version 0.3.1

* MAINT: fix endless loop in `get_one_exchange_neighborhood`.

# Version 0.3

* MAINT: improve speed of `get_one_exchange_neighborhood`. This changes the
  return value of the function from a list to a generator.

# Version 0.2.3

* FIX: allow installation via `python setup.py install`
* MAINT: drop python2.7 support, add python3.6 support
* MAINT: improve performance of `add_configuration_space`
* FIX: a bug in `add_configuration_space` which changed the names of the
  hyperparameters of a child configuration space
* MAINT: new helper functions `add_hyperparameters` and `add_conditions`.

# Version 0.2.2

* ADD: two convenience functions to improve working with HPOlib2
* MAINT: default seed of ConfigurationSpace is now None instead of 1. This makes
  the package behave like other scientific programming packages.
* FIX: improve bounds checking for log-space hyperparameters
* FIX: version information is now in a separate file.
* MAINT: improve sampling speed by breaking the sampling loop earlier when
  possible

# Version 0.2.1

* FIX: bug which changed order of hyperparameters when adding new 
  hyperparameter. This was non-deterministic due to the use of dict instead 
  of OrderedDict.
* FIX: compare configurations with == instead of numpy.allclose.
* FIX: issue 2, syntax error no longer present during installation
* FIX: json serialization of configurations and their hyperparameters can now
       be deserialized by json and still compare equal 

# Version 0.2

* FIX: bug which made integer values have different float values in the 
  underlying vector representation.
* FIX: bug which could make two configuration spaces compare unequal due to 
  the use of defaultdict
* FEATURE: new feature add_configuration_space, which allows to add a whole 
  configuration space into an existing configuration space
* FEATURE: python3.5 support
* FIX: add function get_parent() to Conjunctions (issue #1)
* Added this Changelog.
