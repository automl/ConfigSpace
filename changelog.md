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