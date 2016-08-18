# Version 0.2

* FIX: bug which made integer values have different float values in the 
  underlying vector representation.
* FIX: bug which could make two configuration spaces compare unequal due to 
  the use of defaultdict
* FEATURE: new feature add_configuration_space, which allows to add a whole 
  configuration space into an existing configuration space
* FEATURE: python3.5 support
* FIX: add function get_parent() to Conjunctions (issue #1)