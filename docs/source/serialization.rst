Serialization example
=====================

| Sometimes, it can be useful to serialize the *configuration space*.
| This can be achieved by using the classes **ConfigSpace.read_and_write.pcs**,
  **ConfigSpace.read_and_write.pcs_new** or **ConfigSpace.read_and_write.json**.

1. Serialization to JSON:
-------------------------

This example shows how to write and read a configuration space to *json* - file::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    from ConfigSpace.read_and_write import json

    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))

    # Store the configuration space to file as json-file
    with open('configspace.json', 'w') as fh:
        fh.write(json.write(cs))

    # Read the configuration space from file
    with open('configspace.json', 'r') as fh:
        json_string = fh.read()
        restored_conf = json.read(json_string)

2. Serialization with pcs-new
-----------------------------
To write to pcs is similar to the example above.::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    from ConfigSpace.read_and_write import pcs_new

    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))

    # Store the configuration space to file configspace as pcs-file
    with open('configspace.pcs', 'w') as fh:
        fh.write(pcs_new.write(cs))

    # Read the configuration space from file
    with open('configspace.pcs', 'r') as fh:
        restored_conf = pcs_new.read(fh)


3. Serialization with pcs
-------------------------

::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    from ConfigSpace.read_and_write import pcs

    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))

    # Store the configuration space to file configspace as pcs-file
    with open('configspace.pcs', 'w') as fh:
        fh.write(pcs.write(cs))

    # Read the configuration space from file
    with open('configspace.pcs', 'r') as fh:
        restored_conf = pcs.read(fh)


