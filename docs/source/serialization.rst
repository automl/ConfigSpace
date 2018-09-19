Serialization
=============

| Sometimes, it can be useful to serialize the *configuration space*.
| This can be achieved by using the classes **ConfigSpace.read_and_write.pcs_new** or **ConfigSpace.read_and_write.json**.

This example shows how to write and read a configuration space to *pcs* - file::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    from ConfigSpace.read_and_write import pcs_new

    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))

    # Store the configuration space to file configspace as pcs-file
    with open('configspace.pcs', 'w') as fh:
        fh.write(pcs_new.write(cs))

    # Read the configuration space from file
    with open ('configspace.pcs'), 'r') as fh:
        restored_conf = pcs_new.read(fh)

