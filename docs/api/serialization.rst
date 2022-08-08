.. _Serialization:

Serialization
=============

ConfigSpace offers *json*, *pcs* and *pcs_new* writers/readers.
These classes can serialize and deserialize configuration spaces.
Serializing configuration spaces is useful to share configuration spaces across
experiments, or use them in other tools, for example, to analyze hyperparameter
importance with `CAVE <https://github.com/automl/CAVE>`_.

.. _json:

Serialization to JSON
---------------------

.. automodule:: ConfigSpace.read_and_write.json
   :members: read, write

.. _pcs_new:

6.2 Serialization with pcs-new (new format)
-------------------------------------------

.. automodule:: ConfigSpace.read_and_write.pcs_new
   :members: read, write

Serialization with pcs (old format)
-----------------------------------

.. automodule:: ConfigSpace.read_and_write.pcs
   :members: read, write
