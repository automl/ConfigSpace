.. ConfigSpace documentation master file, created by
   sphinx-quickstart on Mon Jul 23 18:06:55 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ConfigSpace's documentation!
=======================================

ConfigSpace is a simple python module to manage configuration spaces for algorithm configuration and hyperparameter optimization tasks.
It includes various modules to translate between different text formats for configuration space description.

ConfigSpace is often used in our tools such as `SMAC3`_, `BOHB`_ or `auto-sklearn`_.
To read more about our group and projects, visit our homepage `autoML.org <https://www.automl.org>`_.

| The purpose of this documentation is to explain how to use ``ConfigurationSpace`` and show you its abilities.
| In the :doc:`quickstart` you will see how to set up a ``ConfigurationSpace`` and add hyperparameters of different types.
  Besides containing hyperparameters, ``ConfigurationSpace`` is able to realize constraints such as conditions and forbidden clauses
  on the defined configurations space. (:math:`\rightarrow` :doc:`Guide`)
| Furthermore, in the :ref:`serialization chapter <Serialization>`, it will be explained how to serialize a defined *configuration space* for later usage.

.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn

It can be installed using pip::

   pip install ConfigSpace

Basic usage::

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    a = CSH.UniformIntegerHyperparameter('a', lower=10, upper=100, log=False)
    b = CSH.CategoricalHyperparameter('b', choices=['red', 'green', 'blue'])

    cs.add_hyperparameters([a, b])
    cs.sample_configuration()

    # >>> Configuration:
    # >>>   a, Value: 97
    # >>>   b, Value: 'red'


Contents
========

.. toctree::
   :maxdepth: 2
		
   quickstart.rst
   Guide.rst
   API-Doc.rst
   auto_examples/index.rst
