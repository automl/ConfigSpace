.. ConfigSpace documentation master file, created by
   sphinx-quickstart on Mon Jul 23 18:06:55 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ConfigSpace's documentation!
=======================================

ConfigSpace is a simple python module to manage configuration spaces for algorithm configuration and hyperparameter optimization tasks.
It includes various scripts to translate between different text formats for configuration space description.

ConfigSpace is often used in our frameworks, like `SMAC3`_, `BOHB`_ or `auto-sklearn`_.

| The purpose of this documentation is to explain how to use ConfigSpace and show you its abilities.
| In the :doc:`quickstart` you will see how to set up a *configurations space* and *add hyperparameters* of different types.
  Besides containing hyperparameters, **ConfigSpace** is able to realize constraints on the defined *configurations space*. (:math:`\rightarrow` :doc:`constraints`)
| Furthermore, in the :doc:`serialization chapter <serialization>`, it will be explained how to serialize a defined *configuration space* for later usage.

.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn

It can be installed using pip::

   pip install ConfigSpace

Contents
========

.. toctree::
   :maxdepth: 2
		
   quickstart.rst
   AdvancedExample.rst
   hyperparameter.rst
   constraints.rst
   serialization.rst
   auto_examples/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
