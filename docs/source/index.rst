.. ConfigSpace documentation master file, created by
   sphinx-quickstart on Mon Jul 23 18:06:55 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ConfigSpace's documentation!
=======================================

ConfigSpace is a simple python package to manage configuration spaces for
`algorithm configuration <https://ml.informatik.uni-freiburg.de/papers/09-JAIR-ParamILS.pdf>`_ and
`hyperparameter optimization <https://en.wikipedia.org/wiki/Hyperparameter_optimization>`_ tasks.
It includes various modules to translate between different text formats for
configuration space descriptions.

ConfigSpace is often used in AutoML tools such as `SMAC3`_, `BOHB`_ or
`auto-sklearn`_. To read more about our group and projects, visit our homepage
`AutoML.org <https://www.automl.org>`_.

This documentation explains how to use ConfigSpace and demonstrates its features.
In the :doc:`quickstart`, you will see how to set up a
:class:`~ConfigSpace.configuration_space.ConfigurationSpace`
and add hyperparameters of different types to it.
Besides containing hyperparameters, a :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
can contain constraints such as conditions and forbidden clauses.
Those are introduced in the :doc:`user guide <User-Guide>`.

Furthermore, in the :ref:`serialization section <Serialization>`, it will be
explained how to serialize a
:class:`~ConfigSpace.configuration_space.ConfigurationSpace` for later usage.

.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn


Easy usage

.. doctest::

    >>> from ConfigSpace import ConfigurationSpace
    >>> cs = ConfigurationSpace(
    >>>     name="myspace",
    >>>     space={
    >>>         "a": (0.1, 1.5),  # UniformFloat
    >>>         "b": (2, 10),  # UniformInt
    >>>         "c": ["mouse", "cat", "dog"],  # Categorical
    >>>     },
    >>> )
    >>> cs.sample_configuration(2)
    [Configuration(values={
        'a': 1.1352397950032926,
        'b': 6,
        'c': 'dog',
    })
    , Configuration(values={
        'a': 1.4623477411953076,
        'b': 8,
        'c': 'cat',
    })]
    <BLANKLINE>


More flexibility

.. doctest::

    >>> from ConfigSpace import ConfigurationSpace, Int, Float, Categorical, Normal
    >>> cs = ConfigurationSpace(
    >>>     name="myspace",
    >>>     space={
    >>>         # UniformFloat
    >>>         "a": Float("a", bounds=(0.1, 1.5), distribution=Normal(1, 10), log=True),
    >>>         # UniformInt: You can still use the simple form here
    >>>         "b": (2, 10),
    >>>         # Categorical
    >>>         "c": Categorical("c", ["mouse", "cat", "dog"], weights=[0.8, 0.1, 0.1]),
    >>>     },
    >>> )
    >>> configs = cs.sample_configuration(2)
    [Configuration(values={
        'a': 0.13613799281779645,
        'b': 6,
        'c': 'mouse',
    })
    , Configuration(values={
        'a': 0.11621314136407442,
        'b': 6,
        'c': 'mouse',
    })]
    <BLANKLINE>


Maximum flexibility

.. doctest::

    >>> import ConfigSpace as CS
    >>> import ConfigSpace.hyperparameters as CSH
    >>> cs = CS.ConfigurationSpace(seed=1234)
    >>> a = CSH.UniformIntegerHyperparameter('a', lower=10, upper=100, log=False)
    >>> b = CSH.CategoricalHyperparameter('b', choices=['red', 'green', 'blue'])
    >>> cs.add_hyperparameters([a, b])
    [a, Type: UniformInteger, Range: [10, 100], Default: 55,...]
    >>> cs.sample_configuration()
    Configuration(values={
      'a': 27,
      'b': 'green',
    })
    <BLANKLINE>

Installation
============

*ConfigSpace* requires Python 3.7 or higher.

*ConfigSpace* can be installed with *pip*:

.. code:: bash

    pip install ConfigSpace

If installing from source, the *ConfigSpace* package requires *numpy*, *cython*
and *pyparsing*. Additionally, a functioning C compiler is required.

On Ubuntu, the required compiler tools and Python headers can be installed with:

.. code:: bash

    sudo apt-get install build-essential python3 python3-dev

When using Anaconda/Miniconda, the compiler has to be installed with:

.. code:: bash

    conda install gxx_linux-64 gcc_linux-64


Citing the ConfigSpace
======================

.. code:: bibtex

   @article{
       title   = {BOAH: A Tool Suite for Multi-Fidelity Bayesian Optimization & Analysis of Hyperparameters},
       author  = {M. Lindauer and K. Eggensperger and M. Feurer and A. Biedenkapp and J. Marben and P. Müller and F. Hutter},
       journal = {arXiv:1908.06756 {[cs.LG]}},
       date    = {2019},
   }


Contents
========

.. toctree::
   :maxdepth: 2

   quickstart.rst
   User-Guide.rst
   API-Doc.rst
