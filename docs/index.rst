Welcome to ConfigSpace's documentation!
=======================================

.. toctree::
  :hidden:
  :maxdepth: 2

  quickstart
  guide
  api/index

ConfigSpace is a simple python package to manage configuration spaces for
`algorithm configuration <https://ml.informatik.uni-freiburg.de/papers/09-JAIR-ParamILS.pdf>`_ and
`hyperparameter optimization <https://en.wikipedia.org/wiki/Hyperparameter_optimization>`_ tasks.
It includes various modules to translate between different text formats for
configuration space descriptions.

ConfigSpace is often used in AutoML tools such as `SMAC3`_, `BOHB`_ or
`auto-sklearn`_. To read more about our group and projects, visit our homepage
`AutoML.org <https://www.automl.org>`_.

This documentation explains how to use ConfigSpace and demonstrates its features.
In the :doc:`quickstart`, you will see how to set up a :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
and add hyperparameters of different types to it.
Besides containing hyperparameters, a :class:`~ConfigSpace.configuration_space.ConfigurationSpace` can contain constraints such as conditions and forbidden clauses.
Those are introduced in the :doc:`user guide <guide>`.

Furthermore, in the :ref:`serialization section <Serialization>`, it will be
explained how to serialize a :class:`~ConfigSpace.configuration_space.ConfigurationSpace` for later usage.

.. _SMAC3: https://github.com/automl/SMAC3
.. _BOHB: https://github.com/automl/HpBandSter
.. _auto-sklearn: https://github.com/automl/auto-sklearn



Get Started
-----------

Create a simple :class:`~ConfigSpace.configuration_space.ConfigurationSpace` and then sample a :class:`~ConfigSpace.configuration_space.Configuration` from it!

>>> from ConfigSpace import ConfigurationSpace
>>>
>>> cs = ConfigurationSpace({
...     "myfloat": (0.1, 1.5),                # Uniform Float
...     "myint": (2, 10),                     # Uniform Integer
...     "species": ["mouse", "cat", "dog"],   # Categorical
... })
>>> configs = cs.sample_configuration(2)


Use :mod:`~ConfigSpace.api.types.float`, :mod:`~ConfigSpace.api.types.integer`
or :mod:`~ConfigSpace.api.types.categorical` to customize how sampling is done!

>>> from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
>>> cs = ConfigurationSpace(
...     name="myspace",
...     seed=1234,
...     space={
...         "a": Float("a", bounds=(0.1, 1.5), distribution=Normal(1, 10), log=True),
...         "b": Integer("b", bounds=(2, 10)),
...         "c": Categorical("c", ["mouse", "cat", "dog"], weights=[2, 1, 1]),
...     },
... )
>>> cs.sample_configuration(2)
[Configuration(values={
  'a': 0.17013149799713567,
  'b': 5,
  'c': 'dog',
})
, Configuration(values={
  'a': 0.5476203000512754,
  'b': 9,
  'c': 'mouse',
})
]

Maximum flexibility with conditionals, see :ref:`forbidden clauses <Forbidden clauses>` and :ref:`conditionals <conditions>` for more info.

>>> from ConfigSpace import Categorical, ConfigurationSpace, EqualsCondition, Float
...
>>> cs = ConfigurationSpace(seed=1234)
...
>>> c = Categorical("c1", items=["a", "b"])
>>> f = Float("f1", bounds=(1.0, 10.0))
...
>>> # A condition where `f` is only active if `c` is equal to `a` when sampled
>>> cond = EqualsCondition(f, c, "a")
...
>>> # Add them explicitly to the configuration space
>>> cs.add([c, f])
[c1, Type: Categorical, Choices: {a, b}, Default: a, f1, Type: UniformFloat, Range: [1.0, 10.0], Default: 5.5]

>>> cs.add(cond)
f1 | c1 == 'a'



Installation
============

*ConfigSpace* requires Python 3.7 or higher.

*ConfigSpace* can be installed with *pip*:

.. code:: bash

    pip install ConfigSpace

Citing the ConfigSpace
======================

.. code::

   @article{
       title   = {BOAH: A Tool Suite for Multi-Fidelity Bayesian Optimization & Analysis of Hyperparameters},
       author  = {M. Lindauer and K. Eggensperger and M. Feurer and A. Biedenkapp and J. Marben and P. Müller and F. Hutter},
       journal = {arXiv:1908.06756 {[cs.LG]}},
       date    = {2019},
   }

