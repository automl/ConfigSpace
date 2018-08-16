Hyperparameters
===============

In this section, the different types of hyperparameters are introduced. ConfigSpace is able to handle integer, float, categorical as well as ordinal hyperparameters.
Float and integer hyperparameters are available as **uniform** or **normal distributed** ones.
This means, that when a hyperparameter is sampled from the configuration space, its value is distributed according to the specified type.

Example usages are shown in the :doc:`quickstart <quickstart>`

Integer hyperparameters
-----------------------

.. py:class:: ConfigSpace.hyperparameters.UniformIntegerHyperparameter(name: str, lower: int, upper: int, default_value: Union[int, None]=None, q: Union[int, None]=None, log: bool=False) -> None:

    Creates a integer-hyperparameter with values sampled from a uniform distribution with values from *lower* to *upper*

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int lower: lower bound of a range of values from which the hyperparameter will be sampled.
    :param int upper: upper bound.
    :param int,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int q:
    :param bool log: If *bool* is true, the values of the hyperparameter will be sampled on a logarithmic scale.

.. py:class:: ConfigSpace.hyperparameters.NormalIntegerHyperparameter(name: str, mu: int, sigma: Union[int, float], default_value: Union[int, None]=None, q: Union[None, int]=None, log: bool=False) -> None:

    Creates a integer-hyperparameter with values sampled from a normal distribution :math:`\mathcal{N}(\mu, \sigma^2)`

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int mu: Mean of the distribution.
    :param int,float sigma: Standard deviation of the distribution.
    :param int,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int q:
    :param bool log: If *bool* is true, the values of the hyperparameter will be sampled on a logarithmic scale.


Float hyperparameters
---------------------

.. py:class:: ConfigSpace.hyperparameters.UniformFloatHyperparameter(name: str, lower: Union[int, float], upper: Union[int, float], default_value: Union[int, float, None]=None, q: Union[int, float, None]=None, log: bool=False) -> None:

    Creates a float-hyperparameter with values sampled from a uniform distribution with values from *lower* to *upper*

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int,float lower: lower bound of a range of values from which the hyperparameter will be sampled.
    :param int,float upper: upper bound.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int,float q:
    :param bool log: If *bool* is true, the values of the hyperparameter will be sampled on a logarithmic scale.

.. py:class:: ConfigSpace.hyperparameters.NormalFloatHyperparameter(name: str, mu: Union[int, float], sigma: Union[int, float], default_value: Union[float, None]=None, q: Union[None, int, float]=None, log: bool=False) -> None:

    Creates a float-hyperparameter with values sampled from a normal distribution :math:`\mathcal{N}(\mu, \sigma^2)`

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param int,float mu: Mean of the distribution.
    :param int,float sigma: Standard deviation of the distribution.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.
    :param None,int,float q:
    :param bool log: If *bool* is true, the values of the hyperparameter will be sampled on a logarithmic scale.


Categorical hyperparameters
---------------------------

.. py:class:: ConfigSpace.hyperparameters.CategoricalHyperparameter(name: str, choices: Union[List[Union[str, float, int]], Tuple[Union[float, int, str]]], default_value: Union[int, float, str, None]=None) -> None:

    Creates a categorical-hyperparameter

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param List[Union[str,float,int]],Tuple[Union[float,int,str]]] choises: set of values to sample hyperparameter from.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.


OrdinalHyperparameters
----------------------

.. py:class:: ConfigSpace.hyperparameters.CategoricalHyperparameter(name: str, choices: Union[List[Union[str, float, int]], Tuple[Union[float, int, str]]], default_value: Union[int, float, str, None]=None) -> None:

    Creates a ordinal-hyperparameter

    :param str name: Name of the hyperparameter, with which it can be accessed.
    :param List[Union[str,float,int]],Tuple[Union[float,int,str]]] choises: set of values to sample hyperparameter from.
    :param int,float,None default_value: Sets the default value of a hyperparameter to a given value.
