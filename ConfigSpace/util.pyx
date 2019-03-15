# Copyright (c) 2014-2016, ConfigSpace developers
# Matthias Feurer
# Katharina Eggensperger
# and others (see commit history).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# cython: language_level=3

from collections import deque
import copy
from typing import Union, Dict, Generator

import numpy as np  # type: ignore
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
import ConfigSpace.c_util


def impute_inactive_values(configuration: Configuration, strategy: Union[str, float] = 'default') -> Configuration:
    """Impute inactive parameters.

    Iterate through the hyperparameters of a ``Configuration`` and set the
    values of the inactive hyperparamters to their default values if the choosen
    ``strategy`` is 'default'. Otherwise ``strategy`` contains a float number.
    Set the hyperparameters' value to this number.


    Parameters
    ----------
    configuration : :class:`~ConfigSpace.configuration_space.Configuration`
         For this configuration inactive values will be imputed.
    strategy : (str, float, optional)
        The imputation strategy. Defaults to 'default'
        If 'default', replace inactive parameters by their default.
        If float, replace inactive parameters by the given float value,
        which should be able to be splitted apart by a tree-based model.

    Returns
    -------
    :class:`~ConfigSpace.configuration_space.Configuration`
        A new configuration with the imputed values.
        In this new configuration inactive values are included.
    """
    values = dict()
    for hp in configuration.configuration_space.get_hyperparameters():
        value = configuration.get(hp.name)
        if value is None:

            if strategy == 'default':
                new_value = hp.default_value

            elif isinstance(strategy, float):
                new_value = strategy

            else:
                raise ValueError('Unknown imputation strategy %s' % str(strategy))

            value = new_value

        values[hp.name] = value

    new_configuration = Configuration(configuration.configuration_space,
                                      values=values,
                                      allow_inactive_with_values=True)
    return new_configuration


def get_one_exchange_neighbourhood(
        configuration: Configuration,
        seed: int,
        num_neighbors: int = 4,
        stdev: float = 0.2) -> Generator[Configuration]:
    """
    Return all configurations in a one-exchange neighborhood.

    The method is implemented as defined by:
    Frank Hutter, Holger H. Hoos and Kevin Leyton-Brown
    Sequential Model-Based Optimization for General Algorithm Configuration
    In Proceedings of the conference on Learning and Intelligent
    Optimization(LION 5)

    Parameters
    ----------
    configuration : :class:`~ConfigSpace.configuration_space.Configuration`
        for this Configuration object ``num_neighbors`` neighbors are computed
    seed : int
        Sets the random seed to a fixed value
    num_neighbors : (int, optional)
        number of configurations, which are sampled from the neighbourhood
        of the input configuration
    stdev : (float, optional)
        The standard deviation is used to determine the neigbours of
        :class:`~ConfigSpace.hyperparameters.UniformFloatHyperparameter` and
        :class:`~ConfigSpace.hyperparameters.UniformIntegerHyperparameter`.

    Returns
    -------
    Generator
         It contains configurations, with values being situated around
         the given configuration.

    """
    random = np.random.RandomState(seed)
    hyperparameters_list = list(
        list(configuration.configuration_space._hyperparameters.keys())
    )
    hyperparameters_list_length = len(hyperparameters_list)
    hyperparameters_used = [hp.name for hp in configuration.configuration_space.get_hyperparameters()
                            if hp.get_num_neighbors(configuration.get(hp.name)) == 0 and
                            configuration.get(hp.name)is not None]
    number_of_usable_hyperparameters = sum(np.isfinite(configuration.get_array()))
    n_neighbors_per_hp = {
        hp.name: num_neighbors if
        np.isinf(hp.get_num_neighbors(configuration.get(hp.name)))
        else hp.get_num_neighbors(configuration.get(hp.name))
        for hp in configuration.configuration_space.get_hyperparameters()
    }

    finite_neighbors_stack = {}  # type: Dict
    configuration_space = configuration.configuration_space  # type: ConfigSpace

    while len(hyperparameters_used) < number_of_usable_hyperparameters:
        index = int(random.randint(hyperparameters_list_length))
        hp_name = hyperparameters_list[index]
        if n_neighbors_per_hp[hp_name] == 0:
            continue

        else:
            neighbourhood = []
            number_of_sampled_neighbors = 0
            array = configuration.get_array()  # type: np.ndarray
            value = array[index]  # type: float

            # Check for NaNs (inactive value)
            if value != value:
                continue

            iteration = 0
            hp = configuration_space.get_hyperparameter(hp_name)  # type: Hyperparameter
            num_neighbors_for_hp = hp.get_num_neighbors(configuration.get(hp_name))
            while True:
                # Obtain neigbors differently for different possible numbers of
                # neighbors
                if num_neighbors_for_hp == 0:
                    break
                # No infinite loops
                elif iteration > 100:
                    break
                elif np.isinf(num_neighbors_for_hp):
                    if number_of_sampled_neighbors >= 1:
                        break
                    # TODO if code becomes slow remove the isinstance!
                    if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                        neighbor = hp.get_neighbors(value, random,
                                                    number=1, std=stdev)[0]
                    else:
                        neighbor = hp.get_neighbors(value, random,
                                                    number=1)[0]
                else:
                    if iteration > 0:
                        break
                    if hp_name not in finite_neighbors_stack:
                        neighbors = hp.get_neighbors(value, random)
                        random.shuffle(neighbors)
                        finite_neighbors_stack[hp_name] = neighbors
                    else:
                        neighbors = finite_neighbors_stack[hp_name]
                    neighbor = neighbors.pop()

                # Check all newly obtained neigbors
                new_array = array.copy()
                new_array = ConfigSpace.c_util.change_hp_value(
                    configuration_space=configuration_space,
                    configuration_array=new_array,
                    hp_name=hp_name,
                    hp_value=neighbor,
                    index=index)
                try:
                    # Populating a configuration from an array does not check
                    #  if it is a legal configuration - check this (slow)
                    new_configuration = Configuration(configuration_space,
                                                      vector=new_array)  # type: Configuration
                    # Only rigorously check every tenth configuration (
                    # because moving around in the neighborhood should
                    # just work!)
                    if np.random.random() > 0.95:
                        new_configuration.is_valid_configuration()
                    else:
                        configuration_space._check_forbidden(new_array)
                    neighbourhood.append(new_configuration)
                except ForbiddenValueError:
                    pass

                iteration += 1
                if len(neighbourhood) > 0:
                    number_of_sampled_neighbors += 1

            # Some infinite loop happened and no valid neighbor was found OR
            # no valid neighbor is available for a categorical
            if len(neighbourhood) == 0:
                hyperparameters_used.append(hp_name)
                n_neighbors_per_hp[hp_name] = 0
                hyperparameters_used.append(hp_name)
            else:
                if hp_name not in hyperparameters_used:
                    n_ = neighbourhood.pop()
                    n_neighbors_per_hp[hp_name] -= 1
                    if n_neighbors_per_hp[hp_name] == 0:
                        hyperparameters_used.append(hp_name)
                    yield n_


def get_random_neighbor(configuration: Configuration, seed: int) -> Configuration:
    """
    Draw a random neighbor by changing one parameter of a configuration.

    - If the parameter is categorical, it changes it to another value.
    - If the parameter is ordinal, it changes it to the next higher or
      lower value.
    - If parameter is a float, draw a random sample

    If changing a parameter activates new parameters or deactivates
    previously active parameters, the configuration will be rejected. If more
    than 10000 configurations were rejected, this function raises a
    ValueError.

    Parameters
    ----------
    configuration : :class:`~ConfigSpace.configuration_space.Configuration`
        a configuration for which a random neigbour is calculated
    seed : int
        Used to generate a random state.

    Returns
    -------
    :class:`~ConfigSpace.configuration_space.Configuration`
        The new neighbor

    """
    random = np.random.RandomState(seed)
    rejected = True
    values = copy.deepcopy(configuration.get_dictionary())

    while rejected:
        # First, choose an active hyperparameter
        active = False
        iteration = 0
        while not active:
            iteration += 1
            if configuration._num_hyperparameters > 1:
                rand_idx = random.randint(0,
                                          configuration._num_hyperparameters - 1)
            else:
                rand_idx = 0

            value = configuration.get_array()[rand_idx]
            if np.isfinite(value):
                active = True

                hp_name = configuration.configuration_space \
                    .get_hyperparameter_by_idx(rand_idx)
                hp = configuration.configuration_space.get_hyperparameter(hp_name)

                # Only choose if there is a possibility of finding a neigboor
                if not hp.has_neighbors():
                    active = False

            if iteration > 10000:
                raise ValueError('Probably caught in an infinite loop.')
        # Get a neighboor and adapt the rest of the configuration if necessary
        neighbor = hp.get_neighbors(value, random, number=1, transform=True)[0]
        previous_value = values[hp.name]
        values[hp.name] = neighbor

        try:
            new_configuration = Configuration(
                configuration.configuration_space, values=values)
            rejected = False
        except ValueError:
            values[hp.name] = previous_value

    return new_configuration


def deactivate_inactive_hyperparameters(
        configuration: Dict,
        configuration_space: ConfigurationSpace,
        vector: Union[None, np.ndarray] = None,
):
    """
    Remove inactive hyperparameters from a given configuration

    Parameters
    ----------
    configuration : dict
        a configuration as a dictionary. Key: name of the hyperparameter.
        Value: value of this hyperparamter
        configuration from which inactive hyperparameters will be removed
    configuration_space : :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The defined configuration space. It is necessary to find the inactive
        hyperparameters by iterating through the conditions of the
        configuration space.
    vector : (np.ndarray, optional)
        Efficient represantation of a configuration. Either ``configuration`` or
        ``vector`` must be specified. If both are specified only
        ``configuration`` will be used.

    Returns
    -------
    :class:`~ConfigSpace.configuration_space.Configuration`
        A configuration that is equivalent to the given configuration, except
        that inactivate hyperparameters have been removed.

    """
    hyperparameters = configuration_space.get_hyperparameters()
    configuration = Configuration(configuration_space=configuration_space,
                                  values=configuration,
                                  vector=vector,
                                  allow_inactive_with_values=True)

    hps = deque()

    unconditional_hyperparameters = configuration_space.get_all_unconditional_hyperparameters()
    hyperparameters_with_children = list()
    for uhp in unconditional_hyperparameters:
        children = configuration_space._children_of[uhp]
        if len(children) > 0:
            hyperparameters_with_children.append(uhp)
    hps.extendleft(hyperparameters_with_children)

    inactive = set()

    while len(hps) > 0:
        hp = hps.pop()
        children = configuration_space._children_of[hp]
        for child in children:
            conditions = configuration_space._parent_conditions_of[child.name]
            for condition in conditions:
                if not condition.evaluate_vector(configuration.get_array()):
                    dic = configuration.get_dictionary()
                    try:
                        del dic[child.name]
                    except KeyError:
                        continue
                    configuration = Configuration(
                        configuration_space=configuration_space,
                        values=dic,
                        allow_inactive_with_values=True)
                    inactive.add(child.name)
                hps.appendleft(child.name)

    for hp in hyperparameters:
        if hp.name in inactive:
            dic = configuration.get_dictionary()
            try:
                del dic[hp.name]
            except KeyError:
                continue
            configuration = Configuration(
                configuration_space=configuration_space,
                values=dic,
                allow_inactive_with_values=True)

    return Configuration(configuration_space, values=configuration.get_dictionary())


def fix_types(configuration: dict,
              configuration_space: ConfigurationSpace):
    """
    Iterate over all hyperparameters in the ConfigSpace
    and fix the types of the parameter values in configuration.

    Parameters
    ----------
    configuration : dict
        a configuration as a dictionary. Key: name of the hyperparameter.
        Value: value of this hyperparamter
    configuration_space : :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        Configuration space which knows the types for all parameter values

    Returns
    -------
    dict
        configuration with fixed types of parameter values
    """
    for param in configuration_space.get_hyperparameters():
        param_name = param.name
        if configuration.get(param_name) is not None:
            if isinstance(param, (CategoricalHyperparameter)):
                # should be unnecessary, but to be on the safe param_name:
                configuration[param_name] = str(configuration[param_name])
            elif isinstance(param, (OrdinalHyperparameter)):
                # should be unnecessary, but to be on the safe side:
                configuration[param_name] = str(configuration[param_name])
            elif isinstance(param, Constant):
                # should be unnecessary, but to be on the safe side:
                configuration[param_name] = str(configuration[param_name])
            elif isinstance(param, UniformFloatHyperparameter):
                configuration[param_name] = float(configuration[param_name])
            elif isinstance(param, UniformIntegerHyperparameter):
                configuration[param_name] = int(configuration[param_name])
            else:
                raise TypeError("Unknown hyperparameter type %s" % type(param))
    return configuration
