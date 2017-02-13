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

from collections import deque
import copy
from typing import Union, List, Any, Dict

import numpy as np  # type: ignore
from ConfigSpace import Configuration, Constant


def impute_inactive_values(configuration: Configuration, strategy: Union[str, float]='default') -> Configuration:
    """Impute inactive parameters.

    Parameters
    ----------
    strategy : string, optional (default='default')
        The imputation strategy.

        - If 'default', replace inactive parameters by their default.
        - If float, replace inactive parameters by the given float value,
          which should be able to be splitted apart by a tree-based model.
    """
    values = dict()
    for hp_name in configuration:
        value = configuration[hp_name]
        if value is None:

            if strategy == 'default':
                hp = configuration.configuration_space.get_hyperparameter(
                    hp_name)
                new_value = hp.default

            elif isinstance(strategy, float):
                new_value = strategy

            else:
                raise ValueError('Unknown imputation strategy %s' % str(strategy))

            value = new_value

        values[hp_name] = value

    new_configuration = Configuration(configuration.configuration_space,
                                      values=values,
                                      allow_inactive_with_values=True)
    return new_configuration


def get_one_exchange_neighbourhood(configuration: Configuration, seed: int) -> List[Configuration]:
    """Return all configurations in a one-exchange neighborhood.

    The method is implemented as defined by:
    Frank Hutter, Holger H. Hoos and Kevin Leyton-Brown
    Sequential Model-Based Optimization for General Algorithm Configuration
    In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)
    """
    random = np.random.RandomState(seed)
    hyperparameters_list = list(configuration.keys())
    hyperparameters_list_length = len(hyperparameters_list)
    neighbors_to_return = dict()
    hyperparameters_used = list()
    number_of_usable_hyperparameters = sum(np.isfinite(configuration.get_array()))

    while len(hyperparameters_used) != number_of_usable_hyperparameters:
        index = random.randint(hyperparameters_list_length)
        hp_name = hyperparameters_list[index]
        if hp_name in neighbors_to_return:
            random.shuffle(neighbors_to_return[hp_name])
            n_ = neighbors_to_return[hp_name].pop()
            if len(neighbors_to_return[hp_name]) == 0:
                del neighbors_to_return[hp_name]
                hyperparameters_used.append(hp_name)
            yield n_

        else:
            neighbourhood = []
            number_of_sampled_neighbors = 0
            array = configuration.get_array()

            if not np.isfinite(array[index]):
                continue

            iteration = 0
            while True:
                hp = configuration.configuration_space.get_hyperparameter(hp_name)
                configuration._populate_values()
                num_neighbors = hp.get_num_neighbors()

                # Obtain neigbors differently for different possible numbers of
                # neighbors
                if num_neighbors == 0:
                    break
                # No infinite loops
                elif iteration > 1000:
                    break
                elif np.isinf(num_neighbors):
                    if number_of_sampled_neighbors >= 4:
                        break
                    num_samples_to_go = 4 - number_of_sampled_neighbors
                    neighbors = hp.get_neighbors(array[index], random,
                                                 number=num_samples_to_go)
                else:
                    if iteration > 0:
                        break
                    neighbors = hp.get_neighbors(array[index], random)

                # Check all newly obtained neigbors
                for neighbor in neighbors:
                    new_array = array.copy()
                    new_array[index] = neighbor
                    neighbor_value = hp._transform(neighbor)

                    # Activate hyperparameters if their parent node got activated
                    children = configuration.configuration_space.get_children_of(
                        hp_name)
                    if len(children) > 0:
                        to_visit = deque()  #type: deque
                        to_visit.extendleft(children)
                        visited = set()  #type: Set[str]
                        activated_values = dict()  #type: Dict[str, Union[int, float, str]]
                        while len(to_visit) > 0:
                            current = to_visit.pop()
                            if current.name in visited:
                                continue
                            visited.add(current.name)

                            current_idx = configuration.configuration_space. \
                                get_idx_by_hyperparameter_name(current.name)
                            current_value = new_array[current_idx]

                            conditions = configuration.configuration_space.\
                                _get_parent_conditions_of(current.name)

                            active = True
                            for condition in conditions:
                                parent_names = [c.parent.name for c in
                                                condition.get_descendant_literal_conditions()]

                                parents = {parent_name: configuration[parent_name] for
                                           parent_name in parent_names}

                                # parents come from the original configuration.
                                # We change at least one parameter. In order set
                                # other parameters which are conditional on this,
                                #  we have to activate this
                                if hp_name in parents:
                                    parents[hp_name] = neighbor_value
                                # Hyperparameters which are in depth 1 of the
                                # hyperparameter tree might have children which
                                # have to be activated as well. Once we set hp in
                                #  level 1 to active, it's value changes from the
                                #  value of the original configuration and this
                                # must be done here
                                for parent_name in parent_names:
                                    if parent_name in activated_values:
                                        parents[parent_name] = activated_values[
                                            parent_name]

                                # if one of the parents is None, the hyperparameter cannot be
                                # active! Else we have to check this
                                if any([parent_value is None for parent_value in
                                        parents.values()]):
                                    active = False
                                    break
                                else:
                                    if not condition.evaluate(parents):
                                        active = False
                                        break

                            if active and (current_value is None or
                                           not np.isfinite(current_value)):
                                default = current._inverse_transform(current.default)
                                new_array[current_idx] = default
                                children = configuration.configuration_space.get_children_of(
                                    current.name)
                                if len(children) > 0:
                                    to_visit.extendleft(children)
                                activated_values[current.name] = current.default

                            if not active and (current_value is not None
                                               or np.isfinite(current_value)):
                                new_array[current_idx] = np.NaN

                    try:
                        # Populating a configuration from an array does not check
                        #  if it is a legal configuration - check this (slow)
                        new_configuration = Configuration(
                            configuration.configuration_space, vector=new_array)
                        new_configuration.is_valid_configuration()
                        neighbourhood.append(new_configuration)
                        number_of_sampled_neighbors += 1
                    except ValueError as e:
                        pass

                    # Count iterations to not run into an infinite loop when
                    # sampling floats/ints and there is large amount of forbidden
                    #  values; also to find out if we tried to get a neighbor for
                    #  a categorical hyperparameter, and the only possible
                    # neighbor is forbidden together with another active
                    # value/default hyperparameter
                    iteration += 1
            if len(neighbourhood) > 0:
                if hp_name not in hyperparameters_used:
                    neighbors_to_return[hp_name] = neighbourhood
                    random.shuffle(neighbors_to_return[hp_name])
                    n_ = neighbors_to_return[hp_name].pop()
                    if len(neighbors_to_return[hp_name]) == 0:
                        del neighbors_to_return[hp_name]
                        hyperparameters_used.append(hp_name)
                    yield n_
    # return neighbourhood


def get_random_neighbor(configuration: Configuration, seed: int) -> Configuration:
    """Draw a random neighbor by changing one parameter of a configuration.

    * If the parameter is categorical, it changes it to another value.
    * If the parameter is ordinal, it changes it to the next higher or lower
      value.
    * If parameter is a float, draw a random sample

    If changing a parameter activates new parameters or deactivates
    previously active parameters, the configuration will be rejected. If more
    than 10000 configurations were rejected, this function raises a
    ValueError.

    Parameters
    ----------
    configuration : Configuration

    seed : int
        Used to generate a random state.

    Returns
    -------
    Configuration
        The new neighbor.

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
        except ValueError as e:
            values[hp.name] = previous_value

    return new_configuration




