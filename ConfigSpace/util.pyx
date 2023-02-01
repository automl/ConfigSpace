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
from typing import Union, Dict, Generator, List, Tuple, Optional

import numpy as np  # type: ignore
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NumericalHyperparameter,
    OrdinalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
)
import ConfigSpace.c_util
cimport cython


def impute_inactive_values(configuration: Configuration,
                           strategy: Union[str, float] = 'default') -> Configuration:
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
    hyperparameters_used = [
        hp.name
        for hp in configuration.configuration_space.get_hyperparameters()
        if (
            hp.get_num_neighbors(configuration.get(hp.name)) == 0
            and configuration.get(hp.name)is not None
        )
    ]
    number_of_usable_hyperparameters = sum(np.isfinite(configuration.get_array()))
    n_neighbors_per_hp = {
        hp.name: num_neighbors
        if (
            isinstance(hp, NumericalHyperparameter)
            and hp.get_num_neighbors(configuration.get(hp.name))> num_neighbors
        ) else
        hp.get_num_neighbors(configuration.get(hp.name))
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
                    if isinstance(hp, UniformFloatHyperparameter):
                        neighbor = hp.get_neighbors(value, random, number=1, std=stdev)[0]
                    else:
                        neighbor = hp.get_neighbors(value, random, number=1)[0]
                else:
                    if iteration > 0:
                        break
                    if hp_name not in finite_neighbors_stack:
                        if isinstance(hp, UniformIntegerHyperparameter):
                            neighbors = hp.get_neighbors(
                                value, random,
                                number=n_neighbors_per_hp[hp_name], std=stdev,
                            )
                        else:
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
                    if random.random() > 0.95:
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
                rand_idx = random.randint(0, configuration._num_hyperparameters)
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
    def fix_type_from_candidates(value, candidates):
        result = [c for c in candidates if str(value) == str(c)]
        if len(result) != 1:
            raise ValueError("Parameter value %s cannot be matched to candidates %s. "
                             "Either none or too many matching candidates." % (
                                 str(value), candidates
                             )
                             )
        return result[0]

    for param in configuration_space.get_hyperparameters():
        param_name = param.name
        if configuration.get(param_name) is not None:
            if isinstance(param, (CategoricalHyperparameter)):
                configuration[param_name] = fix_type_from_candidates(configuration[param_name],
                                                                     param.choices)
            elif isinstance(param, (OrdinalHyperparameter)):
                configuration[param_name] = fix_type_from_candidates(configuration[param_name],
                                                                     param.sequence)
            elif isinstance(param, Constant):
                configuration[param_name] = fix_type_from_candidates(configuration[param_name],
                                                                     [param.value])
            elif isinstance(param, UniformFloatHyperparameter):
                configuration[param_name] = float(configuration[param_name])
            elif isinstance(param, UniformIntegerHyperparameter):
                configuration[param_name] = int(configuration[param_name])
            else:
                raise TypeError("Unknown hyperparameter type %s" % type(param))
    return configuration


@cython.boundscheck(True)  # Activate bounds checking
@cython.wraparound(True)  # Activate negative indexing
def generate_grid(configuration_space: ConfigurationSpace,
                  num_steps_dict: Optional[Dict[str, int]] = None,
                  ) -> List[Configuration]:
    """
    Generates a grid of Configurations for a given ConfigurationSpace.
    Can be used, for example, for grid search.

    Parameters
    ----------
    configuration_space: :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The Configuration space over which to create a grid of HyperParameter Configuration values.
        It knows the types for all parameter values.

    num_steps_dict: dict
        A dict containing the number of points to divide the grid side formed by Hyperparameters
        which are either of type UniformFloatHyperparameter or type UniformIntegerHyperparameter.
        The keys in the dict should be the names of the corresponding Hyperparameters and the values
        should be the number of points to divide the grid side formed by the corresponding
        Hyperparameter in to.

    Returns
    -------
    list
        List containing Configurations. It is a cartesian product of tuples of
        HyperParameter values.
        Each tuple lists the possible values taken by the corresponding HyperParameter.
        Within the cartesian product, in each element, the ordering of HyperParameters is the same
        for the OrderedDict within the ConfigurationSpace.
    """

    def get_value_set(num_steps_dict: Optional[Dict[str, int]], hp_name: str):
        '''
        Gets values along the grid for a particular hyperparameter.

        Uses the num_steps_dict to determine number of grid values for UniformFloatHyperparameter
        and UniformIntegerHyperparameter. If these values are not present in num_steps_dict, the
        quantization factor, q, of these classes will be used to divide the grid. NOTE: When q
        is used if it is None, a ValueError is raised.

        Parameters
        ----------
        num_steps_dict: dict
            Same description as above

        hp_name: str
            Hyperparameter name

        Returns
        -------
        tuple
            Holds grid values for the given hyperparameter

        '''
        param = configuration_space.get_hyperparameter(hp_name)
        if isinstance(param, (CategoricalHyperparameter)):
            return param.choices

        elif isinstance(param, (OrdinalHyperparameter)):
            return param.sequence

        elif isinstance(param, Constant):
            return tuple([param.value, ])

        elif isinstance(param, UniformFloatHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None and param.name in num_steps_dict:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                # check for log and for rounding issues
                if param.q is not None:
                    grid_points = np.arange(lower, upper + param.q, param.q)
                else:
                    raise ValueError(
                        "num_steps_dict is None or doesn't contain the number of points"
                        f" to divide {param.name} into. And its quantization factor "
                        "is None. Please provide/set one of these values."
                    )

            if param.log:
                grid_points = np.exp(grid_points)

            # Avoiding rounding off issues
            if grid_points[0] < param.lower:
                grid_points[0] = param.lower
            if grid_points[-1] > param.upper:
                grid_points[-1] = param.upper

            return tuple(grid_points)

        elif isinstance(param, UniformIntegerHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None and param.name in num_steps_dict:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                # check for log and for rounding issues
                if param.q is not None:
                    grid_points = np.arange(lower, upper + param.q, param.q)
                else:
                    raise ValueError(
                        "num_steps_dict is None or doesn't contain the number of points "
                        f"to divide {param.name} into. And its quantization factor "
                        "is None. Please provide/set one of these values."
                    )

            if param.log:
                grid_points = np.exp(grid_points)
            grid_points = np.round(grid_points).astype(int)

            # Avoiding rounding off issues
            if grid_points[0] < param.lower:
                grid_points[0] = param.lower
            if grid_points[-1] > param.upper:
                grid_points[-1] = param.upper

            return tuple(grid_points)

        else:
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    def get_cartesian_product(value_sets: List[Tuple], hp_names: List[str]):
        '''
        Returns a grid for a subspace of the configuration with given hyperparameters
        and their grid values.

        Takes a list of tuples of grid values of the hyperparameters and list of
        hyperparameter names. The outer list iterates over the hyperparameters corresponding
        to the order in the list of hyperparameter names.
        The inner tuples contain grid values of the hyperparameters for each hyperparameter.

        Parameters
        ----------
        value_sets: list of tuples
            Same description as return value of get_value_set()

        hp_names: list of strs
            List of hyperparameter names

        Returns
        -------
        list of dicts
            List of configuration dicts

        '''
        grid = []
        import itertools
        if len(value_sets) == 0:
            # Edge case
            pass
        else:
            for element in itertools.product(*value_sets):
                config_dict = {}
                for j, hp_name in enumerate(hp_names):
                    config_dict[hp_name] = element[j]
                grid.append(config_dict)

        return grid

    # list of tuples: each tuple within is the grid values to be taken on by a Hyperparameter
    value_sets = []
    hp_names = []

    # Get HP names and allowed grid values they can take for the HPs at the top
    # level of ConfigSpace tree
    for hp_name in configuration_space._children['__HPOlib_configuration_space_root__']:
        value_sets.append(get_value_set(num_steps_dict, hp_name))
        hp_names.append(hp_name)

    # Create a Cartesian product of above allowed values for the HPs. Hold them in an
    # "unchecked" deque because some of the conditionally dependent HPs may become active
    # for some of the elements of the Cartesian product and in these cases creating a
    # Configuration would throw an Error (see below).
    # Creates a deque of Configuration dicts
    unchecked_grid_pts = deque(get_cartesian_product(value_sets, hp_names))
    checked_grid_pts = []

    while len(unchecked_grid_pts) > 0:
        try:
            grid_point = Configuration(configuration_space, unchecked_grid_pts[0])
            checked_grid_pts.append(grid_point)
        except ValueError as e:
            assert (str(e)[:23] == "Active hyperparameter '" and
                    str(e)[-16:] == "' not specified!"), \
                "Caught exception contains unexpected message."
            value_sets = []
            hp_names = []
            new_active_hp_names = []

            # "for" loop over currently active HP names
            for hp_name in unchecked_grid_pts[0]:
                value_sets.append(tuple([unchecked_grid_pts[0][hp_name], ]))
                hp_names.append(hp_name)
                # Checks if the conditionally dependent children of already active
                # HPs are now active
                for new_hp_name in configuration_space._children[hp_name]:
                    if (
                        new_hp_name not in new_active_hp_names and
                        new_hp_name not in unchecked_grid_pts[0]
                    ):
                        all_cond_ = True
                        for cond in configuration_space._parent_conditions_of[new_hp_name]:
                            if not cond.evaluate(unchecked_grid_pts[0]):
                                all_cond_ = False
                        if all_cond_:
                            new_active_hp_names.append(new_hp_name)

            for hp_name in new_active_hp_names:
                value_sets.append(get_value_set(num_steps_dict, hp_name))
                hp_names.append(hp_name)
            # this check might not be needed, as there is always going to be a new
            # active HP when in this except block?
            if len(new_active_hp_names) > 0:
                new_conditonal_grid = get_cartesian_product(value_sets, hp_names)
                unchecked_grid_pts += new_conditonal_grid
            else:
                raise RuntimeError(
                    "Unexpected error: There should have been a newly activated hyperparameter"
                    f" for the current configuration values: {str(unchecked_grid_pts[0])}. "
                    "Please contact the developers with the code you ran and the stack trace."
                )
        unchecked_grid_pts.popleft()

    return checked_grid_pts
