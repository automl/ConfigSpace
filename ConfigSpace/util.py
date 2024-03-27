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
from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ConfigSpace import Configuration
from ConfigSpace.exceptions import (
    ActiveHyperparameterNotSetError,
    ForbiddenValueError,
    IllegalVectorizedValueError,
    InactiveHyperparameterSetError,
    NoPossibleNeighborsError,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    NumericalHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

if TYPE_CHECKING:
    from ConfigSpace.configuration_space import ConfigurationSpace


def impute_inactive_values(
    configuration: Configuration,
    strategy: str | float = "default",
) -> Configuration:
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

    Returns:
    -------
    :class:`~ConfigSpace.configuration_space.Configuration`
        A new configuration with the imputed values.
        In this new configuration inactive values are included.
    """
    values = {}
    for hp in configuration.config_space.values():
        value = configuration.get(hp.name)
        if value is None:
            if strategy == "default":
                new_value = hp.default_value

            elif isinstance(strategy, float):
                new_value = strategy

            else:
                raise ValueError(f"Unknown imputation strategy {strategy}")

            value = new_value

        values[hp.name] = value

    return Configuration(
        configuration.config_space,
        values=values,
        allow_inactive_with_values=True,
    )


def get_one_exchange_neighbourhood_fast(
    configuration: Configuration,
    seed: int,
    num_neighbors: int = 4,
    stdev: float = 0.2,
) -> Iterator[Configuration]:
    """Return all configurations in a one-exchange neighborhood.

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

    Returns:
    -------
    Iterator
         It contains configurations, with values being situated around
         the given configuration.

    """
    # TODO: If this needs to get faster, hyperparameters should have a
    # `_get_num_neighbors_vector`, preventing the need to convert to value
    space = configuration.config_space
    config_vector = configuration._vector

    hyperparameters_list = list(space)
    n_hps = len(hyperparameters_list)
    hyperparameters_used = []

    # For all in_use hyperparameters with a size of 1, these have no neighbors
    # and are automatically in use.
    for hp in space.values():
        hp_idx = space._hyperparameter_idx[hp.name]
        in_use = not np.isnan(config_vector[hp_idx])
        if in_use and hp.size == 1:
            hyperparameters_used.append(hp.name)

    neighbor_count: dict[str, int | float] = {}
    for hp in space.values():
        # TODO: Not sure why numericals have their num_neighbors restricted
        if isinstance(hp, NumericalHyperparameter):
            if hp.size > num_neighbors:
                neighbor_count[hp.name] = num_neighbors
            else:
                neighbor_count[hp.name] = hp.get_num_neighbors(
                    value=configuration.get(hp.name),
                )
        else:
            value = configuration.get(hp.name)
            neighbor_count[hp.name] = hp.get_num_neighbors(value)

    finite_neighbors_stack: dict[str, list[np.number]] = {}
    number_of_usable_hyperparameters = sum(np.isfinite(configuration._vector))

    # TODO: Better idea of what to set to than 200
    # Seems to loop around 600 times...
    # Is there some way to calculate how many neighbors we can possibly sample?
    random = np.random.RandomState(seed)
    randints = iter(random.randint(n_hps, size=num_neighbors * 200))

    while len(hyperparameters_used) < number_of_usable_hyperparameters:
        index = next(randints, None)
        if index is None:
            randints = iter(random.randint(n_hps, size=num_neighbors * 10))
            index = next(randints)

        hp_name = hyperparameters_list[index]
        if neighbor_count[hp_name] == 0:
            continue

        neighbourhood = []
        number_of_sampled_neighbors = 0
        array = configuration.get_array()
        vector: np.float64 = array[index]  # type: float

        # Inactive value
        if np.isnan(vector):
            continue

        iteration = 0
        hp = space[hp_name]
        num_neighbors_for_hp = hp.get_num_neighbors(configuration.get(hp_name))
        while True:
            # Obtain neigbors differently for different possible numbers of
            # neighbors
            if num_neighbors_for_hp == 0 or iteration > 100:
                break

            if np.isinf(num_neighbors_for_hp):
                if number_of_sampled_neighbors >= 1:
                    break

                if isinstance(hp, UniformFloatHyperparameter):
                    neighbor = hp.neighbors_vectorized(
                        vector,
                        n=1,
                        seed=random,
                        std=stdev,
                    )[0]
                else:
                    neighbor = hp.neighbors_vectorized(vector, n=1, seed=random)[0]

            else:
                if iteration > 0:
                    break

                if hp_name not in finite_neighbors_stack:
                    # TODO: Why only uniform int?
                    if isinstance(hp, UniformIntegerHyperparameter):
                        neighbors = hp.neighbors_vectorized(
                            vector,
                            n=int(neighbor_count[hp_name]),
                            seed=random,
                            std=stdev,
                        )
                    else:
                        neighbors = hp.neighbors_vectorized(
                            vector,
                            n=4,
                            seed=random,
                        )

                    neighbors = list(neighbors)
                    random.shuffle(neighbors)
                    finite_neighbors_stack[hp_name] = neighbors
                else:
                    neighbors = finite_neighbors_stack[hp_name]

                neighbor = neighbors.pop()
                if len(neighbors) == 0:
                    finite_neighbors_stack.pop(hp_name)

            # Check all newly obtained neigbors
            new_array = array.copy()
            new_array = change_hp_value(
                configuration_space=space,
                configuration_array=new_array,
                hp_name=hp_name,
                hp_value=neighbor,
                index=index,
            )
            try:
                # Populating a configuration from an array does not check
                #  if it is a legal configuration - check this (slow)
                new_configuration = Configuration(space, vector=new_array)
                # Only rigorously check every tenth configuration (
                # because moving around in the neighborhood should
                # just work!)
                if random.random() > 0.95:
                    new_configuration.is_valid_configuration()
                else:
                    space._check_forbidden(new_array)
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
            neighbor_count[hp_name] = 0
            hyperparameters_used.append(hp_name)
        elif hp_name not in hyperparameters_used:
            n_ = neighbourhood.pop()
            neighbor_count[hp_name] -= 1
            if neighbor_count[hp_name] == 0:
                hyperparameters_used.append(hp_name)
            yield n_


def get_one_exchange_neighbourhood(
    configuration: Configuration,
    seed: int,
    num_neighbors: int = 4,
    stdev: float = 0.2,
) -> Iterator[Configuration]:
    """Return all configurations in a one-exchange neighborhood.

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

    Returns:
    -------
    Iterator
         It contains configurations, with values being situated around
         the given configuration.

    """
    random = np.random.RandomState(seed)
    space = configuration.config_space
    hyperparameters_list = list(space)
    hyperparameters_list_length = len(hyperparameters_list)
    hyperparameters_used = [
        name
        for name, hp in space.items()
        if (
            hp.get_num_neighbors(configuration.get(name)) == 0
            and configuration.get(name) is not None
        )
    ]
    number_of_usable_hyperparameters = sum(np.isfinite(configuration.get_array()))
    n_neighbors_per_hp = {
        hp.name: num_neighbors
        if (
            isinstance(hp, NumericalHyperparameter)
            and hp.get_num_neighbors(configuration.get(hp.name)) > num_neighbors
        )
        else hp.get_num_neighbors(configuration.get(hp.name))
        for hp in space.values()
    }

    finite_neighbors_stack: dict[str, list[np.number]] = {}

    while len(hyperparameters_used) < number_of_usable_hyperparameters:
        index = int(random.randint(hyperparameters_list_length))
        hp_name = hyperparameters_list[index]
        if n_neighbors_per_hp[hp_name] == 0:
            continue

        else:
            neighbourhood = []
            number_of_sampled_neighbors = 0
            array = configuration.get_array()
            vector: np.float64 = array[index]  # type: float

            # Inactive value
            if np.isnan(vector):
                continue

            iteration = 0
            hp = space[hp_name]
            num_neighbors_for_hp = hp.get_num_neighbors(configuration.get(hp_name))
            while True:
                # Obtain neigbors differently for different possible numbers of
                # neighbors
                if num_neighbors_for_hp == 0 or iteration > 100:
                    break
                elif np.isinf(num_neighbors_for_hp):
                    if number_of_sampled_neighbors >= 1:
                        break
                    if isinstance(hp, UniformFloatHyperparameter):
                        neighbor = hp.neighbors_vectorized(
                            vector,
                            n=1,
                            seed=random,
                            std=stdev,
                        )[0]
                    else:
                        neighbor = hp.neighbors_vectorized(vector, n=1, seed=random)[0]
                else:
                    if iteration > 0:
                        break

                    if hp_name not in finite_neighbors_stack:
                        # TODO: Why only uniform int?
                        if isinstance(hp, UniformIntegerHyperparameter):
                            neighbors = hp.neighbors_vectorized(
                                vector,
                                n=int(n_neighbors_per_hp[hp_name]),
                                seed=random,
                                std=stdev,
                            )
                        else:
                            neighbors = hp.neighbors_vectorized(
                                vector,
                                n=4,
                                seed=random,
                            )

                        neighbors = list(neighbors)
                        random.shuffle(neighbors)
                        finite_neighbors_stack[hp_name] = neighbors
                    else:
                        neighbors = finite_neighbors_stack[hp_name]
                    neighbor = neighbors.pop()
                    if len(neighbors) == 0:
                        finite_neighbors_stack.pop(hp_name)

                # Check all newly obtained neigbors
                new_array = array.copy()
                new_array = change_hp_value(
                    configuration_space=space,
                    configuration_array=new_array,
                    hp_name=hp_name,
                    hp_value=neighbor,
                    index=index,
                )
                try:
                    # Populating a configuration from an array does not check
                    #  if it is a legal configuration - check this (slow)
                    new_configuration = Configuration(space, vector=new_array)
                    # Only rigorously check every tenth configuration (
                    # because moving around in the neighborhood should
                    # just work!)
                    if random.random() > 0.95:
                        new_configuration.is_valid_configuration()
                    else:
                        space._check_forbidden(new_array)
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
            elif hp_name not in hyperparameters_used:
                n_ = neighbourhood.pop()
                n_neighbors_per_hp[hp_name] -= 1
                if n_neighbors_per_hp[hp_name] == 0:
                    hyperparameters_used.append(hp_name)
                yield n_


def get_random_neighbor(configuration: Configuration, seed: int) -> Configuration:
    """Draw a random neighbor by changing one parameter of a configuration.

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

    Returns:
    -------
    :class:`~ConfigSpace.configuration_space.Configuration`
        The new neighbor

    """
    random = np.random.RandomState(seed)
    rejected = True
    values = copy.deepcopy(dict(configuration))
    new_configuration = None

    if configuration.config_space.estimate_size() <= 1:
        raise NoPossibleNeighborsError(
            "Cannot generate a random neighbor for a configuration space with"
            " only one configuration."
            f"\n{configuration.config_space}",
        )

    while rejected:
        # First, choose an active hyperparameter
        active = False
        iteration = 0
        hp: Hyperparameter | None = None
        value = None
        while not active:
            iteration += 1
            rand_idx = (
                random.randint(0, len(configuration)) if len(configuration) > 1 else 0
            )

            value = configuration.get_array()[rand_idx]
            if np.isfinite(value):
                active = True

                hp_name = configuration.config_space.at[rand_idx]
                hp = configuration.config_space[hp_name]

                # Only choose if there is a possibility of finding a neigboor
                if not hp.has_neighbors():
                    active = False

            if iteration > 10000:
                raise ValueError("Probably caught in an infinite loop.")

        assert hp is not None
        assert value is not None

        # Get a neighboor and adapt the rest of the configuration if necessary
        neighbor = hp.to_value(vector=hp.neighbors_vectorized(value, n=1, seed=random))[
            0
        ]
        previous_value = values[hp.name]
        values[hp.name] = neighbor

        try:
            new_configuration = Configuration(configuration.config_space, values=values)
            rejected = False
        except ValueError:
            values[hp.name] = previous_value

    assert new_configuration is not None
    return new_configuration


def deactivate_inactive_hyperparameters(
    configuration: dict,
    configuration_space: ConfigurationSpace,
    vector: None | np.ndarray = None,
) -> Configuration:
    """Remove inactive hyperparameters from a given configuration.

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

    Returns:
    -------
    :class:`~ConfigSpace.configuration_space.Configuration`
        A configuration that is equivalent to the given configuration, except
        that inactivate hyperparameters have been removed.

    """
    space = configuration_space
    hyperparameters = list(space.values())
    config = Configuration(
        configuration_space=configuration_space,
        values=configuration,
        vector=vector,
        allow_inactive_with_values=True,
    )

    hps: deque[Hyperparameter] = deque()
    hps.extendleft(
        [
            space[hp]
            for hp in space.unconditional_hyperparameters
            if len(space.children_of[hp]) > 0
        ],
    )

    inactive = set()

    while len(hps) > 0:
        hp = hps.pop()
        for child in space.children_of[hp.name]:
            for condition in space.parent_conditions_of[child.name]:
                if not condition.satisfied_by_vector(config.get_array()):
                    dic = dict(config)
                    try:
                        del dic[child.name]
                    except KeyError:
                        continue

                    config = Configuration(
                        configuration_space=space,
                        values=dic,
                        allow_inactive_with_values=True,
                    )
                    inactive.add(child.name)
                hps.appendleft(child)

    for hp in hyperparameters:
        if hp.name in inactive:
            dic = dict(config)
            try:
                del dic[hp.name]
            except KeyError:
                continue
            config = Configuration(
                configuration_space=configuration_space,
                values=dic,
                allow_inactive_with_values=True,
            )

    return Configuration(configuration_space, values=dict(config))


def fix_types(
    configuration: dict[str, Any],
    configuration_space: ConfigurationSpace,
) -> dict[str, Any]:
    """Iterate over all hyperparameters in the ConfigSpace
    and fix the types of the parameter values in configuration.

    Parameters
    ----------
    configuration : dict
        a configuration as a dictionary. Key: name of the hyperparameter.
        Value: value of this hyperparamter
    configuration_space : :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        Configuration space which knows the types for all parameter values

    Returns:
    -------
    dict
        configuration with fixed types of parameter values
    """

    def fix_type_from_candidates(value: Any, candidates: Sequence[Any]) -> Any:
        result = [c for c in candidates if str(value) == str(c)]
        if len(result) != 1:
            raise ValueError(
                f"Parameter value {value} cannot be matched to candidates {candidates}."
                " Either none or too many matching candidates.",
            )
        return result[0]

    for param in configuration_space.values():
        param_name = param.name
        if configuration.get(param_name) is not None:
            if isinstance(param, (CategoricalHyperparameter)):
                configuration[param_name] = fix_type_from_candidates(
                    configuration[param_name],
                    param.choices,
                )
            elif isinstance(param, (OrdinalHyperparameter)):
                configuration[param_name] = fix_type_from_candidates(
                    configuration[param_name],
                    param.sequence,
                )
            elif isinstance(param, Constant):
                configuration[param_name] = fix_type_from_candidates(
                    configuration[param_name],
                    [param.value],
                )
            elif isinstance(param, UniformFloatHyperparameter):
                configuration[param_name] = float(configuration[param_name])
            elif isinstance(param, UniformIntegerHyperparameter):
                configuration[param_name] = int(configuration[param_name])
            else:
                raise TypeError(f"Unknown hyperparameter type {type(param)}")
    return configuration


def check_configuration(
    space: ConfigurationSpace,
    vector: np.ndarray,
    allow_inactive_with_values: bool = False,
) -> None:
    for hp_name, hp in space.items():
        hp_idx = space.index_of[hp_name]
        hp_vector_val = vector[hp_idx]
        is_active = ~np.isnan(hp_vector_val)
        if is_active and not hp.legal_vector(hp_vector_val):
            raise IllegalVectorizedValueError(hp, hp_vector_val)

        should_be_active = True
        for _parent_node, condition in space.dag.dependancies(hp_name):
            # If all conditions pass, then the hyperparameter should remain active
            if not condition.satisfied_by_vector(vector):
                should_be_active = False
                if not allow_inactive_with_values and is_active:
                    raise InactiveHyperparameterSetError(hp, hp_vector_val)

        # If all condition checks above are satisfied, then the hyperparameter
        # should be active
        if should_be_active and not is_active:
            raise ActiveHyperparameterNotSetError(hp)

        for clause in space.forbidden_clauses:
            if clause.is_forbidden_vector(vector):
                raise ForbiddenValueError(
                    f"Given vector violates forbidden clause {clause}",
                )


def change_hp_value(
    configuration_space: ConfigurationSpace,
    configuration_array: np.ndarray,
    hp_name: str,
    hp_value: float,
    index: int,
) -> np.ndarray:
    """Change hyperparameter value in configuration array to given value.

    Does not check if the new value is legal. Activates and deactivates other
    hyperparameters if necessary. Does not check if new hyperparameter value
    results in the violation of any forbidden clauses.

    Parameters
    ----------
    configuration_space : ConfigurationSpace

    configuration_array : np.ndarray

    hp_name : str

    hp_value : float

    index : int

    Returns:
    -------
    np.ndarray
    """
    configuration_array[index] = hp_value

    # Hyperparameters which are going to be set to inactive
    disabled = []

    # Hyperparameters which are going to be set activate, we introduce this to resolve
    # the conflict that might be raised by OrConjunction:
    # Suppose that we have a parent HP_p whose possible values are A, B, C; a
    # child HP_d is activate if HP_p is A or B. Then when HP_p switches from A to B,
    # HP_d needs to remain activate.
    hps_to_be_activate = set()

    # Activate hyperparameters if their parent node got activated
    children = configuration_space.children_of[hp_name]
    if len(children) > 0:
        to_visit = deque()  # type: deque
        to_visit.extendleft(children)
        visited = set()

        while len(to_visit) > 0:
            current = to_visit.pop()
            current_name = current.name
            if current_name in visited:
                continue
            visited.add(current_name)
            if current_name in hps_to_be_activate:
                continue

            current_idx = configuration_space.index_of[current_name]
            current_value = configuration_array[current_idx]

            conditions = configuration_space.parent_conditions_of[current_name]

            active = True
            for condition in conditions:
                if condition.satisfied_by_vector(configuration_array) is False:
                    active = False
                    break

            if active:
                hps_to_be_activate.add(current_idx)
                if current_value == current_value:
                    children_ = configuration_space.children_of[current_name]
                    if len(children_) > 0:
                        to_visit.extendleft(children_)

            if current_name in disabled:
                continue

            if active and current_value != current_value:
                default_value = current.normalized_default_value
                configuration_array[current_idx] = default_value
                children_ = configuration_space.children_of[current_name]
                if len(children_) > 0:
                    to_visit.extendleft(children_)

            # If the hyperparameter was made inactive,
            # all its children need to be deactivade as well
            if not active and current_value == current_value:
                configuration_array[current_idx] = np.nan

                children = configuration_space.children_of[current_name]

                if len(children) > 0:
                    to_disable = set()
                    for ch in children:
                        to_disable.add(ch.name)
                    while len(to_disable) > 0:
                        child = to_disable.pop()
                        child_idx = configuration_space.index_of[child]
                        disabled.append(child_idx)
                        children = configuration_space.children_of[child]

                        for ch in children:
                            to_disable.add(ch.name)

    for idx in disabled:
        if idx not in hps_to_be_activate:
            configuration_array[idx] = np.nan

    return configuration_array


def generate_grid(
    configuration_space: ConfigurationSpace,
    num_steps_dict: dict[str, int] | None = None,
) -> list[Configuration]:
    """Generates a grid of Configurations for a given ConfigurationSpace.
    Can be used, for example, for grid search.

    Parameters
    ----------
    configuration_space: :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The Configuration space over which to create a grid of HyperParameter
        Configuration values. It knows the types for all parameter values.

    num_steps_dict: dict
        A dict containing the number of points to divide the grid side formed by
        Hyperparameters which are either of type UniformFloatHyperparameter or
        type UniformIntegerHyperparameter. The keys in the dict should be the names
        of the corresponding Hyperparameters and the values should be the number of
        points to divide the grid side formed by the corresponding Hyperparameter in to.

    Returns:
    -------
    list
        List containing Configurations. It is a cartesian product of tuples of
        HyperParameter values.
        Each tuple lists the possible values taken by the corresponding HyperParameter.
        Within the cartesian product, in each element, the ordering of HyperParameters
        is the same for the OrderedDict within the ConfigurationSpace.
    """

    def get_value_set(num_steps_dict: dict[str, int] | None, hp_name: str) -> tuple:
        """Gets values along the grid for a particular hyperparameter.

        Uses the num_steps_dict to determine number of grid values for
        UniformFloatHyperparameter and UniformIntegerHyperparameter. If these values
        are not present in num_steps_dict, the quantization factor, q, of these
        classes will be used to divide the grid. NOTE: When q is used if it
        is None, a ValueError is raised.

        Parameters
        ----------
        num_steps_dict: dict
            Same description as above

        hp_name: str
            Hyperparameter name

        Returns:
        -------
        tuple
            Holds grid values for the given hyperparameter

        """
        param = configuration_space[hp_name]
        if isinstance(param, (CategoricalHyperparameter)):
            return cast(tuple, param.choices)

        if isinstance(param, (OrdinalHyperparameter)):
            return cast(tuple, param.sequence)

        if isinstance(param, Constant):
            return (param.value,)

        if isinstance(param, UniformFloatHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None and param.name in num_steps_dict:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                raise ValueError(
                    "num_steps_dict is None or doesn't contain the number of points"
                    f" to divide {param.name} into. And its quantization factor "
                    "is None. Please provide/set one of these values.",
                )

            if param.log:
                grid_points = np.exp(grid_points)

            # Avoiding rounding off issues
            if grid_points[0] < param.lower:
                grid_points[0] = param.lower
            if grid_points[-1] > param.upper:
                grid_points[-1] = param.upper

            return tuple(grid_points)

        if isinstance(param, UniformIntegerHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None and param.name in num_steps_dict:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                raise ValueError(
                    "num_steps_dict is None or doesn't contain the number of points "
                    f"to divide {param.name} into. And its quantization factor "
                    "is None. Please provide/set one of these values.",
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

        raise TypeError(f"Unknown hyperparameter type {type(param)}")

    def get_cartesian_product(
        value_sets: list[tuple],
        hp_names: list[str],
    ) -> list[dict[str, Any]]:
        """Returns a grid for a subspace of the configuration with given hyperparameters
        and their grid values.

        Takes a list of tuples of grid values of the hyperparameters and list of
        hyperparameter names. The outer list iterates over the hyperparameters
        corresponding to the order in the list of hyperparameter names.
        The inner tuples contain grid values of the hyperparameters for each
        hyperparameter.

        Parameters
        ----------
        value_sets: list of tuples
            Same description as return value of get_value_set()

        hp_names: list of strs
            List of hyperparameter names

        Returns:
        -------
        list of dicts
            List of configuration dicts
        """
        import itertools

        if len(value_sets) == 0:
            # Edge case
            return []

        grid = []
        for element in itertools.product(*value_sets):
            config_dict = dict(zip(hp_names, element, strict=False))
            grid.append(config_dict)

        return grid

    # Each tuple within is the grid values to be taken on by a Hyperparameter
    value_sets = []
    hp_names = []

    # Get HP names and allowed grid values they can take for the HPs at the top
    # level of ConfigSpace tree
    for hp_name in configuration_space.unconditional_hyperparameters:
        value_sets.append(get_value_set(num_steps_dict, hp_name))
        hp_names.append(hp_name)

    # Create a Cartesian product of above allowed values for the HPs. Hold them in an
    # "unchecked" deque because some of the conditionally dependent HPs may become
    # active for some of the elements of the Cartesian product and in these cases
    # creating a Configuration would throw an Error (see below).
    # Creates a deque of Configuration dicts
    unchecked_grid_pts = deque(get_cartesian_product(value_sets, hp_names))
    checked_grid_pts = []

    while len(unchecked_grid_pts) > 0:
        try:
            grid_point = Configuration(
                configuration_space,
                values=unchecked_grid_pts[0],
            )
            checked_grid_pts.append(grid_point)

        # When creating a configuration that violates a forbidden clause, simply skip it
        except ForbiddenValueError:
            unchecked_grid_pts.popleft()
            continue

        except ActiveHyperparameterNotSetError:
            value_sets = []
            hp_names = []
            new_active_hp_names = []

            # "for" loop over currently active HP names
            for hp_name in unchecked_grid_pts[0]:
                value_sets.append((unchecked_grid_pts[0][hp_name],))
                hp_names.append(hp_name)
                # Checks if the conditionally dependent children of already active
                # HPs are now active
                # TODO: Shorten this
                for new_hp_name in configuration_space.dag.nodes[hp_name].children:
                    if (
                        new_hp_name not in new_active_hp_names
                        and new_hp_name not in unchecked_grid_pts[0]
                    ):
                        all_cond_ = True
                        for cond in configuration_space.parent_conditions_of[
                            new_hp_name
                        ]:
                            if not cond.satisfied_by_value(unchecked_grid_pts[0]):
                                all_cond_ = False
                        if all_cond_:
                            new_active_hp_names.append(new_hp_name)

            for hp_name in new_active_hp_names:
                value_sets.append(get_value_set(num_steps_dict, hp_name))
                hp_names.append(hp_name)

            # this check might not be needed, as there is always going to be a new
            # active HP when in this except block?
            if len(new_active_hp_names) <= 0:
                raise RuntimeError(
                    "Unexpected error: There should have been a newly activated"
                    " hyperparameter for the current configuration values:"
                    f" {unchecked_grid_pts[0]!s}. Please contact the developers with"
                    " the code you ran and the stack trace.",
                ) from None

            new_conditonal_grid = get_cartesian_product(value_sets, hp_names)
            unchecked_grid_pts += new_conditonal_grid
        unchecked_grid_pts.popleft()

    return checked_grid_pts
