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
import numpy.typing as npt

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

    Returns
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

    Returns
    -------
    Iterator
         It contains configurations, with values being situated around
         the given configuration.

    """
    space = configuration.config_space
    config = configuration
    arr = configuration._vector

    # Define how many neighbors and std.dev we should sample for a given hyperparameter
    sample_strategy: dict[str, tuple[int, float | None, bool]] = {}

    # Define the total number of neighbors we should generate for any given
    # hyperparameter
    neighbors_to_generate: list[tuple[Hyperparameter, int, list[np.float64]]] = []

    for hp in space.values():
        hp_idx = space.index_of[hp.name]

        # inactive hyperparameters skipped
        # hps with a size of one can't be modified to a neighbor
        if np.isnan(arr[hp_idx]) or hp.size == 1:
            continue

        if isinstance(hp, NumericalHyperparameter) and hp.size > num_neighbors:
            UFH = UniformFloatHyperparameter
            UIH = UniformIntegerHyperparameter
            _std = stdev if isinstance(hp, (UFH, UIH)) else None

            # For numericals we just cap it at the number of neighbors
            should_shuffle = not np.isinf(hp.size)
            sample_strategy[hp.name] = (int(min(20, hp.size)), _std, should_shuffle)
            neighbors_to_generate.append((hp, num_neighbors, []))
        else:
            # Otherwise, it's either a cateogircal/oridinal (allow all), or a numerical
            # with some number of neighbors less than request
            value = config.get(hp.name)
            n_possible_neighbors = int(hp.get_num_neighbors(value))
            sample_strategy[hp.name] = (n_possible_neighbors, None, False)
            neighbors_to_generate.append((hp, n_possible_neighbors, []))

    random = np.random.RandomState(seed)
    arr = config.get_array()

    assert not any(n_to_gen == 0 for _, n_to_gen, _ in neighbors_to_generate)

    # Generate some random integers based on the number of neighbors
    # we need to generate and number of forbiddens
    sum_total_to_gen = sum(n_to_gen for _, n_to_gen, _ in neighbors_to_generate)
    _size = (
        sum_total_to_gen
        * len(neighbors_to_generate)
        * int(np.sqrt(len(space.forbidden_clauses)))
    )
    n_hps = len(neighbors_to_generate)
    integers = random.randint(n_hps, size=_size)
    _ridx = 0

    # Keep looping until we have used all hyperparameters
    n_hps_left_to_exhuast = len(neighbors_to_generate)
    while n_hps_left_to_exhuast > 0:
        # Our random int's ran out, make more
        if _ridx >= _size:
            # If we got here, we don't need to generate so many more new ones
            _size = len(neighbors_to_generate) * sum_total_to_gen * 2
            integers = random.randint(n_hps, size=_size)
            _ridx = 0

        chosen_hp_idx: int = integers[_ridx]
        _ridx += 1

        hp, n_left, neighbors = neighbors_to_generate[chosen_hp_idx]
        hp_name = hp.name
        hp_idx = space.index_of[hp_name]

        if n_left == 0:
            continue

        neighbor_config: Configuration | None = None

        # The only way we escape this loop and return a value is if we break
        for _ in range(100):
            if len(neighbors) == 0:
                _ns, _std, _should_shuffle = sample_strategy[hp_name]
                vec = arr[hp_idx]
                neighbors = hp._neighborhood(vec, n=_ns, seed=random, std=_std)
                if _should_shuffle:
                    # Inf sized hp's are already basically shuffled. This is more for
                    # finite hps which may give a linear ordering of neighbors...
                    random.shuffle(neighbors)

                neighbors = list(neighbors)
                neighbors_to_generate[chosen_hp_idx] = (hp, n_left, neighbors)

            neighbor_vector_val = neighbors.pop()

            new_arr = change_hp_value(
                configuration_space=space,
                configuration_array=arr.copy(),
                hp_name=hp_name,
                hp_value=neighbor_vector_val,
                index=hp_idx,
            )
            for forbidden in space.dag.forbidden_lookup.get(hp_name, []):
                if forbidden.is_forbidden_vector(new_arr):
                    break
            else:
                neighbor_config = Configuration(space, vector=new_arr)
                one_less = n_left - 1
                neighbors_to_generate[chosen_hp_idx] = (hp, one_less, neighbors)
                if one_less == 0:
                    n_hps_left_to_exhuast -= 1
                yield neighbor_config
                break
        else:
            # We didn't manage to break the for loop, choose the next hp
            n_hps_left_to_exhuast -= 1
            neighbors_to_generate[chosen_hp_idx] = (hp, 0, neighbors)


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

    Returns
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
                if hp.size <= 1:
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

    Returns
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

    Returns
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


def check_configuration(  # noqa: D103
    space: ConfigurationSpace,
    vector: np.ndarray,
    allow_inactive_with_values: bool = False,
) -> None:
    activated = np.isfinite(vector)

    # Make sure the roots are all good
    for root in space.dag.roots.values():
        hp_idx = root.idx
        if not activated[hp_idx]:
            raise ActiveHyperparameterNotSetError(root.hp)

    for cnode in space.dag.minimum_conditions:
        # Everything for the condition is satisfied, make sure active
        # hyperparameters are set, legal and not forbidden
        children_idxs = cnode.children_vector
        if cnode.condition.satisfied_by_vector(vector):
            active_mask = activated[children_idxs]
            if not active_mask.all():
                idx: int = children_idxs[~active_mask][0]
                hp_name = space.at[idx]
                hp = space[hp_name]
                raise ActiveHyperparameterNotSetError(hp)

            for hp_idx, hp_node in cnode.unique_children.items():
                # OPTIM: We bypass the larger safety checking of the hp and access
                # the underlying transformer directly
                transformer = hp_node.hp._transformer
                if not transformer.legal_vector_single(vector[hp_idx]):
                    raise IllegalVectorizedValueError(hp_node.hp, vector[hp_idx])

        elif not allow_inactive_with_values:
            active_mask = activated[children_idxs]
            if active_mask.any():
                idx = children_idxs[active_mask][0]
                hp_name = space.at[idx]
                hp = space[hp_name]
                raise InactiveHyperparameterSetError(hp, hp.to_value(vector[idx]))

    for forbidden in space.dag.fast_forbidden_checks:
        if forbidden.is_forbidden_vector(vector):
            raise ForbiddenValueError(
                f"Given vector violates forbidden clause: {forbidden}",
            )


def change_hp_value(  # noqa: D103
    configuration_space: ConfigurationSpace,
    configuration_array: npt.NDArray[np.float64],
    hp_name: str,
    hp_value: float | np.float64,
    index: int | np.int64,
) -> npt.NDArray[np.float64]:
    space = configuration_space
    arr = configuration_array
    arr[index] = hp_value

    dependants = space.dag.change_hp_lookup.get(hp_name)

    if dependants is None:
        return arr

    for dep in dependants:
        condition = dep.condition
        child_idxs = dep.children_vector
        if condition.satisfied_by_vector(arr):
            children = arr[child_idxs]
            arr[child_idxs] = np.where(
                np.isnan(children),
                space.dag.normalized_defaults[child_idxs],
                children,
            )
        else:
            arr[child_idxs] = np.nan

    return arr


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

    Returns
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

        Returns
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

        Returns
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
            config_dict = dict(zip(hp_names, element))
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
