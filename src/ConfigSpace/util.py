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
import itertools
import math
from collections import deque
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast, Generator

import numpy as np

from ConfigSpace import Configuration
from ConfigSpace.exceptions import (
    ActiveHyperparameterNotSetError,
    ForbiddenValueError,
    IllegalValueError,
    IllegalVectorizedValueError,
    InactiveHyperparameterSetError,
    NoPossibleNeighborsError,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.types import NotSet

if TYPE_CHECKING:
    from ConfigSpace.configuration_space import ConfigurationSpace
    from ConfigSpace.types import Array, f64, i64


def impute_inactive_values(
    configuration: Configuration,
    strategy: str | float = "default",
) -> Configuration:
    """Impute inactive parameters.

    Iterate through the hyperparameters of a `Configuration` and set the
    values of the inactive hyperparamters to their default values if the choosen
    `strategy` is 'default'. Otherwise `strategy` contains a float number.
    Set the hyperparameters' value to this number.


    Args:
        configuration: For this configuration inactive values will be imputed.
        strategy:
            The imputation strategy. Defaults to 'default'
            If 'default', replace inactive parameters by their default.
            If float, replace inactive parameters by the given float value,
            which should be able to be splitted apart by a tree-based model.

    Returns:
        A new configuration with the imputed values. In this new configuration
        inactive values are included.
    """
    values = {}
    for hp in configuration.config_space.values():
        value = configuration.get(hp.name, NotSet)
        if value is NotSet:
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
    seed: int | np.random.RandomState,
    num_neighbors: int = 4,
    stdev: float = 0.2,
) -> Iterator[Configuration]:
    """Return all configurations in a one-exchange neighborhood.

    The method is implemented as defined by:
    Frank Hutter, Holger H. Hoos and Kevin Leyton-Brown
    Sequential Model-Based Optimization for General Algorithm Configuration
    In Proceedings of the conference on Learning and Intelligent
    Optimization(LION 5)

    Args:
        configuration: for this Configuration object `num_neighbors` neighbors
            are computed
        seed: Sets the random seed to a fixed value
        num_neighbors: number of configurations, which are sampled from
            the neighbourhood of the input configuration
        stdev: The standard deviation is used to determine the neigbours of
            hyperparameters which are continuous/integer

    Returns:
         It contains configurations, with values being situated around
         the given configuration.

    """
    OVER_SAMPLE_CONTINUOUS_MULT = 5
    space = configuration.config_space
    config = configuration
    arr = configuration._vector
    dag = space._dag

    # neighbor_sample_size: How many neighbors we should sample for a given
    #   hyperparameter at once.
    # max_iter_per_selection: How many times we loop trying to generate a valid
    #   configuration with a given hyperparameter, every time it gets sampled. If
    #   not a single valid configuration is generated in this many iterations, it's
    #   marked as failed.
    # std: The standard deviation to use for the neighborhood of a hyperparameter when
    #   sampling neighbors.
    # should_shuffle: Whether or not we should shuffle the neighbors of a hyperparameter
    #   once generated
    # generated: Whether or not we have already generated the neighbors for this
    #   hyperparameter, set to false until sampled.
    # should_regen: Whether or not we should regenerate more neighbors for this
    #   hyperparameter at all.
    # -> dict[HP, (neighbor_sample_size, std, should_shuffle, generated, should_regen)]
    sample_strategy: dict[str, tuple[int, int, float | None, bool, bool, bool]] = {}

    # n_to_gen: Per hyperparameter, how many configurations we should generate with this
    #   hyperparameter as the one where the values change.
    # neighbors_generated_for_hp: The neighbors that were generated for this hp that can
    #   be retrieved.
    # -> tuple[HP, hp_idx, n_to_gen, neighbors_generated_for_hp]
    neighbors_to_generate: list[tuple[Hyperparameter, int, int, list[f64]]] = []

    nan_hps = np.isnan(arr)
    UFH = UniformFloatHyperparameter
    UIH = UniformIntegerHyperparameter
    n_randints_to_gen = 0
    for hp_name, node in dag.nodes.items():
        hp = node.hp
        hp_idx = node.idx

        # inactive hyperparameters skipped
        # hps with a size of one can't be modified to a neighbor
        # This catches Constants, single value categoricals and ordinals (ints?)
        if hp.size == 1 or nan_hps[hp_idx]:
            continue

        if isinstance(hp, CategoricalHyperparameter):
            neighbor_sample_size = hp.size - 1
            # NOTE: We ignore argument `num_neighbors` for Categoricals,
            # don't know why
            n_to_gen = neighbor_sample_size
            max_iter_per_selection = neighbor_sample_size
            _std = None
            should_shuffle = True
            should_regen = False
        elif isinstance(hp, OrdinalHyperparameter):
            neighbor_sample_size = int(hp.get_num_neighbors(config[hp_name]))
            # NOTE: We can only generate maximum 2 neighbors for Ordinals
            # so we just generate all possible ones.
            _std = None
            n_to_gen = neighbor_sample_size
            max_iter_per_selection = neighbor_sample_size
            should_shuffle = True
            should_regen = False
        elif np.isinf(hp.size):  # All continuous ones
            # We can oversample here as there are an infinite number of unique neighbors
            # by oversampling, we can hopefully avoid regeneration of neighbors.
            neighbor_sample_size = num_neighbors * OVER_SAMPLE_CONTINUOUS_MULT
            n_to_gen = num_neighbors
            # NOTE: Not sure it should be this high without increasing the std of
            # neighborhood sampling.
            max_iter_per_selection = max(neighbor_sample_size, 100)
            _std = stdev if isinstance(hp, UFH) else None
            should_shuffle = False
            should_regen = True
        else:  # All non-continuous ones
            # We don't want to over sample a finite hyperparameter as by specifying
            # a large number of neighbors, we end up sampling the entire hyperparameter
            # range, not just it's immediate neighbors.
            _possible_neighbors = int(hp.size - 1)
            neighbor_sample_size = int(min(num_neighbors, _possible_neighbors))
            n_to_gen = num_neighbors
            # NOTE: Not sure it should be this high without increasing the std of
            # neighborhood sampling.
            max_iter_per_selection = neighbor_sample_size
            _std = stdev if isinstance(hp, UIH) else None
            should_shuffle = True
            should_regen = _possible_neighbors >= num_neighbors

        n_forbiddens_on_hp = len(dag.forbidden_lookup.get(hp_name, []))
        hueristic_multiplier = 1 + np.sqrt(n_forbiddens_on_hp)
        n_randints_to_gen += int(n_to_gen * hueristic_multiplier)

        generated = False
        sample_strategy[hp_name] = (
            neighbor_sample_size,
            max_iter_per_selection,
            _std,
            should_shuffle,
            generated,
            should_regen,
        )
        neighbors_to_generate.append((hp, hp_idx, n_to_gen, []))

    random = np.random.RandomState(seed) if isinstance(seed, int) else seed

    arr = config.get_array()

    assert not any(n_to_gen == 0 for _, _, n_to_gen, _ in neighbors_to_generate)

    # Generate some random integers based on the number of neighbors
    # we need to generate and number of forbiddens
    n_hps = len(neighbors_to_generate)
    integers = random.randint(n_hps, size=n_randints_to_gen).tolist()
    _randint_idx = 0

    # Keep looping until we have used all hyperparameters
    n_hps_left_to_exhuast = n_hps
    while n_hps_left_to_exhuast > 0:
        # Our random int's ran out, make more
        if _randint_idx >= n_randints_to_gen:
            # If we got here, we don't need to generate so many more new ones
            n_randints_to_gen = len(neighbors_to_generate) * n_hps * 2
            integers = random.randint(n_hps, size=n_randints_to_gen).tolist()
            _randint_idx = 0

        chosen_hp_idx: int = integers[_randint_idx]
        _randint_idx += 1

        hp, hp_idx, n_left, neighbors = neighbors_to_generate[chosen_hp_idx]
        hp_name = hp.name

        if n_left == 0:
            continue

        neighbor_config: Configuration | None = None

        (
            neighbor_sample_size,
            max_iter_per_selection,
            _std,
            _should_shuffle,
            _generated,
            _should_regen,
        ) = sample_strategy[hp_name]

        for _ in range(max_iter_per_selection):
            # Generate neighbors if we don't have any for this hp.
            if len(neighbors) == 0:
                # All possible neighbors of the hp were generated before and were
                # exhausted, no point in trying it again...
                if _generated and not _should_regen:
                    n_hps_left_to_exhuast -= 1
                    neighbors_to_generate[chosen_hp_idx] = (hp, hp_idx, 0, neighbors)
                    break

                # We should never resample something that has already had all it's
                # neighbors sampled.
                vec = arr[hp_idx]
                _neighbors = hp._neighborhood(
                    vec,
                    n=neighbor_sample_size,
                    seed=random,
                    std=_std,
                )

                # Inf sized hp's are already basically shuffled. This is more for
                # finite hps which may give a linear ordering of neighbors...
                if _should_shuffle:
                    random.shuffle(_neighbors)

                neighbors = _neighbors.tolist()
                neighbors_to_generate[chosen_hp_idx] = (hp, hp_idx, n_left, neighbors)

                # Update to say it's been `generated`
                sample_strategy[hp_name] = (
                    neighbor_sample_size,
                    max_iter_per_selection,
                    _std,
                    _should_shuffle,
                    True,  # generated
                    _should_regen,
                )

            neighbor_vector_val = neighbors.pop()

            new_arr = change_hp_value(
                configuration_space=space,
                configuration_array=arr.copy(),
                hp_name=hp_name,
                hp_value=neighbor_vector_val,
                index=hp_idx,
            )

            is_valid = True
            for forbidden in space._dag.forbidden_lookup.get(hp_name, []):
                if forbidden.is_forbidden_vector(new_arr):
                    is_valid = False
                    break

            if not is_valid:  # Next retry
                continue

            # If we manager to not have a forbidden, we can yield the configuration
            # and mark there's one less configuration needed to generate for this hp
            neighbor_config = Configuration(space, vector=new_arr)
            one_less = n_left - 1
            neighbors_to_generate[chosen_hp_idx] = (hp, hp_idx, one_less, neighbors)
            if one_less == 0:
                n_hps_left_to_exhuast -= 1

            yield neighbor_config
            break  # Break out of the outer for loop

        else:
            # Never broke out of the retry loop, so we mark the hp as failed, moving on
            n_hps_left_to_exhuast -= 1
            neighbors_to_generate[chosen_hp_idx] = (hp, hp_idx, 0, neighbors)


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

    Args:
        configuration: a configuration for which a random neigbour is calculated
        seed: Used to generate a random state.

    Returns:
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

    Args:
        configuration: a configuration as a dictionary.

            * Key: name of the hyperparameter.
            * Value: value of this hyperparamter

        configuration_space:
            The defined configuration space. It is necessary to find the inactive
            hyperparameters by iterating through the conditions of the configuration space.
        vector:
            fficient represantation of a configuration. Either `configuration` or
            `vector` must be specified. If both are specified only
            `configuration` will be used.

    Returns:
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

    Args:
    configuration:
        A configuration as a dictionary.

        * Key: name of the hyperparameter.
        * Value: value of this hyperparamter

    configuration_space:
        Configuration space which knows the types for all parameter values

    Returns:
        Configuration with fixed types of parameter values
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
    #yield_all_unset_active_hyperparameters: bool = False,
) -> None:
    activated = np.isfinite(vector)
    #unset_active_hps: list[Hyperparameter] = []

    # Make sure the roots are all good
    for root in space._dag.roots.values():
        hp_idx = root.idx
        if not activated[hp_idx]:
            #if not yield_all_unset_active_hyperparameters:
            raise ActiveHyperparameterNotSetError(root.hp)
            #else:
            #    unset_active_hps.append(hp)

    for cnode in space._dag.minimum_conditions:
        # Everything for the condition is satisfied, make sure active
        # hyperparameters are set, legal and not forbidden
        children_idxs = cnode.children_indices
        if cnode.condition.satisfied_by_vector(vector):
            active_mask = activated[children_idxs]
            if not active_mask.all():
                idx: int = children_idxs[~active_mask][0]
                hp_name = space.at[idx]
                hp = space[hp_name]
                #if not yield_all_unset_active_hyperparameters:
                raise ActiveHyperparameterNotSetError(hp)
                #else:
                #    unset_active_hps.append(hp)                 

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

    for forbidden in space._dag.fast_forbidden_checks:
        if forbidden.is_forbidden_vector(vector):
            raise ForbiddenValueError(
                f"Given vector violates forbidden clause: {forbidden}",
            )

    # All checks passed, except for possible plural ActiveHyperparameterNotSetError
    #if unset_active_hps:
    #    raise ActiveHyperparametersNotSetError(unset_active_hps)


def change_hp_value(  # noqa: D103
    configuration_space: ConfigurationSpace,
    configuration_array: Array[f64],
    hp_name: str,
    hp_value: float | f64,
    index: int | i64,
) -> Array[f64]:
    space = configuration_space
    arr = configuration_array
    arr[index] = hp_value
    dag = space._dag
    defaults = dag.normalized_defaults

    for dep in dag.change_hp_lookup.get(hp_name, []):
        condition = dep.condition
        child_idxs = dep.children_indices
        if condition.satisfied_by_vector(arr):
            # Get indices of nan children
            children = arr[child_idxs]
            nan_mask = np.isnan(children)
            nan_idx = child_idxs[nan_mask]

            # Assign them to defaults
            arr[nan_idx] = defaults[nan_idx]
        else:
            arr[child_idxs] = dep.nan_arr

    return arr


def grid_generator(
    configuration_space: ConfigurationSpace,
    num_steps_dict: dict[str, int] | None = None,
) -> Generator[Configuration, None, None]:
    """Generates a grid of Configurations for a given ConfigurationSpace.
    Can be used, for example, for grid search.

    Args:
    configuration_space:
        The Configuration space over which to create a grid of HyperParameter
        Configuration values. It knows the types for all parameter values.

    num_steps_dict:
        A dict containing the number of points to divide the grid side formed by
        Hyperparameters which are either of type UniformFloatHyperparameter or
        type UniformIntegerHyperparameter. The keys in the dict should be the names
        of the corresponding Hyperparameters and the values should be the number of
        points to divide the grid side formed by the corresponding Hyperparameter in to.

    Returns:
        A generator producing Configurations for a given ConfigurationSpace as a cartesian product of tuples of HyperParameter values.
        It is a cartesian product of tuples, where each tuple lists the possible values taken by the corresponding HyperParameter.
        Within the cartesian product, in each element, the ordering of HyperParameters is the same for the OrderedDict within the ConfigurationSpace.
    """

    # Idea; we can perhaps create a generator for each HP, to avoid taking the entire grid into memory
    # Then we can draw for each HP a value from each generator and test the yielded configuration (masking out the HP values that actually should be inactive)
    # For each combination that **could** result in a duplicate (due to active vs inactive HPs), we need to store a light weight hash of the configuration
    # That we can check each time s.t. we can quickly skip over combinations that are known to be duplicates
    # 1. Build a generator for each HP based on their min/max and step size
    # 2. This generator allows us to build a 'cartesian product' generator s.t. all combinations are made (including inactive HPs....)
    # 3. It would be best if we could make the HPs generate values for active HPs only when applicable but this is complicated due to not knowing the dependency order
    # 4. ??
    # 5. Profit

    def _hyperparameter_range(hp: Hyperparameter, num_steps: int) -> range | tuple | Generator:
        """Constructs the range of the hyperparameter or tuple for categorical / ordinal hyperparameters and constants."""
        
        def frange(lower: float, upper: float, numsteps: int, log: bool=False, as_int: bool=False, conditional: bool=False) -> Generator[float, None, None]:
            """For some reason this does not exist by default in Python, and Numpy returns arrays instead of generators."""
            if log:
                lower_source, upper_source = lower, upper
                lower, upper = math.log(lower), math.log(upper)
            x = lower  # Starting point
            step_size = float((upper - lower) / (numsteps-1))
            if not log:  # Determine precision
                precision = len(str(step_size).split(".", maxsplit=1)[1])  # This is so ugly...
            while x <= upper:
                if log:  # Capping for float rounding errors
                    # NOTE: What if the capping is now letting through a final value that was originally waaaaaay out of bounds? Should it not be rejected?
                    value = min(max(math.exp(x), lower_source), upper_source)
                    if as_int:
                        value = round(value)
                else:
                    value = round(x) if as_int else x
                yield value
                x += step_size
                if not log:  # Linear, thus we can make the precision to be the same as the step_size for accuracy purposes
                    x = round(x, precision)
            #if conditional:
            #    yield NotSet  # Include the 'inactive' option

        conditional_hp = hp.name in configuration_space.conditional_hyperparameters
        if isinstance(hp, (CategoricalHyperparameter)):
            #return cast(tuple, list(hp.choices) + [NotSet] if conditional_hp else hp.choices)
            return cast(tuple, hp.choices)
        elif isinstance(hp, (OrdinalHyperparameter)):
            #return cast(tuple, list(hp.sequence) + [NotSet] if conditional_hp else hp.sequence)
            return cast(tuple, hp.sequence)
        elif isinstance(hp, Constant):
            #return (hp.value, NotSet) if conditional_hp else (hp.value,)
            return (hp.value,)
        elif num_steps is None:  # The latter two hyperparameter require a number of steps, do a quick check if to see if we can proceed
            raise ValueError(f"No number of steps provided for {hp.name} i.e. the number of points to divide {hp.name} into.")
        elif isinstance(hp, UniformIntegerHyperparameter):
            return frange(hp.lower, hp.upper, num_steps, log=hp.log, as_int=True, conditional=conditional_hp)
        elif isinstance(hp, UniformFloatHyperparameter):
            return frange(hp.lower, hp.upper, num_steps, log=hp.log, conditional=conditional_hp)
        raise TypeError(f"Unknown hyperparameter type {type(hp)}")

    def _cartesian_product_generator(hps: list[Hyperparameter]) -> Generator[tuple, None, None]:
        """Constructs a generator that produces a cartesian product of the Hyperparameter values."""
        hp_ranges = [_hyperparameter_range(hp, num_steps_dict.get(hp.name, None) if num_steps_dict else None) for hp in hps]
        if not hp_ranges:
            # Itertools.product returns an empty tuple if hp_ranges is empty, to prevent this we check if the list contains anything before unpacking
            return itertools.product([])
        return itertools.product(*hp_ranges)

    hyperparameter_names = list(configuration_space.keys())
    regular_hyperparameters = [
        hp
        for hp in configuration_space.values()
        if hp.name not in configuration_space.conditional_hyperparameters
    ]

    # hyperparameters = [hp for hp in configuration_space.values() if hp.name not in configuration_space.conditional_hyperparameters]
    # hyperparameter_names = [hp.name for hp in hyperparameters]

    def generate_with_conditionals(regular_configuration: dict[str, Any], active_conditionals: list[Hyperparameter]) -> Generator[Configuration, None, None]:
        """Recursively adds all conditional hyperparameters to some configuration of regular HPs."""
        for conditional_configuration in _cartesian_product_generator(active_conditionals):
            new_configuration = regular_configuration.copy()# + conditional_configuration
            for hp, value in zip(active_conditionals, conditional_configuration):  # Combine the existing configuration with new conditional values
                new_configuration[hp.name] = value
            try:
                grid_point = Configuration(
                    configuration_space,
                    values=new_configuration,
                )
                yield grid_point
            except ActiveHyperparameterNotSetError as ex:
                for configuration_with_conditionals in generate_with_conditionals(new_configuration, [ex.hyperparameter]):
                    yield configuration_with_conditionals
            except ForbiddenValueError as ex:  # The grid generator generates all possible combinations, including those violating the Forbidden rules
                continue
            except InactiveHyperparameterSetError as ex:  # This should not happen?
                raise ex
            except IllegalValueError as ex:  # Should not occur: The grid should only generate legal values for each HP.
                raise ex

    for configuration in _cartesian_product_generator(regular_hyperparameters):
        configuration_dict = {key: value for key, value in zip(hyperparameter_names, configuration)}
        try:
            # NOTE: Build vector instead and call check_configuration here directly?
            grid_point = Configuration(
                configuration_space,
                values=configuration_dict,
            )
            yield grid_point
        except ActiveHyperparameterNotSetError as ex:
            # NOTE: We are not getting all possible known ActiveHyperparameterNotSetErrors at once here; its thrown for the first 'mistake' only.
            for configuration_with_conditionals in generate_with_conditionals(configuration_dict, [ex.hyperparameter]):
                yield configuration_with_conditionals
        except ForbiddenValueError as ex:  # The grid generator generates all possible combinations, including those violating the Forbidden rules
            continue
        except InactiveHyperparameterSetError as ex:  # This should not occur due to how conditionals are handled
            raise ex
        except IllegalValueError as ex:  # Should not occur: The grid should only generate legal values for each HP.
            raise ex
