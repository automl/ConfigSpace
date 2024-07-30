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
import io
import json
import warnings
from collections.abc import ItemsView, Iterable, Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, Mapping, Sequence, overload
from typing_extensions import deprecated

import numpy as np

import ConfigSpace.util
from ConfigSpace._condition_tree import DAG
from ConfigSpace.conditions import (
    Condition,
    ConditionLike,
    Conjunction,
    EqualsCondition,
)
from ConfigSpace.configuration import Configuration, NotSet
from ConfigSpace.exceptions import (
    ActiveHyperparameterNotSetError,
    ForbiddenValueError,
    IllegalVectorizedValueError,
    InactiveHyperparameterSetError,
)
from ConfigSpace.forbidden import (
    ForbiddenClause,
    ForbiddenConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    ForbiddenLike,
    ForbiddenRelation,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.hyperparameters.hyperparameter import NumericalHyperparameter
from ConfigSpace.read_and_write.dictionary import (
    CONDITION_DECODERS,
    CONDITION_ENCODERS,
    FORBIDDEN_DECODERS,
    FORBIDDEN_ENCODERS,
    HYPERPARAMETER_DECODERS,
    HYPERPARAMETER_ENCODERS,
)
from ConfigSpace.types import Array, Mask, f64

if TYPE_CHECKING:
    from ConfigSpace.read_and_write.dictionary import _Decoder, _Encoder


def _parse_hyperparameters_from_dict(
    items: Mapping[str, Any],
) -> Iterator[Hyperparameter]:
    for name, hp in items.items():
        # Anything that is a Hyperparameter already is good
        # Note that we discard the key name in this case in favour
        # of the name given in the dictionary
        if isinstance(hp, Hyperparameter):
            yield hp

        # Tuples are bounds, check if float or int
        elif isinstance(hp, tuple):
            if len(hp) != 2:
                raise ValueError(f"'{name}' must be (lower, upper) bound, got {hp}")

            lower, upper = hp
            if isinstance(lower, float):
                yield UniformFloatHyperparameter(name, lower, upper)
            else:
                yield UniformIntegerHyperparameter(name, lower, upper)

        # Lists are categoricals
        elif isinstance(hp, list):
            if len(hp) == 0:
                raise ValueError(f"Can't have empty list for categorical {name}")

            yield CategoricalHyperparameter(name, hp)
        else:
            # It's a constant
            yield Constant(name, hp)


class ConfigurationSpace(Mapping[str, Hyperparameter]):
    """A collection-like object containing a set of hyperparameter definitions and
    conditions.

    A configuration space organizes all hyperparameters and its conditions
    as well as its forbidden clauses. Configurations can be sampled from
    this configuration space. As underlying data structure, the
    configuration space uses a tree-based approach to represent the
    conditions and restrictions between hyperparameters.
    """

    def __init__(
        self,
        name: str | Mapping[str, Any] | None = None,
        seed: int | None = None,
        meta: dict | None = None,
        *,
        space: None
        | (
            Mapping[
                str,
                tuple[int, int]
                | tuple[float, float]
                | Sequence[Any]
                | int
                | float
                | str
                | Hyperparameter,
            ]
            | Sequence[Hyperparameter]
        ) = None,
    ) -> None:
        """Initialize a configuration space.

        Args:
            name:
                Name of the configuration space. If a dict is passed,
                this is considered the same as the `space=` arg.
            seed:
                Random seed
            meta:
                Field for holding meta data provided by the user.
                Not used by the configuration space.
            space:
                A simple configuration space to use:

                ```python exec="true" result="python" source="material-block"
                from ConfigSpace import ConfigurationSpace

                ConfigurationSpace(
                    name="myspace",
                    space={
                        "uniform_integer": (1, 10),
                        "uniform_float": (1.0, 10.0),
                        "categorical": ["a", "b", "c"],
                        "constant": 1337,
                    }
                )
                ```

                You can also use a sequence of hyperparameters:

                ```python exec="true" result="python" source="material-block"
                from ConfigSpace import ConfigurationSpace, Float, Integer

                ConfigurationSpace(
                    name="myspace",
                    space=[Float("a", (1.0, 10.0)), Integer("b", (1, 10))]
                )
                ```

        """
        # If first arg is a dict, we assume this to be `space`
        if isinstance(name, (Mapping, Sequence)) and not isinstance(name, str):
            space = name
            _name = None
        else:
            _name = name

        self.name = _name
        self.meta = meta
        self.random = np.random.RandomState(seed)
        self._dag = DAG()
        self._len = 0

        if space is not None:
            if isinstance(space, Mapping):
                hyperparameters = list(_parse_hyperparameters_from_dict(space))
            elif isinstance(space, Sequence):
                hyperparameters = list(space)
            self.add(hyperparameters)

    @property
    def index_of(self) -> Mapping[str, int]:
        """The index of hyperparameters by their name."""
        return self._dag.index_of

    @property
    def at(self) -> Sequence[str]:
        """The hyperparameters by their index."""
        return self._dag.at

    @property
    def conditions(self) -> Sequence[ConditionLike]:
        """All conditions from the configuration space."""
        return self._dag.conditions

    @property
    def forbidden_clauses(self) -> Sequence[ForbiddenLike]:
        """All forbidden clauses from the configuration space."""
        return self._dag.forbiddens

    @property
    def conditional_hyperparameters(self) -> Sequence[str]:
        """Names of all conditional hyperparameters.

        Returns:
            Set with all hyperparameter names which are only active under certain
            conditions
        """
        return list(self._dag.non_roots)

    @property
    def unconditional_hyperparameters(self) -> Sequence[str]:
        """Names of all unconditional hyperparameters.

        Returns:
            Set of all hyperparameter names which are always active
        """
        return list(self._dag.roots)

    @property
    def children_of(self) -> Mapping[str, Sequence[Hyperparameter]]:
        """Children of a hyperparameter.

        Returns:
            Mapping from a parent hyperparameter name to all hyperparameters which
            are activate, depending on the value of the parent.
        """
        return self._dag.children_of

    @property
    def parents_of(self) -> Mapping[str, Sequence[Hyperparameter]]:
        """Parents of a hyperparameter.

        Returns:
            Mapping from a child hyperparameter name to all hyperparameters which
            activate the child, depending on their values.
        """
        return self._dag.parents_of

    @property
    def child_conditions_of(self) -> Mapping[str, Sequence[ConditionLike]]:
        """Conditions of a hyperparameter.

        Returns:
            Mapping from a parent hyperparameter name to all conditions which
            check the value of the parent.
        """
        return self._dag.child_conditions_of

    @property
    def parent_conditions_of(self) -> Mapping[str, Sequence[ConditionLike]]:
        """Conditions of a hyperparameter.

        Returns:
            Mapping from a child hyperparameter name to all conditions which
            need to be check to activate the child.
        """
        return self._dag.parent_conditions_of

    def add(
        self,
        *args: (
            Hyperparameter
            | ConditionLike
            | ForbiddenLike
            | Iterable[Hyperparameter | ConditionLike | ForbiddenLike]
        ),
    ) -> None:
        """Add a hyperparameter, condition or forbidden clause to the configuration
        space.

        !!! note

            If adding multiple hyperparameters, conditions or forbidden clauses, it
            is better to add them all at once with one call to `add()`, as we
            rebuilt a cache after each call to `add()`.

        Args:
            args:
                Hyperparameter, condition or forbidden clause to add
        """
        # First turn everything into one large iterable
        hps = []
        conditions = []
        forbiddens = []

        def _put_to_list(
            arg: Hyperparameter
            | ConditionLike
            | ForbiddenLike
            | Iterable[Hyperparameter | ConditionLike | ForbiddenLike],
        ) -> None:
            if isinstance(arg, Hyperparameter):
                hps.append(arg)
            elif isinstance(arg, (Condition, Conjunction)):
                conditions.append(arg)
            elif isinstance(
                arg,
                (ForbiddenClause, ForbiddenConjunction, ForbiddenRelation),
            ):
                forbiddens.append(arg)
            elif isinstance(arg, Iterable):
                for a in arg:
                    _put_to_list(a)
            else:
                raise TypeError(f"Unknown type {type(arg)}")

        for a in args:
            _put_to_list(a)

        with self._dag.update():
            for hp in hps:
                self._dag.add(hp)

            for condition in conditions:
                self._dag.add_condition(condition)

            for forbidden in forbiddens:
                self._dag.add_forbidden(forbidden)

        self._len = len(self._dag.nodes)
        self._check_default_configuration()

    def add_configuration_space(
        self,
        prefix: str,
        configuration_space: ConfigurationSpace,
        delimiter: str = ":",
        parent_hyperparameter: dict | None = None,
    ) -> ConfigurationSpace:
        """Combine two configuration space by adding one the other configuration
        space. The contents of the configuration space, which should be added,
        are renamed to `prefix` + `delimiter` + old_name.

        Args:
            prefix:
                The prefix for the renamed hyperparameter | conditions |
                forbidden clauses
            configuration_space:
                The configuration space which should be added
            delimiter:
                Defaults to ':'
            parent_hyperparameter:
                Adds for each new hyperparameter the condition, that
                `parent_hyperparameter` is active. Must be a dictionary with two keys
                "parent" and "value", meaning that the added configuration space is
                active when `parent` is equal to `value`

        Returns:
            The configuration space, which was added.
        """
        prefix_delim = f"{prefix}{delimiter}"

        def _new_name(_item: Hyperparameter) -> str:
            if _item.name in ("", prefix):
                return prefix

            if not _item.name.startswith(prefix_delim):
                return f"{prefix_delim}{_item.name}"

            return _item.name

        new_parameters = []
        for hp in configuration_space.values():
            new_hp = copy.copy(hp)
            new_hp.name = _new_name(hp)
            new_parameters.append(new_hp)

        conditions_to_add = []
        for condition in configuration_space.conditions:
            new_condition = copy.copy(condition)
            cond_dlcs = (
                new_condition.dlcs
                if isinstance(new_condition, Conjunction)
                else [new_condition]
            )
            for cond_dlc in cond_dlcs:
                # Rename children
                cond_dlc.child.name = _new_name(cond_dlc.child)
                cond_dlc.parent.name = _new_name(cond_dlc.parent)

            conditions_to_add.append(new_condition)

        forbiddens_to_add = []
        for forbidden_clause in configuration_space.forbidden_clauses:
            new_forbidden = copy.copy(forbidden_clause)
            forb_dlcs = (
                new_forbidden.dlcs
                if isinstance(new_forbidden, ForbiddenConjunction)
                else [new_forbidden]
            )
            for forb_dlc in forb_dlcs:
                if isinstance(forb_dlc, ForbiddenRelation):
                    forb_dlc.left.name = _new_name(forb_dlc.left)
                    forb_dlc.right.name = _new_name(forb_dlc.right)
                else:
                    forb_dlc.hyperparameter.name = _new_name(forb_dlc.hyperparameter)
            forbiddens_to_add.append(new_forbidden)

        self.add(new_parameters, conditions_to_add, forbiddens_to_add)

        # Finally, we may need to add conditions to the added search space
        conditions_to_add = []
        if parent_hyperparameter is not None:
            parent = parent_hyperparameter["parent"]
            value = parent_hyperparameter["value"]

            # Only add a condition if the parameter is a top-level parameter of the new
            # configuration space (this will be some kind of tree structure).
            root_params = [
                hp for hp in new_parameters if len(self.parents_of[hp.name]) == 0
            ]
            for param in root_params:
                conditions_to_add.append(EqualsCondition(param, parent, value))

        if len(conditions_to_add) > 0:
            self.add(conditions_to_add)

        return configuration_space

    def generate_all_continuous_from_bounds(
        self,
        bounds: Iterable[tuple[float, float]],
    ) -> None:
        """Fill a ConfigurationSpace from a list of bounded numericals.

        The generated hyperparameters are added to the configuration space.

        Args:
            bounds: List containing lists with two elements: lower and upper bound
        """
        self.add(
            UniformFloatHyperparameter(name=f"x{i}", lower=lower, upper=upper)
            for i, (lower, upper) in enumerate(bounds)
        )

    def get_default_configuration(self) -> Configuration:
        """Configuration containing hyperparameters with default values.

        Returns:
            Configuration with the set default values
        """
        return self._check_default_configuration()

    # For backward compatibility
    @deprecated("Please call `configuration.check_valid_configuration()` instead.")
    def check_configuration(self, configuration: Configuration) -> None:
        """Check if a configuration is legal. Raises an error if not.

        Args:
            configuration: Configuration to check
        """
        ConfigSpace.util.check_configuration(self, configuration.get_array())

    def check_configuration_vector_representation(self, vector: Array[f64]) -> None:
        """Raise error if configuration in vector representation is not legal.

        Args:
            vector: configuration in vector representation
        """
        ConfigSpace.util.check_configuration(self, vector)

    def get_active_hyperparameters(
        self,
        configuration: Configuration | Array[f64],
    ) -> set[str]:
        """Set of active hyperparameter names for a given configuration.

        Args:
            configuration: Configuration to get active hyperparameters of

        Returns:
            The set of names of all active hyperparameter
        """
        vector = (
            configuration.get_array()
            if isinstance(configuration, Configuration)
            else configuration
        )
        active_hyperparameters = set()
        for hp_name in self.keys():
            conditions = self.parent_conditions_of[hp_name]

            active = True
            for condition in conditions:
                parent_vector_idx: np.intp | Array[np.intp]
                if isinstance(condition, Conjunction):
                    assert condition.parent_vector_ids is not None
                    parent_vector_idx = condition.parent_vector_ids
                else:
                    parent_vector_idx = np.asarray(condition.parent_vector_id)

                if np.isnan(vector[parent_vector_idx]).any():
                    active = False
                    break

                if not condition.satisfied_by_vector(vector):
                    active = False
                    break

            if active:
                active_hyperparameters.add(hp_name)

        return active_hyperparameters

    @overload
    def sample_configuration(self, size: None = None) -> Configuration: ...

    # Technically this is wrong given the current behaviour but it's
    # sufficient for most cases. Once deprecation warning is up,
    # we can just have `1` always return a list of configurations
    # because an `int` was specified, `None` for single config.
    @overload
    def sample_configuration(self, size: int) -> list[Configuration]: ...

    def sample_configuration(
        self,
        size: int | None = None,
    ) -> Configuration | list[Configuration]:
        """Sample `size` configurations from the configuration space object.

        Args:
            size: Number of configurations to sample. Default (`None`) is to
                sample a single configuration.

        Returns:
            A single configuration if `size` 1 else a list of Configurations
        """
        if len(self) == 0:
            if size is None:
                return Configuration(self, vector=np.array([]))
            return [Configuration(self, vector=np.array([])) for _ in range(size)]

        if size is not None and not isinstance(size, int):
            raise TypeError(f"Expected int or None, got {type(size)}")

        if size == 1:
            warnings.warn(
                "Please leave at default or explicitly set `size=None`."
                " In the future, specifying a size will always return a list, even if"
                " 1",
                DeprecationWarning,
                stacklevel=2,
            )

        # Maintain old behaviour by setting this
        if size is None:
            size = 1

        if size < 1:
            return []

        accepted_configurations: list[Configuration] = []
        num_hyperparameters = len(self)

        # Main sampling loop
        MULT = (
            len(self.forbidden_clauses) + len(self.conditional_hyperparameters)
        ) / num_hyperparameters
        sample_size = size
        while len(accepted_configurations) < size:
            sample_size = max(int(MULT**2 * sample_size), 5)

            # Sample a vector for each hp, filling out columns
            # OPTIM: We put the hyperparameters as rows as we perform row-wise
            # operations and the matrices themselves are row-oriented in memory,
            # helping to improve cache locality.
            config_matrix: Array[f64] = np.empty(
                (num_hyperparameters, sample_size),
                dtype=f64,
            )
            for i, hp in enumerate(self.values()):
                config_matrix[i] = hp._vector_dist.sample_vector(
                    n=sample_size,
                    seed=self.random,
                )

            # Apply unconditional forbiddens across the columns (hps)
            # We treat this as an OR, i.e. if any of the forbidden clauses are
            # forbidden, the entire configuration (row) is forbidden
            uncond_forbidden: Mask = np.zeros(sample_size, dtype=np.bool_)
            for clause in self._dag.unconditional_forbiddens:
                uncond_forbidden |= clause.is_forbidden_vector_array(config_matrix)

            valid_configs = config_matrix[:, ~uncond_forbidden]

            for cnode in self._dag.minimum_conditions:
                condition = cnode.condition
                satisfied = condition.satisfied_by_vector_array(valid_configs)
                valid_configs[np.ix_(cnode.children_indices, ~satisfied)] = np.nan

            # Now we apply the forbiddens that depend on conditionals
            cond_forbidden: Mask = np.zeros(valid_configs.shape[1], dtype=np.bool_)
            for clause in self._dag.conditional_forbiddens:
                cond_forbidden |= clause.is_forbidden_vector_array(valid_configs)

            valid_configs = valid_configs[:, ~cond_forbidden]

            # And now we have a matrix of valid configurations
            accepted_configurations.extend(
                [Configuration(self, vector=vec) for vec in valid_configs.T],
            )
            sample_size = size - len(accepted_configurations)

        if size <= 1:
            return accepted_configurations[0]

        return accepted_configurations[:size]

    def seed(self, seed: int) -> None:
        """Set the random seed to a number.

        Args:
            seed: The random seed
        """
        self.random = np.random.RandomState(seed)

    def remove_hyperparameter_priors(self) -> ConfigurationSpace:
        """Produces a new ConfigurationSpace where all priors on parameters are removed.

        Non-uniform hyperpararmeters are replaced with uniform ones, and
        CategoricalHyperparameters with weights have their weights removed.

        Returns:
            The resulting configuration space, without priors on the hyperparameters
        """
        uniform_config_space = ConfigurationSpace(
            {
                name: p.to_uniform()
                if isinstance(p, (NumericalHyperparameter, CategoricalHyperparameter))
                else copy.copy(p)
                for name, p in self.items()
            },
        )
        uniform_config_space.add(
            self.substitute_hyperparameters_in_conditions(
                self.conditions,
                uniform_config_space,
            ),
            self.substitute_hyperparameters_in_forbiddens(
                self.forbidden_clauses,
                uniform_config_space,
            ),
        )
        return uniform_config_space

    def estimate_size(self) -> float | int:
        """Estimate the number of unique configurations.

        This is `np.inf` in case if there is a single hyperparameter of size `np.inf`
        (i.e. a `UniformFloatParameter`),
        otherwise it is the product of the size of all hyperparameters. The function
        correctly guesses the number of unique configurations if there are no condition
        and forbidden statements in the configuration spaces. Otherwise, this is an
        upper bound. Use [`generate_grid()`][ConfigSpace.util.generate_grid] to generate
        all valid configurations if required.

        Returns:
            The estimated number of unique configurations
        """
        sizes = [hp.size for hp in self.values()]

        if len(sizes) == 0:
            return 0.0

        acc: int | float = 1
        for size in sizes:
            acc *= size

        return acc

    @staticmethod
    def substitute_hyperparameters_in_conditions(
        conditions: Iterable[ConditionLike],
        new_configspace: ConfigurationSpace,
    ) -> list[ConditionLike]:
        """Takes a set of conditions and generates a new set of conditions with the same
        structure, where each hyperparameter is replaced with its namesake in
        new_configspace. As such, the set of conditions remain unchanged, but the
        included hyperparameters are changed to match those types that exist in
        new_configspace.

        Args:
            conditions: The conditions to adjust
            new_configspace:
                A ConfigurationSpace containing hyperparameters with the same names
                as those in the conditions.

        Returns:
            The list of conditions, adjusted to fit the new ConfigurationSpace
        """
        new_conditions: list[ConditionLike] = []
        for condition in conditions:
            if isinstance(condition, Conjunction):
                conjunction_type = type(condition)
                children = condition.dlcs
                substituted_children = (
                    ConfigurationSpace.substitute_hyperparameters_in_conditions(
                        children,
                        new_configspace,
                    )
                )
                substituted_conjunction = conjunction_type(*substituted_children)
                new_conditions.append(substituted_conjunction)

            elif isinstance(condition, Condition):
                new_conditions.append(
                    condition.__class__(
                        **{
                            **condition.to_dict(),
                            "parent": new_configspace[condition.parent.name],
                            "child": new_configspace[condition.child.name],
                        },
                    ),
                )
            else:
                raise TypeError(
                    f"Did not expect the supplied condition type {type(condition)}.",
                )

        return new_conditions

    @staticmethod
    def substitute_hyperparameters_in_forbiddens(
        forbiddens: Iterable[ForbiddenLike],
        new_configspace: ConfigurationSpace,
    ) -> list[ForbiddenLike]:
        """Takes a set of forbidden clauses and generates a new set of forbidden clauses
        with the same structure, where each hyperparameter is replaced with its
        namesake in new_configspace.
        As such, the set of forbidden clauses remain unchanged, but the included
        hyperparameters are changed to match those types that exist in new_configspace.

        Args:
            forbiddens: An iterable of forbiddens
            new_configspace: ConfigurationSpace
                A ConfigurationSpace containing hyperparameters with the same
                names as those in the forbidden clauses.

        Returns:
            The list of forbidden clauses, adjusted to fit the new ConfigurationSpace
        """
        new_forbiddens: list[ForbiddenLike] = []
        for forbidden in forbiddens:
            if isinstance(forbidden, ForbiddenConjunction):
                substituted_children = (
                    ConfigurationSpace.substitute_hyperparameters_in_forbiddens(
                        forbidden.components,
                        new_configspace,
                    )
                )
                substituted_conjunction = forbidden.__class__(*substituted_children)
                new_forbiddens.append(substituted_conjunction)

            elif isinstance(forbidden, ForbiddenClause):
                if isinstance(forbidden, ForbiddenInClause):
                    new_forbiddens.append(
                        forbidden.__class__(
                            hyperparameter=new_configspace[
                                forbidden.hyperparameter.name
                            ],
                            values=forbidden.values,
                        ),
                    )
                elif isinstance(forbidden, ForbiddenEqualsClause):
                    new_forbiddens.append(
                        forbidden.__class__(
                            hyperparameter=new_configspace[
                                forbidden.hyperparameter.name
                            ],
                            value=forbidden.value,
                        ),
                    )
                else:
                    raise TypeError(
                        f"Forbidden of type '{type(forbidden)}' not recognized.",
                    )

            elif isinstance(forbidden, ForbiddenRelation):
                new_forbiddens.append(
                    forbidden.__class__(
                        left=new_configspace[forbidden.left.name],
                        right=new_configspace[forbidden.right.name],
                    ),
                )
            else:
                raise TypeError(f"Did not expect type {type(forbidden)}.")

        return new_forbiddens

    def __eq__(self, other: Any) -> bool:
        """Override the default Equals behavior."""
        if isinstance(other, self.__class__):
            other_dict = other.__dict__

            # _minimum_condition_span has a np.ndarray which doesn't allow ==
            # to give a direct bool but is based off the others
            for k, v in self.__dict__.items():
                if k in ("random",):
                    continue
                if v != other_dict.get(k):
                    return False

            return True

        return NotImplemented

    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("Configuration space object:\n  Hyperparameters:\n")

        if self.name is not None:
            retval.write(self.name)
            retval.write("\n")

        hyperparameters = sorted(self.values(), key=lambda t: t.name)  # type: ignore
        if hyperparameters:
            retval.write("    ")
            retval.write(
                "\n    ".join(
                    [str(hyperparameter) for hyperparameter in hyperparameters],
                ),
            )
            retval.write("\n")

        conditions = sorted(self.conditions, key=lambda t: str(t))
        if conditions:
            retval.write("  Conditions:\n")
            retval.write("    ")
            retval.write("\n    ".join([str(condition) for condition in conditions]))
            retval.write("\n")

        if self.forbidden_clauses:
            retval.write("  Forbidden Clauses:\n")
            retval.write("    ")
            retval.write(
                "\n    ".join([str(clause) for clause in self.forbidden_clauses]),
            )
            retval.write("\n")

        retval.seek(0)
        return retval.getvalue()

    def __getitem__(self, key: str) -> Hyperparameter:
        return self._dag.nodes[key].hp

    def __iter__(self) -> Iterator[str]:
        """Iterate over the hyperparameter names in the right order."""
        return iter(self._dag.nodes.keys())

    def items(self) -> ItemsView[str, Hyperparameter]:
        """Return an items view of the hyperparameters, same as `dict.items()`."""  # noqa: D402
        return {name: node.hp for name, node in self._dag.nodes.items()}.items()

    def __len__(self) -> int:
        """Return the number of hyperparameters."""
        return self._len

    # TODO: Move these into a single validate function
    def _check_default_configuration(self) -> Configuration:
        # Check if adding that hyperparameter leads to an illegal default configuration
        values: dict[str, Any] = {}
        for hp_name, hp in self.items():
            active: bool = True
            for condition in self.parent_conditions_of[hp_name]:
                if isinstance(condition, Conjunction):
                    parent_names = [c.parent.name for c in condition.dlcs]
                else:
                    parent_names = [condition.parent.name]

                parents = {
                    parent_name: values[parent_name] for parent_name in parent_names
                }

                # OPTIM: Can speed up things here by just breaking early?
                if not condition.satisfied_by_value(parents):
                    active = False

            if not active:
                # the evaluate above will use compares so we need
                # to use NotSet and replace later....
                values[hp_name] = NotSet
            else:
                values[hp_name] = hp.default_value

        return Configuration(self, values=values)

    def _check_configuration_rigorous(
        self,
        configuration: Configuration,
        allow_inactive_with_values: bool = False,
    ) -> None:
        vector = configuration.get_array()
        active_hyperparameters = self.get_active_hyperparameters(configuration)

        for hp_name, node in self._dag.nodes.items():
            hyperparameter = node.hp
            hp_vector = vector[self.index_of[hp_name]]
            active = hp_name in active_hyperparameters

            if not np.isnan(hp_vector) and not hyperparameter.legal_vector(hp_vector):
                raise IllegalVectorizedValueError(hyperparameter, hp_vector)

            if active and np.isnan(hp_vector):
                raise ActiveHyperparameterNotSetError(hyperparameter)

            if (
                not allow_inactive_with_values
                and not active
                and not np.isnan(hp_vector)
            ):
                raise InactiveHyperparameterSetError(hyperparameter, hp_vector)

        self._check_forbidden(vector)

    def _check_forbidden(self, vector: Array[f64]) -> None:
        for clause in self.forbidden_clauses:
            if clause.is_forbidden_vector(vector):
                raise ForbiddenValueError(
                    f"Provided vector violates forbidden clause : {clause}",
                )

    def to_serialized_dict(
        self,
        encoders: Mapping[type, tuple[str, _Encoder]] | None = None,
    ) -> dict[str, Any]:
        """Serialize the configuration space to a dictionary.

        Please see the [serialization reference](../../reference/serialization.md) for more

        Args:
            encoders:
                A mapping from types to a tuple of the type name and an encoder
                function. If not provided, the default encoders will be used.

        Returns:
            A dictionary containing the serialized configuration space
        """
        # NOTE: Used to be called JSON format
        SERIALIZATION_FORMAT_VERSION = 0.4

        _encoders = {
            **HYPERPARAMETER_ENCODERS,
            **CONDITION_ENCODERS,
            **FORBIDDEN_ENCODERS,
            **(encoders or {}),
        }

        def enc(item: Any, _enc: _Encoder) -> dict[str, Any]:
            key = type(item)
            res = _encoders.get(key)
            if res is None:
                raise ValueError(
                    f"No found encoder for '{key}'. Registered encoders are"
                    f" {_encoders.keys()}. Please include a custom `encoders=` if"
                    " you want to encode this type.",
                )

            type_name, encoder = res
            encoding = encoder(item, _enc)
            return {"type": type_name, **encoding}

        from ConfigSpace import __version__

        return {
            "name": self.name,
            "hyperparameters": [enc(hp, enc) for hp in self.values()],
            "conditions": [enc(c, enc) for c in self.conditions],
            "forbiddens": [enc(f, enc) for f in self.forbidden_clauses],
            "python_module_version": __version__,
            "format_version": SERIALIZATION_FORMAT_VERSION,
        }

    @classmethod
    def from_serialized_dict(
        cls,
        d: dict[str, Any],
        decoders: (
            Mapping[
                Literal["hyperparameters", "conditions", "forbiddens"],
                Mapping[str, _Decoder],
            ]
            | None
        ) = None,
    ) -> ConfigurationSpace:
        """Decode a serialized configuration space from a dictionary.

        Please see the [serialization reference](../../reference/serialization.md) for more

        Args:
            d: The serialized configuration space
            decoders:
                A mapping from one of the kinds of things that can be decoded
                to a dictionary of type names to a decoder that will be used.
                Anything not specified will fall back to the default decoders.

        Returns:
            The decoded configuration space
        """
        user_decoders = decoders or {}

        def get_decoder(_decoders: Mapping[str, _Decoder]) -> _Decoder:
            def dec(
                item: dict[str, Any],
                cs: ConfigurationSpace,
                _dec: _Decoder,
            ) -> Any:
                _type = item.pop("type", None)
                if _type is None:
                    raise KeyError(
                        f"Expected a key 'type' in item {item} but did not find it."
                        " Did you include this in the encoding?",
                    )

                decoder = _decoders.get(_type)
                if decoder is None:
                    raise ValueError(
                        f"No found decoder for '{_type}'.  Registered decoders are"
                        f" {_decoders.keys()}. Please include a custom `decoder=` if"
                        " you want to decode this type.",
                    )

                return decoder(item, cs, _dec)

            return dec

        space = ConfigurationSpace(name=d.get("name"))
        _hyperparameters = d.get("hyperparameters", [])
        _conditions = d.get("conditions", [])
        _forbiddens = d.get("forbiddens", [])

        hp_decoder = get_decoder(
            {**HYPERPARAMETER_DECODERS, **user_decoders.get("hyperparameters", {})},
        )
        cond_decoder = get_decoder(
            {**CONDITION_DECODERS, **user_decoders.get("conditions", {})},
        )
        forb_decoder = get_decoder(
            {**FORBIDDEN_DECODERS, **user_decoders.get("forbiddens", {})},
        )

        # Important that we add hyperparameters first as decoding conditions
        # and forbiddens rely on having access to the hyperparameters
        hyperparameters = [hp_decoder(hp, space, hp_decoder) for hp in _hyperparameters]
        space.add(hyperparameters)

        conditions = [cond_decoder(c, space, cond_decoder) for c in _conditions]
        forbidden = [forb_decoder(f, space, forb_decoder) for f in _forbiddens]
        space.add(conditions, forbidden)
        return space

    def to_json(
        self,
        path: str | Path | IO[str],
        *,
        encoders: Mapping[type, tuple[str, _Encoder]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Serialize the configuration space to a JSON file.

        Please see the [serialization reference](../../reference/serialization.md) for more

        Args:
            path: Path to the file or a file object to write to
            encoders:
                A mapping from types to a tuple of the type name and an encoder
                function. If not provided, the default encoders will be used.
            **kwargs: Additional arguments to pass to `json.dump`
        """
        serialized = self.to_serialized_dict(encoders=encoders)
        if isinstance(path, (str, Path)):
            with open(path, "w") as f:
                json.dump(serialized, f, **kwargs)
        else:
            json.dump(serialized, path, **kwargs)

    @classmethod
    def from_json(
        cls,
        path: str | Path | IO[str],
        *,
        decoders: (
            Mapping[
                Literal["hyperparameters", "conditions", "forbiddens"],
                Mapping[str, _Decoder],
            ]
            | None
        ) = None,
        **kwargs: Any,
    ) -> ConfigurationSpace:
        """Decode a serialized configuration space from a json file.

        Please see the [serialization reference](../../reference/serialization.md) for
        more

        Args:
            path: Path to the serialized configuration space
            decoders:
                A mapping from one of the kinds of things that can be decoded
                to a dictionary of type names to a decoder that will be used.
                Anything not specified will fall back to the default decoders.
            **kwargs: Any additional arguments to pass to `json.load`

        Returns:
            The decoded configuration space
        """
        if isinstance(path, (str, Path)):
            p = Path(path)
            with p.open("r") as f:
                d = json.load(f, **kwargs)
        else:
            d = json.load(path, **kwargs)

        return cls.from_serialized_dict(d, decoders=decoders)

    def to_yaml(
        self,
        path: str | Path | IO[str],
        *,
        encoders: Mapping[type, tuple[str, _Encoder]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Serialize the configuration space to a JSON file.

        Please see the [serialization reference](../../reference/serialization.md) for more

        Args:
            path: Path to the file or a file object to write to
            encoders:
                A mapping from types to a tuple of the type name and an encoder
                function. If not provided, the default encoders will be used.
            **kwargs: Additional arguments to pass to `json.dump`
        """
        import yaml

        serialized = self.to_serialized_dict(encoders=encoders)
        if isinstance(path, (str, Path)):
            with open(path, "w") as f:
                yaml.dump(serialized, f, **kwargs)
        else:
            yaml.dump(serialized, path, **kwargs)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path | IO[str],
        *,
        decoders: (
            Mapping[
                Literal["hyperparameters", "conditions", "forbiddens"],
                Mapping[str, _Decoder],
            ]
            | None
        ) = None,
        **kwargs: Any,
    ) -> ConfigurationSpace:
        """Decode a serialized configuration space from a yaml file.

        Please see the [serialization reference](../../reference/serialization.md) for more

        Args:
            path: Path to the serialized configuration space
            decoders:
                A mapping from one of the kinds of things that can be decoded
                to a dictionary of type names to a decoder that will be used.
                Anything not specified will fall back to the default decoders.
            **kwargs: Any additional arguments to pass to `yaml.safe_load`

        Returns:
            The decoded configuration space
        """
        import yaml

        if isinstance(path, (str, Path)):
            p = Path(path)
            with p.open("r") as f:
                d = yaml.safe_load(f, **kwargs)
        else:
            d = yaml.safe_load(path, **kwargs)

        return cls.from_serialized_dict(d, decoders=decoders)

    # ------------ Marked Deprecated --------------------
    # Probably best to only remove these once we actually
    # make some other breaking changes
    # * Search `Marked Deprecated` to find others

    @deprecated("Please use `space[name]`")
    def get_hyperparameter(self, name: str) -> Hyperparameter:  # noqa: D102
        return self[name]

    @property
    @deprecated(
        "Please use map operations directly on the `ConfigurationSpace` object"
        "over private variable `_hyperparameters`",
    )
    def _hyperparameters(self) -> Mapping[str, Hyperparameter]:
        return self

    @property
    @deprecated("Please use `space.at[idx]`")
    def _idx_to_hyperparameter(self) -> Sequence[str]:
        return self.at

    @property
    @deprecated("Please use `space.index_of`")
    def _hyperparameter_idx(self) -> Mapping[str, int]:
        return self.index_of

    @deprecated("Please use `list(space.values())`")
    def get_hyperparameters(self) -> list[Hyperparameter]:  # noqa: D102
        return list(self.values())

    @deprecated("Please use `dict(space)`")
    def get_hyperparameters_dict(self) -> dict[str, Hyperparameter]:  # noqa: D102
        return dict(self)

    @deprecated("Please use `list(space.keys())`")
    def get_hyperparameter_names(self) -> list[str]:  # noqa: D102
        return list(self.keys())

    @deprecated("Please use `list(space.keys())`")
    def _get_parent_conditions_of(self, name: str) -> Sequence[Condition | Conjunction]:
        return self.parent_conditions_of[name]

    @deprecated("Please use `space.children_of[name]`")
    def _get_children_of(self, name: str) -> Sequence[Hyperparameter]:
        return self.children_of[name]

    @deprecated("Please use `space.parents_of[name]`")
    def _get_parents_of(self, name: str) -> Sequence[Hyperparameter]:
        return self.parents_of[name]

    @deprecated("Please use `space.child_conditions_of[name]`")
    def _get_child_conditions_of(self, name: str) -> Sequence[Condition | Conjunction]:
        return self.child_conditions_of[name]

    @deprecated("Please use `space.add(hyperparameter)`")
    def add_hyperparameter(self, hyperparameter: Hyperparameter) -> Hyperparameter:
        """Deprecated. Please use [`space.add()`][ConfigSpace.configuration_space.ConfigurationSpace.add]."""  # noqa: E501
        self.add(hyperparameter)
        return hyperparameter

    @deprecated("Please use `space.add(hyperparameters)`")
    def add_hyperparameters(
        self,
        hyperparameters: Iterable[Hyperparameter],
    ) -> list[Hyperparameter]:
        """Deprecated. Please use [`space.add()`][ConfigSpace.configuration_space.ConfigurationSpace.add]."""  # noqa: E501
        warnings.warn(
            "Please use public function `add()` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        hyperparameters = list(hyperparameters)
        self.add(hyperparameters)
        return hyperparameters

    @deprecated("Please use `space.add(condition)`")
    def add_condition(self, condition: ConditionLike) -> ConditionLike:
        """Deprecated. Please use [`space.add()`][ConfigSpace.configuration_space.ConfigurationSpace.add]."""  # noqa: E501
        self.add(condition)
        return condition

    @deprecated("Please use `space.add(conditions)`")
    def add_conditions(self, conditions: list[ConditionLike]) -> list[ConditionLike]:
        """Deprecated. Please use [`space.add()`][ConfigSpace.configuration_space.ConfigurationSpace.add]."""  # noqa: E501
        self.add(conditions)
        return conditions

    @deprecated("Please use `space.add(clause)`")
    def add_forbidden_clause(self, clause: ForbiddenLike) -> ForbiddenLike:
        """Deprecated. Please use [`space.add()`][ConfigSpace.configuration_space.ConfigurationSpace.add]."""  # noqa: E501
        self.add(clause)
        return clause

    @deprecated("Please use `space.add(clause)`")
    def add_forbidden_clauses(
        self,
        clauses: list[ForbiddenLike],
    ) -> list[ForbiddenLike]:
        """Deprecated. Please use [`space.add()`][ConfigSpace.configuration_space.ConfigurationSpace.add]."""  # noqa: E501
        self.add(clauses)
        return clauses

    @deprecated("Please use `space.index_of[name]`")
    def get_idx_by_hyperparameter_name(self, name: str) -> int:
        """Deprecated. Please use [`space.index_of[name]`][ConfigSpace.configuration_space.ConfigurationSpace.index_of]."""  # noqa: E501
        return self.index_of[name]

    @deprecated("Please use `space.at[idx]`")
    def get_hyperparameter_by_idx(self, idx: int) -> str:
        """Deprecated. Please use [`space.at[idx]`][ConfigSpace.configuration_space.ConfigurationSpace.at]."""  # noqa: E501
        return self.at[idx]

    @deprecated("Please use `space.conditions`")
    def get_conditions(self) -> Sequence[ConditionLike]:
        """Deprecated. Please use [`space.conditions`][ConfigSpace.configuration_space.ConfigurationSpace.conditions]."""  # noqa: E501
        return self.conditions

    @deprecated("Please use `space.forbidden_clauses`")
    def get_forbiddens(self) -> Sequence[ForbiddenLike]:
        """Deprecated. Please use [`space.forbidden_clauses`][ConfigSpace.configuration_space.ConfigurationSpace.forbidden_clauses]."""  # noqa: E501
        return self.forbidden_clauses

    @deprecated("Please use `space.conditional_hyperparameters`")
    def get_all_conditional_hyperparameters(self) -> Sequence[str]:
        """Deprecated. Please use [`space.conditional_hyperparameters`][ConfigSpace.configuration_space.ConfigurationSpace.conditional_hyperparameters]."""  # noqa: E501
        return self.conditional_hyperparameters

    @deprecated("Please use `space.uncoditional_hyperparameters`")
    def get_all_unconditional_hyperparameters(self) -> Sequence[str]:
        """Deprecated. Please use [`space.unconditional_hyperparameters`][ConfigSpace.configuration_space.ConfigurationSpace.unconditional_hyperparameters]."""  # noqa: E501
        return self.unconditional_hyperparameters

    @deprecated("Please use `space.children_of[hyperparameter.name]`")
    def get_children_of(self, name: str | Hyperparameter) -> Sequence[Hyperparameter]:
        """Deprecated. Please use [`space.children_of[name]`][ConfigSpace.configuration_space.ConfigurationSpace.children_of]."""  # noqa: E501
        _name = name.name if isinstance(name, Hyperparameter) else name
        return self.children_of[_name]

    @deprecated("Please use `space.parents_of[hyperparameter.name]`")
    def get_parents_of(self, name: str | Hyperparameter) -> Sequence[Hyperparameter]:
        """Deprecated. Please use [`space.parents_of[name]`][ConfigSpace.configuration_space.ConfigurationSpace.parents_of]."""  # noqa: E501
        _name = name.name if isinstance(name, Hyperparameter) else name
        return self.parents_of[_name]

    @deprecated("Please use `space.child_conditions_of[hyperparameter.name]`")
    def get_child_conditions_of(
        self,
        name: str | Hyperparameter,
    ) -> Sequence[ConditionLike]:
        """Deprecated. Please use [`space.child_conditions_of[name]`][ConfigSpace.configuration_space.ConfigurationSpace.child_conditions_of]."""  # noqa: E501
        _name = name.name if isinstance(name, Hyperparameter) else name
        return self.child_conditions_of[_name]

    @deprecated("Please use `space.parent_conditions_of[hyperparameter.name]`")
    def get_parent_conditions_of(
        self,
        name: str | Hyperparameter,
    ) -> Sequence[Condition | Conjunction]:
        """Deprecated. Please use [`space.parent_conditions_of[name]`][ConfigSpace.configuration_space.ConfigurationSpace.parent_conditions_of]."""  # noqa: E501
        _name = name.name if isinstance(name, Hyperparameter) else name
        return self.parent_conditions_of[_name]

    # ---------------------------------------------------
