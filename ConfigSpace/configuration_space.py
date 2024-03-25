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

import contextlib
import copy
import io
import warnings
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable, Iterator, KeysView, Mapping
from itertools import chain
from typing import Any, Final, cast, overload

import numpy as np
from more_itertools import unique_everseen

import ConfigSpace.c_util
from ConfigSpace import nx
from ConfigSpace.conditions import (
    Condition,
    Conjunction,
    EqualsCondition,
)
from ConfigSpace.configuration import Configuration, NotSet
from ConfigSpace.exceptions import (
    ActiveHyperparameterNotSetError,
    AmbiguousConditionError,
    ChildNotFoundError,
    CyclicDependancyError,
    HyperparameterAlreadyExistsError,
    HyperparameterIndexError,
    HyperparameterNotFoundError,
    IllegalValueError,
    InactiveHyperparameterSetError,
    ParentNotFoundError,
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
from ConfigSpace.hyperparameters.hyperparameter import HyperparameterWithPrior

_ROOT: Final = "__HPOlib_configuration_space_root__"


def _parse_hyperparameters_from_dict(items: dict[str, Any]) -> Iterator[Hyperparameter]:
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

        # If it's an allowed type, it's a constant
        elif isinstance(hp, int | str | float):
            yield Constant(name, hp)

        else:
            raise ValueError(f"Unknown value '{hp}' for '{name}'")


def _assert_type(
    item: Any,
    expected: type | tuple[type, ...],
    method: str | None = None,
) -> None:
    if not isinstance(item, expected):
        msg = f"Expected {expected}, got {type(item)}"
        if method:
            msg += " in method " + method
        raise TypeError(msg)


def _assert_legal(hyperparameter: Hyperparameter, value: tuple | list | Any) -> None:
    if isinstance(value, tuple | list):
        for v in value:
            if not hyperparameter.legal_value(v):
                raise IllegalValueError(hyperparameter, v)
    elif not hyperparameter.legal_value(value):
        raise IllegalValueError(hyperparameter, value)


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
        name: str | dict | None = None,
        seed: int | None = None,
        meta: dict | None = None,
        *,
        space: None
        | (
            dict[
                str,
                tuple[int, int] | tuple[float, float] | list[Any] | int | float | str,
            ]
        ) = None,
    ) -> None:
        """Parameters
        ----------
        name : str | dict, optional
            Name of the configuration space. If a dict is passed, this is considered the
            same as the `space` arg.
        seed : int, optional
            Random seed
        meta : dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        space:
            A simple configuration space to use:

            .. code:: python

                ConfigurationSpace(
                    name="myspace",
                    space={
                        "uniform_integer": (1, 10),
                        "uniform_float": (1.0, 10.0),
                        "categorical": ["a", "b", "c"],
                        "constant": 1337,
                    }
                )

        """
        # If first arg is a dict, we assume this to be `space`
        if isinstance(name, dict):
            space = name
            name = None

        self.name = name
        self.meta = meta

        # NOTE: The idx of a hyperparamter is tied to its order in _hyperparamters
        # Having three variables to keep track of this seems excessive
        self._hyperparameters: OrderedDict[str, Hyperparameter] = OrderedDict()
        self._hyperparameter_idx: dict[str, int] = {}
        self._idx_to_hyperparameter: dict[int, str] = {}

        # Use dictionaries to make sure that we don't accidently add
        # additional keys to these mappings (which happened with defaultdict()).
        # This once broke auto-sklearn's equal comparison of configuration
        # spaces when _children of one instance contained  all possible
        # hyperparameters as keys and empty dictionaries as values while the
        # other instance not containing these.
        self._children: OrderedDict[
            str,
            OrderedDict[str, None | Condition | Conjunction],
        ]
        self._children = OrderedDict()

        self._parents: OrderedDict[
            str,
            OrderedDict[str, None | Condition | Conjunction],
        ]
        self._parents = OrderedDict()

        # Changing this to a normal dict will break sampling because there is
        # no guarantee that the parent of a condition was evaluated before
        self._conditionals: set[str] = set()
        self.forbidden_clauses: list[
            ForbiddenConjunction | ForbiddenRelation | ForbiddenClause
        ] = []
        self.random = np.random.RandomState(seed)

        self._children[_ROOT] = OrderedDict()

        self._parent_conditions_of: dict[str, list[Condition | Conjunction]] = {}
        self._child_conditions_of: dict[str, list[Condition | Conjunction]] = {}
        self._parents_of: dict[str, list[Hyperparameter]] = {}
        self._children_of: dict[str, list[Hyperparameter]] = {}
        self._conditional_dep_graph: dict[str, list[Condition | Conjunction]] = {}

        if space is not None:
            hyperparameters = list(_parse_hyperparameters_from_dict(space))
            self.add_hyperparameters(hyperparameters)

    def add_hyperparameter(self, hyperparameter: Hyperparameter) -> Hyperparameter:
        """Add a hyperparameter to the configuration space.

        Parameters
        ----------
        hyperparameter : :ref:`Hyperparameters`
            The hyperparameter to add

        Returns:
        -------
        :ref:`Hyperparameters`
            The added hyperparameter
        """
        _assert_type(hyperparameter, Hyperparameter, method="add_hyperparameter")

        self._add_hyperparameter(hyperparameter)
        self._update_cache()
        self._sort_hyperparameters()
        self._check_default_configuration()

        return hyperparameter

    def add_hyperparameters(
        self,
        hyperparameters: Iterable[Hyperparameter],
    ) -> list[Hyperparameter]:
        """Add hyperparameters to the configuration space.

        Parameters
        ----------
        hyperparameters : Iterable(:ref:`Hyperparameters`)
            Collection of hyperparameters to add

        Returns:
        -------
        list(:ref:`Hyperparameters`)
            List of added hyperparameters (same as input)
        """
        hyperparameters = list(hyperparameters)
        for hp in hyperparameters:
            _assert_type(hp, Hyperparameter, method="add_hyperparameters")

        for hyperparameter in hyperparameters:
            self._add_hyperparameter(hyperparameter)

        self._update_cache()
        self._sort_hyperparameters()
        self._check_default_configuration()
        return hyperparameters

    def add_condition(
        self,
        condition: Condition | Conjunction,
    ) -> Condition | Conjunction:
        """Add a condition to the configuration space.

        Check if adding the condition is legal:

        - The parent in a condition statement must exist
        - The condition must add no cycles

        The internal array keeps track of all edges which must be
        added to the DiGraph; if the checks don't raise any Exception,
        these edges are finally added at the end of the function.

        Parameters
        ----------
        condition : :ref:`Conditions`
            Condition to add

        Returns:
        -------
        :ref:`Conditions`
            Same condition as input
        """
        if isinstance(condition, Condition):
            self._check_edges([(condition.parent, condition.child)], [condition.value])
            self._check_condition(condition.child, condition)
            self._add_edge(condition.parent, condition.child, condition=condition)
        # Loop over the Conjunctions to find out the conditions we must add!
        elif isinstance(condition, Conjunction):
            dlcs = condition.get_descendant_literal_conditions()
            edges = [(dlc.parent, dlc.child) for dlc in dlcs]
            values = [dlc.value for dlc in dlcs]
            self._check_edges(edges, values)

            for dlc in dlcs:
                self._check_condition(dlc.child, condition)
                self._add_edge(dlc.parent, dlc.child, condition=condition)

        else:
            raise TypeError(f"Unknown condition type {type(condition)}")

        self._sort_hyperparameters()
        self._update_cache()
        return condition

    def add_conditions(
        self,
        conditions: list[Condition | Conjunction],
    ) -> list[Condition | Conjunction]:
        """Add a list of conditions to the configuration space.

        They must be legal. Take a look at
        :meth:`~ConfigSpace.configuration_space.ConfigurationSpace.add_condition`.

        Parameters
        ----------
        conditions : list(:ref:`Conditions`)
            collection of conditions to add

        Returns:
        -------
        list(:ref:`Conditions`)
            Same as input conditions
        """
        edges = []
        values = []
        conditions_to_add = []
        for condition in conditions:
            # TODO: Need to check that we can't add a condition twice!

            if isinstance(condition, Condition):
                edges.append((condition.parent, condition.child))
                values.append(condition.value)
                conditions_to_add.append(condition)

            elif isinstance(condition, Conjunction):
                dlcs = condition.get_descendant_literal_conditions()
                edges.extend([(dlc.parent, dlc.child) for dlc in dlcs])
                values.extend([dlc.value for dlc in dlcs])
                conditions_to_add.extend([condition] * len(dlcs))

            else:
                raise TypeError(f"Unknown condition type {type(condition)}")

        for edge, condition in zip(edges, conditions_to_add, strict=False):
            self._check_condition(edge[1], condition)

        self._check_edges(edges, values)
        for edge, condition in zip(edges, conditions_to_add, strict=False):
            self._add_edge(edge[0], edge[1], condition)

        self._sort_hyperparameters()
        self._update_cache()
        return conditions

    # TODO: Typing could be improved here, as it gives back out what you put in
    def add_forbidden_clause(self, clause: ForbiddenLike) -> ForbiddenLike:
        """Add a forbidden clause to the configuration space.

        Parameters
        ----------
        clause : :ref:`Forbidden clauses`
            Forbidden clause to add

        Returns:
        -------
        :ref:`Forbidden clauses`
            Same as input forbidden clause
        """
        self._check_forbidden_component(clause=clause)
        clause.set_vector_idx(self._hyperparameter_idx)
        self.forbidden_clauses.append(clause)
        self._check_default_configuration()
        return clause

    def add_forbidden_clauses(
        self,
        clauses: list[ForbiddenLike],
    ) -> list[ForbiddenLike]:
        """Add a list of forbidden clauses to the configuration space.

        Parameters
        ----------
        clauses : list(:ref:`Forbidden clauses`)
            Collection of forbidden clauses to add

        Returns:
        -------
        list(:ref:`Forbidden clauses`)
            Same as input clauses
        """
        for clause in clauses:
            self._check_forbidden_component(clause=clause)
            clause.set_vector_idx(self._hyperparameter_idx)
            self.forbidden_clauses.append(clause)

        self._check_default_configuration()
        return clauses

    def add_configuration_space(
        self,
        prefix: str,
        configuration_space: ConfigurationSpace,
        delimiter: str = ":",
        parent_hyperparameter: dict | None = None,
    ) -> ConfigurationSpace:
        """Combine two configuration space by adding one the other configuration
        space. The contents of the configuration space, which should be added,
        are renamed to ``prefix`` + ``delimiter`` + old_name.

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
                ``parent_hyperparameter`` is active. Must be a dictionary with two keys
                "parent" and "value", meaning that the added configuration space is
                active when `parent` is equal to `value`

        Returns:
        -------
            The configuration space, which was added.
        """
        _assert_type(
            configuration_space,
            ConfigurationSpace,
            method="add_configuration_space",
        )

        prefix_delim = f"{prefix}{delimiter}"

        def _new_name(_item: Hyperparameter) -> str:
            if _item.name in ("", prefix):
                return prefix

            if not _item.name.startswith(prefix_delim):
                return f"{prefix_delim}{_item.name}"

            return cast(str, _item.name)

        new_parameters = []
        for hp in configuration_space.values():
            new_hp = copy.copy(hp)
            new_hp.name = _new_name(hp)
            new_parameters.append(new_hp)

        self.add_hyperparameters(new_parameters)

        conditions_to_add = []
        for condition in configuration_space.get_conditions():
            new_condition = copy.copy(condition)
            dlcs = (
                new_condition.get_descendant_literal_conditions()
                if isinstance(new_condition, Conjunction)
                else [new_condition]
            )
            for dlc in dlcs:
                # Rename children
                dlc.child.name = _new_name(dlc.child)
                dlc.parent.name = _new_name(dlc.parent)

            conditions_to_add.append(new_condition)

        self.add_conditions(conditions_to_add)

        forbiddens_to_add = []
        for forbidden_clause in configuration_space.forbidden_clauses:
            new_forbidden = forbidden_clause
            dlcs = (
                new_forbidden.get_descendant_literal_clauses()
                if isinstance(new_forbidden, ForbiddenConjunction)
                else [new_forbidden]
            )
            for dlc in dlcs:
                if isinstance(dlc, ForbiddenRelation):
                    dlc.left.name = _new_name(dlc.left)
                    dlc.right.name = _new_name(dlc.right)
                else:
                    dlc.hyperparameter.name = _new_name(dlc.hyperparameter)
            forbiddens_to_add.append(new_forbidden)

        self.add_forbidden_clauses(forbiddens_to_add)

        conditions_to_add = []
        if parent_hyperparameter is not None:
            parent = parent_hyperparameter["parent"]
            value = parent_hyperparameter["value"]

            # Only add a condition if the parameter is a top-level parameter of the new
            # configuration space (this will be some kind of tree structure).
            for new_hp in new_parameters:
                parents = self.get_parents_of(new_hp)
                if not any(parents):
                    condition = EqualsCondition(new_hp, parent, value)
                    conditions_to_add.append(condition)

        self.add_conditions(conditions_to_add)

        return configuration_space

    def get_hyperparameter_by_idx(self, idx: int) -> str:
        """Name of a hyperparameter from the space given its id.

        Parameters
        ----------
        idx : int
            Id of a hyperparameter

        Returns:
        -------
        str
            Name of the hyperparameter
        """
        hp = self._idx_to_hyperparameter.get(idx)
        if hp is None:
            raise HyperparameterIndexError(idx, self)

        return hp

    def get_idx_by_hyperparameter_name(self, name: str) -> int:
        """The id of a hyperparameter by its ``name``.

        Parameters
        ----------
        name : str
            Name of a hyperparameter

        Returns:
        -------
        int
            Id of the hyperparameter with name ``name``
        """
        idx = self._hyperparameter_idx.get(name)

        if idx is None:
            raise HyperparameterNotFoundError(name, space=self)

        return idx

    def get_conditions(self) -> list[Condition]:
        """All conditions from the configuration space.

        Returns:
        -------
        list(:ref:`Conditions`)
            Conditions of the configuration space
        """
        conditions = []
        added_conditions: set[str] = set()

        # Nodes is a list of nodes
        for source_node in self._hyperparameters.values():
            # This is a list of keys in a dictionary
            # TODO sort the edges by the order of their source_node in the
            # hyperparameter list!
            for target_node in self._children[source_node.name]:
                if target_node not in added_conditions:
                    condition = self._children[source_node.name][target_node]
                    conditions.append(condition)
                    added_conditions.add(target_node)

        return conditions

    def get_forbiddens(self) -> list[ForbiddenLike]:
        """All forbidden clauses from the configuration space.

        Returns:
        -------
        list(:ref:`Forbidden clauses`)
            List with the forbidden clauses
        """
        return self.forbidden_clauses

    def get_children_of(self, name: str | Hyperparameter) -> list[Hyperparameter]:
        """Return a list with all children of a given hyperparameter.

        Parameters
        ----------
        name : str, :ref:`Hyperparameters`
            Hyperparameter or its name, for which all children are requested

        Returns:
        -------
        list(:ref:`Hyperparameters`)
            Children of the hyperparameter
        """
        # Get all possible children of the hyperparameter based on conditionals
        children = chain.from_iterable(
            cond.get_children() if isinstance(cond, Conjunction) else [cond.child]
            for cond in self.get_child_conditions_of(name)
        )
        # Get the unique ones
        return list(unique_everseen(children, key=id))

    def generate_all_continuous_from_bounds(
        self,
        bounds: list[tuple[float, float]],
    ) -> None:
        """Generate :class:`~ConfigSpace.hyperparameters.UniformFloatHyperparameter`
        from a list containing lists with lower and upper bounds.

        The generated hyperparameters are added to the configuration space.

        Parameters
        ----------
        bounds : list[tuple([float, float])]
            List containing lists with two elements: lower and upper bound
        """
        self.add_hyperparameters(
            [
                UniformFloatHyperparameter(name=f"x{i}", lower=lower, upper=upper)
                for i, (lower, upper) in enumerate(bounds)
            ],
        )

    def get_child_conditions_of(
        self,
        name: str | Hyperparameter,
    ) -> list[Condition]:
        """Return a list with conditions of all children of a given
        hyperparameter referenced by its ``name``.

        Parameters
        ----------
        name : str, :ref:`Hyperparameters`
            Hyperparameter or its name, for which conditions are requested

        Returns:
        -------
        list(:ref:`Conditions`)
            List with the conditions on the children of the given hyperparameter
        """
        name = name if isinstance(name, str) else name.name

        # This raises an exception if the hyperparameter does not exist
        self[name]
        return self._get_child_conditions_of(name)

    def get_parents_of(self, name: str | Hyperparameter) -> list[Hyperparameter]:
        """The parents hyperparameters of a given hyperparameter.

        Parameters
        ----------
        name : str, :ref:`Hyperparameters`
            Can either be the name of a hyperparameter or the hyperparameter
            object.

        Returns:
        -------
        list[:ref:`Conditions`]
            List with all parent hyperparameters
        """
        # Get all possible parents of the hyperparameter based on conditionals
        parents = chain.from_iterable(
            cond.get_parents() if isinstance(cond, Conjunction) else [cond.parent]
            for cond in self.get_parent_conditions_of(name)
        )
        # Get the unique ones
        return list(unique_everseen(parents, key=id))

    def get_parent_conditions_of(
        self,
        name: str | Hyperparameter,
    ) -> list[Condition | Conjunction]:
        """The conditions of all parents of a given hyperparameter.

        Parameters
        ----------
        name : str, :ref:`Hyperparameters`
            Can either be the name of a hyperparameter or the hyperparameter
            object

        Returns:
        -------
        list[:ref:`Conditions`]
            List with all conditions on parent hyperparameters
        """
        if isinstance(name, Hyperparameter):
            name = name.name  # type: ignore

        # This raises an exception if the hyperparameter does not exist
        self[name]
        return self._get_parent_conditions_of(name)

    def get_all_unconditional_hyperparameters(self) -> list[str]:
        """Names of unconditional hyperparameters.

        Returns:
        -------
        list[str]
            List with all parent hyperparameters, which are not part of a condition
        """
        return list(self._children[_ROOT])

    def get_all_conditional_hyperparameters(self) -> set[str]:
        """Names of all conditional hyperparameters.

        Returns:
        -------
        set[:ref:`Hyperparameters`]
            Set with all conditional hyperparameter
        """
        return self._conditionals

    def get_default_configuration(self) -> Configuration:
        """Configuration containing hyperparameters with default values.

        Returns:
        -------
        :class:`~ConfigSpace.configuration_space.Configuration`
            Configuration with the set default values

        """
        return self._check_default_configuration()

    # For backward compatibility
    def check_configuration(self, configuration: Configuration) -> None:
        """Check if a configuration is legal. Raises an error if not.

        Parameters
        ----------
        configuration : :class:`~ConfigSpace.configuration_space.Configuration`
            Configuration to check
        """
        _assert_type(configuration, Configuration, method="check_configuration")
        ConfigSpace.c_util.check_configuration(self, configuration.get_array(), False)

    def check_configuration_vector_representation(self, vector: np.ndarray) -> None:
        """Raise error if configuration in vector representation is not legal.

        Parameters
        ----------
        vector : np.ndarray
            Configuration in vector representation
        """
        _assert_type(
            vector,
            np.ndarray,
            method="check_configuration_vector_representation",
        )
        ConfigSpace.c_util.check_configuration(self, vector, False)

    def get_active_hyperparameters(
        self,
        configuration: Configuration,
    ) -> set[Hyperparameter]:
        """Set of active hyperparameter for a given configuration.

        Parameters
        ----------
        configuration : :class:`~ConfigSpace.configuration_space.Configuration`
            Configuration for which the active hyperparameter are returned

        Returns:
        -------
        set(:class:`~ConfigSpace.configuration_space.Configuration`)
            The set of all active hyperparameter

        """
        vector = configuration.get_array()
        active_hyperparameters = set()
        for hp_name, hyperparameter in self._hyperparameters.items():
            conditions = self._parent_conditions_of[hyperparameter.name]

            active = True
            for condition in conditions:
                if isinstance(condition, Conjunction):
                    parent_vector_idx = condition.get_parents_vector()
                else:
                    parent_vector_idx = [condition.parent_vector_id]

                # if one of the parents is None, the hyperparameter cannot be
                # active! Else we have to check this
                # Note from trying to optimize this - this is faster than using
                # dedicated numpy functions and indexing
                if any(vector[i] != vector[i] for i in parent_vector_idx):
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
        """Sample ``size`` configurations from the configuration space object.

        Parameters
        ----------
        size : int, optional
            Number of configurations to sample. Default to 1

        Returns:
        -------
        :class:`~ConfigSpace.configuration_space.Configuration`,
            list[:class:`~ConfigSpace.configuration_space.Configuration`]:
            A single configuration if ``size`` 1 else a list of Configurations
        """
        if size == 1:
            warnings.warn(
                "Please leave at default or explicitly set `size=None`."
                " In the future, specifying a size will always retunr a list, even if"
                " 1",
                DeprecationWarning,
                stacklevel=2,
            )

        # Maintain old behaviour by setting this
        if size is None:
            size = 1

        _assert_type(size, int, method="sample_configuration")
        if size < 1:
            return []

        accepted_configurations: list[Configuration] = []
        num_hyperparameters = len(self._hyperparameters)

        unconditional_hyperparameters = self.get_all_unconditional_hyperparameters()

        # Filter forbiddens into those that have conditionals and those that do not
        _forbidden_clauses_unconditionals: list[ForbiddenLike] = []
        _forbidden_clauses_conditionals: list[ForbiddenLike] = []
        for clause in self.forbidden_clauses:
            based_on_conditionals = False

            dlcs = (
                [clause]
                if not isinstance(clause, ForbiddenConjunction)
                else clause.dlcs
            )
            for subclause in dlcs:
                if isinstance(subclause, ForbiddenRelation):
                    if (
                        subclause.left.name not in unconditional_hyperparameters
                        or subclause.right.name not in unconditional_hyperparameters
                    ):
                        based_on_conditionals = True
                        break
                elif subclause.hyperparameter.name not in unconditional_hyperparameters:
                    based_on_conditionals = True
                    break

            if based_on_conditionals:
                _forbidden_clauses_conditionals.append(clause)
            else:
                _forbidden_clauses_unconditionals.append(clause)

        # Main sampling loop
        MULT = (len(self.forbidden_clauses) + len(self._conditionals)) / len(
            self._hyperparameters,
        )
        sample_size = size
        while len(accepted_configurations) < size:
            sample_size = max(int(MULT**2 * sample_size), 5)

            # Sample a vector for each hp, filling out columns
            config_matrix = np.empty(
                (sample_size, num_hyperparameters),
                dtype=np.float64,
            )
            for i, hp in enumerate(self._hyperparameters.values()):
                config_matrix[:, i] = hp.sample_vector(sample_size, seed=self.random)

            # Apply unconditional forbiddens across the columns (hps)
            # We treat this as an OR, i.e. if any of the forbidden clauses are
            # forbidden, the entire configuration (row) is forbidden
            uncond_forbidden = np.zeros(sample_size, dtype=np.bool_)
            for clause in _forbidden_clauses_unconditionals:
                uncond_forbidden |= clause.is_forbidden_vector_array(config_matrix)

            valid_config_matrix = config_matrix[~uncond_forbidden]

            for condition, effected_mask in self.minimum_condition_span:
                satisfied = condition.satisfied_by_vector_array(valid_config_matrix)
                valid_config_matrix[np.ix_(~satisfied, effected_mask)] = np.nan

            # Now we apply the forbiddens that depend on conditionals
            cond_forbidden = np.zeros(len(valid_config_matrix), dtype=np.bool_)
            for clause in _forbidden_clauses_conditionals:
                cond_forbidden |= clause.is_forbidden_vector_array(valid_config_matrix)

            valid_config_matrix = valid_config_matrix[~cond_forbidden]

            # And now we have a matrix of valid configurations
            accepted_configurations.extend(
                [Configuration(self, vector=vec) for vec in valid_config_matrix],
            )
            sample_size = size - len(accepted_configurations)

        if size <= 1:
            return accepted_configurations[0]

        return accepted_configurations

    def seed(self, seed: int) -> None:
        """Set the random seed to a number.

        Parameters
        ----------
        seed : int
            The random seed
        """
        self.random = np.random.RandomState(seed)

    def remove_hyperparameter_priors(self) -> ConfigurationSpace:
        """Produces a new ConfigurationSpace where all priors on parameters are removed.

        Non-uniform hyperpararmeters are replaced with uniform ones, and
        CategoricalHyperparameters with weights have their weights removed.

        Returns:
        -------
        :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
            The resulting configuration space, without priors on the hyperparameters
        """
        uniform_config_space = ConfigurationSpace()
        for parameter in self.values():
            if isinstance(parameter, HyperparameterWithPrior):
                uniform_config_space.add_hyperparameter(parameter.to_uniform())
            else:
                uniform_config_space.add_hyperparameter(copy.copy(parameter))

        new_conditions = self.substitute_hyperparameters_in_conditions(
            self.get_conditions(),
            uniform_config_space,
        )
        new_forbiddens = self.substitute_hyperparameters_in_forbiddens(
            self.get_forbiddens(),
            uniform_config_space,
        )
        uniform_config_space.add_conditions(new_conditions)
        uniform_config_space.add_forbidden_clauses(new_forbiddens)

        return uniform_config_space

    def estimate_size(self) -> float | int:
        """Estimate the number of unique configurations.

        This is `np.inf` in case if there is a single hyperparameter of size `np.inf`
        (i.e. a :class:`~ConfigSpace.hyperparameters.UniformFloatHyperparameter`),
        otherwise it is the product of the size of all hyperparameters. The function
        correctly guesses the number of unique configurations if there are no condition
        and forbidden statements in the configuration spaces. Otherwise, this is an
        upper bound. Use :func:`~ConfigSpace.util.generate_grid` to generate all
        valid configurations if required.
        """
        sizes = [hp.size for hp in self._hyperparameters.values()]

        if len(sizes) == 0:
            return 0.0

        acc = 1
        for size in sizes:
            acc *= size

        return acc

    @staticmethod
    def substitute_hyperparameters_in_conditions(
        conditions: Iterable[Condition | Conjunction],
        new_configspace: ConfigurationSpace,
    ) -> list[Condition | Conjunction]:
        """Takes a set of conditions and generates a new set of conditions with the same
        structure, where each hyperparameter is replaced with its namesake in
        new_configspace. As such, the set of conditions remain unchanged, but the
        included hyperparameters are changed to match those types that exist in
        new_configspace.

        Parameters
        ----------
        new_configspace: ConfigurationSpace
            A ConfigurationSpace containing hyperparameters with the same names as those
            in the conditions.

        Returns:
        -------
        list[Condition | Conjunction]:
            The list of conditions, adjusted to fit the new ConfigurationSpace
        """
        new_conditions = []
        for condition in conditions:
            if isinstance(condition, Conjunction):
                conjunction_type = type(condition)
                children = condition.get_descendant_literal_conditions()
                substituted_children = (
                    ConfigurationSpace.substitute_hyperparameters_in_conditions(
                        children,
                        new_configspace,
                    )
                )
                substituted_conjunction = conjunction_type(*substituted_children)
                new_conditions.append(substituted_conjunction)

            elif isinstance(condition, Condition):
                condition_type = type(condition)
                child_name = condition.child.name
                parent_name = condition.parent.name
                new_child = new_configspace[child_name]
                new_parent = new_configspace[parent_name]

                # TODO: Fix this up
                if hasattr(condition, "values"):
                    condition_arg = condition.values
                    substituted_condition = condition_type(
                        child=new_child,
                        parent=new_parent,
                        values=condition_arg,
                    )
                elif hasattr(condition, "value"):
                    condition_arg = condition.value
                    substituted_condition = condition_type(
                        child=new_child,
                        parent=new_parent,
                        value=condition_arg,
                    )
                else:
                    raise AttributeError(
                        f"Did not find the expected attribute in condition"
                        f" {type(condition)}.",
                    )

                new_conditions.append(substituted_condition)
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

        Parameters
        ----------
        forbiddens: Iterable[ForbiddenLike]
            An iterable of forbiddens
        new_configspace: ConfigurationSpace
            A ConfigurationSpace containing hyperparameters with the same names as those
            in the forbidden clauses.

        Returns:
        -------
        list[ForbiddenLike]:
            The list of forbidden clauses, adjusted to fit the new ConfigurationSpace
        """
        new_forbiddens = []
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
                new_hyperparameter = new_configspace[forbidden.hyperparameter.name]
                if isinstance(forbidden, ForbiddenInClause):
                    substituted_forbidden = forbidden.__class__(
                        hyperparameter=new_hyperparameter,
                        values=forbidden.values,
                    )
                    new_forbiddens.append(substituted_forbidden)
                elif isinstance(forbidden, ForbiddenEqualsClause):
                    substituted_forbidden = forbidden.__class__(
                        hyperparameter=new_hyperparameter,
                        value=forbidden.value,
                    )
                    new_forbiddens.append(substituted_forbidden)
                else:
                    raise TypeError(
                        f"Forbidden of type '{type(forbidden)}' not recognized.",
                    )

            elif isinstance(forbidden, ForbiddenRelation):
                substituted_forbidden = forbidden.__class__(
                    left=new_configspace[forbidden.left.name],
                    right=new_configspace[forbidden.right.name],
                )
                new_forbiddens.append(substituted_forbidden)
            else:
                raise TypeError(f"Did not expect type {type(forbidden)}.")

        return new_forbiddens

    def __eq__(self, other: Any) -> bool:
        """Override the default Equals behavior."""
        if isinstance(other, self.__class__):
            this_dict = self.__dict__.copy()
            del this_dict["random"]
            other_dict = other.__dict__.copy()
            del other_dict["random"]
            return this_dict == other_dict
        return NotImplemented

    def __hash__(self) -> int:
        """Override the default hash behavior (that returns the id or the object)."""
        return hash(self.__repr__())

    def __getitem__(self, key: str) -> Hyperparameter:
        hp = self._hyperparameters.get(key)
        if hp is None:
            raise HyperparameterNotFoundError(key, space=self)

        return hp

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

        conditions = sorted(self.get_conditions(), key=lambda t: str(t))
        if conditions:
            retval.write("  Conditions:\n")
            retval.write("    ")
            retval.write("\n    ".join([str(condition) for condition in conditions]))
            retval.write("\n")

        if self.get_forbiddens():
            retval.write("  Forbidden Clauses:\n")
            retval.write("    ")
            retval.write(
                "\n    ".join([str(clause) for clause in self.get_forbiddens()]),
            )
            retval.write("\n")

        retval.seek(0)
        return retval.getvalue()

    def __iter__(self) -> Iterator[str]:
        """Iterate over the hyperparameter names in the right order."""
        return iter(self._hyperparameters.keys())

    def keys(self) -> KeysView[str]:
        """Return the hyperparameter names in the right order."""
        return self._hyperparameters.keys()

    def __len__(self) -> int:
        return len(self._hyperparameters)

    def _add_hyperparameter(self, hyperparameter: Hyperparameter) -> None:
        hp_name = hyperparameter.name

        existing = self._hyperparameters.get(hp_name)
        if existing is not None:
            raise HyperparameterAlreadyExistsError(existing, hyperparameter, space=self)

        self._hyperparameters[hp_name] = hyperparameter
        self._children[hp_name] = OrderedDict()

        # TODO remove (_ROOT) __HPOlib_configuration_space_root__, it is only used in
        # to check for cyclic configuration spaces. If it is only added when
        # cycles are checked, the code can become much easier (e.g. the parent
        # caching can be more or less removed).
        self._children[_ROOT][hp_name] = None
        self._parents[hp_name] = OrderedDict()
        self._parents[hp_name][_ROOT] = None

        # Save the index of each hyperparameter name to later on access a
        # vector of hyperparameter values by indices, must be done twice
        # because check_default_configuration depends on it
        self._hyperparameter_idx.update(
            {hp: i for i, hp in enumerate(self._hyperparameters)},
        )

    def _sort_hyperparameters(self) -> None:
        levels: OrderedDict[str, int] = OrderedDict()
        to_visit: deque[str] = deque()
        for hp_name in self._hyperparameters:
            to_visit.appendleft(hp_name)

        while len(to_visit) > 0:
            current = to_visit.pop()
            if _ROOT in self._parents[current]:
                assert len(self._parents[current]) == 1
                levels[current] = 1

            else:
                all_parents_visited = True
                depth = -1
                for parent in self._parents[current]:
                    if parent not in levels:
                        all_parents_visited = False
                        break

                    depth = max(depth, levels[parent] + 1)

                if all_parents_visited:
                    levels[current] = depth
                else:
                    to_visit.appendleft(current)

        by_level: defaultdict[int, list[str]] = defaultdict(list)
        for hp in levels:
            level = levels[hp]
            by_level[level].append(hp)

        nodes = []
        # Sort and add to list
        for level in sorted(by_level):
            sorted_by_level = by_level[level]
            sorted_by_level.sort()
            nodes.extend(sorted_by_level)

        # Resort the OrderedDict
        new_order = OrderedDict()
        for node in nodes:
            new_order[node] = self._hyperparameters[node]
        self._hyperparameters = new_order

        # Update to reflect sorting
        for i, hp in enumerate(self._hyperparameters):
            self._hyperparameter_idx[hp] = i
            self._idx_to_hyperparameter[i] = hp

        # Update order of _children
        new_order = OrderedDict()
        new_order[_ROOT] = self._children[_ROOT]
        for hp in chain([_ROOT], self._hyperparameters):
            # Also resort the children dict
            children_sorting = [
                (self._hyperparameter_idx[child_name], child_name)
                for child_name in self._children[hp]
            ]
            children_sorting.sort()
            children_order = OrderedDict()
            for _, child_name in children_sorting:
                children_order[child_name] = self._children[hp][child_name]
            new_order[hp] = children_order
        self._children = new_order

        # Update order of _parents
        new_order = OrderedDict()
        for hp in self._hyperparameters:
            # Also resort the parent's dict
            if _ROOT in self._parents[hp]:
                parent_sorting = [(-1, _ROOT)]
            else:
                parent_sorting = [
                    (self._hyperparameter_idx[parent_name], parent_name)
                    for parent_name in self._parents[hp]
                ]
            parent_sorting.sort()
            parent_order = OrderedDict()
            for _, parent_name in parent_sorting:
                parent_order[parent_name] = self._parents[hp][parent_name]
            new_order[hp] = parent_order
        self._parents = new_order

        # update conditions
        for condition in self.get_conditions():
            condition.set_vector_idx(self._hyperparameter_idx)

        # forbidden clauses
        for clause in self.get_forbiddens():
            clause.set_vector_idx(self._hyperparameter_idx)

    def _check_condition(
        self,
        child_node: Hyperparameter,
        condition: Condition | Conjunction,
    ) -> None:
        for present_condition in self._get_parent_conditions_of(child_node.name):
            if present_condition != condition:
                raise AmbiguousConditionError(present_condition, condition)

    def _add_edge(
        self,
        parent_node: Hyperparameter,
        child_node: Hyperparameter,
        condition: Condition | Conjunction,
    ) -> None:
        self._children[_ROOT].pop(child_node.name, None)
        self._parents[child_node.name].pop(_ROOT, None)

        if (
            existing := self._children[parent_node.name].pop(child_node.name, None)
        ) is not None and existing != condition:
            raise AmbiguousConditionError(existing, condition)

        if (
            existing := self._parents[child_node.name].pop(parent_node.name, None)
        ) is not None and existing != condition:
            raise AmbiguousConditionError(existing, condition)

        self._children[parent_node.name][child_node.name] = condition
        self._parents[child_node.name][parent_node.name] = condition

        self._conditionals.add(child_node.name)

    def _create_tmp_dag(self) -> nx.DiGraph:
        tmp_dag = nx.DiGraph()
        for hp_name in self._hyperparameters:
            tmp_dag.add_node(hp_name)
            tmp_dag.add_edge(_ROOT, hp_name)

        for parent_node_ in self._children:
            if parent_node_ == _ROOT:
                continue
            for child_node_ in self._children[parent_node_]:
                with contextlib.suppress(Exception):
                    tmp_dag.remove_edge(_ROOT, child_node_)

                condition = self._children[parent_node_][child_node_]
                tmp_dag.add_edge(parent_node_, child_node_, condition=condition)

        return tmp_dag

    def _check_edges(
        self,
        edges: list[tuple[Hyperparameter, Hyperparameter]],
        values: list[Any],
    ) -> None:
        for (parent, child), value in zip(edges, values, strict=False):
            # check if both nodes are already inserted into the graph
            if child.name not in self._hyperparameters:
                raise ChildNotFoundError(child, space=self)

            if parent.name not in self._hyperparameters:
                raise ParentNotFoundError(parent, space=self)

            if child != self._hyperparameters[child.name]:
                existing = self._hyperparameters[child.name]
                raise HyperparameterAlreadyExistsError(existing, child, space=self)

            if parent != self._hyperparameters[parent.name]:
                existing = self._hyperparameters[child.name]
                raise HyperparameterAlreadyExistsError(existing, child, space=self)

            _assert_legal(parent, value)

        # TODO: recursively check everything which is inside the conditions,
        # this means we have to recursively traverse the condition
        tmp_dag = self._create_tmp_dag()
        for parent, child in edges:
            tmp_dag.add_edge(parent.name, child.name)

        if not nx.is_directed_acyclic_graph(tmp_dag):
            cycles: list[list[str]] = list(nx.simple_cycles(tmp_dag))
            for cycle in cycles:
                cycle.sort()
            cycles.sort()
            raise CyclicDependancyError(cycles)

    def _update_cache(self) -> None:
        self._parent_conditions_of = {
            name: self._get_parent_conditions_of(name) for name in self._hyperparameters
        }
        self._child_conditions_of = {
            name: self._get_child_conditions_of(name) for name in self._hyperparameters
        }
        self._parents_of = {
            name: self.get_parents_of(name) for name in self._hyperparameters
        }
        self._children_of = {
            name: self.get_children_of(name) for name in self._hyperparameters
        }
        self._conditional_dep_graph = {
            name: self._parent_conditions_of[name]
            for name in ConfigSpace.c_util.topological_sort(self._parents_of)
        }

        # A lot of hyperparameters often depend on what is essentially they same
        # condition on its parent, i.e. some choice of algorithm dictating if the
        # hyperparameter is active.
        # We can speed up sampling by only computing conditionals that perform some
        # unique operation i.e. whatever they check on their parent.
        # This could be improed by considering transitivity
        # [ (condition, mask_of_hps_effected), ...]
        minimum_condition_span: list[tuple[Condition | Conjunction, list[int]]] = []
        for hp_name, parent_conditions in self._conditional_dep_graph.items():
            # If it's unconditional, skip
            if len(parent_conditions) == 0:
                continue

            hp_idx = self._hyperparameter_idx[hp_name]

            # Otherwise, for each of the conditions effecting this hp
            for condition_effecting_hp in parent_conditions:
                # If there's nothing yet, insert it as a minimum span condition
                if len(minimum_condition_span) == 0:
                    minimum_condition_span.append((condition_effecting_hp, [hp_idx]))
                else:
                    # Otherwise, iterate through the existing conditions and append
                    # it as a hyperparameter that is effected by the same condition
                    for min_span_cond, effected_hps in minimum_condition_span:
                        if condition_effecting_hp.conditionally_equal(min_span_cond):
                            effected_hps.append(hp_idx)
                            break
                    else:
                        minimum_condition_span.append(
                            (condition_effecting_hp, [hp_idx]),
                        )

        self.minimum_condition_span = [
            (c, np.asarray(effected_hps, dtype=np.int64))
            for c, effected_hps in minimum_condition_span
        ]

    def _check_forbidden_component(self, clause: ForbiddenLike) -> None:
        _assert_type(
            clause,
            (ForbiddenClause, ForbiddenConjunction, ForbiddenRelation),
            "_check_forbidden_component",
        )

        to_check: list[ForbiddenClause] = []
        relation_to_check: list[ForbiddenRelation] = []
        if isinstance(clause, ForbiddenClause):
            to_check.append(clause)
        elif isinstance(clause, ForbiddenConjunction):
            to_check.extend(clause.get_descendant_literal_clauses())
        elif isinstance(clause, ForbiddenRelation):
            relation_to_check.append(clause)
        else:
            raise NotImplementedError(type(clause))

        def _check_hp(tmp_clause: ForbiddenLike, hp: Hyperparameter) -> None:
            if hp.name not in self._hyperparameters:
                raise HyperparameterNotFoundError(
                    hp,
                    space=self,
                    preamble=(
                        f"Cannot add '{tmp_clause}' because it references '{hp.name}'"
                    ),
                )

        for tmp_clause in to_check:
            _check_hp(tmp_clause, tmp_clause.hyperparameter)

        for tmp_clause in relation_to_check:
            _check_hp(tmp_clause, tmp_clause.left)
            _check_hp(tmp_clause, tmp_clause.right)

    def _get_children_of(self, name: str) -> list[Hyperparameter]:
        conditions = self._get_child_conditions_of(name)
        children: list[Hyperparameter] = []
        for condition in conditions:
            if isinstance(condition, Conjunction):
                children.extend(condition.get_children())
            else:
                children.append(condition.child)

        return children

    def _get_child_conditions_of(self, name: str) -> list[Condition]:
        children = self._children[name]
        return [children[child_name] for child_name in children if child_name != _ROOT]

    def _get_parents_of(self, name: str) -> list[Hyperparameter]:
        """The parents hyperparameters of a given hyperparameter.

        Parameters
        ----------
        name : str

        Returns:
        -------
        list
            List with all parent hyperparameters
        """
        conditions = self._get_parent_conditions_of(name)
        parents: list[Hyperparameter] = []
        for condition in conditions:
            if isinstance(condition, Conjunction):
                parents.extend(condition.get_parents())
            else:
                parents.append(condition.parent)
        return parents

    def _check_default_configuration(self) -> Configuration:
        # Check if adding that hyperparameter leads to an illegal default configuration
        instantiated_hyperparameters: dict[str, int | float | str | None] = {}
        for hp in self.values():
            conditions = self._get_parent_conditions_of(hp.name)
            active: bool = True

            for condition in conditions:
                if isinstance(condition, Conjunction):
                    parent_names = (
                        c.parent.name
                        for c in condition.get_descendant_literal_conditions()
                    )
                else:
                    parent_names = [condition.parent.name]

                parents = {
                    parent_name: instantiated_hyperparameters[parent_name]
                    for parent_name in parent_names
                }

                # OPTIM: Can speed up things here by just breaking early?
                if not condition.satisfied_by_value(parents):
                    active = False

            if not active:
                # the evaluate above will use compares so we need to use None
                # and replace later....
                instantiated_hyperparameters[hp.name] = NotSet
            elif isinstance(hp, Constant):
                instantiated_hyperparameters[hp.name] = hp.value
            else:
                instantiated_hyperparameters[hp.name] = hp.default_value

                # TODO copy paste from check configuration

        # TODO add an extra Exception type for the case that the default
        # configuration is forbidden!
        return Configuration(self, values=instantiated_hyperparameters)

    def _get_parent_conditions_of(self, name: str) -> list[Condition | Conjunction]:
        parents = self._parents[name]
        return [parents[parent_name] for parent_name in parents if parent_name != _ROOT]

    def _check_configuration_rigorous(
        self,
        configuration: Configuration,
        allow_inactive_with_values: bool = False,
    ) -> None:
        vector = configuration.get_array()
        active_hyperparameters = self.get_active_hyperparameters(configuration)

        for hp_name, hyperparameter in self._hyperparameters.items():
            hp_value = vector[self._hyperparameter_idx[hp_name]]
            active = hp_name in active_hyperparameters

            if not np.isnan(hp_value) and not hyperparameter.legal_vector(hp_value):
                raise IllegalValueError(hyperparameter, hp_value)

            if active and np.isnan(hp_value):
                raise ActiveHyperparameterNotSetError(hyperparameter)

            if not allow_inactive_with_values and not active and not np.isnan(hp_value):
                raise InactiveHyperparameterSetError(hyperparameter, hp_value)

        self._check_forbidden(vector)

    def _check_forbidden(self, vector: np.ndarray) -> None:
        ConfigSpace.c_util.check_forbidden(self.forbidden_clauses, vector)

    # ------------ Marked Deprecated --------------------
    # Probably best to only remove these once we actually
    # make some other breaking changes
    # * Search `Marked Deprecated` to find others

    def get_hyperparameter(self, name: str) -> Hyperparameter:
        """Hyperparameter from the space with a given name.

        Parameters
        ----------
        name : str
            Name of the searched hyperparameter

        Returns:
        -------
        :ref:`Hyperparameters`
            Hyperparameter with the name ``name``
        """
        warnings.warn(
            "Prefer `space[name]` over `get_hyperparameter`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self[name]

    def get_hyperparameters(self) -> list[Hyperparameter]:
        """All hyperparameters in the space.

        Returns:
        -------
        list(:ref:`Hyperparameters`)
            A list with all hyperparameters stored in the configuration space object
        """
        warnings.warn(
            "Prefer using `list(space.values())` over `get_hyperparameters`",
            DeprecationWarning,
            stacklevel=2,
        )
        return list(self._hyperparameters.values())

    def get_hyperparameters_dict(self) -> dict[str, Hyperparameter]:
        """All the ``(name, Hyperparameter)`` contained in the space.

        Returns:
        -------
        dict(str, :ref:`Hyperparameters`)
            An OrderedDict of names and hyperparameters
        """
        warnings.warn(
            "Prefer using `dict(space)` over `get_hyperparameters_dict`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._hyperparameters.copy()

    def get_hyperparameter_names(self) -> list[str]:
        """Names of all the hyperparameter in the space.

        Returns:
        -------
        list(str)
            List of hyperparameter names
        """
        warnings.warn(
            "Prefer using `list(space.keys())` over `get_hyperparameter_names`",
            DeprecationWarning,
            stacklevel=2,
        )
        return list(self._hyperparameters.keys())

    # ---------------------------------------------------
