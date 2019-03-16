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

from collections import defaultdict, deque, OrderedDict
import copy
from itertools import chain

import numpy as np
import io

import ConfigSpace.nx
from ConfigSpace.hyperparameters import (
    Hyperparameter,
    Constant,
    FloatHyperparameter,
)
from ConfigSpace.conditions import (
    ConditionComponent,
    AbstractCondition,
    AbstractConjunction,
    EqualsCondition,
)
from ConfigSpace.forbidden import (
    AbstractForbiddenComponent,
    AbstractForbiddenClause,
    AbstractForbiddenConjunction,
)
from typing import Union, List, Any, Dict, Iterable, Set, Tuple, Optional
from ConfigSpace.exceptions import ForbiddenValueError
import ConfigSpace.c_util


class ConfigurationSpace(object):

    # TODO add a method to add whole configuration spaces as a child "tree"

    def __init__(
            self,
            name: Union[str, None] = None,
            seed: Union[int, None] = None,
            meta: Optional[Dict] = None,
    ) -> None:
        """
        A collection-like object containing a set of hyperparameter definitions and conditions.

        A configuration space organizes all hyperparameters and its conditions
        as well as its forbidden clauses. Configurations can be sampled from
        this configuration space. As underlying data structure, the
        configuration space uses a tree-based approach to represent the
        conditions and restrictions between hyperparameters.

        Parameters
        ----------
        name : (str, optional)
            Name of the configuration space
        seed : (int, optional)
            random seed
        meta : (dict, optional)
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        self.name = name
        self.meta = meta

        self._hyperparameters = OrderedDict()  # type: OrderedDict[str, Hyperparameter]
        self._hyperparameter_idx = dict()  # type: Dict[str, int]
        self._idx_to_hyperparameter = dict()  # type: Dict[int, str]

        # Use dictionaries to make sure that we don't accidently add
        # additional keys to these mappings (which happened with defaultdict()).
        # This once broke auto-sklearn's equal comparison of configuration
        # spaces when _children of one instance contained  all possible
        # hyperparameters as keys and empty dictionaries as values while the
        # other instance not containing these.
        self._children = OrderedDict()   # type: OrderedDict[str, OrderedDict[str, Union[None, AbstractCondition]]]
        self._parents = OrderedDict()   # type: OrderedDict[str, OrderedDict[str, Union[None, AbstractCondition]]]

        # changing this to a normal dict will break sampling because there is
        #  no guarantee that the parent of a condition was evaluated before
        self._conditionals = set()   # type: Set[str]
        self.forbidden_clauses = []  # type: List['AbstractForbiddenComponent']
        self.random = np.random.RandomState(seed)

        self._children['__HPOlib_configuration_space_root__'] = OrderedDict()

        # caching
        self._parent_conditions_of = dict()
        self._child_conditions_of = dict()
        self._parents_of = dict()
        self._children_of = dict()

    def generate_all_continuous_from_bounds(self, bounds: List[List[Any]]) -> None:
        """
        Generate :class:`~ConfigSpace.hyperparameters.UniformFloatHyperparameter`
        from a list containing lists with lower and upper bounds. The generated
        hyperparameters are added to the configuration space.
        
        Parameters
        ----------
        bounds : list(list([Any, Any)])
            List containing lists with two elements: lower and upper bound
        """
        for i, (l, u) in enumerate(bounds):
            hp = ConfigSpace.UniformFloatHyperparameter('x%d' % i, l, u)
            self.add_hyperparameter(hp)

    def add_hyperparameters(self, hyperparameters: List[Hyperparameter]) -> List[Hyperparameter]:
        """
        Add hyperparameters to the configuration space.

        Parameters
        ----------
        hyperparameters : list(:ref:`Hyperparameters`)
            Collection of hyperparameters to add

        Returns
        -------
        list(:ref:`Hyperparameters`)
            List of added hyperparameters (same as input)
        """

        for hyperparameter in hyperparameters:
            if not isinstance(hyperparameter, Hyperparameter):
                raise TypeError("Hyperparameter '%s' is not an instance of "
                                "ConfigSpace.hyperparameters.Hyperparameter." %
                                str(hyperparameter))

        for hyperparameter in hyperparameters:
            self._add_hyperparameter(hyperparameter)

        self._update_cache()
        self._check_default_configuration()
        self._sort_hyperparameters()
        return hyperparameters

    def add_hyperparameter(self, hyperparameter: Hyperparameter) -> Hyperparameter:
        """
        Add a hyperparameter to the configuration space.

        Parameters
        ----------
        hyperparameter : :ref:`Hyperparameters`
            The hyperparameter to add

        Returns
        -------
        :ref:`Hyperparameters`
            The added hyperparameter
        """
        if not isinstance(hyperparameter, Hyperparameter):
            raise TypeError("The method add_hyperparameter must be called "
                            "with an instance of "
                            "ConfigSpace.hyperparameters.Hyperparameter.")

        self._add_hyperparameter(hyperparameter)
        self._update_cache()
        self._check_default_configuration()
        self._sort_hyperparameters()

        return hyperparameter

    def _add_hyperparameter(self, hyperparameter: Hyperparameter) -> None:
        # Check if adding the hyperparameter is legal:
        # * Its name must not already exist
        if hyperparameter.name in self._hyperparameters:
            raise ValueError("Hyperparameter '%s' is already in the "
                             "configuration space." % hyperparameter.name)
        self._hyperparameters[hyperparameter.name] = hyperparameter
        self._children[hyperparameter.name] = OrderedDict()

        # TODO remove __HPOlib_configuration_space_root__, it is only used in
        # to check for cyclic configuration spaces. If it is only added when
        # cycles are checked, the code can become much easier (e.g. the parent
        # caching can be more or less removed).
        self._children['__HPOlib_configuration_space_root__'][
            hyperparameter.name] = None
        self._parents[hyperparameter.name] = OrderedDict()
        self._parents[hyperparameter.name][
            '__HPOlib_configuration_space_root__'] = None
        # Save the index of each hyperparameter name to later on access a
        # vector of hyperparameter values by indices, must be done twice
        # because check_default_configuration depends on it
        for i, hp in enumerate(self._hyperparameters):
            self._hyperparameter_idx[hp] = i

    def add_condition(self, condition: ConditionComponent) -> ConditionComponent:
        """
        Add a condition to the configuration space.
        Check if adding the condition is legal:

        - The parent in a condition statement must exist
        - The condition must add no cycles

        The following array keeps track of all edges which must be
        added to the DiGraph; if the checks don't raise any Exception,
        these edges are finally added at the end of the function.

        Parameters
        ----------
        condition : :ref:`Conditions`
            Condition to add

        Returns
        -------
        :ref:`Conditions`
            Same condition as input
        """
        if not isinstance(condition, ConditionComponent):
            raise TypeError("The method add_condition must be called "
                            "with an instance of "
                            "ConfigSpace.condition.ConditionComponent.")

        if isinstance(condition, AbstractCondition):
            self._check_edges(
                [(condition.parent, condition.child)],
                [condition.value],
            )
            self._check_condition(condition.child, condition)
            self._add_edge(
                condition.parent,
                condition.child,
                condition,
            )

        # Loop over the Conjunctions to find out the conditions we must add!
        elif isinstance(condition, AbstractConjunction):
            dlcs = condition.get_descendant_literal_conditions()
            edges = [(dlc.parent, dlc.child) for dlc in dlcs]
            values = [dlc.value for dlc in dlcs]
            self._check_edges(edges, values)

            for dlc in dlcs:
                self._check_condition(dlc.child, condition)
                self._add_edge(
                    dlc.parent,
                    dlc.child,
                    condition=condition,
                )

        else:
            raise Exception("This should never happen!")

        self._sort_hyperparameters()
        self._update_cache()
        return condition

    def add_conditions(self, conditions: List[ConditionComponent]) -> List[ConditionComponent]:
        """
        Add a list of conditions to the configuration space.
        They must be legal. Take a look at
        :meth:`~ConfigSpace.configuration_space.ConfigurationSpace.add_condition`.

        Parameters
        ----------
        conditions : list(:ref:`Conditions`)
            collection of conditions to add

        Returns
        -------
        list(:ref:`Conditions`)
            Same as input conditions
        """
        for condition in conditions:
            if not isinstance(condition, ConditionComponent):
                raise TypeError("Condition '%s' is not an instance of "
                                "ConfigSpace.condition.ConditionComponent." %
                                str(condition))

        edges = []
        values = []
        conditions_to_add = []
        for condition in conditions:
            if isinstance(condition, AbstractCondition):
                edges.append((condition.parent, condition.child))
                values.append(condition.value)
                conditions_to_add.append(condition)
            elif isinstance(condition, AbstractConjunction):
                dlcs = condition.get_descendant_literal_conditions()
                edges.extend(
                    [(dlc.parent, dlc.child) for dlc in dlcs])
                values.extend([dlc.value for dlc in dlcs])
                conditions_to_add.extend([condition] * len(dlcs))

        for edge, condition in zip(edges, conditions_to_add):
            self._check_condition(edge[1], condition)
        self._check_edges(edges, values)
        for edge, condition in zip(edges, conditions_to_add):
            self._add_edge(edge[0], edge[1], condition)

        self._sort_hyperparameters()
        self._update_cache()
        return conditions

    def _add_edge(
            self,
            parent_node: Hyperparameter,
            child_node: Hyperparameter,
            condition: ConditionComponent,
    ) -> None:
        try:
            # TODO maybe this has to be done more carefully
            del self._children['__HPOlib_configuration_space_root__'][child_node.name]
        except Exception:
            pass

        try:
            del self._parents[child_node.name]['__HPOlib_configuration_space_root__']
        except Exception:
            pass

        self._children[parent_node.name][child_node.name] = condition
        self._parents[child_node.name][parent_node.name] = condition
        self._conditionals.add(child_node.name)

    def _check_condition(self, child_node: Hyperparameter, condition: ConditionComponent) \
            -> None:
        for other_condition in self._get_parent_conditions_of(child_node.name):
            if other_condition != condition:
                raise ValueError("Adding a second condition (different) for a "
                                 "hyperparameter is ambigouos and "
                                 "therefore forbidden. Add a conjunction "
                                 "instead!\nAlready inserted: %s\nNew one: "
                                 "%s" % (str(other_condition), str(condition)))

    def _check_edges(
            self,
            edges: List[Tuple[Hyperparameter, Hyperparameter]],
            values: List[Union[float, str, int]]
    ) -> None:
        for (parent_node, child_node), value in zip(edges, values):
            # check if both nodes are already inserted into the graph
            if child_node.name not in self._hyperparameters:
                raise ValueError(
                    "Child hyperparameter '%s' not in configuration "
                    "space." % child_node.name)
            if child_node != self._hyperparameters[child_node.name]:
                # TODO test this
                raise ValueError(
                    "Child hyperparameter '%s' different to hyperparameter "
                    "with the same name in configuration space: '%s'." %
                    (child_node, self._hyperparameters[child_node.name])
                )
            if parent_node.name not in self._hyperparameters:
                raise ValueError(
                    "Parent hyperparameter '%s' not in configuration "
                    "space." % parent_node.name)
            if parent_node != self._hyperparameters[parent_node.name]:
                # TODO test this
                raise ValueError(
                    "Parent hyperparameter '%s' different to hyperparameter "
                    "with the same name in configuration space: '%s'." %
                    (parent_node, self._hyperparameters[parent_node.name])
                )
            if isinstance(value, (tuple, list)):
                # TODO test this
                for v in value:
                    if not self._hyperparameters[parent_node.name].is_legal(v):
                        raise ValueError(
                            "Value '%s' is not legal for hyperparameter %s." %
                            (v, self._hyperparameters[parent_node.name])
                        )
            else:
                if not self._hyperparameters[parent_node.name].is_legal(value):
                    raise ValueError(
                        "Value '%s' is not legal for hyperparameter %s." %
                        (value, self._hyperparameters[parent_node.name])
                    )

        # TODO: recursively check everything which is inside the conditions,
        # this means we have to recursively traverse the condition

        tmp_dag = self._create_tmp_dag()
        for parent_node, child_node in edges:
            tmp_dag.add_edge(parent_node.name, child_node.name)

        if not ConfigSpace.nx.is_directed_acyclic_graph(tmp_dag):
            cycles = list(
                ConfigSpace.nx.simple_cycles(tmp_dag)
            )  # type: List[List[str]]
            for cycle in cycles:
                cycle.sort()
            cycles.sort()
            raise ValueError("Hyperparameter configuration contains a "
                             "cycle %s" % str(cycles))

    def _sort_hyperparameters(self) -> None:
        levels = OrderedDict()  # type: OrderedDict[str, int]
        to_visit = deque()  # type: ignore
        for hp_name in self._hyperparameters:
            to_visit.appendleft(hp_name)

        while len(to_visit) > 0:
            current = to_visit.pop()
            if '__HPOlib_configuration_space_root__' in self._parents[current]:
                assert len(self._parents[current]) == 1
                levels[current] = 1

            else:
                all_parents_visited = True
                depth = -1
                for parent in self._parents[current]:
                    if parent not in levels:
                        all_parents_visited = False
                        break
                    else:
                        depth = max(depth, levels[parent] + 1)

                if all_parents_visited:
                    levels[current] = depth
                else:
                    to_visit.appendleft(current)

        by_level = defaultdict(list)  # type: defaultdict[int, List[str]]
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
        new_order['__HPOlib_configuration_space_root__'] = self._children['__HPOlib_configuration_space_root__']
        for hp in chain(['__HPOlib_configuration_space_root__'], self._hyperparameters):
            # Also resort the children dict
            children_sorting = [(self._hyperparameter_idx[child_name], child_name)
                                for child_name in self._children[hp]]
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
            if '__HPOlib_configuration_space_root__' in self._parents[hp]:
                parent_sorting = [(-1, '__HPOlib_configuration_space_root__')]
            else:
                parent_sorting = [(self._hyperparameter_idx[parent_name], parent_name)
                                  for parent_name in self._parents[hp]]
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

    def _update_cache(self):
        self._parent_conditions_of = dict()
        self._child_conditions_of = dict()
        self._parents_of = dict()
        self._children_of = dict()

        for hp_name in self._hyperparameters:
            self._parent_conditions_of[hp_name] = self._get_parent_conditions_of(hp_name)
            self._child_conditions_of[hp_name] = self._get_child_conditions_of(hp_name)
            self._parents_of[hp_name] = self.get_parents_of(hp_name)
            self._children_of[hp_name] = self.get_children_of(hp_name)

    def _create_tmp_dag(self) -> ConfigSpace.nx.DiGraph:
        tmp_dag = ConfigSpace.nx.DiGraph()
        for hp_name in self._hyperparameters:
            tmp_dag.add_node(hp_name)
            tmp_dag.add_edge('__HPOlib_configuration_space_root__', hp_name)

        for parent_node_ in self._children:
            if parent_node_ == '__HPOlib_configuration_space_root__':
                continue
            for child_node_ in self._children[parent_node_]:
                try:
                    tmp_dag.remove_edge('__HPOlib_configuration_space_root__',
                                        child_node_)
                except Exception:
                    pass
                condition = self._children[parent_node_][child_node_]
                tmp_dag.add_edge(parent_node_, child_node_, condition=condition)

        return tmp_dag

    def add_forbidden_clause(self, clause: AbstractForbiddenComponent) -> AbstractForbiddenComponent:
        """
        Add a forbidden clause to the configuration space.

        Parameters
        ----------
        clause : :ref:`Forbidden clauses`
            Forbidden clause to add

        Returns
        -------
        :ref:`Forbidden clauses`
            Same as input forbidden clause
        """
        self._check_forbidden_component(clause=clause)
        clause.set_vector_idx(self._hyperparameter_idx)
        self.forbidden_clauses.append(clause)
        self._check_default_configuration()
        return clause

    def add_forbidden_clauses(self, clauses: List[AbstractForbiddenComponent]) -> List[AbstractForbiddenComponent]:
        """
        Add a list of forbidden clauses to the configuration space.

        Parameters
        ----------
        clauses : list(:ref:`Forbidden clauses`)
            Collection of forbidden clauses to add

        Returns
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

    def _check_forbidden_component(self, clause: AbstractForbiddenComponent):
        if not isinstance(clause, AbstractForbiddenComponent):
            raise TypeError("The method add_forbidden_clause must be called "
                            "with an instance of "
                            "ConfigSpace.forbidden.AbstractForbiddenComponent.")
        to_check = list()
        if isinstance(clause, AbstractForbiddenClause):
            to_check.append(clause)
        elif isinstance(clause, AbstractForbiddenConjunction):
            to_check.extend(clause.get_descendant_literal_clauses())
        else:
            raise NotImplementedError(type(clause))

        for tmp_clause in to_check:
            if tmp_clause.hyperparameter.name not in self._hyperparameters:
                raise ValueError(
                    "Cannot add clause '%s' because it references hyperparameter"
                    " %s which is not in the configuration space (allowed "
                    "hyperparameters are: %s)"
                    % (
                        tmp_clause,
                        tmp_clause.hyperparameter.name,
                        list(self._hyperparameters),
                    )
                )

    def add_configuration_space(self,
                                prefix: str,
                                configuration_space: 'ConfigurationSpace',
                                delimiter: str = ":",
                                parent_hyperparameter: Hyperparameter = None
                                ) -> 'ConfigurationSpace':
        """
        Combine two configuration space by adding one the other configuration
        space. The contents of the configuration space, which should be added,
        are renamed to ``prefix`` + ``delimiter`` + old_name.

        Parameters
        ----------
        prefix : str
            The prefix for the renamed hyperparameter | conditions |
            forbidden clauses
        configuration_space : :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
            The configuration space which should be added
        delimiter : (str, optional)
            Defaults to '':''.
        parent_hyperparameter : (:ref:`Hyperparameters`, optional)
            Adds for each new hyperparameter the condition, that
            ``parent_hyperparameter`` is active

        Returns
        -------
        :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
            The configuration space, which was added
        """
        if not isinstance(configuration_space, ConfigurationSpace):
            raise TypeError("The method add_configuration_space must be "
                            "called with an instance of "
                            "ConfigSpace.configuration_space."
                            "ConfigurationSpace.")

        new_parameters = []
        for hp in configuration_space.get_hyperparameters():
            new_parameter = copy.copy(hp)
            # Allow for an empty top-level parameter
            if new_parameter.name == '':
                new_parameter.name = prefix
            else:
                new_parameter.name = "%s%s%s" % (prefix, delimiter,
                                                 new_parameter.name)
            new_parameters.append(new_parameter)
        self.add_hyperparameters(new_parameters)

        conditions_to_add = []
        for condition in configuration_space.get_conditions():
            new_condition = copy.copy(condition)
            dlcs = new_condition.get_descendant_literal_conditions()
            for dlc in dlcs:
                if dlc.child.name == prefix or dlc.child.name == '':
                    dlc.child.name = prefix
                elif not dlc.child.name.startswith(
                                "%s%s" % (prefix, delimiter)):
                    dlc.child.name = "%s%s%s" % (
                        prefix, delimiter, dlc.child.name)
                if dlc.parent.name == prefix or dlc.parent.name == '':
                    dlc.parent.name = prefix
                elif not dlc.parent.name.startswith(
                                "%s%s" % (prefix, delimiter)):
                    dlc.parent.name = "%s%s%s" % (
                        prefix, delimiter, dlc.parent.name)
            conditions_to_add.append(new_condition)
        self.add_conditions(conditions_to_add)

        forbiddens_to_add = []
        for forbidden_clause in configuration_space.forbidden_clauses:
            # new_forbidden = copy.deepcopy(forbidden_clause)
            new_forbidden = forbidden_clause
            dlcs = new_forbidden.get_descendant_literal_clauses()
            for dlc in dlcs:
                if dlc.hyperparameter.name == prefix or \
                                dlc.hyperparameter.name == '':
                    dlc.hyperparameter.name = prefix
                elif not dlc.hyperparameter.name.startswith(
                                "%s%s" % (prefix, delimiter)):
                    dlc.hyperparameter.name = "%s%s%s" % \
                                              (prefix, delimiter,
                                               dlc.hyperparameter.name)
            forbiddens_to_add.append(new_forbidden)
        self.add_forbidden_clauses(forbiddens_to_add)

        conditions_to_add = []
        if parent_hyperparameter is not None:
            for new_parameter in new_parameters:
                # Only add a condition if the parameter is a top-level
                # parameter of the new configuration space (this will be some
                #  kind of tree structure).
                if self.get_parents_of(new_parameter):
                    continue
                condition = EqualsCondition(new_parameter,
                                            parent_hyperparameter['parent'],
                                            parent_hyperparameter['value'])
                conditions_to_add.append(condition)
        self.add_conditions(conditions_to_add)

        return configuration_space

    def get_hyperparameters(self) -> List[Hyperparameter]:
        """
        Return a list with all the hyperparameter, which are contained in the
        configuration space object.

        Returns
        -------
        list(:ref:`Hyperparameters`)
            A list with all hyperparameters stored in the configuration
            space object
        """
        return list(self._hyperparameters.values())

    def get_hyperparameter_names(self) -> List[str]:
        """
        Return a list with all names of hyperparameter, which are contained in
        the configuration space object.

        Returns
        -------
        list(str)
            List of hyperparameter names

        """
        return list(self._hyperparameters.keys())

    def get_hyperparameter(self, name: str) -> Hyperparameter:
        """
        Gives the hyperparameter from the configuration space given its name.

        Parameters
        ----------
        name : str
            Name of the searched hyperparameter

        Returns
        -------
        :ref:`Hyperparameters`
            Hyperparameter with the name ``name``

        """
        hp = self._hyperparameters.get(name)

        if hp is None:
            if self.name is None:
                raise KeyError("Hyperparameter '%s' does not exist in this "
                               "configuration space." % name)
            else:
                raise KeyError("Hyperparameter '%s' does not exist in  "
                               "configuration space %s." % (name, self.name))
        else:
            return hp

    def get_hyperparameter_by_idx(self, idx: int) -> str:
        """
        Return the name of a hyperparameter from the configuration space given
        its id.

        Parameters
        ----------
        idx : int
            Id of a hyperparameter

        Returns
        -------
        str
            Name of the hyperparameter

        """
        hp = self._idx_to_hyperparameter.get(idx)

        if hp is None:
            if self.name is None:
                raise KeyError("Hyperparameter #'%d' does not exist in this "
                               "configuration space." % idx)
            else:
                raise KeyError("Hyperparameter #'%d' does not exist in  "
                               "configuration space %s." % (idx, self.name))
        else:
            return hp

    def get_idx_by_hyperparameter_name(self, name: str) -> int:
        """
        Return the id of a hyperparameter by its ``name``.

        Parameters
        ----------
        name : str
            Name of a hyperparameter

        Returns
        -------
        int
            Id of the hyperparameter with name ``name``

        """
        idx = self._hyperparameter_idx.get(name)

        if idx is None:
            if self.name is None:
                raise KeyError("Hyperparameter '%s' does not exist in this "
                               "configuration space." % name)
            else:
                raise KeyError("Hyperparameter '%s' does not exist in  "
                               "configuration space %s." % (name, self.name))
        else:
            return idx

    def get_conditions(self) -> List[AbstractCondition]:
        """
        Return a list with all conditions from the configuration space.

        Returns
        -------
        list(:ref:`Conditions`)
            Conditions of the configuration space

        """
        conditions = []
        added_conditions = set()  # type: Set[str]

        # Nodes is a list of nodes
        for source_node in self.get_hyperparameters():
            # This is a list of keys in a dictionary
            # TODO sort the edges by the order of their source_node in the
            # hyperparameter list!
            for target_node in self._children[source_node.name]:
                if target_node not in added_conditions:
                    condition = self._children[source_node.name][target_node]
                    conditions.append(condition)
                    added_conditions.add(target_node)

        return conditions

    def get_forbiddens(self) -> List[AbstractForbiddenComponent]:
        """
        Return a list with all forbidden clauses from the configuration space.

        Returns
        -------
        list(:ref:`Forbidden clauses`)
            List with the forbidden clauses
        """
        return self.forbidden_clauses

    def get_children_of(self, name: Union[str, Hyperparameter]) -> List[Hyperparameter]:
        """
        Return a list with all children of a given hyperparameter.

        Parameters
        ----------
        name : (str, :ref:`Hyperparameters`)
            Hyperparameter or its name, for which all children are requested

        Returns
        -------
        list(:ref:`Hyperparameters`)
            Children of the hyperparameter

        """
        conditions = self.get_child_conditions_of(name)
        parents = []  # type: List[Hyperparameter]
        for condition in conditions:
            parents.extend(condition.get_children())
        return parents

    def _get_children_of(self, name: str) -> List[Hyperparameter]:
        conditions = self._get_child_conditions_of(name)
        parents = []  # type: List[Hyperparameter]
        for condition in conditions:
            parents.extend(condition.get_children())
        return parents

    def get_child_conditions_of(self, name: Union[str, Hyperparameter]) -> List[AbstractCondition]:
        """
        Return a list with conditions of all children of a given
        hyperparameter referenced by its ``name``.

        Parameters
        ----------
        name : (str, :ref:`Hyperparameters`)
            Hyperparameter or its name, for which conditions are requested

        Returns
        -------
        list(:ref:`Conditions`)
            List with the conditions on the children of the given hyperparameter

        """
        if isinstance(name, Hyperparameter):
            name = name.name  # type: ignore

        # This raises an exception if the hyperparameter does not exist
        self.get_hyperparameter(name)
        return self._get_child_conditions_of(name)

    def _get_child_conditions_of(self, name: str) -> List[AbstractCondition]:
        children = self._children[name]
        conditions = [children[child_name] for child_name in children
                      if child_name != "__HPOlib_configuration_space_root__"]
        return conditions

    def get_parents_of(self, name: Union[str, Hyperparameter]) -> List[Hyperparameter]:
        """
        Return the parent hyperparameters of a given hyperparameter.

        Parameters
        ----------
        name : (str, :ref:`Hyperparameters`)
            Can either be the name of a hyperparameter or the hyperparameter
            object

        Returns
        -------
        list[:ref:`Conditions`]
            List with all parent hyperparameters
        """

        conditions = self.get_parent_conditions_of(name)
        parents = []  # type: List[Hyperparameter]
        for condition in conditions:
            parents.extend(condition.get_parents())
        return parents

    def _get_parents_of(self, name: str) -> List[Hyperparameter]:
        """
        Return the parent hyperparameters of a given hyperparameter.

        Parameters
        ----------
        name : str

        Returns
        -------
        list
            List with all parent hyperparameters
        """
        conditions = self._get_parent_conditions_of(name)
        parents = []  # type: List[Hyperparameter]
        for condition in conditions:
            parents.extend(condition.get_parents())
        return parents

    def get_parent_conditions_of(self, name: Union[str, Hyperparameter]) -> List[AbstractCondition]:
        """
        Return a list with conditions of all parents of a given hyperparameter.

        Parameters
        ----------
        name : (str, :ref:`Hyperparameters`)
            Can either be the name of a hyperparameter or the hyperparameter
            object

        Returns
        -------
        List[:ref:`Conditions`]
            List with all conditions on parent hyperparameters

        """
        if isinstance(name, Hyperparameter):
            name = name.name  # type: ignore

        # This raises an exception if the hyperparameter does not exist
        self.get_hyperparameter(name)
        return self._get_parent_conditions_of(name)

    def _get_parent_conditions_of(self, name: str) -> List[AbstractCondition]:
        parents = self._parents[name]
        conditions = [parents[parent_name] for parent_name in parents
                      if parent_name != "__HPOlib_configuration_space_root__"]
        return conditions

    def get_all_unconditional_hyperparameters(self) -> List[str]:
        """
        Return a list with names of unconditional hyperparameters.

        Returns
        -------
        list[:ref:`Hyperparameters`]
            List with all parent hyperparameters, which are not part of a condition

        """
        hyperparameters = [hp_name for hp_name in
                           self._children[
                               '__HPOlib_configuration_space_root__']]
        return hyperparameters

    def get_all_conditional_hyperparameters(self) -> List[str]:
        """
        Return a list with names of all conditional hyperparameters.

        Returns
        -------
        list[:ref:`Hyperparameters`]
            List with all conditional hyperparameter

        """
        return self._conditionals

    def get_default_configuration(self) -> 'Configuration':
        """
        Return a configuration containing hyperparameters with default values.

        Returns
        -------
        :class:`~ConfigSpace.configuration_space.Configuration`
            Configuration with the set default values

        """
        return self._check_default_configuration()

    def _check_default_configuration(self) -> 'Configuration':
        # Check if adding that hyperparameter leads to an illegal default configuration
        instantiated_hyperparameters = {}  # type: Dict[str, Optional[Union[int, float, str]]]
        for hp in self.get_hyperparameters():
            conditions = self._get_parent_conditions_of(hp.name)
            active = True
            for condition in conditions:
                parent_names = [c.parent.name for c in
                                condition.get_descendant_literal_conditions()]

                parents = {
                    parent_name: instantiated_hyperparameters[parent_name]
                    for parent_name in parent_names
                }

                if not condition.evaluate(parents):
                    # TODO find out why a configuration is illegal!
                    active = False

            if not active:
                instantiated_hyperparameters[hp.name] = None
            elif isinstance(hp, Constant):
                instantiated_hyperparameters[hp.name] = hp.value
            else:
                instantiated_hyperparameters[hp.name] = hp.default_value

                # TODO copy paste from check configuration

        # TODO add an extra Exception type for the case that the default
        # configuration is forbidden!
        return Configuration(self, instantiated_hyperparameters)

    # For backward compatibility
    def check_configuration(self, configuration: 'Configuration') -> None:
        """
        Check if a configuration is legal. Raises an error if not.

        Parameters
        ----------
        configuration : :class:`~ConfigSpace.configuration_space.Configuration`
            Configuration to check
        """
        if not isinstance(configuration, Configuration):
            raise TypeError("The method check_configuration must be called "
                            "with an instance of %s. "
                            "Your input was of type %s" % (Configuration, type(configuration)))
        ConfigSpace.c_util.check_configuration(
            self, configuration.get_array(), False
        )

    def check_configuration_vector_representation(self, vector: np.ndarray) -> None:
        """
        Raise error f configuration in vector representation is not legal.

        Parameters
        ----------
        vector : np.ndarray
            Configuration in vector representation
        """
        if not isinstance(vector, np.ndarray):
            raise TypeError("The method check_configuration must be called "
                            "with an instance of np.ndarray "
                            "Your input was of type %s" % (type(vector)))
        ConfigSpace.c_util.check_configuration(self, vector, False)

    def get_active_hyperparameters(self, configuration: 'Configuration') -> Set:
        """
        Return a set of active hyperparameter for a given configuration.

        Parameters
        ----------
        configuration : :class:`~ConfigSpace.configuration_space.Configuration`
            Configuration for which the active hyperparameter are returned

        Returns
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

                parent_vector_idx = condition.get_parents_vector()

                # if one of the parents is None, the hyperparameter cannot be
                # active! Else we have to check this
                # Note from trying to optimize this - this is faster than using
                # dedicated numpy functions and indexing
                if any([vector[i] != vector[i] for i in parent_vector_idx]):
                    active = False
                    break

                else:
                    if not condition.evaluate_vector(vector):
                        active = False
                        break

            if active:
                active_hyperparameters.add(hp_name)
        return active_hyperparameters

    def _check_configuration_rigorous(self, configuration: 'Configuration',
                                      allow_inactive_with_values: bool = False) -> None:
        vector = configuration.get_array()
        active_hyperparameters = self.get_active_hyperparameters(configuration)

        for hp_name, hyperparameter in self._hyperparameters.items():
            hp_value = vector[self._hyperparameter_idx[hp_name]]
            active = hp_name in active_hyperparameters

            if not np.isnan(hp_value) and not hyperparameter.is_legal_vector(hp_value):
                raise ValueError("Hyperparameter instantiation '%s' "
                                 "(type: %s) is illegal for hyperparameter %s" %
                                 (hp_value, str(type(hp_value)),
                                  hyperparameter))

            if active and np.isnan(hp_value):
                raise ValueError("Active hyperparameter '%s' not specified!" %
                                 hyperparameter.name)

            if not allow_inactive_with_values and not active and \
               not np.isnan(hp_value):
                raise ValueError("Inactive hyperparameter '%s' must not be "
                                 "specified, but has the vector value: '%s'." %
                                 (hp_name, hp_value))
        self._check_forbidden(vector)

    def _check_forbidden(self, vector: np.ndarray) -> None:
        ConfigSpace.c_util.check_forbidden(self.forbidden_clauses, vector)
        # for clause in self.forbidden_clauses:
        #    if clause.is_forbidden_vector(vector, strict=False):
        #        raise ForbiddenValueError("Given vector violates forbidden
        # clause %s" % (str(clause)))

    # http://stackoverflow.com/a/25176504/4636294
    def __eq__(self, other: Any) -> bool:
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            this_dict = self.__dict__.copy()
            del this_dict['random']
            other_dict = other.__dict__.copy()
            del other_dict['random']
            return this_dict == other_dict
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(self.__repr__())

    def __repr__(self) -> str:
        retval = io.StringIO()
        retval.write("Configuration space object:\n  Hyperparameters:\n")

        if self.name is not None:
            retval.write(self.name)
            retval.write('\n')

        hyperparameters = sorted(self.get_hyperparameters(),
                                 key=lambda t: t.name)
        if hyperparameters:
            retval.write("    ")
            retval.write("\n    ".join(
                [str(hyperparameter) for hyperparameter in hyperparameters]))
            retval.write("\n")

        conditions = sorted(self.get_conditions(),
                            key=lambda t: str(t))
        if conditions:
            retval.write("  Conditions:\n")
            retval.write("    ")
            retval.write("\n    ".join(
                [str(condition) for condition in conditions]))
            retval.write("\n")

        if self.get_forbiddens():
            retval.write("  Forbidden Clauses:\n")
            retval.write("    ")
            retval.write("\n    ".join(
                [str(clause) for clause in self.get_forbiddens()]))
            retval.write("\n")

        retval.seek(0)
        return retval.getvalue()

    def __iter__(self) -> Iterable:
        """ Allows to iterate over the hyperparameter names in (hopefully?) the right order."""
        return iter(self._hyperparameters.keys())

    def sample_configuration(self, size: int = 1) -> Union['Configuration', List['Configuration']]:
        """
        Sample ``size`` configurations from the configuration space object.

        Parameters
        ----------
        size : (int, optional)
            Number of configurations to sample. Default to 1

        Returns
        -------
        :class:`~ConfigSpace.configuration_space.Configuration`, List[:class:`~ConfigSpace.configuration_space.Configuration`]:
            A single configuration if ``size`` 1 else a list of Configurations
        """
        if not isinstance(size, int):
            raise TypeError('Argument size must be of type int, but is %s'
                            % type(size))
        elif size < 1:
            return []

        iteration = 0
        missing = size
        accepted_configurations = []  # type: List['Configuration']
        num_hyperparameters = len(self._hyperparameters)

        unconditional_hyperparameters = self.get_all_unconditional_hyperparameters()
        hyperparameters_with_children = list()

        _forbidden_clauses_unconditionals = []
        _forbidden_clauses_conditionals = []
        for clause in self.get_forbiddens():
            based_on_conditionals = False
            for subclause in clause.get_descendant_literal_clauses():
                if subclause.hyperparameter.name not in unconditional_hyperparameters:
                    based_on_conditionals = True
                    break
            if based_on_conditionals:
                _forbidden_clauses_conditionals.append(clause)
            else:
                _forbidden_clauses_unconditionals.append(clause)

        for uhp in unconditional_hyperparameters:
            children = self._children_of[uhp]
            if len(children) > 0:
                hyperparameters_with_children.append(uhp)

        while len(accepted_configurations) < size:
            if missing != size:
                missing = int(1.1 * missing)
            vector = np.ndarray((missing, num_hyperparameters),
                                dtype=np.float64)

            for i, hp_name in enumerate(self._hyperparameters):
                hyperparameter = self._hyperparameters[hp_name]
                vector[:, i] = hyperparameter._sample(self.random, missing)

            for i in range(missing):
                try:
                    configuration = Configuration(
                        self,
                        vector=ConfigSpace.c_util.correct_sampled_array(
                            vector[i].copy(),
                            _forbidden_clauses_unconditionals,
                            _forbidden_clauses_conditionals,
                            hyperparameters_with_children,
                            num_hyperparameters,
                            unconditional_hyperparameters,
                            self._hyperparameter_idx,
                            self._parent_conditions_of,
                            self._parents_of,
                            self._children_of,
                        ))
                    accepted_configurations.append(configuration)
                except ForbiddenValueError:
                    iteration += 1

                    if iteration == size * 100:
                        raise ForbiddenValueError(
                            "Cannot sample valid configuration for "
                            "%s" % self)

            missing = size - len(accepted_configurations)

        if size <= 1:
            return accepted_configurations[0]
        else:
            return accepted_configurations

    def seed(self, seed: int) -> None:
        """
        Set the random seed to a number.

        Parameters
        ----------
        seed : int
            The random seed
        """
        self.random = np.random.RandomState(seed)


class Configuration(object):
    def __init__(self, configuration_space: ConfigurationSpace,
                 values: Union[None,  Dict[str, Union[str, float, int]]] = None,
                 vector: Union[None, np.ndarray] = None,
                 allow_inactive_with_values: bool = False, origin: Any = None) -> None:
        """
        Class for a single configuration.

        The :class:`~ConfigSpace.configuration_space.Configuration` object holds
        for all active hyperparameters a value. While the
        :class:`~ConfigSpace.configuration_space.ConfigurationSpace` stores the
        definitions for the hyperparameters (value ranges, constraints,...), a
        :class:`~ConfigSpace.configuration_space.Configuration` object is
        more like a instance of it.

        Parameters
        ----------
        configuration_space : :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        values : (dict, optional)
            A dictionary with pairs (hyperparameter_name, value), where value is
            a legal value of the hyperparameter in the above
            configuration_space
        vector : (np.ndarray, optional)
            A numpy array for efficient representation. Either values or vector
            has to be given
        allow_inactive_with_values : (bool, optional)
            Whether an Exception will be raised if a value for an inactive
            hyperparameter is given. Default is to raise an Exception.
            Default to False
        origin : (Any, optional)
            Store information about the origin of this configuration.
            Default to None
        """
        if not isinstance(configuration_space, ConfigurationSpace):
            raise TypeError("Configuration expects an instance of %s, "
                            "you provided '%s'" %
                            (ConfigurationSpace, type(configuration_space)))

        self.configuration_space = configuration_space
        self.allow_inactive_with_values = allow_inactive_with_values
        self._query_values = False
        self._num_hyperparameters = len(self.configuration_space._hyperparameters)
        self.origin = origin
        self._keys = None  # type: Union[None, List[str]]

        if values is not None and vector is not None:
            raise ValueError('Configuration specified both as dictionary and '
                             'vector, can only do one.')
        if values is not None:
            # Using cs._hyperparameters to iterate makes sure that the
            # hyperparameters in the configuration are sorted in the same way as
            # they are sorted in the configuration space
            self._values = dict()  # type: Dict[str, Union[str, float, int]]
            for key in configuration_space._hyperparameters:
                value = values.get(key)
                if value is None:
                    continue
                hyperparameter = configuration_space.get_hyperparameter(key)
                if not hyperparameter.is_legal(value):
                    raise ValueError("Trying to set illegal value '%s' (type '%s') for "
                                     "hyperparameter '%s' (default-value has type '%s')." % (
                                       str(value), type(value), hyperparameter, type(hyperparameter.default_value)))
                # Truncate the representation of the float to be of constant
                # length for a python version
                if isinstance(hyperparameter, FloatHyperparameter):
                    value = float(repr(value))

                self._values[key] = value

            for key in values:
                if key not in configuration_space._hyperparameters:
                    raise ValueError('Tried to specify unknown hyperparameter '
                                     '%s' % key)

            self._query_values = True
            self._vector = np.ndarray((self._num_hyperparameters,),
                                      dtype=np.float)

            # Populate the vector
            # TODO very unintuitive calls...
            for key in configuration_space._hyperparameters:
                self._vector[self.configuration_space._hyperparameter_idx[
                    key]] = self.configuration_space.get_hyperparameter(key). \
                    _inverse_transform(self[key])
            self.is_valid_configuration()

        elif vector is not None:
            self._values = dict()
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=float)
            if len(vector.shape) > 1:
                if len(vector.shape) == 2 and vector.shape[1] == 1:
                    vector = vector.flatten()
                else:
                    raise ValueError(
                        'Only 1d arrays can be converted to a Configuration, '
                        'you passed an array of shape %s.' % str(vector.shape)
                    )
            if len(vector) != len(self.configuration_space.get_hyperparameters()):
                raise ValueError(
                    'Expected array of length %d, got %d' %
                    (len(self.configuration_space.get_hyperparameters()), len(vector))
                )
            self._vector = vector
        else:
            raise ValueError('Configuration neither specified as dictionary '
                             'or vector.')

    def is_valid_configuration(self) -> None:
        """
        Check if the object is a valid
        :class:`~ConfigSpace.configuration_space.Configuration`.
        Raise an error if configuration is not valid.
        """
        ConfigSpace.c_util.check_configuration(
            self.configuration_space,
            self._vector,
            allow_inactive_with_values=self.allow_inactive_with_values
        )

    def __getitem__(self, item: str) -> Any:
        if self._query_values or item in self._values:
            return self._values.get(item)

        hyperparameter = self.configuration_space._hyperparameters[item]
        item_idx = self.configuration_space._hyperparameter_idx[item]

        if not np.isfinite(self._vector[item_idx]):
            raise KeyError()

        value = hyperparameter._transform(self._vector[item_idx])
        # Truncate the representation of the float to be of constant
        # length for a python version
        if isinstance(hyperparameter, FloatHyperparameter):
            value = float(repr(value))
        # TODO make everything faster, then it'll be possible to init all values
        # at the same time and use an OrderedDict instead of only a dict here to
        # support iterating that dict in the same order as the actual order of
        # hyperparameters
        self._values[item] = value
        return self._values[item]

    def get(self, item: str, default: Union[None, Any] = None) -> Union[None, Any]:
        """
        Return for a given hyperparameter name ``item`` the value of this
        hyperparameter. ``default`` if the hyperparameter ``name`` doesn't exist.

        Parameters
        ----------
        item : str
            Name of the desired hyperparameter
        default : (None, Any)

        Returns
        -------
        Any, None
            Value of the hyperparameter
        """
        try:
            return self[item]
        except Exception:
            return default

    def __setitem__(self, key, value):
        param = self.configuration_space.get_hyperparameter(key)
        if not param.is_legal(value):
            raise ValueError(
                "Illegal value '%s' for hyperparameter %s" % (str(value), key))
        idx = self.configuration_space.get_idx_by_hyperparameter_name(key)
        vector_value = param._inverse_transform(value)
        new_array = ConfigSpace.c_util.change_hp_value(
            self.configuration_space,
            self.get_array().copy(),
            param.name,
            vector_value,
            idx
        )
        ConfigSpace.c_util.check_configuration(
            self.configuration_space,
            new_array,
            False
        )
        self._vector = new_array
        self._values = dict()
        self._query_values = False

    def __contains__(self, item: str) -> bool:
        self._populate_values()
        return item in self._values

    # http://stackoverflow.com/a/25176504/4636294
    def __eq__(self, other: Any) -> bool:
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            self._populate_values()
            other._populate_values()
            return self._values == other._values and \
                self.configuration_space == other.configuration_space
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Override the default hash behavior (that returns the id or the object)"""
        self._populate_values()
        return hash(self.__repr__())

    def _populate_values(self) -> None:
        if self._query_values is False:
            for hyperparameter in self.configuration_space.get_hyperparameters():
                self.get(hyperparameter.name)
            self._query_values = True

    def __repr__(self) -> str:
        self._populate_values()

        representation = io.StringIO()
        representation.write("Configuration:\n")

        hyperparameters = self.configuration_space.get_hyperparameters()
        hyperparameters.sort(key=lambda t: t.name)
        for hyperparameter in hyperparameters:
            hp_name = hyperparameter.name
            if hp_name in self._values and self._values[hp_name] is not None:
                representation.write("  ")

                value = repr(self._values[hp_name])
                if isinstance(hyperparameter, Constant):
                    representation.write("%s, Constant: %s" % (hp_name, value))
                else:
                    representation.write("%s, Value: %s" % (hp_name, value))
                representation.write("\n")

        return representation.getvalue()

    def __iter__(self) -> Iterable:
        return iter(self.keys())

    def keys(self) -> List[str]:
        """
        Cache the keys to speed up the process of retrieving the keys.

        Returns
        -------
        list(str)
            list of keys

        """
        if self._keys is None:
            keys = list(self.configuration_space._hyperparameters.keys())
            keys = [
                key for i, key in enumerate(keys) if
                np.isfinite(self._vector[i])
            ]
            self._keys = keys
        return self._keys

    def get_dictionary(self) -> Dict[str, Union[str, float, int]]:
        """
        Return a representation of the
        :class:`~ConfigSpace.configuration_space.Configuration` in dictionary
        form.

        Returns
        -------
        dict
            Configuration as dictionary

        """
        self._populate_values()
        return self._values

    def get_array(self) -> np.ndarray:
        """
        Return the internal vector representation of the
        :class:`~ConfigSpace.configuration_space.Configuration`. All continuous
        values are scaled between zero and one.

        Returns
        -------
        numpy.ndarray
            The vector representation of the configuration
        """
        return self._vector
