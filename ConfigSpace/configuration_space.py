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

from collections import defaultdict, deque, OrderedDict
import copy

import numpy as np
import io

import ConfigSpace.nx
from ConfigSpace.hyperparameters import Hyperparameter, Constant, FloatHyperparameter
from ConfigSpace.conditions import ConditionComponent, \
    AbstractCondition, AbstractConjunction, EqualsCondition
from ConfigSpace.forbidden import AbstractForbiddenComponent
from typing import Union, List, Any, Dict, Iterable, Set, Tuple
from ConfigSpace.exceptions import ForbiddenValueError


class ConfigurationSpace(object):
    # TODO add comments to both the configuration space and single
    # hyperparameters!

    # TODO add a method to add whole configuration spaces as a child "tree"

    """Represent a configuration space.
    """

    def __init__(self, seed: Union[int, None] = None) -> None:
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
        for i, (l, u) in enumerate(bounds):
            hp = ConfigSpace.UniformFloatHyperparameter('x%d' % i, l, u)
            self.add_hyperparameter(hp)

    def add_hyperparameters(self, hyperparameters: List[Hyperparameter]) -> List[Hyperparameter]:
        """Add hyperparameters to the configuration space.

        Parameters
        ----------
        hyperparameters : list
            List of hyperparameters to add.

        Returns
        -------
        hyperparameters : list
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
        """Add a hyperparameter to the configuration space.

        Parameters
        ----------
        hyperparameter : :class:`HPOlibConfigSpace.hyperparameters.
                Hyperparameter`
            The hyperparameter to add.
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
        # Check if adding the condition is legal:
        # * The parent in a condition statement must exist
        # * The condition must add no cycles
        # The following array keeps track of all edges which must be
        # added to the DiGraph; if the checks don't raise any Exception,
        # these edges are finally added at the end of the function
        if not isinstance(condition, ConditionComponent):
            raise TypeError("The method add_condition must be called "
                            "with an instance of "
                            "ConfigSpace.condition.ConditionComponent.")

        if isinstance(condition, AbstractCondition):
            self._check_edges([(condition.parent.name, condition.child.name)])
            self._check_condition(condition.child.name, condition)
            self._add_edge(condition.parent.name, condition.child.name,
                           condition)

        # Loop over the Conjunctions to find out the conditions we must add!
        elif isinstance(condition, AbstractConjunction):
            dlcs = condition.get_descendant_literal_conditions()
            edges = [(dlc.parent.name, dlc.child.name) for dlc in dlcs]
            self._check_edges(edges)

            for dlc in condition.get_descendant_literal_conditions():
                self._check_condition(dlc.child.name, condition)
                self._add_edge(dlc.parent.name,
                               dlc.child.name,
                               condition=condition)

        else:
            raise Exception("This should never happen!")

        self._sort_hyperparameters()
        self._update_cache()
        return condition

    def add_conditions(self, conditions: List[ConditionComponent]) -> List[ConditionComponent]:
        for condition in conditions:
            if not isinstance(condition, ConditionComponent):
                raise TypeError("Condition '%s' is not an instance of "
                                "ConfigSpace.condition.ConditionComponent." %
                                str(condition))

        edges = []
        conditions_to_add = []
        for condition in conditions:
            if isinstance(condition, AbstractCondition):
                edges.append((condition.parent.name, condition.child.name))
                conditions_to_add.append(condition)
            elif isinstance(condition, AbstractConjunction):
                dlcs = condition.get_descendant_literal_conditions()
                edges = [(dlc.parent.name, dlc.child.name) for dlc in dlcs]
                conditions_to_add.extend(dlcs)

        for edge, condition in zip(edges, conditions_to_add):
            self._check_condition(edge[1], condition)
        self._check_edges(edges)
        for edge, condition in zip(edges, conditions_to_add):
            self._add_edge(edge[0], edge[1], condition)

        self._sort_hyperparameters()
        self._update_cache()
        return conditions

    def _add_edge(self, parent_node: str, child_node: str, condition: ConditionComponent) -> None:
        try:
            # TODO maybe this has to be done more carefully
            del self._children['__HPOlib_configuration_space_root__'][
                child_node]
        except Exception:
            pass

        try:
            del self._parents[child_node]['__HPOlib_configuration_space_root__']
        except Exception:
            pass

        self._children[parent_node][child_node] = condition
        self._parents[child_node][parent_node] = condition
        self._conditionals.add(child_node)

    def _check_condition(self, child_node: str, condition: ConditionComponent) -> None:
        for other_condition in self._get_parent_conditions_of(child_node):
            if other_condition != condition:
                raise ValueError("Adding a second condition (different) for a "
                                 "hyperparameter is ambigouos and "
                                 "therefore forbidden. Add a conjunction "
                                 "instead!\nAlready inserted: %s\nNew one: "
                                 "%s" % (str(other_condition), str(condition)))

    def _check_edges(self, edges: List[Tuple[str, str]]) -> None:
        for parent_node, child_node in edges:
            # check if both nodes are already inserted into the graph
            if child_node not in self._hyperparameters:
                raise ValueError("Child hyperparameter '%s' not in configuration "
                                 "space." % child_node)
            if parent_node not in self._hyperparameters:
                raise ValueError("Parent hyperparameter '%s' not in configuration "
                                 "space." % parent_node)

        # TODO: recursively check everything which is inside the conditions,
        # this means we have to recursively traverse the condition

        tmp_dag = self._create_tmp_dag()
        for parent_node, child_node in edges:
            tmp_dag.add_edge(parent_node, child_node)

        if not ConfigSpace.nx.is_directed_acyclic_graph(tmp_dag):
            cycles = list(ConfigSpace.nx.simple_cycles(tmp_dag))  # type: List[List[str]]
            for cycle in cycles:
                cycle.sort()
            cycles.sort()
            raise ValueError("Hyperparameter configuration contains a "
                             "cycle %s" % str(cycles))

    def _sort_hyperparameters(self) -> None:
        levels = OrderedDict()  # type: OrderedDict[str, int]
        to_visit = deque() # type: ignore
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

        # update conditions
        for condition in self.get_conditions():
            condition.set_vector_idx(self._hyperparameter_idx)

        # forbidden clauses
        for clause in self.forbidden_clauses:
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
        if not isinstance(clause, AbstractForbiddenComponent):
            raise TypeError("The method add_forbidden_clause must be called "
                            "with an instance of "
                            "ConfigSpace.forbidden.AbstractForbiddenComponent.")
        clause.set_vector_idx(self._hyperparameter_idx)
        self.forbidden_clauses.append(clause)
        self._check_default_configuration()
        return clause

    def add_forbidden_clauses(self, clauses: List[AbstractForbiddenComponent]) -> List[AbstractForbiddenComponent]:
        for clause in clauses:
            clause.set_vector_idx(self._hyperparameter_idx)
            if not isinstance(clause, AbstractForbiddenComponent):
                raise TypeError("Forbidden '%s' is not an instance of "
                                "ConfigSpace.forbidden.AbstractForbiddenComponent." %
                                str(clause))
            self.forbidden_clauses.append(clause)
        self._check_default_configuration()
        return clauses

    def add_configuration_space(self, prefix: str, configuration_space: 'ConfigurationSpace',
                                delimiter: str=":", parent_hyperparameter: Hyperparameter=None) -> 'ConfigurationSpace':
        if not isinstance(configuration_space, ConfigurationSpace):
            raise TypeError("The method add_configuration_space must be "
                            "called with an instance of "
                            "HPOlibConfigSpace.configuration_space."
                            "ConfigurationSpace.")

        new_parameters = []
        for hp in configuration_space.get_hyperparameters():
            new_parameter = copy.deepcopy(hp)
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
            new_condition = copy.deepcopy(condition)
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
            new_forbidden = copy.deepcopy(forbidden_clause)
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
        return list(self._hyperparameters.values())

    def get_hyperparameter(self, name: str) -> Hyperparameter:
        hp = self._hyperparameters.get(name)

        if hp is None:
            raise KeyError("Hyperparameter '%s' does not exist in this "
                           "configuration space." % name)
        else:
            return hp

    def get_hyperparameter_by_idx(self, idx: int) -> str:
        hp = self._idx_to_hyperparameter.get(idx)

        if hp is None:
            raise KeyError("Hyperparameter #'%d' does not exist in this "
                           "configuration space." % idx)
        else:
            return hp

    def get_idx_by_hyperparameter_name(self, name: str) -> int:
        idx = self._hyperparameter_idx.get(name)

        if idx is None:
            raise KeyError("Hyperparameter '%s' does not exist in this "
                           "configuration space." % name)
        else:
            return idx

    def get_conditions(self) -> List[AbstractCondition]:
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

    def get_children_of(self, name: Union[str, Hyperparameter]) -> List[Hyperparameter]:
        conditions = self.get_child_conditions_of(name)
        parents = [] # type: List[Hyperparameter]
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
        """Return the parent hyperparameters of a given hyperparameter.

        Parameters
        ----------
        name : str or Hyperparameter
            Can either be the name of a hyperparameter or the hyperparameter
            object.

        Returns
        -------
        list
            List with all parent hyperparameters.
        """
        conditions = self.get_parent_conditions_of(name)
        parents = [] # type: List[Hyperparameter]
        for condition in conditions:
            parents.extend(condition.get_parents())
        return parents

    def _get_parents_of(self, name: str) -> List[Hyperparameter]:
        """Return the parent hyperparameters of a given hyperparameter.

        Parameters
        ----------
        name : str

        Returns
        -------
        list
            List with all parent hyperparameters.
        """
        conditions = self._get_parent_conditions_of(name)
        parents = []  # type: List[Hyperparameter]
        for condition in conditions:
            parents.extend(condition.get_parents())
        return parents

    def get_parent_conditions_of(self, name: Union[str, Hyperparameter]) -> List[AbstractCondition]:
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
        hyperparameters = [hp_name for hp_name in
                           self._children[
                               '__HPOlib_configuration_space_root__']]
        return hyperparameters

    def get_all_conditional_hyperparameters(self) -> List[str]:
        return self._conditionals

    def get_default_configuration(self) -> 'Configuration':
        return self._check_default_configuration()

    def _check_default_configuration(self) -> 'Configuration':
        # Check if adding that hyperparameter leads to an illegal default configuration
        instantiated_hyperparameters = {} # type: Dict[str, Union[None, int, float, str]]
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

            if active == False:
                instantiated_hyperparameters[hp.name] = None
            elif isinstance(hp, Constant):
                instantiated_hyperparameters[hp.name] = hp.value
            else:
                instantiated_hyperparameters[hp.name] = hp.default

                # TODO copy paste from check configuration

        # TODO add an extra Exception type for the case that the default
        # configuration is forbidden!
        return Configuration(self, instantiated_hyperparameters)

    # For backward compatibility
    def check_configuration(self, configuration: 'Configuration') -> None:
        if not isinstance(configuration, Configuration):
            raise TypeError("The method check_configuration must be called "
                            "with an instance of %s. "
                            "Your input was of type %s"% (Configuration, type(configuration)))
        self._check_configuration(configuration.get_array())

    def check_configuration_vector_representation(self, vector: np.ndarray) -> None:
        if not isinstance(vector, np.ndarray):
            raise TypeError("The method check_configuration must be called "
                            "with an instance of np.ndarray "
                            "Your input was of type %s" % (type(vector)))
        self._check_configuration(vector)

    def _check_configuration(self, vector: np.ndarray,
                             allow_inactive_with_values: bool = False) -> None:
        unconditional_hyperparameters = self.get_all_unconditional_hyperparameters()
        to_visit = deque()
        to_visit.extendleft(unconditional_hyperparameters)
        active = np.zeros((len(vector),), dtype=bool)
        inactive = set()

        for ch in unconditional_hyperparameters:
            active[self._hyperparameter_idx[ch]] = 1

        while len(to_visit) > 0:
            hp_name = to_visit.pop()
            hp_idx = self._hyperparameter_idx[hp_name]
            hyperparameter = self._hyperparameters[hp_name]
            hp_value = vector[hp_idx]

            if not np.isnan(hp_value) and not hyperparameter.is_legal_vector(hp_value):
                raise ValueError("Hyperparameter instantiation '%s' "
                                 "(type: %s) is illegal for hyperparameter %s" %
                                 (hp_value, str(type(hp_value)),
                                  hyperparameter))

            children = self._children_of[hp_name]
            for child in children:
                if child.name not in inactive:
                    parents = self._parents_of[child.name]
                    if len(parents) == 1:
                        conditions = self._parent_conditions_of[child.name]
                        add = True
                        for condition in conditions:
                            if not condition.evaluate_vector(vector):
                                add = False
                                inactive.add(child.name)
                                break
                        if add:
                            hyperparameter_idx = self._hyperparameter_idx[
                                child.name]
                            active[hyperparameter_idx] = 1
                            to_visit.appendleft(child.name)

                    else:
                        parent_names = set(p.name for p in parents)
                        if not parent_names <= set(to_visit):  # make sure no parents are still unvisited
                            conditions = self._parent_conditions_of[child.name]
                            add = True
                            for condition in conditions:
                                if not condition.evaluate_vector(vector):
                                    add = False
                                    inactive.add(child.name)
                                    break

                            if add:
                                hyperparameter_idx = self._hyperparameter_idx[
                                    child.name]
                                active[hyperparameter_idx] = 1
                                to_visit.appendleft(child.name)

                        else:
                            continue

            if active[hp_idx] and np.isnan(hp_value):
                raise ValueError("Active hyperparameter '%s' not specified!" %
                                 hyperparameter.name)

        for hp_idx in self._idx_to_hyperparameter:

            if not allow_inactive_with_values and not active[hp_idx] and \
                    not np.isnan(vector[hp_idx]):
                    # Only look up the value (in the line above) if the
                    # hyperparameter is inactive!
                hp_name = self._idx_to_hyperparameter[hp_idx]
                hp_value = vector[hp_idx]
                raise ValueError("Inactive hyperparameter '%s' must not be "
                                 "specified, but has the vector value: '%s'." %
                                 (hp_name, hp_value))
        self._check_forbidden(vector)

    def _check_configuration_rigorous(self, configuration: 'Configuration',
                                      allow_inactive_with_values: bool = False) -> None:
        vector = configuration.get_array()

        for hp_name, hyperparameter in self._hyperparameters.items():
            hp_value = vector[self._hyperparameter_idx[hp_name]]

            if not np.isnan(hp_value) and not hyperparameter.is_legal_vector(hp_value):
                raise ValueError("Hyperparameter instantiation '%s' "
                                 "(type: %s) is illegal for hyperparameter %s" %
                                 (hp_value, str(type(hp_value)),
                                  hyperparameter))

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
        for clause in self.forbidden_clauses:
            if clause.is_forbidden_vector(vector, strict=False):
                raise ForbiddenValueError("Given vector violates forbidden clause %s" % (str(clause)))

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

        if self.forbidden_clauses:
            retval.write("  Forbidden Clauses:\n")
            retval.write("    ")
            retval.write("\n    ".join(
                [str(clause) for clause in self.forbidden_clauses]))
            retval.write("\n")

        retval.seek(0)
        return retval.getvalue()

    def __iter__(self) -> Iterable:
        """ Allows to iterate over the hyperparameter names in (hopefully?) the right order."""
        return iter(self._hyperparameters.keys())

    def sample_configuration(self, size: int = 1) -> Union['Configuration', List['Configuration']]:
        if not isinstance(size, int):
            raise TypeError('Argument size must be of type int, but is %s'
                            % type(size))

        iteration = 0
        missing = size
        accepted_configurations = []  # type: List['Configuration']
        num_hyperparameters = len(self._hyperparameters)

        unconditional_hyperparameters = self.get_all_unconditional_hyperparameters()
        hyperparameters_with_children = list()

        forbidden_clauses_unconditionals = []
        forbidden_clauses_conditionals = []
        for clause in self.forbidden_clauses:
            based_on_conditionals = False
            for subclause in clause.get_descendant_literal_clauses():
                if subclause.hyperparameter.name not in unconditional_hyperparameters:
                    based_on_conditionals = True
                    break
            if based_on_conditionals:
                forbidden_clauses_conditionals.append(clause)
            else:
                forbidden_clauses_unconditionals.append(clause)

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
                    for clause in forbidden_clauses_unconditionals:
                        if clause.is_forbidden_vector(vector[i], strict=False):
                            raise ForbiddenValueError(
                                "Given vector violates forbidden clause %s" % (
                                str(clause)))

                    hps = deque()
                    visited = set()
                    hps.extendleft(hyperparameters_with_children)
                    active = np.zeros((num_hyperparameters,), dtype=bool)

                    for ch in unconditional_hyperparameters:
                        active[self._hyperparameter_idx[ch]] = 1

                    inactive = set()

                    while len(hps) > 0:
                        hp = hps.pop()
                        visited.add(hp)
                        children = self._children_of[hp]
                        for child in children:
                            child_name = child.name
                            if child_name not in inactive:
                                parents = self._parents_of[child_name]
                                hyperparameter_idx = self._hyperparameter_idx[child_name]
                                if len(parents) == 1:
                                    conditions = self._parent_conditions_of[child_name]
                                    add = True
                                    for condition in conditions:
                                        if not condition.evaluate_vector(vector[i]):
                                            add = False
                                            vector[i][hyperparameter_idx] = np.NaN
                                            inactive.add(child_name)
                                            break
                                    if add == True:
                                        active[hyperparameter_idx] = 1
                                        hps.appendleft(child_name)

                                else:
                                    parent_names = set(p.name for p in parents)
                                    if parent_names.issubset(visited):  # make sure no parents are still unvisited
                                        conditions = self._parent_conditions_of[child_name]
                                        add = True
                                        for condition in conditions:
                                            if not condition.evaluate_vector(vector[i]):
                                                add = False
                                                vector[i][hyperparameter_idx] = np.NaN
                                                inactive.add(child_name)
                                                break

                                        if add == True:
                                            active[hyperparameter_idx] = 1
                                            hps.appendleft(child_name)

                                    else:
                                        continue

                    vector[i][~active] = np.NaN

                    for clause in forbidden_clauses_conditionals:
                        if clause.is_forbidden_vector(vector[i], strict=False):
                            raise ForbiddenValueError(
                                "Given vector violates forbidden clause %s" % (
                                    str(clause)))

                    configuration = Configuration(self, vector=vector[i])
                    accepted_configurations.append(configuration)
                except ForbiddenValueError as e:
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
        self.random = np.random.RandomState(seed)


class Configuration(object):
    # TODO add a method to eliminate inactive hyperparameters from a configuration
    def __init__(self, configuration_space: ConfigurationSpace, values: Union[None,  Dict[str, Union[str, float, int]]] = None,
                 vector: Union[None, np.ndarray]=None, allow_inactive_with_values: bool=False, origin: Any=None)\
            -> None:
        """A single configuration.

        Parameters
        ----------
        configuration_space : ConfigurationSpace
            The configuration space for this configuration

        values : dict
            A dictionary with pairs (hyperparameter_name, value), where value is
            a legal value of the hyperparameter in the above
            configuration_space.

        vector : np.ndarray
            A numpy array for efficient representation. Either values or
            vector has to be given.

        allow_inactive_with_values : bool (default=False)
            Whether an Exception will be raised if a value for an inactive
            hyperparameter is given. Default is to raise an Exception.
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
                hyperparameter.is_legal(value)
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
            self._vector = vector
        else:
            raise ValueError('Configuration neither specified as dictionary '
                             'or vector.')

    def is_valid_configuration(self) -> None:
        self.configuration_space._check_configuration(
            self._vector, allow_inactive_with_values=self.allow_inactive_with_values)

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

    def get(self, item: str, default: Union[None, Any]=None) -> Union[None, Any]:
        try:
            return self[item]
        except:
            return default

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
        # Cache the keys to speed up the process of retrieving the keys
        if self._keys is None:
            self._keys = list(self.configuration_space._hyperparameters.keys())
        return self._keys

    def get_dictionary(self) -> Dict[str, Union[str, float, int]]:
        self._populate_values()
        return self._values

    def get_array(self) -> np.ndarray:
        """
        Returns
        -------
        numpy.ndarray
            internal vector representation of the configuration. All
            continuous values are scaled between zero and one.
        """
        return self._vector




















