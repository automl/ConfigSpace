#!/usr/bin/env python

##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

from collections import deque, OrderedDict
import StringIO

import HPOlibConfigSpace.nx
from HPOlibConfigSpace.hyperparameters import Hyperparameter, \
    InstantiatedHyperparameter, InactiveHyperparameter, Constant
from HPOlibConfigSpace.conditions import ConditionComponent, \
    AbstractCondition, AbstractConjunction
from HPOlibConfigSpace.forbidden import AbstractForbiddenComponent


class ConfigurationSpace(object):
    # TODO add comments to both the configuration space and single
    # hyperparameters!

    # TODO add a method to add whole configuration spaces as a child "tree"

    """Represent a configuration space.
    """
    def __init__(self):
        self._dg = HPOlibConfigSpace.nx.DiGraph()
        self._dg.add_node('__HPOlib_configuration_space_root__')
        self.forbidden_clauses = []

    def add_hyperparameter(self, hyperparameter):
        """Add a hyperparameter to the configuration space.

        Parameters
        ----------
        hyperparameter : :class:`HPOlibConfigSpace.hyperparameters.Hyperparameter`
            The hyperparameter to add.
        """
        if not isinstance(hyperparameter, Hyperparameter):
            raise TypeError("The method add_hyperparameter must be called "
                            "with an instance of "
                            "HPOlibConfigSpace.hyperparameters.Hyperparameter.")

        # Check if adding the hyperparameter is legal:
        # * Its name must not already exist
        if hyperparameter.name in self._dg.node:
            raise ValueError("Hyperparameter '%s' is already in the "
                             "configuration space." % hyperparameter.name)

        self._dg.add_node(hyperparameter.name, hyperparameter=hyperparameter)
        self._dg.add_edge('__HPOlib_configuration_space_root__',
                          hyperparameter.name)

        self._check_default_configuration()

    def add_condition(self, condition):
        # Check if adding the condition is legal:
        # * The parent in a condition statement must exist
        # * The condition must add no cycles
        # The following array keeps track of all edges which must be
        # added to the DiGraph; if the checks don't raise any Exception,
        # these edges are finally added at the end of the function
        edges_to_add = []

        if not isinstance(condition, ConditionComponent):
            raise TypeError("The method add_condition must be called "
                            "with an instance of "
                            "HPOlibConfigSpace.condition.ConditionComponent.")

        if isinstance(condition, AbstractCondition):
            self._add_edge(condition.parent.name, condition.child.name,
                           condition)

        # Loop over the Conjunctions to find out the conditions we must add!
        elif isinstance(condition, AbstractConjunction):
            # The variable name child is misleading; this is a child in the
            # nested condition. Actually, these are the literal conditions!
            conditions_to_add = []
            for dlc in condition.get_descendant_literal_conditions():
                conditions_to_add.append(dlc)
                self._check_edge(dlc.parent.name,
                                 dlc.child.name,
                                 condition=condition)
                self._add_edge(dlc.parent.name,
                               dlc.child.name,
                               condition=condition)

        else:
            raise Exception("This should never happen!")

    def _add_edge(self, parent_node, child_node, condition):
        self._check_edge(parent_node, child_node, condition)
        try:
            # TODO maybe this has to be done more carefully
            self._dg.remove_edge('__HPOlib_configuration_space_root__',
                                 child_node)
        except:
            pass

        self._dg.add_edge(parent_node, child_node,
                          condition=condition)

    def _check_edge(self, parent_node, child_node, condition):
        # check if both nodes are already inserted into the graph
        if child_node not in self._dg.nodes():
            raise ValueError("Child hyperparameter '%s' not in configuration "
                             "space." % (child_node))
        if parent_node not in self._dg.nodes():
            raise ValueError("Parent hyperparameter '%s' not in configuration "
                             "space." % parent_node)

        # TODO: recursively check everything which is inside the conditions,
        # this means we have to recursively traverse the condition

        tmp_dag = self._dg.copy()
        tmp_dag.add_edge(parent_node, child_node, condition=condition)
        if not HPOlibConfigSpace.nx.is_directed_acyclic_graph(tmp_dag):
            cycles = list(HPOlibConfigSpace.nx.simple_cycles(tmp_dag))
            raise ValueError("Hyperparameter configuration contains a "
                             "cycle %s" % str(cycles))

        for other_condition in self.get_parents_of(child_node):
            if other_condition != condition:
                raise ValueError("Adding a second condition (different) for a "
                                 "hyperparameter is ambigouos and "
                                 "therefore forbidden. Add a conjunction "
                                 "instead!\nAlready inserted: %s\nNew one: "
                                 "%s" % (str(other_condition), str(condition)))

    def add_forbidden_clause(self, clause):
        if not isinstance(clause, AbstractForbiddenComponent):
            raise TypeError("The method add_forbidden_clause must be called "
                            "with an instance of "
                            "HPOlibConfigSpace.forbidden.AbstractForbiddenComponent.")
        self.forbidden_clauses.append(clause)
        self._check_default_configuration()

    def print_configuration_space(self):
        HPOlibConfigSpace.nx.write_dot(self._dg, "hyperparameters.dot")
        import matplotlib.pyplot as plt
        plt.title("draw_networkx")
        pos = HPOlibConfigSpace.nx.graphviz_layout(DG, prog='dot')
        HPOlibConfigSpace.nx.draw(self._dg, pos, with_labels=True)
        plt.savefig('nx_test.png')

    def get_hyperparameters(self, order='dfs_preorder'):
        sorted_dag = self._dg.copy()

        # The children of a node are traversed in a random order. Therefore,
        # copy the graph, sort the children and then traverse it
        for adj in sorted_dag.adj:
            sorted_dag.adj[adj] = OrderedDict(
                sorted(sorted_dag.adj[adj].items(), key=lambda item: item[0]))
        sorted_dag.node = OrderedDict(
            sorted(sorted_dag.node.items(), key=lambda item: item[0]))

        # If needed, preorder can be exchanged with postorder
        if order == 'dfs_preorder':
            nodes = HPOlibConfigSpace.nx.algorithms.traversal.\
                dfs_preorder_nodes(sorted_dag,
                source='__HPOlib_configuration_space_root__')
        elif order == 'dfs_postorder':
            nodes = HPOlibConfigSpace.nx.algorithms.traversal\
                .dfs_postorder_nodes(sorted_dag,
                source='__HPOlib_configuration_space_root__')
        elif order == 'topological':
            nodes = HPOlibConfigSpace.nx.algorithms.topological_sort(sorted_dag)
        else:
            raise NotImplementedError()

        nodes = [self._dg.node[node]['hyperparameter'] for node in nodes if
                 node != '__HPOlib_configuration_space_root__']

        return nodes

    def get_hyperparameter(self, name):
        for node in self._dg.node:
            if node != '__HPOlib_configuration_space_root__' and node == name:
                return self._dg.node[node]['hyperparameter']

        raise KeyError("Hyperparameter '%s' does not exist in this "
                       "configuration space." % name)

    def get_conditions(self):
        edges = []
        added_conditions = set()

        # Nodes is a list of nodes
        for source_node in self.get_hyperparameters():
            # This is a list of keys in a dictionary
            # TODO sort the edges by the order of their source_node in the
            # hyperparameter list!
            for target_node_name in self._dg.edge[source_node.name]:
                condition_ = self._dg[source_node.name][target_node_name][
                    'condition']

                if condition_ not in added_conditions:
                    edges.append(condition_)
                    added_conditions.add(condition_)

        return edges

    def get_children_of(self, name):
        if isinstance(name, Hyperparameter):
            name = name.name

        # This raises an exception if the hyperparameter does not exist
        self.get_hyperparameter(name)

        edges = []
        for target_node_name in self._dg.edge[name]:
            edges.append(self._dg[name][target_node_name][
                'condition'])
        return edges

    def get_parents_of(self, name):
        if isinstance(name, Hyperparameter):
            name = name.name

        # This raises an exception if the hyperparameter does not exist
        self.get_hyperparameter(name)

        edges = []
        # Nodes is a list of nodes
        for source_node in self.get_hyperparameters():
            # This is a list of keys in a dictionary
            for target_node_name in self._dg.edge[source_node.name]:
                if target_node_name == name:
                    edges.append(self._dg[source_node.name][target_node_name][
                        'condition'])

        return edges

    def get_all_uncoditional_hyperparameters(self):
        hyperparameters = []
        for target_node_name in self._dg.edge[
                '__HPOlib_configuration_space_root__']:
            hyperparameters.append(self._dg.node[target_node_name]['hyperparameter'])
        return hyperparameters

    def get_default_configuration(self):
        return self._check_default_configuration()

    def _check_default_configuration(self):
        # Check if adding that hyperparameter leads to an illegal default
        # configuration:
        instantiated_hyperparameters = {}
        for hp in self.get_hyperparameters(order='topological'):
            conditions = self.get_parents_of(hp.name)
            active = True
            for condition in conditions:
                parent_names = [c.parent.name for c in
                                condition.get_descendant_literal_conditions()]

                parents = [instantiated_hyperparameters[parent_name] for
                           parent_name in parent_names]

                if len(parents) == 1:
                    parents = parents[0]
                if not condition.evaluate(parents):
                    # TODO find out why a configuration is illegal!
                    active = False

            if active == False:
                if isinstance(hp, Constant):
                    instantiated_hyperparameters[hp.name] = \
                        InactiveHyperparameter(hp.value, hp)
                else:
                    instantiated_hyperparameters[hp.name] = \
                        InactiveHyperparameter(hp.default, hp)
            elif isinstance(hp, Constant):
                instantiated_hyperparameters[hp.name] = hp.instantiate(hp.value)
            else:
                instantiated_hyperparameters[hp.name] = hp.instantiate(hp.default)

            # TODO copy paste from check configuration


        # TODO get an extra Exception type for the case that the default
        # configuration is forbidden!
        return Configuration(self, **instantiated_hyperparameters)

    def check_configuration(self, configuration):
        # TODO: This should be a method of configuration, as it already knows
        #  the configuration space!

        if not isinstance(configuration, Configuration):
            raise TypeError("The method check_configuration must be called "
                            "with an instance of %s." % Configuration)

        hyperparameters = self.get_hyperparameters(order="topological")
        for hyperparameter in hyperparameters:
            ihp = configuration[hyperparameter.name]

            # TODO test that a hyperparameter is missing in a configuration
            if ihp is not None and not isinstance(ihp, InactiveHyperparameter) \
                    and not ihp.is_legal():
                raise ValueError("Hyperparameter instantiation '%s' is "
                                 "illegal" % ihp)

            conditions = self.get_parents_of(hyperparameter.name)

            # TODO this conditions should all be equal, are they actually?
            active = True
            for condition in conditions:
                parent_names = [c.parent.name for c in
                                condition.get_descendant_literal_conditions()]

                parents = [configuration[parent_name] for
                           parent_name in parent_names]

                # if one of the parents is None, the hyperparameter cannot be
                # active! Else we have to check this
                if any([parent is None for parent in parents]):
                    active = False

                else:
                    if len(parents) == 1:
                        parents = parents[0]
                    if not condition.evaluate(parents):
                        # TODO find out why a configuration is illegal!
                        active = False

            if active and not isinstance(ihp, InstantiatedHyperparameter):
                # TODO test this!
                raise ValueError("Hyperparameter '%s' not specified!" %
                                 hyperparameter.name)

            if not active and ihp is not None and \
                    not isinstance(ihp, InactiveHyperparameter):
                raise ValueError("Inactive hyperparameter '%s' must not be "
                                 "specified, but is: '%s'." %
                                 (ihp.hyperparameter.name, ihp))

        # Check if all forbidden clauses are satisfied
        # TODO check if AutoSklearn default would be a legal!
        for clause in self.forbidden_clauses:
            if clause.is_forbidden(configuration, strict=False):
                raise ValueError("%sviolates forbidden clause %s" % (
                    str(configuration), str(clause)))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self._dg.graph == other._dg.graph and \
                self._dg.node == other._dg.node and \
                self._dg.adj == other._dg.adj and \
                self._dg.pred == other._dg.pred and \
                self._dg.succ == other._dg.succ and \
                self._dg.edge == other._dg.edge

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        retval = StringIO.StringIO()
        retval.write("Configuration space object:\n  Hyperparameters:\n")

        hyperparameters = self.get_hyperparameters()
        if hyperparameters:
            retval.write("    ")
            retval.write("\n    ".join(
                [str(hyperparameter) for hyperparameter in hyperparameters]))
            retval.write("\n")

        conditions = self.get_conditions()
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


class Configuration(object):
    # TODO add a method to eliminate inactive hyperparameters from a
    # configuration
    # TODO maybe this syntax is not the best idea, because it forbids
    # hyperparameters like self and configuration_space
    def __init__(self, configuration_space, hyperparameters=None, **kwargs):
        if not isinstance(configuration_space, ConfigurationSpace):
            raise TypeError("Configuration expects an instance of %s, "
                            "you provided '%s'" %
                            (ConfigurationSpace, type(configuration_space)))

        values = dict()
        if hyperparameters is None:
            hyperparameters = kwargs

        for key in hyperparameters:
            hyperparameter = configuration_space.get_hyperparameter(key)
            value = hyperparameters[key]
            if isinstance(value, InstantiatedHyperparameter):
                instance = value
            else:
                instance = hyperparameter.instantiate(value)
            values[key] = instance

        self.values = values
        self.configuration_space = configuration_space
        self.is_valid_configuration()

    def is_valid_configuration(self):
        self.configuration_space.check_configuration(self)

    def __getitem__(self, item):
        return self.values.get(item)

    def __contains__(self, item):
        return item in self.values

    def __repr__(self):
        repr = StringIO.StringIO()
        repr.write("Configuration:\n")

        hyperparameters = self.configuration_space.get_hyperparameters(
            order='topological')
        for hyperparameter in hyperparameters:
            if hyperparameter.name in self.values:
                repr.write("  ")
                repr.write(self.values[hyperparameter.name])
                repr.write("\n")

        return repr.getvalue()

    def __iter__(self):
        return iter(self.values.values())


