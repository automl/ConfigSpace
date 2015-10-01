#!/usr/bin/env python

# #
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

from collections import defaultdict, deque, OrderedDict
import copy
from itertools import product

import numpy as np
import six

import HPOlibConfigSpace.nx
from HPOlibConfigSpace.hyperparameters import Hyperparameter, Constant, \
    CategoricalHyperparameter
from HPOlibConfigSpace.conditions import ConditionComponent, \
    AbstractCondition, AbstractConjunction
from HPOlibConfigSpace.forbidden import AbstractForbiddenComponent


class ConfigurationSpace(object):
    # TODO add comments to both the configuration space and single
    # hyperparameters!

    # TODO add a method to add whole configuration spaces as a child "tree"

    """Represent a configuration space.
    """

    def __init__(self, seed=1):
        self._hyperparameters = OrderedDict()
        self._children = defaultdict(dict)
        self._parents = defaultdict(dict)
        # changing this to a normal dict will break sampling because there is
        #  no guarantee that the parent of a condition was evaluated before
        self._conditionsals = OrderedDict()
        self.forbidden_clauses = []
        self.random = np.random.RandomState(seed)
        self._vector_types = None

    def add_hyperparameter(self, hyperparameter):
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
                            "HPOlibConfigSpace.hyperparameters.Hyperparameter.")

        # Check if adding the hyperparameter is legal:
        # * Its name must not already exist
        if hyperparameter.name in self._hyperparameters:
            raise ValueError("Hyperparameter '%s' is already in the "
                             "configuration space." % hyperparameter.name)

        self._hyperparameters[hyperparameter.name] = hyperparameter
        self._children['__HPOlib_configuration_space_root__'][
            hyperparameter.name] = None
        self._parents[hyperparameter.name][
            '__HPOlib_configuration_space_root__'] = None

        self._check_default_configuration()
        self._sort_hyperparameters()
        # Update the vector
        types = [(hp.name,
                  int if isinstance(hp, (CategoricalHyperparameter, Constant))
                  else float)
                 for hp in self._hyperparameters.values()]
        self._vector_types = types
        return hyperparameter

    def add_condition(self, condition):
        # Check if adding the condition is legal:
        # * The parent in a condition statement must exist
        # * The condition must add no cycles
        # The following array keeps track of all edges which must be
        # added to the DiGraph; if the checks don't raise any Exception,
        # these edges are finally added at the end of the function
        if not isinstance(condition, ConditionComponent):
            raise TypeError("The method add_condition must be called "
                            "with an instance of "
                            "HPOlibConfigSpace.condition.ConditionComponent.")

        if isinstance(condition, AbstractCondition):
            self._add_edge(condition.parent.name, condition.child.name,
                           condition)

        # Loop over the Conjunctions to find out the conditions we must add!
        elif isinstance(condition, AbstractConjunction):
            for dlc in condition.get_descendant_literal_conditions():
                self._check_edge(dlc.parent.name,
                                 dlc.child.name,
                                 condition=condition)
                self._add_edge(dlc.parent.name,
                               dlc.child.name,
                               condition=condition)

        else:
            raise Exception("This should never happen!")
        return condition

    def _add_edge(self, parent_node, child_node, condition):
        self._check_edge(parent_node, child_node, condition)
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
        self._sort_hyperparameters()
        self._conditionsals[child_node] = child_node

    def _check_edge(self, parent_node, child_node, condition):
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
        tmp_dag.add_edge(parent_node, child_node)

        if not HPOlibConfigSpace.nx.is_directed_acyclic_graph(tmp_dag):
            cycles = list(HPOlibConfigSpace.nx.simple_cycles(tmp_dag))
            for cycle in cycles:
                cycle.sort()
            cycles.sort()
            raise ValueError("Hyperparameter configuration contains a "
                             "cycle %s" % str(cycles))

        for other_condition in self._get_parent_conditions_of(child_node):
            if other_condition != condition:
                raise ValueError("Adding a second condition (different) for a "
                                 "hyperparameter is ambigouos and "
                                 "therefore forbidden. Add a conjunction "
                                 "instead!\nAlready inserted: %s\nNew one: "
                                 "%s" % (str(other_condition), str(condition)))

    def _sort_hyperparameters(self):
        levels = dict()
        to_visit = deque()
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

        by_level = defaultdict(list)
        for hp in levels:
            level = levels[hp]
            by_level[level].append(hp)

        nodes = []
        # Sort and add to list
        for level in by_level:
            by_level[level].sort()
            nodes.extend(by_level[level])

        # Resort the OrderedDict
        for node in nodes:
            hp = self._hyperparameters[node]
            del self._hyperparameters[node]
            self._hyperparameters[node] = hp

            hp = self._conditionsals.get(node)
            if hp is not None:
                del self._conditionsals[node]
                self._conditionsals[node] = hp

    def _create_tmp_dag(self):
        tmp_dag = HPOlibConfigSpace.nx.DiGraph()
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

    def add_forbidden_clause(self, clause):
        if not isinstance(clause, AbstractForbiddenComponent):
            raise TypeError("The method add_forbidden_clause must be called "
                            "with an instance of "
                            "HPOlibConfigSpace.forbidden.AbstractForbiddenComponent.")
        self.forbidden_clauses.append(clause)
        self._check_default_configuration()
        return clause

    # def print_configuration_space(self):
    #     HPOlibConfigSpace.nx.write_dot(self._dg, "hyperparameters.dot")
    #     import matplotlib.pyplot as plt
    #     plt.title("draw_networkx")
    #     pos = HPOlibConfigSpace.nx.graphviz_layout(DG, prog='dot')
    #     HPOlibConfigSpace.nx.draw(self._dg, pos, with_labels=True)
    #     plt.savefig('nx_test.png')

    def add_configuration_space(self, prefix, configuration_space,
                                delimiter=":"):
        if not isinstance(configuration_space, ConfigurationSpace):
            raise TypeError("The method add_configuration_space must be "
                            "called with an instance of "
                            "HPOlibConfigSpace.configuration_space."
                            "ConfigurationSpace.")

        for hp in configuration_space.get_hyperparameters():
            new_parameter = copy.deepcopy(hp)
            new_parameter.name = "%s%s%s" % (prefix, delimiter,
                                             new_parameter.name)
            self.add_hyperparameter(new_parameter)

        for condition in configuration_space.get_conditions():
            dlcs = condition.get_descendant_literal_conditions()
            for dlc in dlcs:
                if not dlc.child.name.startswith("%s%s" % (prefix, delimiter)):
                    dlc.child.name = "%s%s%s" % (
                    prefix, delimiter, dlc.child.name)
                if not dlc.parent.name.startswith("%s%s" % (prefix, delimiter)):
                    dlc.parent.name = "%s%s%s" % (
                    prefix, delimiter, dlc.parent.name)
            self.add_condition(condition)

        for forbidden_clause in configuration_space.forbidden_clauses:
            dlcs = forbidden_clause.get_descendant_literal_clauses()
            for dlc in dlcs:
                if not dlc.hyperparameter.name.startswith(
                                "%s%s" % (prefix, delimiter)):
                    dlc.hyperparameter.name = "%s%s%s" % \
                                              (prefix, delimiter,
                                               dlc.hyperparameter.name)
            self.add_forbidden_clause(forbidden_clause)

        return configuration_space

    def get_hyperparameters(self):
        return list(self._hyperparameters.values())

    def get_hyperparameter(self, name):
        hp = self._hyperparameters.get(name)

        if hp is None:
            raise KeyError("Hyperparameter '%s' does not exist in this "
                           "configuration space." % name)
        else:
            return hp

    def get_conditions(self):
        conditions = []
        added_conditions = set()

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

    def get_children_of(self, name):
        conditions = self.get_child_conditions_of(name)
        parents = []
        for condition in conditions:
            parents.append(condition.child)
        return parents

    def get_child_conditions_of(self, name):
        if isinstance(name, Hyperparameter):
            name = name.name

        # This raises an exception if the hyperparameter does not exist
        self.get_hyperparameter(name)

        #conditions = []
        #for child_name in self._children[name]:
        #    if child_name == "__HPOlib_configuration_space_root__":
        #        continue
        #    condition = self._children[name][child_name]
        #    conditions.append(condition)

        children = self._children[name]
        conditions = [children[child_name] for child_name in children
                      if child_name != "__HPOlib_configuration_space_root__"]
        return conditions

    def get_parents_of(self, name):
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
        parents = []
        for condition in conditions:
            parents.append(condition.parent)
        return parents

    def get_parent_conditions_of(self, name):
        if isinstance(name, Hyperparameter):
            name = name.name

        # This raises an exception if the hyperparameter does not exist
        self.get_hyperparameter(name)
        return self._get_parent_conditions_of(name)

    def _get_parent_conditions_of(self, name):
        parents = self._parents[name]
        conditions = [parents[parent_name] for parent_name in parents
                      if parent_name != "__HPOlib_configuration_space_root__"]
        return conditions

    def get_all_uncoditional_hyperparameters(self):
        hyperparameters = [hp_name for hp_name in
                           self._children[
                               '__HPOlib_configuration_space_root__']]
        return hyperparameters

    def get_all_conditional_hyperparameters(self):
        return self._conditionsals

    def get_default_configuration(self):
        return self._check_default_configuration()

    def _check_default_configuration(self):
        # Check if adding that hyperparameter leads to an illegal default
        # configuration:
        instantiated_hyperparameters = {}
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

    def check_configuration(self, configuration):
        if not isinstance(configuration, Configuration):
            raise TypeError("The method check_configuration must be called "
                            "with an instance of %s." % Configuration)
        self._check_configuration(configuration)

    def _check_configuration(self, configuration):
        for hp_name in self._hyperparameters:
            hyperparameter = self._hyperparameters[hp_name]
            hp_value = configuration[hp_name]

            if hp_value is not None and not hyperparameter.is_legal(hp_value):
                raise ValueError("Hyperparameter instantiation '%s' is "
                                 "illegal for hyperparameter %s" %
                                 (hp_value, hyperparameter))

            conditions = self._get_parent_conditions_of(hyperparameter.name)

            # TODO this conditions should all be equal, are they actually?
            active = True
            for condition in conditions:
                parent_names = [c.parent.name for c in
                                condition.get_descendant_literal_conditions()]

                parents = {parent_name: configuration[parent_name] for
                           parent_name in parent_names}

                # if one of the parents is None, the hyperparameter cannot be
                # active! Else we have to check this
                if any([parent_value is None for parent_value in
                        parents.values()]):
                    active = False

                else:
                    #if len(parents) == 1:
                    #    parents = parents[0]
                    if not condition.evaluate(parents):
                        active = False

            if active and hp_value is None:
                raise ValueError("Active hyperparameter '%s' not specified!" %
                                 hyperparameter.name)

            if not active and hp_value is not None:
                raise ValueError("Inactive hyperparameter '%s' must not be "
                                 "specified, but has the value: '%s'." %
                                 (hp_name, hp_value))
        self._check_forbidden(configuration)

    def _check_forbidden(self, configuration):
        # Check if all forbidden clauses are satisfied
        #import traceback
        #for line in traceback.format_stack():
        #    print line.strip()
        #print
        for clause in self.forbidden_clauses:
            if clause.is_forbidden(configuration, strict=False):
                raise ValueError("%sviolates forbidden clause %s" % (
                    str(configuration), str(clause)))

    # http://stackoverflow.com/a/25176504/4636294
    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            this_dict = self.__dict__.copy()
            del this_dict['random']
            other_dict = other.__dict__.copy()
            del other_dict['random']
            return this_dict == other_dict
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self):
        retval = six.StringIO()
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

    def sample_configuration(self, size=1):
        iteration = 0
        missing = size
        accepted_configurations = []
        num_hyperparameters = len(self._hyperparameters)

        while len(accepted_configurations) < size:
            if missing != size:
                missing = int(1.1 * missing)
            vector = np.ndarray((missing,), dtype=self._vector_types)

            for hp_name in self._hyperparameters:
                hyperparameter = self._hyperparameters[hp_name]
                vector[hp_name] = hyperparameter._sample(self.random, missing)

            for i in range(missing):
                inactive = set()
                for hp_name in self.get_all_conditional_hyperparameters():
                    conditions = self._get_parent_conditions_of(hp_name)
                    add = True
                    for condition in conditions:
                        parent_names = [c.parent.name for c in
                                        condition.get_descendant_literal_conditions()]

                        parents = {parent_name: self._hyperparameters[
                                       parent_name]._transform(vector[i][
                                           parent_name])
                                   for parent_name in parent_names}

                        # A parent condition is not fulfilled
                        if np.sum([parent_name in inactive
                                   for parent_name in parent_names]) > 0:
                            add = False
                            break
                        if not condition.evaluate(parents):
                            add = False
                            break

                    if not add:
                        hyperparameter = self._hyperparameters[hp_name]
                        vector[hp_name][i] = hyperparameter._nan
                        inactive.add(hp_name)


            for i in range(missing):
                try:
                    configuration = Configuration(self, vector=vector[i])
                    self._check_forbidden(configuration)
                    accepted_configurations.append(configuration)
                except ValueError as e:
                    iteration += 1

                    if iteration == size * 100:
                        raise ValueError(
                            "Cannot sample valid configuration for "
                            "%s" % self)

            missing = size - len(accepted_configurations)

        if size <= 1:
            return accepted_configurations[0]
        else:
            return accepted_configurations

    def seed(self, seed):
        self.random = np.random.RandomState(seed)


class Configuration(object):
    # TODO add a method to eliminate inactive hyperparameters from a
    # configuration
    def __init__(self, configuration_space, values=None, vector=None):
        """A single configuration.

        Parameters
        ----------
        configuration_space : ConfigurationSpace
            The configuration space for this configuration

        values : dict
            A dictionary with pairs (hyperparameter_name, value), where value is
            a legal value of the hyperparameter in the above
            configuration_space.
        """
        if not isinstance(configuration_space, ConfigurationSpace):
            raise TypeError("Configuration expects an instance of %s, "
                            "you provided '%s'" %
                            (ConfigurationSpace, type(configuration_space)))

        self.configuration_space = configuration_space
        self._query_values = False

        if values is not None:
            # Using cs._hyperparameters to iterate makes sure that the
            # hyperparameters in the configuration are sorted in the same way as
            # they are sorted in the configuration space
            self._values = dict()
            for key in configuration_space._hyperparameters:
                value = values.get(key)
                if value is None:
                    continue
                hyperparameter = configuration_space.get_hyperparameter(key)
                hyperparameter.is_legal(value)
                self._values[key] = value
            self._query_values = True
            self.is_valid_configuration()
            self._vector = np.ndarray((1,),
                                      dtype=configuration_space._vector_types)

        elif vector is not None:
            self._values = dict()
            self._vector = vector
        else:
            raise ValueError()

    def is_valid_configuration(self):
        self.configuration_space._check_configuration(self)

    def __getitem__(self, item):
        if self._query_values or item in self._values:
            return self._values.get(item)

        hyperparameter = self.configuration_space._hyperparameters[item]
        self._values[item] = hyperparameter._transform(self._vector[item])
        return self._values[item]

    def __contains__(self, item):
        return item in self._values

    # http://stackoverflow.com/a/25176504/4636294
    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

    def _populate_values(self):
        for hyperparameter in self.configuration_space.get_hyperparameters():
            self[hyperparameter.name]

    def __repr__(self):
        if self._query_values is False:
            self._populate_values()

        repr = six.StringIO()
        repr.write("Configuration:\n")

        hyperparameters = self.configuration_space.get_hyperparameters()
        hyperparameters.sort(key=lambda t: t.name)
        for hyperparameter in hyperparameters:
            hp_name = hyperparameter.name
            if hp_name in self._values and self._values[hp_name] is not None:
                repr.write("  ")
                value = str(self._values[hp_name])
                if isinstance(hyperparameter, Constant):
                    repr.write("%s, Constant: %s" % (hp_name, value))
                else:
                    repr.write("%s, Value: %s" % (hp_name, value))
                repr.write("\n")

        return repr.getvalue()

    def __iter__(self):
        # TODO: the hyperparameter names should also be in the configuration
        # object!
        return iter(self.configuration_space._hyperparameters.keys())


