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

__authors__ = ["Katharina Eggensperger", "Matthias Feurer", "Mohsin Ali"]
__contact__ = "automl.org"

from collections import OrderedDict
from itertools import product
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    NumericalHyperparameter, Constant, IntegerHyperparameter, \
    NormalIntegerHyperparameter, NormalFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition, NotEqualsCondition, \
    InCondition, AndConjunction, OrConjunction, ConditionComponent
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction, ForbiddenInClause, AbstractForbiddenComponent, MultipleValueForbiddenClause
import sys
import pyparsing
import six

# Build pyparsing expressions for params
pp_param_name = pyparsing.Word(pyparsing.alphanums + "_" + "-" + "@" + "." + ":" + ";" + "\\" + "/" + "?" + "!" +
                               "$" + "%" + "&" + "*" + "+" + "<" + ">")

def build_categorical(param):
    cat_template = "%s '--%s ' c (%s)"
    return cat_template % (param.name, param.name,
                           ",".join([str(value) for value in param.choices]))


def build_constant(param):
    constant_template = "%s '--%s ' c (%s)"
    return constant_template % (param.name, param.name, param.value)


def build_continuous(param):
    if type(param) in (NormalIntegerHyperparameter,
                       NormalFloatHyperparameter):
        param = param.to_uniform()

    float_template = "%s '--%s ' r (%s, %s) "
    int_template = "%s '--%s ' i (%d, %d)"
    # if param.log:
    #     float_template += "l"
    #     int_template += "l"

    if param.q is not None:
        q_prefix = "Q%d_" % (int(param.q),)
    else:
        q_prefix = ""
    default = param.default

    if isinstance(param, IntegerHyperparameter):
        default = int(default)
        return int_template % (param.name, param.name, param.lower,
                               param.upper)
    else:
        return float_template % (param.name, param.name, str(param.lower),
                                 str(param.upper))


def build_condition(condition):
    if not isinstance(condition, ConditionComponent):
        raise TypeError("build_condition must be called with an instance of "
                        "'%s', got '%s'" %
                        (ConditionComponent, type(condition)))

    # Check if IRACE can handle the condition
    if isinstance(condition, OrConjunction):
        raise NotImplementedError("IRACE cannot handle OR conditions: %s" %
                                  (condition))
    if isinstance(condition, NotEqualsCondition):
        raise NotImplementedError("IRACE cannot handle != conditions: %s" %
                                  (condition))

    # Findout type of parent
    pType = "i"
    if condition.parent.__class__.__name__ == 'CategoricalHyperparameter':
        pType = 'c'
    if condition.parent.__class__.__name__ == 'UniformFloatHyperparameter':
        pType = 'r'

    # Now handle the conditions SMAC can handle
    condition_template = "%s | %s %%in%% %s(%s) "
    if isinstance(condition, AndConjunction):
        raise NotImplementedError("This is not yet implemented!")
    elif isinstance(condition, InCondition):
        return condition_template % (condition.child.name,
                                     condition.parent.name,
                                     pType,
                                     ", ".join(condition.values))
    elif isinstance(condition, EqualsCondition):
        return condition_template % (condition.child.name,
                                     condition.parent.name,
                                     pType,
                                     condition.value)


def build_forbidden(clause):
    if not isinstance(clause, AbstractForbiddenComponent):
        raise TypeError("build_forbidden must be called with an instance of "
                        "'%s', got '%s'" %
                        (AbstractForbiddenComponent, type(clause)))

    if not isinstance(clause, (ForbiddenEqualsClause, ForbiddenAndConjunction)):
        raise NotImplementedError("SMAC cannot handle '%s' of type %s" %
                                  str(clause), (type(clause)))

    retval = six.StringIO()
    retval.write("{")
    # Really simple because everything is an AND-conjunction of equals
    # conditions
    dlcs = clause.get_descendant_literal_clauses()
    for dlc in dlcs:
        if retval.tell() > 1:
            retval.write(", ")
        retval.write("%s=%s" % (dlc.hyperparameter.name, dlc.value))
    retval.write("}")
    retval.seek(0)
    return retval.getvalue()


def write(configuration_space):
    if not isinstance(configuration_space, ConfigurationSpace):
        raise TypeError("pcs_parser.write expects an instance of %s, "
                        "you provided '%s'" % (ConfigurationSpace,
                                               type(configuration_space)))

    param_lines = six.StringIO()
    condition_lines = six.StringIO()
    forbidden_lines = []
    for hyperparameter in configuration_space.get_hyperparameters():
        # Check if the hyperparameter names are valid IRACE names!
        try:
            pp_param_name.parseString(hyperparameter.name)
        except pyparsing.ParseException:
            raise ValueError(
                "Illegal hyperparameter name for IRACE: %s" % hyperparameter.name)

        # First build params
        if param_lines.tell() > 0:
            param_lines.write("\n")
        if isinstance(hyperparameter, NumericalHyperparameter):
            # print "building countinuous param"
            param_lines.write(build_continuous(hyperparameter))
        elif isinstance(hyperparameter, CategoricalHyperparameter):
            # print "building categorical param"
            param_lines.write(build_categorical(hyperparameter))
        elif isinstance(hyperparameter, Constant):
            # print "building constant param"
            param_lines.write(build_constant(hyperparameter))
        else:
            raise TypeError("Unknown type: %s (%s)" % (
                type(hyperparameter), hyperparameter))

    for condition in configuration_space.get_conditions():
        if condition_lines.tell() > 0:
            condition_lines.write("\n")
        condition_lines.write(build_condition(condition))

    for forbidden_clause in configuration_space.forbidden_clauses:
        # Convert in-statement into two or more equals statements
        dlcs = forbidden_clause.get_descendant_literal_clauses()
        # First, get all in statements and convert them to equal statements
        in_statements = []
        other_statements = []
        for dlc in dlcs:
            if isinstance(dlc, MultipleValueForbiddenClause):
                if not isinstance(dlc, ForbiddenInClause):
                    raise ValueError("IRACE cannot handle this forbidden "
                                     "clause: %s" % dlc)
                in_statements.append(
                    [ForbiddenEqualsClause(dlc.hyperparameter, value)
                     for value in dlc.values])
            else:
                other_statements.append(dlc)
        # Second, create the product of all elements in the IN statements,
        # create a ForbiddenAnd and add all ForbiddenEquals
        if len(in_statements) > 0:
            for i, p in enumerate(product(*in_statements)):
                all_forbidden_clauses = list(p) + other_statements
                f = ForbiddenAndConjunction(*all_forbidden_clauses)
                forbidden_lines.append(build_forbidden(f))
        else:
            forbidden_lines.append(build_forbidden(forbidden_clause))

    # Add condtions: first conver param_lines to array then search first part of condition in that array
    # if found append second part of condition to that array part
    splitted_params = param_lines.getvalue().split("\n")
    if condition_lines.tell() > 0:
        condition_lines.seek(0)
        param_lines.write("\n\n")
        for line in condition_lines:
            param_lines.write(line)
            t = filter(lambda x: line.split(" ")[0] in x, splitted_params)[0]
            index = splitted_params.index(t)
            splitted_params[index] = splitted_params[index] + "  ".join(line.split(" ")[1:])

    for i, j in enumerate(splitted_params):
        if j[-1] != "\n":
            splitted_params[i] += "\n"
    # print splitted_params

    # TODO: check if irace supports forbidden lines
    # if len(forbidden_lines) > 0:
    #     forbidden_lines.sort()
    #     param_lines.write("\n\n")
    #     for line in forbidden_lines:
    #         param_lines.write(line)
    #         param_lines.write("\n")

    # Check if the default configuration is a valid configuration!

    # overwrite param_lines with splitted_params which contains lines with conditions
    param_lines = six.StringIO()
    for l in splitted_params:
        param_lines.write(l)
    return param_lines.getvalue()
