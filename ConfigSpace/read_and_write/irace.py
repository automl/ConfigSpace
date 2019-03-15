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

from itertools import product
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    NumericalHyperparameter, Constant, IntegerHyperparameter, \
    NormalIntegerHyperparameter, NormalFloatHyperparameter, OrdinalHyperparameter
from ConfigSpace.conditions import EqualsCondition, NotEqualsCondition, \
    InCondition, AndConjunction, OrConjunction, ConditionComponent, GreaterThanCondition, LessThanCondition
# from ConfigSpace.forbidden import ForbiddenEqualsClause, \
#     ForbiddenAndConjunction, ForbiddenInClause, AbstractForbiddenComponent, MultipleValueForbiddenClause
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction, ForbiddenInClause, AbstractForbiddenComponent, MultipleValueForbiddenClause
import pyparsing
import io
import numpy as np

# Build pyparsing expressions for params
pp_param_name = pyparsing.Word(pyparsing.alphanums + "_" + "-" + "@" + "." + ":" + ";" + "\\" + "/" + "?" + "!" +
                               "$" + "%" + "&" + "*" + "+" + "<" + ">")


def build_categorical(param):
    cat_template = "%s '--%s ' c {%s}"
    return cat_template % (param.name, param.name,
                           ",".join([str(value) for value in param.choices]))


def build_ordinal(param):
    cat_template = "%s '--%s ' o {%s}"
    return cat_template % (param.name, param.name,
                           ",".join([str(value) for value in param.sequence]))


def build_constant(param):
    constant_template = "%s '--%s ' c (%s)"
    return constant_template % (param.name, param.name, param.value)


def build_continuous(param):
    if type(param) in (NormalIntegerHyperparameter,
                       NormalFloatHyperparameter):
        param = param.to_uniform()

    float_template = "%s '--%s ' r (%f, %f)"
    int_template = "%s '--%s ' i (%d, %d)"

    if param.log:
        lower = np.log(param.lower)
        upper = np.log(param.upper)
        # default = np.log(param.default)

    else:
        lower = param.lower
        upper = param.upper
        # default = param.default

    # if param.q is not None
    #     q_prefix = "Q%d_" % (int(param.q),)
    # else:
    #     q_prefix = ""

    if isinstance(param, IntegerHyperparameter):
        return int_template % (param.name, param.name, int(lower),
                               int(upper))
    else:
        return float_template % (param.name, param.name, float(lower),
                                 float(upper))


def build_condition(condition):
    if not isinstance(condition, ConditionComponent):
        raise TypeError("build_condition must be called with an instance of "
                        "'%s', got '%s'" %
                        (ConditionComponent, type(condition)))

    # Now handle the conditions IRACE can handle
    in_template = "%s %%in%% %s(%s)"
    less_template = "%s < %s"
    greater_template = "%s > %s"
    notequal_template = "%s != %s"
    equal_template = "%s==%s"

    full_condition = ''
    child = ''
    conditions = [condition]  # if not a conjunction this makes condition iteratable

    # Check if IRACE can handle the condition
    if isinstance(condition, OrConjunction):
        logic = " || "
        conditions = condition.components

    elif isinstance(condition, AndConjunction):
        logic = " && "
        conditions = condition.components

    else:
        logic = None

    for clause in conditions:
        child = clause.child.name
        # Findout type of parent
        if (isinstance(clause.parent, UniformIntegerHyperparameter) or
           isinstance(clause.parent, NormalIntegerHyperparameter)):
            pType = 'i'
        elif isinstance(clause.parent, UniformFloatHyperparameter) or isinstance(clause.parent,
                                                                                 NormalFloatHyperparameter):
            pType = 'r'
        elif isinstance(clause.parent, CategoricalHyperparameter):
            pType = 'c'
        elif isinstance(clause.parent, OrdinalHyperparameter):
            pType = 'o'
        else:
            raise TypeError("Parent Type of Condition is unknown")

        # check if  parent type is log integer, if it is then convert it to log manually
        # this conversion of log is needed because irace doesnt natively support params of type log
        if pType == "i" or pType == "r":
            if clause.parent.log:
                # unlike other condition types which have variable called 'value'
                # the InCondition has array called 'values'
                if hasattr(clause, 'values'):
                    clause.values = np.log(clause.values)
                else:
                    clause.value = np.log(clause.value)

        if full_condition != '':
            full_condition += logic

        if isinstance(clause, NotEqualsCondition):
            full_condition += notequal_template % (clause.parent.name, clause.value)

        elif isinstance(clause, InCondition):
            full_condition += in_template % (clause.parent.name, pType, ",".join(clause.values))

        elif isinstance(clause, EqualsCondition):
            full_condition += equal_template % (clause.parent.name, clause.value)

        elif isinstance(clause, LessThanCondition):
            full_condition += less_template % (clause.parent.name, clause.value)

        elif isinstance(clause, GreaterThanCondition):
            full_condition += greater_template % (clause.parent.name, clause.value)

    return child + ' | ' + full_condition


def build_forbidden(clause):
    if not isinstance(clause, AbstractForbiddenComponent):
        raise TypeError("build_forbidden must be called with an instance of "
                        "'%s', got '%s'" %
                        (AbstractForbiddenComponent, type(clause)))

    if not isinstance(clause, (ForbiddenEqualsClause, ForbiddenAndConjunction)):
        raise NotImplementedError("IRACE cannot handle '%s' of type %s" %
                                  str(clause), (type(clause)))

    retval = io.StringIO()
    retval.write("(")
    # Really simple because everything is an AND-conjunction of equals
    # conditions
    dlcs = clause.get_descendant_literal_clauses()
    for dlc in dlcs:
        if retval.tell() > 1:
            retval.write(" && ")
        retval.write("%s==%s" % (dlc.hyperparameter.name, dlc.value))
    retval.write(")")
    retval.seek(0)
    return retval.getvalue()


def write(configuration_space):
    if not isinstance(configuration_space, ConfigurationSpace):
        raise TypeError("irace.write expects an instance of %s, "
                        "you provided '%s'" % (ConfigurationSpace,
                                               type(configuration_space)))

    param_lines = io.StringIO()
    condition_lines = io.StringIO()
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

        elif isinstance(hyperparameter, OrdinalHyperparameter):
            # print "building constant param"
            param_lines.write(build_ordinal(hyperparameter))

        else:
            raise TypeError("Unknown type: %s (%s)" % (
                type(hyperparameter), hyperparameter))

    for condition in configuration_space.get_conditions():
        if condition_lines.tell() > 0:
            condition_lines.write("\n")
        condition_lines.write(build_condition(condition))

    for forbidden_clause in configuration_space.get_forbiddens():
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

    # Add conditions: first convert param_lines to array then search first part of condition in that array
    # if found append second part of condition to that array part
    splitted_params = param_lines.getvalue().split("\n")
    if condition_lines.tell() > 0:
        condition_lines.seek(0)
        param_lines.write("\n\n")
        for line in condition_lines:
            param_lines.write(line)
            t = filter(lambda x: line.split(" ")[0] in x, splitted_params)
            index = splitted_params.index(next(t))
            splitted_params[index] = splitted_params[index] + "  ".join(line.split(" ")[1:])

    for i, j in enumerate(splitted_params):
        if j[-1] != "\n":
            splitted_params[i] += "\n"

    forbidden_lines_write = io.StringIO()
    if len(forbidden_lines) > 0:
        for forbidden in forbidden_lines:
            forbidden_lines_write.write(forbidden + '\n')

    output_fh = open('forbidden.txt', 'w')
    output_fh.write(forbidden_lines_write.getvalue())
    output_fh.close()

    # overwrite param_lines with split_params which contains lines with conditions
    param_lines = io.StringIO()
    for l in splitted_params:
        param_lines.write(l)
    return param_lines.getvalue()
