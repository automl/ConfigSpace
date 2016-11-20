#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

__authors__ = ["Katharina Eggensperger", "Matthias Feurer", "Christina Hernández Wunsch"]
__contact__ = "automl.org"

from collections import OrderedDict
from itertools import product

import pyparsing
import six
import sys

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    NumericalHyperparameter, IntegerHyperparameter, \
    NormalIntegerHyperparameter, NormalFloatHyperparameter, OrdinalHyperparameter
from ConfigSpace.conditions import EqualsCondition, NotEqualsCondition,\
    InCondition, AndConjunction, OrConjunction, ConditionComponent,\
    GreaterThanCondition, LessThanCondition
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction, ForbiddenInClause, AbstractForbiddenComponent, MultipleValueForbiddenClause


# Build pyparsing expressions for params
pp_param_name = pyparsing.Word(pyparsing.alphanums + "_" + "-" + "@" + "." + ":" + ";" + "\\" + "/" + "?" + "!" +
                               "$" + "%" + "&" + "*" + "+" + "<" + ">")
pp_param_operation = pyparsing.Word("in" + "!=" + "==" + ">" + "<")
pp_digits = "0123456789"
pp_param_val = pp_param_name + pyparsing.Optional(pyparsing.OneOrMore("," + pp_param_name))
pp_plusorminus = pyparsing.Literal('+') | pyparsing.Literal('-')
pp_int = pyparsing.Combine(pyparsing.Optional(pp_plusorminus) + pyparsing.Word(pp_digits))
pp_float = pyparsing.Combine(pyparsing.Optional(pp_plusorminus) + pyparsing.Optional(pp_int) + "." + pp_int)
pp_eorE = pyparsing.Literal('e') | pyparsing.Literal('E')
pp_param_type = pyparsing.Word("integer" + "real" + "categorical" + "ordinal")
pp_floatorint = pp_float | pp_int
pp_e_notation = pyparsing.Combine(pp_floatorint + pp_eorE + pp_int)
pp_number = pp_e_notation | pp_float | pp_int
pp_numberorname = pp_number | pp_param_name
pp_log = pyparsing.Word("log")
pp_connective = pyparsing.Word("||" + "&&")
pp_choices = pp_param_name + pyparsing.Optional(pyparsing.OneOrMore("," + pp_param_name))
pp_sequence = pp_param_name + pyparsing.Optional(pyparsing.OneOrMore("," + pp_param_name))
pp_ord_param = pp_param_name + pp_param_type + "{" + pp_sequence + "}" + "[" + pp_param_name + "]"
pp_cont_param = pp_param_name + pp_param_type + "[" + pp_number + "," + pp_number + "]" + "[" + pp_number + "]" + pyparsing.Optional(pp_log)
pp_cat_param = pp_param_name + pp_param_type + "{" + pp_choices + "}" + "[" + pp_param_name + "]"
pp_condition = pp_param_name + "|" +  pp_param_name + pp_param_operation + \
    pyparsing.Optional('{') + pp_param_val + pyparsing.Optional('}') + \
    pyparsing.Optional(pyparsing.OneOrMore(pp_connective  + pp_param_name + pp_param_operation + pp_param_val))
pp_forbidden_clause = "{" + pp_param_name + "=" + pp_numberorname + \
    pyparsing.Optional(pyparsing.OneOrMore("," + pp_param_name + "=" + pp_numberorname)) + "}"


def build_categorical(param):
    cat_template = "%s categorical {%s} [%s]"
    return cat_template % (param.name,
                           ", ".join([str(value) for value in param.choices]),
                           str(param.default))

def build_ordinal(param):
    ordinal_template = '%s ordinal {%s} [%s]'
    return ordinal_template % (param.name, 
                               ", ".join([str(value) for value in param.sequence]),
                                str(param.default))
                                
def build_continuous(param):
    if type(param) in (NormalIntegerHyperparameter,
                       NormalFloatHyperparameter):
        param = param.to_uniform()

    float_template = "%s%s real [%s, %s] [%s]"
    int_template = "%s%s integer [%d, %d] [%d]"
    if param.log:
        float_template += "log"
        int_template += "log"

    if param.q is not None:
        q_prefix = "Q%d_" % (int(param.q),)
    else:
        q_prefix = ""
    default = param.default

    if isinstance(param, IntegerHyperparameter):
        default = int(default)
        return int_template % (q_prefix, param.name, param.lower,
                               param.upper, default)
    else:
        return float_template % (q_prefix, param.name,  str(param.lower),
                                 str(param.upper), str(default))

def build_condition(condition):
    if not isinstance(condition, ConditionComponent):
        raise TypeError("build_condition must be called with an instance of "
                        "'%s', got '%s'" %
                        (ConditionComponent, type(condition)))

    # Now handle the conditions SMAC can handle
    in_template = "%s | %s in {%s}"
    less_template = "%s | %s < %s"
    greater_template = "%s | %s > %s"
    notequal_template = "%s | %s != %s"
    equal_template = "%s | %s == %s"    
    if isinstance(condition, NotEqualsCondition):
        return notequal_template % (condition.child.name,
                                    condition.parent.name,
                                    condition.value)
                                     
    elif isinstance(condition, InCondition):
        return in_template % (condition.child.name,
                              condition.parent.name,
                              ", ".join(condition.values))
                                     
    elif isinstance(condition, EqualsCondition):
        return equal_template % (condition.child.name,
                                 condition.parent.name,
                                 condition.value)
    elif isinstance(condition, LessThanCondition):
        return less_template % (condition.child.name,
                                 condition.parent.name,
                                 condition.value)
    elif isinstance(condition, GreaterThanCondition):
        return greater_template % (condition.child.name,
                                 condition.parent.name,
                                 condition.value)

def build_forbidden(clause):
    if not isinstance(clause, AbstractForbiddenComponent):
        raise TypeError("build_forbidden must be called with an instance of "
                        "'%s', got '%s'" %
                        (AbstractForbiddenComponent, type(clause)))
                        
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


def read(pcs_string, debug=False):
    configuration_space = ConfigurationSpace()
    conditions = []
    forbidden = []

    # some statistics
    ct = 0
    cont_ct = 0
    cat_ct = 0
    ord_ct = 0
    line_ct = 0

    for line in pcs_string:
        line_ct += 1

        if "#" in line:
            # It contains a comment
            pos = line.find("#")
            line = line[:pos]

        # Remove quotes and whitespaces at beginning and end
        line = line.replace('"', "").replace("'", "")
        line = line.strip()
        if "|" in line:
            # It's a condition
            try:
                c = pp_condition.parseString(line)
                conditions.append(c)
            except pyparsing.ParseException:
                raise NotImplementedError("Could not parse condition: %s" % line)
            continue
        if "}" not in line and "]" not in line:
            continue
        if line.startswith("{") and line.endswith("}"):
            forbidden.append(line)
            continue
        if len(line.strip()) == 0:
            continue

        ct += 1
        param = None

        create = {"int": UniformIntegerHyperparameter,
                  "float": UniformFloatHyperparameter,
                  "categorical": CategoricalHyperparameter,
                  "ordinal": OrdinalHyperparameter}

        try:
            param_list = pp_cont_param.parseString(line)
            log = param_list[10:]
            if len(log) > 0:
                log = log[0]
            param_list = param_list[:10]
            name = param_list[0]
            paramtype = "int" if "integer" in param_list else "float"
            lower = float(param_list[3])
            upper = float(param_list[5])
            log_on = True if "log" in log else False
            default = float(param_list[8])
            param = create[paramtype](name=name, lower=lower, upper=upper,
                                      q=None, log=log_on, default=default)
            cont_ct += 1
        except pyparsing.ParseException:
            pass

        try:
            if "categorical" in line:
                param_list = pp_cat_param.parseString(line)
                name = param_list[0]
                choices = [choice for choice in param_list[3:-4:2]]
                default = param_list[-2]
                param = create["categorical"](name=name, choices=choices, default=default)
                cat_ct += 1
                
            elif "ordinal" in line:
                param_list = pp_ord_param.parseString(line)
                name = param_list[0]
                sequence = [seq for seq in param_list[3:-4:2]]
                default = param_list[-2]
                param = create["ordinal"](name=name, sequence=sequence, default=default)
                ord_ct += 1
                
        except pyparsing.ParseException:
            pass

        if param is None:
            raise NotImplementedError("Could not parse: %s" % line)

        configuration_space.add_hyperparameter(param)

    for clause in forbidden:
        param_list = pp_forbidden_clause.parseString(clause)
        tmp_list = []
        clause_list = []
        for value in param_list[1:]:
            if len(tmp_list) < 3:
                tmp_list.append(value)
            else:
                # So far, only equals is supported by SMAC
                if tmp_list[1] == '=':
                    # TODO maybe add a check if the hyperparameter is
                    # actually in the configuration space
                    clause_list.append(ForbiddenEqualsClause(
                        configuration_space.get_hyperparameter(tmp_list[0]),
                        tmp_list[2]))
                else:
                    raise NotImplementedError()
                tmp_list = []
        configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
            *clause_list))
    connectives = []
    conditions_per_child = OrderedDict()
    for condition in conditions:        
        child_name = condition[0]
        if child_name not in conditions_per_child:
            conditions_per_child[child_name] = list()
        connective = []
        if ('&&' in str(condition)) or ('||' in str(condition)):
            i = 2
            j = 2
            while i < len(condition):
                if condition[i] == '&&' or condition[i] == '||':
                    connective.append(condition[i])
                    conditions_per_child[child_name].append(condition[j:i])
                    i += 1
                    j = i
                else:
                    i +=1
                    rest = condition[j:i+1]
            connectives.append(connective)
            conditions_per_child[child_name].append(rest)
        else:
            conditions_per_child[child_name].append(condition)
    k = 0
    for child_name in conditions_per_child:
        condition_objects = []
        if len(conditions_per_child[child_name]) > 1:
            connection = connectives[k]
            k +=1
            for condition in conditions_per_child[child_name]:
                child = configuration_space.get_hyperparameter(child_name)
                parent_name = condition[0]
                parent = configuration_space.get_hyperparameter(parent_name)
                operation = condition[1]
                # in template must be treated differently from the rest
                if operation == 'in':
                    restrictions = condition[3:-1:2]
                    if len(restrictions) == 1:
                        condition = EqualsCondition(child, parent, restrictions[0])
                    else:
                        condition = InCondition(child, parent, values=restrictions)
                    condition_objects.append(condition)
            
                else:
                    restrictions = condition[2]
            
                    if operation == '==':
                        condition = EqualsCondition(child, parent, restrictions)
                    elif operation == '!=':
                        condition = NotEqualsCondition(child, parent, restrictions)
                    elif operation == '<':
                        condition = LessThanCondition(child, parent, restrictions)
                    elif operation == '>':
                        condition = GreaterThanCondition(child, parent, restrictions)
        
                    condition_objects.append(condition) 
        else:
            for condition in conditions_per_child[child_name]:
                child = configuration_space.get_hyperparameter(child_name)
                parent_name = condition[2]
                parent = configuration_space.get_hyperparameter(parent_name)
                operation = condition[3]
                # in template must be treated differently from the rest
                if operation == 'in':
                    restrictions = condition[5:-1:2]
                    if len(restrictions) == 1:
                        condition = EqualsCondition(child, parent, restrictions[0])
                    else:
                        condition = InCondition(child, parent, values=restrictions)
                    condition_objects.append(condition)
                
                else:
                    restrictions = condition[4]
                
                    if operation == '==':
                        condition = EqualsCondition(child, parent, restrictions)
                    elif operation == '!=':
                        condition = NotEqualsCondition(child, parent, restrictions)
                    elif operation == '<':
                        condition = LessThanCondition(child, parent, restrictions)
                    elif operation == '>':
                        condition = GreaterThanCondition(child, parent, restrictions)
            
                    condition_objects.append(condition)

        if len(condition_objects) > 1:
            if '||' in connection:
                ands = []
                ors = []
                if '&&' in connection:
                    if (len(connection) % 2 != 0) and (connection.count('&&') % 2 == 0):
                        i = 0
                        j = 0
                    else:        
                        i = 1
                        j = 0
                    while i < len(condition_objects):
                        while j < len(connection):
                            if connection[j] == '&&':
                                ands.append(condition_objects[i-1])
                                ands.append(condition_objects[i])
                                i += 2
                                j += 1
                            else:
                                ors.append(condition_objects[i-1])
                                i += 1
                                j += 1

                    orand_conjunction = OrConjunction(AndConjunction(*ands), *ors)
                    configuration_space.add_condition(orand_conjunction)
                else:
                    or_conjunction = OrConjunction(*condition_objects)
                    configuration_space.add_condition(or_conjunction)
            else:
                and_conjunction = AndConjunction(*condition_objects)
                configuration_space.add_condition(and_conjunction)
        else:
            configuration_space.add_condition(condition_objects[0])
    return configuration_space
    
def write(configuration_space):
    if not isinstance(configuration_space, ConfigurationSpace):
        raise TypeError("pcs_parser.write expects an instance of %s, "
                        "you provided '%s'" % (ConfigurationSpace,
                        type(configuration_space)))

    param_lines = six.StringIO()
    condition_lines = six.StringIO()
    forbidden_lines = []
    for hyperparameter in configuration_space.get_hyperparameters():
        # Check if the hyperparameter names are valid SMAC names!
        try:
            pp_param_name.parseString(hyperparameter.name)
        except pyparsing.ParseException:
            raise ValueError(
                "Illegal hyperparameter name for SMAC: %s" % hyperparameter.name)

        # First build params
        if param_lines.tell() > 0:
            param_lines.write("\n")
        if isinstance(hyperparameter, NumericalHyperparameter):
            param_lines.write(build_continuous(hyperparameter))
        elif isinstance(hyperparameter, CategoricalHyperparameter):
            param_lines.write(build_categorical(hyperparameter))
        elif isinstance(hyperparameter, OrdinalHyperparameter):
            param_lines.write(build_ordinal(hyperparameter))
        else:
            raise TypeError("Unknown type: %s (%s)" % (
                type(hyperparameter), hyperparameter))

    for condition in configuration_space.get_conditions():
        if isinstance(condition, AndConjunction) or isinstance(condition, OrConjunction):
            vals = condition.__repr__()
            condition_lines.write("\n")
            condition_lines.write(vals)
            
        else:
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
                    raise ValueError("SMAC cannot handle this forbidden "
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

    if condition_lines.tell() > 0:
        condition_lines.seek(0)
        param_lines.write("\n\n")
        for line in condition_lines:
            param_lines.write(line)

    if len(forbidden_lines) > 0:
        forbidden_lines.sort()
        param_lines.write("\n\n")
        for line in forbidden_lines:
            param_lines.write(line)
            param_lines.write("\n")

    # Check if the default configuration is a valid configuration!


    param_lines.seek(0)
    return param_lines.getvalue()
