#!/usr/bin/env python
"""
The old PCS format is part of the `Algorithm Configuration Library <http://aclib.net/#>`_.

A detailed explanation of the **old** PCS format can be found
`here. <http://aclib.net/cssc2014/pcs-format.pdf>`_
"""
from __future__ import annotations

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

import sys
from collections import OrderedDict
from io import StringIO
from itertools import product
from typing import Iterable

import pyparsing

from ConfigSpace.conditions import (
    AndConjunction,
    ConditionComponent,
    EqualsCondition,
    InCondition,
    NotEqualsCondition,
    OrConjunction,
)
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import (
    AbstractForbiddenComponent,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    MultipleValueForbiddenClause,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    IntegerHyperparameter,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    NumericalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

# Build pyparsing expressions for params
pp_param_name = pyparsing.Word(
    pyparsing.alphanums
    + "_"
    + "-"
    + "@"
    + "."
    + ":"
    + ";"
    + "\\"
    + "/"
    + "?"
    + "!"
    + "$"
    + "%"
    + "&"
    + "*"
    + "+"
    + "<"
    + ">",
)
pp_digits = "0123456789"
pp_plusorminus = pyparsing.Literal("+") | pyparsing.Literal("-")
pp_int = pyparsing.Combine(pyparsing.Optional(pp_plusorminus) + pyparsing.Word(pp_digits))
pp_float = pyparsing.Combine(
    pyparsing.Optional(pp_plusorminus) + pyparsing.Optional(pp_int) + "." + pp_int,
)
pp_eorE = pyparsing.Literal("e") | pyparsing.Literal("E")
pp_floatorint = pp_float | pp_int
pp_e_notation = pyparsing.Combine(pp_floatorint + pp_eorE + pp_int)
pp_number = pp_e_notation | pp_float | pp_int
pp_numberorname = pp_number | pp_param_name
pp_il = pyparsing.Word("il")
pp_choices = pp_param_name + pyparsing.Optional(pyparsing.OneOrMore("," + pp_param_name))

pp_cont_param = (
    pp_param_name
    + "["
    + pp_number
    + ","
    + pp_number
    + "]"
    + "["
    + pp_number
    + "]"
    + pyparsing.Optional(pp_il)
)
pp_cat_param = pp_param_name + "{" + pp_choices + "}" + "[" + pp_param_name + "]"
pp_condition = pp_param_name + "|" + pp_param_name + "in" + "{" + pp_choices + "}"
pp_forbidden_clause = (
    "{"
    + pp_param_name
    + "="
    + pp_numberorname
    + pyparsing.Optional(pyparsing.OneOrMore("," + pp_param_name + "=" + pp_numberorname))
    + "}"
)


def build_categorical(param: CategoricalHyperparameter) -> str:
    if param.weights is not None:
        raise ValueError(
            "The pcs format does not support categorical hyperparameters with "
            "assigned weights (for hyperparameter %s)" % param.name,
        )
    cat_template = "%s {%s} [%s]"
    return cat_template % (
        param.name,
        ", ".join([str(value) for value in param.choices]),
        str(param.default_value),
    )


def build_constant(param: Constant) -> str:
    constant_template = "%s {%s} [%s]"
    return constant_template % (param.name, param.value, param.value)


def build_continuous(param: NormalIntegerHyperparameter | NormalFloatHyperparameter) -> str:
    if type(param) in (NormalIntegerHyperparameter, NormalFloatHyperparameter):
        param = param.to_uniform()

    float_template = "%s%s [%s, %s] [%s]"
    int_template = "%s%s [%d, %d] [%d]i"
    if param.log:
        float_template += "l"
        int_template += "l"

    q_prefix = "Q%d_" % (int(param.q),) if param.q is not None else ""
    default_value = param.default_value

    if isinstance(param, IntegerHyperparameter):
        default_value = int(default_value)
        return int_template % (q_prefix, param.name, param.lower, param.upper, default_value)

    return float_template % (
        q_prefix,
        param.name,
        str(param.lower),
        str(param.upper),
        str(default_value),
    )


def build_condition(condition: ConditionComponent) -> str:
    if not isinstance(condition, ConditionComponent):
        raise TypeError(
            "build_condition must be called with an instance of "
            "'{}', got '{}'".format(ConditionComponent, type(condition)),
        )

    # Check if SMAC can handle the condition
    if isinstance(condition, OrConjunction):
        raise NotImplementedError("SMAC cannot handle OR conditions: %s" % (condition))
    if isinstance(condition, NotEqualsCondition):
        raise NotImplementedError("SMAC cannot handle != conditions: %s" % (condition))

    # Now handle the conditions SMAC can handle
    condition_template = "%s | %s in {%s}"
    if isinstance(condition, AndConjunction):
        return "\n".join([build_condition(cond) for cond in condition.components])

    if isinstance(condition, InCondition):
        return condition_template % (
            condition.child.name,
            condition.parent.name,
            ", ".join(condition.values),
        )

    if isinstance(condition, EqualsCondition):
        return condition_template % (condition.child.name, condition.parent.name, condition.value)

    raise NotImplementedError(condition)


def build_forbidden(clause: AbstractForbiddenComponent) -> str:
    if not isinstance(clause, AbstractForbiddenComponent):
        raise TypeError(
            "build_forbidden must be called with an instance of "
            "'{}', got '{}'".format(AbstractForbiddenComponent, type(clause)),
        )

    if not isinstance(clause, (ForbiddenEqualsClause, ForbiddenAndConjunction)):
        raise NotImplementedError(
            "SMAC cannot handle '{}' of type {}".format(*str(clause)),
            (type(clause)),
        )

    retval = StringIO()
    retval.write("{")
    # Really simple because everything is an AND-conjunction of equals
    # conditions
    dlcs = clause.get_descendant_literal_clauses()
    for dlc in dlcs:
        if retval.tell() > 1:
            retval.write(", ")
        retval.write(f"{dlc.hyperparameter.name}={dlc.value}")
    retval.write("}")
    retval.seek(0)
    return retval.getvalue()


def read(pcs_string: Iterable[str]) -> ConfigurationSpace:
    """
    Read in a :py:class:`~ConfigSpace.configuration_space.ConfigurationSpace`
    definition from a pcs file.


    .. code:: python

        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.read_and_write import pcs

        cs = ConfigurationSpace({"a": [1, 2, 3]})
        with open('configspace.pcs', 'w') as f:
             f.write(pcs.write(cs))

        with open('configspace.pcs', 'r') as f:
            deserialized_conf = pcs.read(f)

    Parameters
    ----------
    pcs_string : Iterable[str]
        ConfigSpace definition in pcs format as an iterable of strings

    Returns
    -------
    :py:class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The deserialized ConfigurationSpace object
    """
    if isinstance(pcs_string, str):
        pcs_string = pcs_string.split("\n")

    configuration_space = ConfigurationSpace()
    conditions = []
    forbidden = []

    # some statistics
    ct = 0
    cont_ct = 0
    cat_ct = 0
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
            except pyparsing.ParseException as e:
                raise NotImplementedError(f"Could not parse condition: {line}") from e

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

        create = {
            "int": UniformIntegerHyperparameter,
            "float": UniformFloatHyperparameter,
            "categorical": CategoricalHyperparameter,
        }

        try:
            param_list = pp_cont_param.parseString(line)
            il = param_list[9:]
            if len(il) > 0:
                il = il[0]
            param_list = param_list[:9]
            name = param_list[0]
            lower = float(param_list[2])  # type: ignore
            upper = float(param_list[4])  # type: ignore
            paramtype = "int" if "i" in il else "float"
            log = "l" in il
            default_value = float(param_list[7])  # type: ignore
            param = create[paramtype](
                name=name,
                lower=lower,
                upper=upper,
                q=None,
                log=log,
                default_value=default_value,
            )
            cont_ct += 1
        except pyparsing.ParseException:
            pass

        try:
            param_list = pp_cat_param.parseString(line)
            name = param_list[0]
            choices = list(param_list[2:-4:2])
            default_value = param_list[-2]
            param = create["categorical"](name=name, choices=choices, default_value=default_value)
            cat_ct += 1
        except pyparsing.ParseException:
            pass

        if param is None:
            raise NotImplementedError("Could not parse: %s" % line)

        configuration_space.add_hyperparameter(param)

    for clause in forbidden:
        # TODO test this properly!
        # TODO Add a try/catch here!
        # noinspection PyUnusedLocal
        param_list = pp_forbidden_clause.parseString(clause)
        tmp_list: list = []
        clause_list = []
        for value in param_list[1:]:
            if len(tmp_list) < 3:
                tmp_list.append(value)
            else:
                # So far, only equals is supported by SMAC
                if tmp_list[1] == "=":
                    # TODO maybe add a check if the hyperparameter is
                    # actually in the configuration space
                    clause_list.append(
                        ForbiddenEqualsClause(configuration_space[tmp_list[0]], tmp_list[2]),
                    )
                else:
                    raise NotImplementedError()
                tmp_list = []
        configuration_space.add_forbidden_clause(ForbiddenAndConjunction(*clause_list))

    # Now handle conditions
    # If there are two conditions for one child, these two conditions are an
    # AND-conjunction of conditions, thus we have to connect them
    conditions_per_child: dict = OrderedDict()
    for condition in conditions:
        child_name = condition[0]
        if child_name not in conditions_per_child:
            conditions_per_child[child_name] = []
        conditions_per_child[child_name].append(condition)

    for child_name in conditions_per_child:
        condition_objects = []
        for condition in conditions_per_child[child_name]:
            child = configuration_space[child_name]
            parent_name = condition[2]
            parent = configuration_space[parent_name]
            restrictions = condition[5:-1:2]

            # TODO: cast the type of the restriction!
            if len(restrictions) == 1:
                condition = EqualsCondition(child, parent, restrictions[0])
            else:
                condition = InCondition(child, parent, values=restrictions)
            condition_objects.append(condition)

        # Now we have all condition objects for this child, so we can build a
        #  giant AND-conjunction of them (if number of conditions >= 2)!

        if len(condition_objects) > 1:
            and_conjunction = AndConjunction(*condition_objects)
            configuration_space.add_condition(and_conjunction)
        else:
            configuration_space.add_condition(condition_objects[0])

    return configuration_space


def write(configuration_space: ConfigurationSpace) -> str:
    """
    Create a string representation of a
    :class:`~ConfigSpace.configuration_space.ConfigurationSpace` in pcs format.
    This string can be written to file.

    .. code:: python

        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.read_and_write import pcs

        cs = ConfigurationSpace({"a": [1, 2, 3]})

        with open('configspace.pcs', 'w') as fh:
            fh.write(pcs.write(cs))

    Parameters
    ----------
    configuration_space : :py:class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        a configuration space

    Returns
    -------
    str
        The string representation of the configuration space

    """
    if not isinstance(configuration_space, ConfigurationSpace):
        raise TypeError(
            f"pcs_parser.write expects an instance of {ConfigurationSpace}, "
            f"you provided '{type(configuration_space)}'",
        )

    param_lines = StringIO()
    condition_lines = StringIO()
    forbidden_lines = []
    for hyperparameter in configuration_space.values():
        # Check if the hyperparameter names are valid ConfigSpace names!
        try:
            pp_param_name.parseString(hyperparameter.name)
        except pyparsing.ParseException as e:
            raise ValueError(
                f"Illegal hyperparameter name for ConfigSpace: {hyperparameter.name}",
            ) from e

        # First build params
        if param_lines.tell() > 0:
            param_lines.write("\n")
        if isinstance(hyperparameter, NumericalHyperparameter):
            param_lines.write(build_continuous(hyperparameter))
        elif isinstance(hyperparameter, CategoricalHyperparameter):
            param_lines.write(build_categorical(hyperparameter))
        elif isinstance(hyperparameter, Constant):
            param_lines.write(build_constant(hyperparameter))
        else:
            raise TypeError(f"Unknown type: {type(hyperparameter)} ({hyperparameter})")

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
                    raise ValueError("SMAC cannot handle this forbidden " "clause: %s" % dlc)
                in_statements.append(
                    [ForbiddenEqualsClause(dlc.hyperparameter, value) for value in dlc.values],
                )
            else:
                other_statements.append(dlc)

        # Second, create the product of all elements in the IN statements,
        # create a ForbiddenAnd and add all ForbiddenEquals
        if len(in_statements) > 0:
            for p in product(*in_statements):
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


if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        orig_pcs = fh.readlines()

    sp = read(orig_pcs)
    created_pcs = write(sp).split("\n")
    print("============== Writing Results")
    print("#Lines: ", len(created_pcs))
    print("#LostLines: ", len(orig_pcs) - len(created_pcs))
    diff = ["%s\n" % i for i in created_pcs if i not in " ".join(orig_pcs)]
    print("Identical Lines: ", len(created_pcs) - len(diff))
    print()
    print("Up to 10 random different lines (of %d):" % len(diff))
    print("".join(diff[:10]))
