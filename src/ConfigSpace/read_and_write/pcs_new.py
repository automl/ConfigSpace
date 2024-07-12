#!/usr/bin/env python
"""PCS (parameter configuration space) is a simple, human-readable file format for the
description of an algorithm's configurable parameters, their possible values, as well
as any parameter dependencies. There exist an old and a new version.

The new PCS format is part of the
[Algorithm Configuration Library 2.0](https://bitbucket.org/mlindauer/aclib2/src/master/).
A detailed description of the **new** format can be found in the
[ACLIB 2.0 docs](https://bitbucket.org/mlindauer/aclib2/src/aclib2/AClib_Format.md),
in the [SMACv2 docs](https://www.cs.ubc.ca/labs/beta/Projects/SMAC/v2.10.03/manual.pdf)
and further examples are provided in the
[pysmac docs](https://pysmac.readthedocs.io/en/latest/pcs.html)

!!! warning

    The PCS format definition has changed in the year 2016 and is supported by
    AClib 2.0, as well as SMAC (v2 and v3). Please check the
    [serialization guide](../../../reference/serialization.md) for more information.
"""

from __future__ import annotations

__authors__ = [
    "Katharina Eggensperger",
    "Matthias Feurer",
    "Christina Hernández Wunsch",
]
__contact__ = "automl.org"

import warnings
from collections import OrderedDict
from collections.abc import Iterable
from io import StringIO
from itertools import product
from typing_extensions import deprecated

import pyparsing

from ConfigSpace.conditions import (
    AndConjunction,
    Condition,
    Conjunction,
    EqualsCondition,
    GreaterThanCondition,
    InCondition,
    LessThanCondition,
    NotEqualsCondition,
    OrConjunction,
)
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import (
    ForbiddenAndConjunction,
    ForbiddenClause,
    ForbiddenConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    ForbiddenRelation,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    IntegerHyperparameter,
    NumericalHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

warnings.warn(
    "Modules pcs and pcs_new are deprecated but will remain without future support. ",
    DeprecationWarning,
    stacklevel=2,
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
pp_param_operation = pyparsing.Word("in" + "!=" + "==" + ">" + "<")
pp_digits = "0123456789"
pp_param_val = pp_param_name + pyparsing.Optional(
    pyparsing.OneOrMore("," + pp_param_name),
)
pp_plusorminus = pyparsing.Literal("+") | pyparsing.Literal("-")
pp_int = pyparsing.Combine(
    pyparsing.Optional(pp_plusorminus) + pyparsing.Word(pp_digits),
)
pp_float = pyparsing.Combine(
    pyparsing.Optional(pp_plusorminus) + pyparsing.Optional(pp_int) + "." + pp_int,
)
pp_eorE = pyparsing.Literal("e") | pyparsing.Literal("E")
pp_param_type = (
    pyparsing.Literal("integer")
    | pyparsing.Literal("real")
    | pyparsing.Literal("categorical")
    | pyparsing.Literal("ordinal")
)
pp_floatorint = pp_float | pp_int
pp_e_notation = pyparsing.Combine(pp_floatorint + pp_eorE + pp_int)
pp_number = pp_e_notation | pp_float | pp_int
pp_numberorname = pp_number | pp_param_name
pp_log = pyparsing.Literal("log")
# A word matches each character as a set. So && is processed as &
# https://pythonhosted.org/pyparsing/pyparsing.Word-class.html
pp_connectiveOR = pyparsing.Literal("||")
pp_connectiveAND = pyparsing.Literal("&&")
pp_choices = pp_param_name + pyparsing.Optional(
    pyparsing.OneOrMore("," + pp_param_name),
)
pp_sequence = pp_param_name + pyparsing.Optional(
    pyparsing.OneOrMore("," + pp_param_name),
)
pp_ord_param = (
    pp_param_name + pp_param_type + "{" + pp_sequence + "}" + "[" + pp_param_name + "]"
)
pp_cont_param = (
    pp_param_name
    + pp_param_type
    + "["
    + pp_number
    + ","
    + pp_number
    + "]"
    + "["
    + pp_number
    + "]"
    + pyparsing.Optional(pp_log)
)
pp_cat_param = (
    pp_param_name + pp_param_type + "{" + pp_choices + "}" + "[" + pp_param_name + "]"
)
pp_condition = (
    pp_param_name
    + "|"
    + pp_param_name
    + pp_param_operation
    + pyparsing.Optional("{")
    + pp_param_val
    + pyparsing.Optional("}")
    + pyparsing.Optional(
        pyparsing.OneOrMore(
            (pp_connectiveAND | pp_connectiveOR)
            + pp_param_name
            + pp_param_operation
            + pyparsing.Optional("{")
            + pp_param_val
            + pyparsing.Optional("}"),
        ),
    )
)
pp_forbidden_clause = (
    "{"
    + pp_param_name
    + "="
    + pp_numberorname
    + pyparsing.Optional(
        pyparsing.OneOrMore("," + pp_param_name + "=" + pp_numberorname),
    )
    + "}"
)


def build_categorical(param: CategoricalHyperparameter) -> str:
    if param.weights is not None:
        raise ValueError(
            "The pcs format does not support categorical hyperparameters with "
            f"assigned weights (for hyperparameter {param.name})",
        )
    cat_template = "%s categorical {%s} [%s]"
    return cat_template % (
        param.name,
        ", ".join([str(value) for value in param.choices]),
        str(param.default_value),
    )


def build_ordinal(param: OrdinalHyperparameter) -> str:
    ordinal_template = "%s ordinal {%s} [%s]"
    return ordinal_template % (
        param.name,
        ", ".join([str(value) for value in param.sequence]),
        str(param.default_value),
    )


def build_constant(param: Constant) -> str:
    const_template = "%s categorical {%s} [%s]"
    return const_template % (param.name, param.value, param.value)


def build_continuous(param: NumericalHyperparameter) -> str:
    _param = param.to_uniform()

    float_template = "%s%s real [%s, %s] [%s]"
    int_template = "%s%s integer [%d, %d] [%d]"
    if _param.log:
        float_template += "log"
        int_template += "log"

    q_prefix = ""
    default_value = _param.default_value

    if isinstance(_param, IntegerHyperparameter):
        default_value = int(default_value)
        return int_template % (
            q_prefix,
            _param.name,
            _param.lower,
            _param.upper,
            default_value,
        )

    return float_template % (
        q_prefix,
        _param.name,
        str(_param.lower),
        str(_param.upper),
        str(default_value),
    )


def build_condition(condition: Condition) -> str:
    if not isinstance(condition, Condition):
        raise TypeError(
            "build_condition must be called with an instance"
            f" of '{Condition}', got '{type(condition)}'",
        )

    # Now handle the conditions SMAC can handle
    in_template = "%s | %s in {%s}"
    less_template = "%s | %s < %s"
    greater_template = "%s | %s > %s"
    notequal_template = "%s | %s != %s"
    equal_template = "%s | %s == %s"

    if isinstance(condition, InCondition):
        cond_values = ", ".join([str(value) for value in condition.values])
    else:
        cond_values = str(condition.value)

    if isinstance(condition, NotEqualsCondition):
        return notequal_template % (
            condition.child.name,
            condition.parent.name,
            cond_values,
        )

    if isinstance(condition, InCondition):
        return in_template % (
            condition.child.name,
            condition.parent.name,
            cond_values,
        )

    if isinstance(condition, EqualsCondition):
        return equal_template % (
            condition.child.name,
            condition.parent.name,
            cond_values,
        )
    if isinstance(condition, LessThanCondition):
        return less_template % (
            condition.child.name,
            condition.parent.name,
            cond_values,
        )
    if isinstance(condition, GreaterThanCondition):
        return greater_template % (
            condition.child.name,
            condition.parent.name,
            cond_values,
        )

    raise TypeError(f"Didn't find a matching template for type {condition}")


def build_conjunction(conjunction: Conjunction) -> str:
    line: str
    line = conjunction.get_children()[0].name + " | "

    cond_list = []
    for component in conjunction.components:
        if not isinstance(component, Condition):
            # TODO: Fix this
            raise NotImplementedError(
                "build_conjunction only accepts Condition objects, "
                " if this error occurs, please report it to the developers",
            )
        tmp = build_condition(component)

        # This is somehow hacky, but should work for now
        tmp = tmp.split("|")[1].strip()

        cond_list.append(tmp)
    if isinstance(conjunction, AndConjunction):
        line += " && ".join(cond_list)
    elif isinstance(conjunction, OrConjunction):
        line += " || ".join(cond_list)

    return line


def build_forbidden(clause: ForbiddenClause | ForbiddenConjunction) -> str:
    accepted = (ForbiddenRelation, ForbiddenClause, ForbiddenConjunction)
    if not isinstance(clause, accepted):
        raise TypeError(
            "build_forbidden must be called with an instance of "
            f"'{accepted}', got '{type(clause)}'",
        )

    # TODO: Why ...?
    if isinstance(clause, ForbiddenRelation):
        raise NotImplementedError(
            "build_forbidden does not support ForbiddenRelation"
            " objects, please report this to the developers",
        )

    retval = StringIO()
    retval.write("{")
    # Really simple because everything is an AND-conjunction of equals
    # conditions
    dlcs = [clause] if not isinstance(clause, ForbiddenConjunction) else clause.dlcs
    for dlc in dlcs:
        if retval.tell() > 1:
            retval.write(", ")
        # TODO: Fixup
        if isinstance(dlc, ForbiddenRelation):
            raise NotImplementedError(
                "build_forbidden does not support ForbiddenRelation"
                " objects, please report this to the developers",
            )

        if isinstance(dlc, ForbiddenInClause):
            retval.write(f"{dlc.hyperparameter.name}={dlc.values}")
        else:
            sentinal = object()
            _val = getattr(dlc, "value", sentinal)
            assert _val is not sentinal
            retval.write(f"{dlc.hyperparameter.name}={_val}")

    retval.write("}")
    retval.seek(0)
    return retval.getvalue()


def condition_specification(
    child_name: str,
    condition: list[str],
    configuration_space: ConfigurationSpace,
) -> Condition:
    # specifies the condition type
    child = configuration_space[child_name]
    parent_name = condition[0]
    parent = configuration_space[parent_name]
    operation = condition[1]
    if operation == "in":
        restrictions = list(condition[3:-1:2])

        if isinstance(parent, FloatHyperparameter):
            restricted_values = [float(val) for val in restrictions]
        elif isinstance(parent, IntegerHyperparameter):
            restricted_values = [int(val) for val in restrictions]
        else:
            restricted_values = restrictions

        if len(restrictions) == 1:
            cond = EqualsCondition(child, parent, restricted_values[0])
        else:
            cond = InCondition(child, parent, values=restricted_values)
        return cond

    restriction: float | int | str = condition[2]
    if isinstance(parent, FloatHyperparameter):
        restriction = float(restriction)
    elif isinstance(parent, IntegerHyperparameter):
        restriction = int(restriction)

    if operation == "==":
        cond = EqualsCondition(child, parent, restriction)
    elif operation == "!=":
        cond = NotEqualsCondition(child, parent, restriction)
    else:
        if isinstance(parent, FloatHyperparameter):
            restriction = float(restriction)
        elif isinstance(parent, IntegerHyperparameter):
            restriction = int(restriction)
        elif isinstance(parent, OrdinalHyperparameter):
            pass
        else:
            raise ValueError(
                "The parent of a conditional hyperparameter "
                "must be either a float, int or ordinal "
                f"hyperparameter, but is {type(parent)}.",
            )

        if operation == "<":
            cond = LessThanCondition(child, parent, value=restriction)  # type: ignore
        elif operation == ">":
            cond = GreaterThanCondition(child, parent, value=restriction)  # type: ignore
        else:
            raise NotImplementedError(
                f"Could not parse condition: {condition}",
            )

    return cond


@deprecated(
    "pcs_new.read is has stopped being maintained, please use `space.to_json`"
    " or `space.to_yaml` instead",
)
def read(pcs_string: Iterable[str]) -> ConfigurationSpace:
    """Read a [ConfigurationSpace][ConfigSpace.configuration_space.ConfigurationSpace]
    definition from a pcs file.


    ```python exec="true", source="material-block" result="python"
    from ConfigSpace import ConfigurationSpace

    from ConfigSpace.read_and_write import pcs_new
    cs = ConfigurationSpace({"a": [1,2,3]})

    with open('configspace.pcs_new', 'w') as f:
         f.write(pcs_new.write(cs))

    with open('configspace.pcs_new', 'r') as fh:
        deserialized_conf = pcs_new.read(fh)

    print(deserialized_conf)
    ```

    Args:
        pcs_string: ConfigSpace definition in pcs format

    Returns:
        The deserialized ConfigurationSpace object
    """
    configuration_space = ConfigurationSpace()
    forbidden_to_add = []
    conditions_to_add = []

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
            "ordinal": OrdinalHyperparameter,
        }

        try:
            param_list = pp_cont_param.parseString(line)
            name = param_list[0]
            if param_list[1] == "integer":
                paramtype = "int"
                value_cast = int
            elif param_list[1] == "real":
                paramtype = "float"
                value_cast = float
            else:
                paramtype = None
                value_cast = lambda x: x

            if paramtype in ["int", "float"]:
                log = param_list[10:]
                param_list = param_list[:10]
                if len(log) > 0:
                    log = log[0]
                lower = value_cast(param_list[3])  # type: ignore
                upper = value_cast(param_list[5])  # type: ignore
                log_on = "log" in log
                default_value = value_cast(param_list[8])  # type: ignore
                param = create[paramtype](
                    name=name,
                    lower=lower,
                    upper=upper,
                    log=log_on,
                    default_value=default_value,
                )
                cont_ct += 1

        except pyparsing.ParseException:
            pass

        try:
            if "categorical" in line:
                param_list = pp_cat_param.parseString(line)
                name = param_list[0]
                choices = list(param_list[3:-4:2])
                default_value = param_list[-2]
                param = create["categorical"](
                    name=name,
                    choices=choices,
                    default_value=default_value,
                )
                cat_ct += 1

            elif "ordinal" in line:
                param_list = pp_ord_param.parseString(line)
                name = param_list[0]
                sequence = list(param_list[3:-4:2])
                default_value = param_list[-2]
                param = create["ordinal"](
                    name=name,
                    sequence=sequence,
                    default_value=default_value,
                )
                ord_ct += 1

        except pyparsing.ParseException:
            pass

        if param is None:
            raise NotImplementedError(f"Could not parse: {line}")

        configuration_space.add(param)

    for clause in forbidden:
        param_list = pp_forbidden_clause.parseString(clause)
        tmp_list: list = []
        clause_list = []
        for value in param_list[1:]:
            if len(tmp_list) < 3:
                tmp_list.append(value)
            else:
                # So far, only equals is supported by SMAC
                if tmp_list[1] == "=":
                    hp = configuration_space[tmp_list[0]]
                    if isinstance(hp, NumericalHyperparameter):
                        forbidden_value: float | int
                        if isinstance(hp, IntegerHyperparameter):
                            forbidden_value = int(tmp_list[2])
                        elif isinstance(hp, FloatHyperparameter):
                            forbidden_value = float(tmp_list[2])
                        else:
                            raise NotImplementedError

                        if forbidden_value < hp.lower or forbidden_value > hp.upper:
                            raise ValueError(
                                f"forbidden_value is set out of the bound, it needs to"
                                f" be set between [{hp.lower}, {hp.upper}]"
                                f" but its value is {forbidden_value}",
                            )

                    elif isinstance(
                        hp,
                        (CategoricalHyperparameter, OrdinalHyperparameter),
                    ):
                        hp_values = (
                            hp.choices
                            if isinstance(hp, CategoricalHyperparameter)
                            else hp.sequence
                        )
                        forbidden_value_in_hp_values = tmp_list[2] in hp_values

                        if forbidden_value_in_hp_values:
                            forbidden_value = tmp_list[2]
                        else:
                            raise ValueError(
                                f"forbidden_value is set out of the allowed value "
                                f"sets, it needs to be one member from {hp_values} "
                                f"but its value is {tmp_list[2]}",
                            )
                    else:
                        raise ValueError("Unsupported Hyperparamter sorts")

                    clause_list.append(
                        ForbiddenEqualsClause(
                            configuration_space[tmp_list[0]],
                            forbidden_value,
                        ),
                    )
                else:
                    raise NotImplementedError()
                tmp_list = []
        forbidden_to_add.append(ForbiddenAndConjunction(*clause_list))

    configuration_space.add(forbidden_to_add)

    conditions_per_child: dict = OrderedDict()

    for condition in conditions:
        child_name = condition[0]
        if child_name not in conditions_per_child:
            conditions_per_child[child_name] = []
        conditions_per_child[child_name].append(condition)

    for child_name in conditions_per_child:
        for condition in conditions_per_child[child_name]:
            condition = condition[2:]
            condition = " ".join(condition)  # type: ignore
            if "||" in str(condition):
                ors = []
                # 1st case we have a mixture of || and &&
                if "&&" in str(condition):
                    ors_combis = []
                    for cond_parts in str(condition).split("||"):
                        condition = str(cond_parts).split("&&")  # type: ignore
                        # if length is 1 it must be or
                        if len(condition) == 1:
                            element_list = condition[0].split()
                            ors_combis.append(
                                condition_specification(
                                    child_name,
                                    element_list,
                                    configuration_space,
                                ),
                            )
                        else:
                            # now taking care of ands
                            ands = []
                            for and_part in condition:
                                element_list = [
                                    element
                                    for _ in condition
                                    for element in and_part.split()
                                ]
                                ands.append(
                                    condition_specification(
                                        child_name,
                                        element_list,
                                        configuration_space,
                                    ),
                                )
                            ors_combis.append(AndConjunction(*ands))
                    mixed_conjunction = OrConjunction(*ors_combis)
                    conditions_to_add.append(mixed_conjunction)
                else:
                    # 2nd case: we only have ors
                    for cond_parts in str(condition).split("||"):
                        element_list = list(cond_parts.split())
                        ors.append(
                            condition_specification(
                                child_name,
                                element_list,
                                configuration_space,
                            ),
                        )
                    or_conjunction = OrConjunction(*ors)
                    conditions_to_add.append(or_conjunction)

            # 3rd case: we only have ands
            elif "&&" in str(condition):
                ands = []
                for cond_parts in str(condition).split("&&"):
                    element_list = list(cond_parts.split())
                    ands.append(
                        condition_specification(
                            child_name,
                            element_list,
                            configuration_space,
                        ),
                    )
                and_conjunction = AndConjunction(*ands)
                conditions_to_add.append(and_conjunction)

            # 4th case: we have a normal condition
            else:
                element_list = list(condition.split())
                normal_condition = condition_specification(
                    child_name,
                    element_list,
                    configuration_space,
                )
                conditions_to_add.append(normal_condition)

    configuration_space.add(conditions_to_add)

    return configuration_space


@deprecated(
    "pcs_new.write is has stopped being maintained, please use `space.to_json`"
    " or `space.to_yaml` instead",
)
def write(configuration_space: ConfigurationSpace) -> str:
    """Create a string representation of a
    [ConfigurationSpace][ConfigSpace.configuration_space.ConfigurationSpace]
    in pcs_new format. This string can be written to file.

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.read_and_write import pcs_new
    cs = ConfigurationSpace({"a": [1,2,3]})

    with open('configspace.pcs_new', 'w') as fh:
        fh.write(pcs_new.write(cs))
    ```

    Args:
        configuration_space: A configuration space

    Returns:
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
        elif isinstance(hyperparameter, OrdinalHyperparameter):
            param_lines.write(build_ordinal(hyperparameter))
        elif isinstance(hyperparameter, Constant):
            param_lines.write(build_constant(hyperparameter))
        else:
            raise TypeError(f"Unknown type: {type(hyperparameter)} ({hyperparameter})")

    for condition in configuration_space.conditions:
        if condition_lines.tell() > 0:
            condition_lines.write("\n")
        if isinstance(condition, (AndConjunction, OrConjunction)):
            condition_lines.write(build_conjunction(condition))
        elif isinstance(condition, Condition):
            condition_lines.write(build_condition(condition))
        else:
            raise TypeError(f"Unknown type: {type(condition)} ({condition})")

    for forbidden_clause in configuration_space.forbidden_clauses:
        # Convert in-statement into two or more equals statements
        dlcs = (
            forbidden_clause.get_descendant_literal_clauses()
            if isinstance(forbidden_clause, ForbiddenConjunction)
            else [forbidden_clause]
        )
        # First, get all in statements and convert them to equal statements
        in_statements = []
        other_statements = []
        for dlc in dlcs:
            if isinstance(dlc, ForbiddenInClause):
                in_statements.append(
                    [
                        ForbiddenEqualsClause(dlc.hyperparameter, value)
                        for value in dlc.values
                    ],
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
            if isinstance(forbidden_clause, ForbiddenRelation):
                raise TypeError("ForbiddenRelation is not supported")

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
