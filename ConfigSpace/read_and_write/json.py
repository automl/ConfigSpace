#!/usr/bin/env python
from __future__ import annotations

import json

from ConfigSpace import __version__
from ConfigSpace.conditions import (
    AbstractCondition,
    AndConjunction,
    EqualsCondition,
    GreaterThanCondition,
    InCondition,
    LessThanCondition,
    NotEqualsCondition,
    OrConjunction,
)
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import (
    AbstractForbiddenComponent,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenEqualsRelation,
    ForbiddenGreaterThanRelation,
    ForbiddenInClause,
    ForbiddenLessThanRelation,
    ForbiddenRelation,
)
from ConfigSpace.functional import walk_subclasses
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

JSON_FORMAT_VERSION = 0.4


################################################################################
# Builder for hyperparameters
def _build_constant(param: Constant) -> dict:
    return {
        "name": param.name,
        "type": "constant",
        "value": param.value,
    }


def _build_unparametrized_hyperparameter(param: UnParametrizedHyperparameter) -> dict:
    return {
        "name": param.name,
        "type": "unparametrized",
        "value": param.value,
    }


def _build_uniform_float(param: UniformFloatHyperparameter) -> dict:
    return {
        "name": param.name,
        "type": "uniform_float",
        "log": param.log,
        "lower": param.lower,
        "upper": param.upper,
        "default": param.default_value,
    }


def _build_normal_float(param: NormalFloatHyperparameter) -> dict:
    return {
        "name": param.name,
        "type": "normal_float",
        "log": param.log,
        "mu": param.mu,
        "sigma": param.sigma,
        "default": param.default_value,
        "lower": param.lower,
        "upper": param.upper,
    }


def _build_beta_float(param: BetaFloatHyperparameter) -> dict:
    return {
        "name": param.name,
        "type": "beta_float",
        "log": param.log,
        "alpha": param.alpha,
        "beta": param.beta,
        "lower": param.lower,
        "upper": param.upper,
        "default": param.default_value,
    }


def _build_uniform_int(param: UniformIntegerHyperparameter) -> dict:
    return {
        "name": param.name,
        "type": "uniform_int",
        "log": param.log,
        "lower": param.lower,
        "upper": param.upper,
        "default": param.default_value,
    }


def _build_normal_int(param: NormalIntegerHyperparameter) -> dict:
    return {
        "name": param.name,
        "type": "normal_int",
        "log": param.log,
        "mu": param.mu,
        "sigma": param.sigma,
        "lower": param.lower,
        "upper": param.upper,
        "default": param.default_value,
    }


def _build_beta_int(param: BetaIntegerHyperparameter) -> dict:
    return {
        "name": param.name,
        "type": "beta_int",
        "log": param.log,
        "alpha": param.alpha,
        "beta": param.beta,
        "lower": param.lower,
        "upper": param.upper,
        "default": param.default_value,
    }


def _build_categorical(param: CategoricalHyperparameter) -> dict:
    return {
        "name": param.name,
        "type": "categorical",
        "choices": param.choices,
        "default": param.default_value,
        "weights": param.weights,
    }


def _build_ordinal(param: OrdinalHyperparameter) -> dict:
    return {
        "name": param.name,
        "type": "ordinal",
        "sequence": param.sequence,
        "default": param.default_value,
    }


################################################################################
# Builder for Conditions
def _build_condition(condition: AbstractCondition) -> dict:
    methods = {
        AndConjunction: _build_and_conjunction,
        OrConjunction: _build_or_conjunction,
        InCondition: _build_in_condition,
        EqualsCondition: _build_equals_condition,
        NotEqualsCondition: _build_not_equals_condition,
        GreaterThanCondition: _build_greater_than_condition,
        LessThanCondition: _build_less_than_condition,
    }
    return methods[type(condition)](condition)


def _build_and_conjunction(conjunction: AndConjunction) -> dict:
    child = conjunction.get_descendant_literal_conditions()[0].child.name
    cond_list = []
    for component in conjunction.components:
        cond_list.append(_build_condition(component))
    return {
        "child": child,
        "type": "AND",
        "conditions": cond_list,
    }


def _build_or_conjunction(conjunction: OrConjunction) -> dict:
    child = conjunction.get_descendant_literal_conditions()[0].child.name
    cond_list = []
    for component in conjunction.components:
        cond_list.append(_build_condition(component))
    return {
        "child": child,
        "type": "OR",
        "conditions": cond_list,
    }


def _build_in_condition(condition: InCondition) -> dict:
    child = condition.child.name
    parent = condition.parent.name
    values = list(condition.values)
    return {
        "child": child,
        "parent": parent,
        "type": "IN",
        "values": values,
    }


def _build_equals_condition(condition: EqualsCondition) -> dict:
    child = condition.child.name
    parent = condition.parent.name
    value = condition.value
    return {
        "child": child,
        "parent": parent,
        "type": "EQ",
        "value": value,
    }


def _build_not_equals_condition(condition: NotEqualsCondition) -> dict:
    child = condition.child.name
    parent = condition.parent.name
    value = condition.value
    return {
        "child": child,
        "parent": parent,
        "type": "NEQ",
        "value": value,
    }


def _build_greater_than_condition(condition: GreaterThanCondition) -> dict:
    child = condition.child.name
    parent = condition.parent.name
    value = condition.value
    return {
        "child": child,
        "parent": parent,
        "type": "GT",
        "value": value,
    }


def _build_less_than_condition(condition: LessThanCondition) -> dict:
    child = condition.child.name
    parent = condition.parent.name
    value = condition.value
    return {
        "child": child,
        "parent": parent,
        "type": "LT",
        "value": value,
    }


################################################################################
# Builder for forbidden
def _build_forbidden(clause: AbstractForbiddenComponent) -> dict:
    methods = {
        ForbiddenEqualsClause: _build_forbidden_equals_clause,
        ForbiddenInClause: _build_forbidden_in_clause,
        ForbiddenAndConjunction: _build_forbidden_and_conjunction,
        ForbiddenEqualsRelation: _build_forbidden_relation,
        ForbiddenLessThanRelation: _build_forbidden_relation,
        ForbiddenGreaterThanRelation: _build_forbidden_relation,
    }
    return methods[type(clause)](clause)


def _build_forbidden_equals_clause(clause: ForbiddenEqualsClause) -> dict:
    return {
        "name": clause.hyperparameter.name,
        "type": "EQUALS",
        "value": clause.value,
    }


def _build_forbidden_in_clause(clause: ForbiddenInClause) -> dict:
    return {
        "name": clause.hyperparameter.name,
        "type": "IN",
        # The values are a set, but a set cannot be serialized to json
        "values": list(clause.values),
    }


def _build_forbidden_and_conjunction(clause: ForbiddenAndConjunction) -> dict:
    return {
        "name": clause.get_descendant_literal_clauses()[0].hyperparameter.name,
        "type": "AND",
        "clauses": [_build_forbidden(component) for component in clause.components],
    }


def _build_forbidden_relation(clause: ForbiddenRelation) -> dict:
    if isinstance(clause, ForbiddenLessThanRelation):
        lambda_ = "LESS"
    elif isinstance(clause, ForbiddenEqualsRelation):
        lambda_ = "EQUALS"
    elif isinstance(clause, ForbiddenGreaterThanRelation):
        lambda_ = "GREATER"
    else:
        raise ValueError("Unknown relation '%s'" % type(clause))

    return {
        "left": clause.left.name,
        "right": clause.right.name,
        "type": "RELATION",
        "lambda": lambda_,
    }


################################################################################
def write(configuration_space: ConfigurationSpace, indent: int = 2) -> str:
    """Create a string representation of a
    :class:`~ConfigSpace.configuration_space.ConfigurationSpace` in json format.
    This string can be written to file.

    .. code:: python

        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.read_and_write import json as cs_json

        cs = ConfigurationSpace({"a": [1, 2, 3]})

        with open('configspace.json', 'w') as f:
            f.write(cs_json.write(cs))

    Parameters
    ----------
    configuration_space : :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        a configuration space, which should be written to file.
    indent : int
        number of whitespaces to use as indent

    Returns:
    -------
    str
        String representation of the configuration space,
        which will be written to file
    """
    if not isinstance(configuration_space, ConfigurationSpace):
        raise TypeError(
            f"pcs_parser.write expects an instance of {ConfigurationSpace}, "
            f"you provided '{type(configuration_space)}'",
        )

    conditions = []
    forbiddens = []

    # Sanity check to make debugging easier
    for hyperparameter in configuration_space.values():
        try:
            json.dumps(hyperparameter.to_dict())
        except TypeError as e:
            raise TypeError(
                f"Hyperparameter {hyperparameter.name} is not serializable to json",
                f" with the dictionary {hyperparameter.to_dict()}.",
            ) from e

    hyperparameters = [hp.to_dict() for hp in configuration_space.values()]

    for condition in configuration_space.get_conditions():
        conditions.append(_build_condition(condition))

    for forbidden_clause in configuration_space.get_forbiddens():
        forbiddens.append(_build_forbidden(forbidden_clause))

    rval: dict = {}
    if configuration_space.name is not None:
        rval["name"] = configuration_space.name
    rval["hyperparameters"] = hyperparameters
    rval["conditions"] = conditions
    rval["forbiddens"] = forbiddens
    rval["python_module_version"] = __version__
    rval["json_format_version"] = JSON_FORMAT_VERSION

    return json.dumps(rval, indent=indent)


################################################################################
def read(jason_string: str) -> ConfigurationSpace:
    """Create a configuration space definition from a json string.

    .. code:: python

        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.read_and_write import json as cs_json

        cs = ConfigurationSpace({"a": [1, 2, 3]})

        cs_string = cs_json.write(cs)
        with open('configspace.json', 'w') as f:
             f.write(cs_string)

        with open('configspace.json', 'r') as f:
            json_string = f.read()
            config = cs_json.read(json_string)


    Parameters
    ----------
    jason_string : str
        A json string representing a configuration space definition

    Returns:
    -------
    :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The deserialized ConfigurationSpace object
    """
    jason = json.loads(jason_string)
    if "name" in jason:
        configuration_space = ConfigurationSpace(name=jason["name"])
    else:
        configuration_space = ConfigurationSpace()

    hyperparameters: list[Hyperparameter] = []

    hp_classes = {
        serializable_type_name: hp
        for hp in walk_subclasses(Hyperparameter)
        if (serializable_type_name := getattr(hp, "serializable_type_name", None))
        is not None
    }
    for hyperparameter in jason["hyperparameters"]:
        hp_type = hyperparameter["type"]
        hp_class = hp_classes.get(hp_type)
        if hp_class is None:
            raise ValueError(
                f"Unknown hyperparameter type '{hp_type}'. Registered hyperparameters"
                f" are {hp_classes.keys()}",
            )

        hp = hp_class.from_dict(hyperparameter)
        hyperparameters.append(hp)

    configuration_space.add_hyperparameters(hyperparameters)

    # TODO: Can add multiple at a time?
    conditionals = []
    for condition in jason["conditions"]:
        conditionals.append(_construct_condition(condition, configuration_space))

    configuration_space.add_conditions(conditionals)

    forbiddens = []
    for forbidden in jason["forbiddens"]:
        forbiddens.append(_construct_forbidden(forbidden, configuration_space))

    configuration_space.add_forbidden_clauses(forbiddens)

    return configuration_space


def _construct_condition(
    condition: dict,
    cs: ConfigurationSpace,
) -> AbstractCondition:
    condition_type = condition["type"]
    methods = {
        "AND": _construct_and_condition,
        "OR": _construct_or_condition,
        "IN": _construct_in_condition,
        "EQ": _construct_eq_condition,
        "NEQ": _construct_neq_condition,
        "GT": _construct_gt_condition,
        "LT": _construct_lt_condition,
    }
    return methods[condition_type](condition, cs)


def _construct_and_condition(
    condition: dict,
    cs: ConfigurationSpace,
) -> AndConjunction:
    conditions = [_construct_condition(cond, cs) for cond in condition["conditions"]]
    return AndConjunction(*conditions)


def _construct_or_condition(
    condition: dict,
    cs: ConfigurationSpace,
) -> OrConjunction:
    conditions = [_construct_condition(cond, cs) for cond in condition["conditions"]]
    return OrConjunction(*conditions)


def _construct_in_condition(
    condition: dict,
    cs: ConfigurationSpace,
) -> InCondition:
    return InCondition(
        child=cs[condition["child"]],
        parent=cs[condition["parent"]],
        values=condition["values"],
    )


def _construct_eq_condition(
    condition: dict,
    cs: ConfigurationSpace,
) -> EqualsCondition:
    return EqualsCondition(
        child=cs[condition["child"]],
        parent=cs[condition["parent"]],
        value=condition["value"],
    )


def _construct_neq_condition(
    condition: dict,
    cs: ConfigurationSpace,
) -> NotEqualsCondition:
    return NotEqualsCondition(
        child=cs[condition["child"]],
        parent=cs[condition["parent"]],
        value=condition["value"],
    )


def _construct_gt_condition(
    condition: dict,
    cs: ConfigurationSpace,
) -> GreaterThanCondition:
    return GreaterThanCondition(
        child=cs[condition["child"]],
        parent=cs[condition["parent"]],
        value=condition["value"],
    )


def _construct_lt_condition(
    condition: dict,
    cs: ConfigurationSpace,
) -> LessThanCondition:
    return LessThanCondition(
        child=cs[condition["child"]],
        parent=cs[condition["parent"]],
        value=condition["value"],
    )


def _construct_forbidden(
    clause: dict,
    cs: ConfigurationSpace,
) -> AbstractForbiddenComponent:
    forbidden_type = clause["type"]
    methods = {
        "EQUALS": _construct_forbidden_equals,
        "IN": _construct_forbidden_in,
        "AND": _construct_forbidden_and,
        "RELATION": _construct_forbidden_equals,
    }
    return methods[forbidden_type](clause, cs)


def _construct_forbidden_equals(
    clause: dict,
    cs: ConfigurationSpace,
) -> ForbiddenEqualsClause:
    return ForbiddenEqualsClause(
        hyperparameter=cs[clause["name"]],
        value=clause["value"],
    )


def _construct_forbidden_in(
    clause: dict,
    cs: ConfigurationSpace,
) -> ForbiddenEqualsClause:
    return ForbiddenInClause(hyperparameter=cs[clause["name"]], values=clause["values"])


def _construct_forbidden_and(
    clause: dict,
    cs: ConfigurationSpace,
) -> ForbiddenAndConjunction:
    clauses = [_construct_forbidden(cl, cs) for cl in clause["clauses"]]
    return ForbiddenAndConjunction(*clauses)


def _construct_forbidden_relation(  # pyright: ignore
    clause: dict,
    cs: ConfigurationSpace,
) -> ForbiddenRelation:
    left = cs[clause["left"]]
    right = cs[clause["right"]]

    if clause["lambda"] == "LESS":
        return ForbiddenLessThanRelation(left, right)

    if clause["lambda"] == "EQUALS":
        return ForbiddenEqualsRelation(left, right)

    if clause["lambda"] == "GREATER":
        return ForbiddenGreaterThanRelation(left, right)

    raise ValueError("Unknown relation '%s'" % clause["lambda"])
