#!/usr/bin/env python
from __future__ import annotations

import json
import warnings
from typing import Any, Callable, Hashable, Mapping, TypeAlias

from ConfigSpace import __version__
from ConfigSpace.conditions import (
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
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenEqualsRelation,
    ForbiddenGreaterThanRelation,
    ForbiddenInClause,
    ForbiddenLessThanRelation,
)
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

JSON_FORMAT_VERSION = 0.4

_Decoder: TypeAlias = Callable[
    [dict[str, Any], ConfigurationSpace, "_DecoderLookup"],
    Any,
]
_DecoderLookup: TypeAlias = Mapping[Hashable, _Decoder]
_Encoder: TypeAlias = Callable[[Any, "_EncoderLookup"], dict[str, Any]]

_EncoderLookup: TypeAlias = Mapping[type, tuple[str | tuple[str, str], _Encoder]]


def _pop_q(item: dict[str, Any]) -> dict[str, Any]:
    if item.pop("q", None) is not None:
        warnings.warn(
            "The field 'q' was removed! Please update your json file as necessary!"
            f"\nFound in item {item}",
            stacklevel=3,
        )
    return item


def _decode_uniform_float(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decoders: _DecoderLookup,  # noqa: ARG001
) -> UniformFloatHyperparameter:
    item = _pop_q(item)
    return UniformFloatHyperparameter(**item)


def _decode_uniform_int(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decoders: _DecoderLookup,  # noqa: ARG001
) -> UniformIntegerHyperparameter:
    item = _pop_q(item)
    return UniformIntegerHyperparameter(**item)


def _decode_normal_int(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decoders: _DecoderLookup,  # noqa: ARG001
) -> NormalIntegerHyperparameter:
    item = _pop_q(item)
    return NormalIntegerHyperparameter(**item)


def _decode_normal_float(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decoders: _DecoderLookup,  # noqa: ARG001
) -> NormalFloatHyperparameter:
    item = _pop_q(item)
    return NormalFloatHyperparameter(**item)


def _decode_beta_int(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decoders: _DecoderLookup,  # noqa: ARG001
) -> BetaIntegerHyperparameter:
    item = _pop_q(item)
    return BetaIntegerHyperparameter(**item)


def _decode_beta_float(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decoders: _DecoderLookup,  # noqa: ARG001
) -> BetaFloatHyperparameter:
    item = _pop_q(item)
    return BetaFloatHyperparameter(**item)


def _decode_categorical(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decoders: _DecoderLookup,  # noqa: ARG001
) -> CategoricalHyperparameter:
    return CategoricalHyperparameter(**item)


def _decode_ordinal(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decoders: _DecoderLookup,  # noqa: ARG001
) -> OrdinalHyperparameter:
    return OrdinalHyperparameter(**item)


def _decode_constant(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decoders: _DecoderLookup,  # noqa: ARG001
) -> Constant:
    return Constant(**item)


def _decode_equals_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> EqualsCondition:
    return EqualsCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        value=item["value"],
    )


def _decode_not_equals_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> NotEqualsCondition:
    return NotEqualsCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        value=item["value"],
    )


def _decode_less_than_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> LessThanCondition:
    return LessThanCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        value=item["value"],
    )


def _decode_greater_than_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> GreaterThanCondition:
    return GreaterThanCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        value=item["value"],
    )


def _decode_in_condition_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> InCondition:
    return InCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        values=item["values"],
    )


def _decode_and_conjunction(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,
) -> AndConjunction:
    return AndConjunction(
        *[_decode_item(cond, cs, decoders=decoders) for cond in item["conditions"]],
    )


def _decode_or_conjunction(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,
) -> OrConjunction:
    return OrConjunction(
        *[_decode_item(cond, cs, decoders=decoders) for cond in item["conditions"]],
    )


def _decode_forbidden_in(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> ForbiddenInClause:
    return ForbiddenInClause(hyperparameter=cs[item["name"]], values=item["values"])


def _decode_forbidden_equal(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> ForbiddenEqualsClause:
    return ForbiddenEqualsClause(hyperparameter=cs[item["name"]], value=item["value"])


def _decode_forbidden_and(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,
) -> ForbiddenAndConjunction:
    return ForbiddenAndConjunction(
        *[_decode_item(cl, cs, decoders=decoders) for cl in item["clauses"]],
    )


def _decode_forbidden_equals_relation(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> ForbiddenEqualsRelation:
    return ForbiddenEqualsRelation(left=cs[item["left"]], right=cs[item["right"]])


def _decode_forbidden_less_than_relation(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> ForbiddenLessThanRelation:
    return ForbiddenLessThanRelation(left=cs[item["left"]], right=cs[item["right"]])


def _decode_forbidden_greater_than_relation(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decoders: _DecoderLookup,  # noqa: ARG001
) -> ForbiddenGreaterThanRelation:
    return ForbiddenGreaterThanRelation(left=cs[item["left"]], right=cs[item["right"]])


HYPERPARAMETER_DECODERS: _DecoderLookup = {
    "uniform_float": _decode_uniform_float,
    "uniform_int": _decode_uniform_int,
    "normal_int": _decode_normal_int,
    "normal_float": _decode_normal_float,
    "beta_int": _decode_beta_int,
    "beta_float": _decode_beta_float,
    "categorical": _decode_categorical,
    "ordinal": _decode_ordinal,
    "constant": _decode_constant,
}
CONDITION_DECODERS: _DecoderLookup = {
    "EQ": _decode_equals_condition,
    "NEQ": _decode_not_equals_condition,
    "LT": _decode_less_than_condition,
    "GT": _decode_greater_than_condition,
    "IN": _decode_in_condition_condition,
    "AND": _decode_and_conjunction,
    "OR": _decode_or_conjunction,
}
FORBIDDEN_DECODERS: _DecoderLookup = {
    "EQUALS": _decode_forbidden_equal,
    "IN": _decode_forbidden_in,
    "AND": _decode_forbidden_and,
    ("RELATION", "LESS"): _decode_forbidden_less_than_relation,
    ("RELATION", "EQUALS"): _decode_forbidden_equals_relation,
    ("RELATION", "GREATER"): _decode_forbidden_greater_than_relation,
}


def _decode_item(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    *,
    decoders: _DecoderLookup,
) -> Any:
    _type = item.pop("type", None)
    if _type is None:
        raise KeyError(
            f"Expected a key 'type' in item {item} but did not find it."
            " Did you include this in the encoding?",
        )

    # NOTE: Previous iterations basically put the type of ForbiddenRelation
    # into two fields, "type" and "lambda", hence this check here.
    key: Hashable
    if _type != "RELATION":
        key = _type
    else:
        _lambda = item.pop("lambda", None)
        if _lambda is None:
            raise KeyError(
                f"Expected a key 'lambda' in ForbiddenRelation of type {_type}"
                f" in item {item} but did not find it. Did you include this"
                " in the encoding?",
            )
        _type = (_type, _lambda)

    decoder = decoders.get(key)
    if decoder is None:
        raise ValueError(
            f"No found decoder for '{key}'.  Registered decoders are"
            f" {decoders.keys()}. Please include a custom `decoder=` if"
            " you want to decode this type.",
        )

    return decoder(item, cs, decoders)


def _encode_uniform_float(
    hp: UniformFloatHyperparameter,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "name": hp.name,
        "lower": float(hp.lower),
        "upper": float(hp.upper),
        "default_value": float(hp.default_value),
        "log": hp.log,
        "meta": hp.meta,
    }


def _encode_uniform_int(
    hp: UniformIntegerHyperparameter,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "name": hp.name,
        "lower": int(hp.lower),
        "upper": int(hp.upper),
        "default_value": int(hp.default_value),
        "log": hp.log,
        "meta": hp.meta,
    }


def _encode_normal_int(
    hp: NormalIntegerHyperparameter,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "name": hp.name,
        "mu": float(hp.mu),
        "sigma": float(hp.sigma),
        "lower": int(hp.lower),
        "upper": int(hp.upper),
        "default_value": int(hp.default_value),
        "log": hp.log,
        "meta": hp.meta,
    }


def _encode_normal_float(
    hp: NormalFloatHyperparameter,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "name": hp.name,
        "mu": float(hp.mu),
        "sigma": float(hp.sigma),
        "lower": float(hp.lower),
        "upper": float(hp.upper),
        "default_value": float(hp.default_value),
        "log": hp.log,
        "meta": hp.meta,
    }


def _encode_beta_int(
    hp: BetaIntegerHyperparameter,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "name": hp.name,
        "alpha": float(hp.alpha),
        "beta": float(hp.beta),
        "lower": int(hp.lower),
        "upper": int(hp.upper),
        "default_value": int(hp.default_value),
        "log": hp.log,
        "meta": hp.meta,
    }


def _encode_beta_float(
    hp: BetaFloatHyperparameter,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "name": hp.name,
        "alpha": float(hp.alpha),
        "beta": float(hp.beta),
        "lower": float(hp.lower),
        "upper": float(hp.upper),
        "default_value": float(hp.default_value),
        "log": hp.log,
        "meta": hp.meta,
    }


def _encode_categorical(
    hp: CategoricalHyperparameter,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "name": hp.name,
        "choices": list(hp.choices),
        "weights": list(hp.weights) if hp.weights else None,
        "default_value": hp.default_value,
        "meta": hp.meta,
    }


def _encode_ordinal(
    hp: OrdinalHyperparameter,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "name": hp.name,
        "sequence": list(hp.sequence),
        "default_value": hp.default_value,
        "meta": hp.meta,
    }


def _encode_constant(hp: Constant, encoders: _EncoderLookup) -> dict[str, Any]:  # noqa: ARG001
    return {"name": hp.name, "value": hp.value, "meta": hp.meta}


def _encode_equals_condition(
    cond: EqualsCondition,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "parent": cond.parent.name,
        "value": cond.value,
    }


def _encode_not_equals_condition(
    cond: NotEqualsCondition,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "parent": cond.parent.name,
        "value": cond.value,
    }


def _encode_less_than_condition(
    cond: LessThanCondition,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "parent": cond.parent.name,
        "value": cond.value,
    }


def _encode_greater_than_condition(
    cond: GreaterThanCondition,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "parent": cond.parent.name,
        "value": cond.value,
    }


def _encode_in_condition(cond: InCondition, encoders: _EncoderLookup) -> dict[str, Any]:  # noqa: ARG001
    return {
        "child": cond.child.name,
        "parent": cond.parent.name,
        "values": cond.values,
    }


def _encode_and_conjuction(
    cond: AndConjunction,
    encoders: _EncoderLookup,
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "conditions": [_encode_item(c, encoders) for c in cond.components],
    }


def _encode_or_conjuction(
    cond: AndConjunction,
    encoders: _EncoderLookup,
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "conditions": [_encode_item(c, encoders) for c in cond.components],
    }


def _encode_forbidden_equals(
    cond: ForbiddenEqualsClause,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {"name": cond.hyperparameter.name, "value": cond.value}


def _encode_forbidden_in(
    cond: ForbiddenInClause,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {"name": cond.hyperparameter.name, "values": cond.values}


def _encode_forbidden_and(
    cond: ForbiddenAndConjunction,
    encoders: _EncoderLookup,
) -> dict[str, Any]:
    return {"clauses": [_encode_item(c, encoders) for c in cond.components]}


def _encoder_forbidden_less_than(
    cond: ForbiddenLessThanRelation,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {"left": cond.left.name, "right": cond.right.name}


def _encoder_forbidden_equals(
    cond: ForbiddenEqualsRelation,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {"left": cond.left.name, "right": cond.right.name}


def _encoder_forbidden_greater_than(
    cond: ForbiddenGreaterThanRelation,
    encoders: _EncoderLookup,  # noqa: ARG001
) -> dict[str, Any]:
    return {"left": cond.left.name, "right": cond.right.name}


HYPERPARAMETER_ENCODERS: _EncoderLookup = {
    UniformFloatHyperparameter: ("uniform_float", _encode_uniform_float),
    UniformIntegerHyperparameter: ("uniform_int", _encode_uniform_int),
    NormalFloatHyperparameter: ("normal_float", _encode_normal_float),
    NormalIntegerHyperparameter: ("normal_int", _encode_normal_int),
    BetaIntegerHyperparameter: ("beta_int", _encode_beta_int),
    BetaFloatHyperparameter: ("beta_float", _encode_beta_float),
    CategoricalHyperparameter: ("categorical", _encode_categorical),
    OrdinalHyperparameter: ("ordinal", _encode_ordinal),
    Constant: ("constant", _encode_constant),
}
CONDITION_ENCODERS: _EncoderLookup = {
    EqualsCondition: ("EQ", _encode_equals_condition),
    NotEqualsCondition: ("NEQ", _encode_not_equals_condition),
    LessThanCondition: ("LT", _encode_less_than_condition),
    GreaterThanCondition: ("GT", _encode_greater_than_condition),
    InCondition: ("IN", _encode_in_condition),
    AndConjunction: ("AND", _encode_and_conjuction),
    OrConjunction: ("OR", _encode_or_conjuction),
}

# NOTE: The two part for relations is due to a legacy issue
FORBIDDEN_ENCODERS: _EncoderLookup = {
    ForbiddenEqualsClause: ("EQUALS", _encode_forbidden_equals),
    ForbiddenInClause: ("IN", _encode_forbidden_in),
    ForbiddenAndConjunction: ("AND", _encode_forbidden_and),
    ForbiddenLessThanRelation: (("RELATION", "LESS"), _encoder_forbidden_less_than),
    ForbiddenEqualsRelation: (("RELATION", "EQUALS"), _encoder_forbidden_equals),
    ForbiddenGreaterThanRelation: (
        ("RELATION", "GREATER"),
        _encoder_forbidden_greater_than,
    ),
}


def _encode_item(item: Any, encoders: _EncoderLookup) -> Mapping[str, Any]:
    key = type(item)
    res = encoders.get(key)
    if res is None:
        raise ValueError(
            f"No found encoder for '{key}'. Registered encoders are"
            f" {encoders.keys()}. Please include a custom `encoder=` if"
            " you want to encode this type.",
        )

    type_name, encoder = res
    encoding = encoder(item, encoders)
    if isinstance(type_name, tuple):
        # NOTE: This is due to legacy where Forbidden's are delcared using two keys
        encoding.update({"type": type_name[0], "lambda": type_name[1]})
    else:
        encoding.update({"type": type_name})

    try:
        json.dumps(encoding)
    except TypeError as e:
        clsname = item.__class__.__name__
        raise TypeError(
            f"`{clsname}` is not serializable to json with the"
            f" dictionary {encoding}.\n{item}",
        ) from e

    return encoding


################################################################################
def write(
    configuration_space: ConfigurationSpace,
    *,
    indent: int = 2,
    encoders: _EncoderLookup | None = None,
) -> str:
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
    encoders: dict[type, tuple[str, Callable[[Any, encoders], dict]]]
        Additional encoders to include where they key is a type to which the encoder
        applies to and the value is a tuple, where the first element is the type name
        to include in the dictionary and the second element is the encoder function which
        gives back a serializable dictionary.

    Returns
    -------
    str
        String representation of the configuration space,
        which can be written to file
    """
    if not isinstance(configuration_space, ConfigurationSpace):
        raise TypeError(
            f"pcs_parser.write expects an instance of {ConfigurationSpace}, "
            f"you provided '{type(configuration_space)}'",
        )

    user_encoders = encoders or {}

    json_dict: dict[str, Any] = {
        "name": configuration_space.name,
        "hyperparameters": [
            _encode_item(hp, {**HYPERPARAMETER_ENCODERS, **user_encoders})
            for hp in configuration_space.values()
        ],
        "conditions": [
            _encode_item(c, {**CONDITION_ENCODERS, **user_encoders})
            for c in configuration_space.conditions
        ],
        "forbiddens": [
            _encode_item(f, {**FORBIDDEN_ENCODERS, **user_encoders})
            for f in configuration_space.forbidden_clauses
        ],
        "python_module_version": __version__,
        "json_format_version": JSON_FORMAT_VERSION,
    }
    return json.dumps(json_dict, indent=indent)


################################################################################
def read(
    jason_string: str,
    decoders: _DecoderLookup | None = None,
) -> ConfigurationSpace:
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

    Returns
    -------
    :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The deserialized ConfigurationSpace object
    """
    jason: dict[str, Any] = json.loads(jason_string)
    user_decoders = decoders or {}
    space = ConfigurationSpace(name=jason.get("name"))

    hyperparameters_json = jason.get("hyperparameters", [])
    conditions_json = jason.get("conditions", [])
    forbiddens_json = jason.get("forbiddens", [])

    hyperparameters = [
        _decode_item(hp, space, decoders={**HYPERPARAMETER_DECODERS, **user_decoders})
        for hp in hyperparameters_json
    ]
    space.add(hyperparameters)

    conditions = [
        _decode_item(c, space, decoders={**CONDITION_DECODERS, **user_decoders})
        for c in conditions_json
    ]
    forbidden = [
        _decode_item(f, space, decoders={**FORBIDDEN_DECODERS, **user_decoders})
        for f in forbiddens_json
    ]
    space.add(conditions, forbidden)
    return space
