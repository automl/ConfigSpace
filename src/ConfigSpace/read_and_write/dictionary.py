from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict
from typing_extensions import TypeAlias

from ConfigSpace.conditions import (
    AndConjunction,
    EqualsCondition,
    GreaterThanCondition,
    InCondition,
    LessThanCondition,
    NotEqualsCondition,
    OrConjunction,
)
from ConfigSpace.forbidden import (
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenEqualsRelation,
    ForbiddenGreaterThanRelation,
    ForbiddenInClause,
    ForbiddenLessThanRelation,
    ForbiddenRelation,
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

if TYPE_CHECKING:
    from ConfigSpace.configuration_space import ConfigurationSpace

    _Decoder: TypeAlias = Callable[
        [Dict[str, Any], ConfigurationSpace, "_Decoder"],
        Any,
    ]
    """Type alias for the decoder function signature."""

    _Encoder: TypeAlias = Callable[[Any, "_Encoder"], Dict[str, Any]]
    """Type alias for the encoder function signature."""


def _backwards_compat(item: dict[str, Any]) -> dict[str, Any]:
    if item.pop("q", None) is not None:
        warnings.warn(
            "The field 'q' was removed! Please update your serialized format as needed!"
            f"\nFound in item {item}",
            stacklevel=3,
        )
    if (default := item.pop("default", None)) is not None:
        warnings.warn(
            "The field 'default' should be 'default_value' !" f"\nFound in item {item}",
            stacklevel=3,
        )
        item["default_value"] = default

    return item


def _decode_uniform_float(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decode: _Decoder,  # noqa: ARG001
) -> UniformFloatHyperparameter:
    item = _backwards_compat(item)
    return UniformFloatHyperparameter(**item)


def _decode_uniform_int(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decode: _Decoder,  # noqa: ARG001
) -> UniformIntegerHyperparameter:
    item = _backwards_compat(item)
    return UniformIntegerHyperparameter(**item)


def _decode_normal_int(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decode: _Decoder,  # noqa: ARG001
) -> NormalIntegerHyperparameter:
    item = _backwards_compat(item)
    return NormalIntegerHyperparameter(**item)


def _decode_normal_float(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decode: _Decoder,  # noqa: ARG001
) -> NormalFloatHyperparameter:
    item = _backwards_compat(item)
    return NormalFloatHyperparameter(**item)


def _decode_beta_int(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decode: _Decoder,  # noqa: ARG001
) -> BetaIntegerHyperparameter:
    item = _backwards_compat(item)
    return BetaIntegerHyperparameter(**item)


def _decode_beta_float(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decode: _Decoder,  # noqa: ARG001
) -> BetaFloatHyperparameter:
    item = _backwards_compat(item)
    return BetaFloatHyperparameter(**item)


def _decode_categorical(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decode: _Decoder,  # noqa: ARG001
) -> CategoricalHyperparameter:
    item = _backwards_compat(item)
    if item.pop("probabilities", None) is not None:
        warnings.warn(
            "The field 'probabilities' was removed and is called 'weights'!"
            "\nPlease update your serialized format as needed!"
            f"\nFound in item {item}",
            stacklevel=3,
        )
    return CategoricalHyperparameter(**item)


def _decode_ordinal(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decode: _Decoder,  # noqa: ARG001
) -> OrdinalHyperparameter:
    item = _backwards_compat(item)
    return OrdinalHyperparameter(**item)


def _decode_constant(
    item: dict[str, Any],
    cs: ConfigurationSpace,  # noqa: ARG001
    decode: _Decoder,  # noqa: ARG001
) -> Constant:
    item = _backwards_compat(item)
    return Constant(**item)


def _decode_equals_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> EqualsCondition:
    return EqualsCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        value=item["value"],
    )


def _decode_not_equals_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> NotEqualsCondition:
    return NotEqualsCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        value=item["value"],
    )


def _decode_less_than_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> LessThanCondition:
    return LessThanCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        value=item["value"],
    )


def _decode_greater_than_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> GreaterThanCondition:
    return GreaterThanCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        value=item["value"],
    )


def _decode_in_condition_condition(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> InCondition:
    return InCondition(
        child=cs[item["child"]],
        parent=cs[item["parent"]],
        values=item["values"],
    )


def _decode_and_conjunction(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,
) -> AndConjunction:
    return AndConjunction(*[decode(cond, cs, decode) for cond in item["conditions"]])


def _decode_or_conjunction(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,
) -> OrConjunction:
    return OrConjunction(
        *[decode(cond, cs, decode) for cond in item["conditions"]],
    )


def _decode_forbidden_in(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> ForbiddenInClause:
    return ForbiddenInClause(hyperparameter=cs[item["name"]], values=item["values"])


def _decode_forbidden_equal(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> ForbiddenEqualsClause:
    return ForbiddenEqualsClause(hyperparameter=cs[item["name"]], value=item["value"])


def _decode_forbidden_and(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,
) -> ForbiddenAndConjunction:
    return ForbiddenAndConjunction(*[decode(cl, cs, decode) for cl in item["clauses"]])


def _decode_relation_legacy(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> ForbiddenRelation:
    _lambda = item["lambda"]
    if _lambda == "LT":
        return ForbiddenLessThanRelation(left=cs[item["left"]], right=cs[item["right"]])
    if _lambda == "EQ":
        return ForbiddenEqualsRelation(left=cs[item["left"]], right=cs[item["right"]])
    if _lambda == "GT":
        return ForbiddenGreaterThanRelation(
            left=cs[item["left"]],
            right=cs[item["right"]],
        )

    raise ValueError(f"Unknown lambda {_lambda}")


def _decode_forbidden_equals_relation(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> ForbiddenEqualsRelation:
    return ForbiddenEqualsRelation(left=cs[item["left"]], right=cs[item["right"]])


def _decode_forbidden_less_than_relation(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> ForbiddenLessThanRelation:
    return ForbiddenLessThanRelation(left=cs[item["left"]], right=cs[item["right"]])


def _decode_forbidden_greater_than_relation(
    item: dict[str, Any],
    cs: ConfigurationSpace,
    decode: _Decoder,  # noqa: ARG001
) -> ForbiddenGreaterThanRelation:
    return ForbiddenGreaterThanRelation(left=cs[item["left"]], right=cs[item["right"]])


HYPERPARAMETER_DECODERS: dict[str, _Decoder] = {
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
CONDITION_DECODERS: dict[str, _Decoder] = {
    "EQ": _decode_equals_condition,
    "NEQ": _decode_not_equals_condition,
    "LT": _decode_less_than_condition,
    "GT": _decode_greater_than_condition,
    "IN": _decode_in_condition_condition,
    "AND": _decode_and_conjunction,
    "OR": _decode_or_conjunction,
}

FORBIDDEN_DECODERS: dict[str, _Decoder] = {
    "EQUALS": _decode_forbidden_equal,
    "IN": _decode_forbidden_in,
    "AND": _decode_forbidden_and,
    "RELATION_LT": _decode_forbidden_less_than_relation,
    "RELATION_EQ": _decode_forbidden_equals_relation,
    "RELATION_GT": _decode_forbidden_greater_than_relation,
    "RELATION": _decode_relation_legacy,
}


def _encode_uniform_float(
    hp: UniformFloatHyperparameter,
    encode: _Encoder,  # noqa: ARG001
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
    encode: _Encoder,  # noqa: ARG001
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
    encode: _Encoder,  # noqa: ARG001
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
    encode: _Encoder,  # noqa: ARG001
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
    encode: _Encoder,  # noqa: ARG001
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
    encode: _Encoder,  # noqa: ARG001
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
    encode: _Encoder,  # noqa: ARG001
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
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "name": hp.name,
        "sequence": list(hp.sequence),
        "default_value": hp.default_value,
        "meta": hp.meta,
    }


def _encode_constant(
    hp: Constant,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {"name": hp.name, "value": hp.value, "meta": hp.meta}


def _encode_equals_condition(
    cond: EqualsCondition,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {"child": cond.child.name, "parent": cond.parent.name, "value": cond.value}


def _encode_not_equals_condition(
    cond: NotEqualsCondition,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {"child": cond.child.name, "parent": cond.parent.name, "value": cond.value}


def _encode_less_than_condition(
    cond: LessThanCondition,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "parent": cond.parent.name,
        "value": cond.value,
    }


def _encode_greater_than_condition(
    cond: GreaterThanCondition,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "parent": cond.parent.name,
        "value": cond.value,
    }


def _encode_in_condition(
    cond: InCondition,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "parent": cond.parent.name,
        "values": cond.values,
    }


def _encode_and_conjuction(
    cond: AndConjunction,
    encode: _Encoder,
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "conditions": [encode(c, encode) for c in cond.components],
    }


def _encode_or_conjuction(
    cond: AndConjunction,
    encode: _Encoder,
) -> dict[str, Any]:
    return {
        "child": cond.child.name,
        "conditions": [encode(c, encode) for c in cond.components],
    }


def _encode_forbidden_equals(
    cond: ForbiddenEqualsClause,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {"name": cond.hyperparameter.name, "value": cond.value}


def _encode_forbidden_in(
    cond: ForbiddenInClause,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {"name": cond.hyperparameter.name, "values": cond.values}


def _encode_forbidden_and(
    cond: ForbiddenAndConjunction,
    encode: _Encoder,
) -> dict[str, Any]:
    return {"clauses": [encode(c, encode) for c in cond.components]}


def _encode_forbidden_relation_less_than(
    cond: ForbiddenLessThanRelation,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {"left": cond.left.name, "right": cond.right.name}


def _encode_forbidden_relation_equals(
    cond: ForbiddenEqualsRelation,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {"left": cond.left.name, "right": cond.right.name}


def _encode_forbidden_relation_greater_than(
    cond: ForbiddenGreaterThanRelation,
    encode: _Encoder,  # noqa: ARG001
) -> dict[str, Any]:
    return {"left": cond.left.name, "right": cond.right.name}


HYPERPARAMETER_ENCODERS: dict[type, tuple[str, _Encoder]] = {
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

CONDITION_ENCODERS: dict[type, tuple[str, _Encoder]] = {
    EqualsCondition: ("EQ", _encode_equals_condition),
    NotEqualsCondition: ("NEQ", _encode_not_equals_condition),
    LessThanCondition: ("LT", _encode_less_than_condition),
    GreaterThanCondition: ("GT", _encode_greater_than_condition),
    InCondition: ("IN", _encode_in_condition),
    AndConjunction: ("AND", _encode_and_conjuction),
    OrConjunction: ("OR", _encode_or_conjuction),
}

FORBIDDEN_ENCODERS: dict[type, tuple[str, _Encoder]] = {
    ForbiddenEqualsClause: ("EQUALS", _encode_forbidden_equals),
    ForbiddenInClause: ("IN", _encode_forbidden_in),
    ForbiddenAndConjunction: ("AND", _encode_forbidden_and),
    ForbiddenLessThanRelation: ("RELATION_LT", _encode_forbidden_relation_less_than),
    ForbiddenEqualsRelation: ("RELATION_EQ", _encode_forbidden_relation_equals),
    ForbiddenGreaterThanRelation: (
        "RELATION_GT",
        _encode_forbidden_relation_greater_than,
    ),
}
