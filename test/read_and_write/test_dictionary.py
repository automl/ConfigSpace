from __future__ import annotations

from copy import deepcopy

import pytest

from ConfigSpace import (
    ConfigurationSpace,
    EqualsCondition,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    OrConjunction,
)


@pytest.mark.parametrize("include_conditions", [False, True])
@pytest.mark.parametrize("include_forbiddens", [False, True])
def test_from_serialized_dict_preserves_input(
    include_conditions: bool,
    include_forbiddens: bool,
) -> None:
    space = ConfigurationSpace({"a": [0, 1, 2], "b": [0, 1, 2], "c": [0, 1, 2]})
    if include_conditions:
        space.add(
            OrConjunction(
                EqualsCondition(space["c"], space["a"], 1),
                EqualsCondition(space["c"], space["b"], 1),
            ),
        )
    if include_forbiddens:
        space.add(
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(space["a"], 1),
                ForbiddenEqualsClause(space["b"], 1),
            ),
        )
    serialized = space.to_serialized_dict()
    original = deepcopy(serialized)

    assert ConfigurationSpace.from_serialized_dict(serialized) == space
    assert serialized == original
    assert ConfigurationSpace.from_serialized_dict(serialized) == space
    assert serialized == original


def test_from_serialized_dict_empty_space() -> None:
    space = ConfigurationSpace()
    serialized = space.to_serialized_dict()
    original = deepcopy(serialized)

    assert ConfigurationSpace.from_serialized_dict(serialized) == space
    assert ConfigurationSpace.from_serialized_dict(serialized) == space
    assert serialized == original
