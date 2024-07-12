from __future__ import annotations

import numpy as np
from pytest_cases import parametrize

from ConfigSpace.functional import (
    arange_chunked,
    center_range,
    normalize,
    quantize,
    rescale,
    scale,
)


def test_center_range_equal():
    assert list(center_range(5, low=0, high=10)) == [4, 6, 3, 7, 2, 8, 1, 9, 0, 10]


def test_center_range_unequal():
    assert list(center_range(5, low=4, high=10)) == [4, 6, 7, 8, 9, 10]


def test_arange_chunked_uneven():
    expected = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
        np.array([9]),
    ]
    result = list(arange_chunked(0, 10, chunk_size=3))
    assert len(result) == len(expected)
    for row_exp, row_result in zip(result, expected):
        assert np.all(row_exp == row_result)


def test_arange_chunked_even():
    expected = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
    ]
    result = list(arange_chunked(0, 9, chunk_size=3))
    assert len(result) == len(expected)
    for row_exp, row_result in zip(result, expected):
        assert np.all(row_exp == row_result)


@parametrize(
    "x, bounds, expected",
    [
        (
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            (0, 10),
            np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
        ),
        (
            np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
            (0, 1),
            np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
        ),
        (
            np.array([0]),
            (-1, 1),
            np.array([0.5]),
        ),
        (
            np.array([]),
            (-1, 1),
            np.array([]),
        ),
    ],
)
def test_normalize(
    x: np.ndarray,
    bounds: tuple[int | float | np.number, int | float | np.number],
    expected: np.ndarray,
) -> None:
    normed_arr = normalize(x, bounds=bounds)
    np.testing.assert_array_equal(normed_arr, expected)


@parametrize(
    "x, to, expected",
    [
        (
            np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
            (0, 10),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ),
        (
            np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
            (0, 1),
            np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
        ),
        (
            np.array([0.5]),
            (-1, 1),
            np.array([0.0]),
        ),
        (
            np.array([]),
            (-1, 1),
            np.array([]),
        ),
    ],
)
def test_scale(
    x: np.ndarray,
    to: tuple[int | float | np.number, int | float | np.number],
    expected: np.ndarray,
) -> None:
    scaled = scale(x, to=to)
    np.testing.assert_array_equal(scaled, expected)


@parametrize(
    "x, frm, to, expected",
    [
        (
            np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
            (0, 1),
            (0, 10),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ),
        (
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            (0, 10),
            (0, 10),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ),
        (
            np.array([0, 1]),
            (0, 10),
            (0, 1),
            np.array([0, 0.1]),
        ),
        (
            np.array([0.5]),
            (0, 10),
            (-1, 1),
            np.array([-0.9]),
        ),
        (
            np.array([]),
            (0, 10),
            (-1, 1),
            np.array([]),
        ),
    ],
)
def test_rescale(
    x: np.ndarray,
    frm: tuple[int | float | np.number, int | float | np.number],
    to: tuple[int | float | np.number, int | float | np.number],
    expected: np.ndarray,
) -> None:
    rescaled = rescale(x, frm=frm, to=to)
    np.testing.assert_array_equal(rescaled, expected)


"""
(
    np.array(
        [
            0.0,
            0.11111111,
            0.22222222,
            0.33333333,
            0.44444444,
            0.55555556,
            0.66666667,
            0.77777778,
            0.88888889,
            1.0,
        ],
    ),
    (0, 1),
    10,
    np.array(
        [
            0.0,
            0.11111111,
            0.22222222,
            0.33333333,
            0.44444444,
            0.55555556,
            0.66666667,
            0.77777778,
            0.88888889,
            1.0,
        ],
    ),
),
"""


@parametrize(
    "x, bounds, bins, expected",
    [
        (
            np.array([0.0, 0.32, 0.33, 0.34, 0.65, 0.66, 0.67, 0.99, 1.0]),
            (0, 1),
            3,
            np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1]),
        ),
        (
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (0, 10),
            5,
            np.array([0, 0, 2.5, 2.5, 5, 5, 7.5, 7.5, 10, 10]),
        ),
        (np.array([]), (0, 10), 10, np.array([])),
    ],
)
def test_quantize(
    x: np.ndarray,
    bounds: tuple[int | float | np.number, int | float | np.number],
    bins: int,
    expected: np.ndarray,
) -> None:
    quantized = quantize(x, bounds=bounds, bins=bins)
    np.testing.assert_array_equal(quantized, expected)
