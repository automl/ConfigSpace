from __future__ import annotations

import numpy as np

from ConfigSpace.functional import arange_chunked, center_range


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
