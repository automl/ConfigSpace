from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar
from typing_extensions import overload

import numpy as np
import numpy.typing as npt
from more_itertools import pairwise, roundrobin

Number = TypeVar("Number", bound=np.number)


def center_range(
    center: int,
    low: int,
    high: int,
    step: int = 1,
) -> Iterator[int]:
    """Get a range centered around a value.

    >>> list(center_range(5, 0, 10))
    [4, 6, 3, 7, 2, 8, 1, 9, 0, 10]

    Parameters
    ----------
    center: int
        The center of the range

    low: int
        The low end of the range

    high: int
        The high end of the range

    step: int = 1
        The step size

    Returns:
    -------
    Iterator[int]
    """
    assert low <= center <= high
    above_center = range(center + step, high + 1, step)
    below_center = range(center - step, low - 1, -step)
    yield from roundrobin(below_center, above_center)


def arange_chunked(
    start: int,
    stop: int,
    step: int = 1,
    *,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    """Get np.arange in a chunked fashion.

    >>> list(arange_chunked(0, 10, 3))
    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]

    Parameters
    ----------
    start: int
        The start of the range

    stop: int
        The stop of the range

    chunk_size: int
        The size of the chunks

    step: int = 1
        The step size

    Returns:
    -------
    Iterator[np.ndarray]
    """
    assert step > 0
    assert chunk_size > 0
    assert start < stop

    n_items = int(np.ceil((stop - start) / step))
    n_chunks = int(np.ceil(n_items / chunk_size))

    for chunk in range(n_chunks):
        chunk_start = start + (chunk * chunk_size)
        chunk_stop = min(chunk_start + chunk_size, stop)
        yield np.arange(chunk_start, chunk_stop, step)


def linspace_chunked(
    start: float,
    stop: float,
    num: int,
    *,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    assert num > 0
    assert chunk_size > 0
    assert start < stop

    if num <= chunk_size:
        yield np.linspace(start, stop, int(num), endpoint=True)
        return

    n_intervals = int(np.ceil(num / chunk_size))
    intervals = np.linspace(start, stop, n_intervals + 1, endpoint=True)
    steps_per_chunk = int(min(num, chunk_size))

    for i, (_start, _stop) in enumerate(pairwise(intervals), start=1):
        is_last = i == n_intervals
        yield np.linspace(_start, _stop, steps_per_chunk, endpoint=is_last)


def split_arange(
    *bounds: tuple[int | np.int64, int | np.int64],
) -> npt.NDArray[np.int64]:
    """Split an arange into multiple ranges.

    >>> split_arange((0, 2), (3, 5), (6, 10))
    [0, 1, 3, 4, 6, 7, 8, 9]

    Parameters
    ----------
    bounds: tuple[int, int]
        The bounds of the ranges

    Returns:
        The concatenated ranges
    """
    return np.concatenate(
        [np.arange(start, stop, dtype=int) for start, stop in bounds],
        dtype=np.int64,
    )


def quantize_log(
    x: npt.NDArray[np.number],
    *,
    bounds: tuple[int | float | np.number, int | float | np.number],
    scale_slice: tuple[int | float | np.number, int | float | np.number] | None = None,
    bins: int,
) -> npt.NDArray[np.float64]:
    if scale_slice is None:
        scale_slice = bounds

    log_bounds = (np.log(scale_slice[0]), np.log(scale_slice[1]))

    # Lift to the log scale
    x_log = rescale(x, frm=bounds, to=log_bounds)

    # Lift to original scale
    x_orig = np.exp(x_log)

    # Quantize on the scale
    qx_orig = quantize(
        x_orig,
        bounds=scale_slice,
        bins=bins,
    )

    # Now back to log
    qx_log = np.log(qx_orig)

    # And norm back to original scale
    return rescale(qx_log, frm=log_bounds, to=bounds)


def quantize(
    x: npt.NDArray[np.number],
    *,
    bounds: tuple[int | float | np.number, int | float | np.number],
    bins: int,
) -> npt.NDArray[np.float64]:
    """Discretize an array of values to their closest bin.

    Similar to `np.digitize` but does not require the bins to be specified or loaded
    into memory.
    Similar to `np.histogram` but returns the same length as the input array, where each
    element is assigned to their integer bin.

    >>> quantize(np.array([0.0, 0.32, 0.33, 0.34, 0.65, 0.66, 0.67, 0.99, 1.0]), bounds=(0, 1), bins=3)
    array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])

    Args:
        x: np.NDArray[np.float64]
            The values to discretize

        bounds: tuple[int, int]
            The bounds of the range

        bins: int
            The number of bins

    Returns:
    -------
        np.NDArray[np.int64]
    """  # noqa: E501
    # Shortcut out if we have unit norm already
    unitnorm = x if bounds == (0, 1) else normalize(x, bounds=bounds)
    int_bounds = (0, bins - 1)

    quantization_levels = np.floor(unitnorm * bins).clip(*int_bounds)
    return rescale(quantization_levels, frm=int_bounds, to=bounds)


def scale(
    unit_xs: npt.NDArray,
    to: tuple[int | float | np.number, int | float | np.number],
) -> npt.NDArray:
    return unit_xs * (to[1] - to[0]) + to[0]  # type: ignore


def normalize(
    x: npt.NDArray,
    *,
    bounds: tuple[int | float | np.number, int | float | np.number],
) -> npt.NDArray:
    if bounds == (0, 1):
        return x
    return (x - bounds[0]) / (bounds[1] - bounds[0])  # type: ignore


def rescale(
    x: npt.NDArray,
    frm: tuple[int | float | np.number, int | float | np.number],
    to: tuple[int | float | np.number, int | float | np.number],
) -> npt.NDArray:
    if frm == to:
        return x

    normed = normalize(x, bounds=frm)
    return scale(unit_xs=normed, to=to)


@overload
def is_close_to_integer(value: int | float | np.number, decimals: int) -> bool: ...


@overload
def is_close_to_integer(
    value: np.ndarray,
    decimals: int,
) -> npt.NDArray[np.bool_]: ...


def is_close_to_integer(
    value: int | float | np.number | np.ndarray,
    decimals: int,
) -> bool | npt.NDArray[np.bool_]:
    return np.round(value, decimals) == np.rint(value)  # type: ignore


def walk_subclasses(cls: type, seen: set[type] | None = None) -> Iterator[type]:
    seen = set() if seen is None else seen
    for subclass in cls.__subclasses__():
        if subclass not in seen:
            seen.add(subclass)
            yield subclass
            yield from walk_subclasses(subclass)
