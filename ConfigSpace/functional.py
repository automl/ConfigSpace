from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from more_itertools import roundrobin


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


def repr_maker(cls, **kwargs) -> str:
    """Create a repr string for a class.

    >>> class A:
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...
    ...     def __repr__(self):
    ...         return repr(self, a=self.a, b=self.b)
    ...
    >>> A(1, 2)
    A(a=1, b=2)

    Parameters
    ----------
    cls: type
        The class to create a repr for

    kwargs: dict
        The kwargs to include in the repr

    Returns:
    -------
    str
    """
    return f"{cls.__name__}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"


def in_bounds(
    v: int | float | np.number,
    bounds: tuple[int | float | np.number, int | float | np.number],
    *,
    integer: bool = False,
) -> bool:
    """Check if a value is in bounds (inclusive).

    >>> in_bounds(5, 0, 10)
    True
    >>> in_bounds(5, 6, 10)
    False

    Parameters
    ----------
    v: int | float
        The value to check

    low: int | float
        The low end of the range

    high: int | float
        The high end of the range

    Returns:
    -------
    bool
    """
    low, high = bounds
    if integer:
        return bool(low <= v <= high) and int(v) == v

    return bool(low <= v <= high)


def discretize(
    x: npt.NDArray[np.float64],
    *,
    bounds: tuple[int | float | np.number, int | float | np.number],
    bins: int,
) -> npt.NDArray[np.int64]:
    """Discretize an array of values to their closest bin.

    Similar to `np.digitize` but does not require the bins to be specified or loaded
    into memory.
    Similar to `np.histogram` but returns the same length as the input array, where each
    element is assigned to their integer bin.

    >>> discretize(np.array([0.0, 0.1, 0.3, 0.5, 1]), bounds=(0, 1), bins=3)
    array([0, 0, 1, 1, 2])

    Args:
        x: np.NDArray[np.float64]
            The values to discretize

        bounds: tuple[int, int]
            The bounds of the range

        bins: int
            The number of bins

        scale_back: bool = False
            If `True` the discretized values will be scaled back to the original range

    Returns:
    -------
        np.NDArray[np.int64]
    """
    lower, upper = bounds

    norm = (x - lower) / (upper - lower)
    return np.floor(norm * bins).clip(0, bins - 1)
