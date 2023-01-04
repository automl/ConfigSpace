from typing import Iterator

from more_itertools import roundrobin
import numpy as np


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

    Returns
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

    Returns
    -------
    Iterator[np.ndarray]
    """
    assert step > 0
    assert chunk_size > 0
    assert 0 <= start < stop
    n_items = int(np.ceil((stop - start) / step))
    n_chunks = int(np.ceil(n_items / chunk_size))

    for chunk in range(0, n_chunks):
        chunk_start = start + (chunk * chunk_size)
        chunk_stop = min(chunk_start + chunk_size, stop)
        yield np.arange(chunk_start, chunk_stop, step)
