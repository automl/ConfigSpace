from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar
from typing_extensions import overload

import numpy as np
import numpy.typing as npt
from more_itertools import roundrobin

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
    endpoint: bool = False,
) -> Iterator[np.ndarray]:
    """Get np.linspace in a chunked fashion.

    >>> list(linspace_chunked(0, 10, 11, chunk_size=3, endpoint=True))
    [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7., 8.]), array([ 9., 10.])]

    Parameters
    ----------
    start:
        The start of the range

    stop:
        The stop of the range

    num:
        The number of samples to generate

    chunk_size:
        The size of the chunks

    endpoint:
        If True, stop is the last sample. Otherwise, it is not included.
        Simliar to `np.linspace`

    Returns
    -------
    Iterator[np.ndarray]
    """
    assert num > 0
    assert chunk_size > 0
    assert start < stop

    if num <= chunk_size:
        yield np.linspace(start, stop, int(num), endpoint=endpoint)
        return

    _div = num - 1 if endpoint else num
    for chunk in arange_chunked(0, num, chunk_size=chunk_size):
        yield (chunk / _div) * (stop - start) + start


NPDType = TypeVar("NPDType", bound=np.generic)


def split_arange(
    frm: int,
    to: int,
    *,
    pivot: int,
) -> npt.NDArray[np.int64]:
    """Split an arange into multiple ranges.

    >>> split_arange(0, 10, pivot=5)
    [0, 1, 3, 4, 6, 7, 8, 9]

    Parameters
    ----------
    frm:
        Start of range

    to:
        End of range

    pivot:
        The pivot point, ommited from the output

    Returns
    -------
        The concatenated ranges
    """
    bot = np.arange(frm, pivot)
    top = np.arange(pivot + 1, to)
    return np.concatenate([bot, top])


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

    Returns
    -------
        np.NDArray[np.float64]
    """  # noqa: E501
    # Shortcut out if we have unit norm already
    l, u = bounds  # noqa: E741
    unitnorm = x if bounds == (0, 1) else (x - l) / (u - l)

    quantization_levels = np.floor(unitnorm * bins).clip(0, bins - 1)
    unit_norm_quantized = quantization_levels / (bins - 1)
    if bounds == (0, 1):
        return unit_norm_quantized

    return unit_norm_quantized * (u - l) + l  # type: ignore


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
def is_close_to_integer(
    value: int | float | np.number,
    *,
    atol: float = ...,
    rtol: float = ...,
) -> bool: ...


@overload
def is_close_to_integer(
    value: np.ndarray,
    *,
    atol: float = ...,
    rtol: float = ...,
) -> npt.NDArray[np.bool_]: ...


def is_close_to_integer(
    value: int | float | np.number | np.ndarray,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-5,
) -> bool | npt.NDArray[np.bool_]:
    """Check if a value is close to an integer.

    This implements the same logic as `np.isclose` but removes
    a lot of the overhead.

    Parameters
    ----------
    value:
        The value to check

    atol:
        The absolute tolerance

    rtol:
        The relative tolerance

    Returns
    -------
        Whether the value is close to an integer
    """
    a = np.asarray(value)
    b = np.rint(a)
    return np.less_equal(np.abs(a - b), atol + rtol * np.abs(b))


def is_close_to_integer_single(
    value: int | float | np.number,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-5,
) -> bool:
    """Check if a single value is close to an integer.

    This implements the same logic as `np.isclose` but removes
    a lot of the overhead.

    Parameters
    ----------
    value:
        The value to check

    atol:
        The absolute tolerance

    rtol:
        The relative tolerance

    Returns
    -------
        Whether the value is close to an integer
    """
    a = value
    _b = np.rint(a)  # type: ignore
    return abs(a - _b) <= (atol + rtol * abs(_b))


def walk_subclasses(cls: type, seen: set[type] | None = None) -> Iterator[type]:
    """Walk all subclasses of a class.

    Parameters
    ----------
    cls:
        The class to walk

    seen:
        The set of seen classes

    Returns
    -------
        An iterator of subclasses
    """
    seen = set() if seen is None else seen
    for subclass in cls.__subclasses__():
        if subclass not in seen:
            seen.add(subclass)
            yield subclass
            yield from walk_subclasses(subclass)
