from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import overload

import numpy as np
from more_itertools import roundrobin

from ConfigSpace.types import Number, f64, i64

if TYPE_CHECKING:
    from ConfigSpace.types import Array, Mask


def center_range(
    center: int,
    low: int,
    high: int,
    step: int = 1,
) -> Iterator[int]:
    """Get a range centered around a value.

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace.functional import center_range

    print(list(center_range(5, 0, 10)))
    ```

    Args:
        center: The center of the range
        low: The low end of the range
        high: The high end of the range
        step: The step size

    Returns:
        An iterator of ints around a center point
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

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace.functional import arange_chunked

    print(list(arange_chunked(0, 10, chunk_size=3)))
    ```

    Args:
        start: The start of the range
        stop: The stop of the range
        chunk_size: The size of the chunks
        step: The step size

    Returns:
        An iterator of np.ndarrays
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

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace.functional import linspace_chunked

    print(list(linspace_chunked(0, 10, 11, chunk_size=3, endpoint=True)))
    ```

    Args:
        start: The start of the range
        stop: The stop of the range
        num: The number of samples to generate
        chunk_size: The size of the chunks
        endpoint:
            If True, stop is the last sample. Otherwise, it is not included.
            Simliar to `np.linspace`

    Returns:
        An iterator of np.ndarrays
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


def split_arange(frm: int, to: int, *, pivot: int) -> Array[i64]:
    """Split an arange into multiple ranges.

    ```python exec="true", source="material-block" result="python"
    from ConfigSpace.functional import split_arange

    print(split_arange(0, 10, pivot=5))
    ```

    Args:
        frm: Start of range
        to: End of range
        pivot: The pivot point, ommited from the output

    Returns:
        The concatenated ranges without the pivot
    """
    bot = np.arange(frm, pivot)
    top = np.arange(pivot + 1, to)
    return np.concatenate([bot, top])


def quantize_log(
    x: Array[f64],
    *,
    bounds: tuple[int | float | np.number, int | float | np.number],
    scale_slice: tuple[int | float | np.number, int | float | np.number] | None = None,
    bins: int,
) -> Array[f64]:
    """Quantize an array of values on a log scale.

    Works by first lifting the values to the provided slice of the log scale
    (scale_slice), exponentiate back to linear scale and then perform quantization.
    Gives back the values in provided scale (bounds).

    Args:
        x: The values to quantize
        bounds: The bounds on which the values live on
        scale_slice: The specific slice of the log scale they were logged from.
        bins: The number of bins to quantize to

    Returns:
        The quantized values
    """
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


@overload
def quantize(
    x: Array[f64],
    *,
    bounds: tuple[Number, Number],
    bins: int,
) -> Array[f64]: ...


@overload
def quantize(x: f64, *, bounds: tuple[Number, Number], bins: int) -> f64: ...


def quantize(
    x: Array[f64] | f64,
    *,
    bounds: tuple[Number, Number],
    bins: int,
) -> Array[f64] | f64:
    """Discretize an array of values to their closest bin.

    Similar to `np.digitize` but does not require the bins to be specified or loaded
    into memory.
    Similar to `np.histogram` but returns the same length as the input array, where each
    element is assigned to their integer bin.

    ```python exec="true", source="material-block" result="python"
    import numpy as np
    from ConfigSpace.functional import quantize

    print(quantize(np.array([0.0, 0.32, 0.33, 0.34, 0.65, 0.66, 0.67, 0.99, 1.0]), bounds=(0, 1), bins=3))
    ```

    Args:
        x: The values to discretize
        bounds: The bounds of the range
        bins: The number of bins

    Returns:
        The discretized values
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
    unit_xs: Array[f64],
    to: tuple[int | float | np.number, int | float | np.number],
) -> Array[f64]:
    """Scale values from unit range to a new range.

    ```python exec="true", source="material-block" result="python"
    import numpy as np
    from ConfigSpace.functional import scale

    print(scale(np.array([0.0, 0.5, 1.0]), to=(0, 10)))
    ```

    Args:
        unit_xs: The values to scale
        to: The new range

    Returns:
        The scaled values
    """
    return unit_xs * (to[1] - to[0]) + to[0]  # type: ignore


def normalize(
    x: Array[np.number],
    *,
    bounds: tuple[int | float | np.number, int | float | np.number],
) -> Array[f64]:
    """Normalize values to the unit range.

    ```python exec="true", source="material-block" result="python"
    import numpy as np
    from ConfigSpace.functional import normalize

    print(normalize(np.array([0.0, 5.0, 10.0]), bounds=(0, 10)))
    ```

    Args:
        x: The values to normalize
        bounds: The bounds of the range

    Returns:
        The normalized values
    """
    if bounds == (0, 1):
        return x.astype(f64)

    return (x - bounds[0]) / (bounds[1] - bounds[0])  # type: ignore


def rescale(
    x: Array[f64],
    frm: tuple[int | float | np.number, int | float | np.number],
    to: tuple[int | float | np.number, int | float | np.number],
) -> Array[f64]:
    """Rescale values from one range to another.

    ```python exec="true", source="material-block" result="python"
    import numpy as np
    from ConfigSpace.functional import rescale

    print(rescale(np.array([0, 10, 20]), frm=(0, 100), to=(0, 10)))
    ```

    Args:
        x: The values to rescale
        frm: The original range
        to: The new range

    Returns:
        The rescaled values
    """
    if frm == to:
        return x.astype(f64)

    normed = normalize(x, bounds=frm)
    return scale(unit_xs=normed, to=to)


@overload
def is_close_to_integer(
    value: f64 | float,
    *,
    atol: float = ...,
    rtol: float = ...,
) -> np.bool_: ...


@overload
def is_close_to_integer(
    value: Array[f64],
    *,
    atol: float = ...,
    rtol: float = ...,
) -> Mask: ...


def is_close_to_integer(
    value: f64 | float | Array[f64],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-5,
) -> np.bool_ | Mask:
    """Check if a value is close to an integer.

    This implements the same logic as `np.isclose` but removes
    a lot of the overhead.

    Args:
        value: The value to check
        atol: The absolute tolerance
        rtol: The relative tolerance

    Returns:
        Whether the value is close to an integer
    """
    a = np.asarray(value)
    b = np.rint(a)
    return np.less_equal(np.abs(a - b), atol + rtol * np.abs(b))  # type: ignore


def is_close_to_integer_single(
    value: Number,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-5,
) -> np.bool_:
    """Check if a single value is close to an integer.

    This implements the same logic as `np.isclose` but removes
    a lot of the overhead.

    Args:
        value: The value to check
        atol: The absolute tolerance
        rtol: The relative tolerance

    Returns:
        Whether the value is close to an integer
    """
    a = value
    _b = np.rint(a)  # type: ignore
    return abs(a - _b) <= (atol + rtol * abs(_b))  # type: ignore


T = TypeVar("T")


def walk_subclasses(
    cls: type[T],
    seen: set[type[T]] | None = None,
) -> Iterator[type[T]]:
    """Walk all subclasses of a class.

    Args:
        cls: The class to walk
        seen: The set of seen classes

    Returns:
        An iterator of subclasses
    """
    seen = set() if seen is None else seen
    for subclass in cls.__subclasses__():
        if subclass not in seen:
            seen.add(subclass)
            yield subclass
            yield from walk_subclasses(subclass)
