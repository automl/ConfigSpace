from itertools import cycle, islice
from typing import Iterator, Iterable, TypeVar

T = TypeVar("T")


def roundrobin(*iterables: Iterable[T]) -> Iterator[T]:
    """Iterate over several iterables in a roundrobin fashion.

    https://docs.python.org/2/library/itertools.html#recipes

    >>> list(roundrobin('ABC', 'D', 'EF'))
    ['A', 'D', 'E', 'B', 'F', 'C']

    Parameters
    ----------
    *iterables: Iterable[T]
        The iterables to use

    Returns
    -------
    Iterator[T]
    """
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def center_range(
    center: int,
    low: int,
    high: int,
    step: int = 1,
) -> Iterator[int]:
    """Get a range centered around a value.

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
    below_center = range(center + step, high + 1, step)
    above_center = range(center - step, low - 1, -step)
    yield from roundrobin(below_center, above_center)
