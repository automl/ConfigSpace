from __future__ import annotations

import warnings
from typing import Any


def deprecate(
    thing: Any,
    instead: str,
    stacklevel: int = 3,
) -> None:
    """Deprecate a thing and warn when it is used.

    Parameters
    ----------
    thing : Any
        The thing to deprecate.

    instead : str
        What to use instead.

    stacklevel : int, optional
        How many levels up in the stack to place the warning.
        Defaults to 3.
    """
    msg = f"{thing} is deprecated and will be removed in a future version." f"\n{instead}"
    warnings.warn(msg, stacklevel=stacklevel)
