"""Stack-introspection helpers for accurate `warnings.warn` stack levels."""

import inspect
from pathlib import Path


def find_stack_level() -> int:
    """Find the first place in the stack that is not inside confusius.

    Adapted from
    [pandas](https://github.com/pandas-dev/pandas/tree/main/pandas/util/_exceptions.py#L37)
    and
    [Nilearn](https://github.com/nilearn/nilearn/blob/2d1a2c6d901ef4aba2737ed84e08ad1956afd123/nilearn/_utils/logger.py#L150).

    Returns
    -------
    int
        Stack level pointing to the first frame outside the confusius package.
    """
    import confusius

    pkg_dir = Path(confusius.__file__).parent

    frame = inspect.currentframe()
    try:
        n = 0
        while frame:
            filename = inspect.getfile(frame)
            if not filename.startswith(str(pkg_dir)):
                break
            frame = frame.f_back
            n += 1
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    return n
