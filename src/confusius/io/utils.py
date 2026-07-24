"""Miscellaneous I/O utility functions."""

from pathlib import Path
from typing import Any, Literal


def check_path(
    path: Path | str | Any,
    label: str = "path",
    type: Literal["file", "dir"] | None = None,
) -> Path:
    """Resolve full path and check its type.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to check validity.
    label : str, default: "path"
        Name of the variable passed to `check_path`, used in the error message.
    type : {"file", "dir"}, optional
        Type of path to check for. If `"file"`, checks that the path is a file. If
        `"dir"`, checks that the path is a directory. If not provided, no type check
        is performed, meaning the path may not exist.

    Returns
    -------
    pathlib.Path
        If successful, the path resolved to its full path.

    Raises
    ------
    TypeError
        If `path` cannot be cast to pathlib.Path.
    ValueError
        If type is not `None` and the path is not of the correct type.
    """
    try:
        path = Path(path)
    except TypeError as e:
        raise TypeError(
            f"{label} argument must be a pathlib.Path (or a type that supports"
            " casting to pathlib.Path, such as string)."
        ) from e

    path = path.expanduser().resolve()

    if type == "file" and not path.is_file():
        raise ValueError(f"{label} argument must be a valid file path.")
    elif type == "dir" and not path.is_dir():
        raise ValueError(f"{label} argument must be a valid directory path.")

    return path
