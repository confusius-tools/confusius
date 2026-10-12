"""Shared helpers for ConfUSIus I/O modules."""

import json
import math
import warnings
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import dask.array as da
import numpy as np
import numpy.typing as npt

from confusius._utils.stack import find_stack_level

ZARR_V3_CONSOLIDATED_METADATA_WARNING = (
    "Consolidated metadata is currently not part in the Zarr format 3 specification."
)
"""Zarr v3 warning text emitted when consolidated metadata is written."""


def convert_to_json_serializable(value: Any) -> Any:
    """Recursively convert numpy containers and scalars to native Python objects.

    Parameters
    ----------
    value : Any
        Attribute value, possibly a numpy array or scalar or a `dict`/`list`/`tuple`
        nesting them.

    Returns
    -------
    Any
        Equivalent value with every `numpy.ndarray` replaced by a nested list and every
        numpy scalar replaced by its Python counterpart. Non-numpy values are returned
        unchanged.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: convert_to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [convert_to_json_serializable(item) for item in value]
    return value


def make_attrs_zarr_safe(attrs: dict[str, Any]) -> dict[str, Any]:
    """Make attributes safe to store as Zarr attributes.

    Zarr stores attributes as JSON. Numpy arrays and scalars, including those nested
    inside dicts or lists such as `attrs["affines"]`, are converted to native Python
    objects. Any remaining value that cannot be JSON-encoded (e.g. matplotlib colormap
    and normalization objects on atlas-derived data) is dropped with a warning.

    Parameters
    ----------
    attrs : dict[str, Any]
        Attributes to sanitize.

    Returns
    -------
    dict[str, Any]
        Copy of `attrs` with numpy values converted to native Python and
        non-JSON-serializable entries removed.
    """
    safe: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in attrs.items():
        converted = convert_to_json_serializable(value)
        try:
            json.dumps(converted)
        except TypeError:
            dropped.append(key)
        else:
            safe[key] = converted

    if dropped:
        warnings.warn(
            f"Dropping non-JSON-serializable attrs from Zarr store: {dropped}.",
            stacklevel=find_stack_level(),
        )

    return safe


def restore_affines_in_attrs(attrs: dict[str, Any]) -> None:
    """Restore `attrs["affines"]` dict values to numpy arrays in place.

    Zarr stores the affines as nested lists (see
    [`make_attrs_zarr_safe`][confusius.io._utils.make_attrs_zarr_safe]); this converts
    them back to numpy arrays so a Zarr round-trip matches the NIfTI and SCAN loaders. A
    no-op when `affines` is absent or not a dict.

    Parameters
    ----------
    attrs : dict[str, Any]
        Attributes to update in place.

    Returns
    -------
    None
        This function mutates `attrs` and returns nothing.
    """
    affines = attrs.get("affines")
    if not isinstance(affines, dict):
        return
    attrs["affines"] = {key: np.asarray(value) for key, value in affines.items()}


def _read_binary_block(
    *,
    path: str,
    shape: tuple[int, ...],
    file_dtype: np.dtype,
    offset: int,
    order: Literal["C", "F"],
    block_info: dict[Any, Any],
) -> npt.NDArray:
    """Reopen a read-only mapping and return the requested chunk without copying.

    Parameters
    ----------
    path : str
        Absolute binary file path accessible from the executing worker.
    shape : tuple of int
        Full physical payload shape.
    file_dtype : numpy.dtype
        Payload element type, possibly a structured acquisition record.
    offset : int
        Payload offset in bytes.
    order : {"C", "F"}
        Physical payload ordering.
    block_info : dict
        Dask output chunk locations.

    Returns
    -------
    numpy.ndarray
        Read-only view retaining its mapping until the chunk is released.
    """
    slices = tuple(
        slice(start, stop) for start, stop in block_info[None]["array-location"]
    )
    mapping = np.memmap(
        path, mode="r", dtype=file_dtype, offset=offset, shape=shape, order=order
    )
    return np.asarray(mapping[slices])


def map_binary_array(
    path: str | Path,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    offset: int,
    chunks: int | tuple[int, ...] | str | None,
    order: Literal["C", "F"] = "C",
) -> da.Array:
    """Build a lazy array that reopens the binary payload inside each chunk task.

    No mapping or payload is stored in the graph or cached between tasks. Mapped
    chunks are read-only; callers needing in-place processing must copy them.

    Parameters
    ----------
    path : str or pathlib.Path
        Binary file path, resolved before distributing the graph.
    shape : tuple of int
        Full physical payload shape.
    dtype : dtype_like
        Payload element type, including structured types for padded records.
    offset : int
        Payload offset in bytes.
    chunks : int or tuple of int or str or None
        Dask chunk specification in physical payload axis order.
    order : {"C", "F"}, default: "C"
        Physical payload ordering.

    Returns
    -------
    dask.array.Array
        Lazy array whose chunks are worker-local mapped views.

    Raises
    ------
    ValueError
        If the dimensions or offset are invalid or the payload is truncated.
    """
    path = Path(path).resolve()
    dtype = np.dtype(dtype)
    if offset < 0 or any(size < 1 for size in shape):
        raise ValueError(
            "Binary payload dimensions must be positive and offset nonnegative."
        )
    if path.stat().st_size < offset + math.prod(shape) * dtype.itemsize:
        raise ValueError("Binary file is shorter than the expected payload.")
    return da.map_blocks(
        _read_binary_block,
        path=str(path),
        shape=shape,
        dtype=dtype,
        file_dtype=dtype,
        offset=offset,
        order=order,
        chunks=da.core.normalize_chunks(
            shape if chunks is None else chunks, shape=shape, dtype=dtype
        ),
        meta=np.empty((0,) * len(shape), dtype=dtype),
        name=f"read-binary-{uuid4().hex}",
    )
