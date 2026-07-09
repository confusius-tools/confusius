"""Atlas Dataset validation utilities."""

import json

import numpy as np
import xarray as xr

from confusius._dims import SPATIAL_DIMS

_REQUIRED_DATA_VARS = ("reference", "annotation", "hemispheres")
"""Data variables every atlas Dataset must carry."""

_REQUIRED_ATTRS = (
    "name",
    "citation",
    "species",
    "orientation",
    "structures",
    "rl_midline",
)
"""Attributes every atlas Dataset must carry (the self-describing metadata).

`mesh_vertex_transform` is required too but checked separately: it is an `attrs` entry for
the common affine case and a data variable for a nonlinear (displacement-field) transform.
"""


def validate_atlas_dataset(ds: xr.Dataset) -> None:
    """Validate that a Dataset is a well-formed atlas.

    Companion to [`validate_fusi_dataarray`][confusius.validation.validate_fusi_dataarray]
    and [`validate_iq_dataarray`][confusius.validation.validate_iq_dataarray]. Checks that
    `ds` matches the atlas schema produced by
    [`atlas_from_brainglobe`][confusius.atlas.atlas_from_brainglobe] and consumed by the
    `.atlas` accessor:

    1. **Type**: `ds` is an `xarray.Dataset`.
    2. **Data variables**: `reference`, `annotation`, and `hemispheres` are all present as
       data variables (a `hemispheres` stored as a coordinate is reported as missing).
    3. **Grid**: the three variables share identical dimensions, and those dimensions are a
       subset of `(z, y, x)` (2D or 3D, so a resampled single slice is accepted).
    4. **Data types**: `reference` is floating-point; `annotation` and `hemispheres` are
       integer-valued.
    5. **Attributes**: the self-describing metadata `name`, `citation`, `species`,
       `orientation`, `structures`, and `rl_midline` are present, plus
       `mesh_vertex_transform` (an `attrs` affine, or a data variable for a nonlinear
       transform).
    6. **Structures**: `attrs["structures"]` parses as a JSON list.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate as an atlas.

    Raises
    ------
    TypeError
        If `ds` is not an `xarray.Dataset`, or if `reference` is not floating-point or
        `annotation`/`hemispheres` are not integer-valued.
    ValueError
        If any required data variable or attribute is missing, if the variables do not
        share dimensions that are a subset of `(z, y, x)`, or if `attrs["structures"]`
        does not parse as a JSON list.

    Examples
    --------
    >>> from confusius.datasets import fetch_brainglobe_atlas
    >>> atlas = fetch_brainglobe_atlas("allen_mouse_100um")
    >>> validate_atlas_dataset(atlas)
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Expected an xarray.Dataset, got {type(ds).__name__}.")

    missing_vars = [v for v in _REQUIRED_DATA_VARS if v not in ds.data_vars]
    if missing_vars:
        raise ValueError(
            f"Atlas Dataset is missing required data variables: {missing_vars}. "
            f"An atlas must have data variables {list(_REQUIRED_DATA_VARS)} "
            "(hemispheres must be a data variable, not a coordinate)."
        )

    reference_dims = ds["reference"].dims
    for name in _REQUIRED_DATA_VARS:
        if ds[name].dims != reference_dims:
            raise ValueError(
                f"Atlas variables must share dimensions; '{name}' has dims "
                f"{ds[name].dims} but 'reference' has {reference_dims}."
            )
    if not set(reference_dims).issubset(SPATIAL_DIMS):
        raise ValueError(
            f"Atlas dimensions must be a subset of {SPATIAL_DIMS}, got {reference_dims}."
        )

    if not np.issubdtype(ds["reference"].dtype, np.floating):
        raise TypeError(
            f"Atlas 'reference' must be floating-point, got dtype {ds['reference'].dtype}."
        )
    for name in ("annotation", "hemispheres"):
        if not np.issubdtype(ds[name].dtype, np.integer):
            raise TypeError(
                f"Atlas '{name}' must be integer-valued, got dtype {ds[name].dtype}."
            )

    missing_attrs = [a for a in _REQUIRED_ATTRS if a not in ds.attrs]
    if missing_attrs:
        raise ValueError(
            f"Atlas Dataset is missing required attributes: {missing_attrs}. "
            "xarray drops attrs on many operations by default; run atlas pipelines "
            "under xarray.set_options(keep_attrs=True)."
        )

    # mesh_vertex_transform is an affine in attrs for the common case, or a data variable
    # (a displacement field) after a nonlinear resample.
    if (
        "mesh_vertex_transform" not in ds.attrs
        and "mesh_vertex_transform" not in ds.data_vars
    ):
        raise ValueError(
            "Atlas Dataset is missing 'mesh_vertex_transform' (expected either an attrs "
            "affine or a data variable holding a nonlinear displacement field)."
        )

    try:
        structures = json.loads(ds.attrs["structures"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Atlas attribute 'structures' must be a JSON string; "
            f"failed to parse it ({error})."
        ) from error
    if not isinstance(structures, list):
        raise ValueError(
            "Atlas attribute 'structures' must be a JSON list of structure records, "
            f"got {type(structures).__name__}."
        )
