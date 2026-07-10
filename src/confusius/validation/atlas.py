"""Atlas Dataset validation utilities."""

import numpy as np
import xarray as xr

from confusius._dims import SPATIAL_DIMS

_REQUIRED_ATLAS_DATA_VARS = ("reference", "annotation", "hemispheres")
"""Data variables every atlas Dataset must carry."""

_REQUIRED_ATLAS_ATTRS = (
    "name",
    "citation",
    "species",
    "orientation",
    "structures",
)
"""Attributes every atlas Dataset must carry (the self-describing metadata).

The `base_to_current` mesh transform is required too but checked separately: it is an
`attrs["affines"]` entry for the common affine case and a data variable for a nonlinear
(displacement-field) transform.
"""


def _validate_variable_affines(ds: xr.Dataset) -> None:
    """Check that same-named affines agree across the Dataset's data variables.

    Each data variable may carry an `attrs["affines"]` dict mapping a name (e.g.
    `physical_to_sform`) to a matrix. Two variables that both define an affine of a given
    name describe the same grid, so those matrices must be equal; a mismatch means the
    variables are not on a common physical frame and the atlas is invalid.

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset whose data variables' `affines` attrs are cross-checked.

    Raises
    ------
    ValueError
        If two data variables hold different matrices for the same affine name.
    """
    seen: dict[str, tuple[str, np.ndarray]] = {}
    for var_name in ds.data_vars:
        affines = ds[var_name].attrs.get("affines")
        if not isinstance(affines, dict):
            continue
        for affine_name, matrix in affines.items():
            matrix = np.asarray(matrix, dtype=np.float64)
            if affine_name not in seen:
                seen[affine_name] = (str(var_name), matrix)
                continue
            first_var, first_matrix = seen[affine_name]
            if first_matrix.shape != matrix.shape or not np.allclose(
                first_matrix, matrix
            ):
                raise ValueError(
                    f"Atlas variables disagree on affine '{affine_name}': "
                    f"'{first_var}' and '{var_name}' hold different matrices, so they are "
                    "not on a common physical frame."
                )


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
       `orientation`, and `structures` are present, plus the `base_to_current` mesh
       transform (an `attrs["affines"]` affine, or a data variable for a nonlinear
       transform).
    6. **Affines**: where two data variables both define an affine of the same name (in
       `attrs["affines"]`), the matrices must be equal — a mismatch means the variables
       are not on a common physical frame.
    7. **Structures**: `attrs["structures"]` is a brainglobe `StructuresDict`.

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
        is not a brainglobe `StructuresDict`.

    Examples
    --------
    >>> from confusius.datasets import fetch_brainglobe_atlas
    >>> atlas = fetch_brainglobe_atlas("allen_mouse_100um")
    >>> validate_atlas_dataset(atlas)
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Expected an xarray.Dataset, got {type(ds).__name__}.")

    missing_vars = [v for v in _REQUIRED_ATLAS_DATA_VARS if v not in ds.data_vars]
    if missing_vars:
        raise ValueError(
            f"Atlas Dataset is missing required data variables: {missing_vars}. "
            f"An atlas must have data variables {list(_REQUIRED_ATLAS_DATA_VARS)} "
            "(hemispheres must be a data variable, not a coordinate)."
        )

    reference_dims = ds["reference"].dims
    for name in _REQUIRED_ATLAS_DATA_VARS:
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

    missing_attrs = [a for a in _REQUIRED_ATLAS_ATTRS if a not in ds.attrs]
    if missing_attrs:
        raise ValueError(
            f"Atlas Dataset is missing required attributes: {missing_attrs}. "
            "xarray drops attrs on many operations by default; run atlas pipelines "
            "under xarray.set_options(keep_attrs=True)."
        )

    # base_to_current is an affine in attrs["affines"] for the common case, or a data
    # variable (a displacement field) after a nonlinear resample.
    if (
        "base_to_current" not in ds.attrs.get("affines", {})
        and "base_to_current" not in ds.data_vars
    ):
        raise ValueError(
            "Atlas Dataset is missing 'base_to_current' (expected either an "
            "attrs['affines'] affine or a data variable holding a nonlinear "
            "displacement field)."
        )

    _validate_variable_affines(ds)

    # In memory the structures ride as a BrainGlobe StructuresDict (serialized to JSON only
    # inside the Zarr store); atlas_from_brainglobe and atlas_from_zarr both produce one.
    from brainglobe_atlasapi.structure_class import StructuresDict

    if not isinstance(ds.attrs["structures"], StructuresDict):
        raise ValueError(
            "Atlas attribute 'structures' must be a brainglobe StructuresDict (as built "
            "by atlas_from_brainglobe or atlas_from_zarr), got "
            f"{type(ds.attrs['structures']).__name__}."
        )
