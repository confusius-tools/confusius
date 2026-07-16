"""Atlas Dataset validation utilities."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius._dims import SPATIAL_DIMS

if TYPE_CHECKING:
    from brainglobe_atlasapi.structure_class import StructuresDict

_REQUIRED_ATLAS_DATA_VARS = ("reference", "annotation", "hemispheres")
"""Data variables every atlas Dataset must carry."""

_REQUIRED_ATLAS_ATTRS = ("structures",)
"""Attributes every atlas Dataset must carry.

Only `structures` is unconditionally required. The `physical_to_base` transform and usable
region meshes are required only when validating for mesh operations
(`require_mesh_use=True`); the descriptive metadata the builder adds (`name`, `citation`,
`species`, `orientation`) is optional.
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


def _validate_meshes_available(structures: "StructuresDict") -> None:
    """Raise ValueError unless some structure references an existing mesh file.

    A structure's `mesh_filename` is either `None` (no mesh) or a path; `get_mesh` needs at
    least one that resolves to a file on disk. The scan short-circuits at the first existing
    mesh, so it does not stat every structure.

    Parameters
    ----------
    structures : brainglobe_atlasapi.structure_class.StructuresDict
        The atlas structure dictionary whose `mesh_filename` entries are checked.

    Raises
    ------
    ValueError
        If no structure references a mesh file that exists on disk.
    """
    has_mesh = any(
        structure["mesh_filename"] is not None
        and Path(structure["mesh_filename"]).is_file()
        for structure in structures.values()
    )
    if not has_mesh:
        raise ValueError(
            "Atlas has no usable region meshes: no structure references an existing mesh "
            "file, so get_mesh cannot run (require_mesh_use=True)."
        )


def validate_atlas_dataset(ds: xr.Dataset, *, require_mesh_use: bool = False) -> None:
    """Validate that a Dataset is a well-formed atlas.

    Companion to [`validate_fusi_dataarray`][confusius.validation.validate_fusi_dataarray]
    and [`validate_iq_dataarray`][confusius.validation.validate_iq_dataarray]. Checks that
    `ds` matches the atlas schema produced by
    [`fetch_brainglobe_atlas`][confusius.datasets.fetch_brainglobe_atlas] and consumed
    by the `.atlas` accessor:

    1. **Type**: `ds` is an `xarray.Dataset`.
    2. **Data variables**: `reference`, `annotation`, and `hemispheres` are all present as
       data variables (a `hemispheres` stored as a coordinate is reported as missing).
    3. **Grid**: the three variables share identical dimensions, and those dimensions are a
       subset of `(z, y, x)` (2D or 3D, so a resampled single slice is accepted).
    4. **Data types**: `reference` is floating-point; `annotation` and `hemispheres` are
       integer-valued.
    5. **Attributes**: `attrs["structures"]` is present and is a brainglobe
       `StructuresDict`. The descriptive metadata the builder adds (`name`, `citation`,
       `species`, `orientation`) is not required.
    6. **Affines**: where two data variables both define an affine of the same name (in
       `attrs["affines"]`), the matrices must be equal — a mismatch means the variables
       are not on a common physical frame.
    7. **Mesh use** (only when `require_mesh_use` is set): `attrs["physical_to_base"]` — the
       pull mesh transform get_mesh needs — is present, and at least one structure
       references a mesh file that exists on disk.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate as an atlas.
    require_mesh_use : bool, default: False
        Whether to also require the machinery `get_mesh` needs: the `physical_to_base`
        transform attribute and at least one existing region mesh file.

    Raises
    ------
    TypeError
        If `ds` is not an `xarray.Dataset`, or if `reference` is not floating-point or
        `annotation`/`hemispheres` are not integer-valued.
    ValueError
        If any required data variable or attribute is missing, if the variables do not
        share dimensions that are a subset of `(z, y, x)`, if `attrs["structures"]` is not a
        brainglobe `StructuresDict`, or if `require_mesh_use` is set and `physical_to_base`
        or usable region meshes are absent.

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

    _validate_variable_affines(ds)

    # In memory the structures ride as a BrainGlobe StructuresDict (serialized to JSON
    # only inside the Zarr store); fetch_brainglobe_atlas and load_atlas both produce
    # one.
    from brainglobe_atlasapi.structure_class import StructuresDict

    if not isinstance(ds.attrs["structures"], StructuresDict):
        raise ValueError(
            "Atlas attribute 'structures' must be a brainglobe StructuresDict (as built "
            "by fetch_brainglobe_atlas or load_atlas), got "
            f"{type(ds.attrs['structures']).__name__}."
        )

    if require_mesh_use:
        if "physical_to_base" not in ds.attrs:
            raise ValueError(
                "Atlas Dataset is missing 'physical_to_base', required for mesh operations "
                "(require_mesh_use=True): it is the transform get_mesh uses to place mesh "
                "vertices in the atlas's physical space."
            )
        _validate_meshes_available(ds.attrs["structures"])
