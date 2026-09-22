"""Tests for validate_atlas."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from brainglobe_atlasapi.structure_class import StructuresDict

from confusius.validation import validate_atlas
from confusius.xarray import create_voxeldata


def _make_atlas(
    shape: tuple[int, int, int] = (3, 3, 3), mesh_filename: str | None = None
) -> xr.Dataset:
    """Build a minimal, schema-valid, canonical (indexed) atlas Dataset for validation tests.

    `shape` is `(k, j, i)`; pass a singleton `k` (e.g. `(1, 3, 3)`) for a resampled
    single-slice atlas. `mesh_filename` sets the root structure's mesh path (left
    `None`, i.e. no mesh, by default) so mesh-availability checks can be exercised.
    """
    structures = [
        {
            "id": 997,
            "acronym": "root",
            "name": "whole brain",
            "rgb_triplet": [200, 200, 200],
            "structure_id_path": [997],
            "mesh_filename": mesh_filename,
        }
    ]

    def mk(data: np.ndarray) -> xr.DataArray:
        return create_voxeldata(
            data, dims=["k", "j", "i"], spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
        )

    return xr.Dataset(
        {
            "reference": mk(np.ones(shape, dtype=np.float32)),
            "annotation": mk(np.zeros(shape, dtype=np.int32)),
            "hemispheres": mk(np.ones(shape, dtype=np.int8)),
        },
        attrs={
            "name": "mock",
            "citation": "Mock et al. (2026)",
            "species": "Mus musculus",
            "orientation": "asr",
            "structures": StructuresDict(structures),
            "world_to_base": np.eye(4),
        },
    )


def test_valid_atlas_passes() -> None:
    validate_atlas(_make_atlas())


def test_valid_single_slice_atlas_passes() -> None:
    """A resampled single-slice atlas (singleton k) is accepted."""
    validate_atlas(_make_atlas(shape=(1, 3, 3)))


def test_non_dataset_raises_type_error() -> None:
    with pytest.raises(TypeError, match="xarray.Dataset"):
        validate_atlas(xr.DataArray(np.zeros((3, 3, 3))))  # ty: ignore[invalid-argument-type]


def test_missing_data_var_raises() -> None:
    ds = _make_atlas().drop_vars("annotation")
    with pytest.raises(ValueError, match="annotation"):
        validate_atlas(ds)


def test_hemispheres_as_coordinate_reported_missing() -> None:
    """hemispheres modelled as a coordinate must fail (it must be a data variable)."""
    ds = _make_atlas()
    ds = ds.set_coords("hemispheres")
    with pytest.raises(ValueError, match="data variable"):
        validate_atlas(ds)


def test_reference_wrong_dtype_raises_type_error() -> None:
    ds = _make_atlas()
    ds["reference"] = ds["reference"].astype(np.int32)
    with pytest.raises(TypeError, match="reference"):
        validate_atlas(ds)


def test_annotation_wrong_dtype_raises_type_error() -> None:
    ds = _make_atlas()
    ds["annotation"] = ds["annotation"].astype(np.float32)
    with pytest.raises(TypeError, match="annotation"):
        validate_atlas(ds)


def test_missing_structures_attr_raises() -> None:
    """structures is the one unconditionally required attribute."""
    ds = _make_atlas()
    del ds.attrs["structures"]
    with pytest.raises(ValueError, match="structures"):
        validate_atlas(ds)


def test_missing_metadata_attrs_pass() -> None:
    """The descriptive metadata attrs are not required."""
    ds = _make_atlas()
    for attr in ("name", "citation", "species", "orientation"):
        del ds.attrs[attr]
    validate_atlas(ds)


def test_missing_world_to_base_passes_without_mesh_use() -> None:
    """world_to_base is not required unless validating for mesh use."""
    ds = _make_atlas()
    del ds.attrs["world_to_base"]
    validate_atlas(ds)


def test_require_mesh_use_passes(tmp_path) -> None:
    """An atlas with world_to_base and an existing mesh file passes mesh-use checks."""
    obj = tmp_path / "997.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    ds = _make_atlas(mesh_filename=str(obj))
    validate_atlas(ds, require_mesh_use=True)


def test_require_mesh_use_missing_world_to_base_raises(tmp_path) -> None:
    obj = tmp_path / "997.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    ds = _make_atlas(mesh_filename=str(obj))
    del ds.attrs["world_to_base"]
    with pytest.raises(ValueError, match="world_to_base"):
        validate_atlas(ds, require_mesh_use=True)


def test_require_mesh_use_without_meshes_raises() -> None:
    """An atlas whose structures reference no existing mesh fails mesh-use validation."""
    ds = _make_atlas()  # root structure has mesh_filename=None.
    with pytest.raises(ValueError, match="mesh"):
        validate_atlas(ds, require_mesh_use=True)


def test_matching_variable_affines_pass() -> None:
    """Two variables sharing an equal same-named affine are valid."""
    ds = _make_atlas()
    aff = np.eye(4)
    aff[0, 3] = 5.0
    ds["reference"].attrs["affines"] = {"world_to_sform": aff}
    ds["annotation"].attrs["affines"] = {"world_to_sform": aff.copy()}
    validate_atlas(ds)


def test_mismatched_variable_affines_raise() -> None:
    """Two variables disagreeing on a same-named affine are invalid."""
    ds = _make_atlas()
    ds["reference"].attrs["affines"] = {"world_to_sform": np.eye(4)}
    other = np.eye(4)
    other[0, 3] = 5.0
    ds["hemispheres"].attrs["affines"] = {"world_to_sform": other}
    with pytest.raises(ValueError, match="world_to_sform"):
        validate_atlas(ds)


_MOCK_STRUCTURES = [
    {
        "id": 997,
        "acronym": "root",
        "name": "whole brain",
        "rgb_triplet": [200, 200, 200],
        "structure_id_path": [997],
        "mesh_filename": None,
    }
]


def test_non_spatial_dims_raise() -> None:
    """A dim outside (k, j, i) fails the grid check, before an index is required."""
    ds = xr.Dataset(
        {
            "reference": xr.DataArray(
                np.ones((3, 3, 3), dtype=np.float32), dims=("k", "j", "w")
            ),
            "annotation": xr.DataArray(
                np.zeros((3, 3, 3), dtype=np.int32), dims=("k", "j", "w")
            ),
            "hemispheres": xr.DataArray(
                np.ones((3, 3, 3), dtype=np.int8), dims=("k", "j", "w")
            ),
        },
        attrs={"structures": StructuresDict(_MOCK_STRUCTURES)},
    )
    with pytest.raises(ValueError, match="subset"):
        validate_atlas(ds)


def test_missing_index_raises() -> None:
    """Variables on canonical voxel dims but without a VoxelToWorldIndex are invalid."""
    ds = xr.Dataset(
        {
            "reference": xr.DataArray(
                np.ones((3, 3, 3), dtype=np.float32), dims=("k", "j", "i")
            ),
            "annotation": xr.DataArray(
                np.zeros((3, 3, 3), dtype=np.int32), dims=("k", "j", "i")
            ),
            "hemispheres": xr.DataArray(
                np.ones((3, 3, 3), dtype=np.int8), dims=("k", "j", "i")
            ),
        },
        attrs={"structures": StructuresDict(_MOCK_STRUCTURES)},
    )
    with pytest.raises(ValueError, match="VoxelToWorldIndex"):
        validate_atlas(ds)


def test_mismatched_dims_raise() -> None:
    ds = _make_atlas()
    # Give annotation a different (but still valid voxel) dim set than reference.
    ds = ds.assign(
        annotation=xr.DataArray(np.zeros((3, 3), dtype=np.int32), dims=("j", "i"))
    )
    with pytest.raises(ValueError, match="share dimensions"):
        validate_atlas(ds)


def test_non_structuresdict_structures_raises() -> None:
    """structures stored as anything other than a StructuresDict is invalid."""
    ds = _make_atlas()
    ds.attrs["structures"] = [{"id": 997, "acronym": "root"}]
    with pytest.raises(ValueError, match="StructuresDict"):
        validate_atlas(ds)
