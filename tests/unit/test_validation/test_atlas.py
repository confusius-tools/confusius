"""Tests for validate_atlas_dataset."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from brainglobe_atlasapi.structure_class import StructuresDict

from confusius.validation import validate_atlas_dataset


def _make_atlas(dims: tuple[str, ...] = ("z", "y", "x")) -> xr.Dataset:
    """Build a minimal, schema-valid atlas Dataset for validation tests."""
    shape = (3,) * len(dims)
    structures = [
        {
            "id": 997,
            "acronym": "root",
            "name": "whole brain",
            "rgb_triplet": [200, 200, 200],
            "structure_id_path": [997],
            "mesh_filename": None,
        }
    ]
    mk = lambda data: xr.DataArray(data, dims=dims)  # noqa: E731
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
            "affines": {"base_to_current": np.eye(4)},
        },
    )


def test_valid_atlas_passes() -> None:
    validate_atlas_dataset(_make_atlas())


def test_valid_2d_atlas_passes() -> None:
    """A resampled single-slice atlas (2D) is accepted."""
    validate_atlas_dataset(_make_atlas(dims=("y", "x")))


def test_non_dataset_raises_type_error() -> None:
    with pytest.raises(TypeError, match="xarray.Dataset"):
        validate_atlas_dataset(xr.DataArray(np.zeros((3, 3, 3))))  # ty: ignore[invalid-argument-type]


def test_missing_data_var_raises() -> None:
    ds = _make_atlas().drop_vars("annotation")
    with pytest.raises(ValueError, match="annotation"):
        validate_atlas_dataset(ds)


def test_hemispheres_as_coordinate_reported_missing() -> None:
    """hemispheres modelled as a coordinate must fail (it must be a data variable)."""
    ds = _make_atlas()
    ds = ds.set_coords("hemispheres")
    with pytest.raises(ValueError, match="data variable"):
        validate_atlas_dataset(ds)


def test_reference_wrong_dtype_raises_type_error() -> None:
    ds = _make_atlas()
    ds["reference"] = ds["reference"].astype(np.int32)
    with pytest.raises(TypeError, match="reference"):
        validate_atlas_dataset(ds)


def test_annotation_wrong_dtype_raises_type_error() -> None:
    ds = _make_atlas()
    ds["annotation"] = ds["annotation"].astype(np.float32)
    with pytest.raises(TypeError, match="annotation"):
        validate_atlas_dataset(ds)


def test_missing_attr_raises() -> None:
    ds = _make_atlas()
    del ds.attrs["citation"]
    with pytest.raises(ValueError, match="citation"):
        validate_atlas_dataset(ds)


def test_base_to_current_as_data_var_passes() -> None:
    """A nonlinear atlas carries base_to_current as a data var, not an affine attr."""
    ds = _make_atlas()
    del ds.attrs["affines"]
    ds["base_to_current"] = xr.DataArray(
        np.zeros((3, 3, 3, 3)), dims=("component", "z", "y", "x")
    )
    validate_atlas_dataset(ds)


def test_missing_base_to_current_raises() -> None:
    ds = _make_atlas()
    del ds.attrs["affines"]
    with pytest.raises(ValueError, match="base_to_current"):
        validate_atlas_dataset(ds)


def test_matching_variable_affines_pass() -> None:
    """Two variables sharing an equal same-named affine are valid."""
    ds = _make_atlas()
    aff = np.eye(4)
    aff[0, 3] = 5.0
    ds["reference"].attrs["affines"] = {"physical_to_sform": aff}
    ds["annotation"].attrs["affines"] = {"physical_to_sform": aff.copy()}
    validate_atlas_dataset(ds)


def test_mismatched_variable_affines_raise() -> None:
    """Two variables disagreeing on a same-named affine are invalid."""
    ds = _make_atlas()
    ds["reference"].attrs["affines"] = {"physical_to_sform": np.eye(4)}
    other = np.eye(4)
    other[0, 3] = 5.0
    ds["hemispheres"].attrs["affines"] = {"physical_to_sform": other}
    with pytest.raises(ValueError, match="physical_to_sform"):
        validate_atlas_dataset(ds)


def test_non_spatial_dims_raise() -> None:
    with pytest.raises(ValueError, match="subset"):
        validate_atlas_dataset(_make_atlas(dims=("z", "y", "w")))


def test_mismatched_dims_raise() -> None:
    ds = _make_atlas()
    # Give annotation a different (but still spatial) dim set than reference.
    ds = ds.assign(
        annotation=xr.DataArray(np.zeros((3, 3), dtype=np.int32), dims=("y", "x"))
    )
    with pytest.raises(ValueError, match="share dimensions"):
        validate_atlas_dataset(ds)


def test_non_structuresdict_structures_raises() -> None:
    """structures stored as anything other than a StructuresDict is invalid."""
    ds = _make_atlas()
    ds.attrs["structures"] = [{"id": 997, "acronym": "root"}]
    with pytest.raises(ValueError, match="StructuresDict"):
        validate_atlas_dataset(ds)
