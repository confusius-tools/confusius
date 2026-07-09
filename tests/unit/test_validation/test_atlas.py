"""Tests for validate_atlas_dataset."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

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
            "structures": json.dumps(structures),
            "mesh_to_physical": np.diag([1e-3, 1e-3, 1e-3, 1.0]).tolist(),
            "rl_midline_um": 100.0,
        },
    )


def test_valid_atlas_passes() -> None:
    validate_atlas_dataset(_make_atlas())


def test_valid_2d_atlas_passes() -> None:
    """A resampled single-slice atlas (2D) is accepted."""
    validate_atlas_dataset(_make_atlas(dims=("y", "x")))


def test_non_dataset_raises_type_error() -> None:
    with pytest.raises(TypeError, match="xarray.Dataset"):
        validate_atlas_dataset(xr.DataArray(np.zeros((3, 3, 3))))  # type: ignore[arg-type]


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
    del ds.attrs["rl_midline_um"]
    with pytest.raises(ValueError, match="rl_midline_um"):
        validate_atlas_dataset(ds)


def test_non_spatial_dims_raise() -> None:
    with pytest.raises(ValueError, match="subset"):
        validate_atlas_dataset(_make_atlas(dims=("z", "y", "w")))


def test_mismatched_dims_raise() -> None:
    ds = _make_atlas()
    # Give annotation a different (but still spatial) dim set than reference.
    ds = ds.assign(annotation=xr.DataArray(np.zeros((3, 3), dtype=np.int32), dims=("y", "x")))
    with pytest.raises(ValueError, match="share dimensions"):
        validate_atlas_dataset(ds)


def test_unparseable_structures_raises() -> None:
    ds = _make_atlas()
    ds.attrs["structures"] = "not valid json {"
    with pytest.raises(ValueError, match="structures"):
        validate_atlas_dataset(ds)


def test_structures_not_a_list_raises() -> None:
    ds = _make_atlas()
    ds.attrs["structures"] = json.dumps({"not": "a list"})
    with pytest.raises(ValueError, match="JSON list"):
        validate_atlas_dataset(ds)
