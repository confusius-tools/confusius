"""Tests for label-based signal extraction."""

from typing import Any, Literal, cast

import numpy as np
import pytest
import xarray as xr

from confusius.extract import extract_with_labels


def test_extract_with_labels_reduces_each_region(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Flat labels produce one reduced signal per non-background region."""
    result = extract_with_labels(sample_3dt_volume, sample_roi_labels)

    assert result.dims == ("time", "region")
    np.testing.assert_array_equal(result.region.values, [3, 7, 42])
    expected = np.stack(
        [
            sample_3dt_volume.where(sample_roi_labels == label).mean(("z", "y", "x"))
            for label in (3, 7, 42)
        ],
        axis=1,
    )
    np.testing.assert_allclose(result.values, expected)


@pytest.mark.parametrize("reduction", ("sum", "median", "min", "max", "var", "std"))
def test_extract_with_labels_supports_reductions(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
    reduction: Literal["sum", "median", "min", "max", "var", "std"],
) -> None:
    """Each supported reduction is applied over the labelled voxels."""
    result = extract_with_labels(
        sample_3dt_volume, sample_roi_labels, reduction=reduction
    )
    expected = getattr(sample_3dt_volume.where(sample_roi_labels == 3), reduction)(
        ("z", "y", "x")
    )

    np.testing.assert_allclose(result.sel(region=3), expected)


def test_extract_with_labels_supports_stacked_masks(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Stacked label layers retain their named regions."""
    labels = xr.concat(
        [
            (sample_roi_labels == 3).astype(np.int32),
            (sample_roi_labels == 7).astype(np.int32),
        ],
        dim=xr.IndexVariable("mask", ["motor", "somatosensory"]),
    )

    result = extract_with_labels(sample_3dt_volume, labels)

    np.testing.assert_array_equal(result.region.values, ["motor", "somatosensory"])
    np.testing.assert_allclose(
        result.sel(region="motor"),
        sample_3dt_volume.where(sample_roi_labels == 3).mean(("z", "y", "x")),
    )
    np.testing.assert_allclose(
        result.sel(region="somatosensory"),
        sample_3dt_volume.where(sample_roi_labels == 7).mean(("z", "y", "x")),
    )


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (lambda labels: labels.astype(bool), "integer dtype"),
        (lambda labels: labels.astype(float), "integer dtype"),
    ],
)
def test_extract_with_labels_rejects_non_integer_labels(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
    labels,
    message: str,
) -> None:
    """Labels must use an integer dtype."""
    with pytest.raises(TypeError, match=message):
        extract_with_labels(sample_3dt_volume, labels(sample_roi_labels))


def test_extract_with_labels_rejects_misaligned_grid(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Labels from another fUSI grid are rejected."""
    labels = sample_roi_labels.assign_coords(y=sample_roi_labels.y + 1.0)

    with pytest.raises(ValueError, match="does not match between labels and data"):
        extract_with_labels(sample_3dt_volume, labels)


def test_extract_with_labels_supports_generic_feature_dimensions() -> None:
    """Extraction remains available for non-fUSI feature grids."""
    data = xr.DataArray(np.arange(12).reshape(3, 4), dims=("time", "feature"))
    labels = xr.DataArray([1, 0, 1, 2], dims="feature")

    result = extract_with_labels(data, labels, reduction="sum")

    assert result.dims == ("time", "region")
    np.testing.assert_array_equal(result.region.values, [1, 2])
    np.testing.assert_array_equal(result.values, [[2, 3], [10, 7], [18, 11]])


def test_extract_with_labels_restores_scalar_indexed_spatial_dim(
    sample_3dt_volume_with_scalar_z: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Scalar-indexed data and labels are restored before reduction."""
    labels = sample_roi_labels.isel(z=0)

    result = extract_with_labels(sample_3dt_volume_with_scalar_z, labels)

    assert result.dims == ("time", "region")


def test_extract_with_labels_preserves_dask_laziness(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Dask-backed fUSI data remain lazy during regional reduction."""
    chunked = sample_3dt_volume.chunk({"time": 2})

    result = extract_with_labels(chunked, sample_roi_labels)

    assert hasattr(result.data, "chunks")
    xr.testing.assert_allclose(
        result.compute(), extract_with_labels(sample_3dt_volume, sample_roi_labels)
    )


def test_extract_with_labels_rejects_unknown_reduction(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Unsupported reductions fail before data processing."""
    with pytest.raises(ValueError, match="Invalid reduction"):
        extract_with_labels(
            sample_3dt_volume, sample_roi_labels, reduction=cast(Any, "mode")
        )
