"""Tests for fUSI seed-based connectivity maps."""

import numpy as np
import pytest
import xarray as xr
from scipy.stats import pearsonr

from confusius.connectivity import SeedBasedMaps
from confusius.signal import clean


def test_seed_maps_match_pearson_reference(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """A seed map agrees with `scipy.stats.pearsonr` at a voxel."""
    mapper = SeedBasedMaps(seed_masks=sample_roi_labels).fit(sample_3dt_volume)

    seed = mapper.seed_signals_.sel(region=3).values
    voxel = sample_3dt_volume.isel(k=0, j=0, i=0).values
    expected, _ = pearsonr(seed, voxel)

    assert mapper.maps_.dims == ("region", "k", "j", "i")
    np.testing.assert_allclose(mapper.maps_.sel(region=3).isel(k=0, j=0, i=0), expected)


def test_seed_maps_support_stacked_labels(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Stacked seed masks preserve their region names."""
    labels = xr.concat(
        [
            (sample_roi_labels == 3).astype(np.int32),
            (sample_roi_labels == 7).astype(np.int32),
        ],
        dim=xr.IndexVariable("mask", ["motor", "somatosensory"]),
    )

    mapper = SeedBasedMaps(seed_masks=labels).fit(sample_3dt_volume)

    np.testing.assert_array_equal(mapper.maps_.region, ["motor", "somatosensory"])


def test_precomputed_seed_signal_matches_mask_seed(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Precomputed seed signals yield the corresponding connectivity map."""
    mask_mapper = SeedBasedMaps(seed_masks=sample_roi_labels).fit(sample_3dt_volume)
    signal_mapper = SeedBasedMaps(seed_signals=mask_mapper.seed_signals_).fit(
        sample_3dt_volume
    )

    xr.testing.assert_allclose(signal_mapper.maps_, mask_mapper.maps_)


def test_seed_maps_apply_mask(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Voxels outside a fUSI mask are zero after map reconstruction."""
    mask = sample_roi_labels > 0

    mapper = SeedBasedMaps(seed_masks=sample_roi_labels, mask=mask).fit(
        sample_3dt_volume
    )

    assert np.all(mapper.maps_.values[:, ~mask.values] == 0.0)


def test_seed_maps_keep_dask_results_lazy(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Dask-backed fUSI inputs produce lazy maps."""
    mapper = SeedBasedMaps(seed_masks=sample_roi_labels).fit(
        sample_3dt_volume.chunk({"time": 2})
    )

    assert hasattr(mapper.maps_.data, "chunks")


def test_seed_maps_apply_cleaning(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Cleaning produces the same maps as explicitly cleaned fUSI data."""
    expected = SeedBasedMaps(seed_masks=sample_roi_labels).fit(
        clean(sample_3dt_volume, detrend_order=1)
    )
    actual = SeedBasedMaps(
        seed_masks=sample_roi_labels,
        clean_kwargs={"detrend_order": 1},
    ).fit(sample_3dt_volume)

    xr.testing.assert_allclose(actual.maps_, expected.maps_)


def test_seed_maps_require_a_seed(sample_3dt_volume: xr.DataArray) -> None:
    """Exactly one seed source is required."""
    with pytest.raises(ValueError, match="neither"):
        SeedBasedMaps().fit(sample_3dt_volume)


def test_seed_maps_reject_multiple_seed_sources(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Masks and precomputed signals cannot be supplied together."""
    signals = xr.DataArray(
        np.ones(sample_3dt_volume.sizes["time"]),
        dims="time",
        coords={"time": sample_3dt_volume.time},
    )

    with pytest.raises(ValueError, match="both"):
        SeedBasedMaps(seed_masks=sample_roi_labels, seed_signals=signals).fit(
            sample_3dt_volume
        )


def test_seed_maps_reject_non_integer_labels(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Seed labels must have integer dtype."""
    with pytest.raises(TypeError, match="integer dtype"):
        SeedBasedMaps(seed_masks=sample_roi_labels.astype(float)).fit(sample_3dt_volume)


def test_seed_maps_reject_missing_spatial_coordinate(
    sample_3dt_volume: xr.DataArray,
    sample_roi_labels: xr.DataArray,
) -> None:
    """Input data must retain all fUSI spatial coordinates."""
    invalid = sample_3dt_volume.drop_vars("k")

    with pytest.raises(
        ValueError, match="Missing required coordinate for dimension 'k'"
    ):
        SeedBasedMaps(seed_masks=sample_roi_labels).fit(invalid)


def test_seed_maps_validate_precomputed_signal_time_coordinate(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Precomputed signals must share the fUSI time coordinate."""
    signals = xr.DataArray(
        np.ones(sample_3dt_volume.sizes["time"]),
        dims="time",
        coords={"time": sample_3dt_volume.time + 1.0},
    )

    with pytest.raises(ValueError, match="time coordinates"):
        SeedBasedMaps(seed_signals=signals).fit(sample_3dt_volume)


@pytest.mark.parametrize(
    ("signals", "message"),
    [
        (lambda data: xr.DataArray(np.ones(2), dims="region"), "must have a 'time'"),
        (
            lambda data: xr.DataArray(
                np.ones((data.sizes["time"], 2, 2)),
                dims=("time", "region", "extra"),
                coords={"time": data.time},
            ),
            "unexpected dimensions",
        ),
        (
            lambda data: xr.DataArray(np.ones(data.sizes["time"] + 1), dims="time"),
            "timepoints",
        ),
        (
            lambda data: xr.DataArray(
                np.ones(data.sizes["time"]),
                dims="time",
                coords={"time": data.time},
            ).chunk({"time": 2}),
            "chunked",
        ),
    ],
)
def test_seed_maps_validate_precomputed_signal_shape(
    sample_3dt_volume: xr.DataArray,
    signals,
    message: str,
) -> None:
    """Precomputed seed signals must have compatible time and feature dimensions."""
    with pytest.raises(ValueError, match=message):
        SeedBasedMaps(seed_signals=signals(sample_3dt_volume)).fit(sample_3dt_volume)
