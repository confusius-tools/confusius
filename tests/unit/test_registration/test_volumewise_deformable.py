"""Volumewise nonlinear registration and transform retention contracts."""

from threading import Event
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from confusius.registration import register_volume, register_volumewise, resample_like
from confusius.validation import validate_bspline, validate_displacement_field


@pytest.mark.parametrize("transform", ["translation", "bspline", "syn"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_matches_independent_volume_registration(
    sample_voxeldata_2dt_registration,
    transform,
    n_jobs,
):
    """Batch results and retained transforms agree with independent calls in time order."""
    data = sample_voxeldata_2dt_registration.isel(time=slice(0, 2)).copy()
    data.values[1] = np.roll(data.values[1], 1, axis=2)
    fixed = data.isel(time=0)
    masks = data > 0
    affines = np.repeat(np.eye(4)[None], 2, axis=0)
    affines[1, 2, 3] = 0.02
    options: dict[str, Any] = {
        "metric": "ants_cc",
        "mesh_size": (1, 3, 3),
        "correlation_radius": 2,
        "update_field_variance": 2.0,
        "total_field_variance": 0.2,
        "metric_sampling_percentage": 0.8,
        "metric_sampling_seed": 42,
        "number_of_iterations": 2,
        "learning_rate": 0.01,
    }
    result = register_volumewise(
        data,
        fixed=fixed,
        transform=transform,
        n_jobs=n_jobs,
        moving_mask=masks,
        fixed_mask=masks.isel(time=0),
        initialization=affines,
        keep_transforms=True,
        keep_diagnostics=True,
        show_progress=False,
        **options,
    )
    retained = result.attrs["registration_transforms"]
    for t in range(2):
        expected, expected_transform, diagnostics = register_volume(
            data.isel(time=t),
            fixed,
            transform_type=transform,
            moving_mask=masks.isel(time=t),
            fixed_mask=masks.isel(time=0),
            initialization=affines[t],
            itk_threads=1,
            **options,
        )
        assert_allclose(result.isel(time=t), expected)
        if transform == "bspline":
            validate_bspline(retained[t])
        elif transform == "syn":
            validate_displacement_field(retained[t])
        assert_allclose(retained[t], expected_transform)
        reapplied = resample_like(data.isel(time=t), fixed, retained[t], itk_threads=1)
        assert_allclose(reapplied, expected, atol=2e-5)
        assert_allclose(
            result.attrs["registration_diagnostics"][t].metric_values,
            diagnostics.metric_values,
        )
    summaries = result.attrs["motion_params"]
    assert_array_equal(summaries.index, data.time.values)
    if transform != "translation":
        assert set(summaries.columns) == {
            "status",
            "final_metric_value",
            "n_iterations",
        }
    else:
        assert {"rot_x", "trans_x", "mean_fd"} <= set(summaries.columns)


@pytest.mark.parametrize("transform", ["bspline", "syn"])
def test_estimation_only_and_opt_in_retention(
    sample_voxeldata_2dt_registration, transform
):
    """Estimation does not resample data, and repeated calls remove stale retained objects."""
    data = sample_voxeldata_2dt_registration.isel(time=slice(0, 2))
    affine = np.eye(4)
    affine[2, 3] = 0.05
    retained = data.fusi.register.volumewise(
        transform=transform,
        initialization=affine,
        number_of_iterations=1,
        mesh_size=(1, 3, 3),
        learning_rate=1e-9,
        n_jobs=1,
        resample=False,
        keep_transforms=True,
        keep_diagnostics=True,
        show_progress=False,
    )
    assert_array_equal(retained.values, data.values)
    assert len(retained.attrs["registration_transforms"]) == 2
    not_retained = register_volumewise(
        retained,
        transform=transform,
        number_of_iterations=1,
        mesh_size=(1, 3, 3),
        n_jobs=1,
        show_progress=False,
    )
    assert "registration_transforms" not in not_retained.attrs
    assert "registration_diagnostics" not in not_retained.attrs


@pytest.mark.parametrize("resample", [True, False])
def test_cancelled_frames_have_no_transform(
    sample_voxeldata_2dt_registration, resample
):
    """Unstarted frames have missing transforms, not fabricated identity deformations."""
    data = sample_voxeldata_2dt_registration
    event = Event()
    event.set()
    result = register_volumewise(
        data,
        transform="syn",
        abort_event=event,
        n_jobs=1,
        keep_transforms=True,
        show_progress=False,
        resample=resample,
    )
    assert result.attrs["registration_transforms"] == [None] * data.sizes["time"]
    assert set(result.attrs["motion_params"].status) == {"aborted"}
    expected = np.full_like(data.values, data.values.min()) if resample else data.values
    assert_array_equal(result.values, expected)


@pytest.mark.parametrize("shape", [(3, 3), (1, 4, 4), (2, 3, 3)])
def test_invalid_affine_initialization_shape(sample_voxeldata_2dt_registration, shape):
    """Per-frame initializations must cover the complete input time axis."""
    with pytest.raises(ValueError, match="initialization must have shape"):
        register_volumewise(
            sample_voxeldata_2dt_registration,
            initialization=np.zeros(shape),
            n_jobs=1,
            show_progress=False,
        )


def test_time_mask_must_match_recording(sample_voxeldata_2dt_registration):
    """Reject mismatched time masks instead of pairing masks by the wrong position."""
    data = sample_voxeldata_2dt_registration
    mask = (data > 0).assign_coords(time=data.time + 1)
    with pytest.raises(ValueError):
        register_volumewise(data, moving_mask=mask, n_jobs=1, show_progress=False)
