"""Unit tests for motion parameter estimation functions."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from confusius.registration.motion import (
    compute_framewise_displacement,
    create_motion_dataframe,
    extract_motion_parameters,
)


def _make_ref_da(size, spacing=1.0, ndim=2):
    """Create a spatial DataArray for FD reference."""
    import xarray as xr

    if ndim == 2:
        return xr.DataArray(
            np.zeros(size),
            dims=("y", "x"),
            coords={
                "y": np.arange(size[0]) * spacing,
                "x": np.arange(size[1]) * spacing,
            },
        )
    return xr.DataArray(
        np.zeros(size),
        dims=("z", "y", "x"),
        coords={
            "z": np.arange(size[0]) * spacing,
            "y": np.arange(size[1]) * spacing,
            "x": np.arange(size[2]) * spacing,
        },
    )


def _translation_affine_2d(tx, ty):
    """Return a (3, 3) 2D translation affine."""
    A = np.eye(3)
    A[0, 2] = tx
    A[1, 2] = ty
    return A


def _translation_affine_3d(tx, ty, tz):
    """Return a (4, 4) 3D translation affine."""
    A = np.eye(4)
    A[0, 2] = tx
    A[1, 2] = ty
    A[2, 2] = tz
    return A


class TestExtractMotionParameters:
    """Tests for extract_motion_parameters function."""

    def test_2d_translation_only(self):
        """2D pure translation extracts [0, tx, ty]."""
        affine = _translation_affine_2d(tx=2.0, ty=3.0)
        params = extract_motion_parameters([affine])

        assert params.shape == (1, 3)
        assert_allclose(params[0, 0], 0.0, atol=1e-6)  # no rotation
        assert_allclose(params[0, 1], 2.0, atol=1e-6)
        assert_allclose(params[0, 2], 3.0, atol=1e-6)

    def test_2d_identity_gives_zeros(self):
        """2D identity affine extracts all-zero parameters."""
        params = extract_motion_parameters([np.eye(3)])
        assert params.shape == (1, 3)
        assert_allclose(params[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_3d_translation_only(self):
        """3D pure translation extracts [0, 0, 0, tx, ty, tz]."""
        affine = np.eye(4)
        affine[:3, 3] = [1.0, 2.0, 3.0]
        params = extract_motion_parameters([affine])

        assert params.shape == (1, 6)
        assert_allclose(params[0, :3], [0.0, 0.0, 0.0], atol=1e-6)
        assert_allclose(params[0, 3:], [1.0, 2.0, 3.0], atol=1e-6)

    def test_3d_identity_gives_zeros(self):
        """3D identity affine extracts all-zero parameters."""
        params = extract_motion_parameters([np.eye(4)])
        assert params.shape == (1, 6)
        assert_allclose(params[0], np.zeros(6), atol=1e-6)

    def test_multiple_transforms(self):
        """Multiple affines return stacked parameters."""
        affines = [np.eye(3), _translation_affine_2d(2.0, 3.0)]
        params = extract_motion_parameters(affines)

        assert params.shape == (2, 3)
        assert_allclose(params[0], [0.0, 0.0, 0.0], atol=1e-6)
        assert_allclose(params[1, 1:], [2.0, 3.0], atol=1e-6)

    def test_none_affine_treated_as_identity(self):
        """None entry (e.g. B-spline) is treated as identity (zero motion)."""
        params = extract_motion_parameters([None, np.eye(3)])
        assert params.shape == (2, 3)
        assert_allclose(params[0], [0.0, 0.0, 0.0], atol=1e-6)


class TestComputeFramewiseDisplacement:
    """Tests for compute_framewise_displacement function."""

    def test_identical_transforms_zero_fd(self):
        """Identical affines produce zero framewise displacement."""
        ref = _make_ref_da((10, 10), spacing=1.0)
        affines = [np.eye(3), np.eye(3)]
        fd = compute_framewise_displacement(affines, ref)

        assert_allclose(fd["mean_fd"], [0.0, 0.0])
        assert_allclose(fd["max_fd"], [0.0, 0.0])
        assert_allclose(fd["rms_fd"], [0.0, 0.0])

    def test_known_translation_displacement_2d(self):
        """Known 2D translation produces correct FD (Euclidean distance)."""
        ref = _make_ref_da((10, 10), spacing=1.0)
        t1 = np.eye(3)
        t2 = _translation_affine_2d(3.0, 4.0)  # distance = 5.0
        fd = compute_framewise_displacement([t1, t2], ref)

        # Pure translation: all voxels displace equally.
        assert_allclose(fd["mean_fd"][0], 5.0, atol=1e-6)
        assert_allclose(fd["max_fd"][0], 5.0, atol=1e-6)
        assert_allclose(fd["rms_fd"][0], 5.0, atol=1e-6)
        assert fd["mean_fd"][-1] == 0.0

    def test_known_translation_displacement_3d(self):
        """Known 3D translation produces correct FD."""
        ref = _make_ref_da((8, 8, 8), spacing=1.0, ndim=3)
        t1 = np.eye(4)
        t2 = np.eye(4)
        t2[:3, 3] = [1.0, 2.0, 2.0]  # distance = 3.0
        fd = compute_framewise_displacement([t1, t2], ref)

        assert_allclose(fd["mean_fd"][0], 3.0, atol=1e-6)
        assert_allclose(fd["max_fd"][0], 3.0, atol=1e-6)

    def test_with_mask(self):
        """Mask restricts FD computation to masked voxels."""
        ref = _make_ref_da((10, 10), spacing=1.0)
        t1 = np.eye(3)
        t2 = _translation_affine_2d(3.0, 4.0)  # distance = 5.0
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        fd = compute_framewise_displacement([t1, t2], ref, mask=mask)

        # Pure translation: same displacement regardless of mask.
        assert_allclose(fd["mean_fd"][0], 5.0, atol=1e-6)

    def test_none_affine_treated_as_identity(self):
        """None affine treated as identity: displacement equals the other transform."""
        ref = _make_ref_da((10, 10), spacing=1.0)
        t1 = None  # identity
        t2 = _translation_affine_2d(3.0, 4.0)  # distance = 5.0
        fd = compute_framewise_displacement([t1, t2], ref)

        assert_allclose(fd["mean_fd"][0], 5.0, atol=1e-6)


class TestCreateMotionDataframe:
    """Tests for create_motion_dataframe function."""

    def test_2d_dataframe_columns(self):
        """2D affines produce DataFrame with correct columns."""
        ref = _make_ref_da((10, 10), spacing=1.0)
        affines = [np.eye(3), _translation_affine_2d(2.0, 3.0)]
        df = create_motion_dataframe(affines, ref)

        expected_cols = [
            "rotation",
            "trans_x",
            "trans_y",
            "mean_fd",
            "max_fd",
            "rms_fd",
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 2

    def test_3d_dataframe_columns(self):
        """3D affines produce DataFrame with correct columns."""
        ref = _make_ref_da((8, 8, 8), spacing=1.0, ndim=3)
        t1 = np.eye(4)
        t2 = np.eye(4)
        t2[:3, 3] = [1.0, 2.0, 3.0]
        df = create_motion_dataframe([t1, t2], ref)

        expected_cols = [
            "rot_x",
            "rot_y",
            "rot_z",
            "trans_x",
            "trans_y",
            "trans_z",
            "mean_fd",
            "max_fd",
            "rms_fd",
        ]
        assert list(df.columns) == expected_cols

    def test_time_coords_as_index(self):
        """Time coordinates are used as DataFrame index."""
        ref = _make_ref_da((10, 10), spacing=1.0)
        affines = [np.eye(3), np.eye(3)]
        time_coords = np.array([0.0, 0.5])
        df = create_motion_dataframe(affines, ref, time_coords=time_coords)

        assert df.index.name == "time"
        assert_array_equal(df.index, time_coords)

    def test_no_time_coords_uses_frame_index(self):
        """Without time coords, index is named 'frame'."""
        ref = _make_ref_da((10, 10), spacing=1.0)
        df = create_motion_dataframe([np.eye(3)], ref)
        assert df.index.name == "frame"

    def test_motion_values_correct(self):
        """Motion parameter values are correctly populated."""
        ref = _make_ref_da((10, 10), spacing=1.0)
        affines = [np.eye(3), _translation_affine_2d(2.0, 3.0)]
        df = create_motion_dataframe(affines, ref)

        assert_allclose(df.iloc[0]["rotation"], 0.0, atol=1e-6)
        assert_allclose(df.iloc[0]["trans_x"], 0.0, atol=1e-6)
        assert_allclose(df.iloc[0]["trans_y"], 0.0, atol=1e-6)

        assert_allclose(df.iloc[1]["rotation"], 0.0, atol=1e-6)
        assert_allclose(df.iloc[1]["trans_x"], 2.0, atol=1e-6)
        assert_allclose(df.iloc[1]["trans_y"], 3.0, atol=1e-6)
