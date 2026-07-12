"""Unit tests for motion parameter estimation functions."""

from typing import Any, cast

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from confusius.registration.motion import (
    _validate_affines,
    compute_framewise_displacement,
    create_motion_dataframe,
    extract_motion_parameters,
)


def _with_spatial_dims(reference, dims):
    """Return `reference` with reordered spatial dim names and matching coords."""
    return reference.rename(dict(zip(reference.dims, dims, strict=True))).transpose(
        *dims
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
    A[:3, 3] = [tx, ty, tz]
    return A


def _rotation_affine_3d_first_axis(angle):
    """Return a (4, 4) 3D rotation affine around the first axis."""
    A = np.eye(4)
    c = np.cos(angle)
    s = np.sin(angle)
    A[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )
    return A


def _with_singleton_dim(reference, dim):
    """Return `reference` with `dim` reduced to length 1 but preserved."""
    reduced = reference.isel({dim: [0]})
    reduced.coords[dim].attrs["voxdim"] = 1.0
    return reduced


class TestValidateAffines:
    """Tests for affine validation used by motion diagnostics."""

    def test_rejects_empty_affine_list(self):
        """At least one affine matrix is required."""
        import pytest

        with pytest.raises(ValueError, match="at least one"):
            _validate_affines([])

    def test_rejects_non_numpy_affine(self):
        """Affine entries must already be numpy arrays."""
        import pytest

        with pytest.raises(TypeError, match="numpy.ndarray"):
            _validate_affines(cast(Any, [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

    def test_rejects_non_square_affine_array(self):
        """Affine matrices must be square 2D arrays."""
        import pytest

        with pytest.raises(ValueError, match="square 2D array"):
            _validate_affines([np.zeros((3, 4))])

    def test_rejects_wrong_affine_shape(self):
        """Only homogeneous 2D and 3D affine shapes are accepted."""
        import pytest

        with pytest.raises(ValueError, match=r"shape \(3, 3\) or \(4, 4\)"):
            _validate_affines([np.eye(5)])

    def test_rejects_mixed_affine_dimensionality(self):
        """All affine matrices must have the same dimensionality."""
        import pytest

        with pytest.raises(ValueError, match="same dimensionality"):
            _validate_affines([np.eye(3), np.eye(4)])


class TestExtractMotionParameters:
    """Tests for extract_motion_parameters function."""

    def test_2d_translation_only(self):
        """2D pure translation extracts [0, tx, ty]."""
        affine = _translation_affine_2d(tx=2.0, ty=3.0)
        params = extract_motion_parameters([affine])

        assert params.shape == (1, 3)
        assert_allclose(params[0, 0], 0.0, atol=1e-6)
        assert_allclose(params[0, 1], 2.0, atol=1e-6)
        assert_allclose(params[0, 2], 3.0, atol=1e-6)

    def test_2d_identity_gives_zeros(self):
        """2D identity affine extracts all-zero parameters."""
        params = extract_motion_parameters([np.eye(3)])
        assert params.shape == (1, 3)
        assert_allclose(params[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_2d_reflection_affine_keeps_valid_rotation_and_translation(self):
        """2D reflection-like affines still return stable motion parameters."""
        affine = np.eye(3)
        affine[0, 0] = -1.0
        affine[0, 2] = 1.0
        affine[1, 2] = 2.0

        params = extract_motion_parameters([affine])

        assert_allclose(params[0], [0.0, 1.0, 2.0], atol=1e-6)

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


class TestComputeFramewiseDisplacement:
    """Tests for compute_framewise_displacement function."""

    def test_identical_transforms_zero_fd(self, sample_2d_dataarray_spatial):
        """Identical affines produce zero framewise displacement."""
        affines = [np.eye(3), np.eye(3)]
        fd = compute_framewise_displacement(affines, sample_2d_dataarray_spatial)

        assert_allclose(fd["mean_fd"], [0.0, 0.0])
        assert_allclose(fd["max_fd"], [0.0, 0.0])
        assert_allclose(fd["rms_fd"], [0.0, 0.0])

    def test_known_translation_displacement_2d(self, sample_2d_dataarray_spatial):
        """Known 2D translation produces correct FD (Euclidean distance)."""
        t1 = np.eye(3)
        t2 = _translation_affine_2d(3.0, 4.0)
        fd = compute_framewise_displacement([t1, t2], sample_2d_dataarray_spatial)

        assert_allclose(fd["mean_fd"][0], 5.0, atol=1e-6)
        assert_allclose(fd["max_fd"][0], 5.0, atol=1e-6)
        assert_allclose(fd["rms_fd"][0], 5.0, atol=1e-6)
        assert fd["mean_fd"][-1] == 0.0

    def test_known_translation_displacement_3d(self, sample_3d_dataarray_spatial):
        """Known 3D translation produces correct FD."""
        t1 = np.eye(4)
        t2 = np.eye(4)
        t2[:3, 3] = [1.0, 2.0, 2.0]
        fd = compute_framewise_displacement([t1, t2], sample_3d_dataarray_spatial)

        assert_allclose(fd["mean_fd"][0], 3.0, atol=1e-6)
        assert_allclose(fd["max_fd"][0], 3.0, atol=1e-6)

    def test_with_mask(self, sample_2d_dataarray_spatial):
        """Mask restricts FD computation to masked voxels."""
        t1 = np.eye(3)
        t2 = _translation_affine_2d(3.0, 4.0)
        mask = np.zeros(sample_2d_dataarray_spatial.shape, dtype=bool)
        mask[2:8, 2:8] = True
        fd = compute_framewise_displacement(
            [t1, t2], sample_2d_dataarray_spatial, mask=mask
        )

        assert_allclose(fd["mean_fd"][0], 5.0, atol=1e-6)


class TestCreateMotionDataframe:
    """Tests for create_motion_dataframe function."""

    def test_2d_dataframe_columns(self, sample_2d_dataarray_spatial):
        """2D affines produce DataFrame with correct columns."""
        affines = [np.eye(3), _translation_affine_2d(2.0, 3.0)]
        df = create_motion_dataframe(affines, sample_2d_dataarray_spatial)

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

    def test_singleton_z_dataframe_uses_effective_2d_columns(
        self, sample_3d_dataarray_spatial
    ):
        """A singleton z axis collapses to effective 2D motion columns."""
        ref = _with_singleton_dim(sample_3d_dataarray_spatial, "z")
        df = create_motion_dataframe(
            [np.eye(4), _translation_affine_3d(1.0, 2.0, 3.0)], ref
        )

        assert list(df.columns) == [
            "rotation",
            "trans_x",
            "trans_y",
            "mean_fd",
            "max_fd",
            "rms_fd",
        ]
        assert_allclose(df.iloc[1]["trans_x"], 3.0, atol=1e-6)
        assert_allclose(df.iloc[1]["trans_y"], 2.0, atol=1e-6)

    def test_singleton_x_dataframe_uses_effective_2d_columns(
        self, sample_3d_dataarray_spatial
    ):
        """Any singleton spatial axis collapses to effective 2D motion columns."""
        ref = _with_singleton_dim(sample_3d_dataarray_spatial, "x")
        df = create_motion_dataframe(
            [np.eye(4), _translation_affine_3d(1.0, 2.0, 3.0)], ref
        )

        assert list(df.columns) == [
            "rotation",
            "trans_y",
            "trans_z",
            "mean_fd",
            "max_fd",
            "rms_fd",
        ]
        assert_allclose(df.iloc[1]["trans_y"], 2.0, atol=1e-6)
        assert_allclose(df.iloc[1]["trans_z"], 1.0, atol=1e-6)

    def test_too_many_singleton_axes_raise_error(self, sample_3d_dataarray_spatial):
        """Effective motion summaries need at least two non-singleton spatial axes."""
        import pytest

        ref = _with_singleton_dim(
            _with_singleton_dim(sample_3d_dataarray_spatial, "z"), "y"
        )

        with pytest.raises(ValueError, match="at least two non-singleton"):
            create_motion_dataframe([np.eye(4)], ref)

    def test_3d_dataframe_columns(self, sample_3d_dataarray_spatial):
        """3D affines produce DataFrame with correct columns."""
        t1 = np.eye(4)
        t2 = np.eye(4)
        t2[:3, 3] = [1.0, 2.0, 3.0]
        df = create_motion_dataframe([t1, t2], sample_3d_dataarray_spatial)

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

    def test_2d_dataframe_reorders_translations_to_named_axes(
        self, sample_2d_dataarray_spatial
    ):
        """2D translations follow x/y names, not raw axis order."""
        ref = _with_spatial_dims(sample_2d_dataarray_spatial, ("y", "x"))
        df = create_motion_dataframe([np.eye(3), _translation_affine_2d(2.0, 3.0)], ref)

        assert_allclose(df.iloc[1]["trans_x"], 3.0, atol=1e-6)
        assert_allclose(df.iloc[1]["trans_y"], 2.0, atol=1e-6)

    def test_3d_dataframe_reorders_translations_to_named_axes(
        self, sample_3d_dataarray_spatial
    ):
        """3D translations follow x/y/z names, not raw axis order."""
        ref = _with_spatial_dims(sample_3d_dataarray_spatial, ("z", "y", "x"))
        df = create_motion_dataframe(
            [np.eye(4), _translation_affine_3d(1.0, 2.0, 3.0)], ref
        )

        assert_allclose(df.iloc[1]["trans_x"], 3.0, atol=1e-6)
        assert_allclose(df.iloc[1]["trans_y"], 2.0, atol=1e-6)
        assert_allclose(df.iloc[1]["trans_z"], 1.0, atol=1e-6)

    def test_3d_dataframe_reorders_rotations_to_named_axes(
        self, sample_3d_dataarray_spatial
    ):
        """3D rotations follow x/y/z names, not raw axis order."""
        ref = _with_spatial_dims(sample_3d_dataarray_spatial, ("z", "y", "x"))
        angle = 0.1
        df = create_motion_dataframe(
            [np.eye(4), _rotation_affine_3d_first_axis(angle)], ref
        )

        assert_allclose(df.iloc[1]["rot_x"], 0.0, atol=1e-6)
        assert_allclose(df.iloc[1]["rot_y"], 0.0, atol=1e-6)
        assert_allclose(df.iloc[1]["rot_z"], angle, atol=1e-6)

    def test_rejects_unexpected_parameter_count(
        self, monkeypatch, sample_2d_dataarray_spatial
    ):
        """Unexpected motion-parameter widths are rejected."""
        import pytest

        monkeypatch.setattr(
            "confusius.registration.motion.extract_motion_parameters",
            lambda affines: np.zeros((1, 4)),
        )
        monkeypatch.setattr(
            "confusius.registration.motion.compute_framewise_displacement",
            lambda affines, reference, mask=None: {
                "mean_fd": np.zeros(1),
                "max_fd": np.zeros(1),
                "rms_fd": np.zeros(1),
            },
        )

        with pytest.raises(ValueError, match="3 or 6 columns"):
            create_motion_dataframe([np.eye(3)], sample_2d_dataarray_spatial)

    def test_time_coords_as_index(self, sample_2d_dataarray_spatial):
        """Time coordinates are used as DataFrame index."""
        affines = [np.eye(3), np.eye(3)]
        time_coords = np.array([0.0, 0.5])
        df = create_motion_dataframe(
            affines, sample_2d_dataarray_spatial, time_coords=time_coords
        )

        assert df.index.name == "time"
        assert_array_equal(df.index, time_coords)

    def test_no_time_coords_uses_frame_index(self, sample_2d_dataarray_spatial):
        """Without time coords, index is named 'frame'."""
        df = create_motion_dataframe([np.eye(3)], sample_2d_dataarray_spatial)
        assert df.index.name == "frame"

    def test_motion_values_correct(self, sample_2d_dataarray_spatial):
        """Motion parameter values are correctly populated."""
        ref = _with_spatial_dims(sample_2d_dataarray_spatial, ("x", "y"))
        affines = [np.eye(3), _translation_affine_2d(2.0, 3.0)]
        df = create_motion_dataframe(affines, ref)

        assert_allclose(df.iloc[0]["rotation"], 0.0, atol=1e-6)
        assert_allclose(df.iloc[0]["trans_x"], 0.0, atol=1e-6)
        assert_allclose(df.iloc[0]["trans_y"], 0.0, atol=1e-6)

        assert_allclose(df.iloc[1]["rotation"], 0.0, atol=1e-6)
        assert_allclose(df.iloc[1]["trans_x"], 2.0, atol=1e-6)
        assert_allclose(df.iloc[1]["trans_y"], 3.0, atol=1e-6)
