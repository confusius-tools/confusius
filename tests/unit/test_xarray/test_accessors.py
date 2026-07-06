"""Unit tests for xarray accessor."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

import confusius  # noqa: F401  # Import to register accessor.


class TestFUSIAccessor:
    """Tests for the fusi accessor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataArray for testing."""
        return xr.DataArray(
            [1, 10, 100, 1000],
            dims=["x"],
            coords={"x": [0, 1, 2, 3]},
            attrs={"description": "test data"},
        )

    def test_accessor_is_registered(self, sample_data):
        """Accessor is available on DataArrays."""
        assert hasattr(sample_data, "fusi")

    def test_scale_accessor_is_available(self, sample_data):
        """Scale accessor is available as a property."""
        assert hasattr(sample_data.fusi, "scale")
        from confusius.xarray.scale import FUSIScaleAccessor

        assert isinstance(sample_data.fusi.scale, FUSIScaleAccessor)

    def test_extract_accessor_is_available(self, sample_data):
        """Extract accessor is available as a property."""
        from confusius.xarray.extract import FUSIExtractAccessor

        assert isinstance(sample_data.fusi.extract, FUSIExtractAccessor)

    def test_save_dispatches_to_io_loadsave(self, sample_data, tmp_path):
        """save delegates to `confusius.io.loadsave.save`."""
        out = tmp_path / "recording.nii.gz"

        with patch("confusius.io.loadsave.save") as mock_save:
            sample_data.fusi.save(out)

        mock_save.assert_called_once_with(sample_data, out)

    def test_db_scale_factor_20(self, sample_data):
        """db_scale with factor=20 (amplitude)."""
        result = sample_data.fusi.scale.db(factor=20)

        expected = np.array([-60.0, -40.0, -20.0, 0.0])
        np.testing.assert_allclose(result.values, expected)

        assert result.dims == sample_data.dims
        np.testing.assert_array_equal(result.coords["x"], sample_data.coords["x"])

        assert result.attrs["units"] == "dB"
        assert "scaling" in result.attrs

    def test_db_scale_factor_10(self, sample_data):
        """db_scale with factor=10 (power, default)."""
        result = sample_data.fusi.scale.db(factor=10)

        expected = np.array([-30.0, -20.0, -10.0, 0.0])
        np.testing.assert_allclose(result.values, expected)

    def test_db_scale_default_factor(self, sample_data):
        """db_scale uses factor=10 by default."""
        result = sample_data.fusi.scale.db()

        expected = np.array([-30.0, -20.0, -10.0, 0.0])
        np.testing.assert_allclose(result.values, expected)

    def test_db_scale_with_complex_data(self):
        """db_scale handles complex data correctly."""
        data = xr.DataArray([1 + 0j, 3 + 4j, 0 + 5j])
        result = data.fusi.scale.db(factor=20)

        # Magnitude: [1, 5, 5], max=5.
        expected_magnitudes = np.array([1.0, 5.0, 5.0])
        expected_db = 20 * np.log10(expected_magnitudes / 5.0)

        np.testing.assert_allclose(result.values, expected_db)

    def test_db_scale_with_zero(self):
        """db_scale handles zeros (produces -inf)."""
        data = xr.DataArray([0, 1, 10])
        result = data.fusi.scale.db(factor=20)

        assert result.values[0] == -np.inf
        assert result.values[-1] == 0.0

    def test_log_scale(self, sample_data):
        """log_scale applies natural logarithm."""
        result = sample_data.fusi.scale.log()

        expected = np.log([1, 10, 100, 1000])
        np.testing.assert_allclose(result.values, expected)

        assert result.dims == sample_data.dims
        assert "scaling" in result.attrs

    def test_log_scale_with_zero(self):
        """log_scale handles zeros (produces -inf)."""
        data = xr.DataArray([0, 1, np.e])
        result = data.fusi.scale.log()

        assert result.values[0] == -np.inf
        assert np.isclose(result.values[1], 0.0)
        assert np.isclose(result.values[2], 1.0)

    def test_power_scale_sqrt(self, sample_data):
        """power_scale with default exponent=0.5 (square root)."""
        result = sample_data.fusi.scale.power()

        expected = np.sqrt([1, 10, 100, 1000])
        np.testing.assert_allclose(result.values, expected)

    def test_power_scale_square(self, sample_data):
        """power_scale with exponent=2 (square)."""
        result = sample_data.fusi.scale.power(exponent=2.0)

        expected = np.array([1, 100, 10000, 1000000])
        np.testing.assert_allclose(result.values, expected)

    def test_power_scale_preserves_metadata(self, sample_data):
        """power_scale preserves coordinates and updates attributes."""
        result = sample_data.fusi.scale.power(exponent=0.5)

        assert result.dims == sample_data.dims
        np.testing.assert_array_equal(result.coords["x"], sample_data.coords["x"])

        assert "scaling" in result.attrs
        assert "0.5" in result.attrs["scaling"]

    def test_power_scale_with_complex_data(self):
        """power_scale uses absolute value for complex data."""
        data = xr.DataArray([1 + 0j, 3 + 4j, 0 + 5j])
        result = data.fusi.scale.power(exponent=2.0)

        # Magnitudes: [1, 5, 5], squared: [1, 25, 25].
        expected = np.array([1.0, 25.0, 25.0])
        np.testing.assert_allclose(result.values, expected)

    def test_chained_operations(self, sample_data):
        """Multiple accessor operations can be chained."""
        result = sample_data.fusi.scale.power(exponent=0.5).fusi.scale.db(factor=20)

        # First sqrt: [1, 3.16, 10, 31.62].
        # Then dB relative to max (31.62).
        sqrt_vals = np.sqrt(sample_data.values)
        expected_db = 20 * np.log10(sqrt_vals / sqrt_vals.max())

        np.testing.assert_allclose(result.values, expected_db)


class TestOrigin:
    """Tests for fusi.origin."""

    def test_returns_correct_origin(self):
        """Returns the first coordinate value for each dimension."""
        data = xr.DataArray(
            np.zeros((10, 20)),
            dims=["y", "x"],
            coords={"y": np.arange(10) * 0.2 + 1.0, "x": np.arange(20) * 0.1 + 5.0},
        )
        assert data.fusi.origin == {"y": pytest.approx(1.0), "x": pytest.approx(5.0)}

    def test_zero_origin(self):
        """Returns 0.0 when coordinates start at zero."""
        data = xr.DataArray(
            np.zeros((10, 20)),
            dims=["y", "x"],
            coords={"y": np.arange(10) * 0.2, "x": np.arange(20) * 0.1},
        )
        assert data.fusi.origin == {"y": pytest.approx(0.0), "x": pytest.approx(0.0)}

    def test_missing_coord_warns_and_returns_zero(self):
        """Missing coordinate warns and falls back to 0.0."""
        data = xr.DataArray(np.zeros((5, 10)), dims=["y", "x"])
        with pytest.warns(UserWarning, match="no coordinate"):
            origin = data.fusi.origin
        assert origin["y"] == pytest.approx(0.0)
        assert origin["x"] == pytest.approx(0.0)

    def test_spacing_dim_order_preserved(self):
        """Returned dict keys follow DataArray dimension order."""
        data = xr.DataArray(
            np.zeros((5, 10, 20)),
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(5) * 0.5 + 2.0,
                "y": np.arange(10) * 0.2 + 1.0,
                "x": np.arange(20) * 0.1,
            },
        )
        assert list(data.fusi.origin.keys()) == ["z", "y", "x"]

    def test_single_point_coordinate(self):
        """Single-point coordinate returns its value as origin."""
        data = xr.DataArray(
            np.zeros((1, 10)),
            dims=["z", "x"],
            coords={"z": [3.5], "x": np.arange(10) * 0.1},
        )
        assert data.fusi.origin["z"] == pytest.approx(3.5)

    """Tests for fusi.spacing."""

    @pytest.fixture
    def uniform_2d_data(self):
        """2D DataArray with uniform spacing."""
        return xr.DataArray(
            np.zeros((10, 20)),
            dims=["y", "x"],
            coords={"y": np.arange(10) * 0.2, "x": np.arange(20) * 0.1},
        )

    def test_returns_correct_spacing(self, uniform_2d_data):
        """Returns correct spacing values for uniformly spaced coordinates."""
        assert uniform_2d_data.fusi.spacing == {
            "y": pytest.approx(0.2),
            "x": pytest.approx(0.1),
        }

    def test_includes_all_dims(self):
        """All dimensions, including time, are included."""
        data = xr.DataArray(
            np.zeros((5, 10)),
            dims=["time", "x"],
            coords={"time": np.arange(5) * 0.5, "x": np.arange(10) * 0.1},
        )
        assert data.fusi.spacing == {
            "time": pytest.approx(0.5),
            "x": pytest.approx(0.1),
        }

    def test_non_uniform_warns_and_returns_none(self):
        """Non-uniform coordinate raises UserWarning and returns None."""
        coords = np.array([0.0, 0.1, 0.3, 0.7])
        data = xr.DataArray(np.zeros(4), dims=["x"], coords={"x": coords})
        with pytest.warns(UserWarning, match="non-uniform"):
            spacing = data.fusi.spacing
        assert spacing["x"] is None

    def test_single_point_with_voxdim_uses_attr(self):
        """Single-point coordinate uses voxdim attribute when present."""
        y_coord = xr.DataArray([0.0], dims=["y"], attrs={"voxdim": 0.3})
        data = xr.DataArray(
            np.zeros((1, 10)),
            dims=["y", "x"],
            coords={"y": y_coord, "x": np.arange(10) * 0.1},
        )
        spacing = data.fusi.spacing
        assert spacing["y"] == pytest.approx(0.3)
        assert spacing["x"] == pytest.approx(0.1)

    def test_single_point_without_voxdim_warns_and_returns_none(self):
        """Single-point coordinate without voxdim warns and returns None."""
        data = xr.DataArray(
            np.zeros((1, 10)),
            dims=["y", "x"],
            coords={"y": [0.0], "x": np.arange(10) * 0.1},
        )
        with pytest.warns(UserWarning, match="single coordinate point"):
            spacing = data.fusi.spacing
        assert spacing["y"] is None
        assert spacing["x"] == pytest.approx(0.1)

    def test_dim_order_preserved(self):
        """Returned dict keys follow DataArray dimension order."""
        data = xr.DataArray(
            np.zeros((5, 10, 20)),
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(5) * 0.5,
                "y": np.arange(10) * 0.2,
                "x": np.arange(20) * 0.1,
            },
        )
        assert list(data.fusi.spacing.keys()) == ["z", "y", "x"]


class TestAffineToMethod:
    """Tests for fusi.affine.to."""

    @pytest.fixture
    def rng(self):
        """Seeded random number generator."""
        return np.random.default_rng(42)

    def _make_scan(
        self, affine: np.ndarray, via: str = "physical_to_lab"
    ) -> xr.DataArray:
        return xr.DataArray(
            np.zeros((4, 4, 4)),
            attrs={"affines": {via: affine}},
        )

    def test_identity_when_same_affine(self):
        """Returns identity when both scans share the same affine."""
        affine = np.eye(4)
        a = self._make_scan(affine)
        b = self._make_scan(affine)
        result = a.fusi.affine.to(b, via="physical_to_lab")
        np.testing.assert_allclose(result, np.eye(4), atol=1e-12)

    def test_known_relative_transform(self):
        """Returns inv(b_affine) @ a_affine for known matrices."""
        a_affine = np.array(
            [
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 1.0, 0.0, 3.0],
                [1.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        b_affine = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, -1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        a = self._make_scan(a_affine)
        b = self._make_scan(b_affine)
        expected = np.linalg.inv(b_affine) @ a_affine
        result = a.fusi.affine.to(b, via="physical_to_lab")
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_inverse_is_consistent(self, rng):
        """affine.to is consistent: a.affine.to(b) == inv(b.affine.to(a))."""

        # Build two random rotation+translation affines.
        def random_affine(rng: np.random.Generator) -> np.ndarray:
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            m = np.eye(4)
            m[:3, :3] = q
            m[:3, 3] = rng.standard_normal(3)
            return m

        shared = random_affine(rng)
        # Give both scans affines that go through a common lab frame.
        a_affine = shared @ random_affine(rng)
        b_affine = shared @ random_affine(rng)
        a = self._make_scan(a_affine)
        b = self._make_scan(b_affine)

        a_to_b = a.fusi.affine.to(b, via="physical_to_lab")
        b_to_a = b.fusi.affine.to(a, via="physical_to_lab")
        np.testing.assert_allclose(a_to_b, np.linalg.inv(b_to_a), atol=1e-12)

    def test_custom_via_key(self):
        """Works with a via key other than physical_to_lab."""
        affine = np.eye(4)
        a = self._make_scan(affine, via="physical_to_mri")
        b = self._make_scan(affine, via="physical_to_mri")
        result = a.fusi.affine.to(b, via="physical_to_mri")
        np.testing.assert_allclose(result, np.eye(4), atol=1e-12)

    def test_missing_affines_on_self_raises(self):
        """Raises ValueError when self has no affines in attrs."""
        a = xr.DataArray(np.zeros((2, 2)))
        b = self._make_scan(np.eye(4))
        with pytest.raises(ValueError, match="self does not have"):
            a.fusi.affine.to(b, via="physical_to_lab")

    def test_missing_affines_on_other_raises(self):
        """Raises ValueError when other has no affines in attrs."""
        a = self._make_scan(np.eye(4))
        b = xr.DataArray(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="other does not have"):
            a.fusi.affine.to(b, via="physical_to_lab")

    def test_missing_via_key_raises(self):
        """Raises KeyError when via key is absent from the affines dict."""
        a = self._make_scan(np.eye(4), via="physical_to_lab")
        b = self._make_scan(np.eye(4), via="physical_to_lab")
        with pytest.raises(KeyError):
            a.fusi.affine.to(b, via="nonexistent_key")

    def test_output_shape(self, rng):
        """Output is always a (4, 4) array."""
        affine = np.eye(4)
        a = self._make_scan(affine)
        b = self._make_scan(affine)
        result = a.fusi.affine.to(b, via="physical_to_lab")
        assert result.shape == (4, 4)


def _assert_affine_round_trip(result, orientation, applied, old):
    """Check `orientation @ new_physical == applied @ old_physical` at sample voxels.

    A 3D affine is fixed by four affinely independent points, so a handful of
    representative voxels (the origin, one step along each axis, plus an interior
    point) is a complete reference comparison.
    """
    iz, iy, ix = [0, 2, 0, 0, 1], [0, 0, 3, 0, 2], [0, 0, 0, 4, 3]
    ones = np.ones(len(iz))
    old_pts = np.stack([old["z"][iz], old["y"][iy], old["x"][ix], ones])
    new_pts = np.stack(
        [
            result.coords["z"].values[iz],
            result.coords["y"].values[iy],
            result.coords["x"].values[ix],
            ones,
        ]
    )
    np.testing.assert_allclose(orientation @ new_pts, applied @ old_pts, atol=1e-10)


class TestAffineApplyMethod:
    """Tests for fusi.affine.apply."""

    def _make_scan(
        self,
        shape: tuple[int, ...] = (3, 4, 5),
        dims: tuple[str, ...] = ("z", "y", "x"),
        spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
        origin: tuple[float, ...] = (0.0, 0.0, 0.0),
        affines: dict | None = None,
    ) -> xr.DataArray:
        coords = {
            dim: np.arange(n) * sp + orig
            for dim, n, sp, orig in zip(dims, shape, spacing, origin)
        }
        return xr.DataArray(
            np.zeros(shape),
            dims=list(dims),
            coords=coords,
            attrs={"affines": affines} if affines is not None else {},
        )

    def test_identity_leaves_coords_unchanged(self):
        """Applying the identity affine leaves coords unchanged and orientation I."""
        da = self._make_scan(origin=(1.0, 2.0, 3.0))
        result, orientation = da.fusi.affine.apply(np.eye(4))
        for dim in ("z", "y", "x"):
            np.testing.assert_allclose(result.coords[dim].values, da.coords[dim].values)
        np.testing.assert_allclose(orientation, np.eye(4))

    def test_pure_translation_shifts_coords(self):
        """A pure translation shifts all coordinate arrays by the given amount."""
        da = self._make_scan()
        shift = np.eye(4)
        shift[:3, 3] = [10.0, 5.0, -3.0]
        result, _ = da.fusi.affine.apply(shift)
        np.testing.assert_allclose(
            result.coords["z"].values, da.coords["z"].values + 10.0
        )
        np.testing.assert_allclose(
            result.coords["y"].values, da.coords["y"].values + 5.0
        )
        np.testing.assert_allclose(
            result.coords["x"].values, da.coords["x"].values - 3.0
        )

    def test_scaling_stretches_coords(self):
        """A diagonal scaling matrix scales coordinate values."""
        da = self._make_scan(spacing=(1.0, 1.0, 1.0))
        scale = np.diag([2.0, 3.0, 0.5, 1.0])
        result, _ = da.fusi.affine.apply(scale)
        np.testing.assert_allclose(
            result.coords["z"].values, da.coords["z"].values * 2.0
        )
        np.testing.assert_allclose(
            result.coords["y"].values, da.coords["y"].values * 3.0
        )
        np.testing.assert_allclose(
            result.coords["x"].values, da.coords["x"].values * 0.5
        )

    def test_single_axis_flip_negates_only_that_coord(self):
        """A sign flip on any single axis negates only that axis, orientation I.

        Regression: a y- or x-axis flip must negate y or x (not z). A diagonal
        affine is axis-aligned regardless of which axis carries the sign, so the
        axis-mixing test must read the off-diagonal block, not the decomposed
        zoom (decompose_affine relocates its one allowed sign flip onto axis 0).
        """
        base = self._make_scan(spacing=(1.0, 1.0, 1.0), origin=(1.0, 2.0, 3.0))
        for axis, flipped in enumerate(("z", "y", "x")):
            flip = np.eye(4)
            flip[axis, axis] = -1.0
            result, orientation = base.fusi.affine.apply(flip)
            for dim in ("z", "y", "x"):
                expected = (
                    -base.coords[dim].values
                    if dim == flipped
                    else base.coords[dim].values
                )
                np.testing.assert_allclose(result.coords[dim].values, expected)
            np.testing.assert_allclose(orientation, np.eye(4))

    def test_multi_axis_flip_negates_flipped_coords(self):
        """A multi-axis sign flip is diagonal: negates each flipped axis, orientation I."""
        da = self._make_scan(spacing=(1.0, 1.0, 1.0), origin=(1.0, 2.0, 3.0))
        flip = np.diag([-1.0, -1.0, 1.0, 1.0])
        result, orientation = da.fusi.affine.apply(flip)
        np.testing.assert_allclose(result.coords["z"].values, -da.coords["z"].values)
        np.testing.assert_allclose(result.coords["y"].values, -da.coords["y"].values)
        np.testing.assert_allclose(result.coords["x"].values, da.coords["x"].values)
        np.testing.assert_allclose(orientation, np.eye(4))

    def test_scaling_updates_voxdim_attrs(self):
        """A scaling affine rescales each coord's `voxdim` by the absolute zoom.

        Regression: `voxdim` used to be copied verbatim, leaving a stale voxel
        size after any zooming affine (and hence a wrong pixdim/qform when the
        scan is saved to NIfTI).
        """
        da = self._make_scan()
        voxdims = {"z": 0.1, "y": 0.2, "x": 0.3}
        for dim, voxdim in voxdims.items():
            da.coords[dim].attrs["voxdim"] = voxdim
        scale = np.diag([2.0, 3.0, -0.5, 1.0])
        result, _ = da.fusi.affine.apply(scale)
        for dim, zoom in zip(("z", "y", "x"), (2.0, 3.0, 0.5)):
            assert result.coords[dim].attrs["voxdim"] == pytest.approx(
                voxdims[dim] * zoom
            )
            # The input scan's voxdim must not be mutated.
            assert da.coords[dim].attrs["voxdim"] == voxdims[dim]

    def test_wrong_shape_raises_value_error(self):
        """Affines with shape other than (4, 4) raise ValueError."""
        da = self._make_scan()
        with pytest.raises(ValueError, match="shape"):
            da.fusi.affine.apply(np.eye(3))

    def test_string_key_looks_up_stored_affine(self):
        """A string `affine` is resolved from `attrs["affines"]` before applying."""
        shift = np.eye(4)
        shift[:3, 3] = [10.0, 5.0, -3.0]
        da = self._make_scan(affines={"physical_to_lab": shift})
        result, _ = da.fusi.affine.apply("physical_to_lab")
        np.testing.assert_allclose(
            result.coords["z"].values, da.coords["z"].values + 10.0
        )
        np.testing.assert_allclose(
            result.coords["y"].values, da.coords["y"].values + 5.0
        )
        np.testing.assert_allclose(
            result.coords["x"].values, da.coords["x"].values - 3.0
        )

    def test_string_key_missing_affines_attr_raises_value_error(self):
        """A string `affine` raises ValueError when `da` has no `affines` attr."""
        da = self._make_scan()
        with pytest.raises(ValueError, match="does not have an 'affines' entry"):
            da.fusi.affine.apply("physical_to_lab")

    def test_string_key_not_found_raises_key_error(self):
        """A string `affine` absent from `attrs["affines"]` raises KeyError."""
        da = self._make_scan(affines={"physical_to_lab": np.eye(4)})
        with pytest.raises(KeyError, match="physical_to_mri"):
            da.fusi.affine.apply("physical_to_mri")

    def test_stored_affines_updated_single(self):
        """Stored (4, 4) affines are updated by M_new = M_old @ inv(affine)."""
        stored = np.array(
            [
                [1.0, 0.0, 0.0, 5.0],
                [0.0, 1.0, 0.0, 6.0],
                [0.0, 0.0, 1.0, 7.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        da = self._make_scan(affines={"physical_to_lab": stored})
        shift = np.eye(4)
        shift[:3, 3] = [1.0, 2.0, 3.0]
        result, _ = da.fusi.affine.apply(shift)
        expected = stored @ np.linalg.inv(shift)
        np.testing.assert_allclose(
            result.attrs["affines"]["physical_to_lab"], expected, atol=1e-12
        )

    def test_stored_affines_updated_per_pose_stack(self):
        """Stored (npose, 4, 4) affines are updated per pose."""
        rng = np.random.default_rng(0)
        npose = 5
        # Build random per-pose affines (rotation + translation).
        stored = np.zeros((npose, 4, 4))
        for i in range(npose):
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            stored[i, :3, :3] = q
            stored[i, :3, 3] = rng.standard_normal(3)
            stored[i, 3, 3] = 1.0
        da = self._make_scan(affines={"physical_to_lab": stored})
        scale = np.diag([2.0, 1.0, 1.0, 1.0])
        result, _ = da.fusi.affine.apply(scale)
        inv_scale = np.linalg.inv(scale)
        expected = stored @ inv_scale
        np.testing.assert_allclose(
            result.attrs["affines"]["physical_to_lab"], expected, atol=1e-12
        )

    def test_partial_dims_only_updates_present_dims(self):
        """Only dimensions present in da.dims are updated."""
        # 2-D scan with only z and y (no x).
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["z", "y"],
            coords={"z": np.arange(3) * 1.0, "y": np.arange(4) * 1.0},
        )
        shift = np.eye(4)
        shift[:3, 3] = [10.0, 5.0, 99.0]  # x-shift should have no effect.
        result, _ = da.fusi.affine.apply(shift)
        np.testing.assert_allclose(
            result.coords["z"].values, da.coords["z"].values + 10.0
        )
        np.testing.assert_allclose(
            result.coords["y"].values, da.coords["y"].values + 5.0
        )
        assert "x" not in result.coords

    def test_unexpected_affine_shape_is_passed_through(self):
        """Unexpected stored affine shapes are kept unchanged."""
        weird = np.array([1.0, 2.0, 3.0])
        da = self._make_scan(affines={"weird": weird})

        result, _ = da.fusi.affine.apply(np.eye(4))

        np.testing.assert_allclose(result.attrs["affines"]["weird"], weird)

    def test_inplace_updates_and_returns_same_object(self):
        """inplace=True mutates and returns the original DataArray."""
        da = self._make_scan()
        original_z = da.coords["z"].values.copy()
        shift = np.eye(4)
        shift[0, 3] = 2.0

        returned, _ = da.fusi.affine.apply(shift, inplace=True)

        assert returned is da
        np.testing.assert_allclose(da.coords["z"].values, original_z + 2.0)

    def test_rotation_returns_round_trip_orientation(self):
        """A rotation absorbs zoom/translation into coords and returns the
        residual orientation, mapping new physical coords to the affine's world."""
        da = self._make_scan(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
        old = {d: da.coords[d].values.astype(float) for d in ("z", "y", "x")}
        # 90 deg rotation in the z-y plane, zoom [2, 3, 4], translation [1, 2, 3].
        affine = np.array(
            [
                [0.0, -3.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 4.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result, orientation = da.fusi.affine.apply(affine)
        _assert_affine_round_trip(result, orientation, affine, old)

    def test_shear_returns_round_trip_orientation(self):
        """A shear is axis-mixing (identity rotation, nonzero off-diagonal) and
        returns a residual orientation that round-trips."""
        da = self._make_scan(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
        old = {d: da.coords[d].values.astype(float) for d in ("z", "y", "x")}
        shear = np.eye(4)
        shear[0, 1] = 0.5  # z depends on y: off-diagonal, axis-mixing.
        result, orientation = da.fusi.affine.apply(shear)
        _assert_affine_round_trip(result, orientation, shear, old)

    def test_axis_permutation_does_not_introduce_spurious_coord_flip(self):
        """A mixing affine absorbs unsigned zoom only, leaving reflections in orientation."""
        da = self._make_scan(
            shape=(3, 4, 5),
            spacing=(1.0, 1.0, 1.0),
            origin=(10.0, 20.0, 30.0),
        )
        old = {d: da.coords[d].values.astype(float) for d in ("z", "y", "x")}
        affine = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result, orientation = da.fusi.affine.apply(affine)

        np.testing.assert_allclose(result.coords["z"].values, da.coords["z"].values)
        np.testing.assert_allclose(result.coords["y"].values, da.coords["y"].values)
        np.testing.assert_allclose(result.coords["x"].values, da.coords["x"].values)
        _assert_affine_round_trip(result, orientation, affine, old)

    def test_mixing_affine_reexpresses_existing_stored_affine(self):
        """Existing stored affines stay valid after an axis-mixing affine."""
        stored = np.eye(4)
        stored[:3, 3] = [5.0, 6.0, 7.0]
        da = self._make_scan(
            spacing=(1.0, 1.0, 1.0), affines={"physical_to_lab": stored}
        )
        old = {d: da.coords[d].values.astype(float) for d in ("z", "y", "x")}
        affine = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result, _ = da.fusi.affine.apply(affine)
        # lab position is frame-independent: M @ new_physical == stored @ old_physical.
        _assert_affine_round_trip(
            result, result.attrs["affines"]["physical_to_lab"], stored, old
        )
