"""Unit tests for xarray accessor."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

import confusius  # noqa: F401  # Import to register accessor.
from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
)


def _make_voxel_to_world_volume() -> xr.DataArray:
    """Create a small voxel-to-world volume for accessor tests."""
    base = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=["k", "j", "i"],
        coords={
            "k": [0, 1],
            "j": [0, 2, 4],
            "i": [0, 1, 2, 3],
        },
    )
    return attach_voxel_to_world_index(
        base,
        np.array(
            [
                [2.0, 1.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        world_coord_attrs={
            "z": {"units": "mm"},
            "y": {"units": "mm"},
            "x": {"units": "mm"},
        },
    )


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

    def test_origin_raises_without_voxel_to_world_index(self):
        """A plain DataArray without a voxel-to-world index raises ValueError."""
        data = xr.DataArray(
            np.zeros((10, 20)),
            dims=["y", "x"],
            coords={"y": np.arange(10) * 0.2, "x": np.arange(20) * 0.1},
        )
        with pytest.raises(ValueError, match="voxel-to-world index"):
            _ = data.fusi.origin

    def test_voxel_to_world_origin_uses_first_sampled_voxel(self):
        """Voxel-to-world origin is the world location of array index zero."""
        data = _make_voxel_to_world_volume().expand_dims(time=[0.0, 0.5])

        assert data.fusi.origin == {
            "time": pytest.approx(0.0),
            "z": pytest.approx(10.0),
            "y": pytest.approx(20.0),
            "x": pytest.approx(30.0),
        }

    def test_voxel_to_world_origin_respects_nonzero_voxel_coords(self):
        """Voxel-to-world origin uses the first sampled voxel coords, not affine translation."""
        base = xr.DataArray(
            np.zeros((2, 3, 4)),
            dims=["k", "j", "i"],
            coords={
                "k": [10, 11],
                "j": [5, 7, 9],
                "i": [100, 101, 102, 103],
            },
        )
        data = attach_voxel_to_world_index(
            base,
            np.array(
                [
                    [2.0, 0.0, 0.0, 10.0],
                    [0.0, 3.0, 0.0, 20.0],
                    [0.0, 0.0, 4.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        assert data.fusi.origin == {
            "z": pytest.approx(30.0),
            "y": pytest.approx(35.0),
            "x": pytest.approx(430.0),
        }

    """Tests for fusi.spacing."""

    def test_spacing_raises_without_voxel_to_world_index(self):
        """A plain DataArray without a voxel-to-world index raises ValueError."""
        data = xr.DataArray(
            np.zeros((10, 20)),
            dims=["y", "x"],
            coords={"y": np.arange(10) * 0.2, "x": np.arange(20) * 0.1},
        )
        with pytest.raises(ValueError, match="voxel-to-world index"):
            _ = data.fusi.spacing

    def test_voxel_to_world_spacing_uses_world_step_lengths(self):
        """Voxel-to-world spacing comes from voxel steps and affine column norms."""
        data = _make_voxel_to_world_volume().expand_dims(time=[0.0, 0.5])

        assert data.fusi.spacing == {
            "time": pytest.approx(0.5),
            "k": pytest.approx(2.0),
            "j": pytest.approx(2.0 * np.sqrt(10.0)),
            "i": pytest.approx(4.0),
        }

    def test_voxel_to_world_direction_returns_orientation_matrix(self):
        """Voxel-to-world direction is the normalized affine linear part."""
        data = _make_voxel_to_world_volume()

        np.testing.assert_allclose(
            data.fusi.direction,
            np.array(
                [
                    [1.0, 1.0 / np.sqrt(10.0), 0.0],
                    [0.0, 3.0 / np.sqrt(10.0), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )


class TestReindexVoxels:
    """Tests for fusi.reindex_voxels."""

    def test_rebases_voxel_coords_to_dense_positions(self):
        """Voxel coordinates become 0, 1, ..., dim - 1 after reindexing."""
        data = _make_voxel_to_world_volume()
        result = data.fusi.reindex_voxels()
        for dim in ("k", "j", "i"):
            np.testing.assert_array_equal(
                result.coords[dim].values, np.arange(data.sizes[dim], dtype=float)
            )

    def test_preserves_world_coordinates(self):
        """World (z/y/x) coordinates are unchanged by reindexing."""
        data = _make_voxel_to_world_volume()
        result = data.fusi.reindex_voxels()
        for name in ("z", "y", "x"):
            np.testing.assert_allclose(
                result.coords[name].values, data.coords[name].values
            )

    def test_preserves_data_values(self):
        """Array content is unchanged by reindexing."""
        data = _make_voxel_to_world_volume()
        data.values[:] = np.arange(data.size).reshape(data.shape)
        result = data.fusi.reindex_voxels()
        np.testing.assert_array_equal(result.values, data.values)

    def test_affine_maps_dense_positions_to_world(self):
        """The rebuilt affine maps position (0, 0, 0) to the array's actual origin."""
        data = _make_voxel_to_world_volume()
        cropped = data.isel(k=slice(1, 2), j=slice(1, 3), i=slice(2, 4))
        result = cropped.fusi.reindex_voxels()
        affine = get_voxel_to_world_affine(result)
        origin = affine @ np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            origin[:3],
            [cropped.fusi.origin[name] for name in ("z", "y", "x")],
        )

    def test_raises_without_voxel_to_world_geometry(self):
        """A plain DataArray without voxel-to-world geometry raises ValueError."""
        data = xr.DataArray(np.zeros((2, 3)), dims=["j", "i"])
        with pytest.raises(ValueError, match="voxel-to-world index"):
            data.fusi.reindex_voxels()

    def test_raises_when_spacing_undefined(self):
        """Irregular voxel-space coordinates without defined spacing raise ValueError."""
        base = xr.DataArray(
            np.zeros((3, 4)),
            dims=["j", "i"],
            coords={"j": [0, 1, 3], "i": np.arange(4)},
        )
        data = attach_voxel_to_world_index(base, np.eye(3))
        with pytest.raises(ValueError, match="spacing is undefined"):
            data.fusi.reindex_voxels()


class TestReindexVoxelsLike:
    """Tests for fusi.reindex_voxels_like."""

    def _cropped_strided_reference(self) -> xr.DataArray:
        """A voxel-to-world array cropped and strided from a larger one."""
        base = xr.DataArray(
            np.arange(4 * 20 * 20, dtype=np.float64).reshape(4, 20, 20),
            dims=("k", "j", "i"),
            coords={
                "k": np.arange(4),
                "j": np.arange(20),
                "i": np.arange(20),
            },
        )
        base = attach_voxel_to_world_index(
            base,
            np.diag([1.0, 1.0, 1.0, 1.0]),
            world_coord_attrs={
                "z": {"units": "mm"},
                "y": {"units": "mm"},
                "x": {"units": "mm"},
            },
        )
        return base.isel(k=slice(1, 3), j=slice(2, 10, 2), i=slice(1, 15, 3))

    def test_relabels_onto_reference_voxel_coords(self):
        """Voxel coordinates and affine become reference's after reindexing."""
        reference = self._cropped_strided_reference()
        data = reference.fusi.reindex_voxels()

        result = data.fusi.reindex_voxels_like(reference)

        for dim in ("k", "j", "i"):
            np.testing.assert_array_equal(
                result.coords[dim].values, reference.coords[dim].values
            )
        np.testing.assert_allclose(
            get_voxel_to_world_affine(result), get_voxel_to_world_affine(reference)
        )

    def test_preserves_data_values_and_world_coordinates(self):
        """Array content and world coordinates are unchanged by reindexing."""
        reference = self._cropped_strided_reference()
        data = reference.fusi.reindex_voxels()

        result = data.fusi.reindex_voxels_like(reference)

        np.testing.assert_array_equal(result.values, data.values)
        for name in ("z", "y", "x"):
            np.testing.assert_allclose(
                result.coords[name].values, data.coords[name].values
            )

    def test_preserves_reference_coord_attrs(self):
        """Reindexed world coordinates carry reference's attrs (e.g. units)."""
        reference = self._cropped_strided_reference()
        data = reference.fusi.reindex_voxels()

        result = data.fusi.reindex_voxels_like(reference)

        assert result.coords["z"].attrs["units"] == "mm"

    def test_raises_on_shape_mismatch(self):
        """Different voxel grid shapes cannot be reindexed onto each other."""
        reference = self._cropped_strided_reference()
        data = reference.isel(i=slice(0, -1)).fusi.reindex_voxels()

        with pytest.raises(ValueError, match="same voxel grid shape"):
            data.fusi.reindex_voxels_like(reference)

    def test_raises_when_not_physically_aligned(self):
        """Data occupying a different world grid than reference raises."""
        reference = self._cropped_strided_reference()
        misaligned = reference.fusi.reindex_voxels().fusi.affine.apply(
            np.array(
                [
                    [1.0, 0.0, 0.0, 100.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

        with pytest.raises(ValueError, match="not aligned in world space"):
            misaligned.fusi.reindex_voxels_like(reference)

    def test_raises_without_voxel_to_world_geometry(self):
        """Either side lacking voxel-to-world geometry raises ValueError."""
        reference = self._cropped_strided_reference()
        data = reference.fusi.reindex_voxels()
        plain = xr.DataArray(np.zeros((2, 4, 5)), dims=["k", "j", "i"])

        with pytest.raises(ValueError, match="data must have a voxel-to-world index"):
            plain.fusi.reindex_voxels_like(reference)
        with pytest.raises(
            ValueError, match="reference must have a voxel-to-world index"
        ):
            data.fusi.reindex_voxels_like(plain)

    def test_raises_on_voxel_dim_mismatch(self):
        """Different voxel dimensions (e.g. 2D vs. 3D) cannot be reindexed."""
        reference = self._cropped_strided_reference()
        base_2d = xr.DataArray(
            np.zeros((4, 5)),
            dims=["j", "i"],
            coords={"j": np.arange(4), "i": np.arange(5)},
        )
        data_2d = attach_voxel_to_world_index(base_2d, np.eye(3))

        with pytest.raises(ValueError, match="same voxel dimensions"):
            data_2d.fusi.reindex_voxels_like(reference)


class TestAffineSetVoxelToWorldMethod:
    """Tests for fusi.affine.set_voxel_to_world."""

    def _make_data(self) -> xr.DataArray:
        return xr.DataArray(
            np.zeros((2, 3, 4)),
            dims=["k", "j", "i"],
            coords={
                "k": [0, 1],
                "j": [0, 1, 2],
                "i": [0, 1, 2, 3],
            },
        )

    def test_replaces_geometry_with_new_affine(self):
        """The resulting voxel_to_world affine matches the one supplied."""
        data = self._make_data()
        affine = np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = data.fusi.affine.set_voxel_to_world(affine)
        np.testing.assert_allclose(get_voxel_to_world_affine(result), affine)
        # World coordinates are broadcast over all voxel dims; spot-check one
        # voxel per axis against the affine applied by hand.
        assert result.coords["z"].isel(k=1, j=0, i=0).item() == pytest.approx(12.0)
        assert result.coords["y"].isel(k=0, j=1, i=0).item() == pytest.approx(23.0)
        assert result.coords["x"].isel(k=0, j=0, i=1).item() == pytest.approx(34.0)

    def test_inplace_updates_and_returns_same_object(self):
        """inplace=True mutates and returns the original DataArray."""
        data = self._make_data()
        affine = np.diag([2.0, 3.0, 4.0, 1.0])

        returned = data.fusi.affine.set_voxel_to_world(affine, inplace=True)

        assert returned is data
        np.testing.assert_allclose(get_voxel_to_world_affine(data), affine)
        assert data.coords["z"].isel(k=1, j=0, i=0).item() == pytest.approx(2.0)


class TestAffineToMethod:
    """Tests for fusi.affine.to."""

    @pytest.fixture
    def rng(self):
        """Seeded random number generator."""
        return np.random.default_rng(42)

    def _make_scan(self, affine: np.ndarray, via: str = "world_to_lab") -> xr.DataArray:
        return xr.DataArray(
            np.zeros((4, 4, 4)),
            attrs={"affines": {via: affine}},
        )

    def test_identity_when_same_affine(self):
        """Returns identity when both scans share the same affine."""
        affine = np.eye(4)
        a = self._make_scan(affine)
        b = self._make_scan(affine)
        result = a.fusi.affine.to(b, via="world_to_lab")
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
        result = a.fusi.affine.to(b, via="world_to_lab")
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

        a_to_b = a.fusi.affine.to(b, via="world_to_lab")
        b_to_a = b.fusi.affine.to(a, via="world_to_lab")
        np.testing.assert_allclose(a_to_b, np.linalg.inv(b_to_a), atol=1e-12)

    def test_custom_via_key(self):
        """Works with a via key other than world_to_lab."""
        affine = np.eye(4)
        a = self._make_scan(affine, via="world_to_mri")
        b = self._make_scan(affine, via="world_to_mri")
        result = a.fusi.affine.to(b, via="world_to_mri")
        np.testing.assert_allclose(result, np.eye(4), atol=1e-12)

    def test_missing_affines_on_self_raises(self):
        """Raises ValueError when self has no affines in attrs."""
        a = xr.DataArray(np.zeros((2, 2)))
        b = self._make_scan(np.eye(4))
        with pytest.raises(ValueError, match="self does not have"):
            a.fusi.affine.to(b, via="world_to_lab")

    def test_missing_affines_on_other_raises(self):
        """Raises ValueError when other has no affines in attrs."""
        a = self._make_scan(np.eye(4))
        b = xr.DataArray(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="other does not have"):
            a.fusi.affine.to(b, via="world_to_lab")

    def test_missing_via_key_raises(self):
        """Raises KeyError when via key is absent from the affines dict."""
        a = self._make_scan(np.eye(4), via="world_to_lab")
        b = self._make_scan(np.eye(4), via="world_to_lab")
        with pytest.raises(KeyError):
            a.fusi.affine.to(b, via="nonexistent_key")

    def test_output_shape(self, rng):
        """Output is always a (4, 4) array."""
        affine = np.eye(4)
        a = self._make_scan(affine)
        b = self._make_scan(affine)
        result = a.fusi.affine.to(b, via="world_to_lab")
        assert result.shape == (4, 4)


class TestAffineApplyMethod:
    """Tests for fusi.affine.apply."""

    def _make_scan(
        self,
        shape: tuple[int, ...] = (3, 4, 5),
        dims: tuple[str, ...] = ("k", "j", "i"),
        spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
        origin: tuple[float, ...] = (0.0, 0.0, 0.0),
        affines: dict | None = None,
    ) -> xr.DataArray:
        coords = {dim: np.arange(n) for dim, n in zip(dims, shape)}
        affine = np.eye(len(dims) + 1)
        affine[:-1, :-1] = np.diag(spacing[: len(dims)])
        affine[:-1, -1] = origin[: len(dims)]
        base = xr.DataArray(
            np.zeros(shape),
            dims=list(dims),
            coords=coords,
            attrs={"affines": affines} if affines is not None else {},
        )
        return attach_voxel_to_world_index(base, affine)

    def test_identity_leaves_coords_unchanged(self):
        """Applying the identity affine leaves coords unchanged."""
        da = self._make_scan(origin=(1.0, 2.0, 3.0))
        result = da.fusi.affine.apply(np.eye(4))
        for dim in ("z", "y", "x"):
            np.testing.assert_allclose(result.coords[dim].values, da.coords[dim].values)

    def test_pure_translation_shifts_coords(self):
        """A pure translation shifts all coordinate arrays by the given amount."""
        da = self._make_scan()
        shift = np.eye(4)
        shift[:3, 3] = [10.0, 5.0, -3.0]
        result = da.fusi.affine.apply(shift)
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
        result = da.fusi.affine.apply(scale)
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
        """A sign flip on any single axis negates only that axis.

        Regression: a y- or x-axis flip must negate y or x (not z). A diagonal
        affine is axis-aligned regardless of which axis carries the sign, so the
        axis-mixing test must read the off-diagonal block, not the decomposed
        zoom (decompose_affine relocates its one allowed sign flip onto axis 0).
        """
        base = self._make_scan(spacing=(1.0, 1.0, 1.0), origin=(1.0, 2.0, 3.0))
        for axis, flipped in enumerate(("z", "y", "x")):
            flip = np.eye(4)
            flip[axis, axis] = -1.0
            result = base.fusi.affine.apply(flip)
            for dim in ("z", "y", "x"):
                expected = (
                    -base.coords[dim].values
                    if dim == flipped
                    else base.coords[dim].values
                )
                np.testing.assert_allclose(result.coords[dim].values, expected)

    def test_multi_axis_flip_negates_flipped_coords(self):
        """A multi-axis sign flip is diagonal: negates each flipped axis."""
        da = self._make_scan(spacing=(1.0, 1.0, 1.0), origin=(1.0, 2.0, 3.0))
        flip = np.diag([-1.0, -1.0, 1.0, 1.0])
        result = da.fusi.affine.apply(flip)
        np.testing.assert_allclose(result.coords["z"].values, -da.coords["z"].values)
        np.testing.assert_allclose(result.coords["y"].values, -da.coords["y"].values)
        np.testing.assert_allclose(result.coords["x"].values, da.coords["x"].values)

    def test_scaling_updates_voxdim_attrs(self):
        """A scaling affine rescales each coord's `voxdim` by the absolute zoom.

        Regression: `voxdim` used to be copied verbatim, leaving a stale voxel
        size after any zooming affine (and hence a wrong pixdim/qform when the
        scan is saved to NIfTI).
        """
        da = self._make_scan()
        scale = np.diag([2.0, 3.0, -0.5, 1.0])
        result = da.fusi.affine.apply(scale)
        for dim, expected in zip(("z", "y", "x"), (2.0, 3.0, 0.5)):
            assert result.coords[dim].attrs["voxdim"] == pytest.approx(expected)

    def test_wrong_shape_raises_value_error(self):
        """Affines with shape other than (4, 4) raise ValueError."""
        da = self._make_scan()
        with pytest.raises(ValueError, match="shape"):
            da.fusi.affine.apply(np.eye(3))

    def test_raises_without_voxel_to_world_geometry(self):
        """A plain DataArray without voxel-to-world geometry raises ValueError."""
        da = xr.DataArray(np.zeros((2, 3)), dims=["j", "i"])
        with pytest.raises(ValueError, match="voxel-to-world index"):
            da.fusi.affine.apply(np.eye(3))

    def test_string_key_looks_up_stored_affine(self):
        """A string `affine` is resolved from `attrs["affines"]` before applying."""
        shift = np.eye(4)
        shift[:3, 3] = [10.0, 5.0, -3.0]
        da = self._make_scan(affines={"world_to_lab": shift})
        result = da.fusi.affine.apply("world_to_lab")
        np.testing.assert_allclose(
            result.coords["z"].values, da.coords["z"].values + 10.0
        )
        np.testing.assert_allclose(
            result.coords["y"].values, da.coords["y"].values + 5.0
        )
        np.testing.assert_allclose(
            result.coords["x"].values, da.coords["x"].values - 3.0
        )

    def test_string_key_applied_dropped_from_result(self):
        """Applying by key drops that key: composing it with itself is always
        identity, so the entry carries no information after applying."""
        shift = np.eye(4)
        shift[:3, 3] = [10.0, 5.0, -3.0]
        da = self._make_scan(
            affines={"world_to_lab": shift, "world_to_atlas": np.eye(4)}
        )
        result = da.fusi.affine.apply("world_to_lab")
        assert "world_to_lab" not in result.attrs["affines"]
        assert "world_to_atlas" in result.attrs["affines"]

    def test_string_key_missing_affines_attr_raises_value_error(self):
        """A string `affine` raises ValueError when `da` has no `affines` attr."""
        da = self._make_scan()
        with pytest.raises(ValueError, match="does not have an 'affines' entry"):
            da.fusi.affine.apply("world_to_lab")

    def test_string_key_not_found_raises_key_error(self):
        """A string `affine` absent from `attrs["affines"]` raises KeyError."""
        da = self._make_scan(affines={"world_to_lab": np.eye(4)})
        with pytest.raises(KeyError, match="world_to_mri"):
            da.fusi.affine.apply("world_to_mri")

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
        da = self._make_scan(affines={"world_to_lab": stored})
        shift = np.eye(4)
        shift[:3, 3] = [1.0, 2.0, 3.0]
        result = da.fusi.affine.apply(shift)
        expected = stored @ np.linalg.inv(shift)
        np.testing.assert_allclose(
            result.attrs["affines"]["world_to_lab"], expected, atol=1e-12
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
        da = self._make_scan(affines={"world_to_lab": stored})
        scale = np.diag([2.0, 1.0, 1.0, 1.0])
        result = da.fusi.affine.apply(scale)
        inv_scale = np.linalg.inv(scale)
        expected = stored @ inv_scale
        np.testing.assert_allclose(
            result.attrs["affines"]["world_to_lab"], expected, atol=1e-12
        )

    def test_partial_dims_only_updates_present_dims(self):
        """Only dimensions present in da.dims are updated."""
        da = self._make_scan(shape=(3, 4), dims=("j", "i"))
        shift = np.eye(3)
        shift[:2, 2] = [10.0, 5.0]
        result = da.fusi.affine.apply(shift)
        np.testing.assert_allclose(
            result.coords["y"].values, da.coords["y"].values + 10.0
        )
        np.testing.assert_allclose(
            result.coords["x"].values, da.coords["x"].values + 5.0
        )
        assert "z" not in result.coords

    def test_voxel_to_world_accepts_matching_2d_affine_shape(self):
        """Voxel-to-world 2D scans accept 3x3 world-space transforms."""
        base = xr.DataArray(
            np.zeros((3, 4)),
            dims=["j", "i"],
            coords={"j": [0, 1, 2], "i": [0, 1, 2, 3]},
            attrs={"affines": {"world_to_lab": np.eye(3)}},
        )
        da = attach_voxel_to_world_index(
            base,
            np.array(
                [
                    [0.2, 0.05, 10.0],
                    [0.08, 0.18, 20.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )
        shift = np.array(
            [
                [1.0, 0.0, 3.0],
                [0.0, 1.0, -4.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result = da.fusi.affine.apply(shift)

        np.testing.assert_allclose(
            get_voxel_to_world_affine(result), shift @ get_voxel_to_world_affine(da)
        )
        np.testing.assert_allclose(
            result.attrs["affines"]["world_to_lab"], np.linalg.inv(shift)
        )
        np.testing.assert_allclose(
            result.coords["y"].values, da.coords["y"].values + 3.0
        )
        np.testing.assert_allclose(
            result.coords["x"].values, da.coords["x"].values - 4.0
        )

    def test_unexpected_affine_shape_is_passed_through(self):
        """Unexpected stored affine shapes are kept unchanged."""
        weird = np.array([1.0, 2.0, 3.0])
        da = self._make_scan(affines={"weird": weird})

        result = da.fusi.affine.apply(np.eye(4))

        np.testing.assert_allclose(result.attrs["affines"]["weird"], weird)

    def test_inplace_updates_and_returns_same_object(self):
        """inplace=True mutates and returns the original DataArray."""
        da = self._make_scan()
        original_z = da.coords["z"].values.copy()
        shift = np.eye(4)
        shift[0, 3] = 2.0

        returned = da.fusi.affine.apply(shift, inplace=True)

        assert returned is da
        np.testing.assert_allclose(da.coords["z"].values, original_z + 2.0)

    def test_rotation_composes_fully_into_voxel_to_world(self):
        """A rotation composes fully into `voxel_to_world` (no residual)."""
        da = self._make_scan(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
        affine = np.array(
            [
                [0.0, -3.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 4.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = da.fusi.affine.apply(affine)
        np.testing.assert_allclose(get_voxel_to_world_affine(result), affine)

    def test_shear_composes_fully_into_voxel_to_world(self):
        """A shear (identity rotation, nonzero off-diagonal) composes fully too."""
        da = self._make_scan(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
        shear = np.eye(4)
        shear[0, 1] = 0.5
        result = da.fusi.affine.apply(shear)
        np.testing.assert_allclose(get_voxel_to_world_affine(result), shear)

    def test_axis_permutation_does_not_introduce_spurious_coord_flip(self):
        """A mixing affine composes fully, including any reflection."""
        da = self._make_scan(
            shape=(3, 4, 5),
            spacing=(1.0, 1.0, 1.0),
            origin=(10.0, 20.0, 30.0),
        )
        affine = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = da.fusi.affine.apply(affine)

        np.testing.assert_allclose(
            get_voxel_to_world_affine(result), affine @ get_voxel_to_world_affine(da)
        )

    def test_mixing_affine_reexpresses_existing_stored_affine(self):
        """Existing stored affines stay valid after an axis-mixing affine."""
        stored = np.eye(4)
        stored[:3, 3] = [5.0, 6.0, 7.0]
        da = self._make_scan(spacing=(1.0, 1.0, 1.0), affines={"world_to_lab": stored})
        affine = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = da.fusi.affine.apply(affine)
        np.testing.assert_allclose(
            result.attrs["affines"]["world_to_lab"], stored @ np.linalg.inv(affine)
        )
