"""Unit tests for confusius.io.nifti module."""

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
import xarray as xr

from confusius.io.nifti import load_nifti, save_nifti


@pytest.fixture
def nifti_2d_path(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Create a 2D NIfTI file for testing."""
    rng = np.random.default_rng(0)
    data = rng.random((8, 6)).astype(np.float32)
    nifti_path = tmp_path / "test_2d.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    return nifti_path, data


@pytest.fixture
def nifti_3d_path(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Create a 3D NIfTI file for testing."""
    rng = np.random.default_rng(0)
    data = rng.random((10, 8, 6)).astype(np.float32)
    nifti_path = tmp_path / "test_3d.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    return nifti_path, data


@pytest.fixture
def nifti_4d_path(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Create a 4D NIfTI file for testing."""
    rng = np.random.default_rng(0)
    data = rng.random((12, 10, 8, 6)).astype(np.float64)
    nifti_path = tmp_path / "test_4d.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    return nifti_path, data


@pytest.fixture
def nifti_with_sidecar(tmp_path: Path) -> Path:
    """Create a 3D NIfTI file with JSON sidecar."""
    rng = np.random.default_rng(0)
    data = rng.random((8, 6, 4)).astype(np.float32)
    nifti_path = tmp_path / "test_sidecar.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

    sidecar = {
        "TaskName": "test",
        "RepetitionTime": 1.0,
        "custom_meta": "test_value",
        "acquisition": "test_acq",
    }
    sidecar_path = tmp_path / "test_sidecar.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)

    return nifti_path


class TestLoadNifti:
    """Tests for load_nifti function."""

    def test_load_2d_nifti(self, nifti_2d_path: tuple[Path, np.ndarray]) -> None:
        """Loading 2D NIfTI creates DataArray with spatial dims only."""
        nifti_path, expected_data = nifti_2d_path
        da = load_nifti(nifti_path)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("y", "x")
        assert da.shape == (6, 8)
        assert da.dtype == np.float32
        np.testing.assert_array_equal(da.values, expected_data.T)

    def test_load_3d_nifti(self, nifti_3d_path: tuple[Path, np.ndarray]) -> None:
        """Loading 3D NIfTI creates DataArray with spatial dims only."""
        nifti_path, expected_data = nifti_3d_path
        da = load_nifti(nifti_path)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("z", "y", "x")
        assert da.shape == (6, 8, 10)
        assert da.dtype == np.float32
        np.testing.assert_array_equal(da.values, expected_data.transpose(2, 1, 0))

    def test_load_4d_nifti(self, nifti_4d_path: tuple[Path, np.ndarray]) -> None:
        """Loading 4D NIfTI creates a time coordinate from the NIfTI header TR."""
        nifti_path, expected_data = nifti_4d_path
        da = load_nifti(nifti_path)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("time", "z", "y", "x")
        assert da.shape == (6, 8, 10, 12)
        assert da.dtype == np.float64
        np.testing.assert_array_equal(
            da.coords["time"].values, np.arange(6, dtype=float)
        )
        np.testing.assert_array_equal(da.values, expected_data.transpose(3, 2, 1, 0))

    def test_load_5d_nifti_uses_dim4_for_extra_dimension(self, tmp_path: Path) -> None:
        """Loading 5D NIfTI preserves an extra dimension as `dim4`."""
        data = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float32).reshape(2, 3, 4, 5, 6)
        nifti_path = tmp_path / "test_5d.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

        da = load_nifti(nifti_path)

        assert da.dims == ("dim4", "time", "z", "y", "x")
        assert da.shape == (6, 5, 4, 3, 2)
        np.testing.assert_array_equal(da.values, data.transpose(4, 3, 2, 1, 0))

    def test_load_nifti_units_from_header(self, tmp_path: Path) -> None:
        """Units on coordinate attrs come from the NIfTI header, not hardcoded."""
        data = np.random.default_rng(0).random((4, 3, 2)).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        nifti_path = tmp_path / "with_units.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        assert da.coords["z"].attrs["units"] == "mm"
        assert da.coords["y"].attrs["units"] == "mm"
        assert da.coords["x"].attrs["units"] == "mm"

    def test_load_nifti_unknown_units_omits_units_attr(self, tmp_path: Path) -> None:
        """Coordinates have no 'units' attr when the NIfTI header declares unknown."""
        data = np.random.default_rng(0).random((4, 3, 2)).astype(np.float32)
        # Default nibabel image has xyzt_units=0 (unknown).
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_xyzt_units(xyz="unknown", t="unknown")
        nifti_path = tmp_path / "unknown_units.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        assert "units" not in da.coords["z"].attrs
        assert "units" not in da.coords["y"].attrs

    def test_load_nifti_creates_slice_time_coordinate_from_sidecar(
        self, tmp_path: Path
    ) -> None:
        """Slice timing sidecar metadata becomes a `slice_time` coordinate on load."""
        data = np.random.default_rng(0).random((2, 3, 4, 5)).astype(np.float32)
        path = tmp_path / "slice_time_sidecar.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(path)

        with open(tmp_path / "slice_time_sidecar.json", "w") as f:
            json.dump(
                {
                    "RepetitionTime": 1.0,
                    "SliceTiming": [0.0, 0.1, 0.2, 0.3],
                    "SliceEncodingDirection": "k",
                },
                f,
            )

        loaded = load_nifti(path)

        assert loaded.coords["slice_time"].dims == ("time", "z")
        np.testing.assert_allclose(
            loaded.coords["slice_time"].values,
            loaded.coords["time"].values[:, np.newaxis]
            + np.array([0.0, 0.1, 0.2, 0.3]),
        )

    def test_load_nifti_reconstructs_scalar_time_and_slice_time_from_sidecar(
        self, tmp_path: Path
    ) -> None:
        """3D payloads with timing sidecar metadata reconstruct scalar temporal coords."""
        data = np.random.default_rng(0).random((5, 4, 3)).astype(np.float32)
        path = tmp_path / "scalar_time_sidecar.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(path)

        with open(tmp_path / "scalar_time_sidecar.json", "w") as f:
            json.dump(
                {
                    "VolumeTiming": [2.05],
                    "FrameAcquisitionDuration": 0.4,
                    "SliceTiming": [0.0, 0.1, 0.2],
                    "SliceEncodingDirection": "k",
                },
                f,
            )

        loaded = load_nifti(path)

        assert loaded.coords["time"].dims == ()
        assert loaded.coords["time"].item() == pytest.approx(2.05)
        assert loaded.coords["time"].attrs["volume_acquisition_reference"] == "start"
        assert loaded.coords["time"].attrs["volume_acquisition_duration"] == pytest.approx(
            0.4
        )
        assert "volume_timing" not in loaded.attrs
        assert "volume_acquisition_duration" not in loaded.attrs

        assert loaded.coords["slice_time"].dims == ("z",)
        np.testing.assert_allclose(
            loaded.coords["slice_time"].values,
            np.array([2.05, 2.15, 2.25]),
        )

    def test_load_nifti_scalar_time_sidecar_converts_to_header_time_units(
        self, tmp_path: Path
    ) -> None:
        """Scalar sidecar timings are converted from seconds to header time units."""
        data = np.random.default_rng(0).random((5, 4, 3)).astype(np.float32)
        path = tmp_path / "scalar_time_ms_units.nii.gz"
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_xyzt_units(xyz="mm", t="msec")
        img.to_filename(path)

        with open(tmp_path / "scalar_time_ms_units.json", "w") as f:
            json.dump(
                {
                    "VolumeTiming": [2.05],
                    "FrameAcquisitionDuration": 0.4,
                    "SliceTiming": [0.0, 0.1, 0.2],
                    "SliceEncodingDirection": "k",
                },
                f,
            )

        loaded = load_nifti(path)

        assert loaded.coords["time"].attrs["units"] == "ms"
        assert loaded.coords["time"].item() == pytest.approx(2050.0)
        assert loaded.coords["time"].attrs["volume_acquisition_duration"] == pytest.approx(
            400.0
        )
        np.testing.assert_allclose(
            loaded.coords["slice_time"].values,
            np.array([2050.0, 2150.0, 2250.0]),
        )

    def test_load_nifti_scalar_time_invalid_slice_direction_preserves_fields(
        self, tmp_path: Path
    ) -> None:
        """Invalid scalar `SliceEncodingDirection` leaves slice fields untouched."""
        data = np.random.default_rng(0).random((5, 4, 3)).astype(np.float32)
        path = tmp_path / "scalar_time_invalid_slice_direction.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(path)

        with open(tmp_path / "scalar_time_invalid_slice_direction.json", "w") as f:
            json.dump(
                {
                    "VolumeTiming": [2.05],
                    "SliceTiming": [0.0, 0.1, 0.2],
                    "SliceEncodingDirection": "invalid",
                },
                f,
            )

        with pytest.warns(UserWarning, match="SliceEncodingDirection"):
            loaded = load_nifti(path)

        assert loaded.coords["time"].item() == pytest.approx(2.05)
        assert "slice_time" not in loaded.coords
        assert "slice_timing" in loaded.attrs
        assert "slice_encoding_direction" in loaded.attrs

    def test_load_nifti_derives_volume_duration_from_repetition_time_and_delay_time(
        self, tmp_path: Path
    ) -> None:
        """RepetitionTime and DelayTime imply a volume acquisition duration on load."""
        data = np.random.default_rng(0).random((2, 3, 4, 5)).astype(np.float32)
        path = tmp_path / "delay_time_sidecar.nii.gz"
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms((1.0, 1.0, 1.0, 2.0))
        img.to_filename(path)

        with open(tmp_path / "delay_time_sidecar.json", "w") as f:
            json.dump(
                {
                    "RepetitionTime": 2.0,
                    "DelayAfterTrigger": 0.5,
                    "DelayTime": 0.25,
                },
                f,
            )

        loaded = load_nifti(path)

        np.testing.assert_allclose(
            loaded.coords["time"].values, [0.5, 2.5, 4.5, 6.5, 8.5]
        )
        assert loaded.coords["time"].attrs["volume_acquisition_reference"] == "start"
        assert loaded.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(1.75)

    def test_load_nifti_validation_runtime_error_warns(self, tmp_path: Path) -> None:
        """Unexpected sidecar validation failures degrade to a warning when loading."""
        data = np.random.default_rng(0).random((4, 3, 2)).astype(np.float32)
        path = tmp_path / "bad_sidecar_validation.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(path)

        with open(tmp_path / "bad_sidecar_validation.json", "w") as f:
            json.dump({"RepetitionTime": 1.0}, f)

        with patch(
            "confusius.io.nifti.validate_metadata", side_effect=RuntimeError("boom")
        ):
            with pytest.warns(UserWarning, match="validation warning: boom"):
                load_nifti(path)

    def test_load_nifti_both_affines_stores_qform_attrs(self, tmp_path: Path) -> None:
        """Loading a NIfTI with both valid affines stores qform orientation attr."""
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        qform = np.diag([1.0, 1.0, 1.0, 1.0])
        data = np.random.default_rng(0).random((4, 3, 2)).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        img.header.set_qform(qform, code=1)
        nifti_path = tmp_path / "both_affines.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        # Primary (z, y, x) coords come from sform.
        assert "z" in da.coords
        assert "y" in da.coords
        assert "x" in da.coords
        # Qform non-dimension coords are no longer created; only the ConfUSIus-
        # convention transform is stored in da.attrs["affines"].
        assert "z_qform" not in da.coords
        assert "physical_to_qform" in da.attrs["affines"]
        assert da.attrs["affines"]["physical_to_qform"].shape == (4, 4)

    def test_load_nifti_qform_only_uses_qform(self, tmp_path: Path) -> None:
        """Loading a NIfTI with only qform valid uses qform for primary coords."""
        qform = np.diag([2.0, 3.0, 4.0, 1.0])
        data = np.random.default_rng(0).random((4, 3, 2)).astype(np.float32)
        img = nib.Nifti1Image(data, qform)
        img.header.set_sform(np.eye(4), code=0)
        img.header.set_qform(qform, code=1)
        nifti_path = tmp_path / "qform_only.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        assert "z" in da.coords
        assert "z_qform" not in da.coords
        # Primary z coord (NIfTI col 2, size 2) should reflect qform spacing of 4.0.
        np.testing.assert_allclose(da.coords["z"].values, [0.0, 4.0])
        # Affines dict uses the qform key, not sform.
        assert "physical_to_qform" in da.attrs["affines"]
        assert "physical_to_sform" not in da.attrs["affines"]

    def test_load_nifti_rotated_affine_probe_relative_coords(
        self, tmp_path: Path
    ) -> None:
        """Rotated affine produces probe-relative coords with correct spacing."""
        # 90° rotation around z maps x→y, y→-x with spacing [2, 3, 4].
        spacing = np.array([2.0, 3.0, 4.0])
        origin = np.array([10.0, 20.0, 30.0])
        affine = np.array(
            [
                [0.0, -spacing[1], 0.0, origin[0]],
                [spacing[0], 0.0, 0.0, origin[1]],
                [0.0, 0.0, spacing[2], origin[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = np.random.default_rng(0).random((5, 4, 3)).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        nifti_path = tmp_path / "rotated.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        # NIfTI data shape is (x=5, y=4, z=3); ConfUSIus order is (z=3, y=4, x=5).
        # Coords start at origin and are stepped by column norm (not diagonal, which
        # would give spacing 0 for the rotated x and y axes here).
        np.testing.assert_allclose(
            da.coords["x"].values, [10.0, 12.0, 14.0, 16.0, 18.0]
        )
        np.testing.assert_allclose(da.coords["y"].values, [20.0, 23.0, 26.0, 29.0])
        np.testing.assert_allclose(da.coords["z"].values, [30.0, 34.0, 38.0])
        # Physical transform (4×4 probe→world affine) is stored in da.attrs["affines"].
        assert da.attrs["affines"]["physical_to_sform"].shape == (4, 4)

    def test_load_nifti_sheared_affine_correct_spacing(self, tmp_path: Path) -> None:
        """Sheared affine uses decompose44 spacings, not (wrong) column norms."""
        # Pure shear along x for the y axis: physical y-spacing is still 1,
        # but the column norm of the y column is sqrt(1 + 0.5^2) ≈ 1.118.
        affine = np.array(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = np.random.default_rng(0).random((3, 4, 5)).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        nifti_path = tmp_path / "sheared.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)

        # NIfTI data shape is (x=3, y=4, z=5); ConfUSIus order is (z=5, y=4, x=3).
        # All voxel spacings should be 1 (true pixdim), not sqrt(1.25) ≈ 1.118.
        np.testing.assert_allclose(da.coords["x"].values, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(da.coords["y"].values, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(da.coords["z"].values, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_load_nifti_physical_transform_maps_probe_to_world(
        self, tmp_path: Path
    ) -> None:
        """physical_to_sform correctly maps physical-space coords to world-space coords."""
        # Affine with rotation (90° around z) + zoom [2, 3, 4] + translation [10, 20, 30].
        spacing = np.array([2.0, 3.0, 4.0])
        origin = np.array([10.0, 20.0, 30.0])
        affine = np.array(
            [
                [0.0, -spacing[1], 0.0, origin[0]],
                [spacing[0], 0.0, 0.0, origin[1]],
                [0.0, 0.0, spacing[2], origin[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = np.zeros((5, 4, 3), dtype=np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        nifti_path = tmp_path / "transform_check.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)
        A = da.attrs["affines"]["physical_to_sform"]

        # A_physical uses ConfUSIus (z, y, x) convention:
        # A @ [pz, py, px, 1] → [wz, wy, wx, 1].
        for i, j, k in [(0, 0, 0), (2, 1, 0), (4, 3, 2)]:
            world_xyz = (affine @ np.array([i, j, k, 1.0]))[:3]
            world_expected_zyx = world_xyz[[2, 1, 0]]
            px = float(da.coords["x"].values[i])
            py = float(da.coords["y"].values[j])
            pz = float(da.coords["z"].values[k])
            world_actual = (A @ np.array([pz, py, px, 1.0]))[:3]
            np.testing.assert_allclose(world_actual, world_expected_zyx, atol=1e-10)

    def test_load_nifti_no_valid_affine_warns(self, tmp_path: Path) -> None:
        """Loading a NIfTI with both codes 0 warns and uses pixdim for coordinates."""
        import struct

        data = np.random.default_rng(0).random((4, 3, 2)).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms([2.0, 3.0, 4.0])
        # Use uncompressed .nii so we can patch the binary header directly;
        # nibabel resets codes to non-zero values during to_filename().
        nifti_path = tmp_path / "no_affine.nii"
        img.to_filename(nifti_path)

        # Patch qform_code (byte 252) and sform_code (byte 254) to 0.
        # Both are int16 LE per the NIfTI-1 specification.
        with open(nifti_path, "r+b") as f:
            f.seek(252)
            f.write(struct.pack("<hh", 0, 0))

        with pytest.warns(UserWarning, match="sform_code and qform_code are 0"):
            da = load_nifti(nifti_path)

        # No affines stored — neither sform/qform code nor affines dict.
        assert "affines" not in da.attrs
        assert "sform_code" not in da.attrs
        assert "qform_code" not in da.attrs
        # Coordinates built from pixdim only: origin 0, step = voxel size.
        # NIfTI shape is (x=4, y=3, z=2); ConfUSIus order is (z=2, y=3, x=4).
        np.testing.assert_allclose(da.coords["x"].values, [0.0, 2.0, 4.0, 6.0])
        np.testing.assert_allclose(da.coords["y"].values, [0.0, 3.0, 6.0])
        np.testing.assert_allclose(da.coords["z"].values, [0.0, 4.0])

    def test_load_nifti_no_valid_affine_2d_preserves_units(
        self, tmp_path: Path
    ) -> None:
        """Pixdim-only fallback still applies units for present spatial dimensions."""
        import struct

        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms([2.0, 3.0])
        img.header.set_xyzt_units(xyz="mm", t="sec")
        nifti_path = tmp_path / "no_affine_2d.nii"
        img.to_filename(nifti_path)

        with open(nifti_path, "r+b") as f:
            f.seek(252)
            f.write(struct.pack("<hh", 0, 0))

        with pytest.warns(UserWarning, match="sform_code and qform_code are 0"):
            da = load_nifti(nifti_path)

        assert da.dims == ("y", "x")
        assert "z" not in da.coords
        assert da.coords["x"].attrs["units"] == "mm"
        assert da.coords["y"].attrs["units"] == "mm"
        np.testing.assert_allclose(da.coords["x"].values, [0.0, 2.0, 4.0, 6.0])
        np.testing.assert_allclose(da.coords["y"].values, [0.0, 3.0, 6.0])

    def test_load_nifti_lazy(self, tmp_path: Path) -> None:
        """Loading creates lazy Dask array."""
        import dask.array as dask_array

        data = np.random.default_rng(0).random((8, 6, 4)).astype(np.float32)
        nifti_path = tmp_path / "test_lazy.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

        result = load_nifti(nifti_path)

        assert isinstance(result.data, dask_array.Array)

    def test_load_nifti_rejects_non_nifti_image(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Loading rejects objects returned by nibabel that are not NIfTI images."""
        minc_path = tmp_path / "not_really_nifti.mnc"
        minc_path.write_bytes(b"placeholder")

        monkeypatch.setattr(nib, "load", lambda _: object())

        with pytest.raises(ValueError, match="Only NIfTI-1 and NIfTI-2 formats"):
            load_nifti(minc_path)

    def test_load_nifti_with_sidecar(self, nifti_with_sidecar: Path) -> None:
        """Loading with sidecar JSON merges metadata."""
        da = load_nifti(nifti_with_sidecar)

        assert da.attrs.get("custom_meta") == "test_value"
        assert da.attrs.get("acquisition") == "test_acq"

    def test_load_nifti_invalid_sidecar_warns(self, tmp_path: Path) -> None:
        """Loading warns when the sidecar violates fUSI-BIDS validation rules."""
        data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms([1.0, 1.0, 1.0, 1.0])
        nifti_path = tmp_path / "invalid_sidecar.nii.gz"
        img.to_filename(nifti_path)

        sidecar_path = tmp_path / "invalid_sidecar.json"
        with open(sidecar_path, "w") as f:
            json.dump(
                {"RepetitionTime": 1.0, "VolumeTiming": [0.0, 1.0, 2.0, 3.0, 4.0]},
                f,
            )

        with pytest.warns(UserWarning, match="fUSI-BIDS validation warning"):
            da = load_nifti(nifti_path)

        np.testing.assert_allclose(da.coords["time"].values, [0.0, 1.0, 2.0, 3.0, 4.0])


class TestSaveNifti:
    """Tests for `save_nifti` function."""

    def test_save_invalid_extension_raises(self, tmp_path, sample_3d_volume):
        """Saving rejects paths without a NIfTI extension."""
        da = sample_3d_volume.drop_vars("time")

        with pytest.raises(ValueError, match=r"\.nii or \.nii\.gz"):
            save_nifti(da, tmp_path / "not_nifti.json")

    def test_save_2d_dataarray(self, tmp_path) -> None:
        """Saving 2D DataArray inserts missing spatial axes in NIfTI order."""
        data = np.random.default_rng(0).random((6, 8)).astype(np.float32)
        da = xr.DataArray(data, dims=["y", "x"])

        output_path = tmp_path / "output_2d.nii.gz"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(da, output_path)

        loaded = nib.load(output_path)
        assert loaded.shape == (8, 6, 1)
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.dataobj), data.T[..., None]
        )

    def test_save_3d_dataarray(self, tmp_path):
        """Saving 3D DataArray creates valid NIfTI file."""
        data = np.random.default_rng(0).random((6, 8, 10)).astype(np.float32)
        da = xr.DataArray(data, dims=["z", "y", "x"])

        output_path = tmp_path / "output_3d.nii"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(da, output_path)

        assert output_path.exists()
        loaded = nib.nifti1.Nifti1Image.from_filename(output_path)
        assert loaded.shape == (10, 8, 6)  # NIfTI order: (x, y, z)
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.dataobj), data.transpose(2, 1, 0)
        )

    def test_save_4d_dataarray(self, tmp_path):
        """Saving 4D DataArray creates valid NIfTI file."""
        data = np.random.default_rng(0).random((4, 6, 8, 10)).astype(np.float32)
        da = xr.DataArray(data, dims=["time", "z", "y", "x"])

        output_path = tmp_path / "output_4d.nii.gz"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(da, output_path)

        assert output_path.exists()
        assert output_path.suffixes == [".nii", ".gz"]
        loaded = nib.nifti1.Nifti1Image.from_filename(output_path)
        assert loaded.shape == (10, 8, 6, 4)  # NIfTI order: (x, y, z, time)
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.dataobj), data.transpose(3, 2, 1, 0)
        )

    def test_save_5d_dataarray_preserves_extra_dim_order(self, tmp_path) -> None:
        """Saving with an extra non-standard dim keeps it after `time` in NIfTI order."""
        data = np.arange(2 * 4 * 3 * 5 * 6, dtype=np.float32).reshape(2, 4, 3, 5, 6)
        da = xr.DataArray(data, dims=["channel", "time", "z", "y", "x"])

        output_path = tmp_path / "output_5d.nii.gz"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(da, output_path)

        loaded = nib.load(output_path)
        assert loaded.shape == (6, 5, 3, 4, 2)
        np.testing.assert_array_equal(
            np.asarray(loaded.dataobj), data.transpose(4, 3, 2, 1, 0)
        )

        roundtripped = load_nifti(output_path)
        assert roundtripped.dims == ("dim4", "time", "z", "y", "x")
        np.testing.assert_array_equal(roundtripped.values, data)

    def test_save_non_uniform_coords_warns(self, tmp_path):
        """Saving a DataArray with non-uniform coordinate spacing emits a warning."""
        data = np.random.default_rng(0).random((4, 2, 3)).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={"z": [0.0, 1.0, 3.0, 6.0], "y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]},
        )

        output_path = tmp_path / "nonuniform.nii.gz"
        with pytest.warns(UserWarning, match="using the median step"):
            save_nifti(da, output_path)

        loaded = nib.load(output_path)
        assert loaded.header.get_zooms()[:3] == pytest.approx((1.0, 1.0, 2.0))

    def test_save_creates_sidecar(self, tmp_path, sample_3d_volume):
        """Saving always creates a JSON sidecar alongside the NIfTI file."""
        da = sample_3d_volume.drop_vars("time").copy()
        da.attrs["task_name"] = "test"
        da.attrs["custom"] = "value"

        output_path = tmp_path / "with_sidecar.nii.gz"
        save_nifti(da, output_path)

        sidecar_path = tmp_path / "with_sidecar.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar["TaskName"] == "test"
        assert sidecar["custom"] == "value"
        assert "RepetitionTime" not in sidecar

    def test_save_maps_processing_metadata_to_sidecar_fields(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """Processing metadata uses BIDS fields or ConfUSIus-prefixed equivalents."""
        da = sample_4d_volume.copy()
        # Use non-uniform time to force VolumeTiming (instead of RepetitionTime) in the
        # sidecar — BIDS disallows FrameAcquisitionDuration alongside RepetitionTime.
        da = da.assign_coords(
            time=xr.DataArray(
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.6],
                dims=["time"],
                attrs={
                    "units": "s",
                    "volume_acquisition_duration": 0.3,
                    "volume_acquisition_reference": "start",
                },
            )
        )
        da.attrs.update(
            {
                "long_name": "Power Doppler intensity",
                "cmap": "gray",
                "compound_sampling_frequency": 500.0,
                "clutter_filters": "Index-based SVD [50, +inf[",
                "clutter_filter_window_duration": 0.6,
                "clutter_filter_window_stride": 0.6,
                "power_doppler_integration_stride": 0.2,
                "axial_velocity_integration_stride": 0.25,
                "bmode_integration_stride": 0.3,
                "axial_velocity_lag": 2,
                "axial_velocity_absolute": True,
                "axial_velocity_spatial_kernel": 3,
                "axial_velocity_estimation_method": "angle_average",
            }
        )

        output_path = tmp_path / "processing_metadata.nii.gz"
        with pytest.warns(UserWarning, match="non-uniform sampling"):
            save_nifti(da, output_path)

        sidecar_path = tmp_path / "processing_metadata.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["ClutterFilters"] == "Index-based SVD [50, +inf["
        assert sidecar["ClutterFilterWindowDuration"] == pytest.approx(0.6)
        assert sidecar["ClutterFilterWindowStride"] == pytest.approx(0.6)
        assert sidecar["PowerDopplerIntegrationStride"] == pytest.approx(0.2)
        assert sidecar["FrameAcquisitionDuration"] == pytest.approx(0.3)
        assert sidecar["ConfUSIusAxialVelocityIntegrationStride"] == pytest.approx(0.25)
        assert sidecar["ConfUSIusBmodeIntegrationStride"] == pytest.approx(0.3)
        assert sidecar["ConfUSIusAxialVelocityLag"] == 2
        assert sidecar["ConfUSIusAxialVelocityAbsolute"] is True
        assert sidecar["ConfUSIusAxialVelocitySpatialKernel"] == 3
        assert sidecar["ConfUSIusAxialVelocityEstimationMethod"] == "angle_average"
        assert sidecar["ConfUSIusLongName"] == "Power Doppler intensity"
        assert sidecar["ConfUSIusCmap"] == "gray"
        assert "long_name" not in sidecar
        assert "clutter_filters" not in sidecar
        assert "clutter_filter_window_duration" not in sidecar
        assert "volume_acquisition_duration" not in sidecar
        assert "volume_acquisition_reference" not in sidecar

    def test_save_maps_probe_and_beamforming_metadata(
        self, tmp_path, sample_3d_volume
    ) -> None:
        """Canonical probe and beamforming metadata use standard fUSI-BIDS fields."""
        da = sample_3d_volume.drop_vars("time").copy()
        da.attrs["probe_number_of_elements"] = 128
        da.attrs["beamforming_sound_velocity"] = 1520.0

        output_path = tmp_path / "beamforming_metadata.nii.gz"
        save_nifti(da, output_path)

        sidecar_path = tmp_path / "beamforming_metadata.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["ProbeNumberOfElements"] == 128
        assert sidecar["BeamformingSoundVelocity"] == 1520.0
        assert "probe_number_of_elements" not in sidecar
        assert "beamforming_sound_velocity" not in sidecar

    def test_save_sidecar_serializes_extra_affines(self, tmp_path, sample_3d_volume):
        """Extra affines that are not qform/sform are written to the sidecar."""
        da = sample_3d_volume.drop_vars("time").copy()
        da.attrs["affines"] = {
            "physical_to_template": np.array(
                [
                    [1.0, 0.0, 0.0, 4.0],
                    [0.0, 1.0, 0.0, 5.0],
                    [0.0, 0.0, 1.0, 6.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        }

        output_path = tmp_path / "extra_affines.nii.gz"
        save_nifti(da, output_path)

        sidecar_path = tmp_path / "extra_affines.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["ConfUSIusAffines"]["physical_to_template"] == [
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def test_save_sidecar_omits_affines_written_to_header(
        self, tmp_path, sample_3d_volume
    ):
        """Affines already written to qform/sform are omitted from the sidecar."""
        template_affine = np.array(
            [
                [1.0, 0.0, 0.0, 4.0],
                [0.0, 1.0, 0.0, 5.0],
                [0.0, 0.0, 1.0, 6.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        scanner_affine = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [1.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        lab_affine = np.array(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, -3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        da = sample_3d_volume.drop_vars("time").copy()
        da.attrs["affines"] = {
            "physical_to_template": template_affine,
            "physical_to_scanner": scanner_affine,
            "physical_to_lab": lab_affine,
        }

        output_path = tmp_path / "selected_affines.nii.gz"
        save_nifti(
            da,
            output_path,
            sform="physical_to_template",
            qform="physical_to_scanner",
        )

        sidecar_path = tmp_path / "selected_affines.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["ConfUSIusAffines"] == {
            "physical_to_lab": [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, -3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }

    def test_save_sidecar_keeps_qform_affine_when_qform_not_written(self, tmp_path):
        """A stored qform affine stays in the sidecar when `qform_code` is 0."""
        qform_affine = np.array(
            [
                [1.0, 0.0, 0.0, 4.0],
                [0.0, 1.0, 0.0, 5.0],
                [0.0, 0.0, 1.0, 6.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        da = xr.DataArray(
            np.zeros((4, 3, 2), dtype=np.float32),
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4, dtype=float),
                "y": np.arange(3, dtype=float),
                "x": np.arange(2, dtype=float),
            },
            attrs={"affines": {"physical_to_qform": qform_affine}},
        )

        output_path = tmp_path / "qform_not_written.nii.gz"
        save_nifti(da, output_path, qform_code=0)

        sidecar_path = tmp_path / "qform_not_written.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["ConfUSIusAffines"]["physical_to_qform"] == [
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def test_save_invalid_time_units_raises(self, tmp_path, sample_4d_volume):
        """Saving rejects unsupported time units."""
        da = sample_4d_volume.copy()
        da.coords["time"].attrs["units"] = "fortnight"

        with pytest.raises(ValueError, match="Unknown time unit"):
            save_nifti(da, tmp_path / "invalid_time_units.nii.gz")

    def test_save_single_volume_time_coord_uses_volume_timing(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """A single time point is stored via `VolumeTiming`, not `RepetitionTime`."""
        da = sample_4d_volume.isel(time=slice(0, 1)).copy()
        da.coords["time"].attrs["volume_acquisition_duration"] = 0.25
        output_path = tmp_path / "single_volume.nii.gz"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_nifti(da, output_path)

        assert any(
            "Coordinate 'time' has no `volume_acquisition_reference` attribute"
            in str(w.message)
            for w in caught
        )

        sidecar_path = tmp_path / "single_volume.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["VolumeTiming"] == [10.0]
        assert sidecar["FrameAcquisitionDuration"] == 0.25
        assert "RepetitionTime" not in sidecar

        loaded = load_nifti(output_path)
        np.testing.assert_allclose(loaded.coords["time"].values, [10.0])

    def test_save_single_volume_without_duration_omits_frame_acquisition_duration(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """A single time point without metadata does not invent a frame duration."""
        da = sample_4d_volume.isel(time=slice(0, 1)).copy()

        output_path = tmp_path / "single_volume_no_duration.nii.gz"
        with pytest.warns(UserWarning, match="FrameAcquisitionDuration is REQUIRED"):
            save_nifti(da, output_path)

        sidecar_path = tmp_path / "single_volume_no_duration.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["VolumeTiming"] == [10.0]
        assert "FrameAcquisitionDuration" not in sidecar

    def test_save_scalar_time_coordinate_without_time_dim(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """Saving a 3D slice with scalar `time` coord writes single-volume timing."""
        da = sample_4d_volume.isel(time=0).copy()
        output_path = tmp_path / "scalar_time_coord.nii.gz"

        with pytest.warns(UserWarning, match="FrameAcquisitionDuration is REQUIRED"):
            save_nifti(da, output_path)

        sidecar_path = tmp_path / "scalar_time_coord.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["VolumeTiming"] == [10.0]
        assert "RepetitionTime" not in sidecar

    def test_save_boolean_data_writes_uint8(self, tmp_path, sample_3d_volume) -> None:
        """Boolean payloads are stored as uint8 because NIfTI does not support bool."""
        threshold = float(sample_3d_volume.mean().item())
        da = sample_3d_volume.drop_vars("time") > threshold
        output_path = tmp_path / "bool_payload.nii.gz"

        save_nifti(da, output_path)

        loaded = nib.load(output_path)
        assert loaded.get_data_dtype() == np.dtype(np.uint8)
        np.testing.assert_array_equal(
            np.asarray(loaded.dataobj), da.values.transpose(2, 1, 0).astype(np.uint8)
        )

    def test_save_nifti2_writes_nifti2_image(self, tmp_path, sample_3d_volume) -> None:
        """Saving with `nifti_version=2` produces a NIfTI-2 image."""
        da = sample_3d_volume.drop_vars("time")

        output_path = tmp_path / "output_nifti2.nii"
        save_nifti(da, output_path, nifti_version=2)

        loaded = nib.load(output_path)
        assert isinstance(loaded, nib.nifti2.Nifti2Image)

    def test_save_valid_derived_metadata_emits_no_validation_warning(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """Valid derived timing metadata does not trigger BIDS validation warnings."""
        da = sample_4d_volume.copy()
        da.coords["time"].attrs["volume_acquisition_duration"] = 0.25

        output_path = tmp_path / "invalid_metadata.nii.gz"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_nifti(da, output_path)

        assert not any(
            "validation warning when saving" in str(w.message) for w in caught
        )

        sidecar_path = tmp_path / "invalid_metadata.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["RepetitionTime"] == pytest.approx(0.5)
        assert "FrameAcquisitionDuration" not in sidecar

    def test_save_regular_center_reference_converts_delay_to_onset(self, tmp_path):
        """Regular timings with center reference are written as onset-based BIDS timings."""
        time_values = np.array([0.5, 1.0, 1.5, 2.0])
        original = xr.DataArray(
            np.random.default_rng(0).random((4, 2, 2, 2)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(
                    time_values,
                    dims=["time"],
                    attrs={
                        "units": "s",
                        "volume_acquisition_reference": "center",
                        "volume_acquisition_duration": 0.5,
                    },
                )
            },
        )

        nifti_path = tmp_path / "center_reference.nii.gz"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(original, nifti_path)

        with open(tmp_path / "center_reference.json") as f:
            sidecar = json.load(f)

        assert sidecar["RepetitionTime"] == pytest.approx(0.5)
        assert sidecar["DelayAfterTrigger"] == pytest.approx(0.25)

    def test_save_irregular_end_reference_converts_volume_timing_to_onset(
        self, tmp_path
    ):
        """Irregular timings with end reference are written as onset-based VolumeTiming."""
        time_values = np.array([1.0, 2.5, 3.8, 5.6])
        original = xr.DataArray(
            np.random.default_rng(0).random((4, 2, 2, 2)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(
                    time_values,
                    dims=["time"],
                    attrs={
                        "units": "s",
                        "volume_acquisition_reference": "end",
                        "volume_acquisition_duration": 0.4,
                    },
                ),
                "z": [0.0, 1.0],
                "y": [0.0, 1.0],
                "x": [0.0, 1.0],
            },
        )

        nifti_path = tmp_path / "end_reference_irregular.nii.gz"
        with pytest.warns(UserWarning, match="non-uniform sampling"):
            save_nifti(original, nifti_path)

        with open(tmp_path / "end_reference_irregular.json") as f:
            sidecar = json.load(f)

        assert sidecar["VolumeTiming"] == pytest.approx([0.6, 2.1, 3.4, 5.2])
        assert sidecar["FrameAcquisitionDuration"] == pytest.approx(0.4)

    def test_save_serializes_consistent_2d_slice_time_coordinate(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """A 2D absolute `slice_time` is exported when onset-relative timing is constant."""
        da = sample_4d_volume.copy()
        time_values = da.coords["time"].values
        da = da.assign_coords(
            slice_time=xr.DataArray(
                time_values[:, np.newaxis] + np.array([0.0, 0.1, 0.2, 0.3]),
                dims=("time", "z"),
                attrs={"units": "s"},
            )
        )

        output_path = tmp_path / "slice_time_2d.nii.gz"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_nifti(da, output_path)

        assert any(
            "Coordinate 'time' has no `volume_acquisition_reference` attribute"
            in str(w.message)
            for w in caught
        )

        with open(tmp_path / "slice_time_2d.json") as f:
            sidecar = json.load(f)

        assert sidecar["SliceTiming"] == pytest.approx([0.0, 0.1, 0.2, 0.3])
        assert sidecar["SliceEncodingDirection"] == "k"

    def test_save_skips_inconsistent_2d_slice_time_coordinate(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """A 2D `slice_time` is omitted when relative slice timing varies across volumes."""
        da = sample_4d_volume.copy()
        time_values = da.coords["time"].values
        varying_offsets = np.tile(np.array([0.0, 0.1, 0.2, 0.3]), (da.sizes["time"], 1))
        varying_offsets[1, -1] += 0.05
        da = da.assign_coords(
            slice_time=xr.DataArray(
                time_values[:, np.newaxis] + varying_offsets,
                dims=("time", "z"),
                attrs={"units": "s"},
            )
        )

        output_path = tmp_path / "slice_time_2d_inconsistent.nii.gz"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_nifti(da, output_path)

        assert any(
            "Coordinate 'time' has no `volume_acquisition_reference` attribute"
            in str(w.message)
            for w in caught
        )
        assert any(
            "cannot represent per-volume variation" in str(w.message) for w in caught
        )

        with open(tmp_path / "slice_time_2d_inconsistent.json") as f:
            sidecar = json.load(f)

        assert "SliceTiming" not in sidecar
        assert "SliceEncodingDirection" not in sidecar

    def test_save_serializes_1d_slice_time_with_scalar_time_coordinate(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """A 1D absolute `slice_time` is exported for scalar-time 3D snapshots."""
        da = sample_4d_volume.isel(time=0).copy()
        da.coords["time"].attrs["volume_acquisition_duration"] = 0.25
        da.coords["time"].attrs["volume_acquisition_reference"] = "start"
        da = da.assign_coords(
            slice_time=xr.DataArray(
                da.coords["time"].item() + np.array([0.0, 0.1, 0.2, 0.3]),
                dims=("z",),
                attrs={"units": "s"},
            )
        )

        output_path = tmp_path / "slice_time_1d_scalar_time.nii.gz"
        save_nifti(da, output_path)

        with open(tmp_path / "slice_time_1d_scalar_time.json") as f:
            sidecar = json.load(f)

        assert sidecar["SliceTiming"] == pytest.approx([0.0, 0.1, 0.2, 0.3])
        assert sidecar["SliceEncodingDirection"] == "k"

    def test_save_nifti_validation_runtime_error_warns(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """Unexpected sidecar validation failures degrade to a warning when saving."""
        output_path = tmp_path / "save_validation_runtime_error.nii.gz"

        with patch(
            "confusius.io.nifti.validate_metadata", side_effect=RuntimeError("boom")
        ):
            with pytest.warns(
                UserWarning, match="validation warning when saving: boom"
            ):
                save_nifti(sample_4d_volume, output_path)

    def test_named_qform_selects_requested_affine(self, tmp_path):
        """`qform=` selects the requested affine key from `attrs['affines']`."""
        data = np.zeros((4, 3, 2), dtype=np.float32)
        selected_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        different_affine = np.diag([9.0, 9.0, 9.0, 1.0])
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4) * 1.0,
                "y": np.arange(3) * 1.0,
                "x": np.arange(2) * 1.0,
            },
            attrs={
                "affines": {
                    "physical_to_qform": different_affine,
                    "physical_to_template": selected_affine,
                }
            },
        )
        output_path = tmp_path / "selected_qform.nii.gz"
        save_nifti(da, output_path, qform="physical_to_template")

        loaded = nib.load(output_path)
        q = loaded.header.get_qform()
        np.testing.assert_allclose(q[:3, :3], np.eye(3), atol=1e-6)

    def test_default_qform_key_from_attrs_used_when_no_kwarg(self, tmp_path):
        """When `qform` is omitted, `physical_to_qform` is used if present."""
        data = np.zeros((4, 3, 2), dtype=np.float32)
        physical_to_qform = np.eye(4)
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4) * 3.0,
                "y": np.arange(3) * 3.0,
                "x": np.arange(2) * 3.0,
            },
            attrs={"affines": {"physical_to_qform": physical_to_qform}},
        )
        output_path = tmp_path / "default_qform_key.nii.gz"
        save_nifti(da, output_path)

        loaded = nib.load(output_path)
        np.testing.assert_allclose(
            loaded.header.get_qform()[:3, :3], np.diag([3.0, 3.0, 3.0]), atol=1e-6
        )

    def test_named_sform_sets_sform_code(self, tmp_path):
        """Providing `sform=` writes a sform with code=1 by default."""
        data = np.zeros((4, 3, 2), dtype=np.float32)
        physical_to_sform = np.diag([2.0, 2.0, 2.0, 1.0])
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4) * 2.0,
                "y": np.arange(3) * 2.0,
                "x": np.arange(2) * 2.0,
            },
            attrs={"affines": {"physical_to_template": physical_to_sform}},
        )
        output_path = tmp_path / "named_sform.nii.gz"
        save_nifti(da, output_path, sform="physical_to_template")

        loaded = nib.load(output_path)
        assert loaded.header.get_sform(coded=True)[1] == 1

    def test_named_sform_code_kwarg_overrides_attrs(self, tmp_path):
        """sform_code= kwarg takes precedence over attrs['sform_code']."""
        data = np.zeros((4, 3, 2), dtype=np.float32)
        physical_to_sform = np.diag([1.0, 1.0, 1.0, 1.0])
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4) * 1.0,
                "y": np.arange(3) * 1.0,
                "x": np.arange(2) * 1.0,
            },
            attrs={
                "affines": {"physical_to_sform": physical_to_sform},
                "sform_code": 1,
            },
        )
        output_path = tmp_path / "sform_code_override.nii.gz"
        save_nifti(da, output_path, sform="physical_to_sform", sform_code=2)

        loaded = nib.load(output_path)
        assert loaded.header.get_sform(coded=True)[1] == 2

    def test_explicit_qform_code_kwarg_overrides_attrs(self, tmp_path):
        """qform_code= kwarg takes precedence over attrs['qform_code']."""
        data = np.zeros((4, 3, 2), dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4) * 1.0,
                "y": np.arange(3) * 1.0,
                "x": np.arange(2) * 1.0,
            },
            attrs={"qform_code": 1},
        )
        output_path = tmp_path / "qform_code_override.nii.gz"
        save_nifti(da, output_path, qform_code=2)

        loaded = nib.load(output_path)
        assert loaded.header.get_qform(coded=True)[1] == 2

    def test_form_codes_from_attrs_used_when_no_kwarg(self, tmp_path):
        """qform_code and sform_code from attrs are written when no kwarg is given."""
        data = np.zeros((4, 3, 2), dtype=np.float32)
        physical_to_sform = np.diag([1.0, 1.0, 1.0, 1.0])
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4) * 1.0,
                "y": np.arange(3) * 1.0,
                "x": np.arange(2) * 1.0,
            },
            attrs={
                "affines": {"physical_to_sform": physical_to_sform},
                "qform_code": 2,
                "sform_code": 2,
            },
        )
        output_path = tmp_path / "codes_from_attrs.nii.gz"
        save_nifti(da, output_path)

        loaded = nib.load(output_path)
        assert loaded.header.get_qform(coded=True)[1] == 2
        assert loaded.header.get_sform(coded=True)[1] == 2

    def test_invalid_qform_key_raises(self, tmp_path):
        """Selecting a missing qform key raises a clear error."""
        data = np.zeros((4, 3, 2), dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4) * 1.0,
                "y": np.arange(3) * 1.0,
                "x": np.arange(2) * 1.0,
            },
            attrs={"affines": {"physical_to_template": np.eye(4)}},
        )

        with pytest.raises(
            ValueError,
            match=r"qform='physical_to_scanner' not found in data_array.attrs\['affines'\]",
        ):
            save_nifti(
                da, tmp_path / "invalid_qform_key.nii.gz", qform="physical_to_scanner"
            )

    def test_no_sform_kwarg_writes_no_sform(self, tmp_path):
        """Without `sform=` and no attrs `physical_to_sform`, the saved file has sform_code=0."""
        data = np.zeros((4, 3, 2), dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4) * 1.0,
                "y": np.arange(3) * 1.0,
                "x": np.arange(2) * 1.0,
            },
        )
        output_path = tmp_path / "no_sform.nii.gz"
        save_nifti(da, output_path)

        loaded = nib.load(output_path)
        assert loaded.header.get_sform(coded=True)[1] == 0


class TestRoundtrip:
    """Tests for save/load roundtrip consistency."""

    def test_roundtrip_preserves_full_affine(self, tmp_path):
        """Save and load roundtrip preserves the full affine (rotation + shear)."""
        # 45° rotation in XY plane + zoom + translation — not a diagonal affine.
        affine = np.array(
            [
                [1.41421356, -1.41421356, 0.0, 10.0],
                [1.41421356, 2.82842712, 0.0, -5.0],
                [0.0, 0.0, 4.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = np.random.default_rng(0).random((3, 4, 5)).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        nifti_path = tmp_path / "rotated.nii.gz"
        img.to_filename(nifti_path)

        da = load_nifti(nifti_path)
        out_path = tmp_path / "roundtrip_rotated.nii.gz"
        save_nifti(da, out_path)

        reloaded = nib.load(out_path)
        np.testing.assert_allclose(reloaded.header.get_sform(), affine, atol=1e-5)

    def test_roundtrip_3d(self, tmp_path, sample_3d_volume):
        """Save and load preserves 3D data and attributes."""
        # Drop scalar time coordinate for pure 3D test
        original = sample_3d_volume.drop_vars("time")

        nifti_path = tmp_path / "roundtrip_3d.nii.gz"
        save_nifti(original, nifti_path)

        loaded = load_nifti(nifti_path)

        np.testing.assert_array_almost_equal(
            np.asarray(loaded), original.values, decimal=5
        )

    def test_roundtrip_regular_timing(self, tmp_path):
        """Regular time coord roundtrips via RepetitionTime/DelayAfterTrigger."""
        time_values = np.array([0.5, 1.0, 1.5, 2.0])  # TR=0.5, delay=0.5
        rng = np.random.default_rng(0)
        original = xr.DataArray(
            rng.random((4, 6, 4, 2)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={"time": time_values},
        )

        nifti_path = tmp_path / "regular_timing.nii.gz"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(original, nifti_path)

        sidecar_path = tmp_path / "regular_timing.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar["RepetitionTime"] == pytest.approx(0.5)
        assert sidecar["DelayAfterTrigger"] == pytest.approx(0.5)
        assert "VolumeTiming" not in sidecar

        loaded = load_nifti(nifti_path)
        np.testing.assert_allclose(loaded.coords["time"].values, time_values)
        assert "RepetitionTime" not in loaded.attrs
        assert "DelayAfterTrigger" not in loaded.attrs

    def test_roundtrip_regular_timing_no_delay(self, tmp_path):
        """Regular time coord starting at 0 omits DelayAfterTrigger."""
        time_values = np.array([0.0, 0.5, 1.0, 1.5])  # TR=0.5, no delay
        rng = np.random.default_rng(0)
        original = xr.DataArray(
            rng.random((4, 6, 4, 2)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={"time": time_values},
        )

        nifti_path = tmp_path / "no_delay.nii.gz"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(original, nifti_path)

        sidecar_path = tmp_path / "no_delay.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert "DelayAfterTrigger" not in sidecar

        loaded = load_nifti(nifti_path)
        np.testing.assert_allclose(loaded.coords["time"].values, time_values)

    def test_save_end_referenced_scan_timing_uses_explicit_duration(self, tmp_path):
        """Explicit end-reference duration avoids spurious validation warnings."""
        rng = np.random.default_rng(0)
        time_values = np.array([0.4, 2.8, 5.2, 7.6])
        da = xr.DataArray(
            rng.random((4, 4, 3, 2)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(
                    time_values,
                    dims=["time"],
                    attrs={
                        "units": "s",
                        "volume_acquisition_reference": "end",
                        "volume_acquisition_duration": 0.4,
                    },
                ),
                "slice_time": xr.DataArray(
                    time_values[:, np.newaxis] + np.array([0.0, 1.8, 0.6, 1.2]),
                    dims=("time", "z"),
                    attrs={
                        "units": "s",
                        "volume_acquisition_reference": "end",
                        "volume_acquisition_duration": 0.4,
                    },
                ),
            },
        )

        output_path = tmp_path / "end_referenced_scan_timing.nii.gz"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_nifti(da, output_path)

        assert not any(
            "validation warning when saving" in str(w.message) for w in caught
        )

        with open(tmp_path / "end_referenced_scan_timing.json") as f:
            sidecar = json.load(f)

        assert sidecar["RepetitionTime"] == pytest.approx(2.4)
        assert sidecar["DelayTime"] == pytest.approx(2.0)
        assert "DelayAfterTrigger" not in sidecar
        assert sidecar["SliceTiming"] == pytest.approx([0.0, 1.8, 0.6, 1.2])

    def test_save_regular_timing_writes_delay_time_from_duration(self, tmp_path):
        """Regular timing exports dead time when TR exceeds acquisition duration."""
        rng = np.random.default_rng(0)
        da = xr.DataArray(
            rng.random((4, 4, 3, 2)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(
                    [0.0, 2.4, 4.8, 7.2],
                    dims=["time"],
                    attrs={
                        "units": "s",
                        "volume_acquisition_reference": "start",
                        "volume_acquisition_duration": 2.2,
                    },
                ),
                "z": xr.DataArray(
                    np.arange(4) * 0.1, dims=["z"], attrs={"units": "mm"}
                ),
                "y": xr.DataArray(
                    np.arange(3) * 0.1, dims=["y"], attrs={"units": "mm"}
                ),
                "x": xr.DataArray(
                    np.arange(2) * 0.1, dims=["x"], attrs={"units": "mm"}
                ),
            },
        )

        output_path = tmp_path / "regular_delay_time.nii.gz"
        save_nifti(da, output_path)

        with open(tmp_path / "regular_delay_time.json") as f:
            sidecar = json.load(f)

        assert sidecar["RepetitionTime"] == pytest.approx(2.4)
        assert sidecar["DelayTime"] == pytest.approx(0.2)
        assert "FrameAcquisitionDuration" not in sidecar

    def test_save_warns_when_time_reference_is_missing(self, tmp_path):
        """Saving warns when time reference metadata is absent and onset is assumed."""
        rng = np.random.default_rng(0)
        da = xr.DataArray(
            rng.random((4, 4, 3, 2)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(
                    [0.0, 1.0, 2.0, 3.0],
                    dims=["time"],
                    attrs={"units": "s", "volume_acquisition_duration": 0.5},
                ),
                "z": xr.DataArray(
                    np.arange(4) * 0.1, dims=["z"], attrs={"units": "mm"}
                ),
                "y": xr.DataArray(
                    np.arange(3) * 0.1, dims=["y"], attrs={"units": "mm"}
                ),
                "x": xr.DataArray(
                    np.arange(2) * 0.1, dims=["x"], attrs={"units": "mm"}
                ),
            },
        )

        output_path = tmp_path / "missing_reference.nii.gz"
        with pytest.warns(
            UserWarning,
            match="Coordinate 'time' has no `volume_acquisition_reference` attribute",
        ):
            save_nifti(da, output_path)

    def test_roundtrip_volume_timing(self, tmp_path):
        """Irregular time coord roundtrips via VolumeTiming; pixdim[4] is 0."""
        time_values = np.array([0.0, 1.5, 2.8, 4.6])  # non-uniform spacing
        rng = np.random.default_rng(0)
        original = xr.DataArray(
            rng.random((4, 6, 4, 2)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(
                    time_values,
                    dims=["time"],
                    attrs={
                        "units": "s",
                        "volume_acquisition_duration": 4.6,
                        "volume_acquisition_reference": "start",
                    },
                ),
                "z": np.arange(6) * 0.1,
                "y": np.arange(4) * 0.1,
                "x": np.arange(2) * 0.1,
            },
        )

        nifti_path = tmp_path / "volume_timing.nii.gz"
        with pytest.warns(UserWarning, match="non-uniform sampling"):
            save_nifti(original, nifti_path)

        sidecar_path = tmp_path / "volume_timing.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar["VolumeTiming"] == pytest.approx(time_values.tolist())
        assert sidecar["FrameAcquisitionDuration"] == pytest.approx(4.6)
        assert "RepetitionTime" not in sidecar
        assert nib.load(nifti_path).header.get_zooms()[3] == pytest.approx(0.0)

        loaded = load_nifti(nifti_path)
        np.testing.assert_allclose(loaded.coords["time"].values, time_values)
        assert loaded.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(4.6)

    def test_save_uses_time_coordinate_frame_acquisition_duration(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """Irregular timings use `time.attrs['volume_acquisition_duration']`."""
        original = sample_4d_volume.isel(time=slice(0, 4)).copy()
        original = original.assign_coords(
            time=xr.DataArray(
                [0.0, 1.5, 2.8, 4.6],
                dims=["time"],
                attrs={"units": "s", "volume_acquisition_duration": 1.5},
            )
        )

        nifti_path = tmp_path / "inferred_frame_duration.nii.gz"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_nifti(original, nifti_path)

        assert not any(
            "validation warning when saving" in str(w.message) for w in caught
        )

        sidecar_path = tmp_path / "inferred_frame_duration.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["VolumeTiming"] == pytest.approx([0.0, 1.5, 2.8, 4.6])
        assert sidecar["FrameAcquisitionDuration"] == pytest.approx(1.5)

    def test_save_irregular_time_approximates_frame_acquisition_duration_from_spacing(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """Irregular timings fall back to median time spacing when no better duration exists."""
        original = sample_4d_volume.isel(time=slice(0, 4)).copy()
        original = original.assign_coords(
            time=xr.DataArray([0.0, 1.5, 2.8, 4.6], dims=["time"], attrs={"units": "s"})
        )

        nifti_path = tmp_path / "approx_frame_duration.nii.gz"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_nifti(original, nifti_path)

        assert any("non-uniform sampling" in str(w.message) for w in caught)
        assert any(
            "Approximating it from the median time spacing" in str(w.message)
            for w in caught
        )
        assert not any(
            "validation warning when saving" in str(w.message) for w in caught
        )

        sidecar_path = tmp_path / "approx_frame_duration.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["VolumeTiming"] == pytest.approx([0.0, 1.5, 2.8, 4.6])
        assert sidecar["FrameAcquisitionDuration"] == pytest.approx(1.5)

    def test_load_repetition_time_pixdim_mismatch_warns(self, tmp_path):
        """Mismatched RepetitionTime in sidecar vs pixdim[4] emits a warning."""
        data = np.random.default_rng(0).random((3, 4, 5, 6)).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms([1.0, 1.0, 1.0, 2.0])  # pixdim[4] = 2.0
        nifti_path = tmp_path / "mismatch.nii.gz"
        img.to_filename(nifti_path)

        sidecar = {"RepetitionTime": 1.0}  # contradicts pixdim[4]=2.0
        sidecar_path = tmp_path / "mismatch.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        with pytest.warns(UserWarning, match="RepetitionTime"):
            loaded = load_nifti(nifti_path)

        # Sidecar value wins: RepetitionTime=1.0, 6 volumes → [0, 1, 2, 3, 4, 5].
        np.testing.assert_allclose(
            loaded.coords["time"].values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        )

    def test_roundtrip_4d(self, tmp_path, sample_4d_volume):
        """Save/load roundtrip preserves 4D data and non-derived BIDS attrs."""
        original = sample_4d_volume.copy()
        original.attrs.update(
            {
                "task_name": "rest",
                "manufacturer": "Verasonics",
                "probe_central_frequency": 15.0,
            }
        )

        nifti_path = tmp_path / "roundtrip_4d.nii.gz"
        save_nifti(original, nifti_path)

        loaded = load_nifti(nifti_path)

        np.testing.assert_allclose(np.asarray(loaded), original.values)
        assert loaded.attrs["task_name"] == "rest"
        assert loaded.attrs["manufacturer"] == "Verasonics"
        assert loaded.attrs["probe_central_frequency"] == 15.0

    def test_roundtrip_preserves_units(self, tmp_path, sample_3d_volume):
        """Spatial units survive a save/load roundtrip."""
        da = sample_3d_volume.drop_vars("time").copy()
        for dim in ("z", "y", "x"):
            da.coords[dim].attrs["units"] = "um"

        nifti_path = tmp_path / "units_roundtrip.nii.gz"
        save_nifti(da, nifti_path)

        loaded = load_nifti(nifti_path)

        assert loaded.coords["z"].attrs["units"] == "um"
        assert loaded.coords["y"].attrs["units"] == "um"
        assert loaded.coords["x"].attrs["units"] == "um"

    def test_load_nii_uncompressed_with_sidecar(self, tmp_path):
        """load_nifti reads an uncompressed .nii file and merges its JSON sidecar."""
        rng = np.random.default_rng(0)
        data = rng.random((4, 3, 2)).astype(np.float32)
        nifti_path = tmp_path / "plain.nii"
        nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

        sidecar = {
            "TaskName": "test",
            "RepetitionTime": 1.0,
            "instrument": "probe_A",
            "session": 42,
        }
        sidecar_path = tmp_path / "plain.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        da = load_nifti(nifti_path)

        assert da.attrs.get("instrument") == "probe_A"
        assert da.attrs.get("session") == 42

    def test_save_converts_time_to_seconds(self, tmp_path, sample_4d_volume):
        """Time values are converted to seconds when saving for BIDS compliance."""
        original = sample_4d_volume.copy()
        original = original.assign_coords(
            time=xr.DataArray(
                np.arange(original.sizes["time"]) * 1000.0,
                dims=["time"],
                attrs={"units": "ms"},
            )
        )

        nifti_path = tmp_path / "time_ms.nii.gz"
        save_nifti(original, nifti_path)

        # Check sidecar has RepetitionTime in seconds (1.0, not 1000.0).
        sidecar_path = nifti_path.with_suffix("").with_suffix(".json")
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["RepetitionTime"] == 1.0  # Converted from 1000 ms to 1 s.
        assert "DelayAfterTrigger" not in sidecar

    def test_save_converts_processing_duration_attrs_to_seconds(
        self, tmp_path, sample_4d_volume
    ) -> None:
        """Processing duration and stride attrs are converted to seconds in sidecar."""
        da = sample_4d_volume.copy().assign_coords(
            time=xr.DataArray(
                np.arange(sample_4d_volume.sizes["time"]) * 100.0,
                dims=["time"],
                attrs={"units": "ms"},
            )
        )
        da.attrs.update(
            {
                "clutter_filter_window_duration": 300.0,
                "clutter_filter_window_stride": 200.0,
                "power_doppler_integration_duration": 80.0,
                "power_doppler_integration_stride": 40.0,
                "axial_velocity_integration_duration": 160.0,
                "axial_velocity_integration_stride": 50.0,
                "bmode_integration_duration": 100.0,
                "bmode_integration_stride": 60.0,
            }
        )

        nifti_path = tmp_path / "processing_duration_ms.nii.gz"
        save_nifti(da, nifti_path)

        sidecar_path = nifti_path.with_suffix("").with_suffix(".json")
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["ClutterFilterWindowDuration"] == pytest.approx(0.3)
        assert sidecar["ClutterFilterWindowStride"] == pytest.approx(0.2)
        assert sidecar["PowerDopplerIntegrationDuration"] == pytest.approx(0.08)
        assert sidecar["PowerDopplerIntegrationStride"] == pytest.approx(0.04)
        assert sidecar["ConfUSIusAxialVelocityIntegrationDuration"] == pytest.approx(
            0.16
        )
        assert sidecar["ConfUSIusAxialVelocityIntegrationStride"] == pytest.approx(0.05)
        assert sidecar["ConfUSIusBmodeIntegrationDuration"] == pytest.approx(0.1)
        assert sidecar["ConfUSIusBmodeIntegrationStride"] == pytest.approx(0.06)

    def test_save_warns_on_inconsistent_spatial_units(self, tmp_path, sample_3d_volume):
        """Saving warns when spatial dimensions have different units."""
        da = sample_3d_volume.drop_vars("time").copy()
        da.coords["x"].attrs["units"] = "m"

        nifti_path = tmp_path / "mixed_units.nii.gz"
        with pytest.warns(UserWarning, match="different units"):
            save_nifti(da, nifti_path)
