"""Unit tests for confusius.io.nifti module."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import xarray as xr

from confusius.io.nifti import load_nifti, save_nifti


@pytest.fixture
def nifti_2d_path(tmp_path: Path) -> Path:
    """Create a 2D NIfTI file for testing."""
    rng = np.random.default_rng(0)
    data = rng.random((8, 6)).astype(np.float32)
    nifti_path = tmp_path / "test_2d.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    return nifti_path


@pytest.fixture
def nifti_3d_path(tmp_path: Path) -> Path:
    """Create a 3D NIfTI file for testing."""
    rng = np.random.default_rng(0)
    data = rng.random((10, 8, 6)).astype(np.float32)
    nifti_path = tmp_path / "test_3d.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    return nifti_path


@pytest.fixture
def nifti_4d_path(tmp_path: Path) -> Path:
    """Create a 4D NIfTI file for testing."""
    rng = np.random.default_rng(0)
    data = rng.random((12, 10, 8, 6)).astype(np.float64)
    nifti_path = tmp_path / "test_4d.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    return nifti_path


@pytest.fixture
def nifti_with_sidecar(tmp_path: Path) -> Path:
    """Create a 3D NIfTI file with JSON sidecar."""
    rng = np.random.default_rng(0)
    data = rng.random((8, 6, 4)).astype(np.float32)
    nifti_path = tmp_path / "test_sidecar.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

    sidecar = {"custom_meta": "test_value", "acquisition": "test_acq"}
    sidecar_path = tmp_path / "test_sidecar.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)

    return nifti_path


class TestLoadNifti:
    """Tests for load_nifti function."""

    def test_load_2d_nifti(self, nifti_2d_path: Path) -> None:
        """Loading 2D NIfTI creates DataArray with spatial dims only."""
        da = load_nifti(nifti_2d_path)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("y", "x")
        assert da.shape == (6, 8)
        assert da.dtype == np.float32

    def test_load_3d_nifti(self, nifti_3d_path: Path) -> None:
        """Loading 3D NIfTI creates DataArray with spatial dims only."""
        da = load_nifti(nifti_3d_path)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("z", "y", "x")
        assert da.shape == (6, 8, 10)
        assert da.dtype == np.float32

    def test_load_4d_nifti(self, nifti_4d_path: Path) -> None:
        """Loading 4D NIfTI creates DataArray with time dimension."""
        da = load_nifti(nifti_4d_path)

        assert isinstance(da, xr.DataArray)
        assert da.dims == ("time", "z", "y", "x")
        assert da.shape == (6, 8, 10, 12)
        assert da.dtype == np.float64

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
        assert "units" not in da.coords["x"].attrs

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
        # The old diagonal approach (affine[col, col]) would give spacing 0 for
        # x and y; column norms give the correct values [2, 3, 4].
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
        # Coords start at origin and are stepped by column norm (not diagonal,
        # which would give spacing 0 for the rotated x and y axes here).
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

    def test_load_nifti_lazy(self, tmp_path: Path) -> None:
        """Loading creates lazy Dask array."""
        import dask.array as dask_array

        data = np.random.default_rng(0).random((8, 6, 4)).astype(np.float32)
        nifti_path = tmp_path / "test_lazy.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

        result = load_nifti(nifti_path)

        assert isinstance(result.data, dask_array.Array)

    def test_load_nifti_with_sidecar(self, nifti_with_sidecar: Path) -> None:
        """Loading with sidecar JSON merges metadata."""
        da = load_nifti(nifti_with_sidecar)

        assert da.attrs.get("custom_meta") == "test_value"
        assert da.attrs.get("acquisition") == "test_acq"


class TestSaveNifti:
    """Tests for `save_nifti` function."""

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

    def test_save_non_uniform_coords_warns(self, tmp_path):
        """Saving a DataArray with non-uniform coordinate spacing emits a warning."""
        data = np.random.default_rng(0).random((4, 3)).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "x"],
            coords={"z": [0.0, 1.0, 3.0, 6.0], "x": [0.0, 1.0, 2.0]},
        )

        with pytest.warns(UserWarning, match="non-uniform"):
            save_nifti(da, tmp_path / "nonuniform.nii.gz")

    def test_save_creates_sidecar(self, tmp_path):
        """Saving always creates a JSON sidecar alongside the NIfTI file."""
        data = np.random.default_rng(0).random((4, 3, 2)).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.linspace(0, 10, 4),
                "y": np.linspace(0, 5, 3),
                "x": np.linspace(0, 3, 2),
            },
            attrs={"custom": "value"},
        )

        output_path = tmp_path / "with_sidecar.nii.gz"
        save_nifti(da, output_path)

        sidecar_path = tmp_path / "with_sidecar.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar.get("custom") == "value"
        # Spatial coords are encoded in the NIfTI header, not duplicated in the sidecar.
        assert "x" not in sidecar
        assert "y" not in sidecar
        assert "z" not in sidecar

    def test_explicit_qform_overrides_attrs(self, tmp_path):
        """Explicit physical_to_qform= kwarg takes precedence over attrs['affines']['physical_to_qform']."""
        data = np.zeros((4, 3, 2), dtype=np.float32)
        # Diagonal physical_to_qform (ConfUSIus convention, identity rotation).
        physical_to_qform = np.diag([1.0, 1.0, 1.0, 1.0])
        # Store a different affine in attrs that should be overridden.
        different_affine = np.diag([9.0, 9.0, 9.0, 1.0])
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(4) * 1.0,
                "y": np.arange(3) * 1.0,
                "x": np.arange(2) * 1.0,
            },
            attrs={"affines": {"physical_to_qform": different_affine}},
        )
        output_path = tmp_path / "explicit_qform.nii.gz"
        save_nifti(da, output_path, physical_to_qform=physical_to_qform)

        loaded = nib.load(output_path)
        q = loaded.header.get_qform()
        # physical_to_qform is the identity rotation, so the qform rotation block
        # should be identity scaled by the voxel spacing (1.0), not the 9× one.
        np.testing.assert_allclose(q[:3, :3], np.eye(3), atol=1e-6)

    def test_explicit_sform_sets_sform_code(self, tmp_path):
        """Providing physical_to_sform= writes a sform with code=1 even if attrs has sform_code=0."""
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
        )
        output_path = tmp_path / "explicit_sform.nii.gz"
        save_nifti(da, output_path, physical_to_sform=physical_to_sform)

        loaded = nib.load(output_path)
        assert loaded.header.get_sform(coded=True)[1] == 1

    def test_explicit_sform_code_kwarg_overrides_attrs(self, tmp_path):
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
        save_nifti(da, output_path, physical_to_sform=physical_to_sform, sform_code=2)

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

    def test_no_sform_kwarg_writes_no_sform(self, tmp_path):
        """Without physical_to_sform= and no attrs sform, the saved file has sform_code=0."""
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

    def test_roundtrip_3d(self, tmp_path):
        """Save and load preserves 3D data and attributes."""
        original_data = np.random.default_rng(0).random((6, 4, 2)).astype(np.float32)
        original = xr.DataArray(
            original_data,
            dims=["z", "y", "x"],
        )

        nifti_path = tmp_path / "roundtrip_3d.nii.gz"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(original, nifti_path)

        loaded = load_nifti(nifti_path)

        np.testing.assert_array_almost_equal(
            np.asarray(loaded), original_data, decimal=5
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

    def test_roundtrip_volume_timing(self, tmp_path):
        """Irregular time coord roundtrips via VolumeTiming; pixdim[4] is 0."""
        time_values = np.array([0.0, 1.5, 2.8, 4.6])  # non-uniform spacing
        rng = np.random.default_rng(0)
        original = xr.DataArray(
            rng.random((4, 6, 4, 2)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={"time": time_values},
        )

        nifti_path = tmp_path / "volume_timing.nii.gz"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(original, nifti_path)

        sidecar_path = tmp_path / "volume_timing.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar["VolumeTiming"] == pytest.approx(time_values.tolist())
        assert "RepetitionTime" not in sidecar
        assert nib.load(nifti_path).header.get_zooms()[3] == pytest.approx(0.0)

        loaded = load_nifti(nifti_path)
        np.testing.assert_allclose(loaded.coords["time"].values, time_values)
        assert "VolumeTiming" not in loaded.attrs

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

    def test_roundtrip_4d(self, tmp_path):
        """Save and load preserves 4D data."""
        original_data = np.random.default_rng(0).random((3, 6, 4, 2)).astype(np.float32)
        original = xr.DataArray(original_data, dims=["time", "z", "y", "x"])

        nifti_path = tmp_path / "roundtrip_4d.nii.gz"
        with pytest.warns(UserWarning, match="spacing is undefined"):
            save_nifti(original, nifti_path)

        loaded = load_nifti(nifti_path)

        np.testing.assert_array_almost_equal(
            np.asarray(loaded), original_data, decimal=5
        )

    def test_roundtrip_preserves_units(self, tmp_path):
        """Spatial and temporal units survive a save/load roundtrip."""
        data = np.random.default_rng(0).random((4, 3, 2)).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={
                "z": xr.DataArray(np.arange(4) * 1.0, dims="z", attrs={"units": "mm"}),
                "y": xr.DataArray(np.arange(3) * 1.0, dims="y", attrs={"units": "mm"}),
                "x": xr.DataArray(np.arange(2) * 1.0, dims="x", attrs={"units": "mm"}),
            },
        )

        nifti_path = tmp_path / "units_roundtrip.nii.gz"
        save_nifti(da, nifti_path)

        loaded = load_nifti(nifti_path)

        assert loaded.coords["z"].attrs["units"] == "mm"
        assert loaded.coords["y"].attrs["units"] == "mm"
        assert loaded.coords["x"].attrs["units"] == "mm"

    def test_load_nii_uncompressed_with_sidecar(self, tmp_path):
        """load_nifti reads an uncompressed .nii file and merges its JSON sidecar."""
        rng = np.random.default_rng(0)
        data = rng.random((4, 3, 2)).astype(np.float32)
        nifti_path = tmp_path / "plain.nii"
        nib.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)

        sidecar = {"instrument": "probe_A", "session": 42}
        sidecar_path = tmp_path / "plain.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        da = load_nifti(nifti_path)

        assert da.attrs.get("instrument") == "probe_A"
        assert da.attrs.get("session") == 42
