"""Unit tests for confusius.multipose module."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from confusius.io.scan import load_scan
from confusius.multipose import consolidate_poses

_NPOSE = 3
_SIZE_Y = 1
_T = 5


# ---------------------------------------------------------------------------
# Tests: consolidate_poses
# ---------------------------------------------------------------------------


class TestConsolidatePoses:
    """Tests for consolidate_poses."""

    def test_physical_to_lab_consolidated_rotation_orthogonal(
        self, scan_3d: xr.DataArray
    ) -> None:
        """Consolidated affine is 4x4 with an orthogonal rotation block."""
        result = consolidate_poses(scan_3d)
        A = np.asarray(result.attrs["affines"]["physical_to_lab"])
        assert A.shape == (4, 4)
        R = A[:3, :3]
        # R^T @ R should be the identity for an orthogonal matrix.
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_4dscan_updates_time_coord_from_pose_timing(
        self, scan_4d: xr.DataArray
    ) -> None:
        """4Dscan consolidation derives whole-volume timings from pose timings."""
        result = consolidate_poses(scan_4d)

        np.testing.assert_allclose(
            result.coords["time"].values, [0.3, 0.6, 0.9, 1.2, 1.5]
        )
        assert result.coords["time"].attrs["volume_acquisition_reference"] == "end"
        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(0.3)

    def test_4dscan_slice_time_values(self, scan_4d: xr.DataArray) -> None:
        """4Dscan consolidation keeps absolute per-slice timestamps on (time, z)."""
        result = consolidate_poses(scan_4d)
        assert result.dims == ("time", "z", "y", "x")
        assert "pose" not in result.dims
        assert "slice_time" in result.coords
        assert result.coords["slice_time"].dims == ("time", "z")
        assert result.coords["slice_time"].shape == (_T, _NPOSE * _SIZE_Y)
        assert result.coords["slice_time"].attrs.get("units") == "s"
        orig_pt = scan_4d.coords["pose_time"].values  # (T, npose)
        # Recover which original pose each consolidated z-slice came from.
        ptl = np.asarray(scan_4d.attrs["affines"]["physical_to_lab"])
        z_mm = scan_4d.coords["z"].values
        r0 = ptl[:, :3, 0]
        t_lab = ptl[:, :3, 3]
        lab_pos = (
            r0[:, np.newaxis, :] * z_mm[np.newaxis, :, np.newaxis]
            + t_lab[:, np.newaxis, :]
        )
        lab_pos_flat = lab_pos.reshape(-1, 3)
        centered = lab_pos_flat - lab_pos_flat.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        sweep_axis = vt[0]
        if sweep_axis[np.argmax(np.abs(sweep_axis))] < 0:
            sweep_axis = -sweep_axis
        proj = lab_pos_flat @ sweep_axis
        sorted_flat = np.argsort(proj)
        pose_idx = sorted_flat // len(z_mm)
        expected = orig_pt[:, pose_idx]
        np.testing.assert_array_equal(result.coords["slice_time"].values, expected)

    def test_4dscan_updates_volume_timing_when_pose_duration_is_known(
        self, scan_4d: xr.DataArray
    ) -> None:
        """Known per-pose duration lets consolidation derive full-volume timing."""
        scan_4d = scan_4d.assign_coords(
            time=xr.DataArray(
                [0.4, 2.2, 4.0, 5.8, 7.6],
                dims=("time",),
                attrs={
                    "units": "s",
                    "volume_acquisition_reference": "end",
                    "volume_acquisition_duration": 0.4,
                },
            ),
            pose_time=xr.DataArray(
                [
                    [0.4, 1.0, 1.6],
                    [2.2, 2.8, 3.4],
                    [4.0, 4.6, 5.2],
                    [5.8, 6.4, 7.0],
                    [7.6, 8.2, 8.8],
                ],
                dims=("time", "pose"),
                attrs={
                    "units": "s",
                    "volume_acquisition_reference": "end",
                    "volume_acquisition_duration": 0.4,
                },
            ),
        )

        result = consolidate_poses(scan_4d)

        np.testing.assert_allclose(
            result.coords["time"].values, [1.6, 3.4, 5.2, 7.0, 8.8]
        )
        assert result.coords["time"].attrs["volume_acquisition_reference"] == "end"
        assert result.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(1.6)
        assert (
            result.coords["slice_time"].attrs["volume_acquisition_reference"] == "end"
        )
        assert result.coords["slice_time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(0.4)

    def test_3dscan_no_slice_time(self, scan_3d: xr.DataArray) -> None:
        """3Dscan consolidation produces no slice_time coordinate."""
        result = consolidate_poses(scan_3d)
        assert "slice_time" not in result.coords

    def test_physical_to_brain_consolidated_with_bps(
        self, scan_3d_path: Path, bps_path: Path
    ) -> None:
        """Consolidating a 3Dscan loaded with BPS preserves the brain link.

        Per pose, `physical_to_brain[p] = inv(brain_to_lab) @ physical_to_lab[p]`
        with `inv(brain_to_lab)` constant across poses. After consolidation, the
        same relationship must hold against the consolidated `physical_to_lab`,
        i.e. `consolidated_physical_to_brain = inv(brain_to_lab) @
        consolidated_physical_to_lab`.
        """
        da = load_scan(scan_3d_path, bps_path=bps_path)
        affines = da.attrs["affines"]
        # Recover the constant `inv(brain_to_lab)` link from any pose.
        link = affines["physical_to_brain"][0] @ np.linalg.inv(
            affines["physical_to_lab"][0]
        )

        result = consolidate_poses(da)

        assert "physical_to_brain" in result.attrs["affines"]
        expected = link @ result.attrs["affines"]["physical_to_lab"]
        np.testing.assert_allclose(
            result.attrs["affines"]["physical_to_brain"], expected, rtol=1e-10
        )

    def test_unlinked_extra_per_pose_affine_raises(self, scan_3d: xr.DataArray) -> None:
        """An extra per-pose affine that is not a constant left-link of the main
        affine must raise ``ValueError`` rather than silently producing a wrong
        consolidated affine.
        """
        ptl = np.asarray(scan_3d.attrs["affines"]["physical_to_lab"]).copy()
        # Perturb pose 1 only: link derived from pose 0 is identity, so the chain
        # `link @ physical_to_lab` cannot reproduce the perturbed pose.
        unlinked = ptl.copy()
        unlinked[1, :3, 3] += np.array([0.5, 0.0, 0.0])
        scan_3d.attrs["affines"]["physical_to_unlinked"] = unlinked

        with pytest.raises(
            ValueError, match="not a constant left-link of 'physical_to_lab'"
        ):
            consolidate_poses(scan_3d)

    def test_static_affine_passed_through(self, scan_3d: xr.DataArray) -> None:
        """A static `(4, 4)` affine is carried through consolidation unchanged."""
        static = np.eye(4, dtype=np.float64)
        static[:3, 3] = [1.0, 2.0, 3.0]
        scan_3d.attrs["affines"]["physical_to_static"] = static

        result = consolidate_poses(scan_3d)

        assert "physical_to_static" in result.attrs["affines"]
        np.testing.assert_array_equal(
            result.attrs["affines"]["physical_to_static"], static
        )

    def test_no_pose_dim_raises(self, scan_2d: xr.DataArray) -> None:
        """consolidate_poses raises ValueError when there is no pose dimension."""
        with pytest.raises(ValueError, match="no 'pose' dimension"):
            consolidate_poses(scan_2d)

    def test_irregular_positions_raises(self, scan_3d_irregular_path: Path) -> None:
        """consolidate_poses raises ValueError when positions are not regularly spaced."""
        da = load_scan(scan_3d_irregular_path)
        with pytest.raises(ValueError, match="not regularly spaced"):
            consolidate_poses(da)

    def test_non_1d_sweep_warns(self, scan_3d_2d_sweep_path: Path) -> None:
        """consolidate_poses warns when the sweep has a significant secondary component.

        The 2D sweep fixture also produces irregular spacings after projection onto the
        diagonal axis, so a ValueError follows the warning. Both are expected here.
        """
        da = load_scan(scan_3d_2d_sweep_path)
        with pytest.warns(UserWarning, match="not purely 1D"):
            with pytest.raises(ValueError):
                consolidate_poses(da)

    def test_varying_rotation_raises(self, scan_3d_varying_rotation_path: Path) -> None:
        """consolidate_poses raises ValueError when rotation varies across poses."""
        da = load_scan(scan_3d_varying_rotation_path)
        with pytest.raises(ValueError, match="not constant across poses"):
            consolidate_poses(da)

    def test_invalid_sweep_dim_raises(self, scan_3d: xr.DataArray) -> None:
        """consolidate_poses raises ValueError for an unrecognised sweep_dim."""
        with pytest.raises(ValueError, match="sweep_dim must be one of"):
            consolidate_poses(scan_3d, sweep_dim="w")

    def test_custom_affines_key(self, scan_3d: xr.DataArray) -> None:
        """consolidate_poses uses the affines_key argument to select the affine."""
        # Copy the existing affine under a custom key and verify the result is
        # identical to the default call.
        affine = scan_3d.attrs["affines"]["physical_to_lab"]
        custom_attrs = {
            **scan_3d.attrs,
            "affines": {"my_affine": affine},
        }
        da_custom = scan_3d.assign_attrs(custom_attrs)
        result_default = consolidate_poses(scan_3d)
        result_custom = consolidate_poses(da_custom, affines_key="my_affine")
        np.testing.assert_array_equal(result_default.values, result_custom.values)
        np.testing.assert_array_equal(
            result_default.coords["z"].values, result_custom.coords["z"].values
        )

    @pytest.mark.parametrize(
        ("sweep_dim", "sweep_unit"),
        [("z", "um"), ("y", "mm"), ("x", "m")],
    )
    def test_consolidates_all_sweep_dims(self, sweep_dim: str, sweep_unit: str) -> None:
        """consolidate_poses correctly merges poses for any spatial sweep dimension.

        This test constructs a DataArray whose affine translates along the requested
        sweep column and verifies that:

        - the output dims are ``(sweep_dim, <other1>, <other2>)`` with no ``pose``;
        - the consolidated coordinate is the expected regular grid with propagated units;
        - each consolidated slice contains exactly the data values from the correct
          ``(pose, sweep_dim)`` combination.
        """
        npose = 3
        sizes = {"z": 2, "y": 4, "x": 3}
        intra_step = 0.2  # mm voxel pitch
        voxel_size = 0.15

        _SWEEP_DIM_TO_COL = {"z": 0, "y": 1, "x": 2}
        sweep_col = _SWEEP_DIM_TO_COL[sweep_dim]
        n_sweep = sizes[sweep_dim]
        inter_step = n_sweep * intra_step  # poses tile without gaps

        rng = np.random.default_rng(7)
        data = rng.random((npose, sizes["z"], sizes["y"], sizes["x"]))

        affines = np.stack([np.eye(4) for _ in range(npose)])
        for i in range(npose):
            affines[i, :3, 3][sweep_col] = i * inter_step

        coords: dict[str, xr.DataArray] = {
            "pose": xr.DataArray(np.arange(npose), dims=["pose"]),
            **{
                d: xr.DataArray(
                    np.arange(sizes[d]) * intra_step,
                    dims=[d],
                    attrs={
                        "units": sweep_unit if d == sweep_dim else "mm",
                        "voxdim": voxel_size,
                    },
                )
                for d in ("z", "y", "x")
            },
        }
        da = xr.DataArray(
            data,
            dims=["pose", "z", "y", "x"],
            coords=coords,
            attrs={"affines": {"physical_to_lab": affines}},
        )

        result = consolidate_poses(da, sweep_dim=sweep_dim)

        other_dims = [d for d in ["z", "y", "x"] if d != sweep_dim]
        assert result.dims == tuple([sweep_dim] + other_dims)
        assert "pose" not in result.dims
        assert result.sizes[sweep_dim] == npose * n_sweep
        np.testing.assert_allclose(
            result.coords[sweep_dim].values,
            np.arange(npose * n_sweep) * intra_step,
        )
        assert result.coords[sweep_dim].attrs.get("units") == sweep_unit
        assert result.coords[sweep_dim].attrs["voxdim"] == pytest.approx(voxel_size)

        # Verify data values: for each pose p and local sweep index si, the
        # consolidated flat index is p*n_sweep + si (poses are sorted ascending).
        for p in range(npose):
            for si in range(n_sweep):
                flat_idx = p * n_sweep + si
                # Expected slice: fix pose and sweep dim, free other dims.
                dim_order = ["z", "y", "x"]
                idx_dict: dict[str, int | slice] = {d: slice(None) for d in dim_order}
                idx_dict[sweep_dim] = si
                idx_tuple = (p,) + tuple(idx_dict[d] for d in dim_order)
                expected = data[idx_tuple]
                np.testing.assert_array_equal(result.values[flat_idx], expected)
