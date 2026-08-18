"""Unit tests for confusius.multipose module."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from confusius._utils.geometry import get_voxel_to_world_affine
from confusius.io.scan import load_scan
from confusius.multipose import consolidate_poses
from confusius.xarray import create_voxeldata

_NPOSE = 3
_SIZE_Y = 1
_T = 5


# ---------------------------------------------------------------------------
# Tests: consolidate_poses
# ---------------------------------------------------------------------------


class TestConsolidatePoses:
    """Tests for consolidate_poses."""

    def test_world_to_lab_consolidated_rotation_orthogonal(
        self, scan_3d: xr.DataArray
    ) -> None:
        """Consolidated affine is 4x4 with an orthogonal rotation block.

        Normalized to unit-length axes first: the affine itself also carries
        per-axis voxel scale, so its raw columns aren't orthonormal.
        """
        result = consolidate_poses(scan_3d)
        A = get_voxel_to_world_affine(result)
        assert A.shape == (4, 4)
        R = A[:3, :3]
        R_normalized = R / np.linalg.norm(R, axis=0, keepdims=True)
        # R_normalized^T @ R_normalized should be the identity for an orthogonal
        # (axis-aligned or rotated) frame.
        np.testing.assert_allclose(R_normalized.T @ R_normalized, np.eye(3), atol=1e-10)

    def test_no_spurious_world_to_lab_in_attrs(self, scan_3d: xr.DataArray) -> None:
        """Consolidation doesn't inject a world_to_lab attrs entry that never existed.

        Primary geometry for SCAN data is index-based (VoxelToWorldIndex), so
        `attrs["affines"]` starts empty; consolidation must not fabricate one.
        """
        assert "world_to_lab" not in scan_3d.attrs.get("affines", {})
        result = consolidate_poses(scan_3d)
        assert "world_to_lab" not in result.attrs.get("affines", {})

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

    def test_consolidates_plain_1d_time_shared_across_poses(self) -> None:
        """A plain 1D `time` coordinate (identical across poses) consolidates as-is.

        Pose-dependent primary geometry doesn't require a pose-dependent `time`
        coordinate too (see stack_poses's "shared time stays 1D" behavior) -- when
        every pose shares the same timestamps, `time` stays a plain 1D coordinate,
        and consolidation must use it directly rather than the (time, pose) path.
        """
        npose = 3
        n_sweep = 2
        intra_step = 0.2
        inter_step = n_sweep * intra_step
        time_values = np.arange(4) * 0.5
        affines = np.stack([np.eye(4) for _ in range(npose)])
        for p in range(npose):
            affines[p, 0, 3] = p * inter_step

        data = np.random.default_rng(5).random((4, npose, n_sweep, 4, 3))
        da = create_voxeldata(
            data,
            dims=["time", "pose", "k", "j", "i"],
            time=xr.DataArray(
                time_values,
                dims=["time"],
                attrs={
                    "units": "s",
                    "volume_acquisition_reference": "start",
                    "volume_acquisition_duration": 0.5,
                },
            ),
            pose=np.arange(npose),
            voxel_to_world=affines @ np.diag([intra_step, intra_step, intra_step, 1.0]),
        )

        result = consolidate_poses(da)

        np.testing.assert_array_equal(result.coords["time"].values, time_values)
        assert "slice_time" not in result.coords

    def test_4dscan_slice_time_values(self, scan_4d: xr.DataArray) -> None:
        """4Dscan consolidation keeps absolute per-slice timestamps on (time, k)."""
        result = consolidate_poses(scan_4d)
        assert result.dims == ("time", "k", "j", "i")
        assert "pose" not in result.dims
        assert "slice_time" in result.coords
        assert result.coords["slice_time"].dims == ("time", "k")
        assert result.coords["slice_time"].shape == (_T, _NPOSE * _SIZE_Y)
        assert result.coords["slice_time"].attrs.get("units") == "s"
        orig_pt = scan_4d.coords["time"].values  # (T, npose)
        # Recover which original pose each consolidated z-slice came from, reading
        # exact per-(pose, k) world positions directly from scan_4d's own
        # (already lab-space) world coordinates -- (pose, k, j, i)-shaped since
        # scan_4d's primary geometry is itself pose-dependent.
        lab_pos_flat = np.stack(
            [
                scan_4d.coords[name].isel(j=0, i=0).values.reshape(-1)
                for name in ("z", "y", "x")
            ],
            axis=-1,
        )
        centered = lab_pos_flat - lab_pos_flat.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        sweep_axis = vt[0]
        if sweep_axis[np.argmax(np.abs(sweep_axis))] < 0:
            sweep_axis = -sweep_axis
        proj = lab_pos_flat @ sweep_axis
        sorted_flat = np.argsort(proj)
        pose_idx = sorted_flat // _SIZE_Y
        expected = orig_pt[:, pose_idx]
        np.testing.assert_array_equal(result.coords["slice_time"].values, expected)

    def test_4dscan_updates_volume_timing_when_pose_duration_is_known(
        self, scan_4d: xr.DataArray
    ) -> None:
        """Known per-pose duration lets consolidation derive full-volume timing."""
        scan_4d = scan_4d.drop_vars("time").assign_coords(
            time=xr.DataArray(
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

    def test_world_to_brain_consolidated_with_bps(
        self, scan_3d_path: Path, bps_path: Path
    ) -> None:
        """Consolidating a 3Dscan loaded with BPS preserves world_to_brain unchanged.

        World coordinates are already lab space, so world_to_brain is a single
        (4, 4) affine independent of pose (see `_add_world_to_brain`); it does not
        match the (npose, 4, 4) main per-pose stack shape, so
        `_consolidate_linked_affines` passes it through unchanged rather than
        treating it as a per-pose-linked affine.
        """
        da = load_scan(scan_3d_path, bps_path=bps_path)
        world_to_brain = da.attrs["affines"]["world_to_brain"]
        assert np.asarray(world_to_brain).shape == (4, 4)

        result = consolidate_poses(da)

        np.testing.assert_allclose(
            result.attrs["affines"]["world_to_brain"], world_to_brain, rtol=1e-10
        )

    def test_unlinked_extra_per_pose_affine_raises(self, scan_3d: xr.DataArray) -> None:
        """An extra per-pose affine that is not a constant left-link of the main
        affine must raise `ValueError` rather than silently producing a wrong
        consolidated affine.
        """
        ptl = get_voxel_to_world_affine(scan_3d)
        # Perturb pose 1 only: link derived from pose 0 is identity, so the chain
        # `link @ world_to_lab` cannot reproduce the perturbed pose.
        unlinked = ptl.copy()
        unlinked[1, :3, 3] += np.array([0.5, 0.0, 0.0])
        scan_3d.attrs["affines"]["world_to_unlinked"] = unlinked

        with pytest.raises(ValueError, match="not a constant left-link of the primary"):
            consolidate_poses(scan_3d)

    def test_static_affine_passed_through(self, scan_3d: xr.DataArray) -> None:
        """A static `(4, 4)` affine is carried through consolidation unchanged."""
        static = np.eye(4, dtype=np.float64)
        static[:3, 3] = [1.0, 2.0, 3.0]
        scan_3d.attrs["affines"]["world_to_static"] = static

        result = consolidate_poses(scan_3d)

        assert "world_to_static" in result.attrs["affines"]
        np.testing.assert_array_equal(
            result.attrs["affines"]["world_to_static"], static
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
        diagonal axis, so a ValueError follows the warning. Both are expected here,
        regardless of which voxel axis auto-detection picks (equally aligned with `k`
        and `i` here): a sweep that steps along two voxel axes at once can never form
        a regular grid along either single one.
        """
        da = load_scan(scan_3d_2d_sweep_path)
        with (
            pytest.warns(UserWarning, match="not purely 1D"),
            pytest.raises(ValueError, match="not regularly spaced"),
        ):
            consolidate_poses(da)

    def test_varying_rotation_raises(self, scan_3d_varying_rotation_path: Path) -> None:
        """consolidate_poses raises ValueError when rotation varies across poses."""
        da = load_scan(scan_3d_varying_rotation_path)
        with pytest.raises(ValueError, match="not constant across poses"):
            consolidate_poses(da)

    def test_secondary_affine_rebase_before_consolidate(self) -> None:
        """Consolidating around a secondary affine requires rebasing onto it first.

        consolidate_poses always reads positions from primary (index-based)
        geometry. To consolidate around a different, secondary named affine
        linked in attrs["affines"] (e.g. "world_to_brain" alongside a primary
        "world_to_lab"-equivalent index) rather than the current primary,
        `.fusi.affine.apply(<key>)` rebases the primary index onto it first,
        giving the same consolidated result as data whose primary geometry was
        that affine from the start.
        """
        npose = 3
        n_sweep = 2
        intra_step = 0.2
        inter_step = n_sweep * intra_step  # poses tile without gaps
        data = np.random.default_rng(3).random((npose, n_sweep, 4, 3))
        spacing_diag = np.diag([intra_step, intra_step, intra_step, 1.0])
        translations = np.stack([np.eye(4) for _ in range(npose)])
        for p in range(npose):
            translations[p, 0, 3] = p * inter_step  # along k, auto-detected

        da_primary = create_voxeldata(
            data,
            dims=["pose", "k", "j", "i"],
            pose=np.arange(npose),
            voxel_to_world=translations @ spacing_diag,
        )
        # Primary geometry here is trivially pose-dependent (identical spacing per
        # pose, no translation): the real per-pose translation lives only in the
        # secondary "my_affine" entry.
        da_secondary = create_voxeldata(
            data,
            dims=["pose", "k", "j", "i"],
            pose=np.arange(npose),
            voxel_to_world=np.stack([spacing_diag] * npose),
            attrs={"affines": {"my_affine": translations}},
        )

        result_primary = consolidate_poses(da_primary)
        result_rebased = consolidate_poses(da_secondary.fusi.affine.apply("my_affine"))
        np.testing.assert_array_equal(result_primary.values, result_rebased.values)
        np.testing.assert_array_equal(
            result_primary.coords["k"].values, result_rebased.coords["k"].values
        )

    @pytest.mark.parametrize(
        ("sweep_dim", "sweep_unit"),
        [("k", "um"), ("j", "mm"), ("i", "m")],
    )
    def test_consolidates_all_sweep_dims(self, sweep_dim: str, sweep_unit: str) -> None:
        """consolidate_poses auto-detects and merges poses for any sweep dimension.

        This test constructs a DataArray whose affine translates along the requested
        sweep column (so auto-detection should pick it up unprompted) and verifies
        that:

        - the output dims are `(sweep_dim, <other1>, <other2>)` with no `pose`;
        - the consolidated coordinate is the expected regular grid with propagated units;
        - each consolidated slice contains exactly the data values from the correct
          `(pose, sweep_dim)` combination.
        """
        npose = 3
        sizes = {"k": 2, "j": 4, "i": 3}
        intra_step = 0.2  # mm voxel pitch

        _SWEEP_DIM_TO_COL = {"k": 0, "j": 1, "i": 2}
        sweep_col = _SWEEP_DIM_TO_COL[sweep_dim]
        n_sweep = sizes[sweep_dim]
        inter_step = n_sweep * intra_step  # poses tile without gaps

        rng = np.random.default_rng(7)
        data = rng.random((npose, sizes["k"], sizes["j"], sizes["i"]))

        affines = np.stack([np.eye(4) for _ in range(npose)])
        for i in range(npose):
            affines[i, :3, 3][sweep_col] = i * inter_step
        spacing_diag = np.diag([intra_step, intra_step, intra_step, 1.0])

        da = create_voxeldata(
            data,
            dims=["pose", "k", "j", "i"],
            pose=np.arange(npose),
            voxel_to_world=affines @ spacing_diag,
        )
        da.coords[{"k": "z", "j": "y", "i": "x"}[sweep_dim]].attrs["units"] = sweep_unit

        result = consolidate_poses(da)

        other_dims = [d for d in ["k", "j", "i"] if d != sweep_dim]
        assert result.dims == tuple([sweep_dim] + other_dims)
        assert "pose" not in result.dims
        assert result.sizes[sweep_dim] == npose * n_sweep
        world_sweep_dim = {"k": "z", "j": "y", "i": "x"}[sweep_dim]
        # The world coordinate is (k, j, i)-shaped (backed by a single joint
        # VoxelToWorldIndex), but only genuinely varies along sweep_dim for
        # axis-aligned geometry; reduce the other voxel dims to compare.
        sweep_coord = result.coords[world_sweep_dim].isel(dict.fromkeys(other_dims, 0))
        np.testing.assert_allclose(
            sweep_coord.values,
            np.arange(npose * n_sweep) * intra_step,
        )
        assert result.coords[world_sweep_dim].attrs.get("units") == sweep_unit

        # Verify data values: for each pose p and local sweep index si, the
        # consolidated flat index is p*n_sweep + si (poses are sorted ascending).
        for p in range(npose):
            for si in range(n_sweep):
                flat_idx = p * n_sweep + si
                # Expected slice: fix pose and sweep dim, free other dims.
                dim_order = ["k", "j", "i"]
                idx_dict: dict[str, int | slice] = {d: slice(None) for d in dim_order}
                idx_dict[sweep_dim] = si
                idx_tuple = (p,) + tuple(idx_dict[d] for d in dim_order)
                expected = data[idx_tuple]
                np.testing.assert_array_equal(result.values[flat_idx], expected)

    def test_non_pose_dependent_primary_geometry_raises(self) -> None:
        """consolidate_poses raises ValueError when primary geometry isn't pose-dependent.

        A `pose` dimension alone isn't enough; the primary `voxel_to_world` affine
        must itself be a per-pose stack (not shared identically across poses).
        """
        npose = 3
        data = np.random.default_rng(11).random((npose, 2, 4, 3))
        da = create_voxeldata(
            data,
            dims=["pose", "k", "j", "i"],
            pose=np.arange(npose),
            spacing=(0.2, 0.2, 0.2),
        )

        with pytest.raises(ValueError, match="no pose-dependent primary geometry"):
            consolidate_poses(da)

    def test_singleton_output_dim_derives_spacing_from_affine(self) -> None:
        """A non-swept output dim with a single voxel still gets its spacing from the
        input affine column norm.

        With only one voxel along an output spatial dimension, there is no pair of
        consecutive positions from which to infer spacing via `numpy.diff`. The affine
        is ground truth regardless of axis length, so the output voxel-to-world index
        construction must derive spacing from the affine column.
        """
        npose = 3
        intra_step = 0.2
        data = np.random.default_rng(7).random((npose, 2, 1, 3))
        affines = np.stack([np.eye(4) for _ in range(npose)])
        for i in range(npose):
            affines[i, :3, 3][0] = i * 2 * intra_step
        spacing_diag = np.diag([intra_step, intra_step, intra_step, 1.0])

        da = create_voxeldata(
            data,
            dims=["pose", "k", "j", "i"],
            pose=np.arange(npose),
            voxel_to_world=affines @ spacing_diag,
        )

        result = consolidate_poses(da)

        assert result.sizes["j"] == 1
        output_affine = get_voxel_to_world_affine(result)
        spacing_y = np.linalg.norm(output_affine[:3, 1])
        assert spacing_y == pytest.approx(intra_step)
