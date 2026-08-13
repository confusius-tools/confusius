🔴 Critical — real bug, fix first

IQ processing functions silently strip CTI geometry. src/confusius/iq/process.py — process_iq_to_power_doppler (~1393), process_iq_to_bmode (~1497), process_iq_to_axial_velocity (~1750) all build their output via:
xr.DataArray(result, dims=iq.dims, coords={"time": ..., "z": iq.coords["z"], "y": ..., "x": ...}, ...)
iq.coords["z"/"y"/"x"] are lazy VoxelToWorldIndex-backed. Passing them through a fresh xr.DataArray(coords=...) constructor drops the index and materializes them as plain dense coords, and since dims=iq.dims is ("time","k","j","i"), no k/j/i coords get attached either. Verified by actually running it: has_voxel_world_geometry(result) is False, and ensure_fusi(result) raises. Every power-Doppler/B-mode/axial-velocity output from this branch can't round-trip through validation, plotting, registration, or .fusi.spacing/.fusi.origin. Fix: rebuild geometry via add_world_coords_from_voxel_affine(result, get_voxel_to_world_affine(iq), voxel_dims=...), same pattern extract/reconstruction.py::unmask already uses correctly.

Same file's compute_processed_volume_timings docstring example (line ~476) still builds iq on (time,z,y,x) dims — raises today when run verbatim. Tied to the same root cause.

Dead code (delete)

Two independent agents both flagged this one — strong signal:
- validate_registration_dataarray — registration/_utils.py:52. Only used by its own test class; every real registration entry point validates via ensure_fusi/validate_fusi_dataarray instead. Worth confirming intent before deleting (it's a deliberate-looking validator, not incidental leftover).

Also confirmed zero production callers (vulture + grep, only referenced from their own tests):
- get_affine_origin — _utils/geometry.py:1188
- get_affine_in_axis_aligned_space — _utils/coordinates.py:439
- get_grid_kwargs_from_dataarray — _utils/coordinates.py:390 (pure pass-through wrapper around get_grid_info_from_dataarray, which production code calls directly instead — looks like a rename shim nobody migrated off of)

Low priority, leave as-is: sitk_linear_transform_to_affine's bspline assert (registration/affines.py:282) is unreachable but self-documents why in its own comment.

compose_affine was flagged by one pass and correctly ruled out by another — it's intentional public API (registration/__init__.py exports it as a compose/decompose pair), not dead.

Docs still describing the pre-CTI model

- docs/user-guide/spatial-conventions.md — comprehensively stale, needs a real rewrite: claims dims are always (time,z,y,x), that world coords are "stored as three independent 1D
arrays" and "can't encode rotations/shears" (false — that's exactly what CTffine.apply pycon example returning a (da, orientation) tuple that hasn'texisted since this session's early cleanup (it returns just da now).
- docs/user-guide/multipose.md — repr examples show z/y/x as dims; sweep_diactual default is "k" (multipose/consolidate.py:197); example passessweep_dim="x" when it should be a voxel dim like "i".
- docs/user-guide/beamformed-iq.md, atlas.md — repr examples and prose show,i).
- docs/user-guide/quality-control.md:168 — comment claims cv has dims (z,y,x), should be (k,j,i).
- docs/changelog.md — nothing under the current unreleased heading documentr an ~11k-line branch. Worth a real entry before release.

Not stale (checked, false alarms to rule out re-flagging): xarray.md, io.mds.md, visualization.md already match CTI; nifti.py's and the GLM example's

Docs still describing the pre-CTI model

- docs/user-guide/spatial-conventions.md — comprehensively stale, needs a real rewrite: claims dims are always (time,z,y,x), that world coords are "stored as three independent 1D arrays" and "can't encode rotations/shears" (false — that's exactly what CTI fixes), and shows a .fusi.affine.apply pycon example returning a (da, orientation) tuple that hasn't existed since this session's early cleanup (it returns just da now).
- docs/user-guide/multipose.md — repr examples show z/y/x as dims; sweep_dim documented default is "z", actual default is "k" (multipose/consolidate.py:197); example passes sweep_dim="x" when it should be a voxel dim like "i".
- docs/user-guide/beamformed-iq.md, atlas.md — repr examples and prose show (z,y,x) dims instead of (k,j,i).
- docs/user-guide/quality-control.md:168 — comment claims cv has dims (z,y,x), should be (k,j,i).
- docs/changelog.md — nothing under the current unreleased heading documents the CTI redesign at all, for an ~11k-line branch. Worth a real entry before release.

Not stale (checked, false alarms to rule out re-flagging): xarray.md, io.mds.md, visualization.md already match CTI; nifti.py's and the GLM example'sold-model comments are intentional historical contrasts.

Naming inconsistencies

- Three spellings for the same core concept, self-contradicting within one file: voxel_affine (dominant, ~150 occurrences) vs voxel_world (has_voxel_world_geometry eto geometry.py) vs voxel_to_world (class names). has_voxel_world_geometry'sl-affine metadata" — contradicts its own name. Error strings split the same way ("must have voxel-world geometry" in geometry.py:965 vs "must have voxel-affine geometry" everywhere else). Recommend standardizing on voxel_affine for functionskeeping VoxelToWorldIndex/VoxelToWorldTransform as the class names (already
- Docstring drift across sibling registration functions: resample_volume's moving param says "singleton k axis," resample_like's says "singleton z axis" — same parammodule, one wraps the other. sitk_threads documented three different ways ar_volume/resample_like.
- New private helpers violate AGENTS.md's "imperative verb, not noun phrase" rule: _voxel_affine_plane_center, _voxel_affine_slice_normal (registration/_utils.py),  _dim_keyed_origin (bspline.py), _voxel_affine_dim_order (plotting/image.py)array/create.py) — established convention elsewhere is get_*.
- Shadowing name: plotting/image.py:85 defines _has_voxel_world_geometry (a wrapper) right next to importing the public has_voxel_world_geometry it wraps — easy to msites.
                                                                                                                                                                     Simplification opportunities (now that origin/spacing/rotation is one affin
                                                                                                                                                                     - .fusi.spacing/.fusi.origin/.fusi.direction (xarray/accessors.py:170-299) nd re-fold the voxel-to-world affine from scratch. Every SimpleITK interopsite that needs all three (registration/_utils.py, resampling.py, bspline.py — ~5 call sites) pays for 3 redundant affine collections instead of fetching once and dethree from the already-in-hand matrix via the affine-taking helpers that aln, get_affine_axis_scalings, get_affine_orientation_matrix).
- reindex_voxels (xarray/affine.py:186-253) decomposes the affine into direction+spacing, then immediately recombines via direction @ diag(spacing) — a normalize/renround-trip that's mathematically a no-op detour; could scale the existing lsteps directly.
- 9+ call sites hand-roll affine[:-1,:-1] = direction @ diag(spacing); affine[:-1,-1] = origin instead of reusing/extending the one existing helper (get_axis_aligned_utils/coordinates.py): xarray/create.py:409, registration/resampling.py:203 (has a documented rationale, keep that when consolidating),xarray/affine.py:235, atlas/_accessor.py:474, multipose/consolidate.py:483, io/nifti.py:1051 and :1101 (duplicated twice within the same function), io/scan.py:1591, datasets/_brainglobe.py:44. Recommend one shared helper taking an optional
- plotting/_utils.py:333-354 — oblique-geometry spacing fallback does np.diff(values) on what can now be an N-D world coordinate; np.diff defaults to axis=-1, silently computing spacing along the wrong voxel axis for oblique data. Should derive spacing ead.
- Correctly ruled out as non-simplifiable (external API constraint): SimpleITK's sitk.Image/BSplineTransform genuinely require separate SetSpacing/SetOrigin/SetDirection calls — no single-affine constructor exists.
