# %%
"""Minimal iPython-friendly CTI slicing demo on real Nunez-Elizalde data."""

from pathlib import Path

import numpy as np

import confusius as cf
from confusius._utils.geometry import add_physical_coords_from_voxel_affine
from confusius.plotting import plot_volume
from confusius.validation import validate_fusi_dataarray


# %%
def _make_center_preserving_oblique_copy(native: cf.DataArray) -> cf.DataArray:
    """Return `native` with the same voxel grid but an oblique voxel-to-physical affine."""
    rz = np.deg2rad(18.0)
    rx = np.deg2rad(-12.0)
    rot_z = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ]
    )
    rotation = rot_z @ rot_x

    native_affine = np.asarray(native.attrs["voxel_to_physical"], dtype=np.float64)
    linear = native_affine[:3, :3]
    translation = native_affine[:3, 3]

    voxel_center = 0.5 * np.array(
        [native.sizes["k"] - 1, native.sizes["j"] - 1, native.sizes["i"] - 1],
        dtype=np.float64,
    )
    physical_center = linear @ voxel_center + translation

    oblique_affine = np.eye(4, dtype=np.float64)
    oblique_affine[:3, :3] = rotation @ linear
    oblique_affine[:3, 3] = physical_center - oblique_affine[:3, :3] @ voxel_center

    voxel_grid = native.drop_vars(
        [name for name in ("z", "y", "x") if name in native.coords]
    )
    oblique = add_physical_coords_from_voxel_affine(
        voxel_grid,
        oblique_affine,
        voxel_dims=("k", "j", "i"),
        physical_coord_names=("z", "y", "x"),
        physical_coord_attrs={
            "z": dict(native.coords["z"].attrs),
            "y": dict(native.coords["y"].attrs),
            "x": dict(native.coords["x"].attrs),
        },
    )
    oblique.name = native.name
    oblique.attrs.update(native.attrs)
    oblique.attrs["voxel_to_physical"] = oblique_affine
    return oblique


# %%
# Fetch one recording from the Nunez-Elizalde 2022 dataset and load a 3D volume.
bids_root = cf.datasets.fetch_nunez_elizalde_2022(
    subjects="CR022",
    sessions="20201011",
    tasks="spontaneous",
    acqs="slice03",
)

angio_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201011"
    / "angio"
    / "sub-CR022_ses-20201011_pwd.nii.gz"
)

native = cf.load(angio_path).compute()
oblique = _make_center_preserving_oblique_copy(native)

native

# %%
print("native dims:", native.dims)
print("native coord names:", list(native.coords))
print("native voxel_to_physical:\n", np.asarray(native.attrs["voxel_to_physical"]))
print("oblique voxel_to_physical:\n", np.asarray(oblique.attrs["voxel_to_physical"]))
print("native origin:", native.fusi.origin)
print("oblique origin:", oblique.fusi.origin)
print("native spacing:", native.fusi.spacing)
print("oblique spacing:", oblique.fusi.spacing)
print("native direction:\n", native.fusi.direction)
print("oblique direction:\n", oblique.fusi.direction)

validate_fusi_dataarray(
    native, require_regular_spacing=True, regular_spacing_dims="space"
)
validate_fusi_dataarray(
    oblique, require_regular_spacing=True, regular_spacing_dims="space"
)
print("validate_fusi_dataarray(...): ok")

# %%
# Native voxel-plane comparison.
#
# Here `slice_mode="k"` means "take constant-k acquisition planes" for each
# volume. The axes are still labeled in mm because the plotted mesh is drawn in
# projected physical coordinates, even though the slice selection itself is in
# voxel index space.
k_values = np.asarray(native.coords["k"].values, dtype=float)
k_margin = max(1, int(round(0.1 * (k_values.size - 1))))
k_slice_coords = np.linspace(k_values[k_margin], k_values[-k_margin - 1], 10).tolist()
plotter = plot_volume(
    native,
    slice_mode="k",
    slice_coords=k_slice_coords,
    nrows=2,
    ncols=5,
    cmap="gray",
    show_colorbar=False,
)
plotter.add_volume(
    oblique,
    slice_coords=k_slice_coords,
    cmap="autumn",
    alpha=0.45,
    show_colorbar=False,
)
plotter.show()

# %%
# Physical-plane comparison.
#
# Here `slice_mode="z"` means "show the same physical z-planes" in both volumes.
# For the oblique CTI volume this now triggers resampling to an axis-aligned
# physical grid before plotting.
z_min = max(
    float(np.asarray(native.coords["z"].values, dtype=float).min()),
    float(np.asarray(oblique.coords["z"].values, dtype=float).min()),
)
z_max = min(
    float(np.asarray(native.coords["z"].values, dtype=float).max()),
    float(np.asarray(oblique.coords["z"].values, dtype=float).max()),
)
z_slice_coords = np.linspace(z_min, z_max, 10).tolist()
cf.plotting.plot_composite(
    native,
    oblique,
    slice_mode="z",
    slice_coords=z_slice_coords,
    nrows=2,
    ncols=5,
).show()
