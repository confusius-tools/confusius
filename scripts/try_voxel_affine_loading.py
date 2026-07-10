# %%
"""iPython-friendly voxel-to-physical geometry demo on real data."""

from pathlib import Path

import numpy as np
import xarray as xr

import confusius as cf
from confusius._utils.geometry import (
    add_physical_coords_from_voxel_affine,
    has_axis_aligned_voxel_affine_geometry,
)
from confusius.plotting import plot_volume
from confusius.plotting._utils import resample_voxel_affine_to_physical_grid
from confusius.validation import validate_fusi_dataarray


# %%
def _make_center_preserving_transformed_copy(
    native: cf.DataArray,
    transform: np.ndarray,
    *,
    name: str,
) -> cf.DataArray:
    """Return `native` with the same voxel grid but a transformed affine."""
    native_affine = np.asarray(native.attrs["voxel_to_physical"], dtype=np.float64)
    linear = native_affine[:3, :3]
    translation = native_affine[:3, 3]

    voxel_center = 0.5 * np.array(
        [native.sizes["k"] - 1, native.sizes["j"] - 1, native.sizes["i"] - 1],
        dtype=np.float64,
    )
    physical_center = linear @ voxel_center + translation

    transformed_affine = np.eye(4, dtype=np.float64)
    transformed_affine[:3, :3] = transform @ linear
    transformed_affine[:3, 3] = (
        physical_center - transformed_affine[:3, :3] @ voxel_center
    )

    voxel_grid = native.drop_vars(
        [coord_name for coord_name in ("z", "y", "x") if coord_name in native.coords]
    )
    transformed = add_physical_coords_from_voxel_affine(
        voxel_grid,
        transformed_affine,
        voxel_dims=("k", "j", "i"),
        physical_coord_names=("z", "y", "x"),
        physical_coord_attrs={
            "z": dict(native.coords["z"].attrs),
            "y": dict(native.coords["y"].attrs),
            "x": dict(native.coords["x"].attrs),
        },
    )
    transformed.name = name
    transformed.attrs.update(native.attrs)
    transformed.attrs["voxel_to_physical"] = transformed_affine
    return transformed


def _make_center_preserving_oblique_copy(native: cf.DataArray) -> cf.DataArray:
    """Return `native` with an oblique voxel-to-physical affine."""
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
    return _make_center_preserving_transformed_copy(
        native,
        rot_z @ rot_x,
        name="oblique",
    )


def _make_center_preserving_sheared_copy(native: cf.DataArray) -> cf.DataArray:
    """Return `native` with a sheared voxel-to-physical affine."""
    shear = np.array(
        [
            [1.0, 0.18, 0.0],
            [0.0, 1.0, -0.12],
            [0.0, 0.0, 1.0],
        ]
    )
    return _make_center_preserving_transformed_copy(
        native,
        shear,
        name="sheared",
    )


def _as_axis_aligned_physical_grid(data: cf.DataArray) -> cf.DataArray:
    """Expose axis-aligned CTI as a plain z/y/x grid for physical-slice demos."""
    result = xr.DataArray(
        data.data,
        dims=("z", "y", "x"),
        coords={
            "z": data.coords["z"].values,
            "y": data.coords["y"].values,
            "x": data.coords["x"].values,
        },
        name=data.name,
        attrs=data.attrs.copy(),
    )
    result.coords["z"].attrs = dict(data.coords["z"].attrs)
    result.coords["y"].attrs = dict(data.coords["y"].attrs)
    result.coords["x"].attrs = dict(data.coords["x"].attrs)
    result.attrs.pop("voxel_to_physical", None)
    return result


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

native = cf.load(angio_path).compute().rename("axis-aligned")
oblique = _make_center_preserving_oblique_copy(native)
sheared = _make_center_preserving_sheared_copy(native)

native

# %%
for label, volume in {
    "native": native,
    "oblique": oblique,
    "sheared": sheared,
}.items():
    print(f"\n--- {label} ---")
    print("dims:", volume.dims)
    print("coord names:", list(volume.coords))
    print("axis-aligned voxel-to-physical geometry:", has_axis_aligned_voxel_affine_geometry(volume))
    print("voxel_to_physical:\n", np.asarray(volume.attrs["voxel_to_physical"]))
    print("origin:", volume.fusi.origin)
    print("spacing:", volume.fusi.spacing)
    print("direction:\n", volume.fusi.direction)
    validate_fusi_dataarray(
        volume, require_regular_spacing=True, regular_spacing_dims="space"
    )

print("\nvalidate_fusi_dataarray(...): ok for native, oblique, sheared")

native_display = resample_voxel_affine_to_physical_grid(native)
oblique_display = resample_voxel_affine_to_physical_grid(oblique)
sheared_display = resample_voxel_affine_to_physical_grid(sheared)
native_physical_grid = _as_axis_aligned_physical_grid(native)
print("\nDisplay-grid dims after napari/matplotlib fallback:")
print("native:", native_display.dims)
print("oblique:", oblique_display.dims)
print("sheared:", sheared_display.dims)
print("native physical-view dims:", native_physical_grid.dims)

# %%
# Native voxel-plane comparison.
#
# Here `slice_mode="k"` means "take constant-k acquisition planes" for each
# volume. All three CTI variants share the same voxel grid, so this shows the
# axis-aligned, oblique, and sheared cases on matching acquisition planes.
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
plotter.add_volume(
    sheared,
    slice_coords=k_slice_coords,
    cmap="winter",
    alpha=0.3,
    show_colorbar=False,
)
plotter.show()


# %%
# Physical-plane comparisons.
#
# Here `slice_mode in {"z", "y", "x"}` means "show the same physical planes"
# in all volumes. The axis-aligned volume is exposed on a plain `z/y/x` grid,
# while oblique and sheared geometry are resampled onto that display grid.
def _shared_range(dim: str) -> tuple[float, float]:
    return (
        max(
            float(np.asarray(native.coords[dim].values, dtype=float).min()),
            float(np.asarray(oblique.coords[dim].values, dtype=float).min()),
            float(np.asarray(sheared.coords[dim].values, dtype=float).min()),
        ),
        min(
            float(np.asarray(native.coords[dim].values, dtype=float).max()),
            float(np.asarray(oblique.coords[dim].values, dtype=float).max()),
            float(np.asarray(sheared.coords[dim].values, dtype=float).max()),
        ),
    )


for slice_mode in ("z", "y", "x"):
    lower, upper = _shared_range(slice_mode)
    slice_coords = np.linspace(lower, upper, 10).tolist()
    plotter = plot_volume(
        native_physical_grid,
        slice_mode=slice_mode,
        slice_coords=slice_coords,
        nrows=2,
        ncols=5,
        cmap="gray",
        show_colorbar=False,
    )
    plotter.add_volume(
        oblique,
        slice_coords=slice_coords,
        cmap="autumn",
        alpha=0.45,
        show_colorbar=False,
    )
    plotter.add_volume(
        sheared,
        slice_coords=slice_coords,
        cmap="winter",
        alpha=0.3,
        show_colorbar=False,
    )
    plotter.show()

# %%
# Napari comparison.
#
# Napari always shows physical `z/y/x` axes. Axis-aligned data is promoted to a
# plain physical grid without interpolation, while oblique and sheared geometry
# are resampled to a physical display grid.
viewer, native_layer = cf.plotting.plot_napari(
    native,
    show_colorbar=False,
    show_scale_bar=True,
)
_, oblique_layer = cf.plotting.plot_napari(
    oblique,
    viewer=viewer,
    show_colorbar=False,
    show_scale_bar=True,
    colormap="magenta",
    opacity=0.5,
)
_, sheared_layer = cf.plotting.plot_napari(
    sheared,
    viewer=viewer,
    show_colorbar=False,
    show_scale_bar=True,
    colormap="cyan",
    opacity=0.35,
)
print("\nnapari physical display:")
for label, layer in {
    "native": native_layer,
    "oblique": oblique_layer,
    "sheared": sheared_layer,
}.items():
    print(f"{label} displayed dims:", layer.metadata["xarray"].dims)
    print(f"{label} source dims:", layer.metadata["source_xarray"].dims)

viewer
