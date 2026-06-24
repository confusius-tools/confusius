# %%
"""Minimal iPython-friendly voxel-affine 3D loading demo."""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

import confusius as cf
from confusius.plotting import plot_volume
from confusius.validation import validate_fusi_dataarray

# %%
# Build a synthetic 3D volume in NIfTI (x, y, z) array order.
rng = np.random.default_rng(0)
nx, ny, nz = 48, 40, 24
x = np.linspace(-1.0, 1.0, nx)
y = np.linspace(-1.0, 1.0, ny)
z = np.linspace(-1.0, 1.0, nz)
xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
blob1 = np.exp(-((xx + 0.25) ** 2 + (yy - 0.15) ** 2 + (zz + 0.2) ** 2) / 0.12)
blob2 = 0.7 * np.exp(-((xx - 0.2) ** 2 + (yy + 0.25) ** 2 + (zz - 0.1) ** 2) / 0.05)
blob3 = 0.4 * np.exp(-((xx + 0.1) ** 2 + (yy + 0.05) ** 2 + (zz - 0.35) ** 2) / 0.03)
data = (blob1 + blob2 + blob3 + 0.03 * rng.standard_normal((nx, ny, nz))).astype(
    np.float32
)

# %%
# Oblique qform in mm. Disable sform so qform is the primary xform on load.
rz = np.deg2rad(30.0)
rx = np.deg2rad(-20.0)
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
spacing = np.diag([0.12, 0.08, 0.25])
affine = np.eye(4)
affine[:3, :3] = rotation @ spacing
affine[:3, 3] = [10.0, 20.0, 30.0]

# %%
# Save to a temporary NIfTI and reload with the voxel-affine coordinate model.
tmpdir = tempfile.TemporaryDirectory()
path = Path(tmpdir.name) / "rotated_qform_demo_3d.nii.gz"
image = nib.Nifti1Image(data, affine)
image.set_qform(affine, code=1)
image.set_sform(np.eye(4), code=0)
image.header.set_xyzt_units(xyz="mm")
image.to_filename(path)

da = cf.load(path, coordinate_model="voxel_affine")
da

# %%
print("dims:", da.dims)
print("coord names:", list(da.coords))
print("voxel_to_physical:\n", np.asarray(da.attrs["voxel_to_physical"]))
print("affine keys:", list(da.attrs.get("affines", {})))
print("k range:", float(da.k.min()), "->", float(da.k.max()))
print("j range:", float(da.j.min()), "->", float(da.j.max()))
print("i range:", float(da.i.min()), "->", float(da.i.max()))
print("origin:", da.fusi.origin)
print("spacing:", da.fusi.spacing)
print("direction:\n", da.fusi.direction)

validate_fusi_dataarray(da, require_regular_spacing=True, regular_spacing_dims="space")
print("validate_fusi_dataarray(...): ok")

# %%
# Slice along k: constant k, plot the native (j, i) plane in its projected in-plane geometry.
plot_volume(
    da,
    slice_mode="k",
    show_colorbar=False,
).show()

# `slice_mode="z"` is intentionally not supported yet for voxel-affine data:
# slicing is native voxel-plane only for now.

# %%
# Slice along j: constant j, plot the native (k, i) plane.
plot_volume(
    da,
    slice_mode="j",
    show_colorbar=False,
).show()

# %%
# Slice along i: constant i, plot the native (k, j) plane.
plot_volume(
    da,
    slice_mode="i",
    show_colorbar=False,
).show()
