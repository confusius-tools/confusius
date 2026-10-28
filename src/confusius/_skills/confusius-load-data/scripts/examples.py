"""Run offline, assert-based custom-data import examples with installed ConfUSIus.

Requires NumPy, SciPy, h5py, Dask, and ConfUSIus with Zarr support. Files are
synthetic and confined to a temporary directory; no source data is overwritten.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import dask.array as da
import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat, savemat

import confusius as cf
from confusius.validation import validate_voxeldata
from confusius.xarray import reindex_voxels

# A 2D export in (lateral, depth, time) order; elevation spacing is a placeholder.
raw = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
power = cf.create_voxeldata(
    raw,
    dims=("i", "j", "time"),
    dt=0.4,
    spacing=(1.0, 0.05, 0.1),
    origin=(0.0, 0.025, -0.15),
    units="mm",
    volume_acquisition_reference="start",
    volume_acquisition_duration=0.2,
    name="power_doppler",
)
validate_voxeldata(power, require_time=True)
assert power.dims == ("time", "k", "j", "i")
assert_array_equal(power.data[:, 0], raw.transpose(2, 1, 0))
assert_allclose(power.time, np.arange(5) * 0.4)
point = power.isel(time=2, k=0, j=1, i=3)
assert_allclose([point.z.item(), point.y.item(), point.x.item()], [0.0, 0.075, 0.15])
assert point.item() == raw[3, 1, 2]
assert power.time.attrs["volume_acquisition_duration"] == 0.2

with TemporaryDirectory(prefix="confusius-load-data-") as temporary:
    root = Path(temporary)
    np.save(root / "power.npy", raw, allow_pickle=False)
    savemat(root / "power.mat", {"doppler": raw})
    with h5py.File(root / "power.h5", "w") as handle:
        handle.create_dataset("doppler", data=raw)
    with h5py.File(root / "power.h5", "r") as handle:
        hdf5_raw = handle["doppler"][()]
    # Real exports need verified axis metadata; these fixtures deliberately agree.
    for loaded in (
        np.load(root / "power.npy", allow_pickle=False),
        loadmat(root / "power.mat")["doppler"],
        hdf5_raw,
    ):
        imported = cf.create_voxeldata(
            loaded,
            dims=("i", "j", "time"),
            time=[0.0, 0.4, 0.81, 1.2, 1.6],
            spacing=(1.0, 0.05, 0.1),
            origin=(0.0, 0.025, -0.15),
            volume_acquisition_duration=0.2,
        )
        validate_voxeldata(imported, require_time=True)
        assert_array_equal(imported.data, power.data)
        assert_allclose(imported.time, [0.0, 0.4, 0.81, 1.2, 1.6])

    cf.save(power, root / "power.zarr")
    restored = cf.load(root / "power.zarr")
    validate_voxeldata(restored)
    assert_array_equal(restored.data, power.data)
    assert_allclose(restored.time, power.time)
    assert restored.time.attrs == power.time.attrs
    assert restored.fusi.affine.units == "mm"
    assert_allclose(
        restored.fusi.affine.voxel_to_world, power.fusi.affine.voxel_to_world
    )

# This external affine maps ijk -> xyz, including a rotation and translation.
external = np.array(
    [
        [0.0, -0.2, 0.0, 4.0],
        [0.1, 0.0, 0.0, 5.0],
        [0.0, 0.0, 0.3, 6.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
permutation = np.eye(4)[[2, 1, 0, 3]]
affine = permutation @ external @ permutation
volume_raw = np.arange(3 * 4 * 5).reshape(3, 4, 5)
volume = cf.create_voxeldata(volume_raw, dims=("k", "j", "i"), voxel_to_world=affine)
validate_voxeldata(volume)
point = volume.isel(k=1, j=2, i=3)
expected_xyz = external @ [3, 2, 1, 1]
assert_allclose(
    [point.z.item(), point.y.item(), point.x.item()], expected_xyz[:3][::-1]
)
assert point.item() == volume_raw[1, 2, 3]

# An oblique grid exercises world coordinates dependent on multiple voxel axes.
angle = np.pi / 6
rotation = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
    ]
)
oblique = cf.create_voxeldata(
    volume_raw,
    dims=("k", "j", "i"),
    spacing=(0.3, 0.2, 0.1),
    origin=(6.0, 5.0, 4.0),
    direction=rotation,
)
validate_voxeldata(oblique)
point = oblique.isel(k=1, j=2, i=3)
assert_allclose(
    [point.z.item(), point.y.item(), point.x.item()],
    np.array([6.0, 5.0, 4.0]) + rotation @ [0.3, 0.4, 0.3],
)

cropped = volume.isel(k=slice(1, 3), i=slice(1, 5, 2))
rebased = reindex_voxels(cropped)
validate_voxeldata(rebased)
assert_array_equal(rebased.data, volume_raw[1:3, :, 1:5:2])
assert_array_equal(rebased.i, [0, 1])
for axis in ("z", "y", "x"):
    assert_allclose(rebased.coords[axis], cropped.coords[axis])

flip = np.eye(4)
flip[2, 2] = -1
flip[2, 3] = volume_raw.shape[2] - 1
flipped = cf.create_voxeldata(
    volume_raw[:, :, ::-1],
    dims=("k", "j", "i"),
    voxel_to_world=affine @ flip,
)
validate_voxeldata(flipped)
for axis in ("z", "y", "x"):
    assert_allclose(
        flipped.coords[axis], volume.coords[axis].isel(i=slice(None, None, -1))
    )
assert_array_equal(flipped.data, volume_raw[:, :, ::-1])

# Each pose has its own position and timestamp; dt repeats the same pose.
pose_affines = np.stack([affine, affine.copy()])
pose_affines[1, 0, 3] += 2.0
multipose = cf.create_voxeldata(
    np.stack([volume_raw, volume_raw + 100], axis=0)[None].repeat(3, axis=0),
    dims=("time", "pose", "k", "j", "i"),
    voxel_to_world=pose_affines,
    pose=[10, 20],
    dt=1.0,
    t0=[0.0, 0.25],
    volume_acquisition_duration=0.2,
)
validate_voxeldata(multipose, require_time=True)
assert multipose.time.dims == ("time", "pose")
assert_allclose(multipose.time, [[0.0, 0.25], [1.0, 1.25], [2.0, 2.25]])
point = multipose.isel(time=1, pose=1, k=1, j=2, i=3)
assert point.item() == volume_raw[1, 2, 3] + 100
assert_allclose(point.z, expected_xyz[2] + 2.0)
one_pose = multipose.isel(pose=1).set_xindex("time")
assert_array_equal(one_pose.sel(time=1.25).data, volume_raw + 100)

# IQ uses the same spatial model; preserve complex data and keep Dask lazy.
iq_raw = (np.arange(24).reshape(2, 3, 4) * (1 + 2j)).astype(np.complex64)
iq = cf.create_voxeldata(
    da.from_array(iq_raw, chunks=(2, 3, 4)),
    dims=("time", "j", "i"),
    dt=1 / 500,
    spacing=(1.0, 0.05, 0.1),
    origin=(0.0, 0.025, 0.0),
    attrs={"transmit_frequency": 15.625e6, "beamforming_sound_velocity": 1540.0},
)
validate_voxeldata(iq, require_velocity_attrs=True, require_dtype=np.complexfloating)
assert isinstance(iq.data, da.Array)
assert_array_equal(iq.compute().data[:, 0], iq_raw)

components = cf.create_voxeldata(
    np.stack([volume_raw, -volume_raw]),
    dims=("component", "k", "j", "i"),
    extra_coords={"component": ["a", "b"]},
    voxel_to_world=affine,
)
validate_voxeldata(components)
assert_array_equal(components.sel(component="b").data, -volume_raw)
print("All custom-data loading examples passed.")
