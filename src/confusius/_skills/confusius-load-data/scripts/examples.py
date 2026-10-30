"""Offline examples for importing spatial arrays with ConfUSIus.

Requires NumPy, SciPy, h5py, Dask, and ConfUSIus with Zarr support. All files are
synthetic and written to a temporary directory.
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

with TemporaryDirectory(prefix="confusius-load-data-") as tmp:
    root = Path(tmp)

    # MATLAB v7.3-style HDF5 layout, following the ConfUSIus MAT example.
    # This synthetic .h5 is not itself a MATLAB v7.3 MAT file.
    raw = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)  # time, i, j
    timestamps = np.array([0.0, 0.4, 0.81, 1.2, 1.6])
    with h5py.File(root / "matlab_style.h5", "w") as mat:
        mat.create_dataset("dop", data=raw)
        mat.create_dataset("timestamps", data=timestamps[None])
        mat.create_dataset("UF/Lambda", data=[[0.05]])
    with h5py.File(root / "matlab_style.h5", "r") as mat:
        doppler = mat["dop"][()]
        time = mat["timestamps"][0]
        dy = float(mat["UF/Lambda"][0, 0])

    power = cf.create_voxeldata(
        doppler,
        dims=("time", "i", "j"),
        time=time,
        spacing=(1.0, dy, 0.1),  # world z, y, x (mm)
        origin=(0.0, dy / 2, -0.15),
        units="mm",
        volume_acquisition_reference="start",
        volume_acquisition_duration=0.2,  # integration window, not time step
    )
    validate_voxeldata(power, require_time=True)
    assert power.dims == ("time", "k", "j", "i")
    assert_array_equal(power.data[:, 0], raw.transpose(0, 2, 1))
    assert_allclose(power.time, timestamps)
    point = power.isel(time=2, k=0, j=1, i=3)
    assert point.item() == raw[2, 3, 1]
    assert_allclose([point.z.item(), point.y.item(), point.x.item()], [0, 0.075, 0.15])

    # Raw (i, j, k, time) array and metadata in the same order.
    recording_raw = np.arange(4 * 3 * 2 * 5, dtype=np.float32).reshape(4, 3, 2, 5)
    np.save(root / "recording.npy", recording_raw, allow_pickle=False)
    savemat(root / "recording.mat", {"power": recording_raw})  # pre-v7.3 MAT
    spacing_ijk = (0.1, 0.05, 0.4)  # i -> x, j -> y, k -> z
    origin_ijk = (1.0, 2.0, 3.0)
    for loaded in (
        np.load(root / "recording.npy", allow_pickle=False),
        loadmat(root / "recording.mat")["power"],
    ):
        recording = cf.create_voxeldata(
            loaded,
            dims=("i", "j", "k", "time"),  # raw axes unchanged
            dt=0.4,
            spacing=spacing_ijk[::-1],  # convert to z, y, x
            origin=origin_ijk[::-1],
            units="mm",
            volume_acquisition_reference="start",
            volume_acquisition_duration=0.2,
        )
        validate_voxeldata(recording, require_time=True)
        assert_array_equal(recording.data, recording_raw.transpose(3, 2, 1, 0))
        point = recording.isel(time=2, k=1, j=2, i=3)
        assert point.item() == recording_raw[3, 2, 1, 2]
        assert_allclose(
            [point.z.item(), point.y.item(), point.x.item()], [3.4, 2.1, 1.3]
        )

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

# General external affine maps [i, j, k] -> [x, y, z].
external = np.array([[0, -0.2, 0, 4], [0.1, 0, 0, 5], [0, 0, 0.3, 6], [0, 0, 0, 1]])
P = np.eye(4)[[2, 1, 0, 3]]
A = P @ external @ P  # [k, j, i] -> [z, y, x]
labels_raw = (np.arange(60).reshape(3, 4, 5) % 7).astype(np.int16)
labels = cf.create_voxeldata(
    labels_raw, dims=("k", "j", "i"), voxel_to_world=A, name="atlas_labels"
)
validate_voxeldata(labels)
assert "time" not in labels.dims and labels.dtype == np.int16
assert_array_equal(labels.data, labels_raw)
point = labels.isel(k=1, j=2, i=3)
assert_allclose(
    [point.z.item(), point.y.item(), point.x.item()],
    (external @ [3, 2, 1, 1])[:3][::-1],
)

cropped = labels.isel(k=slice(1, 3), i=slice(1, 5, 2))
rebased = reindex_voxels(cropped)
validate_voxeldata(rebased)
assert_array_equal(rebased.i, [0, 1])
for axis in ("z", "y", "x"):
    assert_allclose(rebased.coords[axis], cropped.coords[axis])

F = np.eye(4)
F[2, 2], F[2, 3] = -1, labels_raw.shape[2] - 1
flipped = cf.create_voxeldata(
    labels_raw[..., ::-1], dims=("k", "j", "i"), voxel_to_world=A @ F
)
validate_voxeldata(flipped)
for axis in ("z", "y", "x"):
    assert_allclose(
        flipped.coords[axis], labels.coords[axis].isel(i=slice(None, None, -1))
    )

# Separate pose geometry and acquisition time.
pose_affines = np.stack([A, A.copy()])
pose_affines[1, 0, 3] += 2.0
multipose = cf.create_voxeldata(
    np.repeat(np.stack([labels_raw, labels_raw + 100])[None], 3, axis=0),
    dims=("time", "pose", "k", "j", "i"),
    voxel_to_world=pose_affines,
    pose=[10, 20],
    dt=1.0,
    t0=[0.0, 0.25],
    volume_acquisition_reference="start",
    volume_acquisition_duration=0.2,
)
validate_voxeldata(multipose, require_time=True)
assert_allclose(multipose.time, [[0, 0.25], [1, 1.25], [2, 2.25]])
assert_array_equal(
    multipose.isel(pose=1).set_xindex("time").sel(time=1.25).data,
    labels_raw + 100,
)

# Complex IQ stays complex and Dask-backed; extra dimensions carry labels.
iq_raw = (np.arange(24).reshape(2, 3, 4) * (1 + 2j)).astype(np.complex64)
iq = cf.create_voxeldata(
    da.from_array(iq_raw, chunks=iq_raw.shape),
    dims=("time", "j", "i"),
    dt=1 / 500,
    spacing=(1.0, 0.05, 0.1),
    origin=(0.0, 0.025, 0.0),
    volume_acquisition_reference="start",
    volume_acquisition_duration=0.001,
    attrs={"transmit_frequency": 15.625e6, "beamforming_sound_velocity": 1540.0},
)
validate_voxeldata(iq, require_dtype=np.complexfloating, require_velocity_attrs=True)
assert isinstance(iq.data, da.Array)
assert_array_equal(iq.compute().data[:, 0], iq_raw)
components = cf.create_voxeldata(
    np.stack([labels_raw, -labels_raw]),
    dims=("component", "k", "j", "i"),
    extra_coords={"component": ["a", "b"]},
    voxel_to_world=A,
)
validate_voxeldata(components)
assert_array_equal(components.sel(component="b").data, -labels_raw)
print("All custom-data import examples passed.")
