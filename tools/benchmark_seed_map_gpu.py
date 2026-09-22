"""Benchmark seed-map signal/statistics steps on CPU vs CuPy.

This mirrors docs/examples/04_connectivity/02_atlas_seed_map.py after registration/atlas
resampling: smooth, aCompCor, cosine/confound cleaning, seed extraction, Pearson maps.
Plotting and registration are intentionally outside the timed region.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
import xarray as xr
from cupyx.scipy import ndimage as cp_ndimage

import confusius as cf
from confusius._utils.filtering import make_cosine_drift_regressors


def sync() -> None:
    cp.cuda.Stream.null.synchronize()


def timed(name, func, repeat=3):
    rows = []
    for _ in range(repeat):
        cp.get_default_memory_pool().free_all_blocks()
        sync()
        t0 = time.perf_counter()
        out = func()
        sync()
        rows.append(time.perf_counter() - t0)
    return {
        "name": name,
        "times_s": rows,
        "median_s": float(np.median(rows)),
        "last": out,
    }


def gpu_smooth(data: xr.DataArray, fwhm: float) -> cp.ndarray:
    sigma_factor = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spacing = data.fusi.spacing
    sigmas = [
        0.0 if dim == "time" else fwhm * sigma_factor / float(spacing[dim])
        for dim in data.dims
    ]
    data_gpu = cp.asarray(np.asarray(data.data))
    return cp_ndimage.gaussian_filter(data_gpu, sigma=sigmas)


def gpu_standardize(x: cp.ndarray) -> cp.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    y = (x - mean) / std
    return cp.where(std < np.finfo(np.float64).eps, cp.nan, y)


def gpu_regress(
    x: cp.ndarray, confounds: cp.ndarray, *, standardize_confounds: bool
) -> cp.ndarray:
    c = confounds
    if standardize_confounds:
        c = c - c.mean(axis=0)
        scale = c.std(axis=0, ddof=1)
        c = c / cp.where(scale < np.finfo(np.float64).eps, 1, scale)
    else:
        scale = cp.max(cp.abs(c), axis=0)
        c = c / cp.where(scale == 0, 1, scale)

    q, r = cp.linalg.qr(c, mode="reduced")
    rank = int(cp.sum(cp.abs(cp.diag(r)) > np.finfo(np.float64).eps * 100.0).item())
    q = q[:, :rank]
    return x - q @ (q.T @ x)


def gpu_compcor(
    smoothed: cp.ndarray, white_matter_mask: np.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    flat = smoothed.reshape(smoothed.shape[0], -1)
    selected = cp.asarray(white_matter_mask.ravel().astype(bool))
    noise = flat[:, selected]

    # Match compute_compcor_confounds(..., detrend=False): remove zero-variance voxels,
    # z-score each voxel, SVD, first left-singular vector.
    std0 = noise.std(axis=0)
    noise = noise[:, std0 > np.finfo(np.float64).eps]
    noise = gpu_standardize(noise)
    u, s, _vt = cp.linalg.svd(noise, full_matrices=False)
    explained = (s[:1] ** 2) / cp.sum(s**2)
    return u[:, :1], explained


def gpu_clean_cosine(
    data_4d: cp.ndarray, confounds: cp.ndarray, dt: float, low_cutoff: float
) -> cp.ndarray:
    shape = data_4d.shape
    flat = data_4d.reshape(shape[0], -1)
    regs, _ = make_cosine_drift_regressors(shape[0], low_cutoff, dt)
    all_confounds = cp.concatenate(
        [confounds, cp.asarray(regs, dtype=flat.dtype)], axis=1
    )
    cleaned = gpu_regress(flat, all_confounds, standardize_confounds=False)
    return cleaned.reshape(shape)


def gpu_seed_signals(cleaned: cp.ndarray, seed_masks: np.ndarray) -> cp.ndarray:
    flat = cleaned.reshape(cleaned.shape[0], -1)
    signals = []
    for mask in seed_masks:
        selected = cp.asarray(mask.ravel() != 0)
        signals.append(flat[:, selected].mean(axis=1))
    return cp.stack(signals, axis=1)


def gpu_corr_maps(cleaned: cp.ndarray, seeds: cp.ndarray) -> cp.ndarray:
    t = cleaned.shape[0]
    flat = cleaned.reshape(t, -1)
    x = flat - flat.mean(axis=0)
    s = seeds - seeds.mean(axis=0)
    numerator = x.T @ s
    denom = (
        cp.sqrt(cp.sum(x * x, axis=0))[:, None]
        * cp.sqrt(cp.sum(s * s, axis=0))[None, :]
    )
    maps = cp.where(denom == 0, 0.0, numerator / denom).T
    return maps.reshape(seeds.shape[1], *cleaned.shape[1:])


def prepare_example():
    xr.set_options(display_expand_data=False)

    template = cf.datasets.fetch_template_pepe_mariani_2026().compute()
    bids_root = cf.datasets.fetch_nunez_elizalde_2022(
        subjects="CR022", sessions="20201007", tasks="spontaneous", acqs="slice02"
    )
    data_path = (
        Path(bids_root)
        / "sub-CR022"
        / "ses-20201007"
        / "fusi"
        / "sub-CR022_ses-20201007_task-spontaneous_acq-slice02_pwd.nii.gz"
    )
    data = cf.timing.resample_to_uniform_time(cf.load(data_path))
    moving = data.mean(dim="time").fusi.scale.db().compute()

    napari_affine = np.array(
        [
            [1.0, 0.0, 0.0, 5.594638656430411],
            [0.0, 1.0, 0.0, -2.50293925701927],
            [0.0, 0.0, 1.0, 5.6650243788545875],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    initialization = np.linalg.inv(napari_affine)
    target_z = napari_affine[0, 3] + moving.fusi.origin["z"]
    fixed = template.sel(z=slice(target_z - 1.0, target_z + 1.0))
    registered, affine, _diagnostics = cf.registration.register_volume(
        moving=moving,
        fixed=fixed,
        transform_type="affine",
        metric="correlation",
        convergence_window_size=100,
        number_of_iterations=500,
        learning_rate=1,
        initialization=initialization,
        show_progress=False,
    )
    del registered
    world_to_sform = template.attrs["affines"]["world_to_sform"]
    subject_to_atlas = world_to_sform @ np.linalg.inv(affine)
    atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_100um", check_latest=False)
    atlas_native = atlas.atlas.resample_like(moving, subject_to_atlas)
    seed_masks = atlas_native.atlas.get_masks(
        ["SSp-bfd", "RSP", "HIP", "VPM"], sides="right"
    )
    white_matter = atlas_native.atlas.get_masks("fiber tracts").isel(mask=0)
    return data, seed_masks, white_matter


def cpu_pipeline(data, seed_masks, white_matter):
    smoothed = cf.spatial.smooth_volume(data, fwhm=0.1).compute()
    acompcor = cf.signal.compute_compcor_confounds(
        smoothed, noise_mask=white_matter, n_components=1, variance_threshold=0.95
    )
    mapper = cf.connectivity.SeedBasedMaps(
        seed_masks=seed_masks,
        clean_kwargs={
            "low_cutoff": 0.01,
            "filter_method": "cosine",
            "confounds": acompcor,
        },
    )
    mapper.fit(smoothed)
    return mapper.maps_.compute()


def gpu_pipeline(data, seed_masks, white_matter):
    dt = float(np.median(np.diff(data.time.values.astype(float))))

    smoothed = gpu_smooth(data.astype(np.float32), 0.1)
    acompcor, explained = gpu_compcor(smoothed, np.asarray(white_matter.data))
    cleaned = gpu_clean_cosine(smoothed, acompcor, dt, 0.01)
    seeds = gpu_seed_signals(cleaned, np.asarray(seed_masks.data))
    maps = gpu_corr_maps(cleaned, seeds)
    # Return CPU copy so timing includes final materialization for plotting/API handoff.
    return {"maps": maps.get(), "explained_variance_ratio": explained.get().tolist()}


def main():
    data, seed_masks, white_matter = prepare_example()
    setup = {
        "shape": tuple(int(s) for s in data.shape),
        "dtype": str(data.dtype),
        "seed_masks": tuple(int(s) for s in seed_masks.shape),
        "white_matter_voxels": int(np.count_nonzero(white_matter.data)),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
    }
    print(json.dumps({"setup": setup}, indent=2), flush=True)

    cpu = timed(
        "cpu original signal+stats",
        lambda: cpu_pipeline(data, seed_masks, white_matter),
        repeat=3,
    )
    gpu = timed(
        "gpu all signal+stats incl copy-out",
        lambda: gpu_pipeline(data, seed_masks, white_matter),
        repeat=3,
    )

    cpu_maps = np.asarray(cpu.pop("last"))
    gpu_last = gpu.pop("last")
    gpu_maps = gpu_last["maps"]
    comparison = {
        "max_abs_diff": float(np.nanmax(np.abs(cpu_maps - gpu_maps))),
        "mean_abs_diff": float(np.nanmean(np.abs(cpu_maps - gpu_maps))),
        "cpu_map_range": [float(np.nanmin(cpu_maps)), float(np.nanmax(cpu_maps))],
        "gpu_map_range": [float(np.nanmin(gpu_maps)), float(np.nanmax(gpu_maps))],
        "gpu_compcor_explained_variance_ratio": gpu_last["explained_variance_ratio"],
    }
    print(json.dumps({"cpu": cpu, "gpu": gpu, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
