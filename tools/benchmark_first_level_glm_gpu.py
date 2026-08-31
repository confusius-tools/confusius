"""Benchmark first-level GLM example's signal/statistics steps on CPU vs CuPy.

Timed region starts after loading/registration/resampling and includes:
CompCor per run, Gaussian smoothing, AR(1) GLM fit across all runs, and active contrast.
"""

from __future__ import annotations

import gc
import json
import time
import warnings
from functools import partial
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import scipy.stats as sps
import xarray as xr
from cupyx.scipy import ndimage as cp_ndimage

import confusius as cf


def sync() -> None:
    cp.cuda.Stream.null.synchronize()


def timed(name, func, repeat=3):
    rows = []
    last = None
    for _ in range(repeat):
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        sync()
        t0 = time.perf_counter()
        last = func()
        sync()
        rows.append(time.perf_counter() - t0)
    return {"name": name, "times_s": rows, "median_s": float(np.median(rows)), "last": last}


def load_and_prepare_fusi(pwd_path: Path) -> xr.DataArray:
    """Load a Khallaf et al. 2026 recording and fix it to ConfUSIus's spatial convention.

    Mirrors `_load_and_prepare_fusi` in docs/examples/06_glm/01_first_level.py: the
    dataset stores (k, j, i) = (depth, elevation, lateral) with a non-metric sform
    targeting a custom world space, so we relabel voxel dims, apply the qform affine,
    cyclically permute z/y/x into ConfUSIus's convention, restore a proper (non
    reflective) direction matrix, and convert to millimeters.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="fUSI-BIDS validation warning", category=UserWarning)
        da = cf.load(pwd_path)

    relabeled_dims = tuple({"k": "j", "j": "k"}.get(dim, dim) for dim in da.dims)
    permute_kj = np.eye(4)
    permute_kj[[0, 1]] = permute_kj[[1, 0]]
    swapped_voxel_to_world = da.fusi.affine.voxel_to_world @ permute_kj

    da = cf.create_voxeldata(
        da.values,
        dims=relabeled_dims,
        time=da.time,
        voxel_to_world=swapped_voxel_to_world,
        attrs=da.attrs,
        name=str(da.name) if da.name is not None else None,
    )

    da.fusi.affine.apply(da.affines["world_to_qform"], inplace=True)

    permute_zyx_cycle = np.zeros((4, 4))
    permute_zyx_cycle[0, 2] = 1  # z <- old x (elevation)
    permute_zyx_cycle[1, 0] = 1  # y <- old z (depth)
    permute_zyx_cycle[2, 1] = 1  # x <- old y (lateral)
    permute_zyx_cycle[3, 3] = 1
    da.fusi.affine.apply(permute_zyx_cycle, inplace=True)

    flip_x = np.eye(4)
    flip_x[2, 2] = -1
    flip_x[2, 3] = da.x.max().item() + da.x.min().item()
    da.fusi.affine.apply(flip_x, inplace=True)

    flip_y = np.eye(4)
    flip_y[1, 1] = -1
    flip_y[1, 3] = da.y.max().item() + da.y.min().item()
    da.fusi.affine.apply(flip_y, inplace=True)

    flip_z = np.eye(4)
    flip_z[0, 0] = -1
    flip_z[0, 3] = da.z.max().item() + da.z.min().item()
    da.fusi.affine.apply(flip_z, inplace=True)

    m_to_mm = np.eye(4)
    m_to_mm[:3, :3] *= 1e3
    da.fusi.affine.apply(m_to_mm, inplace=True)
    da.fusi.affine.set_units("mm", inplace=True)

    return da


def prepare_example():
    template = cf.datasets.fetch_template_pepe_mariani_2026()
    bids_root = cf.datasets.fetch_khallaf_2026(
        datasets="rawdata",
        subjects="5622",
        sessions="IPM",
        reconstruction="resampled",
    )
    pattern = (
        Path(bids_root)
        / "sub-5622"
        / "ses-IPM"
        / "fusi"
        / "sub-5622_ses-IPM_task-olfactory_rec-resampled_run-*_space-5622run1_pwd.nii"
    )
    paths = sorted(Path(bids_root).rglob(str(pattern.relative_to(bids_root))))
    events = pd.read_csv(Path(bids_root) / "events.tsv", sep="\t")
    fusi_list = [load_and_prepare_fusi(p) for p in paths]
    average = xr.concat([f.mean("time") for f in fusi_list], dim="extra").mean("extra")

    atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_100um")
    resampled_template = cf.registration.resample_like(
        template,
        atlas.reference,
        np.linalg.inv(template.affines["world_to_sform"]),
    )
    napari_transform = np.array(
        [
            [0.7559553732760649, 0.31697755207337375, 0.0, 1.6997652603607039],
            [-0.27557848987798905, 0.8004409446062637, 0.0, -0.7527078253190659],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    _, transform, _ = cf.registration.register_volume(
        average,
        resampled_template,
        transform_type="affine",
        learning_rate="auto",
        initialization=np.linalg.inv(napari_transform),
    )
    average.affines["world_to_sform"] = np.linalg.inv(transform)
    resampled_runs = []
    for fusi in fusi_list:
        fusi.affines["world_to_sform"] = average.affines["world_to_sform"]
        resampled_runs.append(
            cf.registration.resample_like(fusi, atlas.annotation, np.linalg.inv(fusi.affines["world_to_sform"]))
        )
    wm = atlas.atlas.get_masks("fiber tracts")[0]
    return resampled_runs, events, wm


def gpu_smooth(run: xr.DataArray, fwhm: dict[str, float]) -> cp.ndarray:
    arr = cp.asarray(np.asarray(run.data, dtype=np.float64))
    factor = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spacing = run.fusi.spacing
    sigmas = []
    for dim in run.dims:
        if dim == "time":
            sigmas.append(0.0)
        else:
            sigmas.append(float(fwhm.get(dim, 0.0)) * factor / float(spacing[dim]))
    return cp_ndimage.gaussian_filter(arr, sigma=sigmas)


def gpu_standardize(x: cp.ndarray) -> cp.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    return cp.where(std < np.finfo(np.float64).eps, cp.nan, (x - mean) / std)


def gpu_compcor(run: xr.DataArray, wm_mask: xr.DataArray, n_components=3) -> cp.ndarray:
    flat = cp.asarray(np.asarray(run.data, dtype=np.float64)).reshape(run.sizes["time"], -1)
    selected = cp.asarray(np.asarray(wm_mask.data).ravel().astype(bool))
    noise = flat[:, selected]
    std0 = noise.std(axis=0)
    noise = noise[:, std0 > np.finfo(np.float64).eps]
    noise = gpu_standardize(noise)
    u, _s, _vt = cp.linalg.svd(noise, full_matrices=False)
    return u[:, :n_components]


def gpu_ar1_from_residuals(resid: cp.ndarray) -> cp.ndarray:
    x = resid - resid.mean(axis=0)
    ac0 = cp.mean(x * x, axis=0)
    ac1 = cp.mean(x[:-1] * x[1:], axis=0)
    return cp.where(ac0 == 0, 0.0, ac1 / ac0)[cp.newaxis, :]


def gpu_fit_ar1(Y: cp.ndarray, X_np: np.ndarray):
    X = cp.asarray(X_np, dtype=cp.float64)
    pinv = cp.linalg.pinv(X)
    beta0 = pinv @ Y
    resid0 = Y - X @ beta0
    rho = gpu_ar1_from_residuals(resid0)
    del beta0, resid0, pinv
    cp.get_default_memory_pool().free_all_blocks()

    T, K = X.shape
    V = Y.shape[1]
    wy = Y.copy()
    wy[1:] -= rho[0] * Y[:-1]
    L = cp.zeros((1, T, K), dtype=cp.float64)
    L[0, 1:, :] = X[:-1, :]
    A = X.T @ X
    B = cp.einsum("tk,itl->ikl", X, L)
    C = cp.einsum("itk,jtl->ijkl", L, L)
    Bsym = B + B.transpose(0, 2, 1)
    XtX = A[None] - cp.einsum("iv,ikl->vkl", rho, Bsym) + cp.einsum("iv,jv,ijkl->vkl", rho, rho, C)
    P = X.T @ wy
    Q = cp.einsum("itk,tv->ikv", L, wy)
    XtY = P.T - cp.einsum("iv,ikv->vk", rho, Q)
    del A, B, C, Bsym, P, Q
    cp.get_default_memory_pool().free_all_blocks()
    cov = cp.linalg.pinv(XtX)
    del XtX
    cp.get_default_memory_pool().free_all_blocks()
    beta = cp.einsum("vij,vj->vi", cov, XtY).T
    del XtY
    wresid = wy - X @ beta
    wresid += rho[0] * (L[0] @ beta)
    df = T - int(np.linalg.matrix_rank(X_np, np.abs(X_np).sum() * np.finfo(np.float64).eps))
    dispersion = cp.sum(wresid * wresid, axis=0) / df
    return {"theta": beta, "cov": cov, "dispersion": dispersion, "df": df}


def gpu_fit_ar1_contrast_batched(Y: cp.ndarray, X_np: np.ndarray, c_np: np.ndarray, batch_size: int = 100_000):
    """Exact per-voxel AR(1) fit, but only one voxel batch lives on GPU at a time."""
    X = cp.asarray(X_np, dtype=cp.float64)
    c = cp.asarray(c_np, dtype=cp.float64)
    T, K = X.shape
    V = Y.shape[1]
    rank = int(np.linalg.matrix_rank(X_np, np.abs(X_np).sum() * np.finfo(np.float64).eps))
    df = T - rank

    L = cp.zeros((1, T, K), dtype=cp.float64)
    L[0, 1:, :] = X[:-1, :]
    A = X.T @ X
    B = cp.einsum("tk,itl->ikl", X, L)
    C = cp.einsum("itk,jtl->ijkl", L, L)
    Bsym = B + B.transpose(0, 2, 1)
    pinv = cp.linalg.pinv(X)

    effects = cp.empty(V, dtype=cp.float64)
    variances = cp.empty(V, dtype=cp.float64)

    for start in range(0, V, batch_size):
        stop = min(start + batch_size, V)
        Yb = Y[:, start:stop]

        beta0 = pinv @ Yb
        resid0 = Yb - X @ beta0
        rho = gpu_ar1_from_residuals(resid0)
        del beta0, resid0

        wy = Yb.copy()
        wy[1:] -= rho[0] * Yb[:-1]
        XtX = (
            A[None]
            - cp.einsum("iv,ikl->vkl", rho, Bsym)
            + cp.einsum("iv,jv,ijkl->vkl", rho, rho, C)
        )
        P = X.T @ wy
        Q = cp.einsum("itk,tv->ikv", L, wy)
        XtY = P.T - cp.einsum("iv,ikv->vk", rho, Q)
        cov = cp.linalg.pinv(XtX)
        beta = cp.einsum("vij,vj->vi", cov, XtY).T

        wresid = wy - X @ beta
        wresid += rho[0] * (L[0] @ beta)
        dispersion = cp.sum(wresid * wresid, axis=0) / df
        effects[start:stop] = c @ beta
        variances[start:stop] = cp.einsum("k,vkl,l->v", c, cov, c) * dispersion

        del Yb, rho, wy, XtX, P, Q, XtY, cov, beta, wresid, dispersion
        cp.get_default_memory_pool().free_all_blocks()

    return effects, variances, df


def gpu_contrast(results, design_columns, contrast_name="active"):
    idx = list(design_columns).index(contrast_name)
    combined_effect = None
    combined_var = None
    dof = 0
    c_cache = {}
    for res, cols in zip(results, design_columns, strict=True):
        K = len(cols)
        if K not in c_cache:
            c = cp.zeros(K, dtype=cp.float64)
            c[list(cols).index(contrast_name)] = 1.0
            c_cache[K] = c
        c = c_cache[K]
        effect = c @ res["theta"]
        var = cp.einsum("k,vkl,l->v", c, res["cov"], c) * res["dispersion"]
        combined_effect = effect if combined_effect is None else combined_effect + effect
        combined_var = var if combined_var is None else combined_var + var
        dof += res["df"]
    n = len(results)
    effect = combined_effect / n
    variance = combined_var / (n * n)
    t = effect / cp.sqrt(cp.maximum(variance, 1e-50))
    # CuPy lacks Student-t survival/CDF. Copy final t map and compute exact z on CPU.
    t_cpu = t.get()
    p = sps.t.sf(t_cpu, min(dof, 1e10))
    one_minus = sps.t.cdf(t_cpu, min(dof, 1e10))
    z_sf = sps.norm.isf(np.clip(p, 1e-300, 1 - 1e-16))
    z_cdf = sps.norm.ppf(np.clip(one_minus, 1e-300, 1 - 1e-16))
    z = np.where(z_sf < 0, z_cdf, z_sf)
    return z


def cpu_pipeline(runs, events, wm, hrf):
    confounds = [cf.signal.compute_compcor_confounds(run, noise_mask=wm, n_components=3) for run in runs]
    glm = cf.glm.FirstLevelModel(
        smoothing_fwhm=0.3,
        hrf_model=hrf,
        drift_model="cosine",
        low_cutoff=0.01,
        noise_model="ar1",
    )
    glm.fit(runs, events=events, confounds=confounds)
    return glm.compute_contrast("active").data


def gpu_pipeline(runs, events, wm, hrf):
    confounds_gpu = [gpu_compcor(run, wm, 3) for run in runs]
    confounds_cpu = [c.get() for c in confounds_gpu]
    del confounds_gpu
    cp.get_default_memory_pool().free_all_blocks()
    design_matrices = [
        cf.glm.make_first_level_design_matrix(
            run.coords["time"].values,
            events=events,
            hrf_model=hrf,
            drift_model="cosine",
            low_cutoff=0.01,
            confounds=conf,
        )
        for run, conf in zip(runs, confounds_cpu, strict=True)
    ]
    combined_effect = None
    combined_var = None
    dof = 0
    for run, dm in zip(runs, design_matrices, strict=True):
        smoothed = gpu_smooth(run, {"k": 0.3, "j": 0.3, "i": 0.3})
        Y = smoothed.reshape(run.sizes["time"], -1)
        c = np.zeros(len(dm.columns), dtype=np.float64)
        c[list(dm.columns).index("active")] = 1.0
        effect, variance, run_df = gpu_fit_ar1_contrast_batched(Y, dm.to_numpy(), c)
        combined_effect = effect if combined_effect is None else combined_effect + effect
        combined_var = variance if combined_var is None else combined_var + variance
        dof += run_df
        del smoothed, Y, effect, variance
        cp.get_default_memory_pool().free_all_blocks()

    n = len(runs)
    t = (combined_effect / n) / cp.sqrt(cp.maximum(combined_var / (n * n), 1e-50))
    t_cpu = t.get()
    p = sps.t.sf(t_cpu, min(dof, 1e10))
    one_minus = sps.t.cdf(t_cpu, min(dof, 1e10))
    z_sf = sps.norm.isf(np.clip(p, 1e-300, 1 - 1e-16))
    z_cdf = sps.norm.ppf(np.clip(one_minus, 1e-300, 1 - 1e-16))
    z_flat = np.where(z_sf < 0, z_cdf, z_sf)
    return z_flat.reshape(tuple(runs[0].sizes[d] for d in runs[0].dims if d != "time"))


def main():
    modified_hrf = partial(cf.glm.claron2021_hrf, beta=6.7)
    runs, events, wm = prepare_example()
    gc.collect()
    setup = {
        "n_runs": len(runs),
        "run_shapes": [tuple(int(s) for s in run.shape) for run in runs],
        "wm_voxels": int(np.count_nonzero(wm.data)),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
    }
    print(json.dumps({"setup": setup}, indent=2), flush=True)
    cpu = timed("cpu original compcor+smooth+ar1 glm+contrast", lambda: cpu_pipeline(runs, events, wm, modified_hrf), repeat=3)
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    gpu = timed("gpu compcor+smooth+ar1 glm+contrast", lambda: gpu_pipeline(runs, events, wm, modified_hrf), repeat=3)
    cpu_z = np.asarray(cpu.pop("last"))
    gpu_z = np.asarray(gpu.pop("last"))
    comparison = {
        "max_abs_diff": float(np.nanmax(np.abs(cpu_z - gpu_z))),
        "mean_abs_diff": float(np.nanmean(np.abs(cpu_z - gpu_z))),
        "cpu_range": [float(np.nanmin(cpu_z)), float(np.nanmax(cpu_z))],
        "gpu_range": [float(np.nanmin(gpu_z)), float(np.nanmax(gpu_z))],
    }
    print(json.dumps({"cpu": cpu, "gpu": gpu, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
