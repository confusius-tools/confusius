"""End-to-end demo of the ConfUSIus GLM module.

This script demonstrates three workflows:

1. **Low-level API** – manually building design matrices and fitting OLS/AR
   models using the internal building-block classes (``OLSModel``,
   ``ARModel``).  These classes are not part of the public API but are
   exposed here to illustrate the two-pass fitting procedure.
2. **First-level API** – using ``FirstLevelModel`` on real fUSI data to
   fit a voxel-wise GLM and visualise the resulting statistical maps.
3. **Second-level API** – using ``SecondLevelModel`` for group-level
   inference across multiple synthetic subjects.  Demonstrates both the
   one-sample group mean and a two-group comparison with confounds.

The real data used in part 2 is a single-slice power-Doppler recording
from an awake mouse during visual stimulation.  The stimulation protocol
is a 30 s baseline followed by alternating 5 s ON / 5 s OFF blocks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import confusius as cf
from confusius.glm import (
    Contrast,
    FirstLevelModel,
    make_first_level_design_matrix,
)
from confusius.glm._models import ARModel, OLSModel
from confusius.glm._utils import expression_to_contrast_vector
from confusius.plotting import plot_volume

DATA_PATH = Path("~/Downloads/visual-stim-bv/Visual_stim/Raw_visual/1702R5/Func.nii")


# ═══════════════════════════════════════════════════════════════════════
# Part 1 — Low-level API (synthetic data)
# ═══════════════════════════════════════════════════════════════════════


def low_level_demo() -> None:
    """Run the low-level building-block demo with synthetic voxels."""
    print("=" * 70)
    print("Part 1 — Low-level API (synthetic data)")
    print("=" * 70)

    rng = np.random.default_rng(0)

    # 60 s recording at 10 Hz.
    frame_times = np.arange(0.0, 60.0, 0.1)
    events = pd.DataFrame(
        {
            "trial_type": [
                "stim_left",
                "stim_right",
                "stim_left",
                "stim_right",
                "stim_left",
                "stim_right",
            ],
            "onset": [5.0, 12.0, 22.0, 30.0, 42.0, 50.0],
            "duration": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        }
    )
    confounds = pd.DataFrame(
        {
            "motion_x": 0.05 * np.sin(frame_times / 4.0),
            "motion_y": 0.05 * np.cos(frame_times / 5.0),
        }
    )

    design = make_first_level_design_matrix(
        frame_times,
        events=events,
        confounds=confounds,
        hrf_model="glover",
        drift_model="cosine",
        high_pass=0.01,
    )

    print(f"Design matrix: {design.shape[0]} frames × {design.shape[1]} regressors")
    print(f"Columns: {list(design.columns)}\n")

    # Contrast vectors.
    left_minus_right = expression_to_contrast_vector(
        "stim_left - stim_right",
        design.columns.tolist(),
    )
    task_f_test = np.vstack(
        [
            expression_to_contrast_vector("stim_left", design.columns.tolist()),
            expression_to_contrast_vector("stim_right", design.columns.tolist()),
        ]
    )

    # Simulate 3 voxels with known betas.
    beta_voxel_1 = np.array([1.2, 0.3, 0.0, 0.0] + [0.0] * (design.shape[1] - 4))
    beta_voxel_2 = np.array([0.4, 1.0, 0.1, -0.1] + [0.0] * (design.shape[1] - 4))
    beta_voxel_3 = np.array([0.8, 0.8, -0.1, 0.1] + [0.0] * (design.shape[1] - 4))
    beta = np.column_stack([beta_voxel_1, beta_voxel_2, beta_voxel_3])

    signal = design.to_numpy() @ beta
    noise = 0.15 * rng.standard_normal(signal.shape)
    data = signal + noise

    # OLS fit.
    ols_results = OLSModel(design.to_numpy()).fit(data)
    t_results = ols_results.t_contrast(left_minus_right)
    f_results = ols_results.f_contrast(task_f_test)

    t_contrast = Contrast.from_estimate(
        effect=np.atleast_1d(t_results["effect"]),
        variance=np.atleast_1d(t_results["sd"]) ** 2,
        dof=float(t_results["df_den"]),
        stat_type="t",
    )

    # AR(1) fit — rho shape is (order, n_voxels) for per-voxel whitening.
    n_voxels = data.shape[1]
    ar_results = ARModel(design.to_numpy(), rho=np.full((1, n_voxels), 0.3)).fit(data)
    ar_t_results = ar_results.t_contrast(left_minus_right)

    print("OLS left−right t-statistics:", np.round(t_results["t"], 3))
    print("OLS left−right z-scores:    ", np.round(t_contrast.zscore, 3))
    print("OLS task F-statistics:       ", np.round(f_results["F"], 3))
    print("AR(1) left−right t-stats:   ", np.round(ar_t_results["t"], 3))
    print()


# ═══════════════════════════════════════════════════════════════════════
# Part 2 — High-level API (real fUSI data)
# ═══════════════════════════════════════════════════════════════════════


def _make_visual_stim_events(
    total_duration: float,
    baseline: float = 30.0,
    block_on: float = 5.0,
    block_off: float = 5.0,
) -> pd.DataFrame:
    """Build events for a block-design visual stimulation protocol.

    Starts with a ``baseline`` rest period, then alternates between ON
    and OFF blocks of equal duration until the end of the recording.

    Parameters
    ----------
    total_duration : float
        Total recording duration in seconds.
    baseline : float, default: 30.0
        Initial rest period before the first stimulation block.
    block_on : float, default: 5.0
        Duration of each stimulation ON block in seconds.
    block_off : float, default: 5.0
        Duration of each rest OFF block in seconds.

    Returns
    -------
    pandas.DataFrame
        Events table with ``onset``, ``duration``, and ``trial_type`` columns.
    """
    period = block_on + block_off
    onsets = np.arange(baseline, total_duration - block_on, period)
    return pd.DataFrame(
        {
            "trial_type": "visual_stim",
            "onset": onsets,
            "duration": block_on,
        }
    )


def high_level_demo() -> None:
    """Fit a FirstLevelModel to real fUSI data and plot the results."""
    print("=" * 70)
    print("Part 2 — High-level FirstLevelModel (real fUSI data)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    pwd = (
        cf.load(DATA_PATH)
        .transpose("time", "y", "z", "x")
        .rename({"z": "y", "y": "z"})
        .compute()
    )
    print(f"Loaded power Doppler: {dict(pwd.sizes)}")

    frame_times = pwd.coords["time"].values
    total_duration = float(frame_times[-1] - frame_times[0])
    dt = float(np.median(np.diff(frame_times)))
    print(f"Recording: {total_duration:.0f} s, dt ≈ {dt:.3f} s ({1 / dt:.1f} Hz)\n")

    # ------------------------------------------------------------------
    # Build events: 30 s baseline then 5 s ON / 5 s OFF alternating.
    # ------------------------------------------------------------------
    events = _make_visual_stim_events(total_duration)
    print(f"Stimulation protocol: {len(events)} ON blocks")
    print(events.head())
    print()

    # ------------------------------------------------------------------
    # Fit the GLM
    # ------------------------------------------------------------------
    model = FirstLevelModel(
        hrf_model="glover",
        drift_model="cosine",
        high_pass=0.01,
        noise_model="ar1",
    )
    model.fit(pwd, events=events)

    dm = model.design_matrices_[0]
    print(f"Design matrix: {dm.shape[0]} frames × {dm.shape[1]} regressors")
    print(f"Columns: {list(dm.columns)}\n")

    # ------------------------------------------------------------------
    # Compute contrasts
    # ------------------------------------------------------------------
    z_map = model.compute_contrast("visual_stim", output_type="zscore")
    t_map = model.compute_contrast("visual_stim", output_type="statistic")
    effect_map = model.compute_contrast("visual_stim", output_type="effect")

    print(f"Output dims: {z_map.dims}, shape: {z_map.shape}")
    print(f"z-map range:    [{float(z_map.min()):.2f}, {float(z_map.max()):.2f}]")
    print(f"t-map range:    [{float(t_map.min()):.2f}, {float(t_map.max()):.2f}]")
    print(
        f"effect range:   [{float(effect_map.min()):.2f}, {float(effect_map.max()):.2f}]"
    )
    print()

    # ------------------------------------------------------------------
    # Figure 1 — design matrix
    # ------------------------------------------------------------------
    cols_to_show = [c for c in dm.columns if not c.startswith("cosine")]
    fig_dm, ax_dm = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax_dm.imshow(
        dm[cols_to_show].values,
        aspect="auto",
        cmap="coolwarm",
        extent=(
            -0.5,
            len(cols_to_show) - 0.5,
            float(frame_times[-1]),
            float(frame_times[0]),
        ),
        interpolation="none",
    )
    ax_dm.set_xticks(range(len(cols_to_show)))
    ax_dm.set_xticklabels(cols_to_show, fontsize=9, rotation=45, ha="right")
    ax_dm.set_ylabel("Time (s)")
    ax_dm.set_title("Design matrix (excl. cosine drifts)")
    fig_dm.colorbar(im, ax=ax_dm, shrink=0.8)
    fig_dm.savefig("glm_design_matrix.png", dpi=150)
    print("Design matrix saved to glm_design_matrix.png")

    # ------------------------------------------------------------------
    # Figure 2 — statistical maps
    # ------------------------------------------------------------------
    z_thresh = 3.0

    # Symmetric colormap limits for effect and t maps.
    def _vlim(data: xr.DataArray) -> float:
        vals = data.values[np.isfinite(data.values)]
        return float(np.percentile(np.abs(vals), 98))

    fig_stats, axes_stats = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig_stats.suptitle("FirstLevelModel — visual stimulation contrast", fontsize=13)

    plot_volume(
        effect_map,
        vmin=-_vlim(effect_map),
        vmax=_vlim(effect_map),
        cbar_label="β",
        show_titles=False,
        black_bg=False,
        figure=fig_stats,
        axes=axes_stats[0:1].reshape(1, 1),
    )
    axes_stats[0].set_title("Effect size (β)")

    plot_volume(
        t_map,
        vmin=-_vlim(t_map),
        vmax=_vlim(t_map),
        cbar_label="t",
        show_titles=False,
        black_bg=False,
        figure=fig_stats,
        axes=axes_stats[1:2].reshape(1, 1),
    )
    axes_stats[1].set_title("t-statistic")

    plot_volume(
        z_map,
        threshold=z_thresh,
        vmin=-_vlim(z_map),
        vmax=_vlim(z_map),
        cbar_label="z",
        show_titles=False,
        black_bg=False,
        figure=fig_stats,
        axes=axes_stats[2:3].reshape(1, 1),
    )
    axes_stats[2].set_title(f"z-score (|z| > {z_thresh})")

    fig_stats.savefig("glm_first_level_demo.png", dpi=150)
    print("Statistical maps saved to glm_first_level_demo.png")

    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    low_level_demo()
    high_level_demo()


if __name__ == "__main__":
    main()
