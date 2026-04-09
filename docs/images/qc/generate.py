"""Generate documentation images for the Quality Control user guide.

Data is fetched automatically from the Nunez-Elizalde et al. (2022) fUSI-BIDS
dataset on OSF (https://osf.io/43skw/) via `confusius.datasets`.  The first run
downloads ~30 MB; subsequent runs use the local cache.

Usage
-----
Run from the project root::

    uv run docs/images/qc/generate.py

All images are saved to docs/images/qc/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022
from confusius.plotting import plot_carpet
from confusius.qc import compute_cv, compute_dvars, compute_tsnr

HERE = Path(__file__).parent

_SUBJECT = "CR022"
_SESSION = "20201011"
_TASK = "spontaneous"
_ACQ_SLICE = "slice03"

_DERIVATIVE_ATLAS_REL_PATH = (
    Path("derivatives")
    / "allenccf_align"
    / f"sub-{_SUBJECT}"
    / f"ses-{_SESSION}"
    / "fusi"
    / f"sub-{_SUBJECT}_ses-{_SESSION}_space-fusi_desc-allenccf_dseg.nii.gz"
)

_MEAN_DB_LIMITS = (-20.0, 0.0)
_CV_LIMITS = (0.0, 1.0)

console = Console()


def _section(title: str) -> None:
    console.rule(f"[bold]{title}[/bold]")


def _ok(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


# ---------------------------------------------------------------------------
# Fetch dataset and load data
# ---------------------------------------------------------------------------

_section("Load Data")
console.print("Fetching Nunez-Elizalde 2022 dataset")
bids_root = fetch_nunez_elizalde_2022(
    subjects=[_SUBJECT],
    sessions=[_SESSION],
    tasks=[_TASK],
)

_FUSI_PATH = (
    bids_root
    / f"sub-{_SUBJECT}/ses-{_SESSION}/fusi"
    / f"sub-{_SUBJECT}_ses-{_SESSION}_task-{_TASK}_acq-{_ACQ_SLICE}_pwd.nii.gz"
)

console.print("Loading fUSI data")
pwd = cf.load(_FUSI_PATH)
console.print(f"  {pwd.dims}, shape {dict(pwd.sizes)}")

console.print("Loading Allen dseg mask")
atlas_path = bids_root / _DERIVATIVE_ATLAS_REL_PATH
if not atlas_path.exists():
    raise RuntimeError(
        "Missing required derivative atlas file: "
        f"{_DERIVATIVE_ATLAS_REL_PATH}. "
        "Recreate and publish dataset_index.json with this file included."
    )
atlas_labels = cf.load(atlas_path).round().astype(np.int32)
if (
    "z" in atlas_labels.dims
    and "z" in pwd.dims
    and atlas_labels.sizes["z"] != pwd.sizes["z"]
):
    atlas_labels = atlas_labels.sel(z=pwd["z"], method="nearest")
    atlas_labels = atlas_labels.assign_coords(z=pwd["z"])

brain_mask = atlas_labels > 0
console.print(f"  Brain mask voxels: {int(brain_mask.sum())}")

console.print("Extracting brain signals")
signals = pwd.fusi.extract.with_mask(brain_mask).compute()

_SAVEFIG_KWARGS = {"dpi": 150, "bbox_inches": "tight", "transparent": True}

# ---------------------------------------------------------------------------
# 1. DVARS line plot
# ---------------------------------------------------------------------------

_section("DVARS")
console.print("Computing DVARS")
dvars = compute_dvars(signals)
threshold = 2.5

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    bg_color = "#1a1a1a" if black_bg else "white"
    fg_color = "white" if black_bg else "black"
    line_color = "#4c9be8" if black_bg else "#1f6fbf"
    threshold_color = "#e05c5c" if black_bg else "#c0392b"

    fig, ax = plt.subplots(figsize=(10, 3), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    flagged = dvars > threshold
    flagged_dvars = dvars.where(flagged, drop=True)
    dvars.plot(ax=ax, color=line_color, linewidth=0.8, label=dvars.attrs["long_name"])
    if flagged_dvars.size > 0:
        ax.plot(
            np.asarray(flagged_dvars["time"].values, dtype=float),
            np.asarray(flagged_dvars.values, dtype=float),
            marker="o",
            linestyle="",
            color=threshold_color,
            ms=3,
            zorder=3,
            label="Flagged frames",
        )
    ax.axhline(
        threshold,
        color=threshold_color,
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold ({threshold})",
    )

    ax.xaxis.label.set_color(fg_color)
    ax.yaxis.label.set_color(fg_color)
    ax.tick_params(colors=fg_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(fg_color)
    ax.legend(facecolor=bg_color, labelcolor=fg_color, framealpha=0.5, fontsize=9)

    fig.tight_layout()
    fig.savefig(str(HERE / f"qc-dvars-{suffix}.png"), **_SAVEFIG_KWARGS)
    plt.close(fig)

_ok("Saved qc-dvars-light.png and qc-dvars-dark.png")

# ---------------------------------------------------------------------------
# 2. Carpet plot
# ---------------------------------------------------------------------------

_section("Carpet Plot")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    fig, ax = plot_carpet(pwd, mask=brain_mask, black_bg=black_bg)
    fig.savefig(str(HERE / f"qc-carpet-{suffix}.png"), **_SAVEFIG_KWARGS)
    plt.close(fig)

_ok("Saved qc-carpet-light.png and qc-carpet-dark.png")

# ---------------------------------------------------------------------------
# 3. Mean power Doppler (reference for CV and tSNR sections)
# ---------------------------------------------------------------------------

_section("Mean Power Doppler")
console.print("Computing mean (dB scale)")
mean_pwd = pwd.mean("time").fusi.scale.db()

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = mean_pwd.fusi.plot.volume(
        slice_mode="z",
        vmin=_MEAN_DB_LIMITS[0],
        vmax=_MEAN_DB_LIMITS[1],
        black_bg=black_bg,
    )
    plotter.savefig(str(HERE / f"qc-mean-pwd-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

_ok("Saved qc-mean-pwd-light.png and qc-mean-pwd-dark.png")

# ---------------------------------------------------------------------------
# 4. CV spatial map
# ---------------------------------------------------------------------------

_section("CV Map")
console.print("Computing CV")
cv = compute_cv(pwd)

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = cv.fusi.plot.volume(
        slice_mode="z",
        vmin=_CV_LIMITS[0],
        vmax=_CV_LIMITS[1],
        black_bg=black_bg,
    )
    plotter.savefig(str(HERE / f"qc-cv-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

_ok("Saved qc-cv-light.png and qc-cv-dark.png")

# ---------------------------------------------------------------------------
# 5. tSNR spatial map
# ---------------------------------------------------------------------------

_section("tSNR Map")
console.print("Computing tSNR")
tsnr = compute_tsnr(pwd)

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = tsnr.fusi.plot.volume(slice_mode="z", black_bg=black_bg)
    plotter.savefig(str(HERE / f"qc-tsnr-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

_ok("Saved qc-tsnr-light.png and qc-tsnr-dark.png")

# ---------------------------------------------------------------------------

_ok("Done! Rebuild docs with `just docs` to preview changes")
