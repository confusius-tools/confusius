"""Generate documentation images for the Quality Control user guide.

Data is fetched automatically from the Landemard et al. (2026) fUSI-BIDS dataset
from the ConfUSIus dataset collection via `confusius.datasets`. The first run
downloads ~450 MB; subsequent runs use the local cache.

Usage
-----
Run from the project root::

    uv run docs/images/qc/generate.py

All images are saved to docs/images/qc/.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

import confusius as cf
from confusius.datasets import fetch_landemard_2026
from confusius.plotting import plot_carpet
from confusius.qc import compute_cv, compute_dvars, compute_tsnr

HERE = Path(__file__).parent

_SUBJECT = "ALD001"
_REF = 20

_ACQ = f"ref{_REF:02d}"
_FUSI_REL_PATH = (
    Path(f"sub-{_SUBJECT}")
    / "fusi"
    / f"sub-{_SUBJECT}_task-awake_acq-{_ACQ}_pwd.nii.gz"
)
_SCANS_REL_PATH = Path(f"sub-{_SUBJECT}") / f"sub-{_SUBJECT}_scans.tsv"
_ATLAS_REGION_MATCHES_REL_PATH = (
    Path("derivatives")
    / "atlas_mapping"
    / f"sub-{_SUBJECT}"
    / f"AnatomyRegionMatches_{_SUBJECT}.npz"
)

_MEAN_DB_LIMITS = (-25.0, 0.0)
_CV_LIMITS = (0.0, 1.0)

console = Console()


def _section(title: str) -> None:
    console.rule(f"[bold]{title}[/bold]")


def _ok(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def _get_anatomy_session(scans_path: Path, fusi_rel_path: Path) -> str:
    """Return the anatomy session key matching a fUSI recording.

    Landemard stores per-session atlas region matches under
    ``"<acq_time date>_<experiment_id>"`` keys, derived from the scans table.
    """
    with scans_path.open(newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if Path(row["filename"]).name == fusi_rel_path.name:
                date = row["acq_time"].split("T")[0]
                return f"{date}_{int(row['experiment_id'])}"

    raise RuntimeError(
        f"Could not find {fusi_rel_path.name} in {scans_path}. "
        "The scans table does not map this acquisition to an experiment."
    )


# ---------------------------------------------------------------------------
# Fetch dataset and load data
# ---------------------------------------------------------------------------

_section("Load Data")
console.print("Fetching Landemard 2026 dataset")
bids_root = fetch_landemard_2026(
    datasets=["rawdata", "atlas_mapping"],
    subjects=[_SUBJECT],
    acqs=[_ACQ],
    datatypes=["fusi"],
    print_citation=False,
)

console.print("Loading fUSI data")
pwd = cf.load(bids_root / _FUSI_REL_PATH)
console.print(f"  {pwd.dims}, shape {dict(pwd.sizes)}")

console.print("Loading Allen region map")
atlas_path = bids_root / _ATLAS_REGION_MATCHES_REL_PATH
if not atlas_path.exists():
    raise RuntimeError(
        "Missing required derivative atlas file: "
        f"{_ATLAS_REGION_MATCHES_REL_PATH}. "
        "Recreate and publish dataset_index.json with this file included."
    )
anatomy_session = _get_anatomy_session(bids_root / _SCANS_REL_PATH, _FUSI_REL_PATH)
console.print(f"  Anatomy session: {anatomy_session}")
with np.load(atlas_path, allow_pickle=True) as atlas_data:
    atlas_labels = np.asarray(atlas_data[anatomy_session])

pwd_spatial = pwd.isel(time=0, drop=True)
if atlas_labels.shape != pwd_spatial.shape[-2:]:
    raise RuntimeError(
        f"Atlas region map shape {atlas_labels.shape} does not match the fUSI "
        f"slice shape {pwd_spatial.shape[-2:]}."
    )
brain_mask = pwd_spatial.copy(
    data=np.broadcast_to(atlas_labels != 0, pwd_spatial.shape)
)
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

for is_dark, suffix in [(False, "light"), (True, "dark")]:
    bg_color = "#1a1a1a" if is_dark else "white"
    fg_color = "white" if is_dark else "black"
    line_color = "#4c9be8" if is_dark else "#1f6fbf"
    threshold_color = "#e05c5c" if is_dark else "#c0392b"

    fig, ax = plt.subplots(figsize=(10, 3), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    flagged = dvars > threshold
    flagged_dvars = dvars.where(flagged, drop=True)
    ax.plot(
        dvars.coords["time"].values,
        dvars.values,
        color=line_color,
        linewidth=0.8,
        label=dvars.attrs["long_name"],
    )
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

for bg_color, suffix in [("white", "light"), ("black", "dark")]:
    fig, _ax = plot_carpet(pwd, mask=brain_mask, bg_color=bg_color)
    fig.figure.savefig(str(HERE / f"qc-carpet-{suffix}.png"), **_SAVEFIG_KWARGS)
    plt.close(fig.figure)

_ok("Saved qc-carpet-light.png and qc-carpet-dark.png")

# ---------------------------------------------------------------------------
# 3. Mean power Doppler (reference for CV and tSNR sections)
# ---------------------------------------------------------------------------

_section("Mean Power Doppler")
console.print("Computing mean (dB scale)")
mean_pwd = pwd.mean("time").fusi.scale.db()

for bg_color, suffix in [("white", "light"), ("black", "dark")]:
    plotter = mean_pwd.fusi.plot.volume(
        slice_mode="z",
        vmin=_MEAN_DB_LIMITS[0],
        vmax=_MEAN_DB_LIMITS[1],
        bg_color=bg_color,
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

for bg_color, suffix in [("white", "light"), ("black", "dark")]:
    plotter = cv.fusi.plot.volume(
        slice_mode="z",
        vmin=_CV_LIMITS[0],
        vmax=_CV_LIMITS[1],
        bg_color=bg_color,
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

for bg_color, suffix in [("white", "light"), ("black", "dark")]:
    plotter = tsnr.fusi.plot.volume(slice_mode="z", bg_color=bg_color)
    plotter.savefig(str(HERE / f"qc-tsnr-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

_ok("Saved qc-tsnr-light.png and qc-tsnr-dark.png")

# ---------------------------------------------------------------------------

_ok("Done! Rebuild docs with `just docs` to preview changes")
