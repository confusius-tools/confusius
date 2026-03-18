"""Generate documentation images for the Quality Control user guide.

Usage
-----
1. Fill in the constant variables below with paths to example datasets on your machine:

    - ``ZARR_PATH``: 4D fUSI recording in Zarr format.
      Used for the DVARS line plot and the carpet plot.
    - ``MASK_ZARR_PATH``: binary brain mask Zarr store.
    - ``SCAN_PATH``: 2Dscan SCAN file.
      Used for the CV and tSNR spatial maps and the mean power Doppler reference image.

2. Run from the project root::

       uv run docs/images/qc/generate.py

All images are saved to docs/images/qc/.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import confusius as cf
from confusius.plotting import plot_carpet
from confusius.qc import compute_cv, compute_dvars, compute_tsnr

HERE = Path(__file__).parent

# == Fill in before running ===================================================

ZARR_PATH = "../../../data/sub-ALD030_ses-ChATChR2FibreMidThalamus_task-awake_acq-motor2dot2_proc-staticsvd50_pwd.zarr/"

MASK_ZARR_PATH = "../../../data/brain_mask.zarr/"

SCAN_PATH = "../../../data/sub-tatooine_ses-20221205_task-anesthetized_pd2dt.scan"

# =============================================================================


def _resolve(path: str) -> str:
    p = Path(path)
    return str((HERE / p).resolve() if not p.is_absolute() else p)


ZARR_PATH = _resolve(ZARR_PATH)
MASK_ZARR_PATH = _resolve(MASK_ZARR_PATH)
SCAN_PATH = _resolve(SCAN_PATH)

_SAVEFIG_KWARGS = {"dpi": 150, "bbox_inches": "tight", "transparent": True}

# --------------------------------------------------------------------------- #
# Load data                                                                     #
# --------------------------------------------------------------------------- #

print("Loading Zarr power Doppler data (for DVARS and carpet plot) …")
pwd_zarr = cf.load(ZARR_PATH)
print(f"  {pwd_zarr.dims}, shape {dict(pwd_zarr.sizes)}")

print("Loading brain mask …")
brain_mask = cf.load(MASK_ZARR_PATH).compute() > 0

print("Extracting brain signals …")
signals = pwd_zarr.fusi.extract.with_mask(brain_mask).compute()

print("Loading SCAN power Doppler data (for CV and tSNR) …")
pwd_scan = cf.load(SCAN_PATH).compute()
pwd_scan.attrs.update(
    {
        "long_name": "Power Doppler intensity",
        "units": "a.u.",
    }
)
print(f"  {pwd_scan.dims}, shape {dict(pwd_scan.sizes)}")

# --------------------------------------------------------------------------- #
# 1. DVARS line plot                                                            #
# --------------------------------------------------------------------------- #

print("\n── DVARS ─────────────────────────────────────────────────────────────")
print("Computing DVARS …")
dvars = compute_dvars(signals)
threshold = 3

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    bg_color = "#1a1a1a" if black_bg else "white"
    fg_color = "white" if black_bg else "black"
    line_color = "#4c9be8" if black_bg else "#1f6fbf"
    threshold_color = "#e05c5c" if black_bg else "#c0392b"

    fig, ax = plt.subplots(figsize=(10, 3), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    flagged = dvars > threshold
    dvars.plot(ax=ax, color=line_color, linewidth=0.8, label=dvars.attrs["long_name"])
    dvars[flagged].plot(
        ax=ax,
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

print("  Saved qc-dvars-light.png and qc-dvars-dark.png")

# --------------------------------------------------------------------------- #
# 2. Carpet plot                                                                #
# --------------------------------------------------------------------------- #

print("\n── Carpet plot ───────────────────────────────────────────────────────")

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    fig, ax = plot_carpet(pwd_zarr, mask=brain_mask, black_bg=black_bg)
    fig.savefig(str(HERE / f"qc-carpet-{suffix}.png"), **_SAVEFIG_KWARGS)
    plt.close(fig)

print("  Saved qc-carpet-light.png and qc-carpet-dark.png")

# --------------------------------------------------------------------------- #
# 3. Mean power Doppler (reference for CV and tSNR sections)                   #
# --------------------------------------------------------------------------- #

print("\n── Mean power Doppler ────────────────────────────────────────────────")
print("Computing mean (dB scale) …")
mean_pwd = pwd_scan.mean("time").fusi.scale.db()

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = mean_pwd.fusi.plot.volume(slice_mode="z", black_bg=black_bg)
    plotter.savefig(str(HERE / f"qc-mean-pwd-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

print("  Saved qc-mean-pwd-light.png and qc-mean-pwd-dark.png")

# --------------------------------------------------------------------------- #
# 4. CV spatial map                                                             #
# --------------------------------------------------------------------------- #

print("\n── CV map ────────────────────────────────────────────────────────────")
print("Computing CV …")
cv = compute_cv(pwd_scan)

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = cv.fusi.plot.volume(slice_mode="z", black_bg=black_bg)
    plotter.savefig(str(HERE / f"qc-cv-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

print("  Saved qc-cv-light.png and qc-cv-dark.png")

# --------------------------------------------------------------------------- #
# 5. tSNR spatial map                                                           #
# --------------------------------------------------------------------------- #

print("\n── tSNR map ──────────────────────────────────────────────────────────")
print("Computing tSNR …")
tsnr = compute_tsnr(pwd_scan)

for black_bg, suffix in [(False, "light"), (True, "dark")]:
    plotter = tsnr.fusi.plot.volume(slice_mode="z", black_bg=black_bg)
    plotter.savefig(str(HERE / f"qc-tsnr-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

print("  Saved qc-tsnr-light.png and qc-tsnr-dark.png")

# --------------------------------------------------------------------------- #

print("\nDone! Rebuild the docs with `just docs` to see the images in place.")
