"""Generate Quick Start images for the ConfUSIus home page.

Data is fetched automatically from the Nunez-Elizalde et al. (2022) dataset
on OSF. The first run downloads ~30 MB; subsequent runs use the local cache.

Usage
-----
Run from the project root::

    uv run docs/images/home/generate.py

Images are saved to docs/images/home/ as quickstart-dark.png and
quickstart-light.png.
"""

from pathlib import Path

import confusius as cf  # noqa: F401  # Registers xarray accessors.
from confusius.datasets import fetch_nunez_elizalde_2022

HERE = Path(__file__).parent

_SUBJECT = "CR022"
_SESSION = "20201011"
_TASK = "spontaneous"
_ACQ = "slice03"
_DB_LIMITS = (-20.0, 0.0)

print("Fetching dataset …")
bids_root = fetch_nunez_elizalde_2022(
    subjects=_SUBJECT, sessions=_SESSION, tasks=_TASK, acqs=_ACQ
)

print("Loading power Doppler …")
data = cf.load(
    bids_root
    / f"sub-{_SUBJECT}/ses-{_SESSION}/fusi"
    / f"sub-{_SUBJECT}_ses-{_SESSION}_task-{_TASK}_acq-{_ACQ}_pwd.nii.gz"
)

print("Computing mean and converting to dB …")
mean_db = data.mean("time").compute().fusi.scale.db()

for bg_color, suffix in [("black", "dark"), ("white", "light")]:
    plotter = mean_db.fusi.plot.volume(
        slice_mode="z",
        cmap="gray",
        vmin=_DB_LIMITS[0],
        vmax=_DB_LIMITS[1],
        cbar_label="Power Doppler (dB)",
        bg_color=bg_color,
    )
    out = HERE / f"quickstart-{suffix}.png"
    plotter.savefig(str(out), dpi=150, bbox_inches="tight", transparent=True)
    plotter.close()
    print(f"Saved {out}")

print("Done. Rebuild docs with `just docs` to preview.")
