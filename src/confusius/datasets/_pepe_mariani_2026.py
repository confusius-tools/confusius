"""Fetcher for the Pepe, Mariani et al. (2026) fUSI template."""

from __future__ import annotations

from pathlib import Path

import pooch
import requests
import xarray as xr

from confusius.io.loadsave import load

from ._pooch import quiet_pooch_logger, retrieve_with_retries
from ._utils import get_datasets_dir

_OSF_PROJECT_ID = "43tu9"
_TEMPLATE_ROOT = "pepe-mariani-2026-template"
_FILENAME = "pepe-mariani-2026-fusi-template.nii.gz"
_TOTAL_SIZE_BYTES = 5_507_550


def resolve_template_url(project_id: str = _OSF_PROJECT_ID) -> str:
    """Return the direct OSF download URL for the exported template.

    Parameters
    ----------
    project_id : str, default: "43tu9"
        OSF project identifier.

    Returns
    -------
    str
        Direct download URL for `pepe-mariani-2026-fusi-template.nii.gz`.

    Raises
    ------
    RuntimeError
        If the template file is not present in the OSF project storage root.
    """
    response = requests.get(
        f"https://api.osf.io/v2/nodes/{project_id}/files/osfstorage/"
    )
    response.raise_for_status()

    for item in response.json()["data"]:
        if item["attributes"]["name"] == _FILENAME:
            return item["links"]["download"]

    raise RuntimeError(f"Could not find {_FILENAME!r} in OSF project {project_id}.")


def fetch_template_pepe_mariani_2026(
    data_dir: str | Path | None = None,
    refresh: bool = False,
) -> xr.DataArray:
    """Fetch the Pepe, Mariani et al. (2026) mouse fUSI template.

    Downloads the template from OSF on first call, caches it locally, and returns the
    loaded NIfTI as an Xarray DataArray.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory in which to cache the template. Defaults to the platform cache
        directory (e.g. `~/.cache/confusius` on Linux, `~/Library/Caches/confusius` on
        macOS, `%LOCALAPPDATA%\\confusius\\Cache` on Windows), overridable via the
        `CONFUSIUS_DATA` environment variable.
    refresh : bool, default: False
        Whether to redownload the template even if it is already cached.

    Returns
    -------
    xarray.DataArray
        Template with `physical_to_sform` affine transform required for resampling to
        the Allen Mouse Brain atlas space.

    References
    ----------
    [^1]:
        Pepe, C. et al. (2026). Structural and dynamic embedding of the mouse
        functional connectome revealed by functional ultrasound imaging (fUSI).
        [https://doi.org/10.64898/2026.02.05.704055](https://doi.org/10.64898/2026.02.05.704055)

    [^2]:
        Template hosted on OSF: [https://osf.io/43tu9/](https://osf.io/43tu9/)
    """
    dataset_dir = get_datasets_dir(data_dir) / _TEMPLATE_ROOT
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dest = dataset_dir / _FILENAME

    if refresh and dest.exists():
        dest.unlink()

    if not dest.exists():
        url = resolve_template_url()
        with quiet_pooch_logger():
            retrieve_with_retries(url, dest, logger=pooch.get_logger())

    return load(dest)
