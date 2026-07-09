"""Fetcher for the Huang et al. (2025) vascular fUSI template."""

from __future__ import annotations

from pathlib import Path

import pooch
import requests
import xarray as xr

from confusius.io.loadsave import load

from ._pooch import quiet_pooch_logger, retrieve_with_retries
from ._utils import get_datasets_dir, plain_citation, print_citation_message

_OSF_PROJECT_ID = "am3jw"
_TEMPLATE_ROOT = "huang-2025-template"
_FILENAME = "huang-2026-space-allen50_desc-vascular.nii.gz"
_TOTAL_SIZE_BYTES = 16_338_947
_CITATION = (
    "Huang, Y.-A., Lambert, T., Verbeyst, D., Fitzgerald, N. E., Grillet, M., "
    "Brunner, C., Montaldo, G., Vanduffel, W., & Urban, A. (2025). "
    "[citation.title]OfUSA: OpenfUS Analyzer, a versatile open-source framework for the "
    "analysis and visualization of functional ultrasound imaging data across animal "
    "models.[/citation.title] [italic]bioRxiv[/italic]. "
    "[citation.doi]https://doi.org/10.1101/2025.09.16.676515[/citation.doi]"
)


def resolve_template_url(project_id: str = _OSF_PROJECT_ID) -> str:
    """Return the direct OSF download URL for the exported template.

    Parameters
    ----------
    project_id : str, default: "am3jw"
        OSF project identifier.

    Returns
    -------
    str
        Direct download URL for `huang-2026-space-allen50_desc-vascular.nii.gz`.

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


def fetch_template_huang_2025(
    data_dir: str | Path | None = None,
    refresh: bool = False,
    print_citation: bool = True,
) -> xr.DataArray:
    """Fetch the Huang et al. (2025) mouse vascular fUSI template.

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
    print_citation : bool, default: True
        Whether to print the citation for the template.

    Returns
    -------
    xarray.DataArray
        Vascular template aligned to Allen CCF space.

    References
    ----------
    [^1]:
        Huang, Y.-A. et al. (2025). OfUSA: OpenfUS Analyzer, a versatile open-source
        framework for the analysis and visualization of functional ultrasound imaging
        data across animal models.
        [https://doi.org/10.1101/2025.09.16.676515](https://doi.org/10.1101/2025.09.16.676515)

    [^2]:
        Template hosted on OSF: [https://osf.io/am3jw/](https://osf.io/am3jw/)

    [^3]:
        Template license (CC BY-NC-SA 4.0):
        [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)
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

    da = load(dest)
    da.attrs["citation"] = plain_citation(_CITATION)

    if print_citation:
        print_citation_message(_CITATION, "template")
    return da
