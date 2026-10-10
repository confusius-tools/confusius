"""Fetcher for the Huang et al. (2025) vascular fUSI template."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from confusius.datasets._s3 import (
    download_s3_files,
    get_index,
    get_release_dir,
    update_cached_index,
)
from confusius.datasets._utils import (
    get_datasets_dir,
    plain_citation,
    print_citation_message,
)
from confusius.io.loadsave import load

_TEMPLATE_ROOT = "huang-2025-template"
_FILENAME = "huang-2025-space-allen50_desc-vascular.nii.gz"
_TOTAL_SIZE_BYTES = 16_338_965
_CITATION = (
    "Huang, Y.-A., Lambert, T., Verbeyst, D., Fitzgerald, N. E., Grillet, M., "
    "Brunner, C., Montaldo, G., Vanduffel, W., & Urban, A. (2025). "
    "[citation.title]OfUSA: OpenfUS Analyzer, a versatile open-source framework for the "
    "analysis and visualization of functional ultrasound imaging data across animal "
    "models.[/citation.title] [italic]bioRxiv[/italic]. "
    "[citation.doi]https://doi.org/10.1101/2025.09.16.676515[/citation.doi]"
)


def fetch_template_huang_2025(
    data_dir: str | Path | None = None,
    refresh: bool = False,
    print_citation: bool = True,
) -> xr.DataArray:
    """Fetch the Huang et al. (2025) mouse vascular fUSI template.

    Downloads the template from the ConfUSIus dataset collection, caches it locally,
    and returns the loaded NIfTI as a VoxelData array.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory in which to cache the template. Defaults to the platform cache
        directory (e.g. `~/.cache/confusius` on Linux, `~/Library/Caches/confusius` on
        macOS, `%LOCALAPPDATA%\\confusius\\Cache` on Windows), overridable via the
        `CONFUSIUS_DATA` environment variable.
    refresh : bool, default: False
        Whether to resolve the latest published release. Otherwise the cached release
        is reused offline. Releases are cached in separate version directories.
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
        [ConfUSIus dataset collection](https://github.com/confusius-tools/confusius-datasets).

    [^3]:
        Template license (CC BY-NC-SA 4.0):
        [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)
    """
    cache_dir = get_datasets_dir(data_dir) / _TEMPLATE_ROOT
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = get_index(cache_dir, "templates", _TEMPLATE_ROOT, refresh=refresh)
    dataset_dir = get_release_dir(cache_dir, index)
    download_s3_files(dataset_dir, index, refresh=refresh)
    update_cached_index(cache_dir, index)

    da = load(dataset_dir / _FILENAME)
    da.attrs["citation"] = plain_citation(_CITATION)

    if print_citation:
        print_citation_message(_CITATION, "template")
    return da
