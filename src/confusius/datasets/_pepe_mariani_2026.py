"""Fetcher for the Pepe, Mariani et al. (2026) fUSI template."""

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

_TEMPLATE_ROOT = "pepe-mariani-2026-template"
_FILENAME = "pepe-mariani-2026-fusi-template.nii.gz"
_TOTAL_SIZE_BYTES = 5_507_569
_CITATION = (
    "Pepe, C., Mariani, J.-C., Urosevic, M., Gini, S., Stuefer, A., Ricci, F., "
    "Galbusera, A., Iurilli, G., & Gozzi, A. (2026). "
    "[citation.title]Structural and dynamic embedding of the mouse functional "
    "connectome revealed by functional ultrasound imaging (fUSI).[/citation.title] "
    "[italic]bioRxiv[/italic]. "
    "[citation.doi]https://doi.org/10.64898/2026.02.05.704055[/citation.doi]"
)


def fetch_template_pepe_mariani_2026(
    data_dir: str | Path | None = None,
    refresh: bool = False,
    print_citation: bool = True,
) -> xr.DataArray:
    """Fetch the Pepe, Mariani et al. (2026) mouse fUSI template.

    Downloads the template from the AWS Open Data–sponsored S3 collection, caches
    it locally, and returns the
    loaded NIfTI as a VoxelData array.

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
        Native-resolution template in scanner space, with `world_to_sform` mapping
        its world coordinates into Allen Mouse Brain atlas space.

    References
    ----------
    [^1]:
        Pepe, C. et al. (2026). Structural and dynamic embedding of the mouse
        functional connectome revealed by functional ultrasound imaging (fUSI).
        [https://doi.org/10.64898/2026.02.05.704055](https://doi.org/10.64898/2026.02.05.704055)

    [^2]:
        Template collection: [confusius-datasets](https://github.com/confusius-tools/confusius-datasets).

    [^3]:
        Template license (CC BY 4.0):
        [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
    """
    cache_dir = get_datasets_dir(data_dir) / _TEMPLATE_ROOT
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = get_index(cache_dir, "templates", _TEMPLATE_ROOT, refresh=refresh)
    dataset_dir = get_release_dir(cache_dir, index)
    download_s3_files(dataset_dir, index)
    update_cached_index(cache_dir, index)
    dest = dataset_dir / _FILENAME

    # Keep scanner geometry for registration; the Allen transform remains in sform.
    da = load(dest, coordinate_affine="qform")
    da.attrs["citation"] = plain_citation(_CITATION)

    if print_citation:
        print_citation_message(_CITATION, "template")
    return da
