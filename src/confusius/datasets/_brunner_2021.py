"""Fetcher for the Brunner et al. (2021) vascular fUSI template."""

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

_TEMPLATE_ROOT = "brunner-2021-template"
_FILENAME = "brunner-2021-space-allen50_desc-vascular.nii.gz"
_TOTAL_SIZE_BYTES = 16_338_965
_CITATION = (
    "Brunner, C., Grillet, M., Urban, A., Roska, B., Montaldo, G., & Macé, E. (2021). "
    "[citation.title]Whole-brain functional ultrasound imaging in awake head-fixed "
    "mice.[/citation.title] [italic]Nature Protocols[/italic], 16, 3547–3571. "
    "[citation.doi]https://doi.org/10.1038/s41596-021-00548-8[/citation.doi]\n"
    "Macé, E., & Montaldo, G. (2020). [citation.title]Whole-brain visually evoked "
    "activity maps in head-fixed awake mice[/citation.title] (June 2021 metadata "
    "update). [italic]Zenodo[/italic]. "
    "[citation.doi]https://doi.org/10.5281/zenodo.4905862[/citation.doi]"
)


def fetch_template_brunner_2021(
    data_dir: str | Path | None = None,
    refresh: bool = False,
    print_citation: bool = True,
) -> xr.DataArray:
    """Fetch the Brunner et al. (2021) mouse vascular fUSI template.

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
        Brunner, C. et al. (2021). Whole-brain functional ultrasound imaging in awake
        head-fixed mice. Nature Protocols, 16, 3547–3571.
        [https://doi.org/10.1038/s41596-021-00548-8](https://doi.org/10.1038/s41596-021-00548-8)

    [^2]:
        Macé, E. and Montaldo, G. (2020). Whole-brain visually evoked activity maps in
        head-fixed awake mice (June 2021 metadata update). Zenodo.
        [https://doi.org/10.5281/zenodo.4905862](https://doi.org/10.5281/zenodo.4905862)

    [^3]:
        [ConfUSIus dataset collection](https://github.com/confusius-tools/confusius-datasets).

    [^4]:
        Template license (CC BY 4.0):
        [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
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
