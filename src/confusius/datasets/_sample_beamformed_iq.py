"""Lazy access to a sample beamformed IQ recording."""

from pathlib import Path

import xarray as xr

from confusius.datasets._s3 import get_index, update_cached_index
from confusius.datasets._utils import (
    get_datasets_dir,
    plain_citation,
    print_citation_message,
)
from confusius.io.loadsave import load

_CITATION = (
    "Landemard, A., Krumin, M., Harris, K. D., & Carandini, M. (2026). "
    "[citation.title]Brainwide blood volume reflects opposing neural populations."
    "[/citation.title] [italic]Nature[/italic]. "
    "[citation.doi]https://doi.org/10.1038/s41586-026-10350-9[/citation.doi]"
)


def fetch_sample_beamformed_iq(
    data_dir: str | Path | None = None,
    refresh: bool = False,
    print_citation: bool = True,
) -> xr.DataArray:
    """Open a sample beamformed IQ recording lazily from the ConfUSIus collection.

    The Landemard et al. (2026) awake ALD019 recording is a Zarr store. Data chunks
    are downloaded when computed; no full local copy is created. Repeated reads may
    transfer the same chunks again. The recording is licensed under CC BY-NC 4.0.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory in which to cache release metadata. Defaults to the platform cache
        directory, overridable via the `CONFUSIUS_DATA` environment variable.
        IQ chunks are not cached locally.
    refresh : bool, default: False
        Whether to resolve the latest published release instead of reusing cached
        release metadata. Opening the Zarr store still requires network access.
    print_citation : bool, default: True
        Whether to print the citation for the recording.

    Returns
    -------
    xarray.DataArray
        Dask-backed beamformed IQ VoxelData array.

    References
    ----------
    [^1]:
        Landemard, A. et al. (2026). Brainwide blood volume reflects opposing neural
        populations. Nature. [https://doi.org/10.1038/s41586-026-10350-9](https://doi.org/10.1038/s41586-026-10350-9)
    """
    name = "sample-beamformed-iq"
    store = "sub-ALD019_task-awake_acq-ref25_iq.zarr"
    cache_dir = get_datasets_dir(data_dir) / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = get_index(cache_dir, "datasets", name, refresh=refresh)
    url = index[f"{store}/zarr.json"]["url"].removesuffix("/zarr.json")
    data = load(url, variable="iq", chunks={})
    update_cached_index(cache_dir, index)
    data.attrs["citation"] = plain_citation(_CITATION)
    if print_citation:
        print_citation_message(_CITATION, "dataset")
    return data
