"""Fetcher for the Khallaf et al. (2026) naked mole-rat fUSI dataset."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import requests

from ._dataverse import (
    ZipMemberInfo,
    access_url,
    download_missing_zip_members,
    get_zip_index,
)
from ._utils import get_datasets_dir

_DOI = "doi:10.17617/3.7QCU1F"
"""Persistent identifier of the Edmond dataset hosting the fUSI archive."""

_DATAFILE_ID = 343674
"""Dataverse datafile id of the `fUSI data.zip` archive within the dataset."""

_ZIP_ROOT = "naked_mole_rat_fusi_dataset/"
"""Common in-zip path prefix shared by every archive member."""

_BIDS_ROOT = "khallaf-2026-bids"
"""Local cache directory name for the extracted dataset."""

_TOTAL_SIZE_BYTES = 19_485_480_391
"""Size of the full `fUSI data.zip` archive in bytes."""


_VALID_DATASETS = frozenset({"rawdata", "glm", "bootstrapping"})
"""Valid values for the `datasets` parameter of `fetch_khallaf_2026`."""


_VALID_RECONSTRUCTIONS = frozenset({"raw", "resampled", "both"})
"""Valid values for the `reconstruction` parameter of `fetch_khallaf_2026`."""


def _bucket(parts: tuple[str, ...]) -> str:
    """Classify an archive member into a top-level dataset bucket.

    Parameters
    ----------
    parts : tuple[str, ...]
        Path components of a member's prefix-stripped relative path.

    Returns
    -------
    str
        One of `"rawdata"`, `"sourcedata"`, `"glm"`, `"bootstrapping"`, or
        `"metadata"` (top-level files that belong to no dataset bucket).
    """
    head = parts[0]
    if head == "sourcedata":
        return "sourcedata"
    if head == "derivatives":
        name = parts[1] if len(parts) >= 2 else "derivatives"
        # `bootstrapping/` (folder) and `bootstrapping.zip` (loose file) both
        # belong to the bootstrapping derivative.
        return "bootstrapping" if name.startswith("bootstrapping") else name
    if head.startswith("sub-"):
        return "rawdata"
    return "metadata"


def _entity(rel: str, key: str) -> str | None:
    """Return the value of a BIDS entity in a member path, or None if absent.

    The entity may appear either as a path component (e.g. `sub-5622/`) or as a
    filename token (e.g. `..._run-1_...`); both forms are matched.

    Parameters
    ----------
    rel : str
        Prefix-stripped relative path of a member.
    key : str
        Entity key without its trailing dash, e.g. `"sub"`, `"ses"`, `"run"`.

    Returns
    -------
    str or None
        The entity value, or `None` when the entity is not present.
    """
    match = re.search(rf"{key}-([A-Za-z0-9]+)", rel)
    return match.group(1) if match is not None else None


def _matches_entities(
    rel: str,
    subjects: list[str] | None,
    sessions: list[str] | None,
    runs: list[str] | None,
) -> bool:
    """Return True when a member satisfies the subject/session/run filters.

    For each filter, a member matches when it either omits the corresponding
    BIDS entity entirely or declares a value that is in the requested list.

    Parameters
    ----------
    rel : str
        Prefix-stripped relative path of a member.
    subjects : list[str] or None
        Subject IDs to keep (without `sub-` prefix). If `None`, all subjects
        match.
    sessions : list[str] or None
        Session IDs to keep (without `ses-` prefix). If `None`, all sessions
        match.
    runs : list[str] or None
        Run indices to keep (without `run-` prefix). If `None`, all runs match.

    Returns
    -------
    bool
        Whether the member passes all three entity filters.
    """
    if subjects is not None:
        sub = _entity(rel, "sub")
        if sub is not None and sub not in subjects:
            return False
    if sessions is not None:
        ses = _entity(rel, "ses")
        if ses is not None and ses not in sessions:
            return False
    if runs is not None:
        run = _entity(rel, "run")
        if run is not None and run not in runs:
            return False
    return True


def _matches_reconstruction(parts: tuple[str, ...], reconstruction: str) -> bool:
    """Return True when a rawdata member matches the reconstruction filter.

    Only `fusi/` files carry a reconstruction variant: the resampled variant is
    tagged with a `rec-resampled` entity, the raw variant carries no `rec-`
    entity. Files outside a `fusi/` directory (e.g. `scans.tsv`) are unaffected.

    Parameters
    ----------
    parts : tuple[str, ...]
        Path components of a rawdata member's prefix-stripped relative path.
    reconstruction : str
        One of `"raw"`, `"resampled"`, or `"both"`.

    Returns
    -------
    bool
        Whether the member passes the reconstruction filter.
    """
    if reconstruction == "both" or "fusi" not in parts:
        return True
    variant = "resampled" if "rec-resampled" in parts[-1] else "raw"
    return variant == reconstruction


def _filter_members(
    index: dict[str, ZipMemberInfo],
    datasets: list[str] | None,
    subjects: list[str] | None,
    sessions: list[str] | None,
    runs: list[str] | None,
    reconstruction: str,
    sourcedata: bool,
) -> dict[str, ZipMemberInfo]:
    """Filter the member index to the requested datasets and entities.

    Top-level metadata files (`dataset_description.json`, `participants.*`,
    `events.*`, `README.md`) are always included. The `sourcedata/` tree is
    gated solely by `sourcedata`: when enabled it is included in full with no
    entity filtering; when disabled it is excluded entirely. All other buckets
    honour the `datasets`, `subjects`, `sessions`, and `runs` filters, and
    rawdata additionally honours `reconstruction`.

    Parameters
    ----------
    index : dict[str, ZipMemberInfo]
        Full member index as returned by
        [`get_zip_index`][confusius.datasets._dataverse.get_zip_index].
    datasets : list[str] or None
        Dataset buckets to include (`"rawdata"`, `"glm"`, `"bootstrapping"`).
        If `None`, all non-sourcedata buckets are included.
    subjects : list[str] or None
        Subject IDs to include (without `sub-` prefix). If `None`, all subjects
        are included.
    sessions : list[str] or None
        Session IDs to include (without `ses-` prefix). If `None`, all sessions
        are included.
    runs : list[str] or None
        Run indices to include (without `run-` prefix). If `None`, all runs are
        included.
    reconstruction : str
        Reconstruction variant of rawdata `fusi/` files to keep: `"raw"`,
        `"resampled"`, or `"both"`.
    sourcedata : bool
        Whether to include the entire `sourcedata/` tree (unfiltered).

    Returns
    -------
    dict[str, ZipMemberInfo]
        Subset of the index matching the filters.
    """
    filtered: dict[str, ZipMemberInfo] = {}

    for rel, info in index.items():
        parts = Path(rel).parts
        bucket = _bucket(parts)

        # Top-level metadata is always included.
        if bucket == "metadata":
            filtered[rel] = info
            continue

        # sourcedata is gated only by the flag, and is never entity-filtered.
        if bucket == "sourcedata":
            if sourcedata:
                filtered[rel] = info
            continue

        if datasets is not None and bucket not in datasets:
            continue

        if not _matches_entities(rel, subjects, sessions, runs):
            continue

        if bucket == "rawdata" and not _matches_reconstruction(parts, reconstruction):
            continue

        filtered[rel] = info

    return filtered


def _as_str_list(value: str | int | Iterable[str | int] | None) -> list[str] | None:
    """Normalize a filter argument to a list of strings, or None.

    Accepts a single scalar (`str` or `int`), an iterable of scalars, or
    `None`. Numeric IDs are coerced to `str` so that callers passing e.g.
    `subjects=5622` or `runs=[1, 2]` match the string-valued BIDS entities
    rather than silently selecting nothing.

    Parameters
    ----------
    value : str or int or collections.abc.Iterable or None
        Raw filter argument as passed by the caller.

    Returns
    -------
    list[str] or None
        List of string values, or `None` when `value` is not provided.
    """
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return [str(value)]
    return [str(item) for item in value]


def fetch_khallaf_2026(
    data_dir: str | Path | None = None,
    datasets: str | list[str] | None = None,
    subjects: str | list[str] | None = None,
    sessions: str | list[str] | None = None,
    runs: str | list[str] | None = None,
    reconstruction: str = "both",
    sourcedata: bool = False,
    refresh: bool = False,
) -> Path:
    """Fetch the Khallaf et al. (2026) naked mole-rat fUSI dataset.

    Downloads functional ultrasound imaging data from naked mole-rats exposed to
    olfactory stimulation, organised following BIDS and the proposed fUSI extension
    BEP-040. The data is hosted on Edmond (the Max Planck Society's Dataverse
    repository) as a single ~19.5 GB zip archive; requested members are streamed
    out of it individually by HTTP range request rather than downloading the
    whole archive.

    Files are extracted on first call and cached locally. Subsequent calls with
    the same `data_dir` return immediately for already-cached files.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory in which to cache the dataset. Defaults to the platform
        cache directory (e.g. `~/.cache/confusius` on Linux,
        `~/Library/Caches/confusius` on macOS,
        `%LOCALAPPDATA%\\confusius\\Cache` on Windows),
        overridable via the `CONFUSIUS_DATA` environment variable.
    datasets : str or list[str], optional
        Dataset buckets to download: `"rawdata"` for the raw fUSI data,
        `"glm"` and `"bootstrapping"` for the processed derivatives. Accepts a
        single string or a list. If not provided, all of these are downloaded.
        Does not control `"sourcedata"` (see the `"sourcedata"` parameter).
    subjects : str or list[str], optional
        Subject IDs to download (without `sub-` prefix), e.g. `"5622"` or
        `["5622", "6036"]`. If not provided, all subjects are downloaded. Files
        with no subject entity are always included.
    sessions : str or list[str], optional
        Session IDs to download (without `ses-` prefix), e.g. `"IPM"` or
        `["Air", "Etoh"]`. If not provided, all sessions are downloaded. Files
        with no session entity are always included.
    runs : str or list[str], optional
        Run indices to download (without `run-` prefix), e.g. `"1"` or
        `["1", "2"]`. If not provided, all runs are downloaded. Files with no
        run entity are always included.
    reconstruction : str, default: "both"
        Which reconstruction of the rawdata `fusi/` volumes to download:
        `"raw"` for the unregistered volumes (no `rec-` entity), `"resampled"`
        for the registered `rec-resampled` volumes, or `"both"`. Only affects
        rawdata; derivatives are unaffected.
    sourcedata : bool, default: False
        Whether to download the raw Iconeus acquisition files under
        `sourcedata/`. When `True`, the entire `sourcedata/` directory
        (~7.8 GB) is downloaded with no subject/session filtering applied.
    refresh : bool, default: False
        Whether to re-read the archive index from Edmond and download any files
        that are missing locally. If `False` and all requested files are already
        cached, the function returns immediately without any network access.

    Returns
    -------
    pathlib.Path
        Path to the BIDS root directory of the cached dataset.

    Raises
    ------
    ValueError
        If an unknown dataset name is passed in `datasets`, or an unknown value
        is passed in `reconstruction`.

    References
    ----------
    [^1]:
        Khallaf, M. A. et al. (2026). A queen odour mediates reproductive
        suppression in a eusocial mammal.
        <!-- TODO: add the paper DOI and full citation once published. -->

    [^2]:
        fUSI dataset on Edmond (Max Planck Society):
        [https://doi.org/10.17617/3.7QCU1F](https://doi.org/10.17617/3.7QCU1F)

    [^3]:
        Dataset license (CC0 1.0):
        [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)
    """
    bids_dir = get_datasets_dir(data_dir) / _BIDS_ROOT
    bids_dir.mkdir(parents=True, exist_ok=True)

    # Normalize each filter to a list of strings. Coercion matters because the
    # subject and run IDs are numeric strings ("5622", "1"): without it a
    # natural `subjects=5622` or `runs=[1, 2]` would never match and silently
    # drop the requested data.
    datasets = _as_str_list(datasets)
    subjects = _as_str_list(subjects)
    sessions = _as_str_list(sessions)
    runs = _as_str_list(runs)

    if datasets is not None:
        invalid = set(datasets) - _VALID_DATASETS
        if invalid:
            raise ValueError(
                f"Unknown dataset(s): {invalid}. "
                f"Valid options: {sorted(_VALID_DATASETS)}"
            )

    if reconstruction not in _VALID_RECONSTRUCTIONS:
        raise ValueError(
            f"Unknown reconstruction: {reconstruction!r}. "
            f"Valid options: {sorted(_VALID_RECONSTRUCTIONS)}"
        )

    url = access_url(_DATAFILE_ID)
    # One session pools keep-alive connections across the central-directory read
    # and every per-member range request.
    with requests.Session() as session:
        index = get_zip_index(
            bids_dir, url, _ZIP_ROOT, refresh=refresh, session=session
        )
        members = _filter_members(
            index, datasets, subjects, sessions, runs, reconstruction, sourcedata
        )
        download_missing_zip_members(bids_dir, url, members, _ZIP_ROOT, session=session)

    return bids_dir
