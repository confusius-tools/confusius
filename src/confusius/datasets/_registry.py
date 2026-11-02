"""Dataset registry metadata and listing utilities."""

from __future__ import annotations

from pathlib import Path

from ._brunner_grillet_2021 import _TEMPLATE_ROOT as _brunner_grillet_2021_template_root
from ._brunner_grillet_2021 import _TOTAL_SIZE_BYTES as _brunner_grillet_2021_size
from ._cybis_pereira_2026 import _BIDS_ROOT as _cybis_pereira_2026_bids_root
from ._cybis_pereira_2026 import _TOTAL_SIZE_BYTES as _cybis_pereira_2026_size
from ._khallaf_2026 import _BIDS_ROOT as _khallaf_2026_bids_root
from ._khallaf_2026 import _TOTAL_SIZE_BYTES as _khallaf_2026_size
from ._landemard_2026 import _BIDS_ROOT as _landemard_2026_bids_root
from ._landemard_2026 import _TOTAL_SIZE_BYTES as _landemard_2026_size
from ._nunez_elizalde_2022 import _BIDS_ROOT as _nunez_elizalde_2022_bids_root
from ._nunez_elizalde_2022 import _TOTAL_SIZE_BYTES as _nunez_elizalde_2022_size
from ._pepe_mariani_2026 import _TEMPLATE_ROOT as _pepe_mariani_2026_template_root
from ._pepe_mariani_2026 import _TOTAL_SIZE_BYTES as _pepe_mariani_2026_size
from ._pepe_mariani_2026_bids import _BIDS_ROOT as _pepe_mariani_2026_bids_root
from ._pepe_mariani_2026_bids import _TOTAL_SIZE_BYTES as _pepe_mariani_2026_bids_size
from ._pereira_2025 import _BIDS_ROOT as _pereira_2025_bids_root
from ._pereira_2025 import _TOTAL_SIZE_BYTES as _pereira_2025_size
from ._s3 import _INDEX_FILENAME
from ._utils import get_datasets_dir

_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB")

RegistryEntry = tuple[str, int, str]

_REGISTRY: tuple[RegistryEntry, ...] = (
    (
        "fetch_nunez_elizalde_2022",
        _nunez_elizalde_2022_size,
        _nunez_elizalde_2022_bids_root,
    ),
    (
        "fetch_pereira_2025",
        _pereira_2025_size,
        _pereira_2025_bids_root,
    ),
    (
        "fetch_cybis_pereira_2026",
        _cybis_pereira_2026_size,
        _cybis_pereira_2026_bids_root,
    ),
    (
        "fetch_pepe_mariani_2026",
        _pepe_mariani_2026_bids_size,
        _pepe_mariani_2026_bids_root,
    ),
    (
        "fetch_landemard_2026",
        _landemard_2026_size,
        _landemard_2026_bids_root,
    ),
    (
        "fetch_khallaf_2026",
        _khallaf_2026_size,
        _khallaf_2026_bids_root,
    ),
    (
        "fetch_template_brunner_grillet_2021",
        _brunner_grillet_2021_size,
        _brunner_grillet_2021_template_root,
    ),
    (
        "fetch_template_pepe_mariani_2026",
        _pepe_mariani_2026_size,
        _pepe_mariani_2026_template_root,
    ),
)
"""Registry of (fetcher_name, total_size_bytes, root_dirname) per dataset."""


def _format_bytes(size_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Uses SI (base-1000) units.

    Parameters
    ----------
    size_bytes : int
        Size in bytes.

    Returns
    -------
    str
        Human-readable size string, e.g. `"6.4 GB"`.
    """
    size = float(size_bytes)
    for unit in _SIZE_UNITS[:-1]:
        if abs(size) < 1000:
            return f"{size:.4g} {unit}"
        size /= 1000
    return f"{size:.4g} {_SIZE_UNITS[-1]}"


def is_dataset_cached(fetcher_name: str, data_dir: str | Path | None = None) -> bool:
    """Whether a dataset or template is in the local cache.

    True once the fetcher named `fetcher_name` has recorded a complete release in the
    cache, so calling it again reads from disk without downloading (unless it is
    called with `refresh=True`).

    Parameters
    ----------
    fetcher_name : str
        Name of a `confusius.datasets` fetcher, e.g.
        `"fetch_template_pepe_mariani_2026"`.
    data_dir : str or pathlib.Path, optional
        Cache directory. Defaults to the directory resolved by
        [`get_datasets_dir`][confusius.datasets.get_datasets_dir].

    Returns
    -------
    bool
        Whether the fetcher's release inventory is on disk.

    Raises
    ------
    KeyError
        If `fetcher_name` is not a registered fetcher.

    Examples
    --------
    >>> is_dataset_cached("fetch_template_pepe_mariani_2026")  # doctest: +SKIP
    False
    """
    roots = {name: root for name, _, root in _REGISTRY}
    if fetcher_name not in roots:
        raise KeyError(
            f"Unknown dataset fetcher {fetcher_name!r}; expected one of "
            f"{sorted(roots)}."
        )
    return (
        get_datasets_dir(data_dir) / roots[fetcher_name] / _INDEX_FILENAME
    ).is_file()


def list_datasets(data_dir: str | Path | None = None) -> None:
    """Print a table of available datasets.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Cache directory to check for existing datasets. Defaults to the
        directory resolved by `get_datasets_dir`.
    """
    import sys

    from rich.console import Console
    from rich.table import Table

    datasets_dir = get_datasets_dir(data_dir)
    console = Console()

    # Rich rewrites box-drawing chars when the terminal is ASCII-only, but cell
    # contents are passed through verbatim, so ✓/✗ must be guarded explicitly.
    unicode_ok = "utf" in (sys.stdout.encoding or "").lower()
    yes_mark = "[green]✓[/green]" if unicode_ok else "[green]yes[/green]"
    no_mark = "[red]✗[/red]" if unicode_ok else "[red]no[/red]"

    table = Table(title="Available Datasets")
    table.add_column("Fetch function")
    table.add_column("Size", justify="right")
    table.add_column("On disk", justify="center")

    for fetcher_name, size_bytes, _ in _REGISTRY:
        table.add_row(
            fetcher_name,
            _format_bytes(size_bytes),
            yes_mark if is_dataset_cached(fetcher_name, datasets_dir) else no_mark,
        )

    console.print(table)
