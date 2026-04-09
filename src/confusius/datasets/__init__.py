"""Datasets for confusius examples and tutorials."""

__all__ = [
    "fetch_nunez_elizalde_2022",
    "get_datasets_dir",
    "list_datasets",
]

from ._nunez_elizalde_2022 import _TOTAL_SIZE_BYTES as _nunez_elizalde_2022_size
from ._nunez_elizalde_2022 import fetch_nunez_elizalde_2022
from ._utils import format_bytes, get_datasets_dir

_REGISTRY: tuple[tuple[str, int], ...] = (
    ("fetch_nunez_elizalde_2022", _nunez_elizalde_2022_size),
)
"""Registry of (fetcher_name, total_size_bytes) for each dataset."""


def list_datasets() -> None:
    """Print a table of available datasets.

    Displays each fetch function and the total download size.
    """
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Available Datasets")
    table.add_column("Fetch function")
    table.add_column("Size", justify="right")

    for fetcher_name, size_bytes in _REGISTRY:
        table.add_row(fetcher_name, format_bytes(size_bytes))

    Console().print(table)
