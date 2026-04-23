"""Datasets for confusius examples and tutorials."""

from __future__ import annotations

from ._cybis_pereira_2026 import fetch_cybis_pereira_2026
from ._nunez_elizalde_2022 import fetch_nunez_elizalde_2022
from ._registry import list_datasets
from ._utils import get_datasets_dir

__all__ = [
    "fetch_cybis_pereira_2026",
    "fetch_nunez_elizalde_2022",
    "get_datasets_dir",
    "list_datasets",
]
