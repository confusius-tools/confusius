"""Shared fixtures for napari-related tests."""

from __future__ import annotations

import pytest

from confusius._napari._time_series._store import TimeSeriesStore


@pytest.fixture
def time_series_store():
    """Return a fresh TimeSeriesStore instance."""
    return TimeSeriesStore()


@pytest.fixture
def time_series_csv(tmp_path):
    """Return a path to a valid CSV time series file."""
    path = tmp_path / "series.csv"
    path.write_text("time,a,b\n0,1,4\n1,2,5\n2,3,6\n")
    return path


@pytest.fixture
def time_series_tsv(tmp_path):
    """Return a path to a valid TSV time series file."""
    path = tmp_path / "series.tsv"
    path.write_text("time\tregion\n0\t1.0\n0\t2.0\n1\t3.0\n")
    return path


class SignalSpy:
    """Spy object to count signal emissions."""

    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1


@pytest.fixture
def signal_spy():
    """Return a factory for signal spy objects."""
    return SignalSpy
