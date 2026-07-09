"""Unit tests for confusius.datasets._huang_2025."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from confusius.datasets import fetch_template_huang_2025
from confusius.datasets._huang_2025 import (
    _CITATION,
    _FILENAME,
    _OSF_PROJECT_ID,
    _TEMPLATE_ROOT,
    resolve_template_url,
)
from confusius.datasets._utils import plain_citation


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_resolve_template_url_returns_download_link() -> None:
    payload = {
        "data": [
            {
                "attributes": {"name": _FILENAME},
                "links": {"download": "https://files.osf.io/template"},
            }
        ]
    }
    with patch(
        "confusius.datasets._huang_2025.requests.get",
        return_value=_Response(payload),
    ) as mock_get:
        assert resolve_template_url() == "https://files.osf.io/template"

    mock_get.assert_called_once_with(
        f"https://api.osf.io/v2/nodes/{_OSF_PROJECT_ID}/files/osfstorage/"
    )


def test_resolve_template_url_raises_when_missing() -> None:
    with patch(
        "confusius.datasets._huang_2025.requests.get",
        return_value=_Response({"data": []}),
    ):
        with pytest.raises(RuntimeError, match=_FILENAME):
            resolve_template_url()


@pytest.fixture
def mock_resolve() -> object:
    with patch(
        "confusius.datasets._huang_2025.resolve_template_url",
        return_value="https://files.osf.io/template",
    ) as mock:
        yield mock


@pytest.fixture
def mock_retrieve(tmp_path: Path):
    def _retrieve(url, dest, logger, progressbar=False, on_retry=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    with patch(
        "confusius.datasets._huang_2025.retrieve_with_retries",
        side_effect=_retrieve,
    ) as mock:
        yield mock


@pytest.fixture
def mock_load():
    # Stand-in for the loaded DataArray; needs a mutable `attrs` so the fetcher
    # can stamp the citation onto it.
    sentinel = SimpleNamespace(attrs={})
    with patch("confusius.datasets._huang_2025.load", return_value=sentinel) as mock:
        yield sentinel, mock


def test_fetch_downloads_missing_template(
    tmp_path, mock_resolve, mock_retrieve, mock_load
):
    sentinel, mock_load_fn = mock_load
    result = fetch_template_huang_2025(data_dir=tmp_path)
    dest = tmp_path / _TEMPLATE_ROOT / _FILENAME

    assert result is sentinel
    mock_resolve.assert_called_once_with()
    mock_retrieve.assert_called_once()
    mock_load_fn.assert_called_once_with(dest)
    assert dest.exists()


def test_fetch_skips_download_when_cached(
    tmp_path, mock_resolve, mock_retrieve, mock_load
):
    sentinel, mock_load_fn = mock_load
    dest = tmp_path / _TEMPLATE_ROOT / _FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.touch()

    result = fetch_template_huang_2025(data_dir=tmp_path)

    assert result is sentinel
    mock_resolve.assert_not_called()
    mock_retrieve.assert_not_called()
    mock_load_fn.assert_called_once_with(dest)


def test_fetch_refresh_redownloads_cached_template(
    tmp_path, mock_resolve, mock_retrieve, mock_load
):
    sentinel, mock_load_fn = mock_load
    dest = tmp_path / _TEMPLATE_ROOT / _FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("old")

    result = fetch_template_huang_2025(data_dir=tmp_path, refresh=True)

    assert result is sentinel
    mock_resolve.assert_called_once_with()
    mock_retrieve.assert_called_once()
    mock_load_fn.assert_called_once_with(dest)
    assert dest.exists()


def test_fetch_sets_citation(tmp_path, mock_resolve, mock_retrieve, mock_load, capsys):
    result = fetch_template_huang_2025(data_dir=tmp_path)

    # The attribute holds the plain citation, with rich markup stripped.
    assert result.attrs["citation"] == plain_citation(_CITATION)
    assert "[italic]" not in result.attrs["citation"]

    out = capsys.readouterr().out
    assert out.startswith(
        "If you use this template in your work, please cite the following source:"
    )
    # Rich word-wraps the printed citation, so compare on whitespace-normalized text.
    assert plain_citation(_CITATION) in " ".join(out.split())

    fetch_template_huang_2025(data_dir=tmp_path, print_citation=False)
    assert capsys.readouterr().out == ""
