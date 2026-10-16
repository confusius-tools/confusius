"""Unit tests for the confusius._utils.templates fUSI template registry."""

import pytest

from confusius import datasets
from confusius._utils.templates import FUSI_TEMPLATES, fetch_template


def test_every_entry_names_a_datasets_fetcher():
    """Each template's fetcher resolves to a callable on `confusius.datasets`."""
    for name, spec in FUSI_TEMPLATES.items():
        assert spec["fetcher"] == f"fetch_template_{name}"
        assert callable(getattr(datasets, spec["fetcher"]))


def test_fetch_template_resolves_fetcher_and_forwards_kwargs(
    monkeypatch, sample_voxeldata_3d
):
    """fetch_template calls the named fetcher with the given keyword arguments."""
    calls = []

    def fake_fetcher(**kwargs):
        calls.append(kwargs)
        return sample_voxeldata_3d

    monkeypatch.setattr(datasets, "fetch_template_brunner_grillet_2021", fake_fetcher)

    result = fetch_template("brunner_grillet_2021", print_citation=False, refresh=True)

    assert result is sample_voxeldata_3d
    assert calls == [{"print_citation": False, "refresh": True}]


def test_fetch_template_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown fUSI template 'nope'"):
        fetch_template("nope")
