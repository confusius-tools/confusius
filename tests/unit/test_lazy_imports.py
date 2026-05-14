"""Tests for SPEC-0001 style lazy imports."""

import importlib

import pytest

import confusius
import confusius.xarray as confusius_xarray


def test_confusius_lazy_submodule_and_function_exports():
    """Top-level package resolves submodules and functions lazily."""
    module = importlib.reload(confusius)

    assert "atlas" not in module.__dict__
    assert "load" not in module.__dict__

    assert module.atlas.__name__ == "confusius.atlas"
    assert callable(module.load)


def test_confusius_dir_and_missing_attribute():
    """Top-level package exposes lazy names in `dir` and errors cleanly."""
    module = importlib.reload(confusius)

    exported = dir(module)
    assert "atlas" in exported
    assert "load" in exported
    assert "xarray" in exported

    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(module, "does_not_exist")


def test_confusius_xarray_lazy_exports():
    """xarray package resolves exported helpers lazily."""
    module = importlib.reload(confusius_xarray)

    assert "db_scale" not in module.__dict__
    assert callable(module.db_scale)


def test_confusius_xarray_dir_and_missing_attribute():
    """xarray package exposes lazy names in `dir` and errors cleanly."""
    module = importlib.reload(confusius_xarray)

    exported = dir(module)
    assert "db_scale" in exported
    assert "FUSIAccessor" in exported

    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(module, "does_not_exist")
