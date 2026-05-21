"""Unit tests for confusius.xarray public module helpers."""

import pytest

import confusius.xarray as cxr


def test_dir_includes_public_names():
    """__dir__ exposes public API names declared in __all__."""
    names = dir(cxr)
    assert "FUSIAccessor" in names
    assert "db_scale" in names


def test_getattr_valid_name_lazy_loads_symbol():
    """Known lazy attribute names are resolved from their target module."""
    assert callable(getattr(cxr, "db_scale"))


def test_getattr_invalid_name_raises_attribute_error():
    """Unknown lazy attribute names raise AttributeError."""
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(cxr, "definitely_not_a_public_name")
