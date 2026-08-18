"""Tests for registration transform DataArray validation."""

import numpy as np
import pytest

from confusius.validation import validate_bspline, validate_displacement_field
from confusius.xarray import create_fusi_dataarray


def _make_field(n_components: int, transform_type: str):
    """Build a minimal (component, k, j, i) transform DataArray."""
    data = create_fusi_dataarray(
        np.zeros((n_components, 2, 3, 4), dtype=np.float64),
        dims=("component", "k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
    )
    data.attrs["type"] = transform_type
    data.attrs["transform_type"] = transform_type
    data.attrs["order"] = 3
    return data


class TestValidateBspline:
    """Tests for validate_bspline."""

    def test_component_count_mismatch_raises(self):
        """A component count not matching the spatial dimensionality is rejected."""
        bad = _make_field(2, "bspline_transform")

        with pytest.raises(ValueError, match="component count must match"):
            validate_bspline(bad)


class TestValidateDisplacementField:
    """Tests for validate_displacement_field."""

    def test_component_count_mismatch_raises(self):
        """A component count not matching the spatial dimensionality is rejected."""
        bad = _make_field(2, "displacement_field_transform")

        with pytest.raises(ValueError, match="component count must match"):
            validate_displacement_field(bad)
