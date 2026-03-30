"""Unit tests for the _TimeOverlay class."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from confusius._napari._time_overlay import _TimeOverlay


@pytest.fixture
def nifti_4d_ms(tmp_path: Path) -> Path:
    """4D NIfTI file with time units set to milliseconds."""
    shape = (4, 6, 8, 10)  # x, y, z, time (NIfTI convention)
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    affine = np.diag([0.1, 0.1, 0.2, 1.0])
    img = nib.Nifti1Image(data, affine)
    # NIfTI xyzt_units encodes spatial + time units in a single byte.
    # spatial=mm (0x02), time=msec (0x10) → 0x12 = 18.
    img.header.set_xyzt_units(xyz="mm", t="msec")
    img.header["pixdim"][4] = 50.0  # 50 ms per frame
    path = tmp_path / "vol_ms.nii.gz"
    img.to_filename(path)
    return path


class TestTimeOverlay:
    """_TimeOverlay reads time units from layer xarray metadata."""

    def test_reads_units_from_nifti_layer(self, nifti_4d_ms: Path, make_napari_viewer):
        """Units are correctly read as 'ms' from a 4D NIfTI loaded via the reader."""
        from confusius._napari._io._readers import read_nifti

        viewer = make_napari_viewer()
        overlay = _TimeOverlay(viewer)

        reader = read_nifti(str(nifti_4d_ms))
        assert reader is not None
        layer_data_list = reader(str(nifti_4d_ms))
        data, kwargs, layer_type = layer_data_list[0]
        viewer._add_layer_from_data(data, kwargs, layer_type)

        # After adding a 4D layer, time should be a slider (not displayed).
        overlay.check()
        assert overlay._active
        assert overlay._time_idx is not None
        assert overlay._units == "ms"
        assert "ms" in viewer.text_overlay.text
