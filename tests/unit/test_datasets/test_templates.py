"""Public template fetchers preserve image geometry, sidecars, and citations."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import confusius as cf


@pytest.mark.parametrize(
    "fetcher,name,filename,coordinate_affine",
    [
        (
            cf.datasets.fetch_template_huang_2025,
            "huang-2025-template",
            "huang-2025-space-allen50_desc-vascular.nii.gz",
            "sform",
        ),
        (
            cf.datasets.fetch_template_pepe_mariani_2026,
            "pepe-mariani-2026-template",
            "pepe-mariani-2026-fusi-template.nii.gz",
            "sform",
        ),
    ],
)
def test_fetch_template(
    tmp_path,
    release_server,
    publish_release,
    capsys,
    fetcher,
    name,
    filename,
    coordinate_affine,
):
    source = tmp_path / filename
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    image = nib.Nifti1Image(values, np.diag([0.1, 0.2, 0.3, 1.0]))
    image.set_qform(
        np.diag([0.4, 0.5, 0.6, 1.0]),
        code=0 if name == "pepe-mariani-2026-template" else 1,
    )
    sform = np.diag([0.1, 0.2, 0.3, 1.0])
    sform[0, 1] = 0.02
    image.set_sform(sform, code=1)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, source)
    sidecar = filename.removesuffix(".nii.gz") + ".json"
    publish_release(
        "templates",
        name,
        "1.0.0",
        {
            filename: source.read_bytes(),
            sidecar: b"{}",
        },
    )
    expected = cf.load(source, coordinate_affine=coordinate_affine)
    actual = fetcher(data_dir=tmp_path / "cache")
    assert_array_equal(actual.values, expected.values)
    assert_allclose(
        actual.fusi.affine.voxel_to_world, expected.fusi.affine.voxel_to_world
    )
    assert actual.attrs["citation"]
    assert "[italic]" not in actual.attrs["citation"]
    assert "If you use this template" in capsys.readouterr().out
    cached = tmp_path / "cache" / name / "1.0.0"
    assert (cached / filename).read_bytes() == source.read_bytes()
    assert (cached / sidecar).read_bytes() == b"{}"
    _, calls = release_server
    calls.clear()
    again = fetcher(data_dir=tmp_path / "cache", print_citation=False)
    assert_array_equal(again.values, expected.values)
    assert calls == []
    assert capsys.readouterr().out == ""
