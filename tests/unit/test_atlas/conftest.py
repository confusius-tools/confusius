"""Fixtures for Atlas unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import treelib
import xarray as xr

from confusius.atlas import Atlas


class _MockStructuresDict:
    """Minimal duck-type of StructuresDict for testing without BrainGlobe data."""

    def __init__(self, structure_list: list[dict], tree: treelib.Tree) -> None:
        self._data = {s["id"]: s for s in structure_list}
        self.tree = tree

    def __getitem__(self, key: int) -> dict:  # type: ignore[override]
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def items(self):  # type: ignore[override]
        return self._data.items()


@pytest.fixture(scope="module")
def obj_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """OBJ file with six vertices (3 right, 3 left) and two triangle faces.

    Vertices 1-3 lie at RL coordinate 50 µm (right hemisphere).
    Vertices 4-6 lie at RL coordinate 150 µm (left hemisphere).
    The atlas midline is at 100 µm (shape[2]=8, resolution=25 µm).
    """
    mesh_dir = tmp_path_factory.mktemp("meshes")
    path = mesh_dir / "997.obj"
    path.write_text(
        "v 0.0 0.0 50.0\n"
        "v 100.0 0.0 50.0\n"
        "v 0.0 100.0 50.0\n"
        "v 0.0 0.0 150.0\n"
        "v 100.0 0.0 150.0\n"
        "v 0.0 100.0 150.0\n"
        "f 1 2 3\n"
        "f 4 5 6\n"
    )
    return path


@pytest.fixture(scope="module")
def mock_structures(obj_path: Path) -> _MockStructuresDict:
    """Three-node structure tree: root(997) → child(10) → grandchild(20).

    Only the root region (997) has a mesh file assigned.
    """
    tree = treelib.Tree()
    tree.create_node("root", 997)
    tree.create_node("child", 10, parent=997)
    tree.create_node("grandchild", 20, parent=10)

    structure_list = [
        {
            "id": 997,
            "acronym": "root",
            "name": "whole brain",
            "rgb_triplet": [200, 200, 200],
            "mesh_filename": obj_path,
        },
        {
            "id": 10,
            "acronym": "ch",
            "name": "child region",
            "rgb_triplet": [255, 0, 0],
            "mesh_filename": None,
        },
        {
            "id": 20,
            "acronym": "gc",
            "name": "grandchild region",
            "rgb_triplet": [0, 255, 0],
            "mesh_filename": None,
        },
    ]
    return _MockStructuresDict(structure_list, tree)


@pytest.fixture(scope="module")
def atlas(mock_structures: _MockStructuresDict) -> Atlas:
    """Atlas built from fully controlled mock data.

    Annotation (shape 4, 6, 8):
      - [:2, :, 2:6] = 10  (child)
      - [2:, :, 2:6] = 20  (grandchild)
      - elsewhere   = 0   (background)

    Hemispheres (shape 4, 6, 8):
      - [:, :, :4] = 2  (right — low RL in asr orientation)
      - [:, :, 4:] = 1  (left  — high RL in asr orientation)

    Resolution: 25 µm (0.025 mm) isotropic.
    RL midline: shape[2]/2 * 25 µm = 100 µm.
    """
    shape = (4, 6, 8)
    resolution_mm = 0.025

    annotation_data = np.zeros(shape, dtype=np.int32)
    annotation_data[:2, :, 2:6] = 10
    annotation_data[2:, :, 2:6] = 20

    hemispheres_data = np.zeros(shape, dtype=np.int8)
    hemispheres_data[:, :, :4] = 2  # right
    hemispheres_data[:, :, 4:] = 1  # left

    coords = {
        dim: (
            np.arange(shape[i]) * resolution_mm,
            {"voxdim": resolution_mm, "units": "mm"},
        )
        for i, dim in enumerate(["z", "y", "x"])
    }

    rgb_lookup: dict[int, list[int]] = {
        997: [200, 200, 200],
        10: [255, 0, 0],
        20: [0, 255, 0],
    }

    reference_da = xr.DataArray(
        np.ones(shape, dtype=np.float32),
        dims=["z", "y", "x"],
        coords={d: xr.Variable(d, v, attrs=a) for d, (v, a) in coords.items()},
        attrs={"cmap": "gray"},
    )
    annotation_da = xr.DataArray(
        annotation_data,
        dims=["z", "y", "x"],
        coords={d: xr.Variable(d, v, attrs=a) for d, (v, a) in coords.items()},
        attrs={"rgb_lookup": rgb_lookup},
    )
    hemispheres_da = xr.DataArray(
        hemispheres_data,
        dims=["z", "y", "x"],
        coords={d: xr.Variable(d, v, attrs=a) for d, (v, a) in coords.items()},
    )

    dataset = xr.Dataset(
        {
            "reference": reference_da,
            "annotation": annotation_da,
            "hemispheres": hemispheres_da,
        },
        attrs={"name": "mock_atlas", "species": "Mus musculus", "orientation": "asr"},
    )

    mesh_to_physical = np.diag([1e-3, 1e-3, 1e-3, 1.0])
    # shape[2]=8, resolution=25 µm → midline = 8/2 * 25 = 100 µm.
    rl_midline_um = shape[2] / 2 * 25.0

    return Atlas(dataset, mock_structures, mesh_to_physical, rl_midline_um)
