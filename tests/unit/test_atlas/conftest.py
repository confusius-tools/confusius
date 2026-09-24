"""Fixtures for atlas unit tests."""

from __future__ import annotations

from pathlib import Path

import DracoPy  # ty: ignore[unresolved-import]
import meshio as mio
import numpy as np
import pytest
import xarray as xr
from brainglobe_atlasapi.structure_class import StructuresDict

from confusius.xarray import create_voxeldata


@pytest.fixture(scope="module")
def mock_mesh() -> mio.Mesh:
    """Mesh with six vertices (3 right, 3 left) and two triangle faces, in z/y/x µm.

    Vertices 1-3 lie at RL coordinate 50 µm (right hemisphere).
    Vertices 4-6 lie at RL coordinate 150 µm (left hemisphere).
    The atlas midline is at 100 µm (shape[2]=8, resolution=25 µm).
    """
    points = np.array(
        [
            [0.0, 0.0, 50.0],
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 150.0],
            [100.0, 0.0, 150.0],
            [0.0, 100.0, 150.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    return mio.Mesh(points=points, cells=[("triangle", faces)])


@pytest.fixture(scope="module")
def obj_path(tmp_path_factory: pytest.TempPathFactory, mock_mesh: mio.Mesh) -> Path:
    """Draco-encoded mesh file for region 997, matching `mock_mesh`'s geometry.

    BrainGlobe stores points as x/y/z nanometres and undoes that (nm -> µm, then
    XYZ -> ZYX) on read, so the encoded points are the inverse of `mock_mesh`.
    """
    raw_points_nm = mock_mesh.points[:, ::-1] * 1000.0
    raw_faces = mock_mesh.get_cells_type("triangle")[:, ::-1]
    mesh_dir = tmp_path_factory.mktemp("meshes")
    path = mesh_dir / "997"
    path.write_bytes(DracoPy.encode(raw_points_nm, raw_faces, preserve_order=True))
    return path


@pytest.fixture(scope="module")
def structure_list(obj_path: Path) -> list[dict]:
    """Flat BrainGlobe-style structures list: root(997) → child(10) → grandchild(20).

    Only the root region (997) has a mesh file (its absolute path, so `get_mesh` works
    without a BrainGlobe cache lookup).
    """
    return [
        {
            "id": 997,
            "acronym": "root",
            "name": "whole brain",
            "rgb_triplet": [200, 200, 200],
            "structure_id_path": [997],
            "mesh_filename": obj_path,
        },
        {
            "id": 10,
            "acronym": "ch",
            "name": "child region",
            "rgb_triplet": [255, 0, 0],
            "structure_id_path": [997, 10],
            "mesh_filename": None,
        },
        {
            "id": 20,
            "acronym": "gc",
            "name": "grandchild region",
            "rgb_triplet": [0, 255, 0],
            "structure_id_path": [997, 10, 20],
            "mesh_filename": None,
        },
    ]


@pytest.fixture(scope="module")
def mock_structures(structure_list: list[dict]) -> StructuresDict:
    """Real BrainGlobe StructuresDict built from the mock structures list (no network)."""
    return StructuresDict(structure_list)


@pytest.fixture(scope="module")
def atlas_ds(structure_list: list[dict], mock_structures: StructuresDict) -> xr.Dataset:
    """Atlas Dataset built from fully controlled mock data.

    Annotation (shape 4, 6, 8):
      - [:2, :, 2:6] = 10  (child)
      - [2:, :, 2:6] = 20  (grandchild)
      - elsewhere   = 0   (background)

    Hemispheres (shape 4, 6, 8), with attrs {"left": 1, "right": 2}:
      - [:, :, :2] = 2  (right — RL below the 0.1 mm mesh midline)
      - [:, :, 2:] = 1  (left  — RL at/above the 0.1 mm mesh midline)

    Resolution: 50 µm (0.05 mm) isotropic. The split matches the OBJ mesh's RL midline
    (0.1 mm), so sampling the map splits the mesh 3/3.
    """
    shape = (4, 6, 8)
    # 50 µm so the OBJ mesh (z up to 100 µm) stays inside the reference grid, which the
    # nonlinear get_mesh tests require.
    resolution_mm = 0.05

    annotation_data = np.zeros(shape, dtype=np.int32)
    annotation_data[:2, :, 2:6] = 10
    annotation_data[2:, :, 2:6] = 20

    hemispheres_data = np.zeros(shape, dtype=np.int8)
    hemispheres_data[:, :, :2] = 2  # right (RL < 0.1 mm mesh midline)
    hemispheres_data[:, :, 2:] = 1  # left  (RL >= 0.1 mm mesh midline)

    rgb_lookup: dict[int, list[int]] = {
        997: [200, 200, 200],
        10: [255, 0, 0],
        20: [0, 255, 0],
    }

    reference_da = create_voxeldata(
        np.ones(shape, dtype=np.float32),
        dims=["k", "j", "i"],
        spacing=(resolution_mm, resolution_mm, resolution_mm),
        origin=(0.0, 0.0, 0.0),
        attrs={"cmap": "gray"},
    )
    annotation_da = create_voxeldata(
        annotation_data,
        dims=["k", "j", "i"],
        spacing=(resolution_mm, resolution_mm, resolution_mm),
        origin=(0.0, 0.0, 0.0),
        attrs={"rgb_lookup": rgb_lookup},
    )
    hemispheres_da = create_voxeldata(
        hemispheres_data,
        dims=["k", "j", "i"],
        spacing=(resolution_mm, resolution_mm, resolution_mm),
        origin=(0.0, 0.0, 0.0),
        attrs={"left": 1, "right": 2},
    )

    return xr.Dataset(
        {
            "reference": reference_da,
            "annotation": annotation_da,
            "hemispheres": hemispheres_da,
        },
        attrs={
            "name": "mock_atlas",
            "citation": "Mock et al. (2026)",
            "species": "Mus musculus",
            "orientation": "asr",
            "structures": mock_structures,
            "world_to_base": np.eye(4),
        },
    )
