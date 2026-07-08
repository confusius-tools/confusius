"""Tests for the Atlas class.

Reference implementations are used for get_masks and get_mesh to avoid
testing against the implementation itself.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import confusius.atlas.atlas as atlas_module
from confusius.atlas import Atlas


class TestAtlasConstruction:
    """Tests for Atlas construction and from_brainglobe."""

    def test_reference_dtype(self, atlas: Atlas) -> None:
        assert atlas.reference.dtype == np.float32

    def test_annotation_dtype(self, atlas: Atlas) -> None:
        assert atlas.annotation.dtype == np.int32

    def test_hemispheres_dtype(self, atlas: Atlas) -> None:
        assert atlas.hemispheres.dtype == np.int8

    def test_dataarray_dims(self, atlas: Atlas) -> None:
        for da in [atlas.reference, atlas.annotation, atlas.hemispheres]:
            assert da.dims == ("z", "y", "x")

    def test_physical_coordinates_in_mm(self, atlas: Atlas) -> None:
        """Coordinate step must equal the voxdim attribute (in mm)."""
        for dim in ["z", "y", "x"]:
            coord = atlas.annotation.coords[dim]
            np.testing.assert_allclose(coord.values[1], coord.attrs["voxdim"])

    def test_invalid_mesh_vertex_transform_shape_raises(
        self, atlas: Atlas, mock_structures
    ) -> None:
        with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
            Atlas(atlas._dataset, mock_structures, np.eye(3), 0.1)

    def test_invalid_mesh_vertex_transform_type_raises(
        self, atlas: Atlas, mock_structures
    ) -> None:
        invalid = xr.DataArray(np.zeros((2, 2)), attrs={"type": "wrong"})
        with pytest.raises(ValueError, match="bspline_transform"):
            Atlas(atlas._dataset, mock_structures, invalid, 0.1)

    def test_from_brainglobe_accepts_instance(self, mock_structures) -> None:
        """from_brainglobe should accept any BrainGlobeAtlas-compatible object."""
        shape = (4, 6, 8)
        resolution_um = [25, 25, 25]

        class _MockBgAtlas:
            reference = np.ones(shape, dtype=np.uint16)
            annotation = np.zeros(shape, dtype=np.int32)
            hemispheres = np.zeros(shape, dtype=np.int8)
            structures = mock_structures
            metadata = {
                "name": "test_atlas",
                "species": "Mus musculus",
                "orientation": "asr",
                "shape": list(shape),
                "resolution": resolution_um,
            }

        result = Atlas.from_brainglobe(_MockBgAtlas())  # type: ignore[arg-type]

        assert isinstance(result, Atlas)
        assert result.reference.dtype == np.float32
        assert result.annotation.dtype == np.int32
        # Coordinates should be in mm: step = resolution_um[0] * 1e-3.
        expected_step = resolution_um[0] * 1e-3
        np.testing.assert_allclose(
            result.annotation.coords["z"].values[1], expected_step
        )

    def test_from_brainglobe_accepts_string(self, mock_structures, monkeypatch) -> None:
        shape = (4, 6, 8)
        resolution_um = [25, 25, 25]

        class _MockBgAtlas:
            def __init__(self, name: str, **kwargs) -> None:
                assert name == "test_atlas"
                assert kwargs == {"check_latest": False}
                self.reference = np.ones(shape, dtype=np.uint16)
                self.annotation = np.zeros(shape, dtype=np.int32)
                self.hemispheres = np.zeros(shape, dtype=np.int8)
                self.structures = mock_structures
                self.metadata = {
                    "name": "test_atlas",
                    "species": "Mus musculus",
                    "orientation": "asr",
                    "shape": list(shape),
                    "resolution": resolution_um,
                }

        monkeypatch.setitem(
            sys.modules,
            "brainglobe_atlasapi",
            types.SimpleNamespace(BrainGlobeAtlas=_MockBgAtlas),
        )

        result = Atlas.from_brainglobe("test_atlas", check_latest=False)
        assert isinstance(result, Atlas)


class TestAtlasProperties:
    """Tests for Atlas metadata properties."""

    def test_lookup_columns(self, atlas: Atlas) -> None:
        assert set(atlas.lookup.columns) >= {"acronym", "name", "rgb_triplet"}

    def test_lookup_index_matches_structure_ids(
        self, atlas: Atlas, mock_structures
    ) -> None:
        expected_ids = {sid for sid, _ in mock_structures.items()}
        assert set(atlas.lookup.index) == expected_ids

    def test_lookup_values_match_structures(
        self, atlas: Atlas, mock_structures
    ) -> None:
        df = atlas.lookup
        for sid, info in mock_structures.items():
            assert df.loc[sid, "acronym"] == info["acronym"]
            assert df.loc[sid, "name"] == info["name"]

    def test_lookup_is_cached(self, atlas: Atlas) -> None:
        """Accessing lookup twice must return the same DataFrame object."""
        assert atlas.lookup is atlas.lookup

    def test_norm_maps_background_below_range(self, atlas: Atlas) -> None:
        """Label 0 (background) must map below the colormap range.

        With clip=False, BoundaryNorm returns -1 for values below the first
        boundary, so background voxels are rendered with the under color.
        """
        assert atlas.norm(0) < 0

    def test_cmap_under_color_is_transparent(self, atlas: Atlas) -> None:
        """Under color must be fully transparent (RGBA = [0, 0, 0, 0])."""
        np.testing.assert_allclose(atlas.cmap.get_under(), [0, 0, 0, 0])

    def test_repr_contains_name_and_species(self, atlas: Atlas) -> None:
        r = repr(atlas)
        assert "mock_atlas" in r
        assert "Mus musculus" in r


class TestSearch:
    """Tests for Atlas.search, compared against direct DataFrame filtering."""

    def test_search_all_fields_substring(self, atlas: Atlas) -> None:
        """Substring 'child' should match both 'child region' and 'grandchild region'."""
        result = atlas.search("child")
        assert isinstance(result, pd.DataFrame)
        assert 10 in result.index
        assert 20 in result.index
        assert 997 not in result.index

    def test_search_acronym_exact_match(self, atlas: Atlas) -> None:
        result = atlas.search("gc", field="acronym")
        assert list(result.index) == [20]

    def test_search_name_is_case_insensitive(self, atlas: Atlas) -> None:
        result = atlas.search("Child Region", field="name")
        assert 10 in result.index
        assert 20 not in result.index

    def test_search_name_accepts_regex(self, atlas: Atlas) -> None:
        """Regex '.*child.*' should match both 'child region' and 'grandchild region'."""
        result = atlas.search(".*child.*", field="name")
        assert 10 in result.index
        assert 20 in result.index

    def test_search_no_match_returns_empty_dataframe(self, atlas: Atlas) -> None:
        result = atlas.search("no_such_region_xyz")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_search_result_is_subset_of_lookup(self, atlas: Atlas) -> None:
        """Search result must be a filtered view of atlas.lookup."""
        result = atlas.search("root", field="acronym")
        assert result.columns.tolist() == atlas.lookup.columns.tolist()
        assert 997 in result.index


class TestGetMasks:
    """Tests for Atlas.get_masks.

    The reference implementation is built directly from numpy operations on
    atlas.annotation and atlas.hemispheres to avoid circular reasoning.

    Fixture annotation layout (shape 4, 6, 8):
      - [:2, :, 2:6] = 10   (child, descendant of 997)
      - [2:, :, 2:6] = 20   (grandchild, descendant of 997 and 10)
      - elsewhere    = 0    (background)

    Hemisphere layout:
      - [:, :, :4] = 2   (right)
      - [:, :, 4:] = 1   (left)
    """

    def test_single_region_both_sides_vs_reference(self, atlas: Atlas) -> None:
        """Root mask must cover all descendant-labeled voxels."""
        ann = atlas.annotation.values
        expected = np.where(np.isin(ann, [997, 10, 20]), 997, 0).astype(np.int32)
        result = atlas.get_masks(997)
        np.testing.assert_array_equal(result.values[0], expected)

    def test_single_region_left_side_vs_reference(self, atlas: Atlas) -> None:
        ann = atlas.annotation.values
        hemi = atlas.hemispheres.values
        expected = np.where(np.isin(ann, [10, 20]) & (hemi == 1), 10, 0).astype(
            np.int32
        )
        result = atlas.get_masks(10, sides="left")
        np.testing.assert_array_equal(result.values[0], expected)

    def test_single_region_right_side_vs_reference(self, atlas: Atlas) -> None:
        ann = atlas.annotation.values
        hemi = atlas.hemispheres.values
        expected = np.where(np.isin(ann, [10, 20]) & (hemi == 2), 10, 0).astype(
            np.int32
        )
        result = atlas.get_masks(10, sides="right")
        np.testing.assert_array_equal(result.values[0], expected)

    def test_left_and_right_partition_both(self, atlas: Atlas) -> None:
        """Left + right masks must tile both-sides exactly with no overlap."""
        left = atlas.get_masks(10, sides="left").values[0]
        right = atlas.get_masks(10, sides="right").values[0]
        both = atlas.get_masks(10, sides="both").values[0]

        np.testing.assert_array_equal((left > 0) | (right > 0), both > 0)
        np.testing.assert_array_equal(
            (left > 0) & (right > 0), np.zeros_like(left, dtype=bool)
        )

    def test_descendant_voxels_included(self, atlas: Atlas) -> None:
        """get_masks(10) must include voxels labeled 20 (grandchild of 10)."""
        both = atlas.get_masks(10).values[0]
        ann = atlas.annotation.values
        assert np.any(both[ann == 20] == 10), (
            "grandchild voxels (label 20) should be included"
        )

    def test_multiple_regions_stacked_shape(self, atlas: Atlas) -> None:
        result = atlas.get_masks([10, 20])
        assert result.dims == ("mask", "z", "y", "x")
        assert result.sizes["mask"] == 2

    def test_multiple_regions_masks_coord_contains_acronyms(self, atlas: Atlas) -> None:
        result = atlas.get_masks([997, 10, 20])
        np.testing.assert_array_equal(
            result.coords["mask"].values, ["root", "ch", "gc"]
        )

    def test_per_region_sides(self, atlas: Atlas) -> None:
        """Per-element sides list must be applied independently per region."""
        result = atlas.get_masks([10, 10], sides=["left", "right"])
        left = result.isel(mask=0).values
        right = result.isel(mask=1).values
        both = atlas.get_masks(10).values[0]

        np.testing.assert_array_equal((left > 0) | (right > 0), both > 0)

    def test_per_region_sides_mask_coord_disambiguated(self, atlas: Atlas) -> None:
        """Same region with different sides must not share a `mask` coord value."""
        result = atlas.get_masks([10, 10], sides=["left", "right"])
        np.testing.assert_array_equal(result.coords["mask"].values, ["ch_L", "ch_R"])

    def test_both_side_mask_coord_is_bare_acronym(self, atlas: Atlas) -> None:
        """`sides="both"` (the default) must not suffix the acronym."""
        result = atlas.get_masks(10, sides="both")
        np.testing.assert_array_equal(result.coords["mask"].values, ["ch"])

    def test_str_acronym_gives_same_result_as_integer_id(self, atlas: Atlas) -> None:
        by_id = atlas.get_masks(10).values[0]
        by_acronym = atlas.get_masks("ch").values[0]
        np.testing.assert_array_equal(by_id, by_acronym)

    def test_spatial_coords_match_annotation(self, atlas: Atlas) -> None:
        result = atlas.get_masks(10)
        for dim in ["z", "y", "x"]:
            np.testing.assert_array_equal(
                result.coords[dim].values, atlas.annotation.coords[dim].values
            )

    def test_sides_length_mismatch_raises(self, atlas: Atlas) -> None:
        with pytest.raises(ValueError, match="same length"):
            atlas.get_masks([10, 20], sides=["left"])

    def test_invalid_side_value_raises(self, atlas: Atlas) -> None:
        with pytest.raises(ValueError, match="Invalid side"):
            atlas.get_masks(10, sides="center")  # type: ignore[arg-type]

    def test_unknown_region_id_raises(self, atlas: Atlas) -> None:
        with pytest.raises(KeyError):
            atlas.get_masks(9999)

    def test_unknown_region_acronym_raises(self, atlas: Atlas) -> None:
        with pytest.raises(KeyError):
            atlas.get_masks("NONEXISTENT")


class TestGetMesh:
    """Tests for Atlas.get_mesh.

    The OBJ mesh (from conftest) has:
      Vertices 0-2 at RL = 50 µm  → right hemisphere (< midline 100 µm)
      Vertices 3-5 at RL = 150 µm → left  hemisphere (≥ midline 100 µm)
      Face 0: triangle (0, 1, 2) — entirely right
      Face 1: triangle (3, 4, 5) — entirely left

    mesh_to_physical scales µm → mm (factor 1e-3).
    """

    def test_vertices_transformed_to_mm(self, atlas: Atlas) -> None:
        vertices_mm, _ = atlas.get_mesh(997, side="both")
        expected = np.array(
            [
                [0.0, 0.0, 0.05],
                [0.1, 0.0, 0.05],
                [0.0, 0.1, 0.05],
                [0.0, 0.0, 0.15],
                [0.1, 0.0, 0.15],
                [0.0, 0.1, 0.15],
            ]
        )
        np.testing.assert_allclose(vertices_mm, expected)

    def test_both_sides_returns_all_faces(self, atlas: Atlas) -> None:
        vertices, faces = atlas.get_mesh(997, side="both")
        assert len(vertices) == 6
        assert len(faces) == 2

    def test_right_side_clips_to_right_hemisphere(self, atlas: Atlas) -> None:
        vertices, faces = atlas.get_mesh(997, side="right")
        assert len(vertices) == 3
        assert len(faces) == 1
        np.testing.assert_array_equal(faces, [[0, 1, 2]])
        # All right-hemisphere vertices have RL < 0.1 mm.
        assert np.all(vertices[:, 2] < 0.1)

    def test_left_side_clips_to_left_hemisphere(self, atlas: Atlas) -> None:
        vertices, faces = atlas.get_mesh(997, side="left")
        assert len(vertices) == 3
        assert len(faces) == 1
        # Surviving vertex indices are reindexed starting from 0.
        np.testing.assert_array_equal(faces, [[0, 1, 2]])
        # All left-hemisphere vertices have RL ≥ 0.1 mm.
        assert np.all(vertices[:, 2] >= 0.1)

    def test_faces_dtype_is_int32(self, atlas: Atlas) -> None:
        _, faces = atlas.get_mesh(997)
        assert faces.dtype == np.int32

    def test_str_acronym_gives_same_result_as_integer_id(self, atlas: Atlas) -> None:
        vertices_id, faces_id = atlas.get_mesh(997)
        vertices_str, faces_str = atlas.get_mesh("root")
        np.testing.assert_array_equal(vertices_id, vertices_str)
        np.testing.assert_array_equal(faces_id, faces_str)

    def test_region_without_mesh_raises(self, atlas: Atlas) -> None:
        with pytest.raises(ValueError, match="No mesh file"):
            atlas.get_mesh(10)

    def test_unknown_region_raises(self, atlas: Atlas) -> None:
        with pytest.raises(KeyError):
            atlas.get_mesh(9999)

    def test_nonlinear_mesh_transform_crops_vertices_outside_domain(
        self, atlas: Atlas, mock_structures
    ) -> None:
        # The field domain (z, y, x in [0, 0.01]) excludes the whole mesh (vertices at
        # 0.05-0.15), so every vertex is dropped and the returned mesh is empty.
        transform = xr.DataArray(
            np.zeros((3, 2, 2, 2), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": np.arange(3),
                "z": [0.0, 0.01],
                "y": [0.0, 0.01],
                "x": [0.0, 0.01],
            },
            attrs={"type": "displacement_field_transform"},
        )
        nonlinear_atlas = Atlas(atlas._dataset, mock_structures, transform, 0.1)
        vertices, faces = nonlinear_atlas.get_mesh(997)
        assert vertices.shape == (0, 3)
        assert faces.shape == (0, 3)

    def test_bspline_transform_with_initialization_uses_inverse_seed(
        self, atlas: Atlas, mock_structures, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        field = xr.DataArray(
            np.zeros((3, 4, 6, 8), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": np.arange(3),
                "z": atlas.reference.coords["z"],
                "y": atlas.reference.coords["y"],
                "x": atlas.reference.coords["x"],
            },
            attrs={"type": "displacement_field_transform"},
        )
        monkeypatch.setattr(
            atlas_module,
            "sample_displacement_field_like",
            lambda *args, **kwargs: field,
        )

        bspline = xr.DataArray(
            np.zeros((3, 2, 2, 2), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": np.arange(3),
                "z": [0.0, 0.1],
                "y": [0.0, 0.1],
                "x": [0.0, 0.1],
            },
            attrs={
                "type": "bspline_transform",
                "order": 3,
                "direction": np.eye(3).tolist(),
                "affines": {"bspline_initialization": np.eye(4).tolist()},
            },
        )
        nonlinear_atlas = Atlas(atlas._dataset, mock_structures, bspline, 0.1)
        monkeypatch.setattr(
            atlas_module,
            "_interpolate_displacement_field",
            lambda field, points: np.zeros_like(points),
        )
        vertices, _ = nonlinear_atlas.get_mesh(997)
        expected, _ = atlas.get_mesh(997)
        np.testing.assert_allclose(vertices, expected)

    def test_displacement_inversion_can_return_last_iterate(
        self, atlas: Atlas, mock_structures, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transform = xr.DataArray(
            np.zeros((3, 4, 6, 8), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": np.arange(3),
                "z": atlas.reference.coords["z"],
                "y": atlas.reference.coords["y"],
                "x": atlas.reference.coords["x"],
            },
            attrs={"type": "displacement_field_transform"},
        )
        nonlinear_atlas = Atlas(atlas._dataset, mock_structures, transform, 0.1)

        state = {"flip": False}

        def _oscillate(field, points):
            state["flip"] = not state["flip"]
            sign = 1.0 if state["flip"] else -1.0
            return np.full_like(points, sign * 0.01)

        monkeypatch.setattr(atlas_module, "_interpolate_displacement_field", _oscillate)
        vertices, _ = nonlinear_atlas.get_mesh(997)
        # The oscillating field never converges, so the fixed-point solver returns its
        # last iterate rather than hanging; get_mesh completes with all six vertices,
        # which stay inside the grid and are finite.
        assert vertices.shape == (6, 3)
        assert np.isfinite(vertices).all()


class TestResampleLike:
    """Tests for Atlas.resample_like."""

    def test_accepts_displacement_field_and_warps_mesh(self, atlas: Atlas) -> None:
        reference = xr.DataArray(
            np.zeros((4, 3, 4), dtype=np.float32),
            dims=["z", "y", "x"],
            coords={
                "z": xr.Variable("z", [0.0, 0.05, 0.1, 0.15], attrs={"voxdim": 0.05}),
                "y": xr.Variable("y", [0.0, 0.05, 0.1], attrs={"voxdim": 0.05}),
                "x": xr.Variable("x", [0.0, 0.05, 0.1, 0.15], attrs={"voxdim": 0.05}),
            },
        )
        dims = list(reference.dims)
        coords = {dim: reference.coords[dim] for dim in dims}

        field = xr.DataArray(
            np.zeros((3, *reference.shape), dtype=np.float64),
            dims=["component", *dims],
            coords={"component": np.arange(3), **coords},
            attrs={"type": "displacement_field_transform"},
        )
        field.loc[dict(component=2)] = 0.01

        resampled = atlas.resample_like(reference, field)
        vertices_mm, _ = resampled.get_mesh(997)

        # component 2 is the x-displacement (dim order (z, y, x)); the +0.01 pull
        # inverts to a -0.01 shift of the mesh x column (col 2), leaving z (col 0) and
        # y (col 1) untouched. Vertices stay inside the field domain (0.05 -> 0.04,
        # 0.15 -> 0.14).
        expected = np.array(
            [
                [0.0, 0.0, 0.04],
                [0.1, 0.0, 0.04],
                [0.0, 0.1, 0.04],
                [0.0, 0.0, 0.14],
                [0.1, 0.0, 0.14],
                [0.0, 0.1, 0.14],
            ]
        )
        np.testing.assert_allclose(vertices_mm, expected, atol=1e-6)

    def test_bspline_transform_uses_same_mesh_warp_path(
        self, atlas: Atlas, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reference = xr.DataArray(
            np.zeros((4, 3, 4), dtype=np.float32),
            dims=["z", "y", "x"],
            coords={
                "z": xr.Variable("z", [0.0, 0.05, 0.1, 0.15], attrs={"voxdim": 0.05}),
                "y": xr.Variable("y", [0.0, 0.05, 0.1], attrs={"voxdim": 0.05}),
                "x": xr.Variable("x", [0.0, 0.05, 0.1, 0.15], attrs={"voxdim": 0.05}),
            },
        )
        dims = list(reference.dims)
        coords = {dim: reference.coords[dim] for dim in dims}
        field = xr.DataArray(
            np.zeros((3, *reference.shape), dtype=np.float64),
            dims=["component", *dims],
            coords={"component": np.arange(3), **coords},
            attrs={"type": "displacement_field_transform"},
        )
        field.loc[dict(component=2)] = 0.01

        def _fake_sample_displacement_field_like(transform, reference, **kwargs):
            return field

        monkeypatch.setattr(
            "confusius.atlas.atlas.sample_displacement_field_like",
            _fake_sample_displacement_field_like,
        )
        monkeypatch.setattr(
            "confusius.atlas.atlas.resample_like_da",
            lambda moving, reference, transform, **kwargs: moving.copy(deep=True),
        )

        fake_bspline = xr.DataArray(
            np.zeros((3, 2, 2, 2), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": np.arange(3),
                "z": [0.0, 0.1],
                "y": [0.0, 0.1],
                "x": [0.0, 0.1],
            },
            attrs={
                "type": "bspline_transform",
                "order": 3,
                "direction": np.eye(3).tolist(),
            },
        )

        resampled = atlas.resample_like(reference, fake_bspline)
        vertices_mm, _ = resampled.get_mesh(997)

        # component 2 is the x-displacement (dim order (z, y, x)); the +0.01 pull
        # inverts to a -0.01 shift of the mesh x column (col 2), leaving z (col 0) and
        # y (col 1) untouched. Vertices stay inside the field domain (0.05 -> 0.04,
        # 0.15 -> 0.14).
        expected = np.array(
            [
                [0.0, 0.0, 0.04],
                [0.1, 0.0, 0.04],
                [0.0, 0.1, 0.04],
                [0.0, 0.0, 0.14],
                [0.1, 0.0, 0.14],
                [0.0, 0.1, 0.14],
            ]
        )
        np.testing.assert_allclose(vertices_mm, expected, atol=1e-6)

    def test_compose_mesh_vertex_transforms_affine_affine(self) -> None:
        old = np.array(
            [
                [1.0, 0.0, 0.0, 0.02],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        new = np.array(
            [
                [1.0, 0.0, 0.0, -0.01],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        reference = xr.DataArray(
            np.zeros((2, 2, 2)),
            dims=["z", "y", "x"],
            coords={"z": [0.0, 1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
        result = atlas_module._compose_mesh_vertex_transforms(
            old, new, reference, reference
        )
        np.testing.assert_allclose(result, old @ new)

    def test_invert_displacement_field_at_points_empty_returns_empty(self) -> None:
        field = xr.DataArray(
            np.zeros((3, 2, 2, 2), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": ["z", "y", "x"],
                "z": [0.0, 1.0],
                "y": [0.0, 1.0],
                "x": [0.0, 1.0],
            },
            attrs={"type": "displacement_field_transform"},
        )
        points = np.empty((0, 3), dtype=np.float64)

        result = atlas_module._invert_displacement_field_at_points(field, points)

        assert result.shape == (0, 3)
        assert result.dtype == np.float64
        assert result is not points

    def test_transform_points_affine(self) -> None:
        reference = xr.DataArray(
            np.zeros((2, 2, 2)),
            dims=["z", "y", "x"],
            coords={"z": [0.0, 1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
        transform = np.array(
            [
                [1.0, 0.0, 0.0, 0.01],
                [0.0, 1.0, 0.0, 0.02],
                [0.0, 0.0, 1.0, 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        points = np.array([[0.0, 0.0, 0.0]])
        result = atlas_module._transform_points(transform, points, reference)
        np.testing.assert_allclose(result, [[0.01, 0.02, 0.03]])

    def test_transform_points_bspline_samples_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reference = xr.DataArray(
            np.zeros((2, 2, 2)),
            dims=["z", "y", "x"],
            coords={"z": [0.0, 1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
        field = xr.DataArray(
            np.zeros((3, 2, 2, 2), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": np.arange(3),
                "z": [0.0, 1.0],
                "y": [0.0, 1.0],
                "x": [0.0, 1.0],
            },
            attrs={"type": "displacement_field_transform"},
        )
        field.loc[dict(component=2)] = 0.01
        monkeypatch.setattr(
            atlas_module,
            "sample_displacement_field_like",
            lambda *args, **kwargs: field,
        )
        bspline = xr.DataArray(
            np.zeros((3, 2, 2, 2), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": np.arange(3),
                "z": [0.0, 1.0],
                "y": [0.0, 1.0],
                "x": [0.0, 1.0],
            },
            attrs={
                "type": "bspline_transform",
                "order": 3,
                "direction": np.eye(3).tolist(),
            },
        )
        points = np.array([[0.0, 0.0, 0.0]])
        result = atlas_module._transform_points(bspline, points, reference)
        # component 2 is the x-displacement (dim order (z, y, x)), so it lands in x.
        np.testing.assert_allclose(result, [[0.0, 0.0, 0.01]])

    def test_compose_general_path_preserves_component_axis_order(
        self, atlas: Atlas
    ) -> None:
        """Composing through the general path must not swap displacement axes.

        `_compose_mesh_vertex_transforms` and the field consumers all work in DataArray
        dim order `(z, y, x)`, where the `component` labels match those spatial dims.
        Composing two transforms and applying the result must equal applying them in
        sequence; a stray axis reversal in the compose path would break that on
        per-axis-different displacements (a symmetric field would hide it).
        """
        reference = xr.DataArray(
            np.zeros((2, 2, 2)),
            dims=["z", "y", "x"],
            coords={"z": [0.0, 1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
        )

        # A constant displacement field translating +0.3 along z and +0.1 along x.
        new_transform = xr.DataArray(
            np.zeros((3, 2, 2, 2), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": np.arange(3),
                "z": [0.0, 1.0],
                "y": [0.0, 1.0],
                "x": [0.0, 1.0],
            },
            attrs={"type": "displacement_field_transform"},
        )
        new_transform.loc[dict(component=0)] = 0.3  # Displacement along dim z.
        new_transform.loc[dict(component=2)] = 0.1  # Displacement along dim x.

        # An affine translating +0.05 along z and +0.2 along x (dim order (z, y, x)).
        old_transform = np.eye(4)
        old_transform[0, 3] = 0.05  # z.
        old_transform[2, 3] = 0.2  # x.

        composed = atlas_module._compose_mesh_vertex_transforms(
            old_transform, new_transform, reference, reference
        )
        np.testing.assert_array_equal(
            composed.coords["component"].values, ["z", "y", "x"]
        )

        points = np.array([[0.5, 0.5, 0.5], [0.2, 0.8, 0.3]])  # (z, y, x).
        result = atlas_module._transform_points(composed, points, reference)
        # Composition semantics: applying the composed transform equals applying
        # new then old in sequence.
        expected = atlas_module._transform_points(
            old_transform,
            atlas_module._transform_points(new_transform, points, reference),
            reference,
        )
        np.testing.assert_allclose(result, expected)

    def test_resample_like_keeps_existing_nonlinear_transform_on_identity_affine(
        self, atlas: Atlas, mock_structures, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transform = xr.DataArray(
            np.zeros((3, 4, 6, 8), dtype=np.float64),
            dims=["component", "z", "y", "x"],
            coords={
                "component": np.arange(3),
                "z": atlas.reference.coords["z"],
                "y": atlas.reference.coords["y"],
                "x": atlas.reference.coords["x"],
            },
            attrs={"type": "displacement_field_transform"},
        )
        nonlinear_atlas = Atlas(atlas._dataset, mock_structures, transform, 0.1)
        monkeypatch.setattr(
            atlas_module,
            "resample_like_da",
            lambda moving, reference, transform, **kwargs: moving.copy(deep=True),
        )
        monkeypatch.setattr(
            atlas_module,
            "_interpolate_displacement_field",
            lambda field, points: np.zeros_like(points),
        )
        result = nonlinear_atlas.resample_like(atlas.reference, np.eye(4))
        vertices, _ = result.get_mesh(997)
        expected, _ = nonlinear_atlas.get_mesh(997)
        np.testing.assert_allclose(vertices, expected)

    def test_resample_matches_resample_like(self, atlas: Atlas) -> None:
        reference = xr.DataArray(
            np.zeros((4, 3, 4), dtype=np.float32),
            dims=["z", "y", "x"],
            coords={
                "z": xr.Variable("z", [0.0, 0.05, 0.1, 0.15], attrs={"voxdim": 0.05}),
                "y": xr.Variable("y", [0.0, 0.05, 0.1], attrs={"voxdim": 0.05}),
                "x": xr.Variable("x", [0.0, 0.05, 0.1, 0.15], attrs={"voxdim": 0.05}),
            },
        )

        by_like = atlas.resample_like(reference, np.eye(4))
        by_grid = atlas.resample(
            np.eye(4),
            shape=reference.shape,
            spacing=[0.05, 0.05, 0.05],
            origin=[0.0, 0.0, 0.0],
            dims=reference.dims,
        )

        np.testing.assert_allclose(by_grid.reference.values, by_like.reference.values)
        np.testing.assert_array_equal(
            by_grid.annotation.values, by_like.annotation.values
        )
        np.testing.assert_array_equal(
            by_grid.hemispheres.values, by_like.hemispheres.values
        )


class TestAncestors:
    """Tests for Atlas.ancestors, compared against direct treelib traversal."""

    def test_show_tree_prints(
        self, atlas: Atlas, capsys: pytest.CaptureFixture[str]
    ) -> None:
        atlas.show_tree()
        captured = capsys.readouterr()
        assert "root" in captured.out

    def test_root_has_no_ancestors(self, atlas: Atlas) -> None:
        assert atlas.ancestors(997) == []

    def test_child_has_root_as_sole_ancestor(self, atlas: Atlas) -> None:
        result = atlas.ancestors(10)
        assert len(result) == 1
        assert result[0].identifier == 997

    def test_grandchild_ancestors_ordered_root_first(self, atlas: Atlas) -> None:
        """Ancestors must be ordered from root toward the target node."""
        result = atlas.ancestors(20)
        assert len(result) == 2
        assert result[0].identifier == 997
        assert result[1].identifier == 10

    def test_str_acronym_gives_same_result_as_integer_id(self, atlas: Atlas) -> None:
        by_id = [n.identifier for n in atlas.ancestors(20)]
        by_acronym = [n.identifier for n in atlas.ancestors("gc")]
        assert by_id == by_acronym

    def test_unknown_region_raises(self, atlas: Atlas) -> None:
        with pytest.raises(KeyError):
            atlas.ancestors(9999)
