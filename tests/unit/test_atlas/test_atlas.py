"""Tests for the Atlas class.

Reference implementations are used for get_masks and get_mesh to avoid
testing against the implementation itself.
"""

import numpy as np
import pandas as pd
import pytest

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


class TestAncestors:
    """Tests for Atlas.ancestors, compared against direct treelib traversal."""

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
