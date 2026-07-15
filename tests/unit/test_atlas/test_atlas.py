"""Tests for the atlas Dataset builder, `.atlas` accessor, and IO.

Reference implementations are used for get_masks and get_mesh to avoid
testing against the implementation itself.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from brainglobe_atlasapi.structure_class import StructuresDict

import confusius as cf
from confusius.atlas import atlas_from_brainglobe
from confusius.io import load_atlas, save_atlas
from confusius.io.atlas import structures_from_json, structures_to_json
from confusius.validation import validate_atlas_dataset

from .conftest import _MockBgAtlas


def _field_on_grid(reference: xr.DataArray, data: np.ndarray) -> xr.DataArray:
    """Build a displacement-field transform DataArray on `reference`'s grid."""
    dims = [str(d) for d in reference.dims]
    return xr.DataArray(
        data,
        dims=["component", *dims],
        coords={
            "component": np.array(dims, dtype=np.str_),
            **{d: reference.coords[d] for d in dims},
        },
        attrs={"type": "displacement_field_transform"},
    )


def _with_physical_to_base_transform(
    atlas_ds: xr.Dataset, transform: xr.DataArray
) -> xr.Dataset:
    """Return a copy of `atlas_ds` carrying `transform` as its physical_to_base transform."""
    ds = atlas_ds.copy()
    ds.attrs = {**atlas_ds.attrs, "physical_to_base": transform}
    return ds


class TestBuilder:
    """Tests for atlas_from_brainglobe and the Dataset schema."""

    def test_reference_dtype(self, atlas_ds: xr.Dataset) -> None:
        assert atlas_ds.atlas.reference.dtype == np.float32

    def test_annotation_dtype(self, atlas_ds: xr.Dataset) -> None:
        assert atlas_ds.atlas.annotation.dtype == np.int32

    def test_hemispheres_dtype(self, atlas_ds: xr.Dataset) -> None:
        assert atlas_ds.atlas.hemispheres.dtype == np.int8

    def test_hemispheres_is_data_var(self, atlas_ds: xr.Dataset) -> None:
        """hemispheres must be a data variable, not a coordinate.

        As a coordinate it would ride along on `reference`/`annotation` and be silently
        linear-interpolated (to fractional labels) by regridding ops such as resampling
        into a non-orthogonal frame.
        """
        assert "hemispheres" in atlas_ds.data_vars
        assert "hemispheres" not in atlas_ds.coords
        # It must not leak onto the other variables as a coordinate.
        assert "hemispheres" not in atlas_ds["annotation"].coords
        assert "hemispheres" not in atlas_ds["reference"].coords

    def test_dataarray_dims(self, atlas_ds: xr.Dataset) -> None:
        acc = atlas_ds.atlas
        for da in [acc.reference, acc.annotation, acc.hemispheres]:
            assert da.dims == ("z", "y", "x")

    def test_physical_coordinates_in_mm(self, atlas_ds: xr.Dataset) -> None:
        """Coordinate step must equal the voxdim attribute (in mm)."""
        for dim in ["z", "y", "x"]:
            coord = atlas_ds.atlas.annotation.coords[dim]
            np.testing.assert_allclose(coord.values[1], coord.attrs["voxdim"])

    def test_from_brainglobe_schema(self, mock_structures: StructuresDict) -> None:
        """atlas_from_brainglobe must produce the locked schema (dims/vars/coords/attrs)."""
        result = atlas_from_brainglobe(_MockBgAtlas(mock_structures, (4, 6, 8)))  # ty: ignore[invalid-argument-type]

        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"reference", "annotation", "hemispheres"}
        assert "hemispheres" not in result.coords
        assert result.atlas.reference.dtype == np.float32
        assert result.atlas.annotation.dtype == np.int32
        assert {
            "name",
            "citation",
            "species",
            "orientation",
            "structures",
            "physical_to_base",
        } <= set(result.attrs)
        # Coordinates should be in mm: step = resolution_um[0] * 1e-3.
        np.testing.assert_allclose(
            result.atlas.annotation.coords["z"].values[1], 25 * 1e-3
        )

    def test_from_brainglobe_schema_attr_types(
        self, mock_structures: StructuresDict
    ) -> None:
        """structures is a StructuresDict; physical_to_base is a numpy affine matrix."""
        result = atlas_from_brainglobe(_MockBgAtlas(mock_structures, (4, 6, 8)))  # ty: ignore[invalid-argument-type]
        assert isinstance(result.attrs["structures"], StructuresDict)
        physical_to_base = result.attrs["physical_to_base"]
        assert isinstance(physical_to_base, np.ndarray)
        assert physical_to_base.shape == (4, 4)


class TestStructuresSerialization:
    """Round-trip the structure hierarchy through JSON (W0)."""

    def test_roundtrip_preserves_tree(self, mock_structures: StructuresDict) -> None:
        rebuilt = structures_from_json(structures_to_json(mock_structures))
        assert isinstance(rebuilt, StructuresDict)
        ids = [997, 10, 20]
        # Levels, structure_id_path, and acronym↔id map must survive the round-trip.
        assert {i: rebuilt.tree.level(i) for i in ids} == {997: 0, 10: 1, 20: 2}
        for i in ids:
            assert (
                rebuilt[i]["structure_id_path"]
                == mock_structures[i]["structure_id_path"]
            )
        assert rebuilt.acronym_to_id_map == mock_structures.acronym_to_id_map

    def test_mesh_filename_kept_complete(
        self, mock_structures: StructuresDict, obj_path
    ) -> None:
        """The complete mesh path is preserved (fetched atlases read from the cache)."""
        rebuilt = structures_from_json(structures_to_json(mock_structures))
        assert rebuilt[997]["mesh_filename"] == str(obj_path)


class TestStructureMetadata:
    """Tests for accessor metadata properties."""

    def test_lookup_columns(self, atlas_ds: xr.Dataset) -> None:
        assert set(atlas_ds.atlas.lookup.columns) >= {"acronym", "name", "rgb_triplet"}

    def test_lookup_index_matches_structure_ids(self, atlas_ds: xr.Dataset) -> None:
        assert set(atlas_ds.atlas.lookup.index) == {997, 10, 20}

    def test_lookup_values_match_structures(self, atlas_ds: xr.Dataset) -> None:
        df = atlas_ds.atlas.lookup
        assert df.loc[10, "acronym"] == "ch"
        assert df.loc[10, "name"] == "child region"

    def test_lookup_is_cached(self, atlas_ds: xr.Dataset) -> None:
        """Accessing lookup twice must return the same DataFrame object."""
        assert atlas_ds.atlas.lookup is atlas_ds.atlas.lookup

    def test_norm_maps_background_below_range(self, atlas_ds: xr.Dataset) -> None:
        """Label 0 (background) must map below the colormap range."""
        assert atlas_ds.atlas.norm(0) < 0

    def test_cmap_under_color_is_transparent(self, atlas_ds: xr.Dataset) -> None:
        """Under color must be fully transparent (RGBA = [0, 0, 0, 0])."""
        np.testing.assert_allclose(atlas_ds.atlas.cmap.get_under(), [0, 0, 0, 0])

    def test_missing_structures_attr_raises_clear_error(
        self, atlas_ds: xr.Dataset
    ) -> None:
        """Dropping attrs (as many xarray ops do) must give a clear error, not KeyError obscurity."""
        stripped = atlas_ds.drop_attrs()
        with pytest.raises(KeyError, match="keep_attrs"):
            stripped.atlas.structures


class TestSearch:
    """Tests for the accessor's search, compared against direct DataFrame filtering."""

    def test_search_all_fields_substring(self, atlas_ds: xr.Dataset) -> None:
        result = atlas_ds.atlas.search("child")
        assert isinstance(result, pd.DataFrame)
        assert 10 in result.index
        assert 20 in result.index
        assert 997 not in result.index

    def test_search_acronym_exact_match(self, atlas_ds: xr.Dataset) -> None:
        result = atlas_ds.atlas.search("gc", field="acronym")
        assert list(result.index) == [20]

    def test_search_name_is_case_insensitive(self, atlas_ds: xr.Dataset) -> None:
        result = atlas_ds.atlas.search("Child Region", field="name")
        assert 10 in result.index
        assert 20 not in result.index

    def test_search_name_accepts_regex(self, atlas_ds: xr.Dataset) -> None:
        result = atlas_ds.atlas.search(".*child.*", field="name")
        assert 10 in result.index
        assert 20 in result.index

    def test_search_no_match_returns_empty_dataframe(
        self, atlas_ds: xr.Dataset
    ) -> None:
        result = atlas_ds.atlas.search("no_such_region_xyz")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestGetMasks:
    """Tests for the accessor's get_masks, compared against a numpy reference.

    Fixture annotation layout (shape 4, 6, 8):
      - [:2, :, 2:6] = 10   (child, descendant of 997)
      - [2:, :, 2:6] = 20   (grandchild, descendant of 997 and 10)
      - elsewhere    = 0    (background)

    Hemisphere layout:
      - [:, :, :2] = 2   (right)
      - [:, :, 2:] = 1   (left)
    """

    def test_single_region_both_sides_vs_reference(self, atlas_ds: xr.Dataset) -> None:
        ann = atlas_ds.atlas.annotation.values
        expected = np.where(np.isin(ann, [997, 10, 20]), 997, 0).astype(np.int32)
        result = atlas_ds.atlas.get_masks(997)
        np.testing.assert_array_equal(result.values[0], expected)

    def test_single_region_left_side_vs_reference(self, atlas_ds: xr.Dataset) -> None:
        ann = atlas_ds.atlas.annotation.values
        hemi = atlas_ds.atlas.hemispheres.values
        expected = np.where(np.isin(ann, [10, 20]) & (hemi == 1), 10, 0).astype(
            np.int32
        )
        result = atlas_ds.atlas.get_masks(10, sides="left")
        np.testing.assert_array_equal(result.values[0], expected)

    def test_single_region_right_side_vs_reference(self, atlas_ds: xr.Dataset) -> None:
        ann = atlas_ds.atlas.annotation.values
        hemi = atlas_ds.atlas.hemispheres.values
        expected = np.where(np.isin(ann, [10, 20]) & (hemi == 2), 10, 0).astype(
            np.int32
        )
        result = atlas_ds.atlas.get_masks(10, sides="right")
        np.testing.assert_array_equal(result.values[0], expected)

    def test_left_and_right_partition_both(self, atlas_ds: xr.Dataset) -> None:
        left = atlas_ds.atlas.get_masks(10, sides="left").values[0]
        right = atlas_ds.atlas.get_masks(10, sides="right").values[0]
        both = atlas_ds.atlas.get_masks(10, sides="both").values[0]

        np.testing.assert_array_equal((left > 0) | (right > 0), both > 0)
        np.testing.assert_array_equal(
            (left > 0) & (right > 0), np.zeros_like(left, dtype=bool)
        )

    def test_descendant_voxels_included(self, atlas_ds: xr.Dataset) -> None:
        """get_masks(10) must include voxels labeled 20 (grandchild of 10)."""
        both = atlas_ds.atlas.get_masks(10).values[0]
        ann = atlas_ds.atlas.annotation.values
        assert np.any(both[ann == 20] == 10), (
            "grandchild voxels (label 20) should be included"
        )

    def test_multiple_regions_stacked_shape(self, atlas_ds: xr.Dataset) -> None:
        result = atlas_ds.atlas.get_masks([10, 20])
        assert result.dims == ("mask", "z", "y", "x")
        assert result.sizes["mask"] == 2

    def test_multiple_regions_masks_coord_contains_acronyms(
        self, atlas_ds: xr.Dataset
    ) -> None:
        result = atlas_ds.atlas.get_masks([997, 10, 20])
        np.testing.assert_array_equal(
            result.coords["mask"].values, ["root", "ch", "gc"]
        )

    def test_per_region_sides_mask_coord_disambiguated(
        self, atlas_ds: xr.Dataset
    ) -> None:
        """Same region with different sides must not share a `mask` coord value."""
        result = atlas_ds.atlas.get_masks([10, 10], sides=["left", "right"])
        np.testing.assert_array_equal(result.coords["mask"].values, ["ch_L", "ch_R"])

    def test_both_side_mask_coord_is_bare_acronym(self, atlas_ds: xr.Dataset) -> None:
        result = atlas_ds.atlas.get_masks(10, sides="both")
        np.testing.assert_array_equal(result.coords["mask"].values, ["ch"])

    def test_str_acronym_gives_same_result_as_integer_id(
        self, atlas_ds: xr.Dataset
    ) -> None:
        by_id = atlas_ds.atlas.get_masks(10).values[0]
        by_acronym = atlas_ds.atlas.get_masks("ch").values[0]
        np.testing.assert_array_equal(by_id, by_acronym)

    def test_spatial_coords_match_annotation(self, atlas_ds: xr.Dataset) -> None:
        result = atlas_ds.atlas.get_masks(10)
        for dim in ["z", "y", "x"]:
            np.testing.assert_array_equal(
                result.coords[dim].values, atlas_ds.atlas.annotation.coords[dim].values
            )

    def test_uses_hemisphere_labels_from_attrs(self, atlas_ds: xr.Dataset) -> None:
        ds = atlas_ds.copy()
        ds["hemispheres"] = ds["hemispheres"].copy(
            data=np.where(ds["hemispheres"].values == 1, 7, 9).astype(np.int8)
        )
        ds["hemispheres"].attrs = {"left": 7, "right": 9}

        left = ds.atlas.get_masks(10, sides="left").values[0]
        right = ds.atlas.get_masks(10, sides="right").values[0]
        both = ds.atlas.get_masks(10, sides="both").values[0]

        np.testing.assert_array_equal((left > 0) | (right > 0), both > 0)

    def test_2d_slice_preserves_spatial_dims(self, atlas_ds: xr.Dataset) -> None:
        sliced = atlas_ds.isel(z=0, drop=True)
        result = sliced.atlas.get_masks(10)
        assert result.dims == ("mask", "y", "x")
        np.testing.assert_array_equal(
            result.coords["y"], sliced["annotation"].coords["y"]
        )
        np.testing.assert_array_equal(
            result.coords["x"], sliced["annotation"].coords["x"]
        )

    def test_sides_length_mismatch_raises(self, atlas_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="same length"):
            atlas_ds.atlas.get_masks([10, 20], sides=["left"])

    def test_invalid_side_value_raises(self, atlas_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="Invalid side"):
            atlas_ds.atlas.get_masks(10, sides="center")

    def test_unknown_region_id_raises(self, atlas_ds: xr.Dataset) -> None:
        with pytest.raises(KeyError):
            atlas_ds.atlas.get_masks(9999)

    def test_unknown_region_acronym_raises(self, atlas_ds: xr.Dataset) -> None:
        with pytest.raises(KeyError):
            atlas_ds.atlas.get_masks("NONEXISTENT")


class TestGetMesh:
    """Tests for the accessor's get_mesh.

    The OBJ mesh (from conftest) has:
      Vertices 0-2 at RL = 50 µm  → right hemisphere (< midline 100 µm)
      Vertices 3-5 at RL = 150 µm → left  hemisphere (≥ midline 100 µm)
      Face 0: triangle (0, 1, 2) — entirely right
      Face 1: triangle (3, 4, 5) — entirely left

    get_mesh converts µm → mm (factor 1e-3); the identity mesh vertex transform leaves the
    millimetre vertices unchanged.
    """

    def test_vertices_transformed_to_mm(self, atlas_ds: xr.Dataset) -> None:
        vertices_mm, _ = atlas_ds.atlas.get_mesh(997, side="both")
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

    def test_both_sides_returns_all_faces(self, atlas_ds: xr.Dataset) -> None:
        vertices, faces = atlas_ds.atlas.get_mesh(997, side="both")
        assert len(vertices) == 6
        assert len(faces) == 2

    def test_right_side_clips_to_right_hemisphere(self, atlas_ds: xr.Dataset) -> None:
        vertices, faces = atlas_ds.atlas.get_mesh(997, side="right")
        assert len(vertices) == 3
        assert len(faces) == 1
        np.testing.assert_array_equal(faces, [[0, 1, 2]])
        assert np.all(vertices[:, 2] < 0.1)

    def test_left_side_clips_to_left_hemisphere(self, atlas_ds: xr.Dataset) -> None:
        vertices, faces = atlas_ds.atlas.get_mesh(997, side="left")
        assert len(vertices) == 3
        assert len(faces) == 1
        np.testing.assert_array_equal(faces, [[0, 1, 2]])
        assert np.all(vertices[:, 2] >= 0.1)

    def test_faces_dtype_is_int32(self, atlas_ds: xr.Dataset) -> None:
        _, faces = atlas_ds.atlas.get_mesh(997)
        assert faces.dtype == np.int32

    def test_str_acronym_gives_same_result_as_integer_id(
        self, atlas_ds: xr.Dataset
    ) -> None:
        vertices_id, faces_id = atlas_ds.atlas.get_mesh(997)
        vertices_str, faces_str = atlas_ds.atlas.get_mesh("root")
        np.testing.assert_array_equal(vertices_id, vertices_str)
        np.testing.assert_array_equal(faces_id, faces_str)

    def test_region_without_mesh_raises(self, atlas_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="No mesh file"):
            atlas_ds.atlas.get_mesh(10)

    def test_unknown_region_raises(self, atlas_ds: xr.Dataset) -> None:
        with pytest.raises(KeyError):
            atlas_ds.atlas.get_mesh(9999)


class TestAncestors:
    """Tests for the accessor's ancestors, compared against direct treelib traversal."""

    def test_root_has_no_ancestors(self, atlas_ds: xr.Dataset) -> None:
        assert atlas_ds.atlas.ancestors(997) == []

    def test_child_has_root_as_sole_ancestor(self, atlas_ds: xr.Dataset) -> None:
        result = atlas_ds.atlas.ancestors(10)
        assert len(result) == 1
        assert result[0].identifier == 997

    def test_grandchild_ancestors_ordered_root_first(
        self, atlas_ds: xr.Dataset
    ) -> None:
        result = atlas_ds.atlas.ancestors(20)
        assert len(result) == 2
        assert result[0].identifier == 997
        assert result[1].identifier == 10

    def test_str_acronym_gives_same_result_as_integer_id(
        self, atlas_ds: xr.Dataset
    ) -> None:
        by_id = [n.identifier for n in atlas_ds.atlas.ancestors(20)]
        by_acronym = [n.identifier for n in atlas_ds.atlas.ancestors("gc")]
        assert by_id == by_acronym

    def test_unknown_region_raises(self, atlas_ds: xr.Dataset) -> None:
        with pytest.raises(KeyError):
            atlas_ds.atlas.ancestors(9999)


class TestIO:
    """Zarr save/load round-trip (W3)."""

    def test_save_rejects_non_zarr_suffix(self, atlas_ds: xr.Dataset, tmp_path) -> None:
        with pytest.raises(ValueError, match=".zarr"):
            save_atlas(atlas_ds, tmp_path / "atlas.zip")

    def test_load_rejects_non_zarr_suffix(self, tmp_path) -> None:
        with pytest.raises(ValueError, match=".zarr"):
            load_atlas(tmp_path / "atlas")

    def test_roundtrip_preserves_fields_bit_for_bit(
        self, atlas_ds: xr.Dataset, tmp_path
    ) -> None:
        path = tmp_path / "atlas.zarr"
        save_atlas(atlas_ds, path)
        loaded = load_atlas(path)

        np.testing.assert_array_equal(
            loaded.atlas.reference.values, atlas_ds.atlas.reference.values
        )
        np.testing.assert_array_equal(
            loaded.atlas.annotation.values, atlas_ds.atlas.annotation.values
        )
        np.testing.assert_array_equal(
            loaded.atlas.hemispheres.values, atlas_ds.atlas.hemispheres.values
        )

    def test_save_does_not_mutate_caller_attrs(
        self, atlas_ds: xr.Dataset, tmp_path
    ) -> None:
        """Stripping cmap/norm on save must not touch the caller's in-memory Dataset."""
        # Give the in-memory annotation a cmap so there is something to strip.
        ds = atlas_ds.copy()
        ds["annotation"].attrs["cmap"] = atlas_ds.atlas.cmap
        save_atlas(ds, tmp_path / "atlas.zarr")
        assert "cmap" in ds["annotation"].attrs

    def test_cmap_norm_rebuilt_on_load(self, atlas_ds: xr.Dataset, tmp_path) -> None:
        path = tmp_path / "atlas.zarr"
        save_atlas(atlas_ds, path)
        loaded = load_atlas(path)
        # cmap/norm are not stored, but must be rebuilt into annotation.attrs on load.
        assert "cmap" in loaded["annotation"].attrs
        assert "norm" in loaded["annotation"].attrs

    def test_physical_to_base_restored_as_array(
        self, atlas_ds: xr.Dataset, tmp_path
    ) -> None:
        """The affine physical_to_base reloads as a numpy array, not a JSON list."""
        path = tmp_path / "atlas.zarr"
        save_atlas(atlas_ds, path)
        loaded = load_atlas(path)
        transform = loaded.attrs["physical_to_base"]
        assert isinstance(transform, np.ndarray)
        np.testing.assert_allclose(transform, np.eye(4))

    def test_resampled_atlas_with_affines_roundtrips(
        self, atlas_ds: xr.Dataset, tmp_path
    ) -> None:
        """A resampled atlas inherits an `affines` dict of arrays; it must round-trip."""
        aff = np.array(
            [[1.0, 0, 0, 0.5], [0, 1, 0, -0.2], [0, 0, 1, 0.3], [0, 0, 0, 1.0]]
        )
        reference = atlas_ds.atlas.reference.copy()
        reference.attrs = {**reference.attrs, "affines": {"physical_to_sform": aff}}
        resampled = atlas_ds.atlas.resample_like(reference, np.eye(4))
        assert "affines" in resampled["reference"].attrs  # guards the scenario.

        path = tmp_path / "atlas.zarr"
        save_atlas(resampled, path)
        loaded = load_atlas(path)
        got = loaded["reference"].attrs["affines"]["physical_to_sform"]
        assert isinstance(got, np.ndarray)
        np.testing.assert_allclose(got, aff)

    def test_structural_queries_work_post_load(
        self, atlas_ds: xr.Dataset, tmp_path
    ) -> None:
        path = tmp_path / "atlas.zarr"
        save_atlas(atlas_ds, path)
        loaded = load_atlas(path)

        assert loaded.atlas.search("gc", field="acronym").index.tolist() == [20]
        assert [n.identifier for n in loaded.atlas.ancestors(20)] == [997, 10]
        # get_masks post-load must match the pre-save result.
        np.testing.assert_array_equal(
            loaded.atlas.get_masks(10).values, atlas_ds.atlas.get_masks(10).values
        )

    def test_meshes_bundled_into_store(self, atlas_ds: xr.Dataset, tmp_path) -> None:
        """The region OBJ files are copied into the store's meshes/ subdirectory."""
        path = tmp_path / "atlas.zarr"
        save_atlas(atlas_ds, path)
        assert (path / "meshes" / "997.obj").is_file()

    def test_save_suppresses_consolidated_metadata_warning(
        self, atlas_ds: xr.Dataset, tmp_path, monkeypatch
    ) -> None:
        """Zarr v3 consolidated-metadata warning from xarray/zarr is hidden."""
        path = tmp_path / "atlas.zarr"

        def fake_to_zarr(self, *args, **kwargs) -> None:
            warnings.warn(
                "Consolidated metadata is currently not part in the Zarr format 3 specification.",
                UserWarning,
            )

        monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_atlas(atlas_ds, path)

        assert not caught

    def test_loaded_mesh_filename_points_into_store(
        self, atlas_ds: xr.Dataset, tmp_path
    ) -> None:
        path = tmp_path / "atlas.zarr"
        save_atlas(atlas_ds, path)
        loaded = load_atlas(path)
        structures = loaded.attrs["structures"]
        assert structures[997]["mesh_filename"] == str(path / "meshes" / "997.obj")
        assert structures[10]["mesh_filename"] is None

    def test_get_mesh_reads_bundle_when_source_is_gone(
        self, atlas_ds: xr.Dataset, structure_list: list[dict], obj_path, tmp_path
    ) -> None:
        """A loaded atlas renders meshes from the bundle, without the original source."""
        # Point a copy of the atlas at a throwaway mesh we are free to delete.
        source = tmp_path / "src_997.obj"
        source.write_text(obj_path.read_text())
        records = [dict(record) for record in structure_list]
        for record in records:
            if record["id"] == 997:
                record["mesh_filename"] = str(source)
        ds = atlas_ds.copy()
        ds.attrs = {**atlas_ds.attrs, "structures": StructuresDict(records)}

        path = tmp_path / "atlas.zarr"
        save_atlas(ds, path)
        source.unlink()  # the BrainGlobe cache / original mesh is now unavailable.

        loaded = load_atlas(path)
        vertices, faces = loaded.atlas.get_mesh(997)
        np.testing.assert_array_equal(vertices, atlas_ds.atlas.get_mesh(997)[0])
        assert len(faces) == 2


class TestNonlinearMesh:
    """Nonlinear (displacement-field / B-spline) mesh resampling on the accessor.

    A displacement field's component `i` displaces along DataArray axis `dims[i]` in
    `(z, y, x)` order; mesh vertex columns are likewise `(z, y, x)`. A `+δ` pull on a
    component warps the matching mesh column by `-δ` (the mesh transform is inverted on
    apply). This axis-order convention is load-bearing (it was a real z/x-swap bug).
    """

    def test_field_warps_mesh_x_column(self, atlas_ds: xr.Dataset) -> None:
        """A uniform +0.01 pull on component 2 (x) shifts only the mesh x-column by -0.01."""
        reference = atlas_ds.atlas.reference
        data = np.zeros((3, *reference.shape))
        data[2] = 0.01  # component 2 == x
        ds = _with_physical_to_base_transform(atlas_ds, _field_on_grid(reference, data))

        warped, _ = ds.atlas.get_mesh(997)
        base, _ = atlas_ds.atlas.get_mesh(997)
        expected = base.copy()
        expected[:, 2] -= 0.01
        np.testing.assert_allclose(warped, expected, atol=1e-6)

    def test_field_drops_vertices_warped_outside_grid(
        self, atlas_ds: xr.Dataset
    ) -> None:
        """Vertices a nonlinear warp pushes outside the reference grid are dropped."""
        reference = atlas_ds.atlas.reference
        data = np.zeros((3, *reference.shape))
        data[2] = 0.5  # inverts to a -0.5 shift, pushing every vertex off the grid
        ds = _with_physical_to_base_transform(atlas_ds, _field_on_grid(reference, data))

        vertices, faces = ds.atlas.get_mesh(997)
        assert vertices.shape == (0, 3)
        assert faces.shape == (0, 3)

    def test_bspline_transform_samples_field(
        self, atlas_ds: xr.Dataset, monkeypatch
    ) -> None:
        """A B-spline mesh transform is sampled to a dense field before warping."""
        reference = atlas_ds.atlas.reference
        data = np.zeros((3, *reference.shape))
        data[2] = 0.01
        field = _field_on_grid(reference, data)
        monkeypatch.setattr(
            "confusius.atlas._physical_to_base_transform.sample_displacement_field_like",
            lambda transform, ref: field,
        )
        bspline = xr.DataArray(
            np.zeros((2, 2, 2, 3)),
            dims=["cz", "cy", "cx", "comp"],
            attrs={"type": "bspline_transform"},
        )
        ds = _with_physical_to_base_transform(atlas_ds, bspline)

        warped, _ = ds.atlas.get_mesh(997)
        base, _ = atlas_ds.atlas.get_mesh(997)
        expected = base.copy()
        expected[:, 2] -= 0.01
        np.testing.assert_allclose(warped, expected, atol=1e-6)

    def test_resample_like_affine_stores_transform_in_attr(
        self, atlas_ds: xr.Dataset
    ) -> None:
        resampled = atlas_ds.atlas.resample_like(atlas_ds.atlas.reference, np.eye(4))
        assert isinstance(resampled.attrs["physical_to_base"], np.ndarray)
        assert "physical_to_base" not in resampled.data_vars

    def test_resample_like_field_stores_transform_as_dataarray_attr(
        self, atlas_ds: xr.Dataset
    ) -> None:
        reference = atlas_ds.atlas.reference
        field = _field_on_grid(reference, np.zeros((3, *reference.shape)))
        resampled = atlas_ds.atlas.resample_like(reference, field)
        assert isinstance(resampled.attrs["physical_to_base"], xr.DataArray)
        assert "physical_to_base" not in resampled.data_vars
        validate_atlas_dataset(resampled)

    def test_resample_like_affine_shifts_mesh(self, atlas_ds: xr.Dataset) -> None:
        """An affine pull translation warps the mesh by its inverse."""
        transform = np.eye(4)
        transform[2, 3] = 0.02  # +0.02 pull on x → mesh x shifted by -0.02
        resampled = atlas_ds.atlas.resample_like(atlas_ds.atlas.reference, transform)

        warped, _ = resampled.atlas.get_mesh(997)
        base, _ = atlas_ds.atlas.get_mesh(997)
        expected = base.copy()
        expected[:, 2] -= 0.02
        np.testing.assert_allclose(warped, expected, atol=1e-5)

    def test_nonlinear_atlas_roundtrips_through_zarr(
        self, atlas_ds: xr.Dataset, tmp_path
    ) -> None:
        """The displacement-field transform serializes and get_mesh works post-load."""
        reference = atlas_ds.atlas.reference
        data = np.zeros((3, *reference.shape))
        data[2] = 0.01
        resampled = atlas_ds.atlas.resample_like(
            reference, _field_on_grid(reference, data)
        )

        path = tmp_path / "atlas.zarr"
        save_atlas(resampled, path)
        loaded = load_atlas(path)

        assert isinstance(loaded.attrs["physical_to_base"], xr.DataArray)
        assert "physical_to_base" not in loaded.data_vars
        np.testing.assert_allclose(
            loaded.atlas.get_mesh(997)[0], resampled.atlas.get_mesh(997)[0], atol=1e-6
        )

    def test_resample_matches_resample_like(self, atlas_ds: xr.Dataset) -> None:
        """Grid-spec resample equals resample_like on an equivalent reference."""
        reference = atlas_ds.atlas.reference
        by_like = atlas_ds.atlas.resample_like(reference, np.eye(4))
        by_grid = atlas_ds.atlas.resample(
            np.eye(4),
            shape=reference.shape,
            spacing=[0.05, 0.05, 0.05],
            origin=[0.0, 0.0, 0.0],
            dims=["z", "y", "x"],
        )
        for var in ["reference", "annotation", "hemispheres"]:
            np.testing.assert_array_equal(by_like[var].values, by_grid[var].values)


class TestTransformComposition:
    """Composing physical_to_base pull transforms across successive resamples.

    resample_like composes the atlas's stored pull transform with the new one, so a chain
    of resamples must agree with a single resample by the composed transform. These cover
    the affine·affine, identity, and affine·nonlinear composition paths, plus the
    initialization-seeded field inversion get_mesh uses.
    """

    def test_composed_affine_resamples_match_single_step(
        self, atlas_ds: xr.Dataset
    ) -> None:
        """Two affine resamples compose to the matrix product and the same mesh."""
        reference = atlas_ds.atlas.reference
        a = np.eye(4)
        a[2, 3] = 0.02
        b = np.eye(4)
        b[2, 3] = 0.03
        two_step = atlas_ds.atlas.resample_like(reference, a).atlas.resample_like(
            reference, b
        )
        one_step = atlas_ds.atlas.resample_like(reference, a @ b)

        np.testing.assert_allclose(two_step.attrs["physical_to_base"], a @ b)
        np.testing.assert_allclose(
            two_step.atlas.get_mesh(997)[0],
            one_step.atlas.get_mesh(997)[0],
            atol=1e-6,
        )

    def test_identity_resample_preserves_existing_transform(
        self, atlas_ds: xr.Dataset
    ) -> None:
        """Resampling an already-transformed atlas by identity leaves the transform intact."""
        reference = atlas_ds.atlas.reference
        a = np.eye(4)
        a[2, 3] = 0.02
        once = atlas_ds.atlas.resample_like(reference, a)
        twice = once.atlas.resample_like(reference, np.eye(4))

        np.testing.assert_allclose(twice.attrs["physical_to_base"], a)

    def test_affine_then_zero_field_matches_the_affine_mesh(
        self, atlas_ds: xr.Dataset
    ) -> None:
        """An affine followed by a zero field composes to a field giving the affine's mesh."""
        reference = atlas_ds.atlas.reference
        a = np.eye(4)
        a[2, 3] = 0.02
        zero_field = _field_on_grid(reference, np.zeros((3, *reference.shape)))

        composed = atlas_ds.atlas.resample_like(reference, a).atlas.resample_like(
            reference, zero_field
        )
        affine_only = atlas_ds.atlas.resample_like(reference, a)

        # A zero field is the identity, so the affine·field composition is stored as a
        # dense displacement field but must warp the mesh exactly like the affine alone.
        assert isinstance(composed.attrs["physical_to_base"], xr.DataArray)
        np.testing.assert_allclose(
            composed.atlas.get_mesh(997)[0],
            affine_only.atlas.get_mesh(997)[0],
            atol=1e-5,
        )

    def test_get_mesh_seeds_field_inversion_with_bspline_initialization(
        self, atlas_ds: xr.Dataset
    ) -> None:
        """A bspline_initialization affine seeds the field inversion get_mesh runs."""
        reference = atlas_ds.atlas.reference
        data = np.zeros((3, *reference.shape))
        data[2] = 0.01
        field = _field_on_grid(reference, data)
        init = np.eye(4)
        init[2, 3] = 0.01  # its inverse seeds the iteration near the true pre-image
        field.attrs["affines"] = {"bspline_initialization": init}
        ds = _with_physical_to_base_transform(atlas_ds, field)

        warped, _ = ds.atlas.get_mesh(997)
        base, _ = atlas_ds.atlas.get_mesh(997)
        expected = base.copy()
        expected[:, 2] -= 0.01
        np.testing.assert_allclose(warped, expected, atol=1e-6)


class TestGroupbyIntegration:
    """The aligned annotation coordinate plugs into flox groupby (W6)."""

    def test_groupby_annotation_mean_matches_naive_loop(
        self, atlas_ds: xr.Dataset
    ) -> None:
        rng = np.random.default_rng(0)
        data = xr.DataArray(
            rng.standard_normal(atlas_ds.atlas.annotation.shape).astype(np.float32),
            dims=["z", "y", "x"],
            coords={d: atlas_ds.atlas.annotation.coords[d] for d in ["z", "y", "x"]},
        )
        grouped = data.groupby(atlas_ds.atlas.annotation.rename("annotation")).mean()

        ann = atlas_ds.atlas.annotation.values
        for label in np.unique(ann):
            expected = data.values[ann == label].mean()
            np.testing.assert_allclose(
                grouped.sel(annotation=label).item(), expected, rtol=1e-6
            )


class TestStandaloneOperations:
    """The module-level get_mesh/get_masks/search take a Dataset and validate it."""

    def test_search_returns_expected_region(self, atlas_ds: xr.Dataset) -> None:
        assert cf.atlas.search(atlas_ds, "gc", field="acronym").index.tolist() == [20]

    def test_get_masks_matches_accessor(self, atlas_ds: xr.Dataset) -> None:
        np.testing.assert_array_equal(
            cf.atlas.get_masks(atlas_ds, 10).values,
            atlas_ds.atlas.get_masks(10).values,
        )

    def test_get_mesh_matches_accessor(self, atlas_ds: xr.Dataset) -> None:
        vertices, faces = cf.atlas.get_mesh(atlas_ds, 997)
        acc_vertices, acc_faces = atlas_ds.atlas.get_mesh(997)
        np.testing.assert_array_equal(vertices, acc_vertices)
        np.testing.assert_array_equal(faces, acc_faces)
        assert len(faces) == 2  # the fixture mesh has two triangles.

    def test_functions_validate_input(self, atlas_ds: xr.Dataset) -> None:
        """A Dataset missing a required variable is rejected before any work."""
        not_atlas = atlas_ds.drop_vars("hemispheres")
        with pytest.raises(ValueError, match="hemispheres"):
            cf.atlas.get_mesh(not_atlas, 997)
        with pytest.raises(ValueError, match="hemispheres"):
            cf.atlas.get_masks(not_atlas, 10)
        with pytest.raises(ValueError, match="hemispheres"):
            cf.atlas.search(not_atlas, "gc")
