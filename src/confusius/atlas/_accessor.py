"""The `.atlas` xarray Dataset accessor: data-aware brain-atlas operations."""

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal, SupportsFloat, SupportsIndex

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius.atlas._structures import (
    _build_lookup_df,
    _get_descendant_ids,
    _resolve_region_id,
)
from confusius.atlas._world_to_base_transform import (
    WorldToBaseTransform,
    _apply_world_to_base_transform,
    _compose_world_to_base_transforms,
    _drop_vertices_outside_grid,
)
from confusius.registration.resampling import resample_volume
from confusius.validation import ensure_voxeldata
from confusius.validation.atlas import validate_atlas

if TYPE_CHECKING:
    import treelib
    from brainglobe_atlasapi.structure_class import StructuresDict
    from matplotlib.colors import BoundaryNorm, ListedColormap


@xr.register_dataset_accessor("atlas")
class AtlasAccessor:
    """Brain-atlas operations on an atlas `xarray.Dataset`.

    Registered as the `.atlas` namespace on any Dataset produced by
    [`fetch_brainglobe_atlas`][confusius.datasets.fetch_brainglobe_atlas] or
    [`load_atlas`][confusius.io.load_atlas]. `Dataset.attrs["structures"]`
    holds the BrainGlobe
    [`StructuresDict`][brainglobe_atlasapi.structure_class.StructuresDict] directly, so
    structural queries keep working for as long as that attribute rides along (xarray
    drops `attrs` on many ops by default; use `xarray.set_options(keep_attrs=True)` in
    pipelines).

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset with `reference`, `annotation`, and `hemispheres` data variables as
        VoxelData arrays on a common `k`/`j`/`i` grid, and the atlas
        metadata in `attrs`.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds
        self._lookup: pd.DataFrame | None = None

    # ── Data properties ───────────────────────────────────────────────────────────────

    @property
    def reference(self) -> xr.DataArray:
        """Reference template VoxelData array.

        Returns
        -------
        xarray.DataArray
            The reference template VoxelData array.
        """
        return self._ds["reference"]

    @property
    def annotation(self) -> xr.DataArray:
        """Region annotations VoxelData array.

        `attrs["rgb_lookup"]` carries a `{id: [r, g, b]}` dict used for colormap
        construction.

        Returns
        -------
        xarray.DataArray
            The region annotation VoxelData array with integer labels.
        """
        return self._ds["annotation"]

    @property
    def hemispheres(self) -> xr.DataArray:
        """Hemisphere map VoxelData array (1 = left, 2 = right).

        Returns
        -------
        xarray.DataArray
            The hemisphere map VoxelData array.
        """
        return self._ds["hemispheres"]

    @property
    def _world_to_base_transform(self) -> WorldToBaseTransform:
        """Pull transform mapping the atlas's world space back to base atlas space.

        Held directly in `attrs["world_to_base"]` as either a `(4, 4)` numpy affine or a
        dense displacement-field DataArray after a nonlinear resample. It is returned as-is
        — the mesh-warping helpers consume the pull form (they invert the affine, or invert
        the field per point, when mapping base vertices into world space).

        Returns
        -------
        numpy.ndarray or xarray.DataArray
            The `(4, 4)` pull affine, or the dense displacement-field DataArray.
        """
        return self._ds.attrs["world_to_base"]

    # ── Structure metadata ────────────────────────────────────────────────────

    @property
    def structures(self) -> "StructuresDict":
        """BrainGlobe structure dictionary held in `Dataset.attrs["structures"]`.

        Returns
        -------
        brainglobe_atlasapi.structure_class.StructuresDict
            The structure dictionary with its hierarchy tree.

        Raises
        ------
        KeyError
            If `Dataset.attrs` has no `structures` entry (e.g. after an xarray op that
            dropped `attrs`; wrap the pipeline in `xarray.set_options(keep_attrs=True)`).
        """
        if "structures" not in self._ds.attrs:
            raise KeyError(
                "This Dataset has no 'structures' attribute, so its structure "
                "hierarchy is unavailable. xarray drops attrs on many operations "
                "by default; run atlas pipelines under "
                "xarray.set_options(keep_attrs=True)."
            )
        return self._ds.attrs["structures"]

    @property
    def lookup(self) -> pd.DataFrame:
        """DataFrame with columns `acronym`, `name`, `rgb_triplet`.

        The DataFrame is indexed by structure index.

        Returns
        -------
        pandas.DataFrame
            The structure lookup DataFrame. Cached on first access.
        """
        if self._lookup is None:
            self._lookup = _build_lookup_df(self.structures)
        return self._lookup

    @property
    def cmap(self) -> "ListedColormap":
        """[`ListedColormap`][matplotlib.colors.ListedColormap] derived from `annotation.attrs["rgb_lookup"]`.

        Returns
        -------
        matplotlib.colors.ListedColormap
            The colormap to use for atlas rendering.
        """
        cmap, _ = build_atlas_cmap_and_norm(self.annotation.attrs["rgb_lookup"])
        return cmap

    @property
    def norm(self) -> "BoundaryNorm":
        """[`BoundaryNorm`][matplotlib.colors.BoundaryNorm] derived from `annotation.attrs["rgb_lookup"]`.

        Returns
        -------
        matplotlib.colors.BoundaryNorm
            The norm to use for atlas rendering.
        """
        _, norm = build_atlas_cmap_and_norm(self.annotation.attrs["rgb_lookup"])
        return norm

    # ── Search ────────────────────────────────────────────────────────────────────────

    def search(
        self,
        pattern: str,
        field: Literal["all", "acronym", "name"] = "all",
    ) -> pd.DataFrame:
        """Search structures by name or acronym.

        Parameters
        ----------
        pattern : str
            Substring or regex pattern.
        field : {"all", "acronym", "name"}, default: "all"
            Which column to search.

            - `"all"`: case-insensitive regex search on both `acronym` and `name`.
            - `"acronym"` / `"name"`: case-insensitive full regex match on that
              column only.

        Returns
        -------
        pandas.DataFrame
            Filtered view of [`lookup`][confusius.atlas.AtlasAccessor.lookup] matching the
            search criteria.

        Examples
        --------
        >>> ds.atlas.search("visual cortex")
        >>> ds.atlas.search("VISp", field="acronym")
        """
        df = self.lookup
        if field == "acronym":
            mask = df["acronym"].str.fullmatch(pattern, case=False)
        elif field == "name":
            mask = df["name"].str.fullmatch(pattern, case=False)
        else:
            mask = df["acronym"].str.contains(pattern, case=False, na=False) | df[
                "name"
            ].str.contains(pattern, case=False, na=False)
        return df[mask]

    # ── Masks ─────────────────────────────────────────────────────────────────────────

    def get_masks(
        self,
        regions: int | str | Sequence[int | str],
        sides: (
            Literal["left", "right", "both"]
            | Sequence[Literal["left", "right", "both"]]
        ) = "both",
    ) -> xr.DataArray:
        """Return integer region masks stacked along a `mask` dimension.

        Each layer along `mask` has values in `{0, region_id}`; voxels
        belonging to the requested region (including all descendants in the
        hierarchy) carry the region's index, all others are zero.

        Parameters
        ----------
        regions : int or str or sequence of int or str
            One or more regions, each given as a structure index or acronym.
        sides : {"left", "right", "both"} or sequence thereof, default: "both"
            Hemisphere filter. Pass a scalar to apply the same side to all regions, or a
            sequence of the same length as `regions` for per-region control.

        Returns
        -------
        xarray.DataArray
            Integer VoxelData array with dims
            `["mask", *annotation.dims]`. The `mask` coordinate holds the region acronym
            for each layer, suffixed with `_L`/`_R`
            when the corresponding `side` is `"left"`/`"right"` (left/right requests for
            the same region would otherwise share an acronym).

        Raises
        ------
        TypeError
            If `ds` is not a well-formed atlas Dataset.
        KeyError
            If any requested region acronym or index is not found in the atlas.
        ValueError
            If `sides` is a sequence whose length does not match `regions`, or if
            any element of `sides` is not `"left"`, `"right"`, or `"both"`.

        Examples
        --------
        >>> ds.atlas.get_masks("VISp")
        >>> ds.atlas.get_masks("VISp", sides="left")
        >>> ds.atlas.get_masks(["VISp", "AUDp", "MOp"])
        >>> ds.atlas.get_masks(["VISp", "AUDp"], sides=["left", "both"])
        >>> ds.atlas.get_masks(["VISp", "VISp"], sides=["left", "right"]).coords["mask"].values
        array(['VISp_L', 'VISp_R'], dtype=object)
        """
        return get_atlas_masks(self._ds, regions, sides)

    # ── Meshes ────────────────────────────────────────────────────────────────────────

    def get_mesh(
        self,
        region: int | str,
        side: Literal["left", "right", "both"] = "both",
        *,
        clip: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """Return vertex coordinates and face indices for a region's mesh.

        Reads the region's OBJ mesh, transforms its vertices from micron space to the
        DataArrays' current world space (millimetres), then optionally drops
        out-of-grid vertices and clips to one hemisphere. The mesh comes from the
        structure's `mesh_filename`: for a freshly fetched atlas this points into the
        BrainGlobe cache; for an atlas loaded with
        [`load_atlas`][confusius.io.load_atlas] it points at the mesh bundled
        inside the store.

        Parameters
        ----------
        region : int or str
            Structure index or acronym.
        side : {"left", "right", "both"}, default: "both"
            Hemisphere to include. `"both"` keeps the full mesh. `"left"` and `"right"`
            keep only vertices whose nearest `hemispheres` voxel carries that side's label
            (`hemispheres.attrs["left"]` / `["right"]`), sampled in the current world
            space. Faces are kept only when all three of their vertices survive, so the
            cut face is not closed. Sampling the hemisphere map makes this
            orientation-agnostic and correct after an arbitrary resample.
        clip : bool, default: True
            Whether to clip the final mesh to the current reference grid. If `False`,
            the mesh will still be transformed to the current world space, but the
            bounding box will not be respected.

        Returns
        -------
        vertices : numpy.ndarray, shape (N, 3)
            Vertex coordinates in the current world space (millimetres). After a
            nonlinear resample, vertices warped outside the reference grid are dropped.
        faces : numpy.ndarray, shape (M, 3)
            Zero-indexed triangle face indices (int32).

        Raises
        ------
        TypeError
            If `ds` is not a well-formed atlas Dataset.
        KeyError
            If the requested region is not found in the atlas.
        ValueError
            If the region has no mesh file, or the mesh file cannot be located.
        """
        return get_atlas_mesh(self._ds, region, side, clip=clip)

    # ── Resampling ────────────────────────────────────────────────────────────────────

    def resample(
        self,
        transform: WorldToBaseTransform,
        *,
        output_sizes: Mapping[Hashable, SupportsIndex],
        output_spacing: Mapping[Hashable, SupportsFloat | SupportsIndex],
        output_origin: Mapping[Hashable, SupportsFloat | SupportsIndex],
        output_direction: npt.ArrayLike,
        interpolation: Literal["linear", "nearest", "bspline"] = "linear",
        sitk_threads: int = -1,
    ) -> xr.Dataset:
        """Resample the atlas onto an explicit output grid.

        The atlas reference volume is resampled with `interpolation`; `annotation` and
        `hemispheres` are always resampled with nearest-neighbor interpolation to
        preserve integer labels. The returned Dataset stores the composed pull transform
        from its new world space back to the atlas base space in `attrs["world_to_base"]`.

        Parameters
        ----------
        transform : (4, 4) numpy.ndarray or xarray.DataArray
            Pull transform mapping output world coordinates to current atlas world
            coordinates.
        output_sizes : mapping of str to int
            Number of voxels along output axes, read by `k`/`j`/`i` keys.
        output_spacing : mapping of str to float
            World distance between output positions, read by `k`/`j`/`i` keys.
        output_origin : mapping of str to float
            World location of output position `(0, 0, 0)`, read by `z`/`y`/`x` keys.
        output_direction : (3, 3) numpy.ndarray
            Unit world-space direction columns for native `k`/`j`/`i` output axes.
        interpolation : {"linear", "nearest", "bspline"}, default: "linear"
            Interpolation used for the atlas reference volume.
        sitk_threads : int, default: -1
            Number of SimpleITK threads.

        Returns
        -------
        xarray.Dataset
            Resampled atlas Dataset on the requested grid. Meshes returned by
            `get_mesh` are transformed through the composed `world_to_base` attribute.
        """
        resampled_ref = resample_volume(
            self.reference,
            transform,
            output_sizes=output_sizes,
            output_spacing=output_spacing,
            output_origin=output_origin,
            output_direction=output_direction,
            interpolation=interpolation,
            fill_value=0.0,
            sitk_threads=sitk_threads,
        )
        resampled_ann = resample_volume(
            self.annotation,
            transform,
            output_sizes=output_sizes,
            output_spacing=output_spacing,
            output_origin=output_origin,
            output_direction=output_direction,
            interpolation="nearest",
            fill_value=0,
            sitk_threads=sitk_threads,
        )
        resampled_ann.attrs = self.annotation.attrs.copy()
        resampled_hemi = resample_volume(
            self.hemispheres,
            transform,
            output_sizes=output_sizes,
            output_spacing=output_spacing,
            output_origin=output_origin,
            output_direction=output_direction,
            interpolation="nearest",
            fill_value=0,
            sitk_threads=sitk_threads,
        )
        composed = _compose_world_to_base_transforms(
            self._world_to_base_transform, transform, resampled_ref, self.reference
        )
        return xr.Dataset(
            {
                "reference": resampled_ref,
                "annotation": resampled_ann,
                "hemispheres": resampled_hemi,
            },
            attrs={**self._ds.attrs, "world_to_base": composed},
        )

    def resample_like(
        self,
        reference: xr.DataArray,
        transform: WorldToBaseTransform,
        *,
        interpolation: Literal["linear", "nearest", "bspline"] = "linear",
        sitk_threads: int = -1,
    ) -> xr.Dataset:
        """Resample the atlas onto `reference`'s VoxelData grid.

        Parameters
        ----------
        reference : xarray.DataArray
            VoxelData array defining the target grid. Only its `k`/`j`/`i` grid
            (`sizes`/`spacing`/`origin`/`direction`) is used, so a `time` dimension, if
            present, is ignored. Must not have a `pose` or extra non-spatial dimension.
        transform : (4, 4) numpy.ndarray or xarray.DataArray
            Pull transform mapping reference world coordinates to atlas world
            coordinates.
        interpolation : {"linear", "nearest", "bspline"}, default: "linear"
            Interpolation used for the atlas reference volume.
        sitk_threads : int, default: -1
            Number of SimpleITK threads.

        Returns
        -------
        xarray.Dataset
            Resampled atlas Dataset with exactly `reference`'s voxel labels and
            voxel-to-world affine. Meshes returned by `get_mesh` are transformed
            through the composed `world_to_base` attribute.

        Raises
        ------
        ValueError
            If `reference` has a `pose` or extra non-spatial dimension, or is not a
            VoxelData array.
        """
        reference = ensure_voxeldata(
            reference,
            require_time=False,
            allow_pose=False,
            allow_extra_dims=False,
        )
        grid = reference.fusi
        output = self.resample(
            transform,
            output_sizes=reference.sizes,
            output_spacing=grid.spacing,
            output_origin=grid.origin,
            output_direction=grid.direction,
            interpolation=interpolation,
            sitk_threads=sitk_threads,
        )
        data_vars = {
            name: output[name].fusi.affine.reindex_voxels_like(reference)
            for name in output.data_vars
        }
        reference_attrs = dict(data_vars["reference"].attrs)
        if "affines" in reference.attrs:
            reference_attrs["affines"] = deepcopy(reference.attrs["affines"])
        else:
            reference_attrs.pop("affines", None)
        data_vars["reference"] = data_vars["reference"].assign_attrs(reference_attrs)
        world_to_base = output.attrs["world_to_base"]
        if isinstance(world_to_base, xr.DataArray):
            world_to_base = world_to_base.fusi.affine.reindex_voxels_like(reference)
        return xr.Dataset(
            data_vars, attrs={**output.attrs, "world_to_base": world_to_base}
        )

    # ── Tree helpers  ─────────────────────────────────────────────────────────────────

    def ancestors(self, region: int | str) -> list["treelib.Node"]:
        """Return the ancestor nodes of `region`, from root down (exclusive).

        Parameters
        ----------
        region : int or str
            Structure index or acronym.

        Returns
        -------
        list[treelib.Node]
            Ancestor nodes ordered from root toward `region`, not including `region`
            itself.
        """
        rid = _resolve_region_id(self.structures, region)
        tree = self.structures.tree
        level = tree.level(rid)
        return [tree.ancestor(rid, lvl) for lvl in range(level)]

    def show_tree(self, **kwargs: object) -> None:
        """Print the structure hierarchy tree.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to
            [`treelib.Tree.show`][treelib.Tree.show].

        Returns
        -------
        None
            The tree is printed to standard output.
        """
        kwargs.setdefault("stdout", False)
        print(self.structures.tree.show(**kwargs))  # ty: ignore[invalid-argument-type]


# ── Standalone atlas operations ─────────────────────────────────────────────────────────
#
# These free functions are the implementation behind the matching `AtlasAccessor` methods:
# each validates `ds` as an atlas, then operates on it, and the accessor method is a thin
# wrapper (`ds.atlas.get_mesh(...)` calls `get_mesh(ds, ...)`). Import them as
# `confusius.atlas.get_atlas_mesh` / `search_atlas` / `get_atlas_masks` to operate on a
# Dataset directly.


def search_atlas(
    ds: xr.Dataset,
    pattern: str,
    field: Literal["all", "acronym", "name"] = "all",
) -> pd.DataFrame:
    """Search an atlas Dataset's structures by name or acronym.

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset to search; validated as an atlas before use.
    pattern : str
        Substring or regex pattern.
    field : {"all", "acronym", "name"}, default: "all"
        Which column to search.

        - `"all"`: case-insensitive regex search on both `acronym` and `name`.
        - `"acronym"` / `"name"`: case-insensitive full regex match on that column
          only.

    Returns
    -------
    pandas.DataFrame
        Filtered view of the atlas structure lookup table matching the search criteria.

    Raises
    ------
    TypeError
        If `ds` is not a well-formed atlas Dataset.
    ValueError
        If `ds` is not a well-formed atlas Dataset.

    Examples
    --------
    >>> import confusius as cf
    >>> cf.atlas.search_atlas(ds, "visual cortex")
    >>> cf.atlas.search_atlas(ds, "VISp", field="acronym")
    """
    validate_atlas(ds)
    return ds.atlas.search(pattern, field)


def get_atlas_masks(
    ds: xr.Dataset,
    regions: int | str | Sequence[int | str],
    sides: (
        Literal["left", "right", "both"] | Sequence[Literal["left", "right", "both"]]
    ) = "both",
) -> xr.DataArray:
    """Return integer region masks stacked along a `mask` dimension.

    Each layer along `mask` has values in `{0, region_id}`; voxels belonging to the
    requested region (including all descendants in the hierarchy) carry the region's index,
    all others are zero.

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset; validated as an atlas before use.
    regions : int or str or sequence of int or str
        One or more regions, each given as a structure index or acronym.
    sides : {"left", "right", "both"} or sequence thereof, default: "both"
        Hemisphere filter. Pass a scalar to apply the same side to all regions, or a
        sequence of the same length as `regions` for per-region control.

    Returns
    -------
    xarray.DataArray
        Integer VoxelData array with dims `["mask", *annotation.dims]`.
        The `mask` coordinate holds the region acronym for each layer, suffixed with
        `_L`/`_R` when the
        corresponding `side` is `"left"`/`"right"` (left/right requests for the same
        region would otherwise share an acronym).

    Raises
    ------
    TypeError
        If `ds` is not a well-formed atlas Dataset.
    KeyError
        If any requested region acronym or index is not found in the atlas.
    ValueError
        If `ds` is not a well-formed atlas, if `sides` is a sequence whose length does not
        match `regions`, or if any element of `sides` is not `"left"`, `"right"`, or
        `"both"`.

    Examples
    --------
    >>> import confusius as cf
    >>> cf.atlas.get_atlas_masks(ds, "VISp")
    >>> cf.atlas.get_atlas_masks(ds, ["VISp", "AUDp"], sides=["left", "both"])
    """
    validate_atlas(ds)

    region_list: list[int | str] = (
        [regions] if isinstance(regions, (int, str)) else list(regions)
    )

    if isinstance(sides, str):
        side_list = [sides] * len(region_list)
    else:
        side_list = list(sides)
        if len(side_list) != len(region_list):
            raise ValueError(
                f"'sides' has {len(side_list)} elements but 'regions' has "
                f"{len(region_list)} elements; they must have the same length."
            )

    _valid_sides = {"left", "right", "both"}
    invalid = [s for s in side_list if s not in _valid_sides]
    if invalid:
        raise ValueError(
            f"Invalid side value(s): {invalid!r}. "
            f"Each element must be one of {sorted(_valid_sides)}."
        )

    annotation = ds["annotation"]
    hemispheres = ds["hemispheres"]
    structures = ds.attrs["structures"]

    annotation_np = annotation.values
    hemispheres_np = hemispheres.values
    left_value = hemispheres.attrs["left"]
    right_value = hemispheres.attrs["right"]

    layers = []
    acronyms = []
    for reg, s in zip(region_list, side_list):
        rid = _resolve_region_id(structures, reg)
        descendant_ids = _get_descendant_ids(structures, rid)

        layer = np.zeros_like(annotation_np, dtype=np.int32)
        # Using kind="table" here will use a lookup table approach that is much
        # faster at the cost of higher memory usage.
        layer[np.isin(annotation_np, descendant_ids, kind="table")] = rid

        acronym = structures[rid]["acronym"]
        if s == "left":
            layer[hemispheres_np != left_value] = 0
            acronym = f"{acronym}_L"
        elif s == "right":
            layer[hemispheres_np != right_value] = 0
            acronym = f"{acronym}_R"

        layers.append(annotation.copy(data=layer))
        acronyms.append(acronym)

    result = xr.concat(
        layers,
        dim=xr.DataArray(
            np.asarray(acronyms, dtype=np.str_), dims=("mask",), name="mask"
        ),
    )
    result.attrs = annotation.attrs.copy()
    return result


def get_atlas_mesh(
    ds: xr.Dataset,
    region: int | str,
    side: Literal["left", "right", "both"] = "both",
    *,
    clip: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Return vertex coordinates and face indices for a region's mesh.

    Reads the region's OBJ mesh, transforms its vertices from micron space to the atlas's
    world space (millimetres), then optionally drops out-of-grid vertices and clips to
    one hemisphere. The mesh comes from the structure's `mesh_filename`: for a freshly
    fetched atlas this points into the BrainGlobe cache; for an atlas loaded with
    [`load_atlas`][confusius.io.load_atlas] it points at the mesh bundled
    inside the store.

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset; validated as an atlas before use.
    region : int or str
        Structure index or acronym.
    side : {"left", "right", "both"}, default: "both"
        Hemisphere to include. `"both"` keeps the full mesh. `"left"` and `"right"` keep
        only vertices whose nearest `hemispheres` voxel carries that side's label
        (`hemispheres.attrs["left"]` / `["right"]`), sampled in the atlas's world space.
        Faces are kept only when all three of their vertices survive, so the cut face is
        not closed. Sampling the hemisphere map makes this orientation-agnostic and correct
        after an arbitrary resample.
    clip : bool, default: True
        Whether to clip the final mesh to the reference grid. If `False`, the mesh is still
        transformed to the atlas's world space, but the bounding box is not respected.

    Returns
    -------
    vertices : numpy.ndarray, shape (N, 3)
        Vertex coordinates in the atlas's world space (millimetres). After a nonlinear
        resample, vertices warped outside the reference grid are dropped.
    faces : numpy.ndarray, shape (M, 3)
        Zero-indexed triangle face indices (int32).

    Raises
    ------
    TypeError
        If `ds` is not a well-formed atlas Dataset.
    KeyError
        If the requested region is not found in the atlas.
    ValueError
        If `ds` is not a well-formed atlas, if the region has no mesh file, or if the mesh
        file cannot be located.

    Examples
    --------
    >>> import confusius as cf
    >>> vertices, faces = cf.atlas.get_atlas_mesh(ds, "root")
    """
    validate_atlas(ds, require_mesh_use=True)

    structures = ds.attrs["structures"]
    reference = ds["reference"]
    hemispheres = ds["hemispheres"]
    world_to_base = ds.attrs["world_to_base"]

    rid = _resolve_region_id(structures, region)
    info = structures[rid]

    mesh_filename = info.get("mesh_filename")
    if mesh_filename is None:
        raise ValueError(
            f"No mesh file available for region '{region}' (id {rid}). "
            "Not all BrainGlobe atlases include mesh files."
        )

    mesh_path = Path(mesh_filename)
    if not mesh_path.is_file():
        raise ValueError(
            f"Mesh file for region '{region}' (id {rid}) not found at {mesh_path}. "
            "A freshly fetched atlas reads meshes from the BrainGlobe cache; a loaded "
            "atlas reads them from the meshes bundled in its Zarr store."
        )

    mesh = structures[rid]["mesh"]
    vertices_um = mesh.points  # (N, 3) in microns
    faces = mesh.get_cells_type("triangle")

    vertices_mm = vertices_um * 1e-3  # Convert microns to millimetres.

    vertices_mm = _apply_world_to_base_transform(world_to_base, vertices_mm, reference)

    if clip:
        vertices_mm, faces = _drop_vertices_outside_grid(vertices_mm, faces, reference)

    if side != "both":
        from confusius.plotting._utils import (
            _materialize_axis_aligned_world_grid_for_display,
        )

        hemispheres_grid = _materialize_axis_aligned_world_grid_for_display(hemispheres)
        sel = {
            d: xr.DataArray(vertices_mm[:, i], dims="point")
            for i, d in enumerate("zyx")
        }
        side_value = hemispheres.attrs[side]
        hem_points = hemispheres_grid.sel(sel, method="nearest").compute()

        keep_idx = np.where(hem_points == side_value)[0]
        old_to_new = np.full(len(vertices_mm), -1, dtype=np.int64)
        old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

        new_face_idx = old_to_new[faces]  # (M, 3); -1 for dropped vertices.
        valid = np.all(new_face_idx >= 0, axis=1)

        vertices_mm = vertices_mm[keep_idx]
        faces = new_face_idx[valid].astype(np.int32)

    return vertices_mm, faces
