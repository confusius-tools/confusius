"""The `.atlas` xarray Dataset accessor: data-aware brain-atlas operations."""

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
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
        regions: int | str | Sequence[int | str],
        sides: (
            Literal["left", "right", "both"]
            | Sequence[Literal["left", "right", "both"]]
        ) = "both",
        *,
        clip: bool = True,
    ) -> dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]]:
        """Return one surface mesh per requested region, keyed by acronym.

        Reads each region's mesh, transforms its vertices from micron space to the
        DataArrays' current world space (millimetres), then optionally drops
        out-of-grid vertices and clips to one hemisphere. The mesh comes from the
        structure's `mesh_filename`: for a freshly fetched atlas this points into the
        BrainGlobe cache; for an atlas loaded with
        [`load_atlas`][confusius.io.load_atlas] it points at the mesh bundled
        inside the store.

        Parameters
        ----------
        regions : int or str or sequence of int or str
            One or more regions, each given as a structure index or acronym.
        sides : {"left", "right", "both"} or sequence thereof, default: "both"
            Hemisphere filter. Pass a scalar to apply the same side to all regions, or a
            sequence of the same length as `regions` for per-region control. `"both"`
            keeps the full mesh. `"left"` and `"right"` keep only vertices whose nearest
            `hemispheres` voxel carries that side's label (`hemispheres.attrs["left"]` /
            `["right"]`), sampled in the current world space. Faces are kept only when
            all three of their vertices survive, so the cut face is not closed. Sampling
            the hemisphere map makes this orientation-agnostic and correct after an
            arbitrary resample.
        clip : bool, default: True
            Whether to clip each mesh to the current reference grid. Unclipped meshes
            are still transformed to the current world space but may extend past the
            grid's bounding box.

        Returns
        -------
        dict[str, tuple[(N, 3) numpy.ndarray, (M, 3) numpy.ndarray]]
            One `(vertices, faces)` pair per requested region, keyed by the region's
            acronym, suffixed with `_L`/`_R` when the corresponding `side` is
            `"left"`/`"right"` (left/right requests for the same region would otherwise
            share a key). `vertices` holds coordinates in the current world space
            (millimetres); after a nonlinear resample, vertices warped outside the
            reference grid are dropped. `faces` holds zero-indexed triangle indices
            (int32).

        Raises
        ------
        TypeError
            If `ds` is not a well-formed atlas Dataset.
        KeyError
            If any requested region is not found in the atlas.
        ValueError
            If `sides` is a sequence whose length does not match `regions`, if any element
            of `sides` is not `"left"`, `"right"`, or `"both"`, or if a region has no mesh
            file.
        RuntimeError
            If a region's mesh cannot be downloaded or read.

        Examples
        --------
        >>> vertices, faces = ds.atlas.get_mesh("VISp")["VISp"]
        >>> ds.atlas.get_mesh(["VISp", "AUDp"], sides=["left", "both"]).keys()
        dict_keys(['VISp_L', 'AUDp'])
        """
        return get_atlas_mesh(self._ds, regions, sides, clip=clip)

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


def _normalize_regions_and_sides(
    regions: int | str | Sequence[int | str],
    sides: (
        Literal["left", "right", "both"] | Sequence[Literal["left", "right", "both"]]
    ),
) -> tuple[list[int | str], list[str]]:
    """Broadcast `regions` and `sides` into two validated, equal-length lists.

    Shared by [`get_atlas_masks`][confusius.atlas.get_atlas_masks] and
    [`get_atlas_mesh`][confusius.atlas.get_atlas_mesh], which both accept a single region
    or a sequence, with either a single side applied to all of them or one side per region.

    Parameters
    ----------
    regions : int or str or sequence of int or str
        One or more regions, each given as a structure index or acronym.
    sides : {"left", "right", "both"} or sequence thereof
        Hemisphere filter, either a scalar applied to every region or a sequence of the
        same length as `regions`.

    Returns
    -------
    region_list : list[int | str]
        The requested regions as a list.
    side_list : list[str]
        One side per region, in `region_list` order.

    Raises
    ------
    ValueError
        If `sides` is a sequence whose length does not match `regions`, or if any element
        of `sides` is not `"left"`, `"right"`, or `"both"`.
    """
    region_list: list[int | str]
    if isinstance(regions, (int, str)):
        region_list = [regions]
    elif isinstance(regions, np.integer):
        region_list = [int(regions)]
    else:
        region_list = list(regions)
    if not region_list:
        raise ValueError("'regions' must contain at least one region.")

    side_list: list[str]
    if isinstance(sides, str):
        side_list = [sides] * len(region_list)
    else:
        side_list = list(sides)
        if len(side_list) != len(region_list):
            raise ValueError(
                f"'sides' has {len(side_list)} elements but 'regions' has "
                f"{len(region_list)} elements; they must have the same length."
            )

    valid_sides = {"left", "right", "both"}
    invalid = [s for s in side_list if s not in valid_sides]
    if invalid:
        raise ValueError(
            f"Invalid side value(s): {invalid!r}. "
            f"Each element must be one of {sorted(valid_sides)}."
        )

    return region_list, side_list


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


_MASK_BITMASK_DTYPE = np.uint8
"""Bitmask dtype for `get_atlas_masks`'s batched region lookup.

Benchmarked against `uint8`/`uint16`/`uint32`/`uint64`: `uint8` (8 regions per batch)
wins consistently, ~1.2-2.2x over a plain `isin` loop. Wider dtypes batch more regions
per gather but move more bytes per voxel, and lose to `isin`'s own single-pass table
lookup past `uint16`.
"""
_MASK_BITMASK_BATCH_SIZE = np.iinfo(_MASK_BITMASK_DTYPE).bits


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

    region_list, side_list = _normalize_regions_and_sides(regions, sides)

    annotation = ds["annotation"]
    hemispheres = ds["hemispheres"]
    structures = ds.attrs["structures"]

    annotation_np = annotation.values
    hemispheres_np = hemispheres.values
    left_value = hemispheres.attrs["left"]
    right_value = hemispheres.attrs["right"]

    resolved = [
        (_resolve_region_id(structures, reg), s)
        for reg, s in zip(region_list, side_list)
    ]
    max_structure_id = max(int(sid) for sid in structures)

    layers = []
    acronyms = []
    # Requesting many regions independently isin()-scanned the full annotation volume
    # once per region. Instead, pack up to _MASK_BITMASK_BATCH_SIZE regions' descendant-id
    # sets into a single bitmask lookup table and gather it over the annotation array once
    # per batch, so N regions cost ceil(N / batch size) full-array gathers instead of N.
    for batch_start in range(0, len(resolved), _MASK_BITMASK_BATCH_SIZE):
        batch = resolved[batch_start : batch_start + _MASK_BITMASK_BATCH_SIZE]

        id_to_bitmask = np.zeros(max_structure_id + 1, dtype=_MASK_BITMASK_DTYPE)
        for bit, (rid, _) in enumerate(batch):
            descendant_ids = _get_descendant_ids(structures, rid)
            id_to_bitmask[descendant_ids] |= _MASK_BITMASK_DTYPE(1 << bit)
        voxel_bitmask = id_to_bitmask[annotation_np]

        for bit, (rid, s) in enumerate(batch):
            layer = np.zeros_like(annotation_np, dtype=np.int32)
            layer[(voxel_bitmask & _MASK_BITMASK_DTYPE(1 << bit)) != 0] = rid

            acronym = structures[rid]["acronym"]
            if s == "left":
                layer[hemispheres_np != left_value] = 0
                acronym = f"{acronym}_L"
            elif s == "right":
                layer[hemispheres_np != right_value] = 0
                acronym = f"{acronym}_R"

            layers.append(annotation.copy(data=layer))
            acronyms.append(acronym)

    # Every layer is annotation.copy(data=...), so they share one k/j/i grid and one
    # VoxelToWorldIndex; xr.concat's default compat="equals" would otherwise recompute
    # and compare the full lazily derived z/y/x world-coordinate grid across every
    # layer pair, which costs orders of magnitude more than building the layers
    # themselves. coords="minimal"/compat="override" skips that redundant check.
    result = xr.concat(
        layers,
        dim=xr.DataArray(
            np.asarray(acronyms, dtype=np.str_), dims=("mask",), name="mask"
        ),
        coords="minimal",
        compat="override",
    )
    result.attrs = annotation.attrs.copy()
    return result


def get_atlas_mesh(
    ds: xr.Dataset,
    regions: int | str | Sequence[int | str],
    sides: (
        Literal["left", "right", "both"] | Sequence[Literal["left", "right", "both"]]
    ) = "both",
    *,
    clip: bool = True,
) -> dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]]:
    """Return one surface mesh per requested region, keyed by acronym.

    Reads each region's mesh, transforms its vertices from micron space to the atlas's
    world space (millimetres), then optionally drops out-of-grid vertices and clips to
    one hemisphere. The mesh comes from the structure's `mesh_filename`: for a freshly
    fetched atlas this points into the BrainGlobe cache; for an atlas loaded with
    [`load_atlas`][confusius.io.load_atlas] it points at the mesh bundled
    inside the store.

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset; validated as an atlas before use.
    regions : int or str or sequence of int or str
        One or more regions, each given as a structure index or acronym.
    sides : {"left", "right", "both"} or sequence thereof, default: "both"
        Hemisphere filter. Pass a scalar to apply the same side to all regions, or a
        sequence of the same length as `regions` for per-region control. `"both"` keeps
        the full mesh. `"left"` and `"right"` keep only vertices whose nearest
        `hemispheres` voxel carries that side's label (`hemispheres.attrs["left"]` /
        `["right"]`), sampled in the atlas's world space. Faces are kept only when all
        three of their vertices survive, so the cut face is not closed. Sampling the
        hemisphere map makes this orientation-agnostic and correct after an arbitrary
        resample.
    clip : bool, default: True
        Whether to clip each mesh to the reference grid. Unclipped meshes are still
        transformed to the atlas's world space but may extend past the grid's bounding
        box.

    Returns
    -------
    dict[str, tuple[(N, 3) numpy.ndarray, (M, 3) numpy.ndarray]]
        One `(vertices, faces)` pair per requested region, keyed by the region's acronym,
        suffixed with `_L`/`_R` when the corresponding `side` is `"left"`/`"right"`
        (left/right requests for the same region would otherwise share a key). `vertices`
        holds coordinates in the atlas's world space (millimetres); after a nonlinear
        resample, vertices warped outside the reference grid are dropped. `faces` holds
        zero-indexed triangle indices (int32).

    Raises
    ------
    TypeError
        If `ds` is not a well-formed atlas Dataset.
    KeyError
        If any requested region is not found in the atlas.
    ValueError
        If `ds` is not a well-formed atlas, if `sides` is a sequence whose length does not
        match `regions`, if any element of `sides` is not `"left"`, `"right"`, or `"both"`,
        or if a region has no mesh file.
    RuntimeError
        If a region's mesh cannot be downloaded or read.

    Examples
    --------
    >>> import confusius as cf
    >>> vertices, faces = cf.atlas.get_atlas_mesh(ds, "root")["root"]
    >>> cf.atlas.get_atlas_mesh(ds, ["VISp", "AUDp"], sides=["left", "both"]).keys()
    dict_keys(['VISp_L', 'AUDp'])
    """
    validate_atlas(ds, require_mesh_use=True)

    region_list, side_list = _normalize_regions_and_sides(regions, sides)

    structures = ds.attrs["structures"]
    reference = ds["reference"]
    hemispheres = ds["hemispheres"]
    world_to_base = ds.attrs["world_to_base"]

    def _get_single_mesh(
        region: int | str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        rid = _resolve_region_id(structures, region)
        info = structures[rid]

        mesh_filename = info.get("mesh_filename")
        if mesh_filename is None:
            raise ValueError(
                f"No mesh file available for region '{region}' (id {rid}). "
                "Not all BrainGlobe atlases include mesh files."
            )

        # fetch_brainglobe_atlas prefetches every mesh, so this normally reads a cached
        # file. BrainGlobe still downloads lazily on this access when the file is
        # absent: prefetch skipped offline, or a store saved before meshes were cached.
        try:
            mesh = structures[rid]["mesh"]
        except RuntimeError as error:
            raise RuntimeError(
                f"Could not load the mesh for region '{region}' (id {rid}): {error} "
                "BrainGlobe may provide no mesh for this region, the mesh may not be "
                "cached while offline, or, for an atlas loaded with load_atlas, it may "
                "not have been downloaded before save_atlas ran."
            ) from error
        vertices_um = mesh.points  # (N, 3) in microns
        faces = mesh.get_cells_type("triangle")

        vertices_mm = vertices_um * 1e-3  # Convert microns to millimetres.
        return vertices_mm, faces

    region_ids = [_resolve_region_id(structures, r) for r in region_list]
    vertices_per_region, faces_per_region = zip(
        *(_get_single_mesh(r) for r in region_list)
    )

    vertices_sections = np.cumsum([v.shape[0] for v in vertices_per_region[:-1]])

    # One transform call for all regions: the base transform can be nonlinear, and
    # evaluating it once on the concatenated vertices is much cheaper than per region.
    vertices_mm = _apply_world_to_base_transform(
        world_to_base, np.concatenate(vertices_per_region, axis=0), reference
    )

    vertices_per_region = np.split(vertices_mm, vertices_sections, axis=0)

    if any(side != "both" for side in side_list):
        from confusius.plotting._utils import (
            materialize_axis_aligned_world_grid_for_display,
        )

        hemispheres_grid = materialize_axis_aligned_world_grid_for_display(hemispheres)
    else:
        hemispheres_grid = hemispheres

    mesh_dict: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]] = {}
    acronyms = [structures[rid]["acronym"] for rid in region_ids]
    for side, region_vertices, region_faces, acronym in zip(
        side_list, vertices_per_region, faces_per_region, acronyms
    ):
        if side != "both":
            sel = {
                d: xr.DataArray(region_vertices[:, i], dims="point")
                for i, d in enumerate("zyx")
            }
            side_value = hemispheres.attrs[side]
            hem_points = hemispheres_grid.sel(sel, method="nearest").compute()

            keep_idx = np.where(hem_points == side_value)[0]
            old_to_new = np.full(len(region_vertices), -1, dtype=np.int64)
            old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

            new_face_idx = old_to_new[region_faces]  # (M, 3); -1 for dropped vertices.
            valid = np.all(new_face_idx >= 0, axis=1)

            region_vertices = region_vertices[keep_idx]
            region_faces = new_face_idx[valid].astype(np.int32)
            acronym = f"{acronym}_{side[0].upper()}"

        if clip:
            region_vertices, region_faces = _drop_vertices_outside_grid(
                region_vertices, region_faces, reference
            )

        mesh_dict[acronym] = (region_vertices, region_faces)

    return mesh_dict
