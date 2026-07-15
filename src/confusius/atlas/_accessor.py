"""The `.atlas` xarray Dataset accessor: data-aware brain-atlas operations."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius.atlas._physical_to_base_transform import (
    PhysicalToBaseTransform,
    _apply_physical_to_base_transform,
    _compose_physical_to_base_transforms,
    _drop_vertices_outside_grid,
)
from confusius.atlas._structures import (
    _build_lookup_df,
    _get_descendant_ids,
    _resolve_region_id,
)
from confusius.registration.resampling import resample_like as resample_like_da
from confusius.validation.atlas import validate_atlas_dataset

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
        Atlas Dataset with `reference`, `annotation`, and `hemispheres` data variables on
        a common `(z, y, x)` grid, and the atlas metadata in `attrs`.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds
        self._lookup: pd.DataFrame | None = None

    # ── Data properties ───────────────────────────────────────────────────────────────

    @property
    def reference(self) -> xr.DataArray:
        """Reference template DataArray.

        Returns
        -------
        xarray.DataArray
            The reference template DataArray.
        """
        return self._ds["reference"]

    @property
    def annotation(self) -> xr.DataArray:
        """Region annotations DataArray.

        `attrs["rgb_lookup"]` carries a `{id: [r, g, b]}` dict used for colormap
        construction.

        Returns
        -------
        xarray.DataArray
            The region annotation DataArray with integer labels.
        """
        return self._ds["annotation"]

    @property
    def hemispheres(self) -> xr.DataArray:
        """Hemisphere map DataArray (1 = left, 2 = right).

        Returns
        -------
        xarray.DataArray
            The hemisphere map data variable.
        """
        return self._ds["hemispheres"]

    @property
    def _physical_to_base_transform(self) -> PhysicalToBaseTransform:
        """Pull transform mapping the atlas's physical space back to base atlas space.

        Held directly in `attrs["physical_to_base"]` as either a `(4, 4)` numpy affine or a
        dense displacement-field DataArray after a nonlinear resample. It is returned as-is
        — the mesh-warping helpers consume the pull form (they invert the affine, or invert
        the field per point, when mapping base vertices into physical space).

        Returns
        -------
        numpy.ndarray or xarray.DataArray
            The `(4, 4)` pull affine, or the dense displacement-field DataArray.
        """
        return self._ds.attrs["physical_to_base"]

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

            - `"all"`: case-insensitive substring match on both `acronym`
              and `name`.
            - `"acronym"` / `"name"`: full regex match on that column only.

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
        return search(self._ds, pattern, field)

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
            Integer DataArray with dims `["mask", *annotation.dims]`. The `mask`
            coordinate holds the region acronym for each layer, suffixed with `_L`/`_R`
            when the corresponding `side` is `"left"`/`"right"` (left/right requests for
            the same region would otherwise share an acronym).

        Raises
        ------
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
        return get_masks(self._ds, regions, sides)

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
        DataArrays' current physical space (millimetres), then optionally drops
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
            (`hemispheres.attrs["left"]` / `["right"]`), sampled in the current physical
            space. Faces are kept only when all three of their vertices survive, so the
            cut face is not closed. Sampling the hemisphere map makes this
            orientation-agnostic and correct after an arbitrary resample.
        clip : bool, default: True
            Whether to clip the final mesh to the current reference grid. If `False`,
            the mesh will still be transformed to the current physical space, but the
            bounding box will not be respected.

        Returns
        -------
        vertices : numpy.ndarray, shape (N, 3)
            Vertex coordinates in the current physical space (millimetres). After a
            nonlinear resample, vertices warped outside the reference grid are dropped.
        faces : numpy.ndarray, shape (M, 3)
            Zero-indexed triangle face indices (int32).

        Raises
        ------
        KeyError
            If the requested region is not found in the atlas.
        ValueError
            If the region has no mesh file, or the mesh file cannot be located.
        """
        return get_mesh(self._ds, region, side, clip=clip)

    # ── Resampling ────────────────────────────────────────────────────────────────────

    def resample_like(
        self,
        reference: xr.DataArray,
        transform: PhysicalToBaseTransform,
        *,
        reference_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
        sitk_threads: int = -1,
    ) -> xr.Dataset:
        """Resample the atlas onto the grid of `reference`.

        Mirrors
        [`confusius.registration.resample_like`][confusius.registration.resample_like].
        Returns a new atlas Dataset whose variables live on `reference`'s grid.

        - `reference`: resampled with `reference_interpolation`.
        - `annotation` and `hemispheres`: resampled with nearest-neighbour
          to preserve integer labels.
        - Meshes returned by `get_mesh` will also be in the new physical space.

        `reference` and any DataArray transform must use the same physical coordinate
        units when such metadata is defined.

        Parameters
        ----------
        reference : xarray.DataArray
            Target grid. Must be 2D or 3D and must not have a `time` dimension.
        transform : (N+1, N+1) numpy.ndarray or xarray.DataArray
            Pull/inverse transform returned by `register_volume`, mapping `reference`
            physical coordinates to atlas physical coordinates.

            - **Affine** (`numpy.ndarray`): homogeneous matrix.
            - **B-spline** (`xarray.DataArray`): control-point DataArray.
            - **Displacement field** (`xarray.DataArray`): dense field with
              `attrs["type"] == "displacement_field_transform"`.
        reference_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
            Interpolation used for the `reference` variable.
        sitk_threads : int, default: -1
            Number of SimpleITK threads. Negative values use
            `max(1, cpu_count + 1 + sitk_threads)`.

        Returns
        -------
        xarray.Dataset
            New atlas Dataset on `reference`'s grid. The composed physical→base pull
            transform is stored in the `attrs["physical_to_base"]` attribute — a numpy
            affine, or a displacement-field DataArray when `transform` (or a previously
            composed one) is nonlinear.

        Examples
        --------
        >>> _, affine = atlas.atlas.reference.fusi.register.to_volume(
        ...     fusi_mean, metric="mattes_mi", transform="affine"
        ... )
        >>> atlas_fusi = atlas.atlas.resample_like(fusi_mean, affine)
        """
        resampled_ref = resample_like_da(
            self.reference,
            reference,
            transform,
            interpolation=reference_interpolation,
            fill_value=0.0,
            sitk_threads=sitk_threads,
        )
        resampled_ann = resample_like_da(
            self.annotation,
            reference,
            transform,
            interpolation="nearest",
            fill_value=0,
            sitk_threads=sitk_threads,
        )
        resampled_ann.attrs = self.annotation.attrs.copy()

        resampled_hemi = resample_like_da(
            self.hemispheres,
            reference,
            transform,
            interpolation="nearest",
            fill_value=0,
            sitk_threads=sitk_threads,
        )

        composed = _compose_physical_to_base_transforms(
            self._physical_to_base_transform, transform, reference, self.reference
        )

        data_vars: dict[str, xr.DataArray] = {
            "reference": resampled_ref,
            "annotation": resampled_ann,
            "hemispheres": resampled_hemi,
        }
        # `composed` is the pull (physical→base) transform: a numpy affine when both inputs
        # are affine, otherwise a dense displacement-field DataArray. Either rides directly
        # in the single `physical_to_base` attr.
        new_attrs = dict(self._ds.attrs)
        new_attrs["physical_to_base"] = composed

        return xr.Dataset(data_vars, attrs=new_attrs)

    def resample(
        self,
        transform: PhysicalToBaseTransform,
        *,
        shape: Sequence[int],
        spacing: Sequence[float],
        origin: Sequence[float],
        dims: Sequence[str],
        reference_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
        sitk_threads: int = -1,
    ) -> xr.Dataset:
        """Resample the atlas onto an explicit output grid.

        Mirrors
        [`confusius.registration.resample_volume`][confusius.registration.resample_volume]
        for atlas Datasets by constructing a temporary reference grid and delegating to
        [`resample_like`][confusius.atlas.AtlasAccessor.resample_like].

        Parameters
        ----------
        transform : (N+1, N+1) numpy.ndarray or xarray.DataArray
            Pull transform mapping output physical coordinates to the current atlas
            physical coordinates. Affine, B-spline, or displacement-field, as for
            [`resample_like`][confusius.atlas.AtlasAccessor.resample_like].
        shape : sequence of int
            Number of voxels along each output axis, in `dims` order.
        spacing : sequence of float
            Voxel spacing along each output axis, in `dims` order.
        origin : sequence of float
            Physical origin along each output axis, in `dims` order.
        dims : sequence of str
            Dimension names of the output atlas grid.
        reference_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
            Interpolation used for the `reference` volume.
        sitk_threads : int, default: -1
            Number of SimpleITK threads.

        Returns
        -------
        xarray.Dataset
            Resampled atlas Dataset on the requested grid.
        """
        coords = {}
        for i, dim in enumerate(dims):
            dim_str = str(dim)
            coord_attrs = {}
            if dim_str in self.reference.coords:
                coord_attrs = self.reference.coords[dim_str].attrs.copy()
            coord_attrs["voxdim"] = float(spacing[i])
            coords[dim_str] = xr.Variable(
                dim_str,
                np.asarray(origin[i]) + np.arange(shape[i]) * np.asarray(spacing[i]),
                attrs=coord_attrs,
            )

        reference = xr.DataArray(
            np.zeros(tuple(shape), dtype=np.float32),
            dims=[str(dim) for dim in dims],
            coords=coords,
        )
        return self.resample_like(
            reference,
            transform,
            reference_interpolation=reference_interpolation,
            sitk_threads=sitk_threads,
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
# `confusius.atlas.get_mesh` / `search` / `get_masks` to operate on a Dataset directly.


def search(
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

        - `"all"`: case-insensitive substring match on both `acronym` and `name`.
        - `"acronym"` / `"name"`: full regex match on that column only.

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
    >>> cf.atlas.search(ds, "visual cortex")
    >>> cf.atlas.search(ds, "VISp", field="acronym")
    """
    validate_atlas_dataset(ds)
    df = _build_lookup_df(ds.attrs["structures"])
    if field == "acronym":
        mask = df["acronym"].str.fullmatch(pattern)
    elif field == "name":
        mask = df["name"].str.fullmatch(pattern, case=False)
    else:
        mask = df["acronym"].str.contains(pattern, case=False, na=False) | df[
            "name"
        ].str.contains(pattern, case=False, na=False)
    return df[mask]


def get_masks(
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
        Integer DataArray with dims `["mask", *annotation.dims]`. The `mask` coordinate
        holds the region acronym for each layer, suffixed with `_L`/`_R` when the
        corresponding `side` is `"left"`/`"right"` (left/right requests for the same
        region would otherwise share an acronym).

    Raises
    ------
    KeyError
        If any requested region acronym or index is not found in the atlas.
    ValueError
        If `ds` is not a well-formed atlas, if `sides` is a sequence whose length does not
        match `regions`, or if any element of `sides` is not `"left"`, `"right"`, or
        `"both"`.

    Examples
    --------
    >>> import confusius as cf
    >>> cf.atlas.get_masks(ds, "VISp")
    >>> cf.atlas.get_masks(ds, ["VISp", "AUDp"], sides=["left", "both"])
    """
    validate_atlas_dataset(ds)

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

        layers.append(layer)
        acronyms.append(acronym)

    stacked = np.stack(layers, axis=0)

    spatial_dims = [str(dim) for dim in annotation.dims]
    spatial_coords = {dim: annotation.coords[dim] for dim in spatial_dims}
    return xr.DataArray(
        stacked,
        dims=["mask", *spatial_dims],
        coords={"mask": acronyms, **spatial_coords},
        attrs=annotation.attrs.copy(),
    )


def get_mesh(
    ds: xr.Dataset,
    region: int | str,
    side: Literal["left", "right", "both"] = "both",
    *,
    clip: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Return vertex coordinates and face indices for a region's mesh.

    Reads the region's OBJ mesh, transforms its vertices from micron space to the atlas's
    physical space (millimetres), then optionally drops out-of-grid vertices and clips to
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
        (`hemispheres.attrs["left"]` / `["right"]`), sampled in the atlas's physical space.
        Faces are kept only when all three of their vertices survive, so the cut face is
        not closed. Sampling the hemisphere map makes this orientation-agnostic and correct
        after an arbitrary resample.
    clip : bool, default: True
        Whether to clip the final mesh to the reference grid. If `False`, the mesh is still
        transformed to the atlas's physical space, but the bounding box is not respected.

    Returns
    -------
    vertices : numpy.ndarray, shape (N, 3)
        Vertex coordinates in the atlas's physical space (millimetres). After a nonlinear
        resample, vertices warped outside the reference grid are dropped.
    faces : numpy.ndarray, shape (M, 3)
        Zero-indexed triangle face indices (int32).

    Raises
    ------
    KeyError
        If the requested region is not found in the atlas.
    ValueError
        If `ds` is not a well-formed atlas, if the region has no mesh file, or if the mesh
        file cannot be located.

    Examples
    --------
    >>> import confusius as cf
    >>> vertices, faces = cf.atlas.get_mesh(ds, "root")
    """
    validate_atlas_dataset(ds, require_mesh_use=True)

    structures = ds.attrs["structures"]
    reference = ds["reference"]
    hemispheres = ds["hemispheres"]
    physical_to_base = ds.attrs["physical_to_base"]

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

    # defer loading mesh to BrainGlobe's structured dict
    mesh = structures[rid]["mesh"]
    vertices_um = mesh.points  # (N, 3) in microns
    faces = mesh.get_cells_type("triangle")

    vertices_mm = vertices_um * 1e-3  # Convert microns to millimetres.

    vertices_m = _apply_physical_to_base_transform(
        physical_to_base, vertices_mm, reference
    )

    if clip:
        vertices_m, faces = _drop_vertices_outside_grid(vertices_m, faces, reference)

    if side != "both":
        sel = {
            d: xr.DataArray(vertices_m[:, i], dims="point") for i, d in enumerate("zyx")
        }
        side_value = hemispheres.attrs[side]
        hem_points = hemispheres.sel(sel, method="nearest").compute()

        keep_idx = np.where(hem_points == side_value)[0]
        old_to_new = np.full(len(vertices_m), -1, dtype=np.int64)
        old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

        new_face_idx = old_to_new[faces]  # (M, 3); -1 for dropped vertices.
        valid = np.all(new_face_idx >= 0, axis=1)

        vertices_m = vertices_m[keep_idx]
        faces = new_face_idx[valid].astype(np.int32)

    return vertices_m, faces
