"""The `.atlas` xarray Dataset accessor: data-aware brain-atlas operations."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius.atlas._structures import (
    _build_lookup_df,
    _get_descendant_ids,
    _load_obj,
    _resolve_region_id,
    structures_from_json,
)
from confusius.registration.resampling import resample_like as resample_like_da

if TYPE_CHECKING:
    import treelib
    from brainglobe_atlasapi.structure_class import StructuresDict
    from matplotlib.colors import BoundaryNorm, ListedColormap


@xr.register_dataset_accessor("atlas")
class AtlasAccessor:
    """Brain-atlas operations on an atlas `xarray.Dataset`.

    Registered as the `.atlas` namespace on any Dataset produced by
    [`atlas_from_brainglobe`][confusius.atlas.atlas_from_brainglobe] or
    [`atlas_from_zarr`][confusius.atlas.atlas_from_zarr]. The structure hierarchy is
    lazily rebuilt from `Dataset.attrs["structures"]`, so structural queries keep working
    for as long as that attribute rides along (xarray drops `attrs` on many ops by
    default; use `xarray.set_options(keep_attrs=True)` in pipelines).

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset with `reference`, `annotation`, and `hemispheres` data variables on
        a common `(z, y, x)` grid, and the atlas metadata in `attrs`.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds
        self._structures: StructuresDict | None = None
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

    # ── Structure metadata ────────────────────────────────────────────────────

    @property
    def structures(self) -> "StructuresDict":
        """Structure dictionary rebuilt from `Dataset.attrs["structures"]`.

        Returns
        -------
        brainglobe_atlasapi.structure_class.StructuresDict
            The structure dictionary with its hierarchy tree. Parsed once and cached.

        Raises
        ------
        KeyError
            If `Dataset.attrs` has no `structures` entry (e.g. after an xarray op that
            dropped `attrs`; wrap the pipeline in `xarray.set_options(keep_attrs=True)`).
        """
        if self._structures is None:
            if "structures" not in self._ds.attrs:
                raise KeyError(
                    "This Dataset has no 'structures' attribute, so its structure "
                    "hierarchy cannot be rebuilt. xarray drops attrs on many operations "
                    "by default; run atlas pipelines under "
                    "xarray.set_options(keep_attrs=True)."
                )
            self._structures = structures_from_json(self._ds.attrs["structures"])
        return self._structures

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
        df = self.lookup
        if field == "acronym":
            mask = df["acronym"].str.fullmatch(pattern)
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
            Integer DataArray with dims `["mask", "z", "y", "x"]`. The
            `mask` coordinate holds the region acronym for each layer, suffixed with
            `_L`/`_R` when the corresponding `side` is `"left"`/`"right"` (left/right
            requests for the same region would otherwise share an acronym).

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

        annotation_np = self.annotation.values
        hemispheres_np = self.hemispheres.values

        layers = []
        acronyms = []
        for reg, s in zip(region_list, side_list):
            rid = _resolve_region_id(self.structures, reg)
            descendant_ids = _get_descendant_ids(self.structures, rid)

            layer = np.zeros_like(annotation_np, dtype=np.int32)
            # Using kind="table" here will use a lookup table approach that is much
            # faster at the cost of higher memory usage.
            layer[np.isin(annotation_np, descendant_ids, kind="table")] = rid

            acronym = self.structures[rid]["acronym"]
            if s == "left":
                layer[hemispheres_np != 1] = 0
                acronym = f"{acronym}_L"
            elif s == "right":
                layer[hemispheres_np != 2] = 0
                acronym = f"{acronym}_R"

            layers.append(layer)
            acronyms.append(acronym)

        stacked = np.stack(layers, axis=0)

        spatial_coords = {d: self.annotation.coords[d] for d in ["z", "y", "x"]}
        return xr.DataArray(
            stacked,
            dims=["mask", "z", "y", "x"],
            coords={"mask": acronyms, **spatial_coords},
            attrs=self.annotation.attrs.copy(),
        )

    # ── Meshes ────────────────────────────────────────────────────────────────────────

    def get_mesh(
        self,
        region: int | str,
        side: Literal["left", "right", "both"] = "both",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """Return vertex coordinates and face indices for a region's mesh.

        Reads the region's OBJ file, optionally clips to one hemisphere, then transforms
        vertices from micron space to the DataArrays' current physical space
        (millimetres). The OBJ path comes from the structure's `mesh_filename`: for a
        freshly fetched atlas this points into the BrainGlobe cache; for an atlas loaded
        with [`atlas_from_zarr`][confusius.atlas.atlas_from_zarr] it points at the mesh
        bundled inside the store.

        Parameters
        ----------
        region : int or str
            Structure index or acronym.
        side : {"left", "right", "both"}, default: "both"
            Hemisphere to include. `"both"` keeps the full mesh. `"left"` and
            `"right"` clip in the original atlas micron space along the RL axis at the
            volume midline. Only triangles whose three vertices all fall on the
            requested side are retained; the cut face is not closed.

            !!! note
               Generalising axis detection from the orientation attribute for non-`asr`
               atlases is not yet implemented.

        Returns
        -------
        vertices : numpy.ndarray, shape (N, 3)
            Vertex coordinates in the current physical space (millimetres).
        faces : numpy.ndarray, shape (M, 3)
            Zero-indexed triangle face indices (int32).

        Raises
        ------
        KeyError
            If the requested region is not found in the atlas.
        ValueError
            If the region has no mesh file, or the mesh file cannot be located.
        """
        rid = _resolve_region_id(self.structures, region)
        info = self.structures[rid]

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

        vertices_um, faces = _load_obj(mesh_path)

        if side != "both":
            # Clip in micron space along the RL axis (column 2 for asr
            # orientation) before applying the physical transform.
            # For asr, axis 2 increases from right (0) to left (max), so:
            #   right hemisphere → RL < midline
            #   left  hemisphere → RL >= midline
            # TODO: generalize axis detection for non-asr atlases.
            rl_midline_um = self._ds.attrs["rl_midline_um"]
            if side == "right":
                keep = vertices_um[:, 2] < rl_midline_um
            else:  # "left"
                keep = vertices_um[:, 2] >= rl_midline_um

            keep_idx = np.where(keep)[0]
            old_to_new = np.full(len(vertices_um), -1, dtype=np.int64)
            old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

            # Retain only faces where all three vertices survive the clip.
            new_face_idx = old_to_new[faces]  # (M, 3); -1 if vertex removed.
            valid = np.all(new_face_idx >= 0, axis=1)
            vertices_um = vertices_um[keep_idx]
            faces = new_face_idx[valid].astype(np.int32)

        # Apply homogeneous transform: microns → physical millimetres.
        # mesh_to_physical maps [x_um, y_um, z_um, 1]^T → [x_mm, y_mm, z_mm, 1]^T.
        mesh_to_physical = np.asarray(self._ds.attrs["mesh_to_physical"])
        n = len(vertices_um)
        vertices_h = np.hstack([vertices_um, np.ones((n, 1), dtype=np.float64)])
        vertices_m = (mesh_to_physical @ vertices_h.T).T[:, :3]

        return vertices_m, faces

    # ── Resampling ────────────────────────────────────────────────────────────────────

    def resample_like(
        self,
        reference: xr.DataArray,
        transform: "npt.NDArray[np.float64]",
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

        Parameters
        ----------
        reference : xarray.DataArray
            Target grid. Must be 2D or 3D and must not have a `time` dimension.
        transform : (N+1, N+1) numpy.ndarray
            Pull/inverse affine returned by `register_volume`, mapping
            `reference` physical coordinates to atlas physical coordinates.
        reference_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
            Interpolation used for the `reference` variable.
        sitk_threads : int, default: -1
            Number of SimpleITK threads. Negative values use
            `max(1, cpu_count + 1 + sitk_threads)`.

        Returns
        -------
        xarray.Dataset
            New atlas Dataset on `reference`'s grid.

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
            default_value=0.0,
            sitk_threads=sitk_threads,
        )
        resampled_ann = resample_like_da(
            self.annotation,
            reference,
            transform,
            interpolation="nearest",
            default_value=0,
            sitk_threads=sitk_threads,
        )
        resampled_ann.attrs = self.annotation.attrs.copy()

        resampled_hemi = resample_like_da(
            self.hemispheres,
            reference,
            transform,
            interpolation="nearest",
            default_value=0,
            sitk_threads=sitk_threads,
        )

        new_attrs = dict(self._ds.attrs)
        # The mesh→physical affine composes with the resampling pull transform;
        # rl_midline_um is a property of the original atlas micron space and is unchanged.
        new_attrs["mesh_to_physical"] = (
            np.linalg.inv(transform) @ np.asarray(self._ds.attrs["mesh_to_physical"])
        ).tolist()

        return xr.Dataset(
            {
                "reference": resampled_ref,
                "annotation": resampled_ann,
                "hemispheres": resampled_hemi,
            },
            attrs=new_attrs,
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
        print(self.structures.tree.show(**kwargs))
