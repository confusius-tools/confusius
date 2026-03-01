"""Atlas class for brain atlas integration via BrainGlobe."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from confusius.atlas._structures import (
    _build_atlas_cmap_and_norm,
    _build_lookup_df,
    _build_rgb_lookup,
    _get_descendant_ids,
    _load_obj,
    _resolve_region_id,
)
from confusius.registration.resampling import resample_like as resample_like_da

if TYPE_CHECKING:
    import treelib
    from brainglobe_atlasapi import BrainGlobeAtlas
    from brainglobe_atlasapi.structure_class import StructuresDict
    from matplotlib.colors import BoundaryNorm, ListedColormap


def _build_dataset(bg_atlas: "BrainGlobeAtlas") -> xr.Dataset:
    """Build an Xarray Dataset from a BrainGlobe atlas.

    Parameters
    ----------
    bg_atlas : brainglobe_atlasapi.BrainGlobeAtlas
        Loaded BrainGlobe atlas.

    Returns
    -------
    xarray.Dataset
        Dataset with variables `reference`, `annotation`, and
        `hemispheres`, each with physical coordinates in millimetres.
    """
    meta = bg_atlas.metadata
    resolution_mm = [r * 1e-3 for r in meta["resolution"]]
    shape = meta["shape"]

    coords = {
        dim: (
            np.arange(shape[i]) * resolution_mm[i],
            {"voxdim": resolution_mm[i], "units": "mm"},
        )
        for i, dim in enumerate(["z", "y", "x"])
    }

    rgb_lookup = _build_rgb_lookup(bg_atlas.structures)
    cmap, norm = _build_atlas_cmap_and_norm(rgb_lookup)

    reference = xr.DataArray(
        bg_atlas.reference.astype(np.float32),
        dims=["z", "y", "x"],
        coords={d: xr.Variable(d, v, attrs=a) for d, (v, a) in coords.items()},
        attrs={"cmap": "gray"},
    )

    annotation = xr.DataArray(
        bg_atlas.annotation.astype(np.int32),
        dims=["z", "y", "x"],
        coords={d: xr.Variable(d, v, attrs=a) for d, (v, a) in coords.items()},
        # cmap and norm are non-serializable but are skipped automatically when saving
        # to zarr; rgb_lookup is the serializable source of truth.
        attrs={"rgb_lookup": rgb_lookup, "cmap": cmap, "norm": norm},
    )

    hemispheres = xr.DataArray(
        bg_atlas.hemispheres.astype(np.int8),
        dims=["z", "y", "x"],
        coords={d: xr.Variable(d, v, attrs=a) for d, (v, a) in coords.items()},
    )

    return xr.Dataset(
        {
            "reference": reference,
            "annotation": annotation,
            "hemispheres": hemispheres,
        },
        attrs={
            "name": meta["name"],
            "species": meta["species"],
            "orientation": meta["orientation"],
        },
    )


class Atlas:
    """Brain atlas wrapper backed by BrainGlobe, exposing DataArrays.

    Use [`from_brainglobe`][confusius.atlas.Atlas.from_brainglobe] to construct an
    instance.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with variables `reference`, `annotation`, and
        `hemispheres` on a common `(z, y, x)` grid with physical coordinates
        in millimetres.
    structures : brainglobe_atlasapi.structure_class.StructuresDict
        BrainGlobe structure dictionary (carries the `treelib` hierarchy tree).
    mesh_to_physical : (4, 4) numpy.ndarray
        Homogeneous affine that maps mesh vertex coordinates (microns, atlas
        voxel space) to the DataArrays' physical space (millimetres).
    rl_midline_um : float, default: 0.0
        Midpoint of the RL axis in microns (atlas voxel space). Used to clip
        mesh vertices to a single hemisphere.

    Attributes
    ----------
    reference : xarray.DataArray
        Reference template DataArray.
    annotation : xarray.DataArray
        Region annotations DataArray with integer labels.
    hemispheres : xarray.DataArray
        Hemisphere map DataArray (1 = left, 2 = right).
    lookup : pandas.DataFrame
        DataFrame with columns `acronym`, `name`, `rgb_triplet` indexed by structure
        index.
    cmap : matplotlib.colors.ListedColormap
        [`ListedColormap`][matplotlib.colors.ListedColormap] derived from
        `annotation.attrs["rgb_lookup"]`.
    norm : matplotlib.colors.BoundaryNorm
        [`BoundaryNorm`][matplotlib.colors.BoundaryNorm] derived from
        `annotation.attrs["rgb_lookup"]`.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        structures: "StructuresDict",
        mesh_to_physical: "npt.NDArray[np.float64]",
        rl_midline_um: float = 0.0,
    ) -> None:
        self._dataset = dataset
        self._structures = structures
        self._mesh_to_physical = mesh_to_physical
        # Midpoint of the RL axis in microns (atlas voxel space). Used to clip
        # mesh vertices to a single hemisphere without requiring pyvista.
        self._rl_midline_um = rl_midline_um
        self._lookup: pd.DataFrame | None = None

    @classmethod
    def from_brainglobe(
        cls, atlas: "str | BrainGlobeAtlas", **kwargs: object
    ) -> "Atlas":
        """Construct an Atlas from a BrainGlobe atlas name or instance.

        Parameters
        ----------
        atlas : str or brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas
            Either a BrainGlobe atlas name string (e.g. `"allen_mouse_25um"`) or an
            already-loaded
            [`BrainGlobeAtlas`][brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas] instance.
        **kwargs
            Additional keyword arguments forwarded to
            [`BrainGlobeAtlas`][brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas] when
            `atlas` is a string. Common options include `brainglobe_dir` (override the
            atlas cache directory) and `check_latest` (disable the latest-version
            check). Ignored when `atlas` is already a
            [`BrainGlobeAtlas`][brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas] instance.

        Returns
        -------
        Atlas
            Atlas with DataArrays in the atlas physical space (millimetres).

        Examples
        --------
        >>> atlas = Atlas.from_brainglobe("allen_mouse_25um")
        >>> atlas = Atlas.from_brainglobe("allen_mouse_25um", check_latest=False)
        >>> atlas = Atlas.from_brainglobe(bg_atlas_instance)
        """
        from brainglobe_atlasapi import BrainGlobeAtlas

        if isinstance(atlas, str):
            atlas = BrainGlobeAtlas(atlas, **kwargs)

        dataset = _build_dataset(atlas)
        # OBJ mesh vertices are in microns; scale to millimetres.
        mesh_to_physical = np.diag([1e-3, 1e-3, 1e-3, 1.0])

        meta = atlas.metadata
        # For asr orientation: shape[2] is the RL axis length (voxels);
        # resolution[2] is the voxel size in microns. The midline sits at the
        # centre of the volume.
        rl_midline_um = meta["shape"][2] / 2 * meta["resolution"][2]

        return cls(dataset, atlas.structures, mesh_to_physical, rl_midline_um)

    # ── Data properties ───────────────────────────────────────────────────────────────

    @property
    def reference(self) -> xr.DataArray:
        """Reference template DataArray.

        Returns
        -------
        xarray.DataArray
            The reference template DataArray.
        """
        return self._dataset["reference"]

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
        return self._dataset["annotation"]

    @property
    def hemispheres(self) -> xr.DataArray:
        """Hemisphere map DataArray (1 = left, 2 = right).

        Returns
        -------
        xarray.DataArray
            The hemisphere map DataArray.
        """
        return self._dataset["hemispheres"]

    # ── Structure metadata ────────────────────────────────────────────────────

    @property
    def lookup(self) -> pd.DataFrame:
        """DataFrame with columns `acronym`, `name`, `rgb_triplet`.

        The DataFrame is indexed by structure index.

        Returns
        -------
        pandas.DataFrame
            The structure lookup DataFrame, built from the BrainGlobe atlas's
            `StructuresDict`. Cached on first access.
        """
        if self._lookup is None:
            self._lookup = _build_lookup_df(self._structures)
        return self._lookup

    @property
    def cmap(self) -> "ListedColormap":
        """[`ListedColormap`][matplotlib.colors.ListedColormap] derived from `annotation.attrs["rgb_lookup"]`.

        Returns
        -------
        matplotlib.colors.ListedColormap
            The colormap to use for atlas rendering.
        """
        cmap, _ = _build_atlas_cmap_and_norm(self.annotation.attrs["rgb_lookup"])
        return cmap

    @property
    def norm(self) -> "BoundaryNorm":
        """[`BoundaryNorm`][matplotlib.colors.BoundaryNorm] derived from `annotation.attrs["rgb_lookup"]`.

        Returns
        -------
        matplotlib.colors.BoundaryNorm
            The norm to use for atlas rendering.
        """
        _, norm = _build_atlas_cmap_and_norm(self.annotation.attrs["rgb_lookup"])
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
            Filtered view of [`lookup`][confusius.atlas.Atlas.lookup] matching the
            search criteria.

        Examples
        --------
        >>> atlas.search("visual cortex")
        >>> atlas.search("VISp", field="acronym")
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
        """Return integer region masks stacked along a `masks` dimension.

        Each layer along `masks` has values in `{0, region_id}`; voxels
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
            Integer DataArray with dims `["masks", "z", "y", "x"]`. The
            `masks` coordinate holds the region acronym for each layer.

        Raises
        ------
        KeyError
            If any requested region acronym or index is not found in the atlas.
        ValueError
            If `sides` is a sequence whose length does not match `regions`, or if
            any element of `sides` is not ``"left"``, ``"right"``, or ``"both"``.

        Examples
        --------
        >>> atlas.get_masks("VISp")
        >>> atlas.get_masks("VISp", sides="left")
        >>> atlas.get_masks(["VISp", "AUDp", "MOp"])
        >>> atlas.get_masks(["VISp", "AUDp"], sides=["left", "both"])
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
            rid = _resolve_region_id(self._structures, reg)
            descendant_ids = _get_descendant_ids(self._structures, rid)

            layer = np.zeros_like(annotation_np, dtype=np.int32)
            # Using kind="table" here will use a lookup table approach that is much
            # faster at the cost of higher memory usage.
            layer[np.isin(annotation_np, descendant_ids, kind="table")] = rid

            if s == "left":
                layer[hemispheres_np != 1] = 0
            elif s == "right":
                layer[hemispheres_np != 2] = 0

            layers.append(layer)
            acronyms.append(self._structures[rid]["acronym"])

        stacked = np.stack(layers, axis=0)

        spatial_coords = {d: self.annotation.coords[d] for d in ["z", "y", "x"]}
        return xr.DataArray(
            stacked,
            dims=["masks", "z", "y", "x"],
            coords={"masks": acronyms, **spatial_coords},
            attrs=self.annotation.attrs.copy(),
        )

    # ── Meshes ────────────────────────────────────────────────────────────────────────

    def get_mesh(
        self,
        region: int | str,
        side: Literal["left", "right", "both"] = "both",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """Return vertex coordinates and face indices for a region's mesh.

        Reads the OBJ file bundled with the BrainGlobe atlas, optionally clips to one
        hemisphere, then transforms vertices from micron space to the DataArrays'
        current physical space (millimetres).

        Parameters
        ----------
        region : int or str
            Structure index or acronym.
        side : {"left", "right", "both"}, default: "both"
            Hemisphere to include. ``"both"`` keeps the full mesh. ``"left"`` and
            ``"right"`` clip in the original atlas micron space along the RL axis at the
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
            If the atlas does not have mesh files.
        """
        from pathlib import Path

        rid = _resolve_region_id(self._structures, region)
        info = self._structures[rid]

        mesh_filename = info.get("mesh_filename")
        if mesh_filename is None:
            raise ValueError(
                f"No mesh file available for region '{region}' (id {rid}). "
                "Not all BrainGlobe atlases include mesh files."
            )

        vertices_um, faces = _load_obj(Path(str(mesh_filename)))

        if side != "both":
            # Clip in micron space along the RL axis (column 2 for asr
            # orientation) before applying the physical transform.
            # For asr, axis 2 increases from right (0) to left (max), so:
            #   right hemisphere → RL < midline
            #   left  hemisphere → RL >= midline
            # TODO: generalize axis detection for non-asr atlases.
            if side == "right":
                keep = vertices_um[:, 2] < self._rl_midline_um
            else:  # "left"
                keep = vertices_um[:, 2] >= self._rl_midline_um

            keep_idx = np.where(keep)[0]
            old_to_new = np.full(len(vertices_um), -1, dtype=np.int64)
            old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

            # Retain only faces where all three vertices survive the clip.
            new_face_idx = old_to_new[faces]  # (M, 3); -1 if vertex removed.
            valid = np.all(new_face_idx >= 0, axis=1)
            vertices_um = vertices_um[keep_idx]
            faces = new_face_idx[valid].astype(np.int32)

        # Apply homogeneous transform: microns → physical millimetres.
        # _mesh_to_physical maps [x_um, y_um, z_um, 1]^T → [x_mm, y_mm, z_mm, 1]^T.
        n = len(vertices_um)
        vertices_h = np.hstack([vertices_um, np.ones((n, 1), dtype=np.float64)])
        vertices_m = (self._mesh_to_physical @ vertices_h.T).T[:, :3]

        return vertices_m, faces

    # ── Resampling ────────────────────────────────────────────────────────────────────

    def resample_like(
        self,
        reference: xr.DataArray,
        transform: "npt.NDArray[np.float64]",
        *,
        reference_interpolation: Literal["linear", "bspline"] = "linear",
        sitk_threads: int = -1,
    ) -> "Atlas":
        """Resample the atlas onto the grid of `reference`.

        Mirrors
        [`confusius.registration.resample_like`][confusius.registration.resample_like].
        Returns a new [`Atlas`][confusius.atlas.Atlas] whose DataArrays live on
        `reference`'s grid.

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
        reference_interpolation : {"linear", "bspline"}, default: "linear"
            Interpolation used for the `reference` variable.
        sitk_threads : int, default: -1
            Number of SimpleITK threads. Negative values use
            `max(1, cpu_count + 1 + sitk_threads)`.

        Returns
        -------
        Atlas
            New Atlas with DataArrays on `reference`'s grid.

        Examples
        --------
        >>> _, affine = atlas.reference.fusi.register.to_volume(
        ...     fusi_mean, metric="mattes_mi", transform="affine"
        ... )
        >>> atlas_fusi = atlas.resample_like(fusi_mean, affine)
        """

        resampled_ref = resample_like_da(
            self.reference,
            reference,
            transform,
            interpolation=reference_interpolation,
            default_value=0.0,
            sitk_threads=sitk_threads,
        )
        resampled_ann = (
            resample_like_da(
                self.annotation.astype(np.float32),
                reference,
                transform,
                interpolation="linear",
                default_value=0.0,
                sitk_threads=sitk_threads,
            )
            .round()
            .astype(np.int32)
        )
        resampled_ann.attrs = self.annotation.attrs.copy()

        resampled_hemi = (
            resample_like_da(
                self.hemispheres.astype(np.float32),
                reference,
                transform,
                interpolation="linear",
                default_value=0.0,
                sitk_threads=sitk_threads,
            )
            .round()
            .astype(np.int8)
        )

        new_dataset = xr.Dataset(
            {
                "reference": resampled_ref,
                "annotation": resampled_ann,
                "hemispheres": resampled_hemi,
            },
            attrs=self._dataset.attrs.copy(),
        )

        new_mesh_to_physical = np.linalg.inv(transform) @ self._mesh_to_physical

        # _rl_midline_um is a property of the original atlas space and does not change
        # when the DataArrays are resampled to a new grid.
        return Atlas(
            new_dataset, self._structures, new_mesh_to_physical, self._rl_midline_um
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
        rid = _resolve_region_id(self._structures, region)
        tree = self._structures.tree
        level = tree.level(rid)
        return [tree.ancestor(rid, lvl) for lvl in range(level)]

    def show_tree(self, **kwargs) -> None:
        """Print the structure hierarchy tree.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to
            [`treelib.Tree.show`][treelib.Tree.show].
        """
        kwargs.setdefault("stdout", False)
        print(self._structures.tree.show(**kwargs))

    # ── Dunder ────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        meta = self._dataset.attrs
        shape = self.annotation.shape
        return (
            f"Atlas("
            f"name={meta.get('name', 'unknown')!r}, "
            f"species={meta.get('species', 'unknown')!r}, "
            f"orientation={meta.get('orientation', 'unknown')!r}, "
            f"shape={shape}"
            f")"
        )
