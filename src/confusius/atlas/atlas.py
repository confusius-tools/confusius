"""Atlas class for brain atlas integration via BrainGlobe."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy.interpolate import interpn

from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius.atlas._structures import (
    _build_lookup_df,
    _build_rgb_lookup,
    _get_descendant_ids,
    _load_obj,
    _resolve_region_id,
)
from confusius.registration.bspline import sample_displacement_field_like
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
        Dataset with variables `reference`, `annotation`, and `hemispheres`, each with
        physical coordinates. `Atlas.from_brainglobe` currently converts BrainGlobe
        metadata and meshes from microns to millimeters.
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
    cmap, norm = build_atlas_cmap_and_norm(rgb_lookup)
    roi_labels = {
        int(sid): str(info["name"] + f" ({info['acronym']})")
        for sid, info in bg_atlas.structures.items()
    }

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
        # to Zarr/NIfTI; rgb_lookup is the serializable source of truth.
        attrs={
            "rgb_lookup": rgb_lookup,
            "roi_labels": roi_labels,
            "cmap": cmap,
            "norm": norm,
        },
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


MeshVertexTransform: TypeAlias = npt.NDArray[np.float64] | xr.DataArray
"""Transform that maps current atlas physical coordinates back to base atlas space.

This follows the same pull/inverse convention as `confusius.registration`: a point in
current atlas physical coordinates is mapped into the original atlas physical
coordinates. Affine transforms use homogeneous matrices; nonlinear transforms use
B-spline or displacement-field DataArrays.
"""


def _validate_mesh_vertex_transform(transform: MeshVertexTransform) -> None:
    """Raise ValueError if `transform` is not a valid mesh vertex transform.

    Parameters
    ----------
    transform : numpy.ndarray or xarray.DataArray
        Candidate transform.

    Raises
    ------
    ValueError
        If `transform` is neither a homogeneous affine nor a supported nonlinear
        transform DataArray.
    """
    if isinstance(transform, np.ndarray):
        if transform.shape != (4, 4):
            raise ValueError(
                f"Mesh affine transform must have shape (4, 4); got {transform.shape}."
            )
        return

    transform_type = transform.attrs.get("type")
    if transform_type not in {"bspline_transform", "displacement_field_transform"}:
        raise ValueError(
            "Mesh nonlinear transform must have attrs['type'] equal to "
            "'bspline_transform' or 'displacement_field_transform'; got "
            f"{transform_type!r}."
        )


def _transform_points(
    transform: MeshVertexTransform,
    points: npt.NDArray[np.float64],
    reference: xr.DataArray,
) -> npt.NDArray[np.float64]:
    """Apply a pull transform to physical points.

    Parameters
    ----------
    transform : numpy.ndarray or xarray.DataArray
        Pull transform mapping points from `reference` space into some moving/base
        space.
    points : (N, D) numpy.ndarray
        Physical points in the same axis order as atlas mesh vertices `(x, y, z)`.
    reference : xarray.DataArray
        Reference grid on which nonlinear transforms live.

    Returns
    -------
    numpy.ndarray
        Transformed points with shape `(N, D)`.
    """
    _validate_mesh_vertex_transform(transform)

    if isinstance(transform, np.ndarray):
        n_points, ndim = points.shape
        points_h = np.hstack([points, np.ones((n_points, 1), dtype=np.float64)])
        return (transform @ points_h.T).T[:, :ndim]

    field = transform
    if transform.attrs.get("type") == "bspline_transform":
        field = sample_displacement_field_like(transform, reference)
    displacement = _interpolate_displacement_field(field, points)
    return points + displacement


def _compose_mesh_vertex_transforms(
    old_transform: MeshVertexTransform,
    new_transform: MeshVertexTransform,
    new_reference: xr.DataArray,
    old_reference: xr.DataArray,
) -> MeshVertexTransform:
    """Compose mesh pull transforms as `old_transform ∘ new_transform`.

    Parameters
    ----------
    old_transform : numpy.ndarray or xarray.DataArray
        Pull transform from the current atlas physical space to the base atlas space.
    new_transform : numpy.ndarray or xarray.DataArray
        Pull transform from the new atlas physical space to the current atlas space.
    new_reference : xarray.DataArray
        Reference grid of the new atlas space.
    old_reference : xarray.DataArray
        Reference grid of the current atlas space.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Pull transform from the new atlas physical space to the base atlas space.
    """
    if isinstance(old_transform, np.ndarray) and np.allclose(old_transform, np.eye(4)):
        return (
            new_transform.copy(deep=False)
            if isinstance(new_transform, xr.DataArray)
            else new_transform.copy()
        )
    if isinstance(new_transform, np.ndarray) and np.allclose(new_transform, np.eye(4)):
        return (
            old_transform.copy(deep=False)
            if isinstance(old_transform, xr.DataArray)
            else old_transform.copy()
        )
    if isinstance(old_transform, np.ndarray) and isinstance(new_transform, np.ndarray):
        return old_transform @ new_transform

    dims = [str(dim) for dim in new_reference.dims]
    grid = np.meshgrid(
        *[
            np.asarray(new_reference.coords[dim].values, dtype=np.float64)
            for dim in dims
        ],
        indexing="ij",
    )
    reference_points = np.stack(grid, axis=-1).reshape(-1, len(dims))
    current_points = _transform_points(new_transform, reference_points, new_reference)
    base_points = _transform_points(old_transform, current_points, old_reference)
    # Points and displacement components are both in DataArray dim order (component
    # `dims[i]` displaces along axis `dims[i]`), matching `_transform_points` and
    # `_interpolate_displacement_field`, so no axis reversal is needed.
    displacement = (base_points - reference_points).T.reshape(
        len(dims), *new_reference.shape
    )

    coords = {"component": np.array(dims, dtype=np.str_)}
    coords.update({dim: new_reference.coords[dim] for dim in dims})
    return xr.DataArray(
        displacement,
        dims=["component", *dims],
        coords=coords,
        attrs={"type": "displacement_field_transform"},
    )


def _interpolate_displacement_field(
    field: xr.DataArray, points: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Interpolate a dense displacement field at physical points.

    Parameters
    ----------
    field : xarray.DataArray
        Dense displacement field with dimensions `("component", *spatial_dims)`.
    points : (N, D) numpy.ndarray
        Physical points in the same axis order as `field.dims[1:]`.

    Returns
    -------
    numpy.ndarray
        Interpolated displacement vectors with shape `(N, D)`. Points beyond the field
        domain and its one-voxel edge-padded margin are returned as NaN.
    """
    spatial_dims = [str(dim) for dim in field.dims[1:]]
    spacing = field.fusi.spacing

    # Pad the field by one voxel (edge replication) at each end of every spatial dim, so
    # points within one voxel of a boundary, as barely-outside mesh vertices are, can
    # still be interpolated. Points beyond the padded margin resolve to NaN; the mesh
    # caller drops the vertices that could not be warped.
    padded_field = field.pad({dim: (1, 1) for dim in spatial_dims}, mode="edge")
    grid = []
    for dim in spatial_dims:
        padded_coord = np.asarray(padded_field.coords[dim], dtype=np.float64).copy()
        padded_coord[0] = padded_coord[1] - spacing[dim]
        padded_coord[-1] = padded_coord[-2] + spacing[dim]
        grid.append(padded_coord)

    displacements = interpn(
        grid,
        np.moveaxis(np.asarray(padded_field.values, dtype=np.float64), 0, -1),
        points,
        bounds_error=False,
        fill_value=np.nan,
    )
    return np.asarray(displacements, dtype=np.float64)


def _invert_displacement_field_at_points(
    field: xr.DataArray,
    points: npt.NDArray[np.float64],
    initial_guess_affine: npt.NDArray[np.float64] | None = None,
    *,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> npt.NDArray[np.float64]:
    """Map moving-space points back to fixed space with fixed-point iteration.

    Parameters
    ----------
    field : xarray.DataArray
        Dense forward displacement field that maps fixed-space points to moving-space
        points as `moving = fixed + field(fixed)`.
    points : (N, D) numpy.ndarray
        Moving-space points to invert, in the same axis order as `field.dims[1:]`.
    initial_guess_affine : (D+1, D+1) numpy.ndarray, optional
        Affine inverse used to seed the iteration. If not provided, the moving-space
        points themselves are used as the initial guess.
    max_iterations : int, default: 20
        Maximum number of fixed-point updates.
    tolerance : float, default: 1e-6
        Convergence threshold on the maximum point update, in physical units.

    Returns
    -------
    numpy.ndarray
        Approximate fixed-space points with shape `(N, D)`.
    """
    if points.shape[0] == 0:
        return points.copy()

    if initial_guess_affine is None:
        fixed_points = points.copy()
    else:
        n_points, ndim = points.shape
        points_h = np.hstack([points, np.ones((n_points, 1), dtype=np.float64)])
        fixed_points = (initial_guess_affine @ points_h.T).T[:, :ndim]

    for _ in range(max_iterations):
        displaced = _interpolate_displacement_field(field, fixed_points)
        updated = points - displaced
        if np.max(np.linalg.norm(updated - fixed_points, axis=1)) <= tolerance:
            return updated
        fixed_points = updated

    return fixed_points


def _apply_mesh_vertex_transform(
    transform: MeshVertexTransform,
    vertices: npt.NDArray[np.float64],
    reference: xr.DataArray,
) -> npt.NDArray[np.float64]:
    """Transform mesh vertices from BrainGlobe OBJ space into current physical space.

    Parameters
    ----------
    transform : numpy.ndarray or xarray.DataArray
        Pull transform from the current atlas physical space back to the base atlas
        physical space.
    vertices : (N, 3) numpy.ndarray
        Mesh vertices expressed in the base atlas physical coordinates.
    reference : xarray.DataArray
        Current atlas reference grid. Used to sample B-spline transforms into a dense
        displacement field when needed.

    Returns
    -------
    numpy.ndarray
        Mesh vertices in the current physical space.
    """
    if isinstance(transform, np.ndarray):
        n_vertices, ndim = vertices.shape
        vertices_h = np.hstack([vertices, np.ones((n_vertices, 1), dtype=np.float64)])
        return (np.linalg.inv(transform) @ vertices_h.T).T[:, :ndim]

    field = transform
    if transform.attrs.get("type") == "bspline_transform":
        field = sample_displacement_field_like(transform, reference)

    initial_guess_affine = None
    pre_affine = transform.attrs.get("affines", {}).get("bspline_initialization")
    if pre_affine is not None:
        initial_guess_affine = np.linalg.inv(np.asarray(pre_affine, dtype=np.float64))

    return _invert_displacement_field_at_points(field, vertices, initial_guess_affine)


def _drop_vertices_outside_grid(
    vertices: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int32],
    reference: xr.DataArray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Drop mesh vertices outside the reference grid (plus a margin) and reindex faces.

    Applied to warped mesh vertices: a nonlinear warp can move vertices outside the grid
    and returns NaN for vertices too far outside to interpolate. Vertices beyond the grid
    by more than one voxel (the padded interpolation margin), and NaN vertices, are
    dropped along with any face that references them (NaN fails the bounds comparison, so
    NaN vertices are removed too).

    Parameters
    ----------
    vertices : (N, 3) numpy.ndarray
        Warped mesh vertices in DataArray dim order `(z, y, x)`.
    faces : (M, 3) numpy.ndarray
        Zero-indexed triangle face indices into `vertices`.
    reference : xarray.DataArray
        Reference grid whose coordinate bounds define the valid domain.

    Returns
    -------
    vertices : numpy.ndarray
        Surviving vertices, shape `(K, 3)` with `K <= N`.
    faces : numpy.ndarray
        Faces whose three vertices all survived, reindexed into the new vertex array.
    """
    dims = [str(dim) for dim in reference.dims]
    spacing = reference.fusi.spacing
    inside = np.ones(len(vertices), dtype=bool)
    for axis, dim in enumerate(dims):
        coord = reference.coords[dim].values
        # Keep the same one-voxel margin the field interpolation is padded to, so a
        # vertex within `spacing` of a boundary (e.g. the anterior/posterior tips of the
        # Allen brain) is retained rather than clipped.
        margin = spacing[dim] if spacing[dim] is not None else 0.0
        inside &= (vertices[:, axis] >= coord.min() - margin) & (
            vertices[:, axis] <= coord.max() + margin
        )

    keep_idx = np.where(inside)[0]
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

    new_face_idx = old_to_new[faces]  # (M, 3); -1 for dropped vertices.
    valid = np.all(new_face_idx >= 0, axis=1)
    return vertices[keep_idx], new_face_idx[valid].astype(np.int32)


class Atlas:
    """Brain atlas wrapper backed by BrainGlobe, exposing DataArrays.

    Use [`from_brainglobe`][confusius.atlas.Atlas.from_brainglobe] to construct an
    instance.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with variables `reference`, `annotation`, and `hemispheres` on a common
        `(z, y, x)` physical grid. Atlases constructed with
        [`from_brainglobe`][confusius.atlas.Atlas.from_brainglobe] currently use
        millimeters, but `Atlas` itself only requires that units stay internally
        consistent.
    structures : brainglobe_atlasapi.structure_class.StructuresDict
        BrainGlobe structure dictionary (carries the `treelib` hierarchy tree).
    mesh_vertex_transform : (4, 4) numpy.ndarray or xarray.DataArray
        Pull transform that maps the atlas' current physical coordinates back to the
        base atlas physical coordinates.
    rl_midline : float, default: 0.0
        Midpoint of the RL axis in the base atlas physical units. Used to clip mesh
        vertices to a single hemisphere before applying any mesh vertex transform.

    Attributes
    ----------
    reference : xarray.DataArray
        Reference template DataArray.
    annotation : xarray.DataArray
        Region annotations DataArray with integer labels on the same physical grid.
    hemispheres : xarray.DataArray
        Hemisphere map DataArray (1 = left, 2 = right) on the same physical grid.
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
        mesh_vertex_transform: MeshVertexTransform,
        rl_midline: float = 0.0,
    ) -> None:
        self._dataset = dataset
        self._structures = structures
        _validate_mesh_vertex_transform(mesh_vertex_transform)
        self._mesh_vertex_transform = mesh_vertex_transform
        self._rl_midline = rl_midline
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
            Atlas with DataArrays in atlas physical space. BrainGlobe metadata and
            meshes are currently converted from microns to millimeters.

        Examples
        --------
        >>> atlas = Atlas.from_brainglobe("allen_mouse_25um")
        >>> atlas = Atlas.from_brainglobe("allen_mouse_25um", check_latest=False)
        >>> atlas = Atlas.from_brainglobe(bg_atlas_instance)
        """
        from brainglobe_atlasapi import BrainGlobeAtlas

        if isinstance(atlas, str):
            atlas = BrainGlobeAtlas(atlas, **kwargs)  # type: ignore

        dataset = _build_dataset(atlas)
        mesh_vertex_transform = np.eye(4, dtype=np.float64)

        # For asr orientation: shape[2] is the RL axis length (voxels); resolution[2]
        # is the voxel size in microns in BrainGlobe metadata. Convert the midline to
        # the base atlas physical units used by Atlas.from_brainglobe(), i.e.
        # millimeters.
        meta = atlas.metadata
        rl_midline = meta["shape"][2] / 2 * meta["resolution"][2] * 1e-3

        return cls(dataset, atlas.structures, mesh_vertex_transform, rl_midline)

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
        >>> atlas.get_masks("VISp")
        >>> atlas.get_masks("VISp", sides="left")
        >>> atlas.get_masks(["VISp", "AUDp", "MOp"])
        >>> atlas.get_masks(["VISp", "AUDp"], sides=["left", "both"])
        >>> atlas.get_masks(["VISp", "VISp"], sides=["left", "right"]).coords["mask"].values
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
            rid = _resolve_region_id(self._structures, reg)
            descendant_ids = _get_descendant_ids(self._structures, rid)

            layer = np.zeros_like(annotation_np, dtype=np.int32)
            # Using kind="table" here will use a lookup table approach that is much
            # faster at the cost of higher memory usage.
            layer[np.isin(annotation_np, descendant_ids, kind="table")] = rid

            acronym = self._structures[rid]["acronym"]
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

        Reads the OBJ file bundled with the BrainGlobe atlas, optionally clips to one
        hemisphere, then transforms vertices from micron space to the DataArrays'
        current physical space.

        Parameters
        ----------
        region : int or str
            Structure index or acronym.
        side : {"left", "right", "both"}, default: "both"
            Hemisphere to include. `"both"` keeps the full mesh. `"left"` and
            `"right"` clip in the base atlas physical space along the RL axis at the
            volume midline. Only triangles whose three vertices all fall on the
            requested side are retained; the cut face is not closed.

            !!! note
               Generalising axis detection from the orientation attribute for non-`asr`
               atlases is not yet implemented.

        Returns
        -------
        vertices : numpy.ndarray, shape (N, 3)
            Vertex coordinates in millimeters.
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
        vertices_mm = vertices_um * 1e-3  # Convert microns to millimeters.

        if side != "both":
            # Clip in atlas physical space along the RL axis (column 2 for asr
            # orientation) before applying any additional mesh vertex transform.
            # For asr, axis 2 increases from right (0) to left (max), so:
            #   right hemisphere → RL < midline
            #   left  hemisphere → RL >= midline
            # TODO: generalize axis detection for non-asr atlases.
            if side == "right":
                keep = vertices_mm[:, 2] < self._rl_midline
            else:  # "left"
                keep = vertices_mm[:, 2] >= self._rl_midline

            keep_idx = np.where(keep)[0]
            old_to_new = np.full(len(vertices_mm), -1, dtype=np.int64)
            old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

            # Retain only faces where all three vertices survive the clip.
            new_face_idx = old_to_new[faces]  # (M, 3); -1 if vertex removed.
            valid = np.all(new_face_idx >= 0, axis=1)
            vertices_mm = vertices_mm[keep_idx]
            faces = new_face_idx[valid].astype(np.int32)

        vertices_m = _apply_mesh_vertex_transform(
            self._mesh_vertex_transform,
            vertices_mm,
            self.reference,
        )

        if isinstance(self._mesh_vertex_transform, xr.DataArray):
            # A nonlinear warp can move vertices outside the grid, and vertices too far
            # outside it to interpolate come back as NaN; drop both, and any face that
            # references them.
            vertices_m, faces = _drop_vertices_outside_grid(
                vertices_m, faces, self.reference
            )

        return vertices_m, faces

    # ── Resampling ────────────────────────────────────────────────────────────────────

    def resample_like(
        self,
        reference: xr.DataArray,
        transform: "npt.NDArray[np.float64] | xr.DataArray",
        *,
        reference_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
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

        `moving`, `reference`, and any DataArray transform must use the same physical
        coordinate units when such metadata is defined.

        Parameters
        ----------
        reference : xarray.DataArray
            Target grid. Must be 2D or 3D and must not have a `time` dimension.
        transform : (N+1, N+1) numpy.ndarray or xarray.DataArray
            Pull/inverse transform returned by `register_volume`, mapping
            `reference` physical coordinates to atlas physical coordinates.

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

        new_dataset = xr.Dataset(
            {
                "reference": resampled_ref,
                "annotation": resampled_ann,
                "hemispheres": resampled_hemi,
            },
            attrs=self._dataset.attrs.copy(),
        )

        new_mesh_vertex_transform = _compose_mesh_vertex_transforms(
            self._mesh_vertex_transform,
            transform,
            reference,
            self.reference,
        )

        # _rl_midline is a property of the base atlas physical space and does not
        # change when the DataArrays are resampled to a new grid.
        return Atlas(
            new_dataset,
            self._structures,
            new_mesh_vertex_transform,
            self._rl_midline,
        )

    def resample(
        self,
        transform: MeshVertexTransform,
        *,
        shape: Sequence[int],
        spacing: Sequence[float],
        origin: Sequence[float],
        dims: Sequence[str],
        reference_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
        sitk_threads: int = -1,
    ) -> "Atlas":
        """Resample the atlas onto an explicit output grid.

        Mirrors [`confusius.registration.resample_volume`][confusius.registration.resample_volume]
        for atlas objects by constructing a temporary reference grid and delegating to
        [`resample_like`][confusius.atlas.Atlas.resample_like].

        Parameters
        ----------
        transform : (N+1, N+1) numpy.ndarray or xarray.DataArray
            Pull transform mapping output physical coordinates to the current atlas
            physical coordinates.
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
        Atlas
            Resampled atlas on the requested grid.
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
