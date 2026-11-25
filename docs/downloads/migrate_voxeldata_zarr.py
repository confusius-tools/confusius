"""Upgrade geometry metadata in a local, single-array ConfUSIus Zarr store."""

import argparse
from pathlib import Path

import numpy as np
import zarr


def migrate(path: Path) -> None:
    """Migrate one store written by the legacy ConfUSIus array writer.

    Parameters
    ----------
    path : pathlib.Path
        Existing Zarr v2 or v3 directory, opened for metadata updates.

    Raises
    ------
    ValueError
        If the store is ambiguous, has partial metadata, or the legacy geometry
        cannot be represented by the new model.
    """
    consolidated = zarr.open_group(path, mode="r").metadata.consolidated_metadata
    group = zarr.open_group(path, mode="r+", use_consolidated=False)
    arrays = [a for _, a in group.arrays() if "voxel_to_world" in a.attrs]
    if len(arrays) != 1:
        raise ValueError("Expected one array written by confusius.save().")
    array = arrays[0]
    attrs = array.attrs.asdict()
    dims_key = "voxel_to_world_geometry_dims"
    frames_key = "voxel_to_world_named_affines"
    if (dims_key in attrs) != (frames_key in attrs):
        raise ValueError("Geometry metadata is partially migrated.")
    if dims_key not in attrs:
        affine = np.asarray(attrs["voxel_to_world"], dtype=float)
        dims = getattr(array.metadata, "dimension_names", None)
        if dims is None:
            dims = attrs.get("_ARRAY_DIMENSIONS")
        if not isinstance(dims, (list, tuple)) or len(dims) != array.ndim:
            raise ValueError("Array is missing its Xarray dimension names.")
        if affine.shape == (4, 4):
            geometry_dims = []
        elif "pose" in dims and affine.shape == (array.shape[dims.index("pose")], 4, 4):
            geometry_dims = ["pose"]
        else:
            raise ValueError("Expected a shared affine or one affine per pose.")

        frames = dict(attrs.pop("affines", {}))
        if "bspline_initialization" in frames and (
            attrs.get("transform_type") == "bspline_transform"
            or attrs.get("type") == "bspline_transform"
        ):
            initial = frames.pop("bspline_initialization")
            if (
                "bspline_initialization" in attrs
                and attrs["bspline_initialization"] != initial
            ):
                raise ValueError("Conflicting B-spline initializations.")
            attrs["bspline_initialization"] = initial
        for name, matrix in [("voxel_to_world", affine), *frames.items()]:
            values = np.asarray(matrix, dtype=float)
            if (
                values.shape not in ((4, 4), affine.shape)
                or not np.isfinite(values).all()
                or not np.allclose(values[..., -1, :], [0, 0, 0, 1])
            ):
                raise ValueError(f"Invalid affine {name!r}.")
        attrs[dims_key] = geometry_dims
        attrs[frames_key] = frames
        array.attrs.put(attrs)

    # Refresh cached metadata even on retry after an interrupted consolidation.
    if consolidated is not None:
        zarr.consolidate_metadata(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    migrate(parser.parse_args().store)
