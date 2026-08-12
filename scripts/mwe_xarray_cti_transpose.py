"""Minimal reproduction for Xarray CTI transpose handling.

Run with::

    uv run python scripts/mwe_xarray_cti_transpose.py

The coordinate transform has unequal axis lengths so stale shape metadata is visible
after transposing the dimensions.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from xarray.indexes import CoordinateTransform, CoordinateTransformIndex


class AffineTransform(CoordinateTransform):
    """Three-dimensional oblique affine coordinate transform."""

    def __init__(self) -> None:
        super().__init__(
            coord_names=("z", "y", "x"),
            dim_size={"k": 2, "j": 3, "i": 4},
        )
        self.affine = np.array(
            [
                [1.0, 0.1, 0.01],
                [0.2, 1.0, 0.02],
                [0.03, 0.04, 1.0],
            ]
        )

    def forward(self, dim_positions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map voxel positions to oblique world coordinates."""
        positions = np.stack([dim_positions[dim] for dim in self.dims], axis=0)
        world = np.einsum("ab,b...->a...", self.affine, positions)
        return dict(zip(self.coord_names, world, strict=True))

    def reverse(self, coord_labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map oblique world coordinates back to voxel positions."""
        world = np.stack([coord_labels[name] for name in self.coord_names], axis=0)
        positions = np.einsum("ab,b...->a...", np.linalg.inv(self.affine), world)
        return dict(zip(self.dims, positions, strict=True))

    def equals(
        self,
        other: CoordinateTransform,
        *,
        exclude: frozenset[str] | None = None,
    ) -> bool:
        """Return whether another transform has the same affine geometry."""
        return isinstance(other, AffineTransform) and np.array_equal(
            self.affine, other.affine
        )


def main() -> None:
    """Construct and transpose the reproducing DataArray."""
    transform = AffineTransform()
    data = xr.DataArray(
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
        dims=transform.dims,
        coords=xr.Coordinates.from_xindex(CoordinateTransformIndex(transform)),
    )
    transposed = data.transpose("i", "k", "j")

    coordinate = transposed.coords["z"]
    expected_shape = tuple(transposed.sizes[dim] for dim in coordinate.dims)
    print(f"xarray: {xr.__version__}")
    print(f"transposed data dims: {transposed.dims}")
    print(f"coordinate dims: {coordinate.dims}")
    print(f"coordinate shape: {coordinate.shape}")
    print(f"expected shape: {expected_shape}")

    if coordinate.shape != expected_shape:
        print("FAIL: transposed coordinate has stale shape metadata.")
    else:
        print("PASS: transposed coordinate shape is consistent.")

    point = {"k": 1.0, "j": 2.0, "i": 3.0}
    world = transform.forward({dim: np.asarray([point[dim]]) for dim in transform.dims})
    try:
        selected = transposed.sel(
            **{name: xr.Variable("point", values) for name, values in world.items()},
            method="nearest",
        )
    except ValueError as error:
        print(f"FAIL: transposed .sel() raised {type(error).__name__}: {error}")
    else:
        print(f"PASS: transposed .sel() returned {selected.values!r}.")


if __name__ == "__main__":
    main()
