"""Generate review PNGs for the matplotlib fontsize heuristic.

Run with:

    uv run python tools/generate_fontsize_heuristic_review.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

matplotlib.use("Agg", force=True)

OUTPUT_DIR = PROJECT_ROOT / "tmp" / "fontsize_heuristic_review"


def make_volume(
    shape: tuple[int, int, int],
    *,
    seed: int,
    name: str,
    y_extent: float = 1.5,
    x_extent: float = 1.0,
) -> xr.DataArray:
    """Create a smooth synthetic volume with physical coordinates."""
    z_size, y_size, x_size = shape
    z = np.linspace(-2.0, 2.0, z_size)
    y = np.linspace(-y_extent, y_extent, y_size)
    x = np.linspace(-x_extent, x_extent, x_size)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    rng = np.random.default_rng(seed)
    blob1 = np.exp(-((xx + 0.35) ** 2 + (yy - 0.2) ** 2 + (zz + 0.4) ** 2) * 3.0)
    blob2 = 0.8 * np.exp(-((xx - 0.2) ** 2 + (yy + 0.45) ** 2 + (zz - 0.5) ** 2) * 6.0)
    stripes = 0.15 * np.sin(4 * np.pi * xx) * np.cos(3 * np.pi * yy)
    noise = 0.03 * rng.standard_normal(shape)
    data = blob1 + blob2 + stripes + noise

    return xr.DataArray(
        data,
        name=name,
        dims=["z", "y", "x"],
        coords={
            "z": xr.DataArray(
                z,
                dims=["z"],
                attrs={
                    "units": "mm",
                    "voxdim": float(z[1] - z[0]) if z_size > 1 else 1.0,
                },
            ),
            "y": xr.DataArray(
                y,
                dims=["y"],
                attrs={
                    "units": "mm",
                    "voxdim": float(y[1] - y[0]) if y_size > 1 else 1.0,
                },
            ),
            "x": xr.DataArray(
                x,
                dims=["x"],
                attrs={
                    "units": "mm",
                    "voxdim": float(x[1] - x[0]) if x_size > 1 else 1.0,
                },
            ),
        },
        attrs={"long_name": "Synthetic intensity", "units": "a.u."},
    )


def make_stat_map_like(volume: xr.DataArray) -> xr.DataArray:
    """Create a diverging synthetic stat map on `volume`'s grid."""
    z = volume.coords["z"].values[:, None, None]
    y = volume.coords["y"].values[None, :, None]
    x = volume.coords["x"].values[None, None, :]
    data = 4 * np.sin(2 * x) * np.cos(2 * y) + z
    return xr.DataArray(
        data,
        name="stat_map",
        dims=volume.dims,
        coords=volume.coords,
        attrs={"long_name": "Synthetic statistic", "units": "t"},
    )


def save_case(filename: str, volume: xr.DataArray, **kwargs: object) -> None:
    """Render one review volume case to disk."""
    from confusius.plotting import plot_volume

    plotter = plot_volume(volume, **kwargs)
    try:
        assert plotter.figure is not None
        plotter.savefig(str(OUTPUT_DIR / filename), dpi=180)
    finally:
        plotter.close()


def save_stat_map_case(
    filename: str,
    stat_map: xr.DataArray,
    bg_volume: xr.DataArray | None = None,
    **kwargs: object,
) -> None:
    """Render one review stat-map case to disk."""
    from confusius.plotting import plot_stat_map

    plotter = plot_stat_map(stat_map, bg_volume=bg_volume, **kwargs)
    try:
        assert plotter.figure is not None
        plotter.savefig(str(OUTPUT_DIR / filename), dpi=180)
    finally:
        plotter.close()


def save_two_colorbar_case(
    filename: str,
    bg_volume: xr.DataArray,
    stat_map: xr.DataArray,
    **kwargs: object,
) -> None:
    """Render an overlay case with two right-side colorbars."""
    from confusius.plotting import plot_volume

    plotter = plot_volume(bg_volume, cbar_label="background", **kwargs)
    try:
        plotter.add_volume(
            stat_map,
            match_coordinates=True,
            cmap="coolwarm",
            alpha=0.65,
            cbar_label="statistic",
            show_colorbar=True,
        )
        assert plotter.figure is not None
        plotter.savefig(str(OUTPUT_DIR / filename), dpi=180)
    finally:
        plotter.close()


def main() -> None:
    """Generate a small gallery of representative volume plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    compact = make_volume((1, 24, 32), seed=0, name="compact")
    standard = make_volume((4, 24, 32), seed=1, name="standard")
    tall = make_volume((6, 40, 16), seed=2, name="tall", y_extent=3.5, x_extent=0.8)
    wide = make_volume((6, 16, 40), seed=3, name="wide", y_extent=0.8, x_extent=3.5)
    dense = make_volume((9, 24, 32), seed=4, name="dense")
    stat = make_stat_map_like(standard)

    cases: list[tuple[str, xr.DataArray, dict[str, object]]] = [
        (
            "01_single_slice_colorbar_axes.png",
            compact,
            dict(slice_mode="z", show_colorbar=True, show_axes=True),
        ),
        (
            "02_single_slice_no_colorbar_no_axes.png",
            compact,
            dict(slice_mode="z", show_colorbar=False, show_axes=False),
        ),
        (
            "03_four_slices_auto_grid.png",
            standard,
            dict(slice_mode="z", show_colorbar=True, show_axes=True),
        ),
        (
            "04_four_slices_one_row.png",
            standard,
            dict(slice_mode="z", show_colorbar=True, show_axes=True, nrows=1),
        ),
        (
            "05_four_slices_one_column.png",
            standard,
            dict(slice_mode="z", show_colorbar=True, show_axes=True, ncols=1),
        ),
        (
            "06_tall_volume_six_slices_auto_grid.png",
            tall,
            dict(slice_mode="z", show_colorbar=True, show_axes=True),
        ),
        (
            "07_wide_volume_six_slices_auto_grid.png",
            wide,
            dict(slice_mode="z", show_colorbar=True, show_axes=True),
        ),
        (
            "08_tall_volume_three_by_two.png",
            tall,
            dict(slice_mode="z", show_colorbar=True, show_axes=True, nrows=3, ncols=2),
        ),
        (
            "09_wide_volume_two_by_three.png",
            wide,
            dict(slice_mode="z", show_colorbar=True, show_axes=True, nrows=2, ncols=3),
        ),
        (
            "10_tall_volume_no_axes.png",
            tall,
            dict(slice_mode="z", show_colorbar=False, show_axes=False),
        ),
        (
            "11_wide_volume_subset.png",
            wide,
            dict(
                slice_mode="z",
                slice_coords=list(wide.coords["z"].values[::2]),
                show_colorbar=True,
                show_axes=True,
            ),
        ),
        (
            "12_dense_volume_three_by_three.png",
            dense,
            dict(slice_mode="z", show_colorbar=True, show_axes=True, nrows=3, ncols=3),
        ),
        (
            "13_dense_volume_one_row.png",
            dense,
            dict(slice_mode="z", show_colorbar=True, show_axes=True, nrows=1),
        ),
    ]

    for filename, volume, kwargs in cases:
        save_case(filename, volume, **kwargs)

    save_stat_map_case(
        "14_stat_map_background_colorbar.png",
        stat,
        standard,
        slice_mode="z",
        show_colorbar=True,
        show_axes=True,
    )
    save_two_colorbar_case(
        "15_volume_overlay_two_colorbars.png",
        standard,
        stat,
        slice_mode="z",
        show_colorbar=True,
        show_axes=True,
    )
    save_case(
        "16_four_slices_no_titles.png",
        standard,
        slice_mode="z",
        show_colorbar=True,
        show_titles=False,
        show_axes=True,
    )
    save_case(
        "17_four_slices_no_ticks.png",
        standard,
        slice_mode="z",
        show_colorbar=True,
        show_axis_ticks=False,
        show_axes=True,
    )
    save_case(
        "18_long_colorbar_label.png",
        standard,
        slice_mode="z",
        show_colorbar=True,
        show_axes=True,
        cbar_label="Very long synthetic intensity label with arbitrary units",
    )
    save_case(
        "19_slice_mode_x_four_slices.png",
        wide,
        slice_mode="x",
        slice_coords=list(wide.coords["x"].values[::12]),
        show_colorbar=True,
        show_axes=True,
    )
    save_case(
        "20_slice_mode_y_four_slices.png",
        tall,
        slice_mode="y",
        slice_coords=list(tall.coords["y"].values[::12]),
        show_colorbar=True,
        show_axes=True,
    )
    save_stat_map_case(
        "21_stat_map_no_background.png",
        stat,
        slice_mode="z",
        show_colorbar=True,
        show_axes=True,
    )
    save_two_colorbar_case(
        "22_two_colorbars_one_row.png",
        standard,
        stat,
        slice_mode="z",
        show_colorbar=True,
        show_axes=True,
        nrows=1,
    )
    save_two_colorbar_case(
        "23_two_colorbars_one_column.png",
        standard,
        stat,
        slice_mode="z",
        show_colorbar=True,
        show_axes=True,
        ncols=1,
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
