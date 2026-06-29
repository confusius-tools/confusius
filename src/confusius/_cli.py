"""ConfUSIus command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import napari
    import napari.layers


def build_parser() -> argparse.ArgumentParser:
    """Build the `confusius` command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser configured with the CLI's positional path argument and
        `--lazy` / `--video` options.
    """
    parser = argparse.ArgumentParser(
        prog="confusius",
        description="Launch the ConfUSIus napari plugin.",
    )
    parser.add_argument(
        "path",
        nargs="*",
        type=Path,
        metavar="PATH",
        help=(
            "Path to a fUSI data file (.nii, .nii.gz, .scan, .zarr) to open on "
            "launch. May be passed multiple times to open several layers at "
            "once, e.g. `confusius fixed.nii moving.nii`."
        ),
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help=(
            "Load the files as Dask-backed arrays without computing. "
            "By default the full arrays are loaded into memory."
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        help=(
            "Path to a video file (.mp4, .mov, .avi) to display side-by-side "
            "with the first fUSI data file. Requires at least one data file."
        ),
    )
    return parser


def run(args: argparse.Namespace, viewer: napari.Viewer | None = None) -> napari.Viewer:
    """Open the requested files in a napari viewer and attach the ConfUSIus widget.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments, as produced by [`build_parser`][confusius._cli.build_parser].
    viewer : napari.Viewer, optional
        Existing viewer to populate. If not provided, a new viewer is created.
        Tests pass a headless viewer to avoid opening a Qt window.

    Returns
    -------
    napari.Viewer
        The viewer with the ConfUSIus widget docked and the requested data
        files added as image layers (one layer per file, in the same order as
        `args.path`). The first loaded layer is paired with `args.video`
        if it is also set.
    """
    import napari

    from confusius._napari import ConfUSIusWidget

    if viewer is None:
        viewer = napari.Viewer()
    widget = ConfUSIusWidget(viewer)
    viewer.window.add_dock_widget(widget, name="ConfUSIus")

    layers = _open_paths(viewer, args.path, lazy=args.lazy) if args.path else []

    if args.video is not None:
        if not layers:
            build_parser().error(
                "--video requires a data file to be specified as well."
            )
        video_panel = widget._accordion_panels["Video"]
        video_panel._add_video(args.video, layers[0])

    return viewer


def main() -> None:
    """Parse CLI arguments, run the plugin, and start the napari event loop."""
    args = build_parser().parse_args()
    run(args)
    import napari

    napari.run()


def _open_paths(
    viewer: napari.Viewer,
    paths: list[Path],
    *,
    lazy: bool,
) -> list[napari.layers.Image]:
    """Load each path and add it to the viewer as a ConfUSIus image layer.

    Parameters
    ----------
    viewer : napari.Viewer
        Active napari viewer to add the loaded layers to.
    paths : list of pathlib.Path
        One path per fUSI data file. Each becomes its own image layer named
        after the file's basename.
    lazy : bool
        Whether to keep the data Dask-backed (`True`) or materialise it into
        memory before plotting (`False`).

    Returns
    -------
    list of napari.layers.Image
        The layers added to the viewer, in the same order as `paths`.
    """
    from confusius.io import load
    from confusius.plotting.napari import plot_napari

    layers: list[napari.layers.Image] = []
    for path in paths:
        da = load(path)
        if not lazy:
            da = da.compute()
        _, layer = plot_napari(da, viewer=viewer, name=path.name)
        # `plot_napari` returns Image | Labels; the CLI only loads image
        # data files, so the resulting layer is always an Image.
        layers.append(cast("napari.layers.Image", layer))
    return layers
