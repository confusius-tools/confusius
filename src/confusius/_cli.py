"""ConfUSIus command-line interface."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    import napari.layers


def _add_help_option(parser: argparse.ArgumentParser) -> None:
    """Add a harmonized `-h`/`--help` option to a parser.

    Registered explicitly (with `add_help=False` on the parser) so its help
    string follows the same capitalized, full-stop style as the other
    arguments, instead of argparse's lowercase default.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the option to. It must be created with
        `add_help=False`.
    """
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the default (Napari plugin) `confusius` argument parser.

    A bare `confusius PATH...` invocation launches the napari plugin, so the launcher is
    the nameless default rather than a subcommand. Alternative namespaces (i.e.
    [`build_datasets_parser`][confusius._cli.build_datasets_parser]) are dispatched by
    [`main`][confusius._cli.main] and advertised under the `namespaces` help section.

    Returns
    -------
    argparse.ArgumentParser
        The parser configured with the launcher's positional path argument and
        `--lazy` / `--video` options.
    """
    parser = argparse.ArgumentParser(
        prog="confusius",
        description=(
            "Launch the ConfUSIus napari plugin. Alternative subcommands are "
            "listed under `namespaces` below."
        ),
        epilog="Run `confusius <namespace> --help` for the namespace's own options.",
        add_help=False,
    )
    _add_help_option(parser)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"ConfUSIus {metadata.version('confusius')}",
        help="Show the version number and exit.",
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
    # The `datasets` namespace is dispatched in `main`, not by this parser, so
    # it cannot be a real subparser without clashing with the greedy `path`
    # positional. Advertise it with a display-only pseudo-action (as argparse
    # does internally for subcommand choices) so it is colored and aligned like
    # a native entry while staying invisible to parsing.
    namespaces = parser.add_argument_group("namespaces")
    namespaces._group_actions.append(
        argparse.Action(
            option_strings=[],
            dest=argparse.SUPPRESS,
            nargs=0,
            metavar="datasets",
            help="Work with datasets, e.g. `confusius datasets --list`.",
        )
    )
    return parser


def build_datasets_parser() -> argparse.ArgumentParser:
    """Build the parser for the `confusius datasets` namespace.

    Returns
    -------
    argparse.ArgumentParser
        The parser configured with the `datasets` namespace options.
    """
    parser = argparse.ArgumentParser(
        prog="confusius datasets",
        description="Interact with ConfUSIus fetchable datasets.",
        add_help=False,
    )
    _add_help_option(parser)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets, their sizes, and whether each is cached.",
    )
    return parser


def run_datasets(args: argparse.Namespace) -> None:
    """Handle a parsed `confusius datasets` invocation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, as produced by
        [`build_datasets_parser`][confusius._cli.build_datasets_parser].
    """
    if args.list:
        from confusius.datasets import list_datasets

        list_datasets()
    else:
        build_datasets_parser().print_help()


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
        files added as layers (one layer per file, in the same order as
        `args.path`; see [`_open_paths`][confusius._cli._open_paths] for how
        the layer type is chosen). The first loaded layer is paired with
        `args.video` if it is also set.
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
        video_panel._add_video(args.video, layers[0])  # type: ignore

    return viewer


def main() -> None:
    """Parse CLI arguments, dispatch the namespace, and run the requested action.

    A bare `confusius PATH...` invocation launches the napari plugin, while
    `confusius datasets ...` routes to the `datasets` namespace.
    """
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] == "datasets":
        run_datasets(build_datasets_parser().parse_args(argv[1:]))
        return

    args = build_parser().parse_args(argv)
    run(args)
    import napari

    napari.run()


def _open_paths(
    viewer: napari.Viewer,
    paths: list[Path],
    *,
    lazy: bool,
) -> list[napari.layers.Image | napari.layers.Labels]:
    """Load each path and add it to the viewer as an image or labels layer.

    Parameters
    ----------
    viewer : napari.Viewer
        Active napari viewer to add the loaded layers to.
    paths : list of pathlib.Path
        One path per fUSI data file. Each becomes its own layer named after
        the file's basename.
    lazy : bool
        Whether to keep the data Dask-backed (`True`) or materialise it into
        memory before plotting (`False`).

    Returns
    -------
    list of napari.layers.Image or napari.layers.Labels
        The layers added to the viewer, in the same order as `paths`. Files
        with an integer dtype (e.g. atlas annotations, ROI masks) are added
        as `Labels` layers; all others are added as `Image` layers.
    """
    from confusius._utils.napari import infer_layer_type
    from confusius.io import load
    from confusius.plotting.napari import plot_napari

    layers: list[napari.layers.Image | napari.layers.Labels] = []
    for path in paths:
        da = load(path)
        if not lazy:
            da = da.compute()
        layer_type = infer_layer_type(da.dtype)
        _, layer = plot_napari(da, viewer=viewer, name=path.name, layer_type=layer_type)
        layers.append(layer)
    return layers
