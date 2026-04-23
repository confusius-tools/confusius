"""ConfUSIus command-line interface."""


def main() -> None:
    """Launch napari with the ConfUSIus plugin open."""
    import argparse
    from pathlib import Path

    import napari

    from confusius._napari import ConfUSIusWidget

    parser = argparse.ArgumentParser(
        prog="confusius",
        description="Launch the ConfUSIus napari plugin.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to a fUSI data file (.nii, .nii.gz, .scan, .zarr) to open on launch.",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help=(
            "Load the file as a Dask-backed array without computing. "
            "By default the full array is loaded into memory."
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        help=(
            "Path to a video file (.mp4, .mov, .avi) to display side-by-side "
            "with the fUSI data. Requires a data file to be specified as well."
        ),
    )
    args = parser.parse_args()

    viewer = napari.Viewer()
    widget = ConfUSIusWidget(viewer)
    viewer.window.add_dock_widget(widget, name="ConfUSIus")

    if args.path is not None:
        from confusius.io import load
        from confusius.plotting.image import plot_napari

        da = load(args.path)
        if not args.lazy:
            da = da.compute()
        _viewer, layer = plot_napari(da, viewer=viewer)

        if args.video is not None:
            video_panel = widget._accordion_panels["Video"]
            video_panel._add_video(args.video, layer)

    elif args.video is not None:
        parser.error("--video requires a data file to be specified as well.")

    napari.run()
