"""Unit tests for the ConfUSIus CLI entry point."""

from __future__ import annotations

from pathlib import Path


class TestBuildParser:
    """The CLI parser accepts zero, one, or many positional paths."""

    def test_no_path(self) -> None:
        from confusius._cli import build_parser

        ns = build_parser().parse_args([])
        assert ns.path == []
        assert ns.lazy is False
        assert ns.video is None

    def test_one_path(self) -> None:
        from confusius._cli import build_parser

        ns = build_parser().parse_args(["fixed.nii"])
        assert ns.path == [Path("fixed.nii")]

    def test_two_paths(self) -> None:
        from confusius._cli import build_parser

        ns = build_parser().parse_args(["fixed.nii", "moving.nii"])
        assert ns.path == [Path("fixed.nii"), Path("moving.nii")]

    def test_three_paths_with_various_extensions(self) -> None:
        from confusius._cli import build_parser

        ns = build_parser().parse_args(["a.nii", "b.nii.gz", "c.scan"])
        assert ns.path == [Path("a.nii"), Path("b.nii.gz"), Path("c.scan")]

    def test_lazy_and_video(self) -> None:
        from confusius._cli import build_parser

        ns = build_parser().parse_args(["fixed.nii", "--lazy", "--video", "v.mp4"])
        assert ns.lazy is True
        assert ns.video == Path("v.mp4")


class TestRun:
    """`run` opens each requested file as a layer named after the file."""

    def test_no_path_adds_nothing(self, make_napari_viewer) -> None:
        from confusius._cli import build_parser, run

        args = build_parser().parse_args([])
        viewer = make_napari_viewer()
        run(args, viewer=viewer)
        assert len(viewer.layers) == 0

    def test_one_path_adds_one_layer(
        self, make_napari_viewer, scan_2d_path: Path
    ) -> None:
        from confusius._cli import build_parser, run

        args = build_parser().parse_args([str(scan_2d_path)])
        viewer = make_napari_viewer()
        run(args, viewer=viewer)

        assert len(viewer.layers) == 1
        assert viewer.layers[scan_2d_path.name].name == scan_2d_path.name

    def test_two_paths_adds_two_layers_in_order(
        self, make_napari_viewer, tmp_path: Path, scan_2d_path: Path
    ) -> None:
        from confusius._cli import build_parser, run

        # Make a second file via copy so the two paths have distinct basenames.
        second = tmp_path / "second.scan"
        second.write_bytes(scan_2d_path.read_bytes())

        args = build_parser().parse_args([str(scan_2d_path), str(second)])
        viewer = make_napari_viewer()
        run(args, viewer=viewer)

        assert len(viewer.layers) == 2
        names = {layer.name for layer in viewer.layers}
        assert names == {scan_2d_path.name, second.name}
        # Returned order matches the input order.
        layer_names = [layer.name for layer in viewer.layers]
        assert layer_names.index(scan_2d_path.name) < layer_names.index(second.name)

    def test_video_without_data_file_errors(self, make_napari_viewer) -> None:
        import pytest

        from confusius._cli import build_parser, run

        args = build_parser().parse_args(["--video", "v.mp4"])
        viewer = make_napari_viewer()
        with pytest.raises(SystemExit):
            run(args, viewer=viewer)
