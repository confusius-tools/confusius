"""Unit tests for the ConfUSIus CLI entry point."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def integer_nifti_path(tmp_path: Path) -> Path:
    """3D NIfTI file with an integer dtype, as produced by atlas annotations."""
    data = np.zeros((4, 6, 8), dtype=np.int16)
    data[1:3, 2:4, 3:5] = 7
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / "labels.nii.gz"
    img.to_filename(path)
    return path


@pytest.fixture
def float_nifti_path(tmp_path: Path) -> Path:
    """3D NIfTI file with a float dtype, as produced by fUSI power Doppler data."""
    data = np.random.default_rng(0).random((4, 6, 8)).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / "power_doppler.nii.gz"
    img.to_filename(path)
    return path


class TestBuildParser:
    """The default parser accepts zero, one, or many positional paths."""

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

    def test_version_flag(self, capsys) -> None:
        from importlib import metadata

        import pytest

        from confusius._cli import build_parser

        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["--version"])
        assert exc.value.code == 0
        assert metadata.version("confusius") in capsys.readouterr().out


class TestDatasetsNamespace:
    """The `datasets` namespace parses `--list` and dispatches to `list_datasets`."""

    def test_list_flag_parsed(self) -> None:
        from confusius._cli import build_datasets_parser

        ns = build_datasets_parser().parse_args(["--list"])
        assert ns.list is True

    def test_list_defaults_false(self) -> None:
        from confusius._cli import build_datasets_parser

        ns = build_datasets_parser().parse_args([])
        assert ns.list is False

    def test_run_datasets_list_calls_list_datasets(self, monkeypatch) -> None:
        import confusius.datasets
        from confusius._cli import build_datasets_parser, run_datasets

        called = False

        def fake_list_datasets() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(confusius.datasets, "list_datasets", fake_list_datasets)
        run_datasets(build_datasets_parser().parse_args(["--list"]))
        assert called

    def test_main_dispatches_to_datasets(self, monkeypatch) -> None:
        import confusius._cli as cli

        seen = {}

        def fake_run_datasets(args) -> None:
            seen["list"] = args.list

        monkeypatch.setattr("sys.argv", ["confusius", "datasets", "--list"])
        monkeypatch.setattr(cli, "run_datasets", fake_run_datasets)
        cli.main()
        assert seen == {"list": True}


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

    def test_integer_nifti_opens_as_labels_layer(
        self, make_napari_viewer, integer_nifti_path: Path
    ) -> None:
        from napari.layers import Labels

        from confusius._cli import build_parser, run

        args = build_parser().parse_args([str(integer_nifti_path)])
        viewer = make_napari_viewer()
        run(args, viewer=viewer)

        layer = viewer.layers[integer_nifti_path.name]
        assert isinstance(layer, Labels)

    def test_float_nifti_opens_as_image_layer(
        self, make_napari_viewer, float_nifti_path: Path
    ) -> None:
        from napari.layers import Image

        from confusius._cli import build_parser, run

        args = build_parser().parse_args([str(float_nifti_path)])
        viewer = make_napari_viewer()
        run(args, viewer=viewer)

        layer = viewer.layers[float_nifti_path.name]
        assert isinstance(layer, Image)
