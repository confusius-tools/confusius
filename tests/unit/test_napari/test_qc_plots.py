"""Unit tests for the QCPlotsWidget export controls."""

from __future__ import annotations

import numpy as np
import xarray as xr


def test_dvars_export_button_and_tsv_output(make_napari_viewer, tmp_path):
    from confusius._napari._qc_plots import QCPlotsWidget

    viewer = make_napari_viewer()
    widget = QCPlotsWidget(viewer)
    dvars = xr.DataArray(
        np.array([0.2, 0.4, 0.6]),
        dims=["time"],
        coords={"time": xr.DataArray(np.array([1.0, 1.5, 2.0]), dims=["time"])},
    )

    widget.update_dvars(dvars, layer_name="scan")

    out_path = tmp_path / "dvars.tsv"
    widget._write_dvars_delimited(out_path, delimiter="\t")

    rows = [line.split("\t") for line in out_path.read_text().splitlines()]
    assert widget._dvars_export_button.isEnabled()
    assert rows == [
        ["time", "DVARS"],
        ["1", "0.2"],
        ["1.5", "0.4"],
        ["2", "0.6"],
    ]


def test_dvars_csv_output_without_time_coordinate(make_napari_viewer, tmp_path):
    from confusius._napari._qc_plots import QCPlotsWidget

    viewer = make_napari_viewer()
    widget = QCPlotsWidget(viewer)
    dvars = xr.DataArray(np.array([1.0, 2.5, 3.0]), dims=["time"])

    widget.update_dvars(dvars)

    out_path = tmp_path / "dvars.csv"
    widget._write_dvars_delimited(out_path, delimiter=",")

    rows = [line.split(",") for line in out_path.read_text().splitlines()]
    assert rows == [
        ["time", "DVARS"],
        ["0", "1"],
        ["1", "2.5"],
        ["2", "3"],
    ]
