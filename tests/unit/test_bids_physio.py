"""Tests for the confusius.bids.physio module."""

from __future__ import annotations

import gzip
import json

import numpy as np
import numpy.testing as npt
import pytest
from pandas.testing import assert_frame_equal

from confusius.bids import load_physio


class TestReadPhysio:
    """Reading BIDS physio files into a DataFrame."""

    def test_reads_columns_and_synthesizes_time_from_sidecar(self, tmp_path):
        """Sidecar columns and timing metadata produce a ready-to-compare table."""
        path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv.gz"
        with gzip.open(path, "wt") as f:
            f.write("1\t10\n2\t11\n3\t12\n")

        path.with_suffix("").with_suffix(".json").write_text(
            json.dumps(
                {
                    "Columns": ["pulse", "resp"],
                    "SamplingFrequency": 2.0,
                    "StartTime": -0.5,
                    "Manufacturer": "Acme",
                }
            )
        )

        frame = load_physio(path)

        expected = {
            "time": np.array([-0.5, 0.0, 0.5]),
            "pulse": np.array([1, 2, 3]),
            "resp": np.array([10, 11, 12]),
        }
        assert_frame_equal(frame, frame.__class__(expected))
        assert frame.attrs["Columns"] == ["pulse", "resp"]
        assert frame.attrs["SamplingFrequency"] == 2.0
        assert frame.attrs["StartTime"] == -0.5
        assert frame.attrs["Manufacturer"] == "Acme"

    def test_keeps_existing_time_column(self, tmp_path):
        """A sidecar-declared time column is kept instead of being synthesized."""
        path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv"
        path.write_text("0.1\t1\n0.2\t2\n")
        path.with_suffix(".json").write_text(json.dumps({"Columns": ["time", "pulse"]}))

        frame = load_physio(path)

        assert list(frame.columns) == ["time", "pulse"]
        npt.assert_allclose(frame["time"], np.array([0.1, 0.2]))
        npt.assert_array_equal(frame["pulse"], np.array([1, 2]))

    def test_rejects_column_count_mismatch(self, tmp_path):
        """A sidecar whose column count disagrees with the TSV is rejected."""
        path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv"
        path.write_text("1\t10\n2\t11\n")
        path.with_suffix(".json").write_text(
            json.dumps({"Columns": ["pulse"], "SamplingFrequency": 10.0})
        )

        with pytest.raises(
            ValueError, match="Columns' length does not match TSV width"
        ):
            load_physio(path)

    def test_rejects_missing_sampling_frequency_when_time_absent(self, tmp_path):
        """Synthesizing time requires a positive numeric sampling frequency."""
        path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv"
        path.write_text("1\n2\n")
        path.with_suffix(".json").write_text(json.dumps({"Columns": ["pulse"]}))

        with pytest.raises(ValueError, match="SamplingFrequency"):
            load_physio(path)

    def test_rejects_missing_sidecar(self, tmp_path):
        """A BIDS physio table without its JSON sidecar is rejected."""
        path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv"
        path.write_text("1\n2\n")

        with pytest.raises(FileNotFoundError, match="sidecar"):
            load_physio(path)

    def test_rejects_invalid_sidecar_json(self, tmp_path):
        """A malformed sidecar JSON file is rejected."""
        path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv"
        path.write_text("1\n2\n")
        path.with_suffix(".json").write_text("{not json")

        with pytest.raises(ValueError, match="Invalid BIDS physio sidecar JSON"):
            load_physio(path)

    def test_rejects_missing_columns_definition(self, tmp_path):
        """A sidecar must define a non-empty Columns list."""
        path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv"
        path.write_text("1\n2\n")
        path.with_suffix(".json").write_text(json.dumps({"Columns": []}))

        with pytest.raises(ValueError, match="non-empty 'Columns' list"):
            load_physio(path)

    def test_rejects_non_numeric_start_time_when_time_absent(self, tmp_path):
        """Synthesizing time requires a numeric start time."""
        path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv"
        path.write_text("1\n2\n")
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "Columns": ["pulse"],
                    "SamplingFrequency": 10.0,
                    "StartTime": "early",
                }
            )
        )

        with pytest.raises(ValueError, match="StartTime"):
            load_physio(path)
