"""Read BIDS physio files into a signals table.

A BIDS physio recording is stored as a tab-separated values file without a header,
accompanied by a JSON sidecar that defines the column names and timing metadata.
This reader returns a `pandas.DataFrame` with those column names applied.

When the sidecar does not already declare a `time` column, one is synthesized from
`SamplingFrequency` and `StartTime` so the result can be aligned directly with fUSI
acquisitions and other time-based ConfUSIus data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_physio"]


def load_physio(path: str | Path) -> pd.DataFrame:
    """Load a BIDS physio table into a DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a BIDS physio TSV file, typically ending in `_physio.tsv` or
        `_physio.tsv.gz`.

    Returns
    -------
    pandas.DataFrame
        Physio table with sidecar-defined column names. When the sidecar does not
        include a `time` column, one is inserted first using `SamplingFrequency`
        and `StartTime`. The sidecar JSON fields are copied to `DataFrame.attrs`.

    Raises
    ------
    FileNotFoundError
        If the matching JSON sidecar does not exist.
    ValueError
        If the sidecar JSON is invalid, does not define a non-empty `Columns` list,
        if the TSV width does not match `Columns`, or if timing metadata is needed
        to synthesize `time` but missing or invalid.
    """
    path = Path(path)
    sidecar_path = _sidecar_path(path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Could not find BIDS physio sidecar: {sidecar_path}.")

    try:
        metadata = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid BIDS physio sidecar JSON: {sidecar_path}.") from exc

    columns = metadata.get("Columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("BIDS physio sidecar must define a non-empty 'Columns' list.")

    frame = pd.read_csv(path, sep="\t", header=None)
    if frame.shape[1] != len(columns):
        raise ValueError(
            "BIDS physio sidecar 'Columns' length does not match TSV width."
        )

    frame.columns = [str(column) for column in columns]
    if "time" not in frame.columns:
        sampling_frequency = metadata.get("SamplingFrequency")
        if not isinstance(sampling_frequency, int | float) or sampling_frequency <= 0:
            raise ValueError(
                "BIDS physio sidecar must define a positive numeric "
                "'SamplingFrequency' when 'time' is absent from 'Columns'."
            )

        start_time = metadata.get("StartTime", 0.0)
        if not isinstance(start_time, int | float):
            raise ValueError("BIDS physio sidecar 'StartTime' must be numeric.")

        frame.insert(
            0,
            "time",
            start_time + np.arange(len(frame), dtype=float) / float(sampling_frequency),
        )

    frame.attrs.update(metadata)
    return frame


def _sidecar_path(path: Path) -> Path:
    """Return the JSON sidecar path matching a BIDS physio TSV file."""
    if path.suffix.lower() == ".gz":
        return path.with_suffix("").with_suffix(".json")
    return path.with_suffix(".json")
