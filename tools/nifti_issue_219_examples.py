"""Small manual test cases for NIfTI save/load behavior around issue 219.

Run with:
    uv run python tools/nifti_issue_219_examples.py

Optionally keep the generated files:
    uv run python tools/nifti_issue_219_examples.py --output-dir /tmp/confusius-nifti-219
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import xarray as xr
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from confusius.io.nifti import load_nifti, save_nifti

console = Console()

Status = Literal["ok", "changed", "info"]


def _jsonify(value: object) -> str:
    """Pretty-print small Python values for terminal display."""
    return json.dumps(value, indent=2, default=str)


def _status_text(status: Status, text: str) -> str:
    """Return rich-colored status text."""
    styles = {
        "ok": "bold green",
        "changed": "bold red",
        "info": "bold yellow",
    }
    return f"[{styles[status]}]{text}[/{styles[status]}]"


def _compare_values(
    before: object,
    after: object,
    *,
    expect_equal: bool,
) -> Status:
    """Compare two values and return a display status."""
    if isinstance(before, np.ndarray | list | tuple) or isinstance(
        after, np.ndarray | list | tuple
    ):
        before_array = np.asarray(before)
        after_array = np.asarray(after)
        if before_array.dtype.kind in "iuf" and after_array.dtype.kind in "iuf":
            equal = bool(np.allclose(before_array, after_array))
        else:
            equal = bool(np.array_equal(before_array, after_array))
    else:
        equal = before == after

    if expect_equal:
        return "ok" if equal else "changed"
    return "info" if not equal else "ok"


def _compare_attr_subset(
    before: dict[str, object],
    after: dict[str, object],
    *,
    expect_equal: bool,
) -> Status:
    """Compare attrs while allowing header-derived attrs to be added on load."""
    missing_or_changed = {
        key: value
        for key, value in before.items()
        if key not in after or after[key] != value
    }
    if expect_equal:
        return "ok" if not missing_or_changed else "changed"
    return "info" if missing_or_changed else "ok"


def _coord_snapshot(
    data_array: xr.DataArray, coord_name: str
) -> tuple[list[float], dict[str, object]]:
    """Return a coordinate snapshot as plain Python values."""
    coord = data_array.coords[coord_name]
    return np.asarray(coord.values).tolist(), dict(coord.attrs)


def summarize_case(
    name: str,
    original: xr.DataArray,
    output_path: Path,
    *,
    expect_exact_coord_values: bool = True,
    expect_exact_coord_attrs: bool = True,
) -> None:
    """Save, load, print warnings inline, and show a before/after comparison."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_nifti(original, output_path)
        roundtripped = load_nifti(output_path).compute()

    nifti_img = nib.load(output_path)
    sidecar_path = output_path.with_suffix("").with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}

    np.testing.assert_array_equal(roundtripped.values, original.values)
    assert roundtripped.dims == original.dims
    for attr_name, attr_value in original.attrs.items():
        assert roundtripped.attrs[attr_name] == attr_value
    for coord_name in original.coords:
        if expect_exact_coord_values:
            np.testing.assert_allclose(
                np.asarray(roundtripped.coords[coord_name].values),
                np.asarray(original.coords[coord_name].values),
            )
        if expect_exact_coord_attrs:
            for attr_name, attr_value in original.coords[coord_name].attrs.items():
                assert roundtripped.coords[coord_name].attrs[attr_name] == attr_value

    console.print()
    console.print(Panel.fit(f"[bold]{name}[/bold]\n{output_path}"))

    if caught:
        warnings_table = Table(title="Warnings", show_lines=True)
        warnings_table.add_column("Status", style="yellow", no_wrap=True)
        warnings_table.add_column("Message")
        for warning in caught:
            warnings_table.add_row("⚠", str(warning.message))
        console.print(warnings_table)
    else:
        console.print(_status_text("ok", "No warnings."))

    summary = Table(title="Overview", show_lines=True)
    summary.add_column("Field", no_wrap=True)
    summary.add_column("Before")
    summary.add_column("After")
    summary.add_column("Status", no_wrap=True)
    summary.add_row(
        "dims",
        str(original.dims),
        str(roundtripped.dims),
        _status_text(
            _compare_values(original.dims, roundtripped.dims, expect_equal=True),
            "match",
        ),
    )
    summary.add_row(
        "data shape",
        str(original.shape),
        str(roundtripped.shape),
        _status_text("ok", "match"),
    )
    summary.add_row(
        "nifti shape",
        "-",
        str(nifti_img.shape),
        _status_text("info", "on-disk layout"),
    )
    summary.add_row(
        "sidecar keys",
        "-",
        str(sorted(sidecar)),
        _status_text("info", "metadata"),
    )
    console.print(summary)

    coords_table = Table(title="Coordinates", show_lines=True)
    coords_table.add_column("Coord", no_wrap=True)
    coords_table.add_column("Before values")
    coords_table.add_column("After values")
    coords_table.add_column("Status", no_wrap=True)
    coords_table.add_column("Before attrs")
    coords_table.add_column("After attrs")

    for coord_name in sorted(set(original.coords) | set(roundtripped.coords)):
        before_values, before_attrs = (
            _coord_snapshot(original, coord_name)
            if coord_name in original.coords
            else ("<missing>", {})
        )
        after_values, after_attrs = (
            _coord_snapshot(roundtripped, coord_name)
            if coord_name in roundtripped.coords
            else ("<missing>", {})
        )
        values_status = _compare_values(
            before_values,
            after_values,
            expect_equal=expect_exact_coord_values,
        )
        attrs_status = _compare_attr_subset(
            before_attrs,
            after_attrs,
            expect_equal=expect_exact_coord_attrs,
        )
        has_added_attrs = before_attrs != after_attrs and attrs_status == "ok"

        if expect_exact_coord_values or expect_exact_coord_attrs:
            if values_status == "changed":
                status = "changed"
                label = (
                    "values differ" if expect_exact_coord_values else "header-derived"
                )
            elif attrs_status == "changed":
                status = "changed"
                label = "attrs differ"
            elif has_added_attrs:
                status = "ok"
                label = "exact (+derived attrs)"
            else:
                status = "ok"
                label = "exact"
        else:
            if values_status == "info":
                status = "info"
                label = "header-derived"
            elif attrs_status == "info":
                status = "info"
                label = "attrs differ"
            elif has_added_attrs:
                status = "info"
                label = "attrs added"
            else:
                status = "ok"
                label = "exact"

        coords_table.add_row(
            coord_name,
            _jsonify(before_values),
            _jsonify(after_values),
            _status_text(status, label),
            _jsonify(before_attrs),
            _jsonify(after_attrs),
        )
    console.print(coords_table)

    if expect_exact_coord_values and expect_exact_coord_attrs:
        console.print(_status_text("ok", "Exact coord roundtrip OK."))
    elif expect_exact_coord_values:
        console.print(
            _status_text(
                "info",
                "Data/dims and coord values roundtrip OK. Coordinate attrs may differ.",
            )
        )
    else:
        console.print(
            _status_text(
                "info",
                "Data/dims roundtrip OK. Canonical coords are header-derived and may differ.",
            )
        )


def build_cases() -> list[tuple[str, xr.DataArray, str, bool, bool]]:
    """Create a few representative save/load scenarios."""
    cases: list[tuple[str, xr.DataArray, str, bool, bool]] = []

    cases.append(
        (
            "01_basic_3d_no_attrs",
            xr.DataArray(
                np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
                dims=["z", "y", "x"],
                coords={
                    "z": [0.0, 1.0],
                    "y": [0.0, 1.0, 2.0],
                    "x": [0.0, 1.0, 2.0, 3.0],
                },
                name="basic_3d",
            ),
            "basic_3d.nii.gz",
            True,
            True,
        )
    )

    cases.append(
        (
            "02_4d_time_with_attrs",
            xr.DataArray(
                np.arange(3 * 2 * 3 * 4, dtype=np.float32).reshape(3, 2, 3, 4),
                dims=["time", "z", "y", "x"],
                coords={
                    "time": xr.DataArray(
                        [0.0, 0.5, 1.0],
                        dims=["time"],
                        attrs={
                            "units": "s",
                            "volume_acquisition_reference": "start",
                            "volume_acquisition_duration": 0.4,
                        },
                    ),
                    "z": xr.DataArray([0.0, 0.1], dims=["z"], attrs={"units": "m"}),
                    "y": xr.DataArray(
                        [0.0, 0.2, 0.4], dims=["y"], attrs={"units": "m"}
                    ),
                    "x": xr.DataArray(
                        [0.0, 0.3, 0.6, 0.9], dims=["x"], attrs={"units": "m"}
                    ),
                },
                attrs={"custom_meta": "time-series"},
                name="time_with_attrs",
            ),
            "time_with_attrs.nii.gz",
            True,
            True,
        )
    )

    cases.append(
        (
            "03_extra_dim_no_time",
            xr.DataArray(
                np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4),
                dims=["component", "z", "x"],
                coords={
                    "component": [0.0, 1.0, 2.0],
                    "z": [0.0, 0.5],
                    "x": [0.0, 1.0, 2.0, 3.0],
                },
                attrs={"custom_meta": "extra-no-time"},
                name="extra_dim_no_time",
            ),
            "extra_dim_no_time.nii.gz",
            True,
            True,
        )
    )

    cases.append(
        (
            "04_extra_dim_with_coord_attrs_and_negative_spacing",
            xr.DataArray(
                np.arange(3 * 2 * 3 * 4, dtype=np.float32).reshape(3, 2, 3, 4),
                dims=["channel", "z", "y", "x"],
                coords={
                    "channel": xr.DataArray(
                        [0.0, -2.0, -4.0],
                        dims=["channel"],
                        attrs={"units": "a.u.", "long_name": "Channel"},
                    ),
                    "z": [0.0, 1.0],
                    "y": [0.0, 1.0, 2.0],
                    "x": [0.0, 1.0, 2.0, 3.0],
                },
                attrs={"custom_meta": "extra-negative"},
                name="extra_negative",
            ),
            "extra_negative.nii.gz",
            True,
            True,
        )
    )

    cases.append(
        (
            "05_negative_spatial_coord_and_coord_attrs",
            xr.DataArray(
                np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
                dims=["z", "y", "x"],
                coords={
                    "z": xr.DataArray(
                        [0.0, 1.0],
                        dims=["z"],
                        attrs={"units": "mm", "long_name": "Depth"},
                    ),
                    "y": [0.0, 1.0, 2.0],
                    "x": [0.0, -1.0, -2.0, -3.0],
                },
                attrs={"custom_meta": "negative-spatial"},
                name="negative_spatial",
            ),
            "negative_spatial.nii.gz",
            True,
            False,
        )
    )

    cases.append(
        (
            "06_irregular_spatial_coord_falls_back_to_header",
            xr.DataArray(
                np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
                dims=["z", "y", "x"],
                coords={
                    "z": [0.0, 2.0],
                    "y": [0.0, 1.0, 3.0],
                    "x": [0.0, 1.0, 2.0, 4.0],
                },
                attrs={"custom_meta": "irregular-spatial"},
                name="irregular_spatial",
            ),
            "irregular_spatial.nii.gz",
            False,
            False,
        )
    )

    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the .nii.gz/.json files should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="confusius-nifti-219-"))
        console.print(f"using temporary directory: [bold]{output_dir}[/bold]")
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"using output directory: [bold]{output_dir}[/bold]")

    for (
        name,
        data_array,
        filename,
        expect_exact_coord_values,
        expect_exact_coord_attrs,
    ) in build_cases():
        summarize_case(
            name,
            data_array,
            output_dir / filename,
            expect_exact_coord_values=expect_exact_coord_values,
            expect_exact_coord_attrs=expect_exact_coord_attrs,
        )

    console.print()
    console.print(_status_text("ok", "All example cases passed."))


if __name__ == "__main__":
    main()
