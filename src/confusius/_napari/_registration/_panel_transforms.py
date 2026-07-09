"""Transform payload and panel-specific transform helpers for the napari registration panel."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from napari.layers.utils.layer_utils import calc_data_range
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import QFileDialog

from confusius._dims import SPATIAL_DIMS
from confusius._napari._registration._panel_utils import (
    _get_image_display_kwargs_from_layer,
    _get_source_dataarray,
    _prepare_between_scan_data,
)
from confusius._napari._registration._panel_worker_state import on_registration_failed
from confusius._napari._registration._transform_payloads import (
    AffineTransformPayload,
    OutputGridPayload,
    TransformPayload,
    get_affine_transform_from_payload,
    get_bspline_transform_from_payload,
    get_input_grid_from_payload,
    get_output_grid_from_payload,
    load_transform_payload,
    make_output_grid_payload,
    save_transform_payload,
)
from confusius.plotting.napari import plot_napari
from confusius.registration import (
    invert_displacement_field,
    resample_volume,
    sample_displacement_field,
)

if TYPE_CHECKING:
    from napari.layers import Layer

    from confusius._napari._registration._panel import (
        ApplyTransformPayload,
        RegistrationPanel,
        TransformSourceData,
    )


def _get_affine_payload_from_layer(layer: "Layer") -> AffineTransformPayload | None:
    """Return the stored affine transform payload for a napari layer.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer whose metadata should be inspected.

    Returns
    -------
    AffineTransformPayload or None
        Stored payload when present and affine, otherwise `None`.
    """
    payload = layer.metadata.get("confusius_transform")
    if not isinstance(payload, dict) or payload.get("kind") != "affine":
        return None
    get_affine_transform_from_payload(payload)
    return cast("AffineTransformPayload", payload)


def _get_spatial_manual_affine_from_layer(
    layer: "Layer", *, spatial_dims: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Return the spatial sub-affine from a napari layer's manual transform.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer whose manual napari affine should be extracted.
    spatial_dims : sequence of str
        Spatial dimension names, in the exact order expected by registration.

    Returns
    -------
    (N+1, N+1) numpy.ndarray
        Spatial homogeneous affine in world coordinates.

    Raises
    ------
    ValueError
        If the layer does not contain the requested spatial dimensions.
    ValueError
        If the layer affine has an unexpected shape.
    ValueError
        If the manual affine mixes selected spatial axes with ignored axes.
    """
    data = _get_source_dataarray(layer)
    layer_dims = [str(dim) for dim in data.dims]
    missing_dims = [dim for dim in spatial_dims if dim not in layer_dims]
    if missing_dims:
        raise ValueError(
            "Selected manual napari transform does not contain spatial dims "
            f"{missing_dims}."
        )

    affine = np.asarray(layer.affine.affine_matrix, dtype=float)
    expected_shape = (len(layer_dims) + 1, len(layer_dims) + 1)
    if affine.shape != expected_shape:
        raise ValueError(
            f"Selected manual napari transform has shape {affine.shape}, "
            f"but layer '{layer.name}' expects {expected_shape}."
        )

    spatial_indices = [layer_dims.index(dim) for dim in spatial_dims]
    ignored_indices = [i for i in range(len(layer_dims)) if i not in spatial_indices]
    linear = affine[:-1, :-1]

    if ignored_indices:
        spatial_to_ignored = linear[np.ix_(spatial_indices, ignored_indices)]
        ignored_to_spatial = linear[np.ix_(ignored_indices, spatial_indices)]
        if not np.allclose(spatial_to_ignored, 0.0) or not np.allclose(
            ignored_to_spatial, 0.0
        ):
            raise ValueError(
                "Selected manual napari transform mixes spatial axes with ignored "
                "non-spatial axes, so it cannot be used as a registration "
                "initialization."
            )

    spatial_affine = np.eye(len(spatial_dims) + 1, dtype=float)
    spatial_affine[:-1, :-1] = linear[np.ix_(spatial_indices, spatial_indices)]
    spatial_affine[:-1, -1] = affine[np.ix_(spatial_indices, [-1])].ravel()
    return spatial_affine


def _make_manual_transform_payload(layer: "Layer") -> AffineTransformPayload:
    """Build an affine payload from a layer's manual napari transform.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer whose current manual napari transform should be serialized.

    Returns
    -------
    AffineTransformPayload
        JSON-serializable affine payload representing the visible manual layer transform
        on the layer's own spatial output grid.
    """
    data = _get_source_dataarray(layer)
    spatial_data = _prepare_between_scan_data(data)
    spatial_dims = [str(dim) for dim in spatial_data.dims if dim in SPATIAL_DIMS]
    manual_affine = _get_spatial_manual_affine_from_layer(
        layer, spatial_dims=spatial_dims
    )
    pull_affine = np.linalg.inv(manual_affine)
    return {
        "kind": "affine",
        "name": f"{layer.name} (manual)",
        "affine": pull_affine.tolist(),
        "source_layer_name": layer.name,
        "target_layer_name": layer.name,
        "operation": "manual_napari_transform",
        "transform_model": "affine",
        "metric": "manual",
        "output_grid": make_output_grid_payload(spatial_data),
        "input_grid": make_output_grid_payload(spatial_data),
        "diagnostics": {
            "metric": "manual",
            "final_metric_value": 0.0,
            "n_iterations": 0,
            "stop_condition": "Saved from manual napari layer transform.",
            "status": "completed",
        },
    }


def get_transform_source_data(value: object) -> "TransformSourceData | None":
    """Return validated transform-source combo data.

    Parameters
    ----------
    value : object
        Raw combo-box payload to validate.

    Returns
    -------
    tuple[str, str] or None
        Validated `(kind, name)` pair, or `None` when the payload does not match the
        expected transform-source schema.
    """
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    source_kind, source_name = value
    if not isinstance(source_name, str):
        return None
    if source_kind == "loaded":
        return ("loaded", source_name)
    if source_kind == "layer":
        return ("layer", source_name)
    if source_kind == "manual":
        return ("manual", source_name)
    return None


def get_transform_payload_from_metadata(payload: object) -> TransformPayload | None:
    """Return a validated transform payload stored in layer metadata.

    Parameters
    ----------
    payload : object
        Raw metadata payload to validate.

    Returns
    -------
    TransformPayload or None
        Validated transform payload, or `None` when the metadata does not contain a
        supported transform payload.
    """
    if not isinstance(payload, dict):
        return None
    payload_mapping = cast("dict[str, object]", payload)
    kind = payload_mapping.get("kind")
    if kind == "affine":
        get_affine_transform_from_payload(payload_mapping)
        return cast("TransformPayload", payload_mapping)
    if kind == "bspline":
        get_bspline_transform_from_payload(payload_mapping)
        return cast("TransformPayload", payload_mapping)
    return None


def get_transform_source_label(
    payload: TransformPayload, *, suffix: str | None = None
) -> str:
    """Return a user-facing label for a transform payload.

    Parameters
    ----------
    payload : TransformPayload
        Transform payload to label.
    suffix : str, optional
        Unused legacy suffix parameter kept to avoid wider churn.

    Returns
    -------
    str
        Label shown in transform selectors.
    """
    del suffix
    return payload["name"]


def get_available_transform_payloads(
    panel: "RegistrationPanel",
) -> list[TransformPayload]:
    """Return all transform payloads currently available in the UI.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose loaded payload and viewer layers are queried.

    Returns
    -------
    list of TransformPayload
        Loaded payload plus any validated payloads found on viewer layers.
    """
    payloads: list[TransformPayload] = []
    if panel._loaded_transform_payload is not None:
        payloads.append(panel._loaded_transform_payload)
    for layer in panel.viewer.layers:
        payload = get_transform_payload_from_metadata(
            layer.metadata.get("confusius_transform")
        )
        if payload is not None:
            payloads.append(payload)
    return payloads


def update_apply_transform_button_tooltips(panel: "RegistrationPanel") -> None:
    """Refresh apply-transform button tooltips for the current transform selection.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose transform action buttons should be updated.
    """
    forward_tooltip = "Apply the selected transform onto its target/output grid."
    inverse_tooltip = (
        "Apply the inverse of the selected transform onto its source/input grid."
    )

    payload = get_selected_transform_payload(panel)
    if payload is not None and payload["kind"] == "bspline":
        inverse_tooltip = (
            inverse_tooltip
            + " For B-spline transforms this inverse is approximate: ConfUSIus "
            "samples the transform onto the source grid, then inverts the resulting "
            "displacement field."
        )

    panel._apply_transform_btn.setToolTip(forward_tooltip)
    panel._apply_inverse_transform_btn.setToolTip(inverse_tooltip)


def refresh_transform_controls(panel: "RegistrationPanel") -> None:
    """Refresh the transform, initialization, and target selectors from the current viewer state.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose transform selectors are updated.
    """
    source_data = panel._transform_source_combo.currentData()
    initialization_data = panel._initialization_combo.currentData()
    target_name = panel._transform_target_combo.currentText()

    transform_options: list[tuple[str, tuple[str, str]]] = []
    if panel._loaded_transform_payload is not None:
        transform_options.append(
            (
                get_transform_source_label(
                    panel._loaded_transform_payload,
                    suffix="loaded",
                ),
                ("loaded", ""),
            )
        )
    for layer in panel.viewer.layers:
        payload = get_transform_payload_from_metadata(
            layer.metadata.get("confusius_transform")
        )
        if payload is None:
            continue
        transform_options.append(
            (
                get_transform_source_label(payload, suffix=layer.name),
                ("layer", layer.name),
            )
        )

    manual_transform_options: list[tuple[str, tuple[str, str]]] = []
    manual_initialization_options: list[tuple[str, tuple[str, str]]] = []
    for layer in panel.viewer.layers:
        try:
            data = _get_source_dataarray(layer)
            spatial_dims = [str(dim) for dim in data.dims if dim in SPATIAL_DIMS]
            if not spatial_dims:
                continue
            manual_affine = _get_spatial_manual_affine_from_layer(
                layer,
                spatial_dims=spatial_dims,
            )
        except Exception:  # noqa: BLE001
            continue
        if np.allclose(manual_affine, np.eye(len(spatial_dims) + 1)):
            continue
        manual_option = (f"{layer.name} (manual)", ("manual", layer.name))
        manual_transform_options.append(manual_option)
        manual_initialization_options.append(manual_option)

    panel._transform_source_combo.blockSignals(True)
    panel._transform_source_combo.clear()
    for label, data in transform_options:
        panel._transform_source_combo.addItem(label, data)
    for label, data in manual_transform_options:
        panel._transform_source_combo.addItem(label, data)
    panel._transform_source_combo.blockSignals(False)

    panel._initialization_combo.blockSignals(True)
    panel._initialization_combo.clear()
    panel._initialization_combo.addItem("center_geometry", "center_geometry")
    panel._initialization_combo.addItem("center_moments", "center_moments")
    panel._initialization_combo.addItem("none", None)
    for label, data in transform_options:
        source_kind, source_name = data
        if source_kind == "loaded":
            if panel._loaded_transform_payload is None:
                continue
            if panel._loaded_transform_payload["kind"] != "affine":
                continue
        elif source_kind == "layer":
            layer = panel._get_layer_by_name(source_name)
            if layer is None or _get_affine_payload_from_layer(layer) is None:
                continue
        panel._initialization_combo.addItem(label, data)
    for label, data in manual_initialization_options:
        panel._initialization_combo.addItem(label, data)
    panel._initialization_combo.blockSignals(False)

    panel._transform_target_combo.blockSignals(True)
    panel._transform_target_combo.clear()
    panel._transform_target_combo.addItems(
        [layer.name for layer in panel.viewer.layers]
    )
    panel._transform_target_combo.blockSignals(False)

    if source_data is not None:
        for i in range(panel._transform_source_combo.count()):
            if panel._transform_source_combo.itemData(i) == source_data:
                panel._transform_source_combo.setCurrentIndex(i)
                break

    if initialization_data is not None:
        for i in range(panel._initialization_combo.count()):
            if panel._initialization_combo.itemData(i) == initialization_data:
                panel._initialization_combo.setCurrentIndex(i)
                break

    target_index = panel._transform_target_combo.findText(target_name)
    if target_index >= 0:
        panel._transform_target_combo.setCurrentIndex(target_index)

    update_apply_transform_button_tooltips(panel)


def get_selected_transform_payload(
    panel: "RegistrationPanel",
) -> TransformPayload | None:
    """Return the currently selected transform payload.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose transform selection is read.

    Returns
    -------
    TransformPayload or None
        Selected transform payload, or `None` when no valid selection is available.
    """
    source_data = get_transform_source_data(panel._transform_source_combo.currentData())
    if source_data is None:
        return None

    source_kind, source_name = source_data
    if source_kind == "loaded":
        return panel._loaded_transform_payload
    if not source_name:
        return None
    layer = panel._get_layer_by_name(source_name)
    if layer is None:
        return None
    if source_kind == "layer":
        return get_transform_payload_from_metadata(
            layer.metadata.get("confusius_transform")
        )
    if source_kind == "manual":
        return _make_manual_transform_payload(layer)
    return None


def get_selected_center_initialization(
    panel: "RegistrationPanel",
) -> Literal["center_geometry", "center_moments"] | None:
    """Return the selected built-in centering initialization, if any.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose initialization selector is read.

    Returns
    -------
    {"center_geometry", "center_moments"} or None
        Selected built-in centering initialization, or `None` when a different kind of
        initialization is currently selected.
    """
    value = panel._initialization_combo.currentData()
    if value in {"center_geometry", "center_moments"}:
        return value
    return None


def get_selected_initial_transform_payload(
    panel: "RegistrationPanel",
) -> AffineTransformPayload | None:
    """Return the payload selected for registration initialization, if any.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose initialization selector is read.

    Returns
    -------
    AffineTransformPayload or None
        Selected affine transform payload, or `None` when no affine payload is currently
        selected for initialization.
    """
    source_data = get_transform_source_data(panel._initialization_combo.currentData())
    if source_data is None:
        return None

    source_kind, source_name = source_data
    if source_kind == "loaded":
        if (
            panel._loaded_transform_payload is not None
            and panel._loaded_transform_payload["kind"] == "affine"
        ):
            return panel._loaded_transform_payload
        return None
    if source_kind != "layer" or not source_name:
        return None
    layer = panel._get_layer_by_name(source_name)
    if layer is None:
        return None
    return _get_affine_payload_from_layer(layer)


def get_selected_manual_initialization_layer(
    panel: "RegistrationPanel",
) -> "Layer | None":
    """Return the layer selected for manual napari initialization, if any.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose initialization selector is read.

    Returns
    -------
    napari.layers.Layer or None
        Layer selected as the manual napari initialization source, or `None` when no
        manual initialization is currently selected.
    """
    source_data = get_transform_source_data(panel._initialization_combo.currentData())
    if source_data is None:
        return None

    source_kind, source_name = source_data
    if source_kind != "manual" or not source_name:
        return None
    return panel._get_layer_by_name(source_name)


def get_selected_initial_transform(
    panel: "RegistrationPanel",
    moving: xr.DataArray,
    *,
    moving_layer: "Layer | None" = None,
    fixed_layer: "Layer | None" = None,
) -> tuple[npt.NDArray[np.float64] | None, str | None]:
    """Return the selected initialization affine and its source label.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose initialization selector is read.
    moving : xarray.DataArray
        Moving DataArray defining the spatial dimensions of the registration.
    moving_layer : napari.layers.Layer, optional
        Layer used as the moving input; required when a manual initialization is
        selected.
    fixed_layer : napari.layers.Layer, optional
        Layer used as the fixed input; required when a manual initialization is
        selected.

    Returns
    -------
    affine : (N+1, N+1) numpy.ndarray or None
        Selected initialization affine in homogeneous coordinates, or `None` when no
        initialization is selected.
    label : str or None
        Human-readable label for the selected initialization source, or `None` when no
        initialization is selected.

    Raises
    ------
    ValueError
        If a manual initialization is selected but the moving and fixed layers are not
        provided, or the selected manual layer is not the current moving or fixed layer.
    """
    payload = get_selected_initial_transform_payload(panel)
    if payload is not None:
        return get_affine_transform_from_payload(payload), payload["name"]

    layer = get_selected_manual_initialization_layer(panel)
    if layer is None:
        return None, None
    if moving_layer is None or fixed_layer is None:
        raise ValueError("Select moving and fixed layers.")
    if layer not in {moving_layer, fixed_layer}:
        raise ValueError(
            "Selected manual initialization must come from the current moving "
            "or fixed layer."
        )

    spatial_dims = [str(dim) for dim in moving.dims if dim in SPATIAL_DIMS]
    moving_affine = _get_spatial_manual_affine_from_layer(
        moving_layer,
        spatial_dims=spatial_dims,
    )
    fixed_affine = _get_spatial_manual_affine_from_layer(
        fixed_layer,
        spatial_dims=spatial_dims,
    )
    affine = np.linalg.inv(moving_affine) @ fixed_affine
    return affine, f"{layer.name} (manual)"


def validate_initial_transform_selection(
    panel: "RegistrationPanel",
    *,
    operation: Literal["register_volume", "register_volumewise"],
    moving: xr.DataArray,
    fixed: xr.DataArray | None = None,
) -> str | None:
    """Return an inline validation message for transform initialization.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose selection is validated.
    operation : {"register_volume", "register_volumewise"}
        Registration operation the panel is currently configured to run.
    moving : xarray.DataArray
        Moving DataArray used to check initialization transform shapes.
    fixed : xarray.DataArray, optional
        Fixed DataArray; required when an initialization is selected.

    Returns
    -------
    str or None
        Human-readable validation message, or `None` when the current selection is valid
        (or no initialization is selected).
    """
    if operation != "register_volume":
        return None
    if (
        get_selected_initial_transform_payload(panel) is None
        and get_selected_manual_initialization_layer(panel) is None
    ):
        return None
    if fixed is None:
        return "Select a fixed layer."

    moving_layer = panel._selected_layer(panel._moving_combo)
    fixed_layer = panel._selected_layer(panel._fixed_combo)

    try:
        affine, _ = get_selected_initial_transform(
            panel,
            moving,
            moving_layer=moving_layer,
            fixed_layer=fixed_layer,
        )
    except Exception as exc:  # noqa: BLE001
        return str(exc)

    if affine is None:
        return None

    expected_shape = (moving.ndim + 1, moving.ndim + 1)
    if affine.shape != expected_shape:
        return (
            f"Selected initialization transform has shape {affine.shape}, "
            f"but this registration expects {expected_shape}."
        )
    return None


def save_selected_transform(panel: "RegistrationPanel") -> None:
    """Prompt for a destination path and save the currently selected transform payload.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose selected transform should be saved.
    """
    payload = get_selected_transform_payload(panel)
    if payload is None:
        panel._set_error("Select a transform to save.")
        return

    default_name = payload["name"].replace("/", "-")
    suffix = ".json" if payload["kind"] == "affine" else ".nii.gz"
    file_filter = (
        "JSON files (*.json)"
        if payload["kind"] == "affine"
        else "NIfTI files (*.nii *.nii.gz)"
    )
    start = str(Path.home() / f"{default_name}{suffix}")
    path_str, _ = QFileDialog.getSaveFileName(
        panel, "Save transform", start, file_filter
    )
    if not path_str:
        return

    save_transform_payload(path_str, payload)
    show_info(f"Saved transform: {path_str}")


def load_transform(panel: "RegistrationPanel") -> None:
    """Prompt for a transform file, load it into the panel state, and refresh the selectors.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel that should receive the loaded transform.
    """
    start = str(Path.home())
    path_str, _ = QFileDialog.getOpenFileName(
        panel,
        "Load transform",
        start,
        "Transform files (*.json *.nii *.nii.gz *.zarr)",
    )
    if not path_str:
        return

    try:
        panel._loaded_transform_payload = load_transform_payload(path_str)
    except Exception as exc:  # noqa: BLE001
        panel._set_error(str(exc))
        show_error(str(exc))
        return

    refresh_transform_controls(panel)
    for i in range(panel._transform_source_combo.count()):
        if panel._transform_source_combo.itemData(i) == ("loaded", ""):
            panel._transform_source_combo.setCurrentIndex(i)
            break
    show_info(f"Loaded transform: {panel._loaded_transform_payload['name']}")


def _get_inverse_output_grid(
    panel: "RegistrationPanel", payload: TransformPayload
) -> OutputGridPayload:
    """Return the output grid to use when applying a transform inverse.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel that owns the transform selection.
    payload : TransformPayload
        Selected transform payload.

    Returns
    -------
    OutputGridPayload
        Grid of the original moving/source layer.

    Raises
    ------
    ValueError
        If the payload predates `input_grid` and the source layer is not available to
        re-derive it.
    """
    input_grid = get_input_grid_from_payload(payload)
    if input_grid is not None:
        return input_grid

    source_layer = panel._get_layer_by_name(payload["source_layer_name"])
    if source_layer is None:
        raise ValueError(
            "Transform payload does not contain an input grid. Reload the original "
            "source layer or re-save the transform from a newer registration result."
        )

    source = _prepare_between_scan_data(_get_source_dataarray(source_layer))
    return make_output_grid_payload(source)


def apply_selected_transform(panel: "RegistrationPanel") -> None:
    """Start a background resampling worker for the selected transform and target layer.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose selected transform and target layer should be used.
    """
    payload = get_selected_transform_payload(panel)
    if payload is None:
        panel._set_error("Select a transform to apply.")
        return

    moving_layer = panel._selected_layer(panel._transform_target_combo)
    if moving_layer is None:
        panel._set_error("Select an input layer to transform.")
        return

    try:
        moving = _get_source_dataarray(moving_layer)
        if payload["kind"] == "affine":
            transform = get_affine_transform_from_payload(payload)
        else:
            transform = get_bspline_transform_from_payload(payload)
        output_grid = get_output_grid_from_payload(payload)
    except Exception as exc:  # noqa: BLE001
        panel._set_error(str(exc))
        return

    worker = thread_worker(resample_volume)(
        moving,
        transform,
        shape=output_grid["shape"],
        spacing=output_grid["spacing"],
        origin=output_grid["origin"],
        dims=output_grid["dims"],
        interpolation=panel._current_resample_interpolation(),
    )
    apply_payload: ApplyTransformPayload = {
        "moving_layer_name": moving_layer.name,
        "target_layer_name": payload["target_layer_name"],
        "transform_source": payload["name"],
        "direction": "forward",
    }
    panel._worker = worker
    panel._begin_work()

    worker.returned.connect(
        lambda result: on_apply_transform_finished(panel, apply_payload, result)
    )
    worker.errored.connect(lambda exc: on_registration_failed(panel, exc))
    worker.finished.connect(panel._end_work)
    worker.start()


def apply_selected_inverse_transform(panel: "RegistrationPanel") -> None:
    """Start a background resampling worker for the inverse of the selected transform.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose selected transform and target layer should be used.
    """
    payload = get_selected_transform_payload(panel)
    if payload is None:
        panel._set_error("Select a transform to apply.")
        return

    moving_layer = panel._selected_layer(panel._transform_target_combo)
    if moving_layer is None:
        panel._set_error("Select an input layer to transform.")
        return

    try:
        moving = _get_source_dataarray(moving_layer)
        output_grid = _get_inverse_output_grid(panel, payload)
        if payload["kind"] == "affine":
            transform = np.linalg.inv(get_affine_transform_from_payload(payload))
        else:
            transform = invert_displacement_field(
                sample_displacement_field(
                    get_bspline_transform_from_payload(payload),
                    shape=output_grid["shape"],
                    spacing=output_grid["spacing"],
                    origin=output_grid["origin"],
                    dims=output_grid["dims"],
                )
            )
    except Exception as exc:  # noqa: BLE001
        panel._set_error(str(exc))
        return

    worker = thread_worker(resample_volume)(
        moving,
        transform,
        shape=output_grid["shape"],
        spacing=output_grid["spacing"],
        origin=output_grid["origin"],
        dims=output_grid["dims"],
        interpolation=panel._current_resample_interpolation(),
    )
    apply_payload: ApplyTransformPayload = {
        "moving_layer_name": moving_layer.name,
        "target_layer_name": payload["source_layer_name"],
        "transform_source": payload["name"],
        "direction": "inverse",
    }
    panel._worker = worker
    panel._begin_work()

    worker.returned.connect(
        lambda result: on_apply_transform_finished(panel, apply_payload, result)
    )
    worker.errored.connect(lambda exc: on_registration_failed(panel, exc))
    worker.finished.connect(panel._end_work)
    worker.start()


def on_apply_transform_finished(
    panel: "RegistrationPanel", payload: "ApplyTransformPayload", result: xr.DataArray
) -> None:
    """Add the finished transformed layer to the viewer and attach apply-transform metadata.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel that initiated the resampling worker.
    payload : ApplyTransformPayload
        UI snapshot captured when the worker was started; carries the moving and target
        layer names together with the source transform label.
    result : xarray.DataArray
        Resampled DataArray returned by the worker.
    """
    registered = result.copy(deep=False)
    registered.attrs = registered.attrs.copy()
    registered.attrs["registration_operation"] = (
        "apply_inverse_transform"
        if payload["direction"] == "inverse"
        else "apply_transform"
    )

    name = panel._make_unique_layer_name(
        f"{payload['moving_layer_name']} → {payload['target_layer_name']}"
    )
    source_layer = panel._get_layer_by_name(payload["moving_layer_name"])
    display_kwargs = (
        _get_image_display_kwargs_from_layer(source_layer)
        if source_layer is not None
        else {}
    )
    contrast_limits = tuple(calc_data_range(registered.data))
    _, layer = plot_napari(
        registered,
        viewer=panel.viewer,
        name=name,
        show_colorbar=False,
        contrast_limits=contrast_limits,
        **display_kwargs,
    )
    layer.metadata["xarray"] = registered
    layer.metadata["transform_source"] = payload["transform_source"]
    layer.metadata["registration_operation"] = registered.attrs[
        "registration_operation"
    ]
    layer.metadata["registration_parameters"] = payload.copy()
    panel.viewer.layers.selection.active = layer
    panel._status.hide()
    show_info(f"Added transformed layer: {layer.name}")
