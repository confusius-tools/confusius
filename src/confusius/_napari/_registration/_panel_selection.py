"""Layer-selection and validation helpers for the napari registration panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qtpy.QtWidgets import QComboBox

from confusius._dims import TIME_DIM
from confusius._napari._registration._panel_transforms import (
    refresh_transform_controls,
    validate_initial_transform_selection,
)
from confusius._napari._registration._panel_utils import (
    _get_source_dataarray,
    _is_registration_source_layer,
    _prepare_between_scan_data,
)

if TYPE_CHECKING:
    from napari.layers import Layer

    from confusius._napari._registration._panel import (
        MetricName,
        RegistrationPanel,
        ResampleInterpolation,
        ScaleMode,
        VolumeTransformType,
        VolumewiseTransformType,
    )


def refresh_layers(panel: "RegistrationPanel") -> None:
    """Repopulate the layer selectors from the viewer.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose layer selectors should be refreshed.
    """
    moving_name = panel._moving_combo.currentText()
    fixed_name = panel._fixed_combo.currentText()

    layer_names = [
        layer.name
        for layer in panel.viewer.layers
        if _is_registration_source_layer(layer)
    ]

    panel._moving_combo.blockSignals(True)
    panel._fixed_combo.blockSignals(True)
    panel._moving_combo.clear()
    panel._fixed_combo.clear()
    panel._moving_combo.addItems(layer_names)
    panel._fixed_combo.addItems(layer_names)
    panel._moving_combo.blockSignals(False)
    panel._fixed_combo.blockSignals(False)

    moving_index = panel._moving_combo.findText(moving_name)
    if moving_index >= 0:
        panel._moving_combo.setCurrentIndex(moving_index)

    fixed_index = panel._fixed_combo.findText(fixed_name)
    if fixed_index >= 0:
        panel._fixed_combo.setCurrentIndex(fixed_index)
    elif (
        panel._fixed_combo.count() > 1
        and panel._fixed_combo.currentText() == panel._moving_combo.currentText()
    ):
        panel._fixed_combo.setCurrentIndex(1)

    update_reference_time_bounds(panel)
    panel._sync_manual_transform_event_connections()
    refresh_transform_controls(panel)
    validate_registration_selection(panel)


def get_layer_by_name(panel: "RegistrationPanel", name: str) -> "Layer | None":
    """Return a viewer layer by name, if present.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose viewer should be searched.
    name : str
        Layer name to look up in the viewer.

    Returns
    -------
    napari.layers.Layer or None
        Matching layer when present, otherwise `None`.
    """
    try:
        return cast("Layer", panel.viewer.layers[name])
    except KeyError:
        return None


def selected_layer(panel: "RegistrationPanel", combo: QComboBox) -> "Layer | None":
    """Return the currently selected viewer layer for a combo box.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose viewer should be searched.
    combo : QComboBox
        Combo box containing layer names.

    Returns
    -------
    napari.layers.Layer or None
        Selected layer, or `None` when no valid selection exists.
    """
    name = combo.currentText()
    if not name:
        return None
    return get_layer_by_name(panel, name)


def current_scale_mode(panel: "RegistrationPanel") -> "ScaleMode":
    """Return the validated registration scale mode from the combo box.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose scale selector should be read.

    Returns
    -------
    {"off", "dB", "sqrt"}
        Selected registration scale mode.

    Raises
    ------
    ValueError
        If the combo box contains an unexpected value.
    """
    value = panel._scale_combo.currentData()
    if value in {"off", "dB", "sqrt"}:
        return value
    raise ValueError(f"Unknown registration scale mode: {value!r}.")


def current_metric(panel: "RegistrationPanel") -> "MetricName":
    """Return the validated registration metric from the combo box.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose metric selector should be read.

    Returns
    -------
    {"correlation", "mattes_mi"}
        Selected registration metric.

    Raises
    ------
    ValueError
        If the combo box contains an unexpected value.
    """
    value = panel._metric_combo.currentText()
    if value == "correlation":
        return "correlation"
    if value == "mattes_mi":
        return "mattes_mi"
    raise ValueError(f"Unknown registration metric: {value!r}.")


def current_resample_interpolation(
    panel: "RegistrationPanel",
) -> "ResampleInterpolation":
    """Return the validated resampling interpolation from the combo box.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose interpolation selector should be read.

    Returns
    -------
    {"linear", "bspline"}
        Selected resampling interpolation.

    Raises
    ------
    ValueError
        If the combo box contains an unexpected value.
    """
    value = panel._interpolation_combo.currentText()
    if value == "linear":
        return "linear"
    if value == "bspline":
        return "bspline"
    raise ValueError(f"Unknown resampling interpolation: {value!r}.")


def current_transform_model(
    panel: "RegistrationPanel",
) -> "VolumeTransformType | VolumewiseTransformType":
    """Return the validated transform model for the active mode.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose transform selector should be read.

    Returns
    -------
    {"translation", "rigid", "affine", "bspline"}
        Selected transform model, constrained by the active workflow.

    Raises
    ------
    ValueError
        If the combo box contains an unexpected value.
    """
    value = panel._transform_combo.currentText()
    if panel._operation() == "register_volume":
        if value == "translation":
            return "translation"
        if value == "rigid":
            return "rigid"
        if value == "affine":
            return "affine"
        if value == "bspline":
            return "bspline"
    else:
        if value == "translation":
            return "translation"
        if value == "rigid":
            return "rigid"
        if value == "affine":
            return "affine"
    raise ValueError(f"Unknown transform model: {value!r}.")


def update_reference_time_bounds(panel: "RegistrationPanel") -> None:
    """Clamp the volumewise reference-volume widget to the moving layer.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose reference-volume bounds should be updated.
    """
    moving_layer = selected_layer(panel, panel._moving_combo)
    if moving_layer is None:
        panel._reference_time_spin.setMaximum(0)
        panel._reference_time_spin.setValue(0)
        return

    data = _get_source_dataarray(moving_layer)
    if TIME_DIM not in data.dims:
        panel._reference_time_spin.setMaximum(0)
        panel._reference_time_spin.setValue(0)
        return

    panel._reference_time_spin.setMaximum(max(0, data.sizes[TIME_DIM] - 1))


def set_layer_validation_style(
    panel: "RegistrationPanel",
    *,
    moving_invalid: bool = False,
    fixed_invalid: bool = False,
    message: str | None = None,
) -> None:
    """Update inline validation state for the layer selectors.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose validation widgets should be updated.
    moving_invalid : bool, default: False
        Whether to mark the moving-layer selector as invalid.
    fixed_invalid : bool, default: False
        Whether to mark the fixed-layer selector as invalid.
    message : str, optional
        Validation message to show below the layer selectors.
    """
    error_style = "border: 1px solid #e05555;"
    normal_style = ""
    panel._moving_combo.setStyleSheet(error_style if moving_invalid else normal_style)
    panel._fixed_combo.setStyleSheet(error_style if fixed_invalid else normal_style)
    panel._moving_label.setStyleSheet("color: #e05555;" if moving_invalid else "")
    panel._fixed_label.setStyleSheet("color: #e05555;" if fixed_invalid else "")
    panel._reference_time_label.setStyleSheet("")
    if message:
        panel._layer_validation.setText(message)
        panel._layer_validation.show()
    else:
        panel._layer_validation.hide()
        panel._layer_validation.clear()


def set_run_btn_enabled(panel: "RegistrationPanel", enabled: bool) -> None:
    """Enable or disable the Run button without changing its busy text.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose Run button should be updated.
    enabled : bool
        Whether to enable the idle-state Run button.

    Notes
    -----
    The button is also disabled in `_begin_work` while a registration is
    running; this helper only handles the idle-state gating driven by
    layer-selection validation.
    """
    if panel._run_btn.text() == "Registering…":
        return
    panel._run_btn.setEnabled(enabled)


def validate_registration_selection(panel: "RegistrationPanel") -> bool:
    """Validate the current registration-layer selection and show inline feedback.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose selection state should be validated.

    Returns
    -------
    bool
        `True` when the selection is valid and a registration can be started,
        `False` otherwise. As a side effect, the Run button is enabled or disabled
        to match the validation result.
    """
    moving_layer = selected_layer(panel, panel._moving_combo)
    fixed_layer = selected_layer(panel, panel._fixed_combo)
    operation = panel._operation()

    if moving_layer is None:
        set_layer_validation_style(panel)
        set_run_btn_enabled(panel, False)
        return False

    try:
        moving = _get_source_dataarray(moving_layer)
    except Exception:
        set_layer_validation_style(
            panel,
            moving_invalid=True,
            message="Could not read the selected moving layer.",
        )
        set_run_btn_enabled(panel, False)
        return False

    if operation == "register_volumewise":
        if TIME_DIM not in moving.dims:
            set_layer_validation_style(
                panel,
                moving_invalid=True,
                message="Within-scan registration requires a layer with a time dimension.",
            )
            set_run_btn_enabled(panel, False)
            return False
        init_message = validate_initial_transform_selection(
            panel,
            operation=operation,
            moving=moving,
        )
        set_layer_validation_style(panel, message=init_message)
        set_run_btn_enabled(panel, init_message is None)
        return init_message is None

    moving_invalid = False
    fixed_invalid = False
    message: str | None = None

    if fixed_layer is None:
        set_layer_validation_style(
            panel,
            moving_invalid=moving_invalid,
            fixed_invalid=True,
            message="Between-scans registration requires different moving and fixed layers.",
        )
        set_run_btn_enabled(panel, False)
        return False

    try:
        fixed = _get_source_dataarray(fixed_layer)
    except Exception:
        set_layer_validation_style(
            panel,
            fixed_invalid=True,
            message="Could not read the selected fixed layer.",
        )
        set_run_btn_enabled(panel, False)
        return False

    if fixed_layer is moving_layer:
        moving_invalid = True
        fixed_invalid = True
        message = "Moving and fixed layers must be different."

    if message is None:
        message = validate_initial_transform_selection(
            panel,
            operation=operation,
            moving=_prepare_between_scan_data(moving),
            fixed=_prepare_between_scan_data(fixed),
        )

    valid = not (moving_invalid or fixed_invalid or message is not None)
    set_layer_validation_style(
        panel,
        moving_invalid=moving_invalid,
        fixed_invalid=fixed_invalid,
        message=message,
    )
    set_run_btn_enabled(panel, valid)
    return valid


def on_moving_layer_changed(panel: "RegistrationPanel", _name: str) -> None:
    """Update dependent widgets when the moving layer changes.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose dependent widgets should be refreshed.
    _name : str
        Unused emitted layer name from the combo-box signal.
    """
    del _name
    update_reference_time_bounds(panel)
    refresh_transform_controls(panel)
    validate_registration_selection(panel)
