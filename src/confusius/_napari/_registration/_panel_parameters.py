"""Registration-parameter helpers for the napari registration panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from confusius._napari._registration._panel import (
        ModeParameters,
        RegistrationPanel,
        RegistrationParameterMode,
    )


def get_default_registration_parameters(
    *, mode: RegistrationParameterMode
) -> ModeParameters:
    """Return the default parameter state for one registration mode.

    Parameters
    ----------
    mode : {"volume", "volumewise"}
        Registration workflow whose defaults should be returned.

    Returns
    -------
    ModeParameters
        Default parameter values for the requested workflow.
    """
    is_volumewise = mode == "volumewise"
    return {
        "transform": "rigid",
        "metric": "correlation",
        "scale": "dB",
        "initialization": "center_geometry",
        "learning_rate_auto": not is_volumewise,
        "learning_rate_value": 0.01 if is_volumewise else 1.0,
        "number_of_iterations": 100,
        "number_of_histogram_bins": 50,
        "mesh_size": (10, 10, 10),
        "convergence_minimum_value": 1e-6,
        "convergence_window_size": 10,
        "use_multi_resolution": False,
        "shrink_factors": "6, 2, 1",
        "smoothing_sigmas": "6, 2, 1",
        "resample_interpolation": "linear",
        "fill_value_auto": True,
        "fill_value": 0.0,
        "reference_time": 0,
        "n_jobs": -1,
        "sitk_threads": -1,
        "optimizer_weights_enabled": False,
        "optimizer_weights_values": [],
        "keep_diagnostics": False,
        "advanced_open": False,
    }


def get_registration_parameters(panel: RegistrationPanel) -> ModeParameters:
    """Return the current parameter state shown in the panel.

    Parameters
    ----------
    panel : RegistrationPanel
        Panel whose widgets should be read.

    Returns
    -------
    ModeParameters
        Current parameter values read from the visible widgets.
    """
    return {
        "transform": panel._transform_combo.currentText() or "rigid",
        "metric": panel._current_metric(),
        "scale": panel._current_scale_mode(),
        "initialization": panel._initialization_combo.currentData(),
        "learning_rate_auto": panel._learning_rate_auto_check.isChecked(),
        "learning_rate_value": panel._learning_rate_edit.value(),
        "number_of_iterations": panel._iterations_spin.value(),
        "number_of_histogram_bins": panel._histogram_bins_spin.value(),
        "mesh_size": (
            panel._mesh_size_z_spin.value(),
            panel._mesh_size_y_spin.value(),
            panel._mesh_size_x_spin.value(),
        ),
        "convergence_minimum_value": panel._convergence_min_edit.value(),
        "convergence_window_size": panel._convergence_window_spin.value(),
        "use_multi_resolution": panel._multi_resolution_check.isChecked(),
        "shrink_factors": panel._shrink_factors_edit.text(),
        "smoothing_sigmas": panel._smoothing_sigmas_edit.text(),
        "resample_interpolation": panel._current_resample_interpolation(),
        "fill_value_auto": panel._fill_value_auto_check.isChecked(),
        "fill_value": panel._fill_value_spin.value(),
        "reference_time": panel._reference_time_spin.value(),
        "n_jobs": panel._n_jobs_spin.value(),
        "sitk_threads": panel._sitk_threads_spin.value(),
        "optimizer_weights_enabled": panel._optimizer_weights_check.isChecked(),
        "optimizer_weights_values": panel._optimizer_weight_values(),
        "keep_diagnostics": panel._keep_diagnostics_check.isChecked(),
        "advanced_open": panel._advanced_toggle.isChecked(),
    }


def set_registration_parameters(
    panel: RegistrationPanel,
    params: ModeParameters,
    *,
    mode: RegistrationParameterMode,
) -> None:
    """Restore the parameter state for one registration mode.

    Parameters
    ----------
    panel : RegistrationPanel
        Panel whose widgets should be updated.
    params : ModeParameters
        Parameter values to push back into the widgets.
    mode : {"volume", "volumewise"}
        Registration workflow whose UI should be restored.
    """
    panel._transform_combo.blockSignals(True)
    panel._transform_combo.clear()
    is_volumewise = mode == "volumewise"
    if is_volumewise:
        panel._transform_combo.addItems(["translation", "rigid", "affine"])
    else:
        panel._transform_combo.addItems(["translation", "rigid", "affine", "bspline"])
    transform = params["transform"]
    transform_index = panel._transform_combo.findText(transform)
    if transform_index < 0:
        transform_index = panel._transform_combo.findText("rigid")
    if transform_index >= 0:
        panel._transform_combo.setCurrentIndex(transform_index)
    panel._transform_combo.blockSignals(False)

    panel._metric_combo.setCurrentText(params["metric"])
    scale_mode = params["scale"]
    scale_index = panel._scale_combo.findData(scale_mode)
    if scale_index >= 0:
        panel._scale_combo.setCurrentIndex(scale_index)
    initialization_data = params.get("initialization")
    for i in range(panel._initialization_combo.count()):
        if panel._initialization_combo.itemData(i) == initialization_data:
            panel._initialization_combo.setCurrentIndex(i)
            break
    panel._learning_rate_auto_check.setChecked(
        False if is_volumewise else params["learning_rate_auto"]
    )
    panel._learning_rate_edit.setValue(params["learning_rate_value"])
    panel._iterations_spin.setValue(params["number_of_iterations"])
    panel._histogram_bins_spin.setValue(params["number_of_histogram_bins"])
    mesh_size = params["mesh_size"]
    panel._mesh_size_z_spin.setValue(mesh_size[0])
    panel._mesh_size_y_spin.setValue(mesh_size[1])
    panel._mesh_size_x_spin.setValue(mesh_size[2])
    panel._convergence_min_edit.setValue(params["convergence_minimum_value"])
    panel._convergence_window_spin.setValue(params["convergence_window_size"])
    panel._multi_resolution_check.setChecked(params["use_multi_resolution"])
    panel._shrink_factors_edit.setText(params["shrink_factors"])
    panel._smoothing_sigmas_edit.setText(params["smoothing_sigmas"])
    panel._interpolation_combo.setCurrentText(params["resample_interpolation"])
    panel._fill_value_auto_check.setChecked(params["fill_value_auto"])
    panel._fill_value_spin.setValue(params["fill_value"])
    panel._reference_time_spin.setValue(params["reference_time"])
    panel._n_jobs_spin.setValue(params["n_jobs"])
    panel._sitk_threads_spin.setValue(params["sitk_threads"])
    panel._keep_diagnostics_check.setChecked(params["keep_diagnostics"])
    panel._advanced_toggle.setChecked(params["advanced_open"])
    panel._on_advanced_toggled(panel._advanced_toggle.isChecked())
    panel._update_metric_dependent_visibility(panel._metric_combo.currentText())
    panel._update_multi_resolution_enabled(panel._multi_resolution_check.isChecked())
    panel._update_transform_dependent_visibility(panel._transform_combo.currentText())
    panel._sync_optimizer_weight_editor(
        values=params.get("optimizer_weights_values"),
        enabled=params.get("optimizer_weights_enabled", False),
    )
