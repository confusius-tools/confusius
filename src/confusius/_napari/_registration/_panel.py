"""Registration panel for the ConfUSIus napari plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr
from napari.layers.utils.layer_utils import calc_data_range
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from confusius._dims import SPATIAL_DIMS, TIME_DIM
from confusius.plotting.napari import plot_napari
from confusius.registration import register_volume, register_volumewise

if TYPE_CHECKING:
    import napari
    import numpy.typing as npt
    from napari.layers import Layer

    from confusius.registration import RegistrationDiagnostics


def _default_dims_for_ndim(ndim: int) -> tuple[str, ...]:
    """Return fallback dimension names for a raw napari layer.

    Parameters
    ----------
    ndim : int
        Number of array dimensions.

    Returns
    -------
    tuple of str
        Default dimension names compatible with ConfUSIus conventions when
        possible.
    """
    defaults: dict[int, tuple[str, ...]] = {
        1: SPATIAL_DIMS[-1:],
        2: SPATIAL_DIMS[-2:],
        3: SPATIAL_DIMS,
        4: (TIME_DIM, *SPATIAL_DIMS),
    }
    return defaults.get(ndim, tuple(f"dim{i}" for i in range(ndim)))


def _normalize_layer_sequence(values: Any, ndim: int, fill: Any) -> list[Any]:
    """Return a layer property as a list with length `ndim`.

    Parameters
    ----------
    values : Any
        Layer property such as `scale`, `translate`, `units`, or
        `axis_labels`.
    ndim : int
        Number of dimensions expected on the layer data.
    fill : Any
        Value used to pad missing entries.

    Returns
    -------
    list of Any
        Normalized sequence with exactly `ndim` elements.
    """
    if values is None:
        return [fill] * ndim
    seq = list(values)
    if len(seq) < ndim:
        return ([fill] * (ndim - len(seq))) + seq
    if len(seq) > ndim:
        return seq[-ndim:]
    return seq


def _layer_to_dataarray(layer: "Layer") -> xr.DataArray:
    """Return an `xarray.DataArray` view of a napari layer.

    Parameters
    ----------
    layer : napari.layers.Layer
        Napari layer to convert.

    Returns
    -------
    xarray.DataArray
        Original ConfUSIus DataArray when present in `layer.metadata`,
        otherwise a reconstructed DataArray derived from the layer state.
    """
    existing = layer.metadata.get("xarray")
    if existing is not None:
        return cast("xr.DataArray", existing)

    data = np.asarray(layer.data)
    ndim = data.ndim

    raw_labels = _normalize_layer_sequence(
        getattr(layer, "axis_labels", None), ndim, None
    )
    axis_labels = tuple(
        str(label) if label not in (None, "") else default
        for label, default in zip(
            raw_labels, _default_dims_for_ndim(ndim), strict=False
        )
    )

    scale = [
        float(v)
        for v in _normalize_layer_sequence(getattr(layer, "scale", None), ndim, 1.0)
    ]
    translate = [
        float(v)
        for v in _normalize_layer_sequence(getattr(layer, "translate", None), ndim, 0.0)
    ]
    raw_units = _normalize_layer_sequence(getattr(layer, "units", None), ndim, None)
    units = [None if u is None or str(u) == "pixel" else str(u) for u in raw_units]

    coords: dict[str, xr.DataArray] = {}
    for dim, n, spacing, origin, unit in zip(
        axis_labels, data.shape, scale, translate, units, strict=False
    ):
        attrs: dict[str, Any] = {"voxdim": abs(spacing)}
        if unit is not None:
            attrs["units"] = unit
        coords[dim] = xr.DataArray(
            origin + np.arange(n) * spacing, dims=[dim], attrs=attrs
        )

    return xr.DataArray(data, dims=axis_labels, coords=coords)


def _image_display_kwargs_from_layer(layer: "Layer") -> dict[str, Any]:
    """Return image-display kwargs copied from an existing napari layer.

    Parameters
    ----------
    layer : napari.layers.Layer
        Source layer whose visual settings should be reused when possible.

    Returns
    -------
    dict[str, Any]
        Keyword arguments suitable for [`plot_napari`][confusius.plotting.plot_napari].
    """
    kwargs: dict[str, Any] = {}
    for attr in ("colormap", "gamma", "opacity"):
        if hasattr(layer, attr):
            kwargs[attr] = getattr(layer, attr)
    return kwargs


def _run_register_volume(
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    transform_type: Literal["translation", "rigid", "affine", "bspline"],
    metric: Literal["correlation", "mattes_mi"],
    learning_rate: float | Literal["auto"],
    number_of_iterations: int,
    use_multi_resolution: bool,
    resample_interpolation: Literal["linear", "bspline"],
) -> tuple[
    xr.DataArray, npt.NDArray[np.floating] | xr.DataArray, RegistrationDiagnostics
]:
    """Run `register_volume` with GUI-friendly defaults.

    Parameters
    ----------
    moving : xarray.DataArray
        Moving volume.
    fixed : xarray.DataArray
        Fixed reference volume.
    transform_type : {"translation", "rigid", "affine", "bspline"}
        Registration model.
    metric : {"correlation", "mattes_mi"}
        Similarity metric.
    learning_rate : float or {"auto"}
        Optimizer learning rate.
    number_of_iterations : int
        Maximum number of optimizer iterations.
    use_multi_resolution : bool
        Whether to enable the registration pyramid.
    resample_interpolation : {"linear", "bspline"}
        Interpolator for the resampled output.

    Returns
    -------
    registered : xarray.DataArray
        Resampled registered volume.
    transform : numpy.ndarray or xarray.DataArray
        Estimated transform.
    diagnostics : confusius.registration.RegistrationDiagnostics
        Optimizer diagnostics.
    """
    return register_volume(
        moving,
        fixed,
        transform_type=transform_type,
        metric=metric,
        learning_rate=learning_rate,
        number_of_iterations=number_of_iterations,
        use_multi_resolution=use_multi_resolution,
        resample=True,
        resample_interpolation=resample_interpolation,
        show_progress=False,
    )


def _run_register_volumewise(
    data: xr.DataArray,
    *,
    reference_time: int,
    n_jobs: int,
    transform: Literal["translation", "rigid", "affine"],
    metric: Literal["correlation", "mattes_mi"],
    learning_rate: float | Literal["auto"],
    number_of_iterations: int,
    use_multi_resolution: bool,
    resample_interpolation: Literal["linear", "bspline"],
) -> xr.DataArray:
    """Run `register_volumewise` with GUI-friendly defaults.

    Parameters
    ----------
    data : xarray.DataArray
        Time-series data to motion-correct.
    reference_time : int
        Reference frame index.
    n_jobs : int
        Number of joblib workers to use.
    transform : {"translation", "rigid", "affine"}
        Registration model.
    metric : {"correlation", "mattes_mi"}
        Similarity metric.
    learning_rate : float or {"auto"}
        Optimizer learning rate.
    number_of_iterations : int
        Maximum number of optimizer iterations per frame.
    use_multi_resolution : bool
        Whether to enable the registration pyramid.
    resample_interpolation : {"linear", "bspline"}
        Interpolator for the resampled output.

    Returns
    -------
    xarray.DataArray
        Registered time series.
    """
    return register_volumewise(
        data,
        reference_time=reference_time,
        n_jobs=n_jobs,
        transform=transform,
        metric=metric,
        learning_rate=learning_rate,
        number_of_iterations=number_of_iterations,
        use_multi_resolution=use_multi_resolution,
        resample_interpolation=resample_interpolation,
        show_progress=False,
    )


class RegistrationPanel(QWidget):
    """Right-side panel for running registration from napari.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._worker = None
        self._setup_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        operation_group = QGroupBox("Registration")
        operation_layout = QFormLayout(operation_group)
        operation_layout.setSpacing(6)
        operation_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        operation_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self._mode_group = QButtonGroup(self)
        mode_row = QHBoxLayout()
        self._single_volume_radio = QRadioButton("Between scans")
        self._time_series_radio = QRadioButton("Within scan")
        self._single_volume_radio.setChecked(True)
        self._mode_group.addButton(self._single_volume_radio)
        self._mode_group.addButton(self._time_series_radio)
        mode_row.addWidget(self._single_volume_radio)
        mode_row.addWidget(self._time_series_radio)
        operation_layout.addRow("Mode", mode_row)

        self._moving_combo = QComboBox()
        self._moving_combo.setMinimumContentsLength(18)
        self._moving_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._moving_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._moving_combo.currentTextChanged.connect(self._on_moving_layer_changed)
        operation_layout.addRow("Moving layer", self._moving_combo)

        self._fixed_label = QLabel("Fixed layer")
        self._fixed_combo = QComboBox()
        self._fixed_combo.setMinimumContentsLength(18)
        self._fixed_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._fixed_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        operation_layout.addRow(self._fixed_label, self._fixed_combo)

        self._reference_time_label = QLabel("Ref. time")
        self._reference_time_spin = QSpinBox()
        self._reference_time_spin.setMinimum(0)
        operation_layout.addRow(self._reference_time_label, self._reference_time_spin)

        self._n_jobs_label = QLabel("Jobs")
        self._n_jobs_spin = QSpinBox()
        self._n_jobs_spin.setRange(-128, 128)
        self._n_jobs_spin.setSpecialValueText("auto")
        self._n_jobs_spin.setValue(-1)
        self._n_jobs_spin.setToolTip(
            "Number of workers for time-series registration. -1 uses all CPUs."
        )
        operation_layout.addRow(self._n_jobs_label, self._n_jobs_spin)

        layout.addWidget(operation_group)

        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(6)
        params_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        params_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self._transform_combo = QComboBox()
        self._transform_combo.setMinimumContentsLength(14)
        self._transform_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._transform_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        params_layout.addRow("Transform", self._transform_combo)

        self._metric_combo = QComboBox()
        self._metric_combo.setMinimumContentsLength(14)
        self._metric_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._metric_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._metric_combo.addItems(["correlation", "mattes_mi"])
        params_layout.addRow("Metric", self._metric_combo)

        self._interpolation_combo = QComboBox()
        self._interpolation_combo.setMinimumContentsLength(14)
        self._interpolation_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._interpolation_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._interpolation_combo.addItems(["linear", "bspline"])
        params_layout.addRow("Interpolation", self._interpolation_combo)
        self._interpolation_combo.setToolTip(
            "Interpolator used for the resampled output."
        )

        learning_rate_row = QHBoxLayout()
        self._learning_rate_auto_check = QCheckBox("Auto")
        self._learning_rate_auto_check.setChecked(True)
        self._learning_rate_spin = QDoubleSpinBox()
        self._learning_rate_spin.setRange(1e-6, 1e3)
        self._learning_rate_spin.setDecimals(4)
        self._learning_rate_spin.setSingleStep(0.01)
        self._learning_rate_spin.setValue(0.1)
        self._learning_rate_spin.setEnabled(False)
        learning_rate_row.addWidget(self._learning_rate_auto_check)
        learning_rate_row.addWidget(self._learning_rate_spin, stretch=1)
        params_layout.addRow("Learning rate", learning_rate_row)

        self._iterations_spin = QSpinBox()
        self._iterations_spin.setRange(1, 100_000)
        self._iterations_spin.setValue(100)
        params_layout.addRow("Iterations", self._iterations_spin)

        self._multi_resolution_check = QCheckBox("Use multi-resolution")
        self._multi_resolution_check.setToolTip(
            "Run registration from coarse to fine resolution levels."
        )
        self._multi_resolution_check.setChecked(False)
        params_layout.addRow(self._multi_resolution_check)

        layout.addWidget(params_group)

        self._run_btn = QPushButton("Run registration")
        self._run_btn.setObjectName("primary_btn")
        self._run_btn.clicked.connect(self._run_registration)
        layout.addWidget(self._run_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setObjectName("status_err")
        self._status.hide()
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumHeight(4)
        self._progress.hide()
        layout.addWidget(self._progress)

        layout.addStretch()

        self._single_volume_radio.toggled.connect(self._on_mode_changed)
        self._time_series_radio.toggled.connect(self._on_mode_changed)
        self._learning_rate_auto_check.toggled.connect(
            self._learning_rate_spin.setDisabled
        )

        self._refresh_layers()
        self._on_mode_changed()

    def _refresh_layers(self) -> None:
        """Repopulate the layer selectors from the viewer."""
        moving_name = self._moving_combo.currentText()
        fixed_name = self._fixed_combo.currentText()

        layer_names = [layer.name for layer in self.viewer.layers]

        self._moving_combo.blockSignals(True)
        self._fixed_combo.blockSignals(True)
        self._moving_combo.clear()
        self._fixed_combo.clear()
        self._moving_combo.addItems(layer_names)
        self._fixed_combo.addItems(layer_names)
        self._moving_combo.blockSignals(False)
        self._fixed_combo.blockSignals(False)

        moving_index = self._moving_combo.findText(moving_name)
        if moving_index >= 0:
            self._moving_combo.setCurrentIndex(moving_index)

        fixed_index = self._fixed_combo.findText(fixed_name)
        if fixed_index >= 0:
            self._fixed_combo.setCurrentIndex(fixed_index)
        elif (
            self._fixed_combo.count() > 1
            and self._fixed_combo.currentText() == self._moving_combo.currentText()
        ):
            self._fixed_combo.setCurrentIndex(1)

        self._update_reference_time_bounds()

    def _selected_layer(self, combo: QComboBox) -> Layer | None:
        """Return the currently selected viewer layer for a combo box.

        Parameters
        ----------
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
        try:
            return cast("Layer", self.viewer.layers[name])
        except KeyError:
            return None

    def _update_reference_time_bounds(self) -> None:
        """Clamp the volumewise reference-time widget to the moving layer."""
        moving_layer = self._selected_layer(self._moving_combo)
        if moving_layer is None:
            self._reference_time_spin.setMaximum(0)
            self._reference_time_spin.setValue(0)
            return

        data = _layer_to_dataarray(moving_layer)
        if TIME_DIM not in data.dims:
            self._reference_time_spin.setMaximum(0)
            self._reference_time_spin.setValue(0)
            return

        self._reference_time_spin.setMaximum(max(0, data.sizes[TIME_DIM] - 1))

    def _on_moving_layer_changed(self, _name: str) -> None:
        """Update dependent widgets when the moving layer changes."""
        self._update_reference_time_bounds()

    def _operation(self) -> Literal["register_volume", "register_volumewise"]:
        """Return the currently selected registration workflow."""
        if self._time_series_radio.isChecked():
            return "register_volumewise"
        return "register_volume"

    def _on_mode_changed(self) -> None:
        """Update the panel when the registration mode changes."""
        is_volumewise = self._operation() == "register_volumewise"

        self._fixed_label.setVisible(not is_volumewise)
        self._fixed_combo.setVisible(not is_volumewise)
        self._fixed_combo.setEnabled(not is_volumewise)
        self._reference_time_label.setVisible(is_volumewise)
        self._reference_time_spin.setVisible(is_volumewise)
        self._n_jobs_label.setVisible(is_volumewise)
        self._n_jobs_spin.setVisible(is_volumewise)

        self._transform_combo.clear()
        if is_volumewise:
            self._transform_combo.addItems(["translation", "rigid", "affine"])
        else:
            self._transform_combo.addItems(
                ["translation", "rigid", "affine", "bspline"]
            )
        rigid_index = self._transform_combo.findText("rigid")
        if rigid_index >= 0:
            self._transform_combo.setCurrentIndex(rigid_index)

        self._update_reference_time_bounds()

    def _begin_work(self) -> None:
        """Put the panel into its busy state."""
        self._run_btn.setEnabled(False)
        self._run_btn.setText("Registering…")
        self._status.hide()
        self._progress.show()
        QApplication.processEvents()

    def _end_work(self) -> None:
        """Restore the idle UI state after background work."""
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run registration")
        self._progress.hide()

    def _set_error(self, message: str) -> None:
        """Show a validation or execution error in the panel.

        Parameters
        ----------
        message : str
            Error message to display.
        """
        self._status.setText(message)
        self._status.show()

    def _run_registration(self) -> None:
        """Validate inputs and start the selected registration workflow."""
        operation = self._operation()
        moving_layer = self._selected_layer(self._moving_combo)
        fixed_layer = self._selected_layer(self._fixed_combo)

        if moving_layer is None:
            self._set_error("Select a moving layer.")
            return

        try:
            learning_rate: float | Literal["auto"]
            if self._learning_rate_auto_check.isChecked():
                learning_rate = "auto"
            else:
                learning_rate = float(self._learning_rate_spin.value())
            moving = _layer_to_dataarray(moving_layer)
        except Exception as exc:  # noqa: BLE001
            self._set_error(str(exc))
            return

        payload: dict[str, Any] = {
            "operation": operation,
            "moving_layer_name": moving_layer.name,
            "transform": self._transform_combo.currentText(),
            "metric": self._metric_combo.currentText(),
            "learning_rate": learning_rate,
            "number_of_iterations": self._iterations_spin.value(),
            "use_multi_resolution": self._multi_resolution_check.isChecked(),
            "resample_interpolation": self._interpolation_combo.currentText(),
        }

        if operation == "register_volume":
            if fixed_layer is None:
                self._set_error("Select a fixed layer.")
                return
            if fixed_layer is moving_layer:
                self._set_error("Moving and fixed layers must be different.")
                return

            try:
                fixed = _layer_to_dataarray(fixed_layer)
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))
                return

            if TIME_DIM in moving.dims or TIME_DIM in fixed.dims:
                self._set_error("register_volume requires spatial-only layers.")
                return

            payload["fixed_layer_name"] = fixed_layer.name

            worker = thread_worker(_run_register_volume)(
                moving,
                fixed,
                transform_type=cast(
                    "Literal['translation', 'rigid', 'affine', 'bspline']",
                    payload["transform"],
                ),
                metric=cast("Literal['correlation', 'mattes_mi']", payload["metric"]),
                learning_rate=learning_rate,
                number_of_iterations=payload["number_of_iterations"],
                use_multi_resolution=payload["use_multi_resolution"],
                resample_interpolation=cast(
                    "Literal['linear', 'bspline']", payload["resample_interpolation"]
                ),
            )
        else:
            if TIME_DIM not in moving.dims:
                self._set_error(
                    "register_volumewise requires a layer with a time dimension."
                )
                return

            payload["reference_time"] = self._reference_time_spin.value()
            payload["n_jobs"] = self._n_jobs_spin.value()

            worker = thread_worker(_run_register_volumewise)(
                moving,
                reference_time=payload["reference_time"],
                n_jobs=payload["n_jobs"],
                transform=cast(
                    "Literal['translation', 'rigid', 'affine']", payload["transform"]
                ),
                metric=cast("Literal['correlation', 'mattes_mi']", payload["metric"]),
                learning_rate=learning_rate,
                number_of_iterations=payload["number_of_iterations"],
                use_multi_resolution=payload["use_multi_resolution"],
                resample_interpolation=cast(
                    "Literal['linear', 'bspline']", payload["resample_interpolation"]
                ),
            )

        self._worker = worker
        self._begin_work()
        worker.returned.connect(
            lambda result: self._on_registration_finished(payload, result)
        )
        worker.errored.connect(self._on_registration_failed)
        worker.finished.connect(self._end_work)
        worker.start()

    def _on_registration_finished(self, payload: dict[str, Any], result: Any) -> None:
        """Add a successful registration result back to the viewer.

        Parameters
        ----------
        payload : dict[str, Any]
            UI parameter snapshot captured before the worker started.
        result : Any
            Worker return value.
        """
        operation = cast(str, payload["operation"])

        if operation == "register_volume":
            registered, transform, diagnostics = cast(
                "tuple[xr.DataArray, npt.NDArray[np.floating] | xr.DataArray, RegistrationDiagnostics]",
                result,
            )
            registered = registered.copy(deep=False)
            registered.attrs = registered.attrs.copy()
            registered.attrs["registration_transform"] = transform
            registered.attrs["registration_diagnostics"] = diagnostics
            registered.attrs["registration_operation"] = operation
            layer_name = (
                f"{payload['moving_layer_name']} → {payload['fixed_layer_name']}"
            )
            metadata: dict[str, Any] = {
                "registration_transform": transform,
                "registration_diagnostics": diagnostics,
            }
        else:
            registered = cast("xr.DataArray", result).copy(deep=False)
            registered.attrs = registered.attrs.copy()
            registered.attrs["registration_operation"] = operation
            layer_name = f"{payload['moving_layer_name']} registered"
            metadata = {
                "motion_params": registered.attrs.get("motion_params"),
                "reference_time": payload["reference_time"],
            }

        metadata["registration_operation"] = operation
        metadata["registration_parameters"] = payload.copy()

        source_layer_name = cast(str, payload["moving_layer_name"])
        try:
            source_layer = self.viewer.layers[source_layer_name]
        except KeyError:
            display_kwargs: dict[str, Any] = {}
        else:
            display_kwargs = _image_display_kwargs_from_layer(source_layer)
        display_kwargs["contrast_limits"] = calc_data_range(registered.data)

        _, layer = plot_napari(
            registered,
            viewer=self.viewer,
            name=layer_name,
            show_colorbar=False,
            **display_kwargs,
        )
        layer.metadata.update(metadata)
        layer.metadata["xarray"] = registered
        self.viewer.layers.selection.active = layer
        show_info(f"Added registered layer: {layer.name}")

    def _on_registration_failed(self, exc: BaseException) -> None:
        """Handle a failed worker execution.

        Parameters
        ----------
        exc : BaseException
            Exception raised by the worker.
        """
        self._set_error(str(exc))
        show_error(str(exc))
