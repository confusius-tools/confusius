"""Registration panel for the ConfUSIus napari plugin."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

import numpy as np
import xarray as xr
from napari.layers.utils.layer_utils import calc_data_range
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import QRegularExpression, Qt, QTimer
from qtpy.QtGui import QValidator
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from confusius._dims import SPATIAL_DIMS, TIME_DIM
from confusius._napari._registration._metric_plotter import (
    RegistrationMetricPlotter,
)
from confusius._napari._registration._progress import (
    NapariProgressBridge,
    NapariVolumewiseProgress,
    NapariVolumewiseProgressBridge,
    make_napari_progress_factory,
)
from confusius._napari._registration._transforms import (
    AffineTransformPayload,
    affine_transform_from_payload,
    load_affine_transform_payload,
    make_affine_transform_payload,
    output_grid_from_payload,
    save_affine_transform_payload,
)
from confusius.plotting.napari import plot_napari
from confusius.registration import (
    register_volume,
    register_volumewise,
    resample_like,
    resample_volume,
)

if TYPE_CHECKING:
    import napari
    import numpy.typing as npt
    from napari.layers import Image, Layer

    from confusius.registration import RegistrationDiagnostics, RegistrationProgress


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


def _prepare_between_scan_data(data: xr.DataArray) -> xr.DataArray:
    """Return a spatial-only DataArray for between-scan registration.

    Parameters
    ----------
    data : xarray.DataArray
        Input layer data.

    Returns
    -------
    xarray.DataArray
        Spatial-only data. If the input has a time dimension, it is averaged
        over time with attributes preserved.
    """
    if TIME_DIM not in data.dims:
        return data
    averaged = data.mean(dim=TIME_DIM, keep_attrs=True)
    averaged.attrs = data.attrs.copy()
    return averaged


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


def _parse_sequence(text: str, expected_len: int = 3) -> tuple[int, ...]:
    """Parse comma-separated integers from a text field."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return tuple()
    try:
        values = tuple(int(float(p)) for p in parts)
    except ValueError:
        return tuple()
    if len(values) != expected_len:
        return tuple()
    return values


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """`QDoubleSpinBox` variant that accepts scientific notation.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    _ACCEPTABLE_RE = QRegularExpression(
        r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
    )
    _INTERMEDIATE_RE = QRegularExpression(
        r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))?(?:[eE][+-]?\d*)?$"
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDecimals(10)
        self.setKeyboardTracking(False)
        self.setAccelerated(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def validate(  # ty: ignore[invalid-method-override]
        self, text: str | None, pos: int
    ) -> tuple[QValidator.State, str, int]:
        """Validate decimals and scientific notation while the user types.

        Parameters
        ----------
        text : str, optional
            Current text being edited.
        pos : int
            Cursor position.

        Returns
        -------
        state : QValidator.State
            Validation state.
        text : str
            Normalized text.
        pos : int
            Cursor position.
        """
        normalized = text or ""
        if normalized in {"", "+", "-", ".", "+.", "-."}:
            return (QValidator.State.Intermediate, normalized, pos)
        if self._ACCEPTABLE_RE.match(normalized).hasMatch():
            return (QValidator.State.Acceptable, normalized, pos)
        if self._INTERMEDIATE_RE.match(normalized).hasMatch():
            return (QValidator.State.Intermediate, normalized, pos)
        return (QValidator.State.Invalid, normalized, pos)

    def valueFromText(self, text):
        """Parse the current text into a float value.

        Parameters
        ----------
        text : str, optional
            Text to parse.

        Returns
        -------
        float
            Parsed numeric value.
        """
        return float(text or 0.0)

    def textFromValue(self, value: float) -> str:  # ty: ignore[invalid-method-override]
        """Format values compactly, using scientific notation when helpful.

        Parameters
        ----------
        value : float
            Value to format.

        Returns
        -------
        str
            Formatted text.
        """
        return f"{value:.12g}"

    def stepBy(self, steps: int) -> None:
        """Apply additive stepping using the configured single-step size.

        Parameters
        ----------
        steps : int
            Number of steps to apply.
        """
        self.setValue(self.value() + (steps * self.singleStep()))


def _run_register_volume_registration_volume(
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    transform_type: Literal["translation", "rigid", "affine", "bspline"],
    metric: Literal["correlation", "mattes_mi"],
    learning_rate: float | Literal["auto"],
    number_of_iterations: int,
    use_multi_resolution: bool,
    resample_interpolation: Literal["linear", "bspline"],
    mesh_size: tuple[int, int, int] = (10, 10, 10),
    number_of_histogram_bins: int = 50,
    convergence_minimum_value: float = 1e-6,
    convergence_window_size: int = 10,
    initialization: Literal["center_geometry", "center_moments"]
    | None = "center_geometry",
    initial_transform: npt.NDArray[np.floating] | None = None,
    shrink_factors: Sequence[int] = (6, 2, 1),
    smoothing_sigmas: Sequence[int] = (6, 2, 1),
    fill_value: float | None = None,
    progress_plotter: Callable[..., RegistrationProgress] | None = None,
    abort_event: Event | None = None,
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
    mesh_size : tuple of int, default: (10, 10, 10)
        B-spline mesh size.
    number_of_histogram_bins : int
        Histogram bins for Mattes MI metric.
    convergence_minimum_value : float
        Convergence threshold.
    convergence_window_size : int
        Window size for convergence estimation.
    initialization : {"center_geometry", "center_moments"} or None
        Transform initializer.
    initial_transform : numpy.ndarray, optional
        Pre-computed affine transform used as a warm start before optimization.
    shrink_factors : sequence of int
        Shrink factors per resolution level.
    smoothing_sigmas : sequence of int
        Smoothing sigmas per resolution level.
    fill_value : float or None
        Fill value for resampled output outside input domain.
    progress_plotter : callable, optional
        Optional progress-plotter factory forwarded to `register_volume`.
    abort_event : threading.Event, optional
        Cooperative cancellation flag forwarded to `register_volume`.

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
        mesh_size=mesh_size,
        number_of_histogram_bins=number_of_histogram_bins,
        convergence_minimum_value=convergence_minimum_value,
        convergence_window_size=convergence_window_size,
        initialization=initialization
        if initial_transform is None
        else initial_transform,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        fill_value=fill_value,
        show_progress=progress_plotter is not None,
        progress_plotter=progress_plotter,
        abort_event=abort_event,
    )


def _affine_payload_from_layer(layer: "Layer") -> AffineTransformPayload | None:
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
    affine_transform_from_payload(payload)
    return cast("AffineTransformPayload", payload)


def _run_register_volumewise(
    data: xr.DataArray,
    *,
    reference_time: int,
    n_jobs: int,
    transform: Literal["translation", "rigid", "affine"],
    metric: Literal["correlation", "mattes_mi"],
    learning_rate: float | Literal["auto"] = 0.01,
    number_of_iterations: int = 100,
    use_multi_resolution: bool,
    resample_interpolation: Literal["linear", "bspline"],
    number_of_histogram_bins: int = 50,
    convergence_minimum_value: float = 1e-6,
    convergence_window_size: int = 10,
    initialization: Literal["center_geometry", "center_moments"]
    | None = "center_geometry",
    shrink_factors: Sequence[int] = (6, 2, 1),
    smoothing_sigmas: Sequence[int] = (6, 2, 1),
    keep_diagnostics: bool = False,
    abort_event: Event | None = None,
    progress_reporter: NapariVolumewiseProgress | None = None,
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
    learning_rate : float or {"auto"}, default: 0.01
        Optimizer learning rate.
    number_of_iterations : int
        Maximum number of optimizer iterations per frame.
    use_multi_resolution : bool
        Whether to enable the registration pyramid.
    resample_interpolation : {"linear", "bspline"}
        Interpolator for the resampled output.
    number_of_histogram_bins : int
        Histogram bins for Mattes MI metric.
    convergence_minimum_value : float
        Convergence threshold.
    convergence_window_size : int
        Window size for convergence estimation.
    initialization : {"center_geometry", "center_moments"} or None
        Transform initializer.
    shrink_factors : tuple of int or None
        Shrink factors per resolution level.
    smoothing_sigmas : tuple of int or None
        Smoothing sigmas per resolution level.
    keep_diagnostics : bool
        Store detailed optimization diagnostics.
    abort_event : threading.Event, optional
        Cooperative cancellation flag forwarded to `register_volumewise`.
    progress_reporter : NapariVolumewiseProgress, optional
        GUI-thread bridge-backed reporter forwarded to `register_volumewise`.

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
        number_of_histogram_bins=number_of_histogram_bins,
        convergence_minimum_value=convergence_minimum_value,
        convergence_window_size=convergence_window_size,
        initialization=initialization,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        keep_diagnostics=keep_diagnostics,
        show_progress=False,
        abort_event=abort_event,
        progress_reporter=progress_reporter,
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
        self._abort_event: Event | None = None
        self._loaded_transform_payload: AffineTransformPayload | None = None
        # Per-run progress state. Set on the GUI thread before the worker starts.
        self._progress_bridge: NapariProgressBridge | None = None
        self._progress_layer: Image | None = None
        self._progress_fixed_layer: Image | None = None
        self._progress_moving_layer: Image | None = None
        self._volumewise_progress_bridge: NapariVolumewiseProgressBridge | None = None
        self._volumewise_progress_layer: Image | None = None
        self._volumewise_moving_preview_layer: Image | None = None
        self._volumewise_progress_time_axis: int | None = None
        self._volumewise_progress_total: int | None = None
        # Bottom-dock metric curve. Created lazily on the first run, reused
        # across subsequent runs, and torn down with the progress state.
        self._metric_plotter: RegistrationMetricPlotter | None = None
        self._metric_dock: QDockWidget | None = None
        self._active_mode: Literal["register_volume", "register_volumewise"] = (
            "register_volume"
        )
        self._mode_parameters: dict[str, dict[str, Any]] = {}
        self._setup_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)

    def _make_form_label(self, text: str, *, tooltip: str | None = None) -> QLabel:
        """Return a form label with an optional tooltip."""
        label = QLabel(text)
        if tooltip is not None:
            label.setToolTip(tooltip)
        return label

    def _make_advanced_row(
        self,
        layout: QFormLayout,
        label: str,
        widget: QWidget,
        *,
        tooltip: str | None = None,
    ) -> QWidget:
        """Create a row container for advanced parameters that can be shown/hidden together."""
        container = QWidget()
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        lbl = self._make_form_label(label, tooltip=tooltip)
        lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row_layout.addWidget(lbl)
        row_layout.addWidget(widget, stretch=1)
        layout.addRow(container)
        return container

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._panel_group = QButtonGroup(self)
        panel_row = QHBoxLayout()
        panel_row.setSpacing(0)
        self._register_panel_radio = QPushButton("Register")
        self._register_panel_radio.setCheckable(True)
        self._transforms_panel_radio = QPushButton("Transforms")
        self._transforms_panel_radio.setCheckable(True)
        self._register_panel_radio.setChecked(True)
        self._panel_group.addButton(self._register_panel_radio)
        self._panel_group.addButton(self._transforms_panel_radio)
        self._register_panel_radio.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._transforms_panel_radio.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        _segment_btn_style = """
        QPushButton {
            border-radius: 0;
        }
        QPushButton:checked {
            background: #e94b5f;
            color: white;
            font-weight: bold;
        }
        """
        self._register_panel_radio.setStyleSheet(
            _segment_btn_style
            + "border-top-right-radius: 0; border-bottom-right-radius: 0;"
        )
        self._transforms_panel_radio.setStyleSheet(
            _segment_btn_style
            + "border-top-left-radius: 0; border-bottom-left-radius: 0;"
        )
        panel_row.addWidget(self._register_panel_radio)
        panel_row.addWidget(self._transforms_panel_radio)
        layout.addLayout(panel_row)

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
        self._time_series_radio = QRadioButton("Within-scan")
        self._single_volume_radio.setChecked(True)
        self._mode_group.addButton(self._single_volume_radio)
        self._mode_group.addButton(self._time_series_radio)
        mode_row.addWidget(self._single_volume_radio)
        mode_row.addWidget(self._time_series_radio)
        operation_layout.addRow(
            self._make_form_label(
                "Mode",
                tooltip="Registration workflow. Use 'Between scans' for moving/fixed registration and 'Within-scan' for frame-to-reference motion correction.",
            ),
            mode_row,
        )

        self._moving_label = QLabel("Moving layer")
        self._moving_combo = QComboBox()
        self._moving_combo.setMinimumContentsLength(18)
        self._moving_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._moving_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._moving_combo.currentTextChanged.connect(self._on_moving_layer_changed)
        self._moving_label.setToolTip(
            "Layer containing the moving image or time series to register."
        )
        operation_layout.addRow(self._moving_label, self._moving_combo)

        self._fixed_label = QLabel("Fixed layer")
        self._fixed_combo = QComboBox()
        self._fixed_combo.setMinimumContentsLength(18)
        self._fixed_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._fixed_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._fixed_label.setToolTip(
            "Reference layer that defines the registration target grid."
        )
        operation_layout.addRow(self._fixed_label, self._fixed_combo)

        self._reference_time_label = QLabel("Ref. time")
        self._reference_time_spin = QSpinBox()
        self._reference_time_spin.setMinimum(0)
        self._reference_time_label.setToolTip(
            "Time index used as the registration target for within-scan motion correction."
        )
        operation_layout.addRow(self._reference_time_label, self._reference_time_spin)

        self._n_jobs_spin = QSpinBox()
        self._n_jobs_spin.setRange(-128, 128)
        self._n_jobs_spin.setSpecialValueText("auto")
        self._n_jobs_spin.setValue(-1)
        self._n_jobs_spin.setToolTip(
            "Number of workers for time-series registration. -1 uses all CPUs."
        )

        self._layer_validation = QLabel("")
        self._layer_validation.setWordWrap(True)
        self._layer_validation.setObjectName("status_err")
        self._layer_validation.hide()
        operation_layout.addRow(self._layer_validation)

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
        params_layout.addRow(
            self._make_form_label(
                "Transform",
                tooltip="Transform model optimized during registration: translation, rigid, affine, or bspline for between-scan registration.",
            ),
            self._transform_combo,
        )

        self._mesh_size_z_spin = QSpinBox()
        self._mesh_size_z_spin.setRange(1, 512)
        self._mesh_size_z_spin.setValue(10)
        self._mesh_size_z_spin.setMaximumWidth(48)
        self._mesh_size_z_spin.setToolTip("B-spline mesh size along z.")
        self._mesh_size_y_spin = QSpinBox()
        self._mesh_size_y_spin.setRange(1, 512)
        self._mesh_size_y_spin.setValue(10)
        self._mesh_size_y_spin.setMaximumWidth(48)
        self._mesh_size_y_spin.setToolTip("B-spline mesh size along y.")
        self._mesh_size_x_spin = QSpinBox()
        self._mesh_size_x_spin.setRange(1, 512)
        self._mesh_size_x_spin.setValue(10)
        self._mesh_size_x_spin.setMaximumWidth(48)
        self._mesh_size_x_spin.setToolTip("B-spline mesh size along x.")
        self._mesh_size_row = QWidget()
        mesh_size_layout = QVBoxLayout(self._mesh_size_row)
        mesh_size_layout.setContentsMargins(0, 0, 0, 0)
        mesh_size_layout.setSpacing(4)
        mesh_size_label = self._make_form_label(
            "Mesh size",
            tooltip="B-spline mesh size used for B-spline registration.",
        )
        mesh_size_layout.addWidget(mesh_size_label)
        mesh_size_inputs = QHBoxLayout()
        mesh_size_inputs.setContentsMargins(0, 0, 0, 0)
        mesh_size_inputs.setSpacing(6)
        mesh_size_inputs.addWidget(QLabel("Z"))
        mesh_size_inputs.addWidget(self._mesh_size_z_spin)
        mesh_size_inputs.addWidget(QLabel("Y"))
        mesh_size_inputs.addWidget(self._mesh_size_y_spin)
        mesh_size_inputs.addWidget(QLabel("X"))
        mesh_size_inputs.addWidget(self._mesh_size_x_spin)
        mesh_size_inputs.addStretch(1)
        mesh_size_layout.addLayout(mesh_size_inputs)
        params_layout.addRow(self._mesh_size_row)

        self._metric_combo = QComboBox()
        self._metric_combo.setMinimumContentsLength(14)
        self._metric_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._metric_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._metric_combo.addItems(["correlation", "mattes_mi"])
        params_layout.addRow(
            self._make_form_label(
                "Metric",
                tooltip="Similarity metric optimized during registration. 'correlation' suits same-modality data; 'mattes_mi' is more robust across intensity changes.",
            ),
            self._metric_combo,
        )

        self._initialization_combo = QComboBox()
        self._initialization_combo.addItems(
            ["center_geometry", "center_moments", "none"]
        )
        self._initialization_combo.setToolTip(
            "Transform initializer before optimization. 'center_geometry' aligns centers; "
            "'center_moments' aligns centers of mass; 'none' uses identity."
        )
        params_layout.addRow(
            self._make_form_label(
                "Initialization",
                tooltip="How to initialize the transform before optimization: center geometry, center moments, or identity.",
            ),
            self._initialization_combo,
        )

        self._initial_transform_combo = QComboBox()
        self._initial_transform_combo.setMinimumContentsLength(18)
        self._initial_transform_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._initial_transform_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._initial_transform_combo.setToolTip(
            "Optional pre-computed affine transform used as a warm start before optimization."
        )
        self._initial_transform_row_label = self._make_form_label(
            "Initial transform",
            tooltip="Optional pre-computed affine transform from the Transforms tab used as a warm start before optimization.",
        )
        params_layout.addRow(
            self._initial_transform_row_label,
            self._initial_transform_combo,
        )

        learning_rate_row = QHBoxLayout()
        self._learning_rate_auto_check = QCheckBox("Auto")
        self._learning_rate_auto_check.setChecked(True)
        self._learning_rate_edit = ScientificDoubleSpinBox()
        self._learning_rate_edit.setRange(1e-10, 1e3)
        self._learning_rate_edit.setSingleStep(0.1)
        self._learning_rate_edit.setValue(0.1)
        self._learning_rate_edit.setToolTip(
            "Optimizer step size. Accepts decimal (0.1) or scientific notation (1e-5)."
        )
        self._learning_rate_edit.setEnabled(False)
        self._learning_rate_auto_check.toggled.connect(
            lambda checked: self._learning_rate_edit.setEnabled(not checked)
        )
        learning_rate_row.addWidget(self._learning_rate_auto_check)
        learning_rate_row.addWidget(self._learning_rate_edit, stretch=1)
        params_layout.addRow(
            self._make_form_label(
                "Learning rate",
                tooltip="Optimizer step size. Auto re-estimates it each iteration; otherwise enter a fixed decimal or scientific-notation value.",
            ),
            learning_rate_row,
        )

        self._iterations_spin = QSpinBox()
        self._iterations_spin.setRange(1, 100_000)
        self._iterations_spin.setSingleStep(100)
        self._iterations_spin.setValue(100)
        params_layout.addRow(
            self._make_form_label(
                "Iterations",
                tooltip="Maximum number of optimizer iterations.",
            ),
            self._iterations_spin,
        )

        self._advanced_group = QWidget()
        advanced_group_layout = QVBoxLayout(self._advanced_group)
        advanced_group_layout.setContentsMargins(6, 6, 6, 6)
        advanced_group_layout.setSpacing(6)

        advanced_header = QWidget()
        advanced_header_layout = QHBoxLayout(advanced_header)
        advanced_header_layout.setContentsMargins(0, 0, 0, 0)
        advanced_header_layout.setSpacing(6)

        self._advanced_toggle = QToolButton()
        self._advanced_toggle.setCheckable(True)
        self._advanced_toggle.setChecked(False)
        self._advanced_toggle.setAutoRaise(True)
        self._advanced_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._advanced_toggle.setText("Advanced")
        self._advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        advanced_header_layout.addWidget(self._advanced_toggle)
        advanced_header_layout.addStretch(1)
        advanced_group_layout.addWidget(advanced_header)

        self._advanced_content = QWidget()
        advanced_layout = QFormLayout(self._advanced_content)
        advanced_layout.setSpacing(6)
        advanced_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        advanced_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self._histogram_bins_spin = QSpinBox()
        self._histogram_bins_spin.setRange(8, 512)
        self._histogram_bins_spin.setValue(50)
        self._histogram_bins_spin.setToolTip(
            "Number of histogram bins for Mattes mutual information metric."
        )
        self._histogram_bins_row = self._make_advanced_row(
            advanced_layout,
            "Histogram bins",
            self._histogram_bins_spin,
            tooltip="Number of histogram bins used by the Mattes mutual information metric.",
        )

        self._convergence_min_edit = ScientificDoubleSpinBox()
        self._convergence_min_edit.setRange(1e-10, 1.0)
        self._convergence_min_edit.setSingleStep(1e-6)
        self._convergence_min_edit.setValue(1e-6)
        self._convergence_min_edit.setToolTip(
            "Convergence threshold. Accepts decimal (0.000001) or scientific notation (1e-6)."
        )
        self._convergence_min_row = self._make_advanced_row(
            advanced_layout,
            "Convergence min",
            self._convergence_min_edit,
            tooltip="Convergence threshold below which the optimizer stops early.",
        )

        self._convergence_window_spin = QSpinBox()
        self._convergence_window_spin.setRange(1, 100)
        self._convergence_window_spin.setValue(10)
        self._convergence_window_spin.setToolTip(
            "Number of recent metric values for convergence estimation."
        )
        self._convergence_window_row = self._make_advanced_row(
            advanced_layout,
            "Convergence window",
            self._convergence_window_spin,
            tooltip="Number of recent metric values used to estimate convergence.",
        )

        self._multi_resolution_check = QCheckBox("Enabled")
        self._multi_resolution_check.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._multi_resolution_check.setToolTip(
            "Run registration from coarse to fine resolution levels."
        )
        self._multi_resolution_check.setChecked(False)
        self._multi_resolution_row = self._make_advanced_row(
            advanced_layout,
            "Multi-resolution",
            self._multi_resolution_check,
            tooltip="Whether to optimize from coarse to fine resolution levels.",
        )

        self._shrink_factors_edit = QLineEdit("6, 2, 1")
        self._shrink_factors_edit.setToolTip(
            "Comma-separated shrink factors per resolution level for multi-resolution."
        )
        self._shrink_factors_row = self._make_advanced_row(
            advanced_layout,
            "Shrink factors",
            self._shrink_factors_edit,
            tooltip="Comma-separated downsampling factors for each multi-resolution level.",
        )

        self._smoothing_sigmas_edit = QLineEdit("6, 2, 1")
        self._smoothing_sigmas_edit.setToolTip(
            "Comma-separated smoothing sigmas per resolution level for multi-resolution."
        )
        self._smoothing_sigmas_row = self._make_advanced_row(
            advanced_layout,
            "Smoothing sigmas",
            self._smoothing_sigmas_edit,
            tooltip="Comma-separated Gaussian smoothing sigmas applied at each multi-resolution level.",
        )

        self._interpolation_combo = QComboBox()
        self._interpolation_combo.setMinimumContentsLength(14)
        self._interpolation_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._interpolation_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._interpolation_combo.addItems(["linear", "bspline"])
        self._interpolation_combo.setToolTip(
            "Interpolator used for the resampled output."
        )
        self._interpolation_row = self._make_advanced_row(
            advanced_layout,
            "Resample interp.",
            self._interpolation_combo,
            tooltip="Interpolator used when resampling the registered output onto the target grid.",
        )

        self._fill_value_auto_check = QCheckBox("minimum")
        self._fill_value_auto_check.setChecked(True)
        self._fill_value_auto_check.setToolTip(
            "Automatically use the minimum intensity of the fixed image as fill value."
        )
        self._fill_value_spin = QDoubleSpinBox()
        self._fill_value_spin.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._fill_value_spin.setRange(-1e6, 1e6)
        self._fill_value_spin.setDecimals(3)
        self._fill_value_spin.setValue(0.0)
        self._fill_value_spin.setEnabled(False)
        self._fill_value_spin.setToolTip(
            "Fill value for resampled output outside the input domain."
        )
        self._fill_value_auto_check.toggled.connect(
            lambda checked: self._fill_value_spin.setEnabled(not checked)
        )
        self._multi_resolution_check.toggled.connect(
            self._update_multi_resolution_enabled
        )
        fill_value_row = QHBoxLayout()
        fill_value_row.addWidget(self._fill_value_auto_check)
        fill_value_row.addWidget(self._fill_value_spin, stretch=1)
        fill_value_container = QWidget()
        fill_value_container.setLayout(fill_value_row)
        self._fill_value_row = self._make_advanced_row(
            advanced_layout,
            "Fill value",
            fill_value_container,
            tooltip="Value written outside the moving image field of view after resampling. Choose 'minimum' to use the image minimum automatically.",
        )

        self._keep_diagnostics_check = QCheckBox("Keep full traces")
        self._keep_diagnostics_check.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._keep_diagnostics_check.setToolTip(
            "Whether to keep the full per-frame optimizer traces for within-scan registration."
        )
        self._keep_diagnostics_row = self._make_advanced_row(
            advanced_layout,
            "Diagnostics",
            self._keep_diagnostics_check,
            tooltip="Whether to store the full per-frame optimizer traces for within-scan registration.",
        )

        self._n_jobs_row = self._make_advanced_row(
            advanced_layout,
            "Parallel jobs",
            self._n_jobs_spin,
            tooltip="Number of parallel workers used for within-scan registration. -1 uses all CPUs.",
        )

        advanced_group_layout.addWidget(self._advanced_content)
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        self._metric_combo.currentTextChanged.connect(
            self._update_metric_dependent_visibility
        )
        self._transform_combo.currentTextChanged.connect(
            self._update_transform_dependent_visibility
        )
        self._on_advanced_toggled(False)
        self._update_multi_resolution_enabled(False)
        self._update_metric_dependent_visibility(self._metric_combo.currentText())
        self._update_transform_dependent_visibility(self._transform_combo.currentText())

        self._register_panel = QWidget()
        register_layout = QVBoxLayout(self._register_panel)
        register_layout.setContentsMargins(0, 0, 0, 0)
        register_layout.setSpacing(8)
        params_layout.addRow(self._advanced_group)

        register_layout.addWidget(operation_group)
        register_layout.addWidget(params_group)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run registration")
        self._run_btn.setObjectName("primary_btn")
        self._run_btn.clicked.connect(self._run_registration)
        btn_row.addWidget(self._run_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setObjectName("danger_btn")
        self._abort_btn.setToolTip("Abort the running registration.")
        self._abort_btn.clicked.connect(self._abort_registration)
        self._abort_btn.hide()
        btn_row.addWidget(self._abort_btn)

        register_layout.addLayout(btn_row)

        layout.addWidget(self._register_panel)

        transforms_group = QGroupBox("Transforms")
        transforms_group.setToolTip(
            "Save, load, and reapply affine transforms estimated from between-scan registration."
        )
        transforms_layout = QFormLayout(transforms_group)
        transforms_layout.setSpacing(6)
        transforms_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        transforms_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self._transform_source_combo = QComboBox()
        self._transform_source_combo.setMinimumContentsLength(18)
        self._transform_source_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._transform_source_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        transforms_layout.addRow("Transform", self._transform_source_combo)

        self._transform_target_combo = QComboBox()
        self._transform_target_combo.setMinimumContentsLength(18)
        self._transform_target_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._transform_target_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        transforms_layout.addRow("Input layer", self._transform_target_combo)

        transform_buttons = QHBoxLayout()
        self._save_transform_btn = QPushButton("Save")
        self._save_transform_btn.clicked.connect(self._save_transform)
        self._load_transform_btn = QPushButton("Load")
        self._load_transform_btn.clicked.connect(self._load_transform)
        self._apply_transform_btn = QPushButton("Apply")
        self._apply_transform_btn.clicked.connect(self._apply_transform)
        transform_buttons.addWidget(self._save_transform_btn)
        transform_buttons.addWidget(self._load_transform_btn)
        transform_buttons.addWidget(self._apply_transform_btn)
        transforms_layout.addRow(transform_buttons)

        self._transforms_panel = transforms_group
        layout.addWidget(self._transforms_panel)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setObjectName("status_err")
        self._status.hide()
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMinimumHeight(16)
        self._progress.setTextVisible(True)
        self._progress.hide()
        layout.addWidget(self._progress)

        layout.addStretch()

        self._register_panel_radio.toggled.connect(self._on_panel_changed)
        self._transforms_panel_radio.toggled.connect(self._on_panel_changed)
        self._single_volume_radio.toggled.connect(self._on_mode_changed)
        self._time_series_radio.toggled.connect(self._on_mode_changed)
        self._fixed_combo.currentTextChanged.connect(
            self._validate_registration_selection
        )
        self._initialization_combo.currentTextChanged.connect(
            self._validate_registration_selection
        )
        self._initial_transform_combo.currentTextChanged.connect(
            self._validate_registration_selection
        )
        self._transform_source_combo.currentTextChanged.connect(
            self._validate_registration_selection
        )
        self._learning_rate_auto_check.toggled.connect(
            lambda checked: self._learning_rate_edit.setEnabled(not checked)
        )

        self._mode_parameters = {
            "register_volume": self._snapshot_mode_parameters(is_volumewise=False),
            "register_volumewise": {
                **self._snapshot_mode_parameters(is_volumewise=False),
                "transform": "rigid",
                "learning_rate_auto": False,
                "learning_rate_value": 0.01,
                "n_jobs": -1,
                "keep_diagnostics": False,
            },
        }

        self._refresh_layers()
        self._on_panel_changed()
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
        self._refresh_transform_controls()
        self._validate_registration_selection()

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

    def _transform_source_label(
        self, payload: AffineTransformPayload, *, suffix: str | None = None
    ) -> str:
        """Return a user-facing label for a transform payload."""
        label = payload["name"]
        if suffix:
            label = f"{label} — {suffix}"
        return label

    def _volume_result_layer_name(self, moving_name: str, fixed_name: str) -> str:
        """Return the napari layer name for between-scan registration output."""
        del moving_name, fixed_name
        return "Registered"

    def _volume_preview_layer_name(self) -> str:
        """Return the napari layer name for between-scan progress preview."""
        return "Resampled moving"

    def _volume_fixed_preview_layer_name(self) -> str:
        """Return the napari layer name for the fixed preview layer."""
        return "Fixed"

    def _volume_moving_preview_layer_name(self) -> str:
        """Return the napari layer name for the moving preview layer."""
        return "Moving"

    def _volumewise_result_layer_name(self, moving_name: str) -> str:
        """Return the napari layer name for within-scan registration output."""
        del moving_name
        return "Motion corrected"

    def _volumewise_moving_preview_layer_name(self) -> str:
        """Return the napari layer name for the within-scan moving preview."""
        return "Moving"

    def _refresh_transform_controls(self) -> None:
        """Refresh transform-related layer selectors."""
        source_data = self._transform_source_combo.currentData()
        initial_transform_data = self._initial_transform_combo.currentData()
        target_name = self._transform_target_combo.currentText()

        transform_options: list[tuple[str, tuple[str, str]]] = []
        if self._loaded_transform_payload is not None:
            transform_options.append(
                (
                    self._transform_source_label(
                        self._loaded_transform_payload,
                        suffix="loaded",
                    ),
                    ("loaded", ""),
                )
            )
        for layer in self.viewer.layers:
            payload = _affine_payload_from_layer(layer)
            if payload is None:
                continue
            transform_options.append(
                (
                    self._transform_source_label(payload, suffix=layer.name),
                    ("layer", layer.name),
                )
            )

        self._transform_source_combo.blockSignals(True)
        self._transform_source_combo.clear()
        for label, data in transform_options:
            self._transform_source_combo.addItem(label, data)
        self._transform_source_combo.blockSignals(False)

        self._initial_transform_combo.blockSignals(True)
        self._initial_transform_combo.clear()
        self._initial_transform_combo.addItem("None", None)
        for label, data in transform_options:
            self._initial_transform_combo.addItem(label, data)
        self._initial_transform_combo.blockSignals(False)

        self._transform_target_combo.blockSignals(True)
        self._transform_target_combo.clear()
        self._transform_target_combo.addItems(
            [layer.name for layer in self.viewer.layers]
        )
        self._transform_target_combo.blockSignals(False)

        if source_data is not None:
            for i in range(self._transform_source_combo.count()):
                if self._transform_source_combo.itemData(i) == source_data:
                    self._transform_source_combo.setCurrentIndex(i)
                    break

        if initial_transform_data is not None:
            for i in range(self._initial_transform_combo.count()):
                if self._initial_transform_combo.itemData(i) == initial_transform_data:
                    self._initial_transform_combo.setCurrentIndex(i)
                    break

        target_index = self._transform_target_combo.findText(target_name)
        if target_index >= 0:
            self._transform_target_combo.setCurrentIndex(target_index)

    def _selected_transform_payload(self) -> AffineTransformPayload | None:
        """Return the currently selected affine transform payload."""
        source_data = self._transform_source_combo.currentData()
        if not isinstance(source_data, tuple) or len(source_data) != 2:
            return None

        source_kind, source_name = source_data
        if source_kind == "loaded":
            return self._loaded_transform_payload
        if source_kind != "layer" or not source_name:
            return None
        try:
            layer = cast("Layer", self.viewer.layers[source_name])
        except KeyError:
            return None
        return _affine_payload_from_layer(layer)

    def _selected_initial_transform_payload(self) -> AffineTransformPayload | None:
        """Return the transform payload selected for registration initialization."""
        source_data = self._initial_transform_combo.currentData()
        if source_data is None:
            return None
        if not isinstance(source_data, tuple) or len(source_data) != 2:
            return None

        source_kind, source_name = source_data
        if source_kind == "loaded":
            return self._loaded_transform_payload
        if source_kind != "layer" or not source_name:
            return None
        try:
            layer = cast("Layer", self.viewer.layers[source_name])
        except KeyError:
            return None
        return _affine_payload_from_layer(layer)

    def _validate_initial_transform_selection(
        self,
        *,
        operation: Literal["register_volume", "register_volumewise"],
        moving: xr.DataArray,
        fixed: xr.DataArray | None = None,
    ) -> str | None:
        """Return an inline validation message for transform initialization."""
        payload = self._selected_initial_transform_payload()
        if payload is None or operation != "register_volume":
            return None
        if fixed is None:
            return "Select a fixed layer."

        try:
            affine = affine_transform_from_payload(payload)
        except Exception as exc:  # noqa: BLE001
            return str(exc)

        expected_shape = (moving.ndim + 1, moving.ndim + 1)
        if affine.shape != expected_shape:
            return (
                f"Selected initialization transform has shape {affine.shape}, "
                f"but this registration expects {expected_shape}."
            )
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

    def _set_layer_validation_style(
        self,
        *,
        moving_invalid: bool = False,
        fixed_invalid: bool = False,
        message: str | None = None,
    ) -> None:
        """Update inline validation state for the layer selectors."""
        error_style = "border: 1px solid #e05555;"
        normal_style = ""
        self._moving_combo.setStyleSheet(
            error_style if moving_invalid else normal_style
        )
        self._fixed_combo.setStyleSheet(error_style if fixed_invalid else normal_style)
        self._moving_label.setStyleSheet("color: #e05555;" if moving_invalid else "")
        self._fixed_label.setStyleSheet("color: #e05555;" if fixed_invalid else "")
        self._reference_time_label.setStyleSheet("")
        if message:
            self._layer_validation.setText(message)
            self._layer_validation.show()
        else:
            self._layer_validation.hide()
            self._layer_validation.clear()

    def _set_run_btn_enabled(self, enabled: bool) -> None:
        """Enable or disable the Run button without changing its busy text.

        The button is also disabled in `_begin_work` while a registration is
        running; this helper only handles the idle-state gating driven by
        layer-selection validation.
        """
        # Don't override the busy state.
        if self._run_btn.text() == "Registering…":
            return
        self._run_btn.setEnabled(enabled)

    def _validate_registration_selection(self) -> bool:
        """Validate the current registration-layer selection and show inline feedback.

        Returns
        -------
        bool
            `True` when the selection is valid and a registration can be
            started, `False` otherwise. As a side effect, the Run button is
            enabled/disabled to match the validation result.
        """
        moving_layer = self._selected_layer(self._moving_combo)
        fixed_layer = self._selected_layer(self._fixed_combo)
        operation = self._operation()

        if moving_layer is None:
            self._set_layer_validation_style()
            self._set_run_btn_enabled(False)
            return False

        try:
            moving = _layer_to_dataarray(moving_layer)
        except Exception:
            self._set_layer_validation_style(
                moving_invalid=True,
                message="Could not read the selected moving layer.",
            )
            self._set_run_btn_enabled(False)
            return False

        if operation == "register_volumewise":
            if TIME_DIM not in moving.dims:
                self._set_layer_validation_style(
                    moving_invalid=True,
                    message="Within-scan registration requires a layer with a time dimension.",
                )
                self._set_run_btn_enabled(False)
                return False
            init_message = self._validate_initial_transform_selection(
                operation=operation,
                moving=moving,
            )
            self._set_layer_validation_style(message=init_message)
            self._set_run_btn_enabled(init_message is None)
            return init_message is None

        moving_invalid = False
        fixed_invalid = False
        message: str | None = None

        if fixed_layer is None:
            self._set_layer_validation_style(
                moving_invalid=moving_invalid,
                fixed_invalid=True,
                message="Between-scans registration requires different moving and fixed layers.",
            )
            self._set_run_btn_enabled(False)
            return False

        try:
            fixed = _layer_to_dataarray(fixed_layer)
        except Exception:
            self._set_layer_validation_style(
                fixed_invalid=True,
                message="Could not read the selected fixed layer.",
            )
            self._set_run_btn_enabled(False)
            return False

        if fixed_layer is moving_layer:
            moving_invalid = True
            fixed_invalid = True
            message = "Moving and fixed layers must be different."

        if message is None:
            message = self._validate_initial_transform_selection(
                operation=operation,
                moving=_prepare_between_scan_data(moving),
                fixed=_prepare_between_scan_data(fixed),
            )

        valid = not (moving_invalid or fixed_invalid or message is not None)
        self._set_layer_validation_style(
            moving_invalid=moving_invalid,
            fixed_invalid=fixed_invalid,
            message=message,
        )
        self._set_run_btn_enabled(valid)
        return valid

    def _on_moving_layer_changed(self, _name: str) -> None:
        """Update dependent widgets when the moving layer changes."""
        self._update_reference_time_bounds()
        self._validate_registration_selection()

    def _operation(self) -> Literal["register_volume", "register_volumewise"]:
        """Return the currently selected registration workflow."""
        if self._time_series_radio.isChecked():
            return "register_volumewise"
        return "register_volume"

    def _on_panel_changed(self) -> None:
        """Switch between the register and transforms subpanels."""
        show_register = self._register_panel_radio.isChecked()
        self._register_panel.setVisible(show_register)
        self._transforms_panel.setVisible(not show_register)

    def _on_advanced_toggled(self, checked: bool) -> None:
        """Expand or collapse the advanced-parameter group."""
        self._advanced_content.setVisible(checked)
        self._advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _update_metric_dependent_visibility(self, metric: str) -> None:
        """Show metric-specific advanced parameters for the current metric."""
        self._histogram_bins_row.setVisible(metric == "mattes_mi")

    def _update_multi_resolution_enabled(self, checked: bool) -> None:
        """Show or hide multi-resolution-only parameter inputs."""
        self._shrink_factors_row.setVisible(checked)
        self._smoothing_sigmas_row.setVisible(checked)

    def _update_transform_dependent_visibility(self, transform: str) -> None:
        """Show or hide transform-specific basic parameters."""
        self._mesh_size_row.setVisible(
            self._operation() == "register_volume" and transform == "bspline"
        )

    def _snapshot_mode_parameters(self, *, is_volumewise: bool) -> dict[str, Any]:
        """Capture the current parameter state for one registration mode."""
        return {
            "transform": self._transform_combo.currentText() or "rigid",
            "metric": self._metric_combo.currentText(),
            "initialization": self._initialization_combo.currentText(),
            "initial_transform_source": self._initial_transform_combo.currentData(),
            "learning_rate_auto": self._learning_rate_auto_check.isChecked(),
            "learning_rate_value": self._learning_rate_edit.value(),
            "number_of_iterations": self._iterations_spin.value(),
            "number_of_histogram_bins": self._histogram_bins_spin.value(),
            "mesh_size": (
                self._mesh_size_z_spin.value(),
                self._mesh_size_y_spin.value(),
                self._mesh_size_x_spin.value(),
            ),
            "convergence_minimum_value": self._convergence_min_edit.value(),
            "convergence_window_size": self._convergence_window_spin.value(),
            "use_multi_resolution": self._multi_resolution_check.isChecked(),
            "shrink_factors": self._shrink_factors_edit.text(),
            "smoothing_sigmas": self._smoothing_sigmas_edit.text(),
            "resample_interpolation": self._interpolation_combo.currentText(),
            "fill_value_auto": self._fill_value_auto_check.isChecked(),
            "fill_value": self._fill_value_spin.value(),
            "reference_time": self._reference_time_spin.value(),
            "n_jobs": self._n_jobs_spin.value(),
            "keep_diagnostics": self._keep_diagnostics_check.isChecked(),
            "advanced_open": self._advanced_toggle.isChecked(),
            "is_volumewise": is_volumewise,
        }

    def _apply_mode_parameters(
        self, params: dict[str, Any], *, is_volumewise: bool
    ) -> None:
        """Restore the parameter state for one registration mode."""
        self._transform_combo.blockSignals(True)
        self._transform_combo.clear()
        if is_volumewise:
            self._transform_combo.addItems(["translation", "rigid", "affine"])
        else:
            self._transform_combo.addItems(
                ["translation", "rigid", "affine", "bspline"]
            )
        transform = cast("str", params.get("transform", "rigid"))
        transform_index = self._transform_combo.findText(transform)
        if transform_index < 0:
            transform_index = self._transform_combo.findText("rigid")
        if transform_index >= 0:
            self._transform_combo.setCurrentIndex(transform_index)
        self._transform_combo.blockSignals(False)

        self._metric_combo.setCurrentText(cast("str", params["metric"]))
        self._initialization_combo.setCurrentText(cast("str", params["initialization"]))
        initial_transform_source = params.get("initial_transform_source")
        if initial_transform_source is not None:
            for i in range(self._initial_transform_combo.count()):
                if (
                    self._initial_transform_combo.itemData(i)
                    == initial_transform_source
                ):
                    self._initial_transform_combo.setCurrentIndex(i)
                    break
        self._learning_rate_auto_check.setChecked(
            cast("bool", params["learning_rate_auto"])
        )
        self._learning_rate_edit.setValue(cast("float", params["learning_rate_value"]))
        self._iterations_spin.setValue(cast("int", params["number_of_iterations"]))
        self._histogram_bins_spin.setValue(
            cast("int", params["number_of_histogram_bins"])
        )
        mesh_size = cast("tuple[int, int, int]", params["mesh_size"])
        self._mesh_size_z_spin.setValue(mesh_size[0])
        self._mesh_size_y_spin.setValue(mesh_size[1])
        self._mesh_size_x_spin.setValue(mesh_size[2])
        self._convergence_min_edit.setValue(
            cast("float", params["convergence_minimum_value"])
        )
        self._convergence_window_spin.setValue(
            cast("int", params["convergence_window_size"])
        )
        self._multi_resolution_check.setChecked(
            cast("bool", params["use_multi_resolution"])
        )
        self._shrink_factors_edit.setText(cast("str", params["shrink_factors"]))
        self._smoothing_sigmas_edit.setText(cast("str", params["smoothing_sigmas"]))
        self._interpolation_combo.setCurrentText(
            cast("str", params["resample_interpolation"])
        )
        self._fill_value_auto_check.setChecked(cast("bool", params["fill_value_auto"]))
        self._fill_value_spin.setValue(cast("float", params["fill_value"]))
        self._reference_time_spin.setValue(cast("int", params["reference_time"]))
        self._n_jobs_spin.setValue(cast("int", params["n_jobs"]))
        self._keep_diagnostics_check.setChecked(
            cast("bool", params["keep_diagnostics"])
        )
        self._advanced_toggle.setChecked(cast("bool", params["advanced_open"]))
        self._on_advanced_toggled(self._advanced_toggle.isChecked())
        self._update_metric_dependent_visibility(self._metric_combo.currentText())
        self._update_multi_resolution_enabled(self._multi_resolution_check.isChecked())
        self._update_transform_dependent_visibility(self._transform_combo.currentText())

    def _on_mode_changed(self) -> None:
        """Update the panel when the registration mode changes."""
        new_mode = self._operation()
        previous_mode = self._active_mode
        previous_is_volumewise = previous_mode == "register_volumewise"
        is_volumewise = new_mode == "register_volumewise"

        if previous_mode in self._mode_parameters:
            self._mode_parameters[previous_mode] = self._snapshot_mode_parameters(
                is_volumewise=previous_is_volumewise
            )

        self._fixed_label.setVisible(not is_volumewise)
        self._fixed_combo.setVisible(not is_volumewise)
        self._fixed_combo.setEnabled(not is_volumewise)
        self._initial_transform_row_label.setVisible(not is_volumewise)
        self._initial_transform_combo.setVisible(not is_volumewise)
        self._initial_transform_combo.setEnabled(not is_volumewise)
        self._reference_time_label.setVisible(is_volumewise)
        self._reference_time_spin.setVisible(is_volumewise)
        self._n_jobs_row.setVisible(is_volumewise)

        self._fill_value_row.setVisible(not is_volumewise)
        self._keep_diagnostics_row.setVisible(is_volumewise)

        self._apply_mode_parameters(
            self._mode_parameters[new_mode],
            is_volumewise=is_volumewise,
        )
        self._active_mode = new_mode

        self._update_reference_time_bounds()
        self._validate_registration_selection()

    def _begin_work(self) -> None:
        """Put the panel into its busy state."""
        self._run_btn.setEnabled(False)
        self._run_btn.setText("Registering…")
        self._abort_btn.setEnabled(True)
        self._abort_btn.setText("Abort")
        self._abort_btn.show()
        self._status.hide()
        if self._volumewise_progress_total is None:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, self._volumewise_progress_total)
            self._progress.setValue(0)
        self._progress.show()
        QApplication.processEvents()

    def _abort_registration(self) -> None:
        """Request cooperative cancellation of the running registration."""
        if self._worker is None or self._abort_event is None:
            return
        self._abort_event.set()
        self._abort_btn.setEnabled(False)
        self._abort_btn.setText("Aborting…")
        self._set_error("Aborting registration…")

    def _setup_volumewise_progress(
        self,
        *,
        moving_layer: "Image",
        moving: xr.DataArray,
        layer_name: str,
    ) -> NapariVolumewiseProgress:
        """Create a progress bridge and output layer for volumewise registration."""
        self._teardown_volumewise_progress(remove_layer=True)

        moving_display_kwargs = _image_display_kwargs_from_layer(moving_layer)
        moving_display_kwargs["colormap"] = "red"

        display_kwargs = dict(moving_display_kwargs)
        display_kwargs["colormap"] = "cyan"
        display_kwargs["blending"] = "additive"
        contrast_limits = tuple(calc_data_range(moving.data))
        preview_data = np.full(
            moving.shape,
            fill_value=float(np.min(moving.data)),
            dtype=np.asarray(moving.data).dtype,
        )
        preview = xr.DataArray(
            preview_data,
            dims=moving.dims,
            coords=moving.coords,
            attrs=moving.attrs.copy(),
        )

        _, moving_preview_layer = plot_napari(
            moving,
            viewer=self.viewer,
            name=self._volumewise_moving_preview_layer_name(),
            show_colorbar=False,
            contrast_limits=contrast_limits,
            **moving_display_kwargs,
        )
        _, layer = plot_napari(
            preview,
            viewer=self.viewer,
            name=layer_name,
            show_colorbar=False,
            contrast_limits=contrast_limits,
            **display_kwargs,
        )
        bridge = NapariVolumewiseProgressBridge()
        bridge.frame_progress.connect(self._update_volumewise_progress_bar)
        bridge.frame_completed.connect(self._update_volumewise_progress_frame)

        self._volumewise_progress_bridge = bridge
        self._volumewise_progress_layer = cast("Image", layer)
        self._volumewise_moving_preview_layer = cast("Image", moving_preview_layer)
        self._volumewise_progress_time_axis = moving.dims.index(TIME_DIM)
        self._volumewise_progress_total = moving.sizes[TIME_DIM]
        self._progress.setRange(0, self._volumewise_progress_total)
        self._progress.setValue(0)
        return NapariVolumewiseProgress(
            bridge,
            n_frames=moving.sizes[TIME_DIM],
        )

    def _update_volumewise_progress_bar(
        self,
        completed_frames: int,
        total_frames: int,
    ) -> None:
        """Update the determinate progress bar for volumewise registration."""
        self._progress.setRange(0, max(total_frames, 1))
        self._progress.setValue(min(completed_frames, total_frames))

    def _update_volumewise_progress_frame(
        self,
        frame_index: int,
        arr: object,
    ) -> None:
        """Write one completed registered frame into the volumewise output layer."""
        layer = self._volumewise_progress_layer
        time_axis = self._volumewise_progress_time_axis
        if layer is None or time_axis is None or not isinstance(arr, np.ndarray):
            return

        data = np.asarray(layer.data)
        if time_axis >= data.ndim:
            return
        index = tuple(
            frame_index if axis == time_axis else slice(None)
            for axis in range(data.ndim)
        )
        data[index] = arr
        layer.refresh()

    def _teardown_volumewise_progress(self, *, remove_layer: bool) -> None:
        """Reset volumewise progress-layer state."""
        if remove_layer:
            for attr_name in (
                "_volumewise_progress_layer",
                "_volumewise_moving_preview_layer",
            ):
                layer = cast("Image | None", getattr(self, attr_name))
                if layer is not None:
                    try:
                        self.viewer.layers.remove(layer)
                    except (KeyError, ValueError):
                        pass
                    setattr(self, attr_name, None)
        self._volumewise_progress_bridge = None
        self._volumewise_progress_time_axis = None
        self._volumewise_progress_total = None

    def _setup_volume_progress(
        self,
        *,
        moving_layer: "Image",
        fixed_layer: "Image",
        moving: xr.DataArray,
        fixed: xr.DataArray,
        layer_name: str,
    ) -> "Callable[..., RegistrationProgress] | None":
        """Build a napari progress bridge and preview layer for register_volume.

        Creates an empty image layer on the fixed grid (with the final target
        name) and wires a
        [`NapariProgressBridge`][confusius._napari._registration._progress.NapariProgressBridge]
        so that every iteration of SimpleITK's optimizer streams the resampled
        array into that layer. The returned factory is forwarded to
        `register_volume` via its `progress_plotter` argument.

        Parameters
        ----------
        moving_layer : napari.layers.Layer
            Moving source layer. Used for display defaults (colormap,
            contrast limits) on the preview layer, since the resampled output
            lives in the moved intensity space.
        fixed_layer : napari.layers.Layer
            Fixed reference layer. Defines the shape, scale, translate, and
            coordinate system of the preview/output layer.
        moving : xarray.DataArray
            Spatial-only moving data used to seed the preview layer.
        fixed : xarray.DataArray
            Spatial-only DataArray view of `fixed_layer`, used to build the
            empty preview grid.
        layer_name : str
            Name for the preview (and later final) layer.

        Returns
        -------
        callable or None
            Factory suited for `register_volume`'s `progress_plotter`
            argument, or `None` when the preview layer could not be created
            (in which case `register_volume` runs without live progress).
        """
        self._teardown_volume_progress()

        fixed_display_kwargs = _image_display_kwargs_from_layer(fixed_layer)
        fixed_display_kwargs["colormap"] = "red"

        moving_display_kwargs = _image_display_kwargs_from_layer(moving_layer)
        moving_display_kwargs["colormap"] = "cyan"
        moving_display_kwargs["blending"] = "additive"

        display_kwargs = dict(moving_display_kwargs)
        # Seed contrast limits with the moving layer so the preview is shown in
        # the same intensity space as the final resampled volume.
        moving_contrast = getattr(moving_layer, "contrast_limits", None)
        if moving_contrast is not None:
            display_kwargs.setdefault("contrast_limits", tuple(moving_contrast))

        # Render the preview in cyan with additive blending. napari sums the
        # RGB channels of the two layers, so red+cyan highlights
        # misregistered regions as a pure colour. `_image_display_kwargs_from_layer`
        # copies the moving layer's colormap, so we override it explicitly
        # rather than rely on `setdefault`.
        display_kwargs["colormap"] = "cyan"
        display_kwargs["blending"] = "additive"

        # Seed the preview with the moving image resampled onto the fixed
        # grid using an identity transform. This makes the first frame a
        # meaningful "unaligned moving on fixed grid" view that the user can
        # compare against the red fixed, instead of a zero-valued blank that
        # would flash a full-FOV tint. The SimpleITK iteration events then
        # overwrite the data in place as the registration progresses.
        try:
            identity = np.eye(fixed.ndim + 1, dtype=float)
            preview = resample_like(
                moving,
                fixed,
                identity,
                interpolation=cast(
                    "Literal['linear', 'bspline']",
                    "linear",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            # Fall back to a zero-valued seed if the initial resample fails
            # for any reason. The first iteration will populate the preview.
            self._set_error(f"Could not seed progress layer: {exc}")
            preview = xr.DataArray(
                np.zeros(fixed.shape, dtype=np.float32),
                coords=fixed.coords,
                dims=fixed.dims,
                attrs=fixed.attrs.copy(),
            )

        try:
            _, fixed_preview_layer = plot_napari(
                fixed,
                viewer=self.viewer,
                name=self._volume_fixed_preview_layer_name(),
                show_colorbar=False,
                **fixed_display_kwargs,
            )
            _, moving_preview_layer = plot_napari(
                moving,
                viewer=self.viewer,
                name=self._volume_moving_preview_layer_name(),
                show_colorbar=False,
                **moving_display_kwargs,
            )
            _, layer = plot_napari(
                preview,
                viewer=self.viewer,
                name=layer_name,
                show_colorbar=False,
                **display_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_error(f"Could not create progress layer: {exc}")
            return None

        bridge = NapariProgressBridge()
        bridge.iterated.connect(self._update_progress_layer)
        # `finished` is informational: we tear the preview down on
        # `_on_registration_finished` / `_on_registration_failed` instead, so
        # no extra slot is required here.
        self._progress_bridge = bridge
        self._progress_layer = cast("Image", layer)
        self._progress_fixed_layer = cast("Image", fixed_preview_layer)
        self._progress_moving_layer = cast("Image", moving_preview_layer)
        self._progress_moving_layer.visible = False

        # Lazily build the bottom-dock metric plotter. The widget is reused
        # across runs; only the data buffer is reset.
        self._ensure_metric_plotter()
        plotter = self._metric_plotter
        if plotter is not None:
            plotter.reset()
            bridge.metric_updated.connect(plotter.add_metric)
        return make_napari_progress_factory(bridge)

    def _update_progress_layer(self, arr: object) -> None:
        """Write an intermediate resampled array into the preview layer.

        Invoked on the GUI thread via `NapariProgressBridge.iterated`. The
        payload is a numpy array in numpy axis order matching the fixed grid
        shape. Shape/coordinate mismatches are silently ignored: they
        indicate that another run's stale signal slipped through or that the
        preview layer has already been torn down.

        Parameters
        ----------
        arr : numpy.ndarray
            Resampled moving image for the current iteration.
        """
        layer = self._progress_layer
        if layer is None:
            return
        if not isinstance(arr, np.ndarray):
            return
        if arr.shape != layer.data.shape:
            return
        layer.data = arr  # type: ignore[invalid-assignment]

    def _teardown_volume_progress(self) -> None:
        """Remove the progress preview layer and bridge references, if any.

        Called by `_on_registration_finished` and `_on_registration_failed`
        so the newly added result layer replaces the preview without leaving
        duplicates behind. The moving layer's hidden state is not restored.
        The metric plotter is kept (docked, with its final trace) so the
        user can inspect the convergence curve after the run.
        """
        for attr_name in (
            "_progress_layer",
            "_progress_fixed_layer",
            "_progress_moving_layer",
        ):
            layer = cast("Image | None", getattr(self, attr_name))
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except (KeyError, ValueError):
                    pass
                setattr(self, attr_name, None)
        # Drop the bridge reference; the plotter connection becomes inert
        # when the bridge is garbage-collected.
        self._progress_bridge = None

    def _ensure_metric_plotter(self) -> RegistrationMetricPlotter | None:
        """Return the bottom-dock metric plotter, creating and docking it on first use.

        Mirrors the lazy-dock pattern used by `SignalPanel`. The plotter widget is
        reused across runs; `_setup_volume_progress` resets its data buffer
        before each run. Returns `None` only when the dock could not be created
        (in which case the registration still runs, just without a live metric
        view).
        """
        if self._metric_plotter is None:
            self._metric_plotter = RegistrationMetricPlotter(self.viewer)

        if self._metric_dock is None or self._metric_plotter.parent() is None:
            dock = self.viewer.window.add_dock_widget(
                self._metric_plotter,
                name="Registration Metric",
                area="bottom",
            )
            self._metric_dock = cast("QDockWidget", dock)

            # Mirror the HiDPI click-offset fix from the SignalPanel so the
            # canvas paints at the right device-pixel ratio the first time.
            def _settle_layout() -> None:
                try:
                    main_win = self._find_main_window(dock)
                except RuntimeError:
                    return
                if main_win is None:
                    return
                from qtpy.QtCore import QSize

                central = main_win.centralWidget()
                if central is None:
                    return
                central.setMinimumSize(QSize(0, 0))
                for w in central.findChildren(QWidget):
                    w.setMinimumSize(QSize(0, 0))
                for side_dock in main_win.findChildren(QDockWidget):
                    if side_dock is dock:
                        continue
                    side_dock.setMinimumHeight(0)
                    widget = side_dock.widget()
                    if widget is not None:
                        widget.setMinimumSize(QSize(0, 0))
                current = main_win.size()
                if current.height() < 800:
                    main_win.resize(current.width(), 800)
                main_win.resizeDocks([dock], [220], Qt.Orientation.Vertical)

            QTimer.singleShot(200, _settle_layout)

        return self._metric_plotter

    def _find_main_window(self, widget: QWidget) -> QMainWindow | None:
        """Traverse up the widget hierarchy to find the QMainWindow.

        Parameters
        ----------
        widget : QWidget
            Starting widget to search from.

        Returns
        -------
        QMainWindow or None
            The main window if found, otherwise None.
        """
        try:
            parent = widget.parent()
        except RuntimeError:
            return None
        while parent is not None:
            if isinstance(parent, QMainWindow):
                return parent
            try:
                parent = parent.parent()
            except RuntimeError:
                return None
        return None

    def _end_work(self) -> None:
        """Restore the idle UI state after background work."""
        self._worker = None
        self._abort_event = None
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run registration")
        self._abort_btn.setEnabled(True)
        self._abort_btn.setText("Abort")
        self._abort_btn.hide()
        self._progress.setRange(0, 0)
        self._progress.setValue(0)
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

    def _save_transform(self) -> None:
        """Save the selected affine transform payload to JSON."""
        payload = self._selected_transform_payload()
        if payload is None:
            self._set_error("Select an affine transform to save.")
            return

        default_name = payload["name"].replace("/", "-")
        start = str(Path.home() / f"{default_name}.json")
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save transform",
            start,
            "JSON files (*.json)",
        )
        if not path_str:
            return

        save_affine_transform_payload(path_str, payload)
        show_info(f"Saved transform: {path_str}")

    def _load_transform(self) -> None:
        """Load an affine transform payload from JSON."""
        start = str(Path.home())
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load transform",
            start,
            "JSON files (*.json)",
        )
        if not path_str:
            return

        try:
            self._loaded_transform_payload = load_affine_transform_payload(path_str)
        except Exception as exc:  # noqa: BLE001
            self._set_error(str(exc))
            show_error(str(exc))
            return

        self._refresh_transform_controls()
        for i in range(self._transform_source_combo.count()):
            if self._transform_source_combo.itemData(i) == ("loaded", ""):
                self._transform_source_combo.setCurrentIndex(i)
                break
        show_info(f"Loaded transform: {self._loaded_transform_payload['name']}")

    def _apply_transform(self) -> None:
        """Apply the selected affine transform to a layer."""
        payload = self._selected_transform_payload()
        if payload is None:
            self._set_error("Select an affine transform to apply.")
            return

        moving_layer = self._selected_layer(self._transform_target_combo)
        if moving_layer is None:
            self._set_error("Select an input layer to transform.")
            return

        try:
            moving = _layer_to_dataarray(moving_layer)
            affine = affine_transform_from_payload(payload)
            output_grid = output_grid_from_payload(payload)
        except Exception as exc:  # noqa: BLE001
            self._set_error(str(exc))
            return

        worker = thread_worker(resample_volume)(
            moving,
            affine,
            shape=output_grid["shape"],
            spacing=output_grid["spacing"],
            origin=output_grid["origin"],
            dims=output_grid["dims"],
            interpolation=cast(
                "Literal['linear', 'bspline']", self._interpolation_combo.currentText()
            ),
        )
        apply_payload = {
            "moving_layer_name": moving_layer.name,
            "target_layer_name": payload["target_layer_name"],
            "transform_source": payload["name"],
        }
        self._worker = worker
        self._begin_work()
        worker.returned.connect(
            lambda result: self._on_apply_transform_finished(apply_payload, result)
        )
        worker.errored.connect(self._on_registration_failed)
        worker.finished.connect(self._end_work)
        worker.start()

    def _run_registration(self) -> None:
        """Validate inputs and start the selected registration workflow."""
        operation = self._operation()
        moving_layer = self._selected_layer(self._moving_combo)
        fixed_layer = self._selected_layer(self._fixed_combo)

        if moving_layer is None:
            self._set_error("Select a moving layer.")
            return
        if not self._validate_registration_selection():
            return

        try:
            learning_rate: float | Literal["auto"]
            if self._learning_rate_auto_check.isChecked():
                learning_rate = "auto"
            else:
                learning_rate = self._learning_rate_edit.value()
            moving = _layer_to_dataarray(moving_layer)
        except Exception as exc:  # noqa: BLE001
            self._set_error(str(exc))
            return

        convergence_minimum_value = self._convergence_min_edit.value()

        # Parse advanced parameters
        shrink_factors = _parse_sequence(self._shrink_factors_edit.text())
        smoothing_sigmas = _parse_sequence(self._smoothing_sigmas_edit.text())
        use_multi_res = self._multi_resolution_check.isChecked()
        if not use_multi_res:
            shrink_factors = None
            smoothing_sigmas = None

        payload: dict[str, Any] = {
            "operation": operation,
            "moving_layer_name": moving_layer.name,
            "transform": self._transform_combo.currentText(),
            "metric": self._metric_combo.currentText(),
            "learning_rate": learning_rate,
            "number_of_iterations": self._iterations_spin.value(),
            "use_multi_resolution": use_multi_res,
            "resample_interpolation": self._interpolation_combo.currentText(),
            "mesh_size": (
                self._mesh_size_z_spin.value(),
                self._mesh_size_y_spin.value(),
                self._mesh_size_x_spin.value(),
            ),
            "number_of_histogram_bins": self._histogram_bins_spin.value(),
            "convergence_minimum_value": convergence_minimum_value,
            "convergence_window_size": self._convergence_window_spin.value(),
            "initialization": self._initialization_combo.currentText(),
            "shrink_factors": shrink_factors,
            "smoothing_sigmas": smoothing_sigmas,
            "keep_diagnostics": self._keep_diagnostics_check.isChecked(),
            "fill_value": None
            if self._fill_value_auto_check.isChecked()
            else self._fill_value_spin.value(),
        }
        self._abort_event = Event()

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

            moving = _prepare_between_scan_data(moving)
            fixed = _prepare_between_scan_data(fixed)

            initial_transform_payload = self._selected_initial_transform_payload()
            initial_transform: npt.NDArray[np.floating] | None = None
            if initial_transform_payload is not None:
                try:
                    initial_transform = affine_transform_from_payload(
                        initial_transform_payload
                    )
                except Exception as exc:  # noqa: BLE001
                    self._set_error(str(exc))
                    return
                payload["initial_transform_source"] = initial_transform_payload["name"]

            payload["fixed_layer_name"] = fixed_layer.name

            progress_plotter = self._setup_volume_progress(
                moving_layer=cast("Image", moving_layer),
                fixed_layer=cast("Image", fixed_layer),
                moving=moving,
                fixed=fixed,
                layer_name=self._volume_preview_layer_name(),
            )

            worker = thread_worker(_run_register_volume_registration_volume)(
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
                mesh_size=payload["mesh_size"] or (10, 10, 10),
                number_of_histogram_bins=payload["number_of_histogram_bins"],
                convergence_minimum_value=payload["convergence_minimum_value"],
                convergence_window_size=payload["convergence_window_size"],
                initialization=cast(
                    "Literal['center_geometry', 'center_moments'] | None",
                    None
                    if payload["initialization"] == "none"
                    else payload["initialization"],
                ),
                initial_transform=initial_transform,
                shrink_factors=payload["shrink_factors"] or (6, 2, 1),
                smoothing_sigmas=payload["smoothing_sigmas"] or (6, 2, 1),
                fill_value=payload["fill_value"],
                progress_plotter=progress_plotter,
                abort_event=self._abort_event,
            )
        else:
            if TIME_DIM not in moving.dims:
                self._set_error(
                    "register_volumewise requires a layer with a time dimension."
                )
                return

            payload["reference_time"] = self._reference_time_spin.value()
            payload["n_jobs"] = self._n_jobs_spin.value()

            progress_reporter = self._setup_volumewise_progress(
                moving_layer=cast("Image", moving_layer),
                moving=moving,
                layer_name=self._volumewise_result_layer_name(
                    payload["moving_layer_name"]
                ),
            )

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
                number_of_histogram_bins=payload["number_of_histogram_bins"],
                convergence_minimum_value=payload["convergence_minimum_value"],
                convergence_window_size=payload["convergence_window_size"],
                initialization=cast(
                    "Literal['center_geometry', 'center_moments'] | None",
                    None
                    if payload["initialization"] == "none"
                    else payload["initialization"],
                ),
                shrink_factors=payload["shrink_factors"] or (6, 2, 1),
                smoothing_sigmas=payload["smoothing_sigmas"] or (6, 2, 1),
                keep_diagnostics=payload["keep_diagnostics"],
                abort_event=self._abort_event,
                progress_reporter=progress_reporter,
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
            registered.attrs["registration_status"] = diagnostics.status
            layer_name = self._volume_result_layer_name(
                cast("str", payload["moving_layer_name"]),
                cast("str", payload["fixed_layer_name"]),
            )
            metadata: dict[str, Any] = {
                "registration_transform": transform,
                "registration_diagnostics": diagnostics,
                "registration_status": diagnostics.status,
            }
            if isinstance(transform, np.ndarray):
                affine_transform = np.asarray(transform, dtype=float)
                metadata["confusius_transform"] = make_affine_transform_payload(
                    affine_transform,
                    reference=registered,
                    source_layer_name=cast(str, payload["moving_layer_name"]),
                    target_layer_name=cast(str, payload["fixed_layer_name"]),
                    operation=operation,
                    transform_model=cast(str, payload["transform"]),
                    metric=cast(str, payload["metric"]),
                    diagnostics=diagnostics,
                )
        else:
            registered = cast("xr.DataArray", result).copy(deep=False)
            registered.attrs = registered.attrs.copy()
            registered.attrs["registration_operation"] = operation
            layer_name = self._volumewise_result_layer_name(
                cast("str", payload["moving_layer_name"])
            )
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
        # The result layer is the registered stand-in for the moving layer:
        # it must use the same cyan + additive styling so the red/cyan
        # overlay persists after the run.
        if operation == "register_volume":
            display_kwargs["colormap"] = "cyan"
            display_kwargs["blending"] = "additive"
        contrast_limits = tuple(calc_data_range(registered.data))

        if operation == "register_volume" and self._progress_layer is not None:
            layer = self._progress_layer
            layer.data = np.asarray(registered.data)  # type: ignore[invalid-assignment]
            layer.name = layer_name
            if hasattr(layer, "contrast_limits"):
                layer.contrast_limits = contrast_limits
            self._progress_bridge = None
        elif (
            operation == "register_volumewise"
            and self._volumewise_progress_layer is not None
        ):
            layer = self._volumewise_progress_layer
            layer.data = np.asarray(registered.data)  # type: ignore[invalid-assignment]
            if hasattr(layer, "contrast_limits"):
                layer.contrast_limits = contrast_limits
            self._teardown_volumewise_progress(remove_layer=False)
        else:
            _, layer = plot_napari(
                registered,
                viewer=self.viewer,
                name=layer_name,
                show_colorbar=False,
                contrast_limits=contrast_limits,
                **display_kwargs,
            )
        layer.metadata.update(metadata)
        layer.metadata["xarray"] = registered
        self.viewer.layers.selection.active = layer
        self._refresh_transform_controls()

        motion_params = metadata.get("motion_params")
        volumewise_aborted = False
        if operation == "register_volumewise" and motion_params is not None:
            try:
                statuses = motion_params["status"]
            except Exception:  # noqa: BLE001
                statuses = None
            if statuses is not None:
                volumewise_aborted = bool((statuses == "aborted").any())
        registration_status = (
            cast("str", metadata["registration_status"])
            if operation == "register_volume"
            else ("aborted" if volumewise_aborted else "completed")
        )
        if operation == "register_volumewise":
            self._progress.setValue(self._progress.maximum())

        if registration_status == "aborted":
            self._set_error("Registration aborted; added partial result.")
            show_info(f"Registration aborted; added partial layer: {layer.name}")
        else:
            show_info(f"Added registered layer: {layer.name}")

    def _on_apply_transform_finished(
        self, payload: dict[str, str], result: xr.DataArray
    ) -> None:
        """Add a resampled layer produced from an existing affine transform.

        Parameters
        ----------
        payload : dict[str, str]
            UI snapshot captured before the worker started.
        result : xarray.DataArray
            Resampled output.
        """
        registered = result.copy(deep=False)
        registered.attrs = registered.attrs.copy()
        registered.attrs["registration_operation"] = "apply_transform"

        layer_name = f"{payload['moving_layer_name']} → {payload['target_layer_name']}"
        contrast_limits = tuple(calc_data_range(registered.data))

        _, layer = plot_napari(
            registered,
            viewer=self.viewer,
            name=layer_name,
            show_colorbar=False,
            contrast_limits=contrast_limits,
        )
        layer.metadata["xarray"] = registered
        layer.metadata["registration_operation"] = "apply_transform"
        layer.metadata["registration_parameters"] = payload.copy()
        self.viewer.layers.selection.active = layer
        show_info(f"Added transformed layer: {layer.name}")

    def _on_registration_failed(self, exc: BaseException) -> None:
        """Handle a failed worker execution.

        Parameters
        ----------
        exc : BaseException
            Exception raised by the worker.
        """
        self._teardown_volume_progress()
        self._teardown_volumewise_progress(remove_layer=True)
        self._set_error(str(exc))
        show_error(str(exc))
