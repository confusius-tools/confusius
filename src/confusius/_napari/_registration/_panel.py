"""Registration panel for the ConfUSIus napari plugin."""

from __future__ import annotations

from threading import Event
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from confusius._dims import TIME_DIM
from confusius._napari._registration._metric_plotter import (
    RegistrationMetricPlotter,
)
from confusius._napari._registration._panel_parameters import (
    get_default_registration_parameters,
    get_registration_parameters,
    set_registration_parameters,
)
from confusius._napari._registration._panel_selection import (
    current_metric,
    current_resample_interpolation,
    current_scale_mode,
    current_transform_model,
    get_layer_by_name,
    on_moving_layer_changed,
    refresh_layers,
    selected_layer,
    set_layer_validation_style,
    set_run_btn_enabled,
    update_reference_time_bounds,
    validate_registration_selection,
)
from confusius._napari._registration._panel_progress import (
    create_volume_progress_plotter,
    setup_volumewise_progress,
)
from confusius._napari._registration._panel_results import (
    on_volume_registration_finished,
    on_volumewise_registration_finished,
)
from confusius._napari._registration._panel_worker_state import on_registration_failed
from confusius._napari._registration._panel_transforms import (
    apply_selected_transform,
    get_available_transform_payloads,
    get_selected_center_initialization,
    get_selected_initial_transform,
    load_transform,
    refresh_transform_controls,
    save_selected_transform,
)
from confusius._napari._registration._transform_payloads import TransformPayload
from confusius._napari._registration._panel_utils import (
    ScientificDoubleSpinBox,
    _apply_registration_scale,
    _get_source_dataarray,
    _is_registration_source_layer,
    _parse_comma_separated_ints,
    _prepare_between_scan_data,
)
from confusius._napari._registration._progress import (
    NapariProgressBridge,
    NapariRegistrationProgressReporterBridge,
)
from confusius.registration import register_volume, register_volumewise

if TYPE_CHECKING:
    import napari
    import numpy.typing as npt
    from napari.layers import Image, Layer


ScaleMode = Literal["off", "dB", "sqrt"]
"""Allowed registration intensity-scaling modes used by the panel."""

MetricName = Literal["correlation", "mattes_mi"]
"""Allowed registration metric names exposed by the panel."""

VolumeTransformType = Literal["translation", "rigid", "affine", "bspline"]
"""Allowed transform models for between-scan registration."""

VolumewiseTransformType = Literal["translation", "rigid", "affine"]
"""Allowed transform models for within-scan registration."""

ResampleInterpolation = Literal["linear", "bspline"]
"""Allowed interpolation modes for resampling previews and outputs."""

CenterInitialization = Literal["center_geometry", "center_moments"]
"""Allowed built-in center-based initialization modes."""

RegistrationOperation = Literal["register_volume", "register_volumewise"]
"""Allowed registration workflows handled by the panel."""

TransformSourceKind = Literal["loaded", "layer", "manual"]
"""Kinds of transform sources offered in the transforms UI."""

TransformSourceData = tuple[TransformSourceKind, str]
"""Validated transform-source selector payload `(kind, name)`."""

InitializationSelection = CenterInitialization | TransformSourceData | None
"""Validated initialization selection from the registration UI."""

RegistrationParameterMode = Literal["volume", "volumewise"]
"""Registration-parameter mode used for UI snapshot and restore helpers."""


class ModeParameters(TypedDict):
    """Session-scoped UI parameters for one registration mode."""

    transform: str
    metric: MetricName
    scale: ScaleMode
    initialization: InitializationSelection
    learning_rate_auto: bool
    learning_rate_value: float
    number_of_iterations: int
    number_of_histogram_bins: int
    mesh_size: tuple[int, int, int]
    convergence_minimum_value: float
    convergence_window_size: int
    use_multi_resolution: bool
    shrink_factors: str
    smoothing_sigmas: str
    resample_interpolation: ResampleInterpolation
    fill_value_auto: bool
    fill_value: float
    reference_time: int
    n_jobs: int
    keep_diagnostics: bool
    advanced_open: bool


class RegistrationRunPayloadBase(TypedDict):
    """Shared UI snapshot fields captured before a registration worker starts."""

    moving_layer_name: str
    metric: MetricName
    scale: ScaleMode
    learning_rate: float | Literal["auto"]
    number_of_iterations: int
    use_multi_resolution: bool
    resample_interpolation: ResampleInterpolation
    number_of_histogram_bins: int
    convergence_minimum_value: float
    convergence_window_size: int
    initialization: InitializationSelection
    shrink_factors: tuple[int, ...] | None
    smoothing_sigmas: tuple[int, ...] | None
    keep_diagnostics: bool
    fill_value: float | None


class VolumeRegistrationRunPayload(RegistrationRunPayloadBase):
    """UI snapshot for between-scan registration."""

    operation: Literal["register_volume"]
    transform: VolumeTransformType
    mesh_size: tuple[int, int, int]
    fixed_layer_name: str
    initial_transform_source: NotRequired[str]


class VolumewiseRegistrationRunPayload(RegistrationRunPayloadBase):
    """UI snapshot for within-scan registration."""

    operation: Literal["register_volumewise"]
    transform: VolumewiseTransformType
    mesh_size: tuple[int, int, int]
    reference_time: int
    n_jobs: int


class ApplyTransformPayload(TypedDict):
    """UI snapshot for applying an existing transform."""

    moving_layer_name: str
    target_layer_name: str
    transform_source: str


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
        self._loaded_transform_payload: TransformPayload | None = None
        # Per-run progress state. Set on the GUI thread before the worker starts.
        self._progress_bridge: NapariProgressBridge | None = None
        self._progress_layer: Image | None = None
        self._progress_fixed_layer: Image | None = None
        self._progress_moving_layer: Image | None = None
        self._manual_transform_event_layers: list[Layer] = []
        self._volumewise_progress_bridge: (
            NapariRegistrationProgressReporterBridge | None
        ) = None
        self._volumewise_progress_layer: Image | None = None
        self._volumewise_moving_preview_layer: Image | None = None
        self._volumewise_progress_time_axis: int | None = None
        self._volumewise_progress_total: int | None = None
        # Bottom-dock metric curve. Created lazily on the first run, reused
        # across subsequent runs, and torn down with the progress state.
        self._metric_plotter: RegistrationMetricPlotter | None = None
        self._metric_dock: QDockWidget | None = None
        self._active_operation: Literal["register_volume", "register_volumewise"] = (
            "register_volume"
        )
        self._registration_parameters_by_operation: dict[
            RegistrationOperation, ModeParameters
        ] = {}
        self._refresh_transform_controls_callback = lambda: refresh_transform_controls(
            self
        )
        self._save_transform_callback = lambda: save_selected_transform(self)
        self._load_transform_callback = lambda: load_transform(self)
        self._apply_transform_callback = lambda: apply_selected_transform(self)
        self._setup_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)

    def _make_form_label(self, text: str, *, tooltip: str | None = None) -> QLabel:
        """Return a form label with an optional tooltip.

        Parameters
        ----------
        text : str
            Label text.
        tooltip : str, optional
            Tooltip shown when hovering the label.

        Returns
        -------
        QLabel
            Configured label widget.
        """
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
        """Create a show/hide-able row container for one advanced parameter.

        Parameters
        ----------
        layout : QFormLayout
            Parent form layout receiving the row.
        label : str
            Row-label text.
        widget : QWidget
            Input widget shown on the row.
        tooltip : str, optional
            Tooltip shown on the row label.

        Returns
        -------
        QWidget
            Container widget added to `layout`.
        """
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

        self._reference_time_label = QLabel("Reference volume")
        self._reference_time_spin = QSpinBox()
        self._reference_time_spin.setMinimum(0)
        self._reference_time_spin.setMaximumWidth(64)
        self._reference_time_label.setToolTip(
            "Volume index used as the registration target for within-scan motion correction."
        )
        operation_layout.addRow(self._reference_time_label, self._reference_time_spin)

        self._n_jobs_spin = QSpinBox()
        self._n_jobs_spin.setRange(-128, 128)
        self._n_jobs_spin.setSpecialValueText("auto")
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
        self._mesh_size_z_spin.setMaximumWidth(48)
        self._mesh_size_z_spin.setToolTip("B-spline mesh size along z.")
        self._mesh_size_y_spin = QSpinBox()
        self._mesh_size_y_spin.setRange(1, 512)
        self._mesh_size_y_spin.setMaximumWidth(48)
        self._mesh_size_y_spin.setToolTip("B-spline mesh size along y.")
        self._mesh_size_x_spin = QSpinBox()
        self._mesh_size_x_spin.setRange(1, 512)
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

        self._scale_combo = QComboBox()
        self._scale_combo.setMinimumContentsLength(10)
        self._scale_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._scale_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._scale_combo.addItem("decibel", "dB")
        self._scale_combo.addItem("square root", "sqrt")
        self._scale_combo.addItem("none", "off")
        self._scale_combo.setToolTip(
            "Optional intensity preprocessing applied before registration and used for registration preview layers."
        )
        params_layout.addRow(
            self._make_form_label(
                "Scale",
                tooltip="Optional intensity preprocessing applied before registration and used for registration preview layers.",
            ),
            self._scale_combo,
        )

        self._initialization_combo = QComboBox()
        self._initialization_combo.setMinimumContentsLength(18)
        self._initialization_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._initialization_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._initialization_combo.setToolTip(
            "How to initialize registration before optimization: center geometry, center moments, identity, or an existing affine transform from the Transforms tab."
        )
        params_layout.addRow(
            self._make_form_label(
                "Initialization",
                tooltip="How to initialize registration before optimization: center geometry, center moments, identity, or an existing affine transform from the Transforms tab.",
            ),
            self._initialization_combo,
        )

        learning_rate_row = QHBoxLayout()
        self._learning_rate_auto_check = QCheckBox("Auto")
        self._learning_rate_edit = ScientificDoubleSpinBox()
        self._learning_rate_edit.setRange(1e-10, 1e3)
        self._learning_rate_edit.setSingleStep(0.1)
        self._learning_rate_edit.setToolTip(
            "Optimizer step size. Accepts decimal (0.1) or scientific notation (1e-5)."
        )
        self._learning_rate_edit.setMaximumWidth(96)
        self._learning_rate_edit.setEnabled(False)
        self._learning_rate_auto_check.toggled.connect(
            lambda checked: self._learning_rate_edit.setEnabled(not checked)
        )
        learning_rate_row.addWidget(self._learning_rate_auto_check)
        learning_rate_row.addWidget(self._learning_rate_edit)
        learning_rate_row.addStretch(1)
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
        self._iterations_spin.setMaximumWidth(96)
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
        self._multi_resolution_row = self._make_advanced_row(
            advanced_layout,
            "Multi-resolution",
            self._multi_resolution_check,
            tooltip="Whether to optimize from coarse to fine resolution levels.",
        )

        self._shrink_factors_edit = QLineEdit()
        self._shrink_factors_edit.setToolTip(
            "Comma-separated shrink factors per resolution level for multi-resolution."
        )
        self._shrink_factors_row = self._make_advanced_row(
            advanced_layout,
            "Shrink factors",
            self._shrink_factors_edit,
            tooltip="Comma-separated downsampling factors for each multi-resolution level.",
        )

        self._smoothing_sigmas_edit = QLineEdit()
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
        self._fill_value_auto_check.setToolTip(
            "Automatically use the minimum intensity of the fixed image as fill value."
        )
        self._fill_value_spin = QDoubleSpinBox()
        self._fill_value_spin.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._fill_value_spin.setRange(-1e6, 1e6)
        self._fill_value_spin.setDecimals(3)
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
            "Save, load, and reapply affine transforms from registration results or manual napari layer transforms."
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
        self._transform_target_combo.setToolTip(
            "Layer to resample when applying the selected transform."
        )
        transforms_layout.addRow("Apply to", self._transform_target_combo)

        transform_buttons = QHBoxLayout()
        self._save_transform_btn = QPushButton("Save")
        self._save_transform_btn.clicked.connect(self._save_transform_callback)
        self._load_transform_btn = QPushButton("Load")
        self._load_transform_btn.clicked.connect(self._load_transform_callback)
        self._apply_transform_btn = QPushButton("Apply")
        self._apply_transform_btn.clicked.connect(self._apply_transform_callback)
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
        self._transform_source_combo.currentTextChanged.connect(
            self._validate_registration_selection
        )
        self._learning_rate_auto_check.toggled.connect(
            lambda checked: self._learning_rate_edit.setEnabled(not checked)
        )

        self._registration_parameters_by_operation = {
            "register_volume": get_default_registration_parameters(mode="volume"),
            "register_volumewise": get_default_registration_parameters(
                mode="volumewise"
            ),
        }
        set_registration_parameters(
            self,
            self._registration_parameters_by_operation["register_volume"],
            mode="volume",
        )

        self._refresh_layers()
        self._on_panel_changed()
        self._on_mode_changed()

    def _sync_manual_transform_event_connections(self) -> None:
        """Keep manual-transform refresh hooks in sync with viewer layers."""
        for layer in self._manual_transform_event_layers:
            try:
                layer.events.affine.disconnect(
                    self._refresh_transform_controls_callback
                )
            except (TypeError, RuntimeError):
                pass
        self._manual_transform_event_layers = []

        for layer in self.viewer.layers:
            if not _is_registration_source_layer(layer):
                continue
            _get_source_dataarray(layer)
            layer.events.affine.connect(self._refresh_transform_controls_callback)
            self._manual_transform_event_layers.append(layer)

    _refresh_layers = refresh_layers
    _get_layer_by_name = get_layer_by_name
    _selected_layer = selected_layer
    _current_scale_mode = current_scale_mode
    _current_metric = current_metric
    _current_resample_interpolation = current_resample_interpolation
    _current_transform_model = current_transform_model

    def _set_image_layer_data(self, layer: Image, data: npt.NDArray[Any]) -> None:
        """Assign image data despite the current napari stub mismatch.

        Parameters
        ----------
        layer : napari.layers.Image
            Image layer whose data should be replaced.
        data : numpy.ndarray
            Replacement array.

        Returns
        -------
        None
            Updates `layer` in place.
        """
        cast("Any", layer).data = data

    def _make_unique_layer_name(self, base_name: str) -> str:
        """Return a viewer-unique layer name based on `base_name`.

        Parameters
        ----------
        base_name : str
            Desired layer name.

        Returns
        -------
        str
            Unique layer name for the current viewer.
        """
        existing_names = {layer.name for layer in self.viewer.layers}
        if base_name not in existing_names:
            return base_name
        index = 1
        while True:
            candidate = f"{base_name} [{index}]"
            if candidate not in existing_names:
                return candidate
            index += 1

    def _make_unique_transform_name(self, base_name: str) -> str:
        """Return a viewer-unique transform payload name based on `base_name`.

        Parameters
        ----------
        base_name : str
            Desired transform name.

        Returns
        -------
        str
            Unique transform payload name for the current viewer.
        """
        existing_names = {
            payload["name"] for payload in get_available_transform_payloads(self)
        }
        if base_name not in existing_names:
            return base_name
        index = 1
        while True:
            candidate = f"{base_name} [{index}]"
            if candidate not in existing_names:
                return candidate
            index += 1

    def _volume_result_layer_name(
        self,
        moving_name: str,
        fixed_name: str,
        *,
        transform_model: str | None = None,
    ) -> str:
        """Return the napari layer name for between-scan registration output.

        Parameters
        ----------
        moving_name : str
            Moving-layer name. Unused, but kept for call-site clarity.
        fixed_name : str
            Fixed-layer name. Unused, but kept for call-site clarity.
        transform_model : str, optional
            Transform model to include in the result-layer label.

        Returns
        -------
        str
            Result-layer name.
        """
        del moving_name, fixed_name
        model = transform_model or self._transform_combo.currentText()
        return f"Registered ({model})"

    def _volume_fixed_preview_layer_name(self) -> str:
        """Return the napari layer name for the fixed preview layer."""
        return "Fixed"

    def _volume_moving_preview_layer_name(self) -> str:
        """Return the napari layer name for the moving preview layer."""
        return "Moving"

    def _volumewise_result_layer_name(self, moving_name: str) -> str:
        """Return the napari layer name for within-scan registration output.

        Parameters
        ----------
        moving_name : str
            Moving-layer name. Unused, but kept for call-site symmetry.

        Returns
        -------
        str
            Result-layer name.
        """
        del moving_name
        return "Motion corrected"

    def _volumewise_moving_preview_layer_name(self) -> str:
        """Return the napari layer name for the within-scan moving preview."""
        return "Moving"

    _update_reference_time_bounds = update_reference_time_bounds
    _set_layer_validation_style = set_layer_validation_style
    _set_run_btn_enabled = set_run_btn_enabled
    _validate_registration_selection = validate_registration_selection
    _on_moving_layer_changed = on_moving_layer_changed

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

    def _on_mode_changed(self) -> None:
        """Update the panel when the registration mode changes."""
        new_mode = self._operation()
        previous_mode = self._active_operation
        is_volumewise = new_mode == "register_volumewise"

        if previous_mode in self._registration_parameters_by_operation:
            self._registration_parameters_by_operation[previous_mode] = (
                get_registration_parameters(self)
            )

        self._fixed_label.setVisible(not is_volumewise)
        self._fixed_combo.setVisible(not is_volumewise)
        self._fixed_combo.setEnabled(not is_volumewise)
        self._reference_time_label.setVisible(is_volumewise)
        self._reference_time_spin.setVisible(is_volumewise)
        self._n_jobs_row.setVisible(is_volumewise)

        self._learning_rate_auto_check.setVisible(not is_volumewise)
        self._fill_value_row.setVisible(not is_volumewise)
        self._keep_diagnostics_row.setVisible(is_volumewise)

        set_registration_parameters(
            self,
            self._registration_parameters_by_operation[new_mode],
            mode="volumewise" if is_volumewise else "volume",
        )
        self._active_operation = new_mode

        self._update_reference_time_bounds()
        self._validate_registration_selection()

    def _begin_work(self) -> None:
        """Put the panel into its busy state."""
        self._run_btn.hide()
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

    def _end_work(self) -> None:
        """Restore the idle UI state after background work."""
        self._worker = None
        self._abort_event = None
        self._run_btn.show()
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
            moving = _get_source_dataarray(moving_layer)
        except Exception as exc:  # noqa: BLE001
            self._set_error(str(exc))
            return

        convergence_minimum_value = self._convergence_min_edit.value()

        # Parse advanced parameters
        shrink_factors = _parse_comma_separated_ints(self._shrink_factors_edit.text())
        smoothing_sigmas = _parse_comma_separated_ints(
            self._smoothing_sigmas_edit.text()
        )
        use_multi_res = self._multi_resolution_check.isChecked()
        if not use_multi_res:
            shrink_factors = None
            smoothing_sigmas = None

        metric = self._current_metric()
        scale_mode = self._current_scale_mode()
        resample_interpolation = self._current_resample_interpolation()
        transform = self._current_transform_model()
        initialization = cast(
            "InitializationSelection", self._initialization_combo.currentData()
        )
        self._abort_event = Event()

        if operation == "register_volume":
            if fixed_layer is None:
                self._set_error("Select a fixed layer.")
                return
            if fixed_layer is moving_layer:
                self._set_error("Moving and fixed layers must be different.")
                return

            try:
                fixed = _get_source_dataarray(fixed_layer)
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))
                return

            moving = _prepare_between_scan_data(moving)
            fixed = _prepare_between_scan_data(fixed)
            moving = _apply_registration_scale(moving, scale_mode)
            fixed = _apply_registration_scale(fixed, scale_mode)

            initial_transform: npt.NDArray[np.floating] | None = None
            try:
                initial_transform, initial_transform_source = (
                    get_selected_initial_transform(
                        self,
                        moving,
                        moving_layer=moving_layer,
                        fixed_layer=fixed_layer,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))
                return

            if transform not in {"translation", "rigid", "affine", "bspline"}:
                self._set_error(f"Unknown transform model: {transform!r}.")
                return
            volume_payload: VolumeRegistrationRunPayload = {
                "operation": "register_volume",
                "moving_layer_name": moving_layer.name,
                "transform": transform,
                "metric": metric,
                "scale": scale_mode,
                "learning_rate": learning_rate,
                "number_of_iterations": self._iterations_spin.value(),
                "use_multi_resolution": use_multi_res,
                "resample_interpolation": resample_interpolation,
                "number_of_histogram_bins": self._histogram_bins_spin.value(),
                "convergence_minimum_value": convergence_minimum_value,
                "convergence_window_size": self._convergence_window_spin.value(),
                "initialization": initialization,
                "shrink_factors": shrink_factors,
                "smoothing_sigmas": smoothing_sigmas,
                "keep_diagnostics": self._keep_diagnostics_check.isChecked(),
                "fill_value": None
                if self._fill_value_auto_check.isChecked()
                else self._fill_value_spin.value(),
                "mesh_size": (
                    self._mesh_size_z_spin.value(),
                    self._mesh_size_y_spin.value(),
                    self._mesh_size_x_spin.value(),
                ),
                "fixed_layer_name": fixed_layer.name,
            }
            if initial_transform_source is not None:
                volume_payload["initial_transform_source"] = initial_transform_source

            initialization_arg = (
                initial_transform
                if initial_transform is not None
                else get_selected_center_initialization(self)
            )

            try:
                progress_plotter = create_volume_progress_plotter(
                    self,
                    moving_layer=cast("Image", moving_layer),
                    fixed_layer=cast("Image", fixed_layer),
                    moving=moving,
                    fixed=fixed,
                    layer_name=self._make_unique_layer_name(
                        self._volume_result_layer_name(
                            volume_payload["moving_layer_name"],
                            volume_payload["fixed_layer_name"],
                            transform_model=volume_payload["transform"],
                        )
                    ),
                    initial_transform=initial_transform,
                    scale_mode=volume_payload["scale"],
                )
            except Exception:
                return

            worker = thread_worker(register_volume)(
                moving,
                fixed,
                transform_type=volume_payload["transform"],
                metric=volume_payload["metric"],
                learning_rate=learning_rate,
                number_of_iterations=volume_payload["number_of_iterations"],
                use_multi_resolution=volume_payload["use_multi_resolution"],
                resample=True,
                resample_interpolation=volume_payload["resample_interpolation"],
                mesh_size=volume_payload["mesh_size"],
                number_of_histogram_bins=volume_payload["number_of_histogram_bins"],
                convergence_minimum_value=volume_payload["convergence_minimum_value"],
                convergence_window_size=volume_payload["convergence_window_size"],
                initialization=initialization_arg,
                shrink_factors=volume_payload["shrink_factors"] or (6, 2, 1),
                smoothing_sigmas=volume_payload["smoothing_sigmas"] or (6, 2, 1),
                fill_value=volume_payload["fill_value"],
                show_progress=True,
                progress_plotter=progress_plotter,
                abort_event=self._abort_event,
            )
            self._worker = worker
            self._begin_work()
            worker.returned.connect(
                lambda result: on_volume_registration_finished(
                    self, volume_payload, result
                )
            )
        else:
            if TIME_DIM not in moving.dims:
                self._set_error(
                    "register_volumewise requires a layer with a time dimension."
                )
                return
            if transform == "bspline":
                self._set_error(f"Unknown transform model: {transform!r}.")
                return

            volumewise_payload: VolumewiseRegistrationRunPayload = {
                "operation": "register_volumewise",
                "moving_layer_name": moving_layer.name,
                "transform": transform,
                "metric": metric,
                "scale": scale_mode,
                "learning_rate": learning_rate,
                "number_of_iterations": self._iterations_spin.value(),
                "use_multi_resolution": use_multi_res,
                "resample_interpolation": resample_interpolation,
                "number_of_histogram_bins": self._histogram_bins_spin.value(),
                "convergence_minimum_value": convergence_minimum_value,
                "convergence_window_size": self._convergence_window_spin.value(),
                "initialization": initialization,
                "shrink_factors": shrink_factors,
                "smoothing_sigmas": smoothing_sigmas,
                "keep_diagnostics": self._keep_diagnostics_check.isChecked(),
                "fill_value": None
                if self._fill_value_auto_check.isChecked()
                else self._fill_value_spin.value(),
                "mesh_size": (
                    self._mesh_size_z_spin.value(),
                    self._mesh_size_y_spin.value(),
                    self._mesh_size_x_spin.value(),
                ),
                "reference_time": self._reference_time_spin.value(),
                "n_jobs": self._n_jobs_spin.value(),
            }
            moving = _apply_registration_scale(moving, volumewise_payload["scale"])

            progress_reporter = setup_volumewise_progress(
                self,
                moving_layer=cast("Image", moving_layer),
                moving=moving,
                layer_name=self._make_unique_layer_name(
                    self._volumewise_result_layer_name(
                        volumewise_payload["moving_layer_name"]
                    )
                ),
                scale_mode=volumewise_payload["scale"],
            )

            worker = thread_worker(register_volumewise)(
                moving,
                reference_time=volumewise_payload["reference_time"],
                n_jobs=volumewise_payload["n_jobs"],
                transform=volumewise_payload["transform"],
                metric=volumewise_payload["metric"],
                learning_rate=learning_rate,
                number_of_iterations=volumewise_payload["number_of_iterations"],
                use_multi_resolution=volumewise_payload["use_multi_resolution"],
                resample_interpolation=volumewise_payload["resample_interpolation"],
                number_of_histogram_bins=volumewise_payload["number_of_histogram_bins"],
                convergence_minimum_value=volumewise_payload[
                    "convergence_minimum_value"
                ],
                convergence_window_size=volumewise_payload["convergence_window_size"],
                initialization=get_selected_center_initialization(self),
                shrink_factors=volumewise_payload["shrink_factors"] or (6, 2, 1),
                smoothing_sigmas=volumewise_payload["smoothing_sigmas"] or (6, 2, 1),
                keep_diagnostics=volumewise_payload["keep_diagnostics"],
                show_progress=False,
                abort_event=self._abort_event,
                progress_reporter=progress_reporter,
            )
            self._worker = worker
            self._begin_work()
            worker.returned.connect(
                lambda result: on_volumewise_registration_finished(
                    self, volumewise_payload, result
                )
            )
        worker.errored.connect(lambda exc: on_registration_failed(self, exc))
        worker.finished.connect(self._end_work)
        worker.start()
