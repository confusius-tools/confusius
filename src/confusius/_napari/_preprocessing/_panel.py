"""Preprocessing control panel for the ConfUSIus sidebar."""

from __future__ import annotations

from typing import Literal, TypeVar, cast

import napari
import numpy as np
import xarray as xr
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from confusius._dims import TIME_DIM
from confusius._napari._signals._store import LiveSignal, SignalStore
from confusius._napari._utils import extract_label_mean_trace, extract_voxel_trace
from confusius._utils.coordinates import get_representative_step
from confusius.plotting.napari import plot_napari
from confusius.signal import clean
from confusius.spatial import smooth_volume
from confusius.timing import resample_time, resample_to_uniform_time

_SPIN_WIDTH = 70
"""Maximum width, in pixels, for compact numeric spin boxes (small integers)."""

_SPIN_WIDTH_WIDE = 92
"""Maximum width, in pixels, for spin boxes showing decimals (cutoffs, tolerance,
pad length) — wide enough to read the value while typing without forcing the
sidebar wider than the napari dock.
"""

_COMBO_CHARS_NARROW = 8
_COMBO_CHARS_MEDIUM = 12
_COMBO_CHARS_WIDE = 16
"""Minimum-contents-length presets for `_narrow_combo`, in characters.

Paired with `QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon`, this
caps a combo box's width to roughly this many characters regardless of how long its
longest item's text is (e.g. "Match reference layer", arbitrary layer/signal names).
The popup itself still shows the full text; only the closed box is capped.
"""

_STANDARDIZE_LABELS: dict[str, str | None] = {
    "None": None,
    "Z-score": "zscore",
    "Percent signal change": "psc",
}

_INTERPOLATE_METHODS = (
    "linear",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "quintic",
    "pchip",
    "barycentric",
    "krogh",
    "akima",
    "makima",
)

_RESAMPLE_METHODS = (
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "previous",
    "next",
)
_ResampleMethod = Literal[
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "previous",
    "next",
]

_NO_SIGNAL = "None"
_NO_RESAMPLING = "No resampling"
_UNIFORM_GRID = "Uniform grid"
_MATCH_REFERENCE = "Match reference layer"

_SpinBoxT = TypeVar("_SpinBoxT", bound=QAbstractSpinBox)

# A raw, unaligned (x, y) signal pulled from either an imported CSV column or a live
# napari source (point/label trace). Alignment onto the pipeline's actual time grid is
# deferred to the background thread, since that grid is only known once resampling
# (which may change its length) has run.
_RawSignal = tuple["np.ndarray | None", "np.ndarray"]


@thread_worker
def _run_pipeline(
    signals: xr.DataArray,
    resample_spec: tuple[str, dict] | None,
    smooth_kwargs: dict | None,
    clean_kwargs: dict,
    confounds_raw: _RawSignal | None,
    mask_raw: _RawSignal | None,
) -> xr.DataArray:
    """Run temporal resampling, spatial smoothing, and signal cleaning.

    Runs in a background thread, in that order, so temporal filtering (which
    requires uniformly-sampled time) always sees a regular grid when resampling is
    requested.

    Parameters
    ----------
    signals : xarray.DataArray
        Source signals to process.
    resample_spec : tuple[str, dict] or None
        `(mode, kwargs)` pair, where `mode` is `"Uniform grid"` (kwargs forwarded to
        `confusius.timing.resample_to_uniform_time`) or anything else (`kwargs`
        forwarded to `confusius.timing.resample_time`, expected to contain
        `new_time` and `method`). If not provided, no resampling is applied.
    smooth_kwargs : dict, optional
        Keyword arguments forwarded to `confusius.spatial.smooth_volume`. If not
        provided, no smoothing is applied.
    clean_kwargs : dict
        Keyword arguments forwarded to `confusius.signal.clean`. Mutated in place to
        add `confounds`/`sample_mask` when `confounds_raw`/`mask_raw` are provided.
    confounds_raw : tuple[numpy.ndarray or None, numpy.ndarray], optional
        Raw `(x, y)` confound series to align onto the post-resampling time grid.
        If not provided, no confound regression is applied.
    mask_raw : tuple[numpy.ndarray or None, numpy.ndarray], optional
        Raw `(x, y)` series to threshold into a sample mask and align onto the
        post-resampling time grid. If not provided, no scrubbing is applied.

    Returns
    -------
    xarray.DataArray
        Fully processed signals.
    """
    if resample_spec is not None:
        mode, kwargs = resample_spec
        if mode == _UNIFORM_GRID:
            signals = resample_to_uniform_time(signals, **kwargs)
        else:
            method = cast(_ResampleMethod, kwargs["method"])
            signals = resample_time(signals, kwargs["new_time"], method=method)

    if smooth_kwargs is not None:
        signals = smooth_volume(signals, **smooth_kwargs)

    if confounds_raw is not None:
        x, y = confounds_raw
        clean_kwargs["confounds"] = _align_series(x, y, signals, as_mask=False)
    if mask_raw is not None:
        x, y = mask_raw
        clean_kwargs["sample_mask"] = _align_series(x, y, signals, as_mask=True)

    return clean(signals, **clean_kwargs)


def _align_series(
    x: np.ndarray | None, y: np.ndarray, reference: xr.DataArray, *, as_mask: bool
) -> xr.DataArray:
    """Align a raw `(x, y)` series onto a reference DataArray's `time` grid.

    Parameters
    ----------
    x : numpy.ndarray or None
        Time values for `y`. If not provided, `y` is assumed to already share
        `reference`'s time axis positionally (its length must match).
    y : numpy.ndarray
        Series values.
    reference : xarray.DataArray
        DataArray whose `time` dimension/coordinate the series is aligned to.
    as_mask : bool
        Whether to threshold the aligned values into a boolean mask (values > 0.5
        are `True`) instead of returning them as float regressors.

    Returns
    -------
    xarray.DataArray
        1D DataArray with a `time` dimension matching `reference`.

    Raises
    ------
    ValueError
        If interpolation is not possible (no time coordinate on `reference`, or no
        `x` values for `y`) and `y`'s length does not match the number of
        timepoints in `reference`.
    """
    n_timepoints = reference.sizes[TIME_DIM]
    time_coord = (
        reference.coords[TIME_DIM].values if TIME_DIM in reference.coords else None
    )

    if time_coord is not None and x is not None:
        source = xr.DataArray(y, dims=TIME_DIM, coords={TIME_DIM: x})
        values = source.interp({TIME_DIM: time_coord}).values
    else:
        if len(y) != n_timepoints:
            raise ValueError(
                f"Signal has {len(y)} samples, but the source has {n_timepoints} "
                "timepoints and no shared 'time' coordinate to align by."
            )
        values = np.asarray(y, dtype=float)

    if as_mask:
        values = values > 0.5

    coords = {TIME_DIM: time_coord} if time_coord is not None else None
    return xr.DataArray(values, dims=TIME_DIM, coords=coords)


def _capped_spin(spin: _SpinBoxT, width: int = _SPIN_WIDTH) -> _SpinBoxT:
    """Cap a spin box's width so it cannot force the sidebar wider than the dock."""
    spin.setMaximumWidth(width)
    return spin


def _narrow_combo(combo: QComboBox, chars: int = _COMBO_CHARS_MEDIUM) -> QComboBox:
    """Cap a combo box's width to `chars` characters, regardless of item length.

    Without this, a combo box grows to fit its longest item (e.g. "Match reference
    layer", or an arbitrary layer/signal name), which forces the sidebar wider than
    the napari dock. The dropdown popup itself is unaffected and still shows full
    item text.
    """
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(chars)
    return combo


def _form_label(text: str, *, tooltip: str | None = None) -> QLabel:
    """Return a `QFormLayout` row label with its own tooltip.

    Building the label explicitly (rather than passing a bare string to
    `QFormLayout.addRow`) lets the label carry its own tooltip. Without one, Qt
    forwards unhandled tooltip events to the parent widget, which is why leaving a
    row's label/field without a tooltip previously surfaced the enclosing
    QGroupBox's tooltip instead.
    """
    label = QLabel(text)
    if tooltip is not None:
        label.setToolTip(tooltip)
    return label


def _wrap_form(form: QFormLayout) -> QFormLayout:
    """Apply the narrow-dock-safe wrap policy used throughout the registration panel.

    Wrapping long rows (label above field, rather than side by side) keeps a form
    row's minimum width to the widest of the label or the field alone, instead of
    their sum — the same fix used for the registration panel's overflow (see PR
    #216 and the signals panel's issue #183).
    """
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    return form


class PreprocessingPanel(QWidget):
    """Right-side panel for resampling, smoothing, and cleaning fUSI signals.

    A single "Apply" runs, in order: temporal resampling
    (`confusius.timing.resample_time` / `confusius.timing.resample_to_uniform_time`,
    unless "No resampling" is selected), spatial smoothing
    (`confusius.spatial.smooth_volume`, when enabled), and signal cleaning
    (`confusius.signal.clean`, always). The pipeline runs in a background thread and
    adds its result to the viewer as a new image layer named
    `"{source layer} — cleaned"`.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    signal_store : SignalStore
        Shared store of imported and live signals, used to populate the confounds
        and sample mask dropdowns. Pass the same instance used by the Signals panel
        so its imported and live (point/label) signals are available here too.
    """

    def __init__(self, viewer: napari.Viewer, signal_store: SignalStore) -> None:
        super().__init__()
        self._viewer = viewer
        self._signal_store = signal_store
        self._setup_ui()

        viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        viewer.layers.events.removed.connect(self._refresh_layer_combos)
        signal_store.changed.connect(self._refresh_signal_combos)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._make_source_group())
        layout.addWidget(self._make_resample_group())
        layout.addWidget(self._make_smooth_group())
        layout.addWidget(self._make_detrend_group())
        layout.addWidget(self._make_filter_group())
        layout.addWidget(self._make_standardize_group())
        layout.addWidget(self._make_confounds_group())
        layout.addWidget(self._make_scrub_group())

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setObjectName("primary_btn")
        self._apply_btn.clicked.connect(self._apply)
        layout.addWidget(self._apply_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumHeight(4)
        self._progress.hide()
        layout.addWidget(self._progress)

        layout.addStretch()

        self._refresh_layer_combos()
        self._refresh_signal_combos()

    def _make_source_group(self) -> QGroupBox:
        group = QGroupBox("Source")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)
        self._source_combo = _narrow_combo(QComboBox(), _COMBO_CHARS_WIDE)
        self._source_combo.setToolTip(
            "Image layer to process. Only layers with a 'time' dimension are listed."
        )
        self._source_combo.currentTextChanged.connect(self._prefill_resample_defaults)
        group_layout.addWidget(self._source_combo)
        return group

    def _make_resample_group(self) -> QGroupBox:
        group = QGroupBox("Temporal Resampling")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        self._resample_mode_combo = _narrow_combo(QComboBox(), _COMBO_CHARS_WIDE)
        self._resample_mode_combo.addItems(
            [_NO_RESAMPLING, _UNIFORM_GRID, _MATCH_REFERENCE]
        )
        self._resample_mode_combo.currentTextChanged.connect(
            self._on_resample_mode_changed
        )
        group_layout.addWidget(self._resample_mode_combo)

        # Uniform grid: step only. Start/stop always default to the recording's own
        # first/last timepoint (resample_to_uniform_time's own default).
        self._uniform_widget = QWidget()
        uniform_row = QHBoxLayout(self._uniform_widget)
        uniform_row.setContentsMargins(0, 0, 0, 0)
        uniform_row.setSpacing(4)
        uniform_row.addWidget(_form_label("Step"))
        self._resample_step_spin = _capped_spin(QDoubleSpinBox(), _SPIN_WIDTH_WIDE)
        self._resample_step_spin.setRange(1e-6, 1e6)
        self._resample_step_spin.setDecimals(4)
        self._resample_step_spin.setValue(0.5)
        self._resample_step_spin.setToolTip(
            "Uniform time step, in seconds. Prefilled with the source layer's "
            "median temporal spacing."
        )
        uniform_row.addWidget(self._resample_step_spin)
        uniform_row.addStretch()
        group_layout.addWidget(self._uniform_widget)
        self._uniform_widget.hide()

        # Match reference: borrow another layer's time coordinate as the new grid.
        self._reference_widget = QWidget()
        reference_form = _wrap_form(QFormLayout(self._reference_widget))
        reference_form.setContentsMargins(0, 0, 0, 0)
        reference_form.setSpacing(4)
        self._resample_reference_combo = _narrow_combo(QComboBox(), _COMBO_CHARS_WIDE)
        self._resample_reference_combo.setToolTip(
            "Layer whose time coordinate is used as the new time grid."
        )
        reference_form.addRow(_form_label("Reference"), self._resample_reference_combo)
        group_layout.addWidget(self._reference_widget)
        self._reference_widget.hide()

        self._resample_method_widget = QWidget()
        method_row = QHBoxLayout(self._resample_method_widget)
        method_row.setContentsMargins(0, 0, 0, 0)
        method_row.setSpacing(4)
        method_row.addWidget(_form_label("Method"))
        self._resample_method_combo = _narrow_combo(QComboBox(), _COMBO_CHARS_MEDIUM)
        self._resample_method_combo.addItems(list(_RESAMPLE_METHODS))
        method_row.addWidget(self._resample_method_combo, stretch=1)
        group_layout.addWidget(self._resample_method_widget)
        self._resample_method_widget.hide()

        return group

    def _make_smooth_group(self) -> QGroupBox:
        group = QGroupBox("Spatial Smoothing")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        fwhm_row = QHBoxLayout()
        self._smooth_enable_check = QCheckBox("FWHM")
        self._smooth_enable_check.setToolTip(
            "Whether to apply isotropic Gaussian spatial smoothing before cleaning."
        )
        self._smooth_fwhm_spin = _capped_spin(QDoubleSpinBox(), _SPIN_WIDTH_WIDE)
        self._smooth_fwhm_spin.setRange(0.0, 100.0)
        self._smooth_fwhm_spin.setDecimals(3)
        self._smooth_fwhm_spin.setValue(0.3)
        self._smooth_fwhm_spin.setEnabled(False)
        self._smooth_fwhm_spin.setToolTip(
            "Full width at half maximum of an isotropic Gaussian kernel, in the "
            "source layer's physical (spatial coordinate) units."
        )
        self._smooth_enable_check.toggled.connect(self._smooth_fwhm_spin.setEnabled)
        fwhm_row.addWidget(self._smooth_enable_check)
        fwhm_row.addWidget(self._smooth_fwhm_spin)
        fwhm_row.addStretch()
        group_layout.addLayout(fwhm_row)

        self._smooth_ensure_finite_check = QCheckBox("Interpolate Inf/Nan")
        self._smooth_ensure_finite_check.setToolTip(
            "Replace non-finite values with zero before filtering, so they don't "
            "spread to neighbouring voxels through the Gaussian kernel."
        )
        group_layout.addWidget(self._smooth_ensure_finite_check)
        return group

    def _make_detrend_group(self) -> QGroupBox:
        group = QGroupBox("Detrending")
        group_layout = QHBoxLayout(group)
        self._detrend_check = QCheckBox("Order")
        self._detrend_check.setToolTip(
            "Whether to remove a polynomial trend before filtering/standardizing."
        )
        self._detrend_order_spin = _capped_spin(QSpinBox())
        self._detrend_order_spin.setRange(0, 10)
        self._detrend_order_spin.setValue(1)
        self._detrend_order_spin.setEnabled(False)
        self._detrend_order_spin.setToolTip(
            "0: remove mean. 1: remove linear trend (default). 2+: polynomial trend."
        )
        self._detrend_check.toggled.connect(self._detrend_order_spin.setEnabled)
        group_layout.addWidget(self._detrend_check)
        group_layout.addWidget(self._detrend_order_spin)
        group_layout.addStretch()
        return group

    def _make_filter_group(self) -> QGroupBox:
        group = QGroupBox("Temporal Filter")
        group.setToolTip("Zero-phase Butterworth low-/high-/band-pass filter.")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        cutoff_form = _wrap_form(QFormLayout())
        cutoff_form.setSpacing(4)

        self._low_cutoff_check = QCheckBox("High-pass (Hz)")
        self._low_cutoff_check.setToolTip(
            "Attenuates frequencies below this cutoff (low_cutoff)."
        )
        self._low_cutoff_spin = _capped_spin(QDoubleSpinBox(), _SPIN_WIDTH_WIDE)
        self._low_cutoff_spin.setRange(0.0, 100.0)
        self._low_cutoff_spin.setDecimals(4)
        self._low_cutoff_spin.setSingleStep(0.01)
        self._low_cutoff_spin.setValue(0.01)
        self._low_cutoff_spin.setEnabled(False)
        self._low_cutoff_spin.setToolTip(
            "Attenuates frequencies below this cutoff (low_cutoff)."
        )
        self._low_cutoff_check.toggled.connect(self._low_cutoff_spin.setEnabled)
        cutoff_form.addRow(self._low_cutoff_check, self._low_cutoff_spin)

        self._high_cutoff_check = QCheckBox("Low-pass (Hz)")
        self._high_cutoff_check.setToolTip(
            "Attenuates frequencies above this cutoff (high_cutoff)."
        )
        self._high_cutoff_spin = _capped_spin(QDoubleSpinBox(), _SPIN_WIDTH_WIDE)
        self._high_cutoff_spin.setRange(0.0, 100.0)
        self._high_cutoff_spin.setDecimals(4)
        self._high_cutoff_spin.setSingleStep(0.01)
        self._high_cutoff_spin.setValue(1.0)
        self._high_cutoff_spin.setEnabled(False)
        self._high_cutoff_spin.setToolTip(
            "Attenuates frequencies above this cutoff (high_cutoff)."
        )
        self._high_cutoff_check.toggled.connect(self._high_cutoff_spin.setEnabled)
        cutoff_form.addRow(self._high_cutoff_check, self._high_cutoff_spin)
        group_layout.addLayout(cutoff_form)

        # Foldable advanced section, matching the registration panel's pattern
        # (PR #216): a checkable QToolButton with an arrow indicator toggles a
        # QFormLayout indented underneath, collapsed by default.
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
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        advanced_header_layout.addWidget(self._advanced_toggle)
        advanced_header_layout.addStretch(1)
        group_layout.addWidget(advanced_header)

        # Every row below sets its own tooltip on both the label and the field:
        # Qt forwards an unhandled ToolTip event to the parent widget, so a
        # tooltip-less child here would otherwise show this group's tooltip
        # ("Zero-phase Butterworth...") instead of its own.
        self._advanced_widget = QWidget()
        advanced_form = _wrap_form(QFormLayout(self._advanced_widget))
        advanced_form.setContentsMargins(9, 6, 0, 0)
        advanced_form.setSpacing(4)

        self._filter_order_spin = _capped_spin(QSpinBox())
        self._filter_order_spin.setRange(1, 20)
        self._filter_order_spin.setValue(5)
        order_tooltip = (
            "Filter order. Higher orders give steeper roll-off but may be less stable."
        )
        self._filter_order_spin.setToolTip(order_tooltip)
        advanced_form.addRow(
            _form_label("Order", tooltip=order_tooltip), self._filter_order_spin
        )

        self._padtype_combo = _narrow_combo(QComboBox(), _COMBO_CHARS_NARROW)
        self._padtype_combo.addItems(["odd", "even", "constant", "None"])
        padtype_tooltip = (
            "Type of padding applied at each end before filtering, to reduce edge "
            "effects. 'None' disables padding."
        )
        self._padtype_combo.setToolTip(padtype_tooltip)
        advanced_form.addRow(
            _form_label("Pad type", tooltip=padtype_tooltip), self._padtype_combo
        )

        self._padlen_spin = _capped_spin(QSpinBox(), _SPIN_WIDTH_WIDE)
        self._padlen_spin.setRange(0, 10000)
        self._padlen_spin.setSpecialValueText("Auto")
        padlen_tooltip = (
            "Number of samples to pad at each end. Auto uses scipy's default "
            "padding length."
        )
        self._padlen_spin.setToolTip(padlen_tooltip)
        advanced_form.addRow(
            _form_label("Pad length", tooltip=padlen_tooltip), self._padlen_spin
        )

        self._uniformity_tolerance_spin = _capped_spin(
            QDoubleSpinBox(), _SPIN_WIDTH_WIDE
        )
        self._uniformity_tolerance_spin.setRange(1e-6, 1.0)
        self._uniformity_tolerance_spin.setDecimals(6)
        self._uniformity_tolerance_spin.setSingleStep(0.001)
        self._uniformity_tolerance_spin.setValue(1e-2)
        tolerance_tooltip = (
            "Maximum allowed relative deviation between consecutive time "
            "intervals. Increase to tolerate slight timestamp jitter (e.g. from "
            "acquisition clocks or dropped volumes)."
        )
        self._uniformity_tolerance_spin.setToolTip(tolerance_tooltip)
        advanced_form.addRow(
            _form_label("Timing tolerance", tooltip=tolerance_tooltip),
            self._uniformity_tolerance_spin,
        )

        group_layout.addWidget(self._advanced_widget)
        self._advanced_widget.hide()

        return group

    def _make_standardize_group(self) -> QGroupBox:
        group = QGroupBox("Standardization")
        group_layout = QVBoxLayout(group)
        self._standardize_combo = _narrow_combo(QComboBox(), _COMBO_CHARS_WIDE)
        self._standardize_combo.addItems(list(_STANDARDIZE_LABELS))
        group_layout.addWidget(self._standardize_combo)
        return group

    def _make_confounds_group(self) -> QGroupBox:
        group = QGroupBox("Confound Regression")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        form = _wrap_form(QFormLayout())
        form.setSpacing(4)
        self._confounds_combo = _narrow_combo(QComboBox(), _COMBO_CHARS_MEDIUM)
        confound_tooltip = (
            "Imported or live (point/label) signal to regress out as a nuisance "
            "confound."
        )
        self._confounds_combo.setToolTip(confound_tooltip)
        form.addRow(
            _form_label("Confound", tooltip=confound_tooltip), self._confounds_combo
        )
        group_layout.addLayout(form)
        return group

    def _make_scrub_group(self) -> QGroupBox:
        group = QGroupBox("Scrubbing")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        form = _wrap_form(QFormLayout())
        form.setSpacing(4)
        self._mask_combo = _narrow_combo(QComboBox(), _COMBO_CHARS_MEDIUM)
        mask_tooltip = (
            "Imported or live (point/label) signal thresholded at 0.5 into a "
            "keep/censor mask (values > 0.5 are kept)."
        )
        self._mask_combo.setToolTip(mask_tooltip)
        form.addRow(_form_label("Sample mask", tooltip=mask_tooltip), self._mask_combo)

        self._interpolate_method_combo = _narrow_combo(QComboBox(), _COMBO_CHARS_NARROW)
        self._interpolate_method_combo.addItems(list(_INTERPOLATE_METHODS))
        interp_tooltip = (
            "Interpolation method used to fill censored samples before "
            "detrending/filtering."
        )
        self._interpolate_method_combo.setToolTip(interp_tooltip)
        form.addRow(
            _form_label("Interpolation", tooltip=interp_tooltip),
            self._interpolate_method_combo,
        )
        group_layout.addLayout(form)

        self._ensure_finite_check = QCheckBox("Interpolate Inf/Nan")
        self._ensure_finite_check.setToolTip(
            "Repair non-finite values in the signals (and confounds) by "
            "interpolating along time before cleaning."
        )
        group_layout.addWidget(self._ensure_finite_check)
        return group

    # ------------------------------------------------------------------
    # Layer / signal list helpers
    # ------------------------------------------------------------------

    def _eligible_source_layers(self) -> list[str]:
        """Return names of image layers with a multi-element `time` dimension."""
        names = []
        for layer in self._viewer.layers:
            if layer._type_string != "image":
                continue
            da = layer.metadata.get("xarray")
            if da is not None and TIME_DIM in da.dims and da.sizes[TIME_DIM] > 1:
                names.append(layer.name)
        return names

    def _refresh_layer_combos(self, _event=None) -> None:
        """Repopulate the source and resample-reference combos from viewer layers."""
        names = self._eligible_source_layers()
        for combo in (self._source_combo, self._resample_reference_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(names)
                index = combo.findText(current)
                if index >= 0:
                    combo.setCurrentIndex(index)
            finally:
                combo.blockSignals(False)
        self._prefill_resample_defaults()

    def _refresh_signal_combos(self) -> None:
        """Repopulate the confounds/sample-mask combos from imported and live signals."""
        for combo in (self._confounds_combo, self._mask_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItem(_NO_SIGNAL)
                for signal in self._signal_store.imported_signals():
                    combo.addItem(signal.name)
                for live in self._signal_store.live_signals():
                    if live.source_type != "mouse":
                        combo.addItem(live.name)
                index = combo.findText(current)
                combo.setCurrentIndex(index if index >= 0 else 0)
            finally:
                combo.blockSignals(False)

    def _prefill_resample_defaults(self, _text: str = "") -> None:
        """Prefill the uniform-grid step from the source layer's median spacing."""
        layer_name = self._source_combo.currentText()
        if not layer_name:
            return
        try:
            layer = self._viewer.layers[layer_name]
        except KeyError:
            return

        da = layer.metadata.get("xarray")
        if da is None or TIME_DIM not in da.coords:
            return

        time_values = np.asarray(da.coords[TIME_DIM].values)
        if len(time_values) < 2:
            return

        step, _approximate = get_representative_step(time_values)
        if step is not None:
            self._resample_step_spin.setValue(float(step))

    def _on_resample_mode_changed(self, mode: str) -> None:
        self._uniform_widget.setVisible(mode == _UNIFORM_GRID)
        self._reference_widget.setVisible(mode == _MATCH_REFERENCE)
        self._resample_method_widget.setVisible(mode != _NO_RESAMPLING)

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._advanced_widget.setVisible(checked)
        self._advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    # ------------------------------------------------------------------
    # Busy helpers
    # ------------------------------------------------------------------

    def _begin_work(self) -> None:
        self._apply_btn.setEnabled(False)
        self._apply_btn.setText("Working…")
        self._progress.show()

    def _end_work(self) -> None:
        self._apply_btn.setEnabled(True)
        self._apply_btn.setText("Apply")
        self._progress.hide()

    # ------------------------------------------------------------------
    # Source lookup
    # ------------------------------------------------------------------

    def _get_source(
        self, combo: QComboBox, *, label: str = "source"
    ) -> tuple[str, xr.DataArray] | None:
        """Look up the layer selected in `combo` and its DataArray metadata.

        Parameters
        ----------
        combo : QComboBox
            Combo box to read the layer name from.
        label : str, default: "source"
            Noun used in the "No {label} layer selected." error message.

        Returns
        -------
        tuple[str, xarray.DataArray] or None
            `(layer_name, dataarray)`, or `None` (after calling `show_error`) if no
            layer is selected or it carries no DataArray metadata.
        """
        layer_name = combo.currentText()
        if not layer_name:
            show_error(f"No {label} layer selected.")
            return None

        layer = self._viewer.layers[layer_name]
        da = layer.metadata.get("xarray")
        if da is None:
            show_error(
                "Selected layer has no DataArray metadata. "
                "Load the file using the Data panel or the File menu."
            )
            return None
        return layer_name, da

    # ------------------------------------------------------------------
    # Confound / sample-mask resolution
    # ------------------------------------------------------------------

    def _extract_live_series(self, live: LiveSignal, image_layer) -> _RawSignal | None:
        """Re-extract a live signal's raw trace against `image_layer`.

        Parameters
        ----------
        live : LiveSignal
            Live signal to extract. Must be of type `"point"` or `"label"`.
        image_layer : napari.layers.Image
            Image layer to pull voxel intensities from (the Preprocessing panel's
            selected source layer, not necessarily the layer the signal was
            plotted against in the Signals panel).

        Returns
        -------
        tuple[numpy.ndarray or None, numpy.ndarray] or None
            `(time_values_or_none, trace)`, or `None` if the backing Points/Labels
            layer no longer exists, the point index is out of range, or no voxel
            matches the label.
        """
        if live.layer_name is None:
            return None
        try:
            spatial_layer = self._viewer.layers[live.layer_name]
        except KeyError:
            return None

        da = image_layer.metadata.get("xarray")
        xaxis_index = (
            list(da.dims).index(TIME_DIM)
            if da is not None and TIME_DIM in da.dims
            else 0
        )
        x = (
            np.asarray(da.coords[TIME_DIM].values)
            if da is not None and TIME_DIM in da.coords
            else None
        )

        if live.source_type == "point":
            point_index = live.source_id
            if point_index is None or point_index >= len(spatial_layer.data):
                return None
            pt_data = np.asarray(spatial_layer.data[point_index], dtype=float)
            n_pt = len(pt_data)
            scale = np.asarray(spatial_layer.scale, dtype=float)[-n_pt:]
            translate = np.asarray(spatial_layer.translate, dtype=float)[-n_pt:]
            pt_world = pt_data * scale + translate
            img_ndim = image_layer.data.ndim
            if n_pt < img_ndim:
                padded = np.zeros(img_ndim)
                padded[-n_pt:] = pt_world
                pt_world = padded
            y = extract_voxel_trace(image_layer, pt_world, xaxis_index)
        elif live.source_type == "label":
            label_id = live.source_id
            if label_id is None:
                return None
            y = extract_label_mean_trace(
                image_layer, np.asarray(spatial_layer.data), label_id, xaxis_index
            )
        else:
            return None

        if y is None:
            return None
        return x, y

    def _resolve_raw_signal(self, name: str, source_layer) -> _RawSignal | None:
        """Look up a confound/mask signal by name and return its raw `(x, y)` trace.

        Parameters
        ----------
        name : str
            Combo box selection. `"None"` (or empty) resolves to `None`.
        source_layer : napari.layers.Image
            Currently selected source layer, used as the reference image when the
            named signal is a live point/label signal.

        Returns
        -------
        tuple[numpy.ndarray or None, numpy.ndarray] or None
            Raw, unaligned `(x, y)` signal, or `None` if no signal was selected.

        Raises
        ------
        ValueError
            If the named signal cannot be found, or a live signal's trace could
            not be re-extracted.
        """
        if not name or name == _NO_SIGNAL:
            return None

        for signal in self._signal_store.imported_signals():
            if signal.name == name:
                return signal.x, signal.y

        for live in self._signal_store.live_signals():
            if live.name == name and live.source_type != "mouse":
                extracted = self._extract_live_series(live, source_layer)
                if extracted is None:
                    raise ValueError(
                        f"Could not extract signal {name!r} from layer "
                        f"{source_layer.name!r}."
                    )
                return extracted

        raise ValueError(f"Signal {name!r} not found.")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _build_clean_kwargs(self) -> dict:
        """Read the panel's controls into `confusius.signal.clean` keyword arguments."""
        padtype_text = self._padtype_combo.currentText()
        return {
            "detrend_order": (
                self._detrend_order_spin.value()
                if self._detrend_check.isChecked()
                else None
            ),
            "standardize_method": _STANDARDIZE_LABELS[
                self._standardize_combo.currentText()
            ],
            "low_cutoff": (
                self._low_cutoff_spin.value()
                if self._low_cutoff_check.isChecked()
                else None
            ),
            "high_cutoff": (
                self._high_cutoff_spin.value()
                if self._high_cutoff_check.isChecked()
                else None
            ),
            "filter_butterworth_kwargs": {
                "order": self._filter_order_spin.value(),
                "padtype": None if padtype_text == "None" else padtype_text,
                "padlen": self._padlen_spin.value() or None,
                "uniformity_tolerance": self._uniformity_tolerance_spin.value(),
            },
            "standardize_confounds": True,
            "ensure_finite": self._ensure_finite_check.isChecked(),
            "interpolate_method": self._interpolate_method_combo.currentText(),
        }

    def _build_resample_spec(self) -> tuple[str, dict] | None:
        """Read the resampling controls into a `(mode, kwargs)` pipeline spec.

        Returns
        -------
        tuple[str, dict] or None
            `None` if `"No resampling"` is selected.

        Raises
        ------
        ValueError
            If "Match reference layer" is selected but no valid reference layer
            (with a `time` coordinate) is available.
        """
        mode = self._resample_mode_combo.currentText()
        if mode == _NO_RESAMPLING:
            return None

        method = self._resample_method_combo.currentText()
        if mode == _UNIFORM_GRID:
            return _UNIFORM_GRID, {
                "start": None,
                "stop": None,
                "step": self._resample_step_spin.value(),
                "method": method,
            }

        # Look up the reference layer directly (rather than via _get_source) so a
        # missing/invalid reference raises once here instead of also emitting its
        # own show_error — _apply's except block is the single place that surfaces
        # this failure to the user.
        reference_name = self._resample_reference_combo.currentText()
        if not reference_name:
            raise ValueError("No reference layer selected.")
        reference_da = self._viewer.layers[reference_name].metadata.get("xarray")
        if reference_da is None:
            raise ValueError(
                "Reference layer has no DataArray metadata. "
                "Load the file using the Data panel or the File menu."
            )
        if TIME_DIM not in reference_da.coords:
            raise ValueError("Reference layer has no 'time' coordinate.")
        return _MATCH_REFERENCE, {
            "new_time": reference_da.coords[TIME_DIM].values,
            "method": method,
        }

    def _build_smooth_kwargs(self) -> dict | None:
        """Read the smoothing controls into `confusius.spatial.smooth_volume` kwargs.

        Returns
        -------
        dict or None
            `None` if smoothing is not enabled.
        """
        if not self._smooth_enable_check.isChecked():
            return None
        return {
            "fwhm": self._smooth_fwhm_spin.value(),
            "ensure_finite": self._smooth_ensure_finite_check.isChecked(),
        }

    def _apply(self) -> None:
        source = self._get_source(self._source_combo)
        if source is None:
            return
        layer_name, da = source
        layer = self._viewer.layers[layer_name]

        try:
            confounds_raw = self._resolve_raw_signal(
                self._confounds_combo.currentText(), layer
            )
            mask_raw = self._resolve_raw_signal(self._mask_combo.currentText(), layer)
            resample_spec = self._build_resample_spec()
        except ValueError as exc:
            show_error(str(exc))
            return

        smooth_kwargs = self._build_smooth_kwargs()
        clean_kwargs = self._build_clean_kwargs()

        self._begin_work()
        worker = _run_pipeline(
            da, resample_spec, smooth_kwargs, clean_kwargs, confounds_raw, mask_raw
        )
        worker.returned.connect(
            lambda result: self._on_pipeline_returned(result, layer_name)
        )
        worker.errored.connect(self._on_worker_error)
        worker.start()

    def _on_pipeline_returned(self, result: xr.DataArray, layer_name: str) -> None:
        try:
            plot_napari(result, viewer=self._viewer, name=f"{layer_name} — cleaned")
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
        finally:
            self._end_work()

    def _on_worker_error(self, exc: Exception) -> None:
        self._end_work()
        show_error(str(exc))
