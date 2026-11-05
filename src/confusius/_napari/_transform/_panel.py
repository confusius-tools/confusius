"""Transform panel: move a volume in world space, with a live resampled preview."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import napari
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from napari.layers import Image, Labels, Layer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error
from qtpy.QtCore import QSignalBlocker, Qt, QTimer, Signal
from qtpy.QtGui import (
    QClipboard,
    QFontDatabase,
    QKeyEvent,
    QKeySequence,
    QShortcut,
    QWheelEvent,
)
from qtpy.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDial,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from confusius._dims import POSE_DIM, VOXEL_DIMS
from confusius._napari._registration._transform_payloads import (
    AffineTransformPayload,
    get_affine_transform_from_payload,
    load_transform_payload,
    make_output_grid_payload,
    save_transform_payload,
)
from confusius._napari._transform._model import (
    WORLD_AXES,
    Frame,
    History,
    TransformState,
    WorldAxis,
    format_matrix,
    parse_matrix,
    to_itk_transform,
    wrap_angle,
)
from confusius._napari._transform._session import (
    PreviewSession,
    build_applied_layer,
    build_resampled_layer,
)
from confusius._napari._transform._view import compute_tight_grid

ITK_SUFFIXES = {".tfm", ".txt"}
"""File suffixes saved as ITK transform files rather than ConfUSIus payloads."""

MAX_3D_DRAG_VOXELS = 12_000_000
"""Size of a rotation drag's grid in 3D, above which the preview coarsens until release.

Set for 150 ms per update and 300 ms for the drag start, on a 20-thread laptop with
integrated graphics, with the grid enclosing a full turn of a 264 × 160 × 228 volume
about its grid origin. Cost scales with voxels, so a slower machine may want this lower.
2D does not need a limit, since it resamples one plane.
"""

HALF_TURN = 180.0
"""Angle at a dial's far side, in degrees."""

ROTATION_DIAL_STEPS = 1800
"""Dial positions either side of 0, for 0.1 degree steps."""

DIAL_SIZE = 48
"""Width and height of a rotation dial, in pixels."""

TRANSLATION_SLIDER_STEPS = 1000
"""Slider positions either side of 0 on a translation row."""

VOXEL_STEP = "voxel"
"""Step selector entry for one voxel along the row's axis."""


class _NoWheelSlider(QSlider):
    """Slider leaving the wheel to the sidebar's scroll area.

    `install_no_scroll_wheel_filter` only covers combo and spin boxes. Without this,
    scrolling the sidebar with the cursor over a slider would move the layer.
    """

    def wheelEvent(self, event: QWheelEvent | None) -> None:  # ty: ignore[invalid-method-override]
        """Ignore the wheel, so it reaches the scroll area.

        Parameters
        ----------
        event : QWheelEvent, optional
            Wheel event.
        """
        if event is not None:
            event.ignore()


class _NoWheelDial(QDial):
    """Dial leaving the wheel to the sidebar's scroll area, like `_NoWheelSlider`."""

    def wheelEvent(self, event: QWheelEvent | None) -> None:  # ty: ignore[invalid-method-override]
        """Ignore the wheel, so it reaches the scroll area.

        Parameters
        ----------
        event : QWheelEvent, optional
            Wheel event.
        """
        if event is not None:
            event.ignore()


class AxisRow(QWidget):
    """Minus button, slider or dial, plus button and value box for one axis.

    A plain value change never echoes back as a signal, which rules out feedback loops
    with the panel.

    Parameters
    ----------
    label : str
        Axis name.
    half_range : float
        Value at the slider's end, or half a turn on a dial. The control spans
        `-half_range` to `half_range`.
    steps : int
        Slider or dial positions either side of 0.
    suffix : str, default: ""
        Unit shown in the value box.
    decimals : int, default: 4
        Decimals shown in the value box.
    dial : bool, default: False
        Whether to turn a dial that goes round, instead of sliding.

    Attributes
    ----------
    stepped : Signal(int)
        A button asked for one step, `1` or `-1`.
    edited : Signal(float)
        The value box, or a slider step outside a drag, set a value.
    drag_started : Signal()
        The slider handle was pressed.
    dragged : Signal(float)
        The slider handle moved to a value.
    drag_finished : Signal()
        The slider handle was released, and the drag was not cancelled.
    """

    stepped = Signal(int)
    edited = Signal(float)
    drag_started = Signal()
    dragged = Signal(float)
    drag_finished = Signal()

    def __init__(
        self,
        label: str,
        half_range: float,
        steps: int,
        *,
        suffix: str = "",
        decimals: int = 4,
        dial: bool = False,
    ) -> None:
        super().__init__()
        self._steps = steps
        self._value = 0.0
        self._dragging = False
        self.resolution = half_range / steps

        name_label = QLabel(label)
        self.minus_btn = QPushButton("−")
        self.minus_btn.setFixedWidth(24)
        self.plus_btn = QPushButton("+")
        self.plus_btn.setFixedWidth(24)

        self.slider: QAbstractSlider
        if dial:
            self.slider = _NoWheelDial()
            self.slider.setWrapping(True)
            self.slider.setNotchesVisible(True)
            self.slider.setFixedSize(DIAL_SIZE, DIAL_SIZE)
            # Arrow keys turn by one degree, Page Up and Page Down by ten.
            self.slider.setSingleStep(round(1.0 / self.resolution))
            self.slider.setPageStep(round(10.0 / self.resolution))
        else:
            self.slider = _NoWheelSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimumWidth(40)
        self.slider.setRange(-steps, steps)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(-1e6, 1e6)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        # napari's stylesheet floors spin boxes at 90 px, and the widest possible value
        # would widen the box further. Overriding both keeps the row inside the sidebar.
        self.spin.setStyleSheet("QDoubleSpinBox { min-width: 0px; padding: 1px 4px; }")

        if dial:
            # A column: the name, the dial at the left with its buttons stacked at the
            # right, and the box below spanning the two.
            column = QVBoxLayout(self)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(2)
            name_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            column.addWidget(name_label)
            knob = QHBoxLayout()
            knob.setSpacing(4)
            knob.addWidget(self.slider)
            knob.addStretch(1)
            buttons = QVBoxLayout()
            buttons.setSpacing(2)
            buttons.addWidget(self.plus_btn)
            buttons.addWidget(self.minus_btn)
            knob.addLayout(buttons)
            column.addLayout(knob)
            self.spin.setMinimumWidth(76)
            column.addWidget(self.spin)
        else:
            self.spin.setFixedWidth(76)
            row = QHBoxLayout(self)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(name_label)
            row.addWidget(self.minus_btn)
            row.addWidget(self.slider, stretch=1)
            row.addWidget(self.plus_btn)
            row.addWidget(self.spin)

        self.minus_btn.clicked.connect(lambda: self.stepped.emit(-1))
        self.plus_btn.clicked.connect(lambda: self.stepped.emit(1))
        self.spin.valueChanged.connect(self.edited)
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)

    def set_half_range(self, half_range: float) -> None:
        """Change the value at the slider's end, keeping the current value.

        Parameters
        ----------
        half_range : float
            Value at the slider's end.
        """
        self.resolution = half_range / self._steps
        self.set_value(self._value)

    def set_value(self, value: float) -> None:
        """Show a value, without emitting anything.

        Parameters
        ----------
        value : float
            Value to show.
        """
        self._value = value
        self._show(value)
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(value / self.resolution))

    def cancel_drag(self) -> None:
        """Abandon the drag in progress and show the value from before it.

        The mouse may stay down afterwards, so the slider is ignored until its release.
        """
        self._dragging = False
        self.set_value(self._value)

    def _show(self, value: float) -> None:
        """Show a value in the value box only.

        Parameters
        ----------
        value : float
            Value to show.
        """
        with QSignalBlocker(self.spin):
            self.spin.setValue(value)

    def _on_slider_value_changed(self, position: int) -> None:
        """Forward slider changes made outside a drag, such as a click beside the handle.

        Parameters
        ----------
        position : int
            Slider position.
        """
        if not self.slider.isSliderDown():
            self.edited.emit(position * self.resolution)

    def _on_slider_pressed(self) -> None:
        """Start a drag."""
        self._dragging = True
        self.drag_started.emit()

    def _on_slider_moved(self, position: int) -> None:
        """Forward a drag's new value.

        Parameters
        ----------
        position : int
            Slider position.
        """
        if self._dragging:
            value = position * self.resolution
            self._show(value)
            self.dragged.emit(value)

    def _on_slider_released(self) -> None:
        """Finish a drag, or settle the slider after a cancelled one."""
        if self._dragging:
            self._dragging = False
            self.drag_finished.emit()
        else:
            self.set_value(self._value)


@dataclass
class _Target:
    """One layer's transform, kept when another layer is selected.

    Attributes
    ----------
    session : PreviewSession
        Preview on the layer.
    state : TransformState
        Latest transform.
    history : History
        Undo history.
    half_range : float
        Value at the end of a translation slider.
    voxel_steps : (3,) numpy.ndarray
        One voxel along world `(z, y, x)`, in data units.
    """

    session: PreviewSession
    state: TransformState
    history: History
    half_range: float
    voxel_steps: npt.NDArray[np.float64]


@dataclass
class _Drag:
    """Slider drag in progress.

    Attributes
    ----------
    row : AxisRow
        Row being dragged.
    start : TransformState
        Transform when the drag started.
    kind : {"rotation", "translation"}
        What the row controls. A rotation keeps one grid during the drag, whereas a
        translation just moves the grid along.
    axis : int
        Index of the row's axis in `(z, y, x)`.
    angle : float
        Dial angle at the latest move of a rotation drag, in degrees.
    turned : float
        Angle a rotation drag has turned since it started, in degrees, beyond half a
        turn when the dial went round.
    """

    row: AxisRow
    start: TransformState
    kind: Literal["rotation", "translation"]
    axis: int
    angle: float = 0.0
    turned: float = 0.0


def _get_clipboard() -> QClipboard:
    """Return the application's clipboard.

    Returns
    -------
    QClipboard
        The clipboard, which exists whenever a panel does.
    """
    clipboard = QApplication.clipboard()
    assert clipboard is not None
    return clipboard


def _is_candidate(layer: Layer) -> bool:
    """Return whether a layer may be offered as a transform target.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer to check.

    Returns
    -------
    bool
        Whether it is an image or labels layer built from VoxelData, without a pose dim.
    """
    if not isinstance(layer, (Image, Labels)):
        return False
    displayed = layer.metadata.get("xarray")
    return displayed is not None and POSE_DIM not in displayed.dims


class TransformPanel(QWidget):
    """Translate and rotate a layer, previewing the result live.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer holding the layers.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._targets: dict[Image | Labels, _Target] = {}
        self._target: _Target | None = None
        self._target_error = ""
        self._drag: _Drag | None = None
        self._preview: TransformState | None = None
        self._update_pending = False
        self._worker: Any = None
        self._marker_layer: Layer | None = None
        self._blending_to_restore: tuple[Layer, str] | None = None

        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(0)
        self._update_timer.timeout.connect(self._flush_update)

        self._setup_ui()
        viewer.layers.events.inserted.connect(self._refresh_layers)
        viewer.layers.events.removed.connect(self._refresh_layers)
        viewer.layers.events.renamed.connect(self._refresh_layers)
        self._refresh_layers()

    @property
    def state(self) -> TransformState | None:
        """Return the latest transform of the selected layer.

        Returns
        -------
        TransformState or None
            Latest transform, or `None` if no layer is selected.
        """
        return self._target.state if self._target is not None else None

    @property
    def _frame(self) -> Frame:
        """Return the axes the rows act along.

        Returns
        -------
        {"global", "local"}
            Selected frame.
        """
        return "local" if self._frame_local_radio.isChecked() else "global"

    def _setup_ui(self) -> None:
        """Build the widgets."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._make_target_group())
        self._translate_group, self._translate_rows = self._make_rows_group(
            "Translate (data units)",
            half_range=1.0,
            steps=TRANSLATION_SLIDER_STEPS,
            suffix="",
        )
        self._translate_step_combo = self._make_step_combo(
            self._translate_group,
            [("0.01", 0.01), ("0.1", 0.1), ("1", 1.0), ("1 voxel", VOXEL_STEP)],
            default=1,
        )
        layout.addWidget(self._translate_group)
        self._rotate_group, self._rotate_rows = self._make_rows_group(
            "Rotate (°)",
            half_range=HALF_TURN,
            steps=ROTATION_DIAL_STEPS,
            suffix="°",
            dial=True,
        )
        self._rotate_step_combo = self._make_step_combo(
            self._rotate_group,
            [("0.1°", 0.1), ("1°", 1.0), ("5°", 5.0)],
            default=1,
        )
        layout.addWidget(self._rotate_group)
        layout.addWidget(self._make_zoom_group())
        layout.addWidget(self._make_center_group())
        layout.addWidget(self._make_matrix_group())
        layout.addWidget(self._make_actions_row())
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.hide()
        layout.addWidget(self._status_label)
        layout.addStretch()

        for axis, name in enumerate(WORLD_AXES):
            self._connect_translate_row(axis, self._translate_rows[name])
            self._connect_rotate_row(axis, self._rotate_rows[name])

        for keys, slot in (
            (Qt.Key.Key_Escape, self._cancel_drag),
            (QKeySequence.StandardKey.Undo, self._undo),
            ("Ctrl+Shift+Z", self._redo),
        ):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    def _make_target_group(self) -> QGroupBox:
        """Build the target layer and frame selectors.

        Returns
        -------
        QGroupBox
            The group.
        """
        group = QGroupBox("Target")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Layer"))
        self._target_combo = QComboBox()
        self._target_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        layout.addWidget(self._target_combo)

        layout.addWidget(QLabel("Reference"))
        self._reference_combo = QComboBox()
        self._reference_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._reference_combo.setToolTip(
            "Layer to overlay while aligning, and whose grid Resample uses."
        )
        self._reference_combo.currentIndexChanged.connect(self._on_reference_changed)
        layout.addWidget(self._reference_combo)

        axes_row = QHBoxLayout()
        axes_row.addWidget(QLabel("Axes"))
        self._frame_global_radio = QRadioButton("Global")
        self._frame_local_radio = QRadioButton("Local")
        self._frame_global_radio.setChecked(True)
        frame_group = QButtonGroup(self)
        frame_group.addButton(self._frame_global_radio)
        frame_group.addButton(self._frame_local_radio)
        self._frame_global_radio.toggled.connect(lambda _: self._sync_widgets())
        axes_row.addWidget(self._frame_global_radio)
        axes_row.addWidget(self._frame_local_radio)
        axes_row.addStretch(1)
        layout.addLayout(axes_row)

        return group

    def _make_rows_group(
        self,
        title: str,
        *,
        half_range: float,
        steps: int,
        suffix: str,
        dial: bool = False,
    ) -> tuple[QGroupBox, dict[WorldAxis, AxisRow]]:
        """Build a group with one row per world axis.

        Parameters
        ----------
        title : str
            Group title.
        half_range : float
            Value at the slider's end, or half a turn on a dial.
        steps : int
            Slider or dial positions either side of 0.
        suffix : str
            Unit shown in the value boxes.
        dial : bool, default: False
            Whether the rows turn dials instead of sliding.

        Returns
        -------
        group : QGroupBox
            The group.
        rows : dict[{"z", "y", "x"}, AxisRow]
            Rows by world axis name.
        """
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        rows = {
            name: AxisRow(name, half_range, steps, suffix=suffix, dial=dial)
            for name in WORLD_AXES
        }
        if dial:
            # Dials are compact enough to sit side by side.
            columns = QHBoxLayout()
            columns.setSpacing(16)
            for row in rows.values():
                columns.addWidget(row)
            layout.addLayout(columns)
        else:
            for row in rows.values():
                layout.addWidget(row)
        return group, rows

    def _make_step_combo(
        self,
        group: QGroupBox,
        entries: list[tuple[str, float | str]],
        *,
        default: int,
    ) -> QComboBox:
        """Add a step selector at the top of a group.

        Parameters
        ----------
        group : QGroupBox
            Group to add to.
        entries : list of tuple
            Label and data of each entry.
        default : int
            Index of the entry selected at first.

        Returns
        -------
        QComboBox
            The selector.
        """
        combo = QComboBox()
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        for label, data in entries:
            combo.addItem(label, data)
        combo.setCurrentIndex(default)
        row = QHBoxLayout()
        row.addWidget(QLabel("Step"))
        row.addWidget(combo, stretch=1)
        layout = group.layout()
        assert isinstance(layout, QVBoxLayout)
        layout.insertLayout(0, row)
        return combo

    def _make_zoom_group(self) -> QGroupBox:
        """Build the collapsed zoom and flip section.

        Returns
        -------
        QGroupBox
            The group.
        """
        self._zoom_group = QGroupBox()
        layout = QVBoxLayout(self._zoom_group)
        layout.setSpacing(4)

        self._zoom_toggle = QToolButton()
        self._zoom_toggle.setCheckable(True)
        self._zoom_toggle.setAutoRaise(True)
        self._zoom_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._zoom_toggle.setText("Zoom and flip")
        self._zoom_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._zoom_toggle.toggled.connect(self._on_zoom_toggled)
        layout.addWidget(self._zoom_toggle)

        self._zoom_content = QWidget()
        content_layout = QVBoxLayout(self._zoom_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)
        self._zoom_spins: dict[str, QDoubleSpinBox] = {}
        self._flip_btns: dict[str, QPushButton] = {}
        for axis, name in enumerate(WORLD_AXES):
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            spin = QDoubleSpinBox()
            spin.setRange(0.01, 100.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setValue(1.0)
            spin.setKeyboardTracking(False)
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.setMinimumWidth(64)
            spin.valueChanged.connect(
                lambda value, axis=axis: self._on_zoom_edited(axis, value)
            )
            row.addWidget(spin, stretch=1)
            flip_btn = QPushButton("Flip")
            flip_btn.clicked.connect(lambda _, name=name: self._on_flip(name))
            row.addWidget(flip_btn)
            self._zoom_spins[name] = spin
            self._flip_btns[name] = flip_btn
            content_layout.addLayout(row)
        self._zoom_lock_check = QCheckBox("Lock uniform")
        content_layout.addWidget(self._zoom_lock_check)
        self._zoom_content.hide()
        layout.addWidget(self._zoom_content)
        return self._zoom_group

    def _make_center_group(self) -> QGroupBox:
        """Build the center of rotation controls.

        Returns
        -------
        QGroupBox
            The group.
        """
        self._center_group = QGroupBox("Center of rotation")
        layout = QVBoxLayout(self._center_group)
        layout.setSpacing(4)
        self._center_readout = QLabel()
        self._center_readout.setToolTip("World position, in z, y, x order.")
        layout.addWidget(self._center_readout)

        buttons = QHBoxLayout()
        self._center_origin_btn = QPushButton("Grid origin")
        self._center_cursor_btn = QPushButton("Cursor")
        self._center_geometry_btn = QPushButton("Geometry")
        self._center_origin_btn.setToolTip("Use the first voxel of the volume.")
        self._center_cursor_btn.setToolTip("Use the cursor's world position.")
        self._center_geometry_btn.setToolTip("Use the center of the volume's box.")
        self._center_origin_btn.clicked.connect(self._on_center_origin)
        self._center_cursor_btn.clicked.connect(self._on_center_cursor)
        self._center_geometry_btn.clicked.connect(self._on_center_geometry)
        buttons.addWidget(self._center_origin_btn)
        buttons.addWidget(self._center_cursor_btn)
        buttons.addWidget(self._center_geometry_btn)
        layout.addLayout(buttons)

        self._center_marker_check = QCheckBox("Show marker")
        self._center_marker_check.toggled.connect(self._on_marker_toggled)
        layout.addWidget(self._center_marker_check)
        return self._center_group

    def _make_matrix_group(self) -> QGroupBox:
        """Build the matrix view and its tools.

        Returns
        -------
        QGroupBox
            The group.
        """
        self._matrix_group = QGroupBox("4×4 matrix, z y x order")
        layout = QVBoxLayout(self._matrix_group)
        layout.setSpacing(4)

        self._matrix_view = QPlainTextEdit()
        self._matrix_view.setReadOnly(True)
        self._matrix_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._matrix_view.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        self._matrix_view.setFixedHeight(
            5 * self._matrix_view.fontMetrics().lineSpacing()
        )
        self._matrix_view.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._matrix_view)

        buttons = QGridLayout()
        buttons.setSpacing(4)
        self._copy_btn = QPushButton("Copy")
        self._paste_btn = QPushButton("Paste")
        self._load_btn = QPushButton("Load…")
        self._save_btn = QPushButton("Save…")
        self._identity_btn = QPushButton("Identity")
        self._invert_btn = QPushButton("Invert")
        self._copy_btn.clicked.connect(self._copy_matrix)
        self._paste_btn.clicked.connect(self._paste_matrix)
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._identity_btn.clicked.connect(self._set_identity)
        self._invert_btn.clicked.connect(self._invert)
        for index, button in enumerate(
            (
                self._copy_btn,
                self._paste_btn,
                self._load_btn,
                self._save_btn,
                self._identity_btn,
                self._invert_btn,
            )
        ):
            buttons.addWidget(button, index // 3, index % 3)
        layout.addLayout(buttons)
        return self._matrix_group

    def _make_actions_row(self) -> QWidget:
        """Build the undo, redo, reset, resample and apply buttons.

        Returns
        -------
        QWidget
            The rows.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        history_row = QHBoxLayout()
        history_row.setSpacing(4)
        self._undo_btn = QPushButton("Undo")
        self._redo_btn = QPushButton("Redo")
        self._reset_btn = QPushButton("Reset")
        self._undo_btn.clicked.connect(self._undo)
        self._redo_btn.clicked.connect(self._redo)
        self._reset_btn.clicked.connect(self._reset)
        history_row.addWidget(self._undo_btn)
        history_row.addWidget(self._redo_btn)
        history_row.addWidget(self._reset_btn)
        self._not_applied_label = QLabel("Not applied")
        self._not_applied_label.hide()
        history_row.addWidget(self._not_applied_label)
        history_row.addStretch(1)
        layout.addLayout(history_row)

        commit_row = QHBoxLayout()
        commit_row.setSpacing(4)
        self._resample_btn = QPushButton("Resample")
        self._resample_btn.setToolTip(
            "Create a new layer with resampled voxels, on the reference layer's grid if "
            "one is selected. The original layer does not change."
        )
        self._apply_btn = QPushButton("Apply to affine")
        self._apply_btn.setObjectName("primary_btn")
        self._apply_btn.setToolTip(
            "Write the transform into the layer's voxel-to-world affine. Voxel values do "
            "not change."
        )
        self._resample_btn.clicked.connect(self._resample)
        self._apply_btn.clicked.connect(self._apply)
        commit_row.addWidget(self._resample_btn, stretch=1)
        commit_row.addWidget(self._apply_btn, stretch=1)
        layout.addLayout(commit_row)
        return container

    def _connect_translate_row(self, axis: int, row: AxisRow) -> None:
        """Connect a translation row.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.
        row : AxisRow
            The row.
        """
        row.stepped.connect(lambda sign: self._on_translate_stepped(axis, sign))
        row.edited.connect(lambda value: self._on_translate_edited(axis, value))
        row.drag_started.connect(lambda: self._begin_translate_drag(axis, row))
        row.dragged.connect(self._on_translate_dragged)
        row.drag_finished.connect(self._finish_drag)

    def _connect_rotate_row(self, axis: int, row: AxisRow) -> None:
        """Connect a rotation row.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.
        row : AxisRow
            The row.
        """
        row.stepped.connect(lambda sign: self._on_rotate_stepped(axis, sign))
        row.edited.connect(lambda value: self._on_rotate_edited(axis, value))
        row.drag_started.connect(lambda: self._begin_rotate_drag(axis, row))
        row.dragged.connect(self._on_rotate_dragged)
        row.drag_finished.connect(self._finish_drag)

    def _refresh_layers(self) -> None:
        """Repopulate the target selector and forget removed layers."""
        self._targets = {
            layer: target
            for layer, target in self._targets.items()
            if layer in self.viewer.layers
        }
        current = self._target_combo.currentData()
        with QSignalBlocker(self._target_combo):
            self._target_combo.clear()
            for layer in self.viewer.layers:
                if _is_candidate(layer):
                    self._target_combo.addItem(layer.name, layer)
            index = self._target_combo.findData(current)
            self._target_combo.setCurrentIndex(max(index, 0))
        if self._target_combo.currentData() is not current or current is None:
            self._on_target_changed()
        self._refresh_reference_combo()
        self._sync_marker()

    def _refresh_reference_combo(self) -> None:
        """Repopulate the reference selector with every layer but the target."""
        target_layer = self._target_combo.currentData()
        current = self._reference_combo.currentData()
        with QSignalBlocker(self._reference_combo):
            self._reference_combo.clear()
            self._reference_combo.addItem("None", None)
            for layer in self.viewer.layers:
                if _is_candidate(layer) and layer is not target_layer:
                    self._reference_combo.addItem(layer.name, layer)
            index = self._reference_combo.findData(current) if current else 0
            self._reference_combo.setCurrentIndex(max(index, 0))
        if self._reference_combo.currentData() is not current:
            self._on_reference_changed()

    def _on_target_changed(self) -> None:
        """Switch to the selected layer."""
        self._cancel_drag()
        self._restore_blending()
        with QSignalBlocker(self._reference_combo):
            self._reference_combo.setCurrentIndex(0)
        self._target = None
        self._target_error = ""
        layer = self._target_combo.currentData()
        if layer is not None:
            try:
                self._target = self._get_target(layer)
            except ValueError as error:
                self._target_error = str(error)
        if self._target is not None:
            for row in self._translate_rows.values():
                row.set_half_range(self._target.half_range)
        self._refresh_reference_combo()
        self._sync_widgets()

    def _on_reference_changed(self) -> None:
        """Overlay the selected reference under the target, or stop overlaying."""
        self._restore_blending()
        reference = self._reference_combo.currentData()
        target_layer = self._target_combo.currentData()
        if reference is None or target_layer is None:
            return
        reference.visible = True
        layers = self.viewer.layers
        if layers.index(reference) > layers.index(target_layer):
            layers.move(layers.index(reference), layers.index(target_layer))
        self._blending_to_restore = (target_layer, str(target_layer.blending))
        target_layer.blending = "additive"

    def _restore_blending(self) -> None:
        """Give the target back the blending it had before overlaying a reference."""
        if self._blending_to_restore is not None:
            layer, blending = self._blending_to_restore
            layer.blending = blending
            self._blending_to_restore = None

    def _get_target(self, layer: Image | Labels) -> _Target:
        """Return a layer's transform, starting it if it is new.

        Parameters
        ----------
        layer : napari.layers.Image or napari.layers.Labels
            Layer to transform.

        Returns
        -------
        _Target
            The layer's transform.

        Raises
        ------
        ValueError
            If the layer cannot be transformed.
        """
        if layer in self._targets:
            return self._targets[layer]
        session = PreviewSession(layer)
        source = session.source
        origin = source.fusi.origin
        state = TransformState.identity(center=[origin[name] for name in WORLD_AXES])
        bounds = source.fusi.affine.bounding_box
        extent = bounds.sel(bound="max") - bounds.sel(bound="min")
        target = _Target(
            session=session,
            state=state,
            history=History(state),
            half_range=float(extent.max()),
            voxel_steps=compute_tight_grid(source, np.eye(4)).spacing,
        )
        self._targets[layer] = target
        return target

    def _sync_widgets(self) -> None:
        """Make every widget show the selected layer's transform."""
        target = self._target
        idle = self._worker is None
        usable = target is not None and idle
        is_moved = target is not None and not target.state.is_identity()
        is_sheared = target is not None and target.state.is_sheared
        self._target_combo.setEnabled(idle)
        self._reference_combo.setEnabled(usable)
        for group in (self._translate_group, self._rotate_group, self._zoom_group):
            group.setEnabled(usable and not is_sheared)
        self._center_group.setEnabled(usable)
        self._matrix_group.setEnabled(usable)
        self._reset_btn.setEnabled(usable)
        self._undo_btn.setEnabled(usable and target.history.can_undo)
        self._redo_btn.setEnabled(usable and target.history.can_redo)
        self._resample_btn.setEnabled(usable)
        self._apply_btn.setEnabled(usable and is_moved)
        self._not_applied_label.setVisible(is_moved)
        status = self._target_error or (
            "Affine mode: this matrix carries shear, so only the matrix tools, Undo and "
            "Reset can change it."
            if is_sheared
            else ""
        )
        self._status_label.setText(status)
        self._status_label.setVisible(bool(status))
        if target is None:
            self._center_readout.setText("")
            self._matrix_view.setPlainText("")
            return
        self._show_matrix(target.state.matrix())
        offsets = self._get_frame_translation(target.state)
        magnitudes = np.abs(target.state.scale)
        for axis, name in enumerate(WORLD_AXES):
            self._translate_rows[name].set_value(float(offsets[axis]))
            self._rotate_rows[name].set_value(float(target.state.turned[axis]))
            with QSignalBlocker(self._zoom_spins[name]):
                self._zoom_spins[name].setValue(float(magnitudes[axis]))
        self._center_readout.setText(
            ", ".join(f"{value:.4g}" for value in target.state.pivot())
        )
        self._sync_marker()

    def _get_frame_translation(self, state: TransformState) -> npt.NDArray[np.float64]:
        """Return a state's translation along the rows' axes.

        Parameters
        ----------
        state : TransformState
            State to read.

        Returns
        -------
        (3,) numpy.ndarray
            Translation along the current frame's `(z, y, x)` axes.
        """
        if self._frame == "local":
            return state.rotation.T @ state.translation
        return state.translation

    @contextmanager
    def _keeping_slice(self) -> Generator[None, None, None]:
        """Keep the displayed slice in place while the target's extent changes.

        napari clamps the slice position to the layers' extent after every geometry
        change, and the session changes a layer's shape and placement one after the
        other, so the slice could end up clamped to an extent that only existed in
        between.

        Yields
        ------
        None
            Control returns to the wrapped block; the slice position is set back when
            it exits.
        """
        point = tuple(self.viewer.dims.point)
        try:
            yield
        finally:
            self.viewer.dims.point = point

    def _commit(self, state: TransformState) -> None:
        """Make a state the latest one, as one undo step.

        Parameters
        ----------
        state : TransformState
            State to display and record.
        """
        assert self._target is not None
        # Show first: if that fails, the state must keep describing what is displayed.
        with self._keeping_slice():
            self._target.session.show(state.matrix())
        self._target.state = state
        self._target.history.push(state)
        self._sync_widgets()

    def _go_to(self, state: TransformState) -> None:
        """Display a state from the history.

        History states carry the center they were made with, whereas moving the center
        is not an undo step. So the current center stays, on the same point of the volume.

        Parameters
        ----------
        state : TransformState
            State to display.
        """
        assert self._target is not None
        matrix = state.matrix()
        center = self._target.state.center
        with self._keeping_slice():
            self._target.session.show(matrix)
        self._target.state = state.with_pivot(matrix[:3, :3] @ center + matrix[:3, 3])
        self._sync_widgets()

    def _undo(self) -> None:
        """Go back one step."""
        if self._target is None or self._drag is not None:
            return
        state = self._target.history.undo()
        if state is not None:
            self._go_to(state)

    def _redo(self) -> None:
        """Go forward one step."""
        if self._target is None or self._drag is not None:
            return
        state = self._target.history.redo()
        if state is not None:
            self._go_to(state)

    def _reset(self) -> None:
        """Restore the layer as it was, keeping the center of rotation."""
        assert self._target is not None
        state = TransformState.identity(center=self._target.state.pivot())
        self._target.state = state
        self._target.history.clear(state)
        with self._keeping_slice():
            self._target.session.restore()
        self._sync_widgets()

    def _get_translate_step(self, axis: int) -> float:
        """Return the step of a translation row.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.

        Returns
        -------
        float
            Step in data units.
        """
        assert self._target is not None
        step = self._translate_step_combo.currentData()
        return float(self._target.voxel_steps[axis]) if step == VOXEL_STEP else step

    def _on_translate_stepped(self, axis: int, sign: int) -> None:
        """Apply one translation step.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.
        sign : int
            Direction, `1` or `-1`.
        """
        assert self._target is not None
        delta = np.zeros(3)
        delta[axis] = sign * self._get_translate_step(axis)
        self._commit(self._target.state.translated(delta, frame=self._frame))

    def _on_translate_edited(self, axis: int, value: float) -> None:
        """Set a translation row's value.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.
        value : float
            New offset along the row's axis.
        """
        assert self._target is not None
        self._commit(self._translated_to(self._target.state, axis, value))

    def _translated_to(
        self, state: TransformState, axis: int, value: float
    ) -> TransformState:
        """Return a state whose offset along one row's axis is `value`.

        Parameters
        ----------
        state : TransformState
            State to start from.
        axis : int
            Index of the row's axis in `(z, y, x)`.
        value : float
            New offset along the row's axis.

        Returns
        -------
        TransformState
            The translated state.
        """
        delta = np.zeros(3)
        delta[axis] = value - self._get_frame_translation(state)[axis]
        return state.translated(delta, frame=self._frame)

    def _on_rotate_stepped(self, axis: int, sign: int) -> None:
        """Apply one rotation step.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.
        sign : int
            Direction, `1` or `-1`.
        """
        assert self._target is not None
        step = sign * self._rotate_step_combo.currentData()
        self._commit(
            self._target.state.rotated(WORLD_AXES[axis], step, frame=self._frame)
        )

    def _on_rotate_edited(self, axis: int, angle: float) -> None:
        """Turn until the dial shows a given angle, the short way round.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.
        angle : float
            Angle turned about the axis to show, in degrees.
        """
        assert self._target is not None
        state = self._target.state
        step = wrap_angle(angle - state.turned[axis])
        self._commit(state.rotated(WORLD_AXES[axis], step, frame=self._frame))

    def _begin_translate_drag(self, axis: int, row: AxisRow) -> None:
        """Start a translation drag.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.
        row : AxisRow
            Row being dragged.
        """
        assert self._target is not None
        self._drag = _Drag(row, self._target.state, "translation", axis)
        self._preview = None

    def _begin_rotate_drag(self, axis: int, row: AxisRow) -> None:
        """Start a rotation drag, on a grid enclosing all of it.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.
        row : AxisRow
            Row being dragged.
        """
        assert self._target is not None
        start = self._target.state
        self._drag = _Drag(
            row, start, "rotation", axis, angle=float(start.turned[axis])
        )
        self._preview = None
        with self._keeping_slice():
            self._target.session.begin_rotation_drag(
                start.matrix(),
                start.rotation_axis(WORLD_AXES[axis], frame=self._frame),
                start.pivot(),
                max_voxels=MAX_3D_DRAG_VOXELS
                if self.viewer.dims.ndisplay == 3
                else None,
            )

    def _on_translate_dragged(self, value: float) -> None:
        """Preview a translation drag's new value.

        Parameters
        ----------
        value : float
            New offset along the row's axis.
        """
        assert self._drag is not None
        self._schedule_update(
            self._translated_to(self._drag.start, self._drag.axis, value)
        )

    def _on_rotate_dragged(self, angle: float) -> None:
        """Preview a rotation drag's new value.

        Parameters
        ----------
        angle : float
            Dial angle in degrees.
        """
        drag = self._drag
        assert drag is not None
        # The dial goes round, so a move across its far side is a short turn, not a
        # long one the other way.
        drag.turned += wrap_angle(angle - drag.angle)
        drag.angle = angle
        self._schedule_update(
            drag.start.rotated(WORLD_AXES[drag.axis], drag.turned, frame=self._frame)
        )

    def _schedule_update(self, state: TransformState) -> None:
        """Keep only the latest drag state for the next turn of the event loop.

        Parameters
        ----------
        state : TransformState
            Latest drag state.
        """
        self._preview = state
        self._update_pending = True
        if not self._update_timer.isActive():
            self._update_timer.start()

    def _flush_update(self) -> None:
        """Display the latest drag state, if it is not displayed yet."""
        self._update_timer.stop()
        if not self._update_pending:
            return
        self._update_pending = False
        if self._target is None or self._drag is None or self._preview is None:
            return
        matrix = self._preview.matrix()
        if self._drag.kind == "rotation":
            self._target.session.drag_to(matrix)
        else:
            with self._keeping_slice():
                self._target.session.show(matrix)
        self._show_matrix(matrix)

    def _finish_drag(self) -> None:
        """Commit a released drag as one undo step, on the tight grid."""
        drag, preview = self._drag, self._preview
        self._drag = None
        self._preview = None
        self._update_pending = False
        self._update_timer.stop()
        if drag is None or self._target is None:
            return
        if preview is None:
            self._go_to(drag.start)
        else:
            self._commit(preview)

    def _cancel_drag(self) -> None:
        """Abandon the drag in progress and go back to where it started."""
        drag = self._drag
        if drag is None:
            return
        self._drag = None
        self._preview = None
        self._update_pending = False
        self._update_timer.stop()
        drag.row.cancel_drag()
        if self._target is not None:
            self._go_to(drag.start)

    def _show_matrix(self, matrix: npt.NDArray[np.float64]) -> None:
        """Show a transform in the read-only matrix view.

        Parameters
        ----------
        matrix : (4, 4) numpy.ndarray
            Transform to show.
        """
        self._matrix_view.setPlainText(
            "\n".join(" ".join(f"{value:>9.6g}" for value in row) for row in matrix)
        )

    def _copy_matrix(self) -> None:
        """Copy the transform to the clipboard, at full precision."""
        assert self._target is not None
        _get_clipboard().setText(format_matrix(self._target.state.matrix()))

    def _paste_matrix(self) -> None:
        """Set the transform from the clipboard, as one undo step."""
        assert self._target is not None
        try:
            matrix = parse_matrix(_get_clipboard().text())
        except ValueError as error:
            show_error(str(error))
            return
        self._commit(
            TransformState.from_matrix(matrix, center=self._target.state.center)
        )

    def _set_identity(self) -> None:
        """Set the transform to the identity, as one undo step."""
        assert self._target is not None
        self._commit(TransformState.identity(center=self._target.state.pivot()))

    def _invert(self) -> None:
        """Set the transform to its inverse, as one undo step."""
        assert self._target is not None
        state = self._target.state
        self._commit(
            TransformState.from_matrix(
                np.linalg.inv(state.matrix()), center=state.center
            )
        )

    def _on_save_clicked(self) -> None:
        """Ask where to save the transform, and save it."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save transform",
            "",
            "ConfUSIus transform (*.json);;ITK transform (*.tfm *.txt)",
        )
        if path:
            try:
                self._save_matrix(path)
            except OSError as error:
                show_error(str(error))

    def _on_load_clicked(self) -> None:
        """Ask which transform to load, and load it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load transform", "", "ConfUSIus transform (*.json)"
        )
        if path:
            self._load_matrix(path)

    def _save_matrix(self, path: str | Path) -> None:
        """Save the transform to a file.

        A `.tfm` or `.txt` file is an ITK transform, for other tools such as 3D Slicer.
        Anything else is the JSON payload the registration panel also reads.

        Parameters
        ----------
        path : str or pathlib.Path
            File to write.
        """
        assert self._target is not None
        matrix = self._target.state.matrix()
        if Path(path).suffix in ITK_SUFFIXES:
            sitk.WriteTransform(to_itk_transform(matrix), str(path))
            return
        source = self._target.session.source
        spatial = source.isel(
            {dim: 0 for dim in source.dims if dim not in VOXEL_DIMS}, drop=True
        )
        name = self._target.session.layer.name
        payload: AffineTransformPayload = {
            "kind": "affine",
            "name": f"{name} (manual)",
            # These payloads hold pull transforms, from output points to input points.
            "affine": np.linalg.inv(matrix).tolist(),
            "source_layer_name": name,
            "target_layer_name": name,
            "operation": "manual_transform_panel",
            "transform_model": "affine",
            "metric": "manual",
            "output_grid": make_output_grid_payload(spatial),
            "input_grid": make_output_grid_payload(spatial),
            "diagnostics": {
                "metric": "manual",
                "final_metric_value": 0.0,
                "n_iterations": 0,
                "stop_condition": "Saved from the transform panel.",
                "status": "completed",
            },
        }
        save_transform_payload(path, payload)

    def _load_matrix(self, path: str | Path) -> None:
        """Set the transform from a saved payload, as one undo step.

        Parameters
        ----------
        path : str or pathlib.Path
            File to read.
        """
        assert self._target is not None
        try:
            pull = get_affine_transform_from_payload(load_transform_payload(path))
            if pull.shape != (4, 4):
                raise ValueError(f"Expected a 4×4 affine, got shape {pull.shape}.")
            matrix = np.linalg.inv(pull)
        except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as error:
            show_error(str(error))
            return
        self._commit(
            TransformState.from_matrix(matrix, center=self._target.state.center)
        )

    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # ty: ignore[invalid-method-override]
        """Translate along the displayed axes with the arrow and page keys.

        Parameters
        ----------
        event : QKeyEvent, optional
            Key press.
        """
        delta = self._get_key_translation(event) if event is not None else None
        if delta is None:
            super().keyPressEvent(event)
            return
        assert self._target is not None
        self._commit(self._target.state.translated(delta, frame="global"))

    def _get_key_translation(self, event: QKeyEvent) -> npt.NDArray[np.float64] | None:
        """Return the world translation a key press asks for.

        Parameters
        ----------
        event : QKeyEvent
            Key press.

        Returns
        -------
        (3,) numpy.ndarray or None
            Translation in `(z, y, x)` order, or `None` if the key does not translate
            right now.
        """
        target = self._target
        if (
            target is None
            or target.state.is_sheared
            or self._worker is not None
            or self._drag is not None
        ):
            return None
        dims = self.viewer.dims
        first_spatial = dims.ndim - 3
        vertical, horizontal = dims.displayed[-2:]
        through = next(
            (
                dim
                for dim in dims.order
                if dim >= first_spatial and dim not in (vertical, horizontal)
            ),
            None,
        )
        dim, sign = {
            Qt.Key.Key_Right: (horizontal, 1.0),
            Qt.Key.Key_Left: (horizontal, -1.0),
            Qt.Key.Key_Down: (vertical, 1.0),
            Qt.Key.Key_Up: (vertical, -1.0),
            Qt.Key.Key_PageUp: (through, 1.0),
            Qt.Key.Key_PageDown: (through, -1.0),
        }.get(event.key(), (None, 0.0))
        if dim is None or dim < first_spatial:
            return None

        axis = dim - first_spatial
        step = self._get_translate_step(axis)
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step *= 10.0
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            step /= 10.0
        delta = np.zeros(3)
        delta[axis] = sign * step
        return delta

    def _on_zoom_toggled(self, expanded: bool) -> None:
        """Show or hide the zoom and flip controls.

        Parameters
        ----------
        expanded : bool
            Whether to show them.
        """
        self._zoom_content.setVisible(expanded)
        self._zoom_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _on_zoom_edited(self, axis: int, value: float) -> None:
        """Set one zoom factor, or all of them when locked.

        Parameters
        ----------
        axis : int
            Index of the row's axis in `(z, y, x)`.
        value : float
            New factor.
        """
        assert self._target is not None
        factors = np.abs(self._target.state.scale)
        if self._zoom_lock_check.isChecked():
            factors = np.full(3, value)
        else:
            factors[axis] = value
        self._commit(self._target.state.zoomed(factors))

    def _on_flip(self, name: WorldAxis) -> None:
        """Mirror the volume along one axis, through the center of rotation.

        Parameters
        ----------
        name : {"z", "y", "x"}
            World axis name.
        """
        assert self._target is not None
        self._commit(self._target.state.flipped(name, frame=self._frame))

    def _set_pivot(self, point: npt.ArrayLike) -> None:
        """Move the center of rotation, without moving the volume.

        This is not an undo step: the transform does not change, so undoing it would look
        like doing nothing.

        Parameters
        ----------
        point : (3,) array-like
            World point in `(z, y, x)` order.
        """
        assert self._target is not None
        self._target.state = self._target.state.with_pivot(point)
        self._sync_widgets()

    def _map_source_point(
        self, point: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return where the current transform puts a point of the committed volume.

        Parameters
        ----------
        point : (3,) numpy.ndarray
            World point of the committed volume, in `(z, y, x)` order.

        Returns
        -------
        (3,) numpy.ndarray
            Transformed world point.
        """
        assert self._target is not None
        matrix = self._target.state.matrix()
        return matrix[:3, :3] @ point + matrix[:3, 3]

    def _on_center_origin(self) -> None:
        """Move the center of rotation to the volume's first voxel."""
        assert self._target is not None
        origin = self._target.session.source.fusi.origin
        self._set_pivot(
            self._map_source_point(np.array([origin[name] for name in WORLD_AXES]))
        )

    def _on_center_cursor(self) -> None:
        """Move the center of rotation to the cursor."""
        self._set_pivot(np.asarray(self.viewer.cursor.position[-3:], dtype=np.float64))

    def _on_center_geometry(self) -> None:
        """Move the center of rotation to the center of the volume's box."""
        assert self._target is not None
        center = self._target.session.source.fusi.affine.geometry_center.to_numpy()
        self._set_pivot(self._map_source_point(center))

    def _on_marker_toggled(self, checked: bool) -> None:
        """Add or remove the center of rotation marker.

        Parameters
        ----------
        checked : bool
            Whether to show the marker.
        """
        marker = self._marker_layer
        if not checked:
            self._marker_layer = None
            if marker is not None and marker in self.viewer.layers:
                self.viewer.layers.remove(marker)
            return
        if self._target is None:
            return
        active = self.viewer.layers.selection.active
        self._marker_layer = self.viewer.add_points(  # type: ignore
            self._target.state.pivot()[np.newaxis],
            name="Center of rotation",
            size=float(self._target.voxel_steps.max()) * 5.0,
            symbol="cross",
            face_color="yellow",
        )
        self.viewer.layers.selection.active = active

    def _sync_marker(self) -> None:
        """Keep the marker on the pivot, and notice if the user deleted it."""
        marker = self._marker_layer
        if marker is None:
            return
        if marker not in self.viewer.layers:
            self._marker_layer = None
            with QSignalBlocker(self._center_marker_check):
                self._center_marker_check.setChecked(False)
        elif self._target is not None:
            marker.data = self._target.state.pivot()[np.newaxis]

    def _start_worker(
        self,
        function: Callable[..., Any],
        *args: Any,
        on_done: Callable[[Any], None],
        **kwargs: Any,
    ) -> None:
        """Run a function in a worker thread, locking the panel meanwhile.

        Parameters
        ----------
        function : collections.abc.Callable
            Function to run. It must not touch Qt or napari.
        *args : Any
            Positional arguments of `function`.
        on_done : collections.abc.Callable
            Called on the GUI thread with the result.
        **kwargs : Any
            Keyword arguments of `function`.
        """
        # Without `ignore_errors`, the worker re-raises after emitting `errored`, which
        # only adds a stray traceback since the error is reported below.
        worker = thread_worker(function, ignore_errors=True)(*args, **kwargs)

        def _on_returned(result: Any) -> None:
            self._worker = None
            on_done(result)
            self._sync_widgets()

        def _on_errored(error: Exception) -> None:
            self._worker = None
            show_error(str(error))
            self._sync_widgets()

        worker.returned.connect(_on_returned)
        worker.errored.connect(_on_errored)
        self._worker = worker
        self._sync_widgets()
        worker.start()

    def _apply(self) -> None:
        """Write the transform into the layer's affine, in a worker thread."""
        target = self._target
        if target is None or self._worker is not None or target.state.is_identity():
            return
        session = target.session
        self._start_worker(
            build_applied_layer,
            session.source,
            target.state.matrix(),
            layer_type=session.layer_type,
            fill_value=session.fill_value,
            name=session.layer.name,
            committed_labels=session.committed_labels,
            on_done=lambda result: self._on_applied(target, result),
        )

    def _on_applied(self, target: _Target, result: Any) -> None:
        """Commit an applied transform, unless its layer is gone.

        Parameters
        ----------
        target : _Target
            Target the worker ran for.
        result : tuple
            Result of `build_applied_layer`.
        """
        if target.session.layer not in self.viewer.layers:
            return
        data, kwargs, _ = result
        state = TransformState.identity(center=target.state.pivot())
        with self._keeping_slice():
            target.session.adopt(data, kwargs)
        target.state = state
        target.history.clear(state)
        target.voxel_steps = compute_tight_grid(
            target.session.source, np.eye(4)
        ).spacing

    def _resample(self) -> None:
        """Create a new layer with the transformed, resampled voxels."""
        target = self._target
        if target is None or self._worker is not None:
            return
        session = target.session
        reference_layer = self._reference_combo.currentData()
        self._start_worker(
            build_resampled_layer,
            session.source,
            target.state.matrix(),
            layer_type=session.layer_type,
            fill_value=session.fill_value,
            name=f"{session.layer.name} (resampled)",
            committed_labels=session.committed_labels,
            reference=(
                reference_layer.metadata["xarray"]
                if reference_layer is not None
                else None
            ),
            on_done=lambda result: self._on_resampled(session.layer, result),
        )

    def _on_resampled(self, layer: Image | Labels, result: Any) -> None:
        """Add a resampled layer to the viewer.

        Parameters
        ----------
        layer : napari.layers.Image or napari.layers.Labels
            Layer it was resampled from.
        result : tuple
            Result of `build_resampled_layer`.
        """
        data, kwargs, layer_type = result
        if isinstance(layer, Image):
            kwargs["contrast_limits"] = tuple(layer.contrast_limits)
        getattr(self.viewer, f"add_{layer_type}")(data, **kwargs)
