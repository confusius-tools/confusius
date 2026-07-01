"""Shared utility helpers for the napari registration panel."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr
from qtpy.QtCore import QRegularExpression
from qtpy.QtGui import QValidator
from qtpy.QtWidgets import QDoubleSpinBox, QSizePolicy, QWidget

from confusius._dims import SPATIAL_DIMS, TIME_DIM
from confusius.xarray.scale import db_scale, power_scale

if TYPE_CHECKING:
    import napari
    from napari.layers import Layer


@contextmanager
def _preserve_view(viewer: "napari.Viewer") -> Iterator[None]:
    """Keep the viewer camera and dims state across a block that adds layers.

    Adding image layers makes napari recompute `camera.center` and re-apply
    `napari.imshow`'s default `ndisplay`/`order` to the dims, which yanks the canvas
    back to a default framing. Wrapping the layer creation in this context manager
    snapshots the current pan, zoom, rotation, and slider position and restores them
    once the block exits, so the user keeps the view they were on when starting a
    registration run.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer whose camera and dims state are snapshotted and restored.

    Yields
    ------
    None
        Control returns to the wrapped block; the saved state is restored when
        it exits, including on early return or exception.
    """
    camera = viewer.camera
    dims = viewer.dims
    center = tuple(camera.center)
    zoom = camera.zoom
    angles = tuple(camera.angles)
    ndisplay = dims.ndisplay
    order = tuple(dims.order)
    current_step = tuple(dims.current_step)
    try:
        yield
    finally:
        dims.ndisplay = ndisplay
        dims.order = order
        dims.current_step = current_step
        camera.center = center
        camera.zoom = zoom
        camera.angles = angles


def _default_dims_for_ndim(ndim: int) -> tuple[str, ...]:
    """Return fallback dimension names for a raw napari layer.

    Parameters
    ----------
    ndim : int
        Number of array dimensions.

    Returns
    -------
    tuple of str
        Default dimension names compatible with ConfUSIus conventions when possible.
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
        Layer property such as `scale`, `translate`, `units`, or `axis_labels`.
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


def _reconstruct_layer_dataarray(layer: "Layer") -> xr.DataArray:
    """Reconstruct a DataArray from the current napari layer state.

    Parameters
    ----------
    layer : napari.layers.Layer
        Napari layer to convert.

    Returns
    -------
    xarray.DataArray
        DataArray reconstructed from the layer's current axis labels, scale, translate,
        and units.
    """
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


def _layer_supports_registration_source(layer: "Layer") -> bool:
    """Return whether `layer` can be converted to a registration source.

    ConfUSIus-managed layers carry the original `xarray.DataArray` in metadata. For
    plain napari image layers we can reconstruct one from eager NumPy data. Lazy
    non-NumPy layers (for example the video panel's frame-on-demand array) are
    intentionally excluded: forcing `np.asarray` on them can trigger expensive decoding
    or backend errors while the registration panel is merely refreshing.
    """
    if layer.metadata.get("xarray") is not None:
        return True
    if layer.metadata.get("confusius_cached_registration_xarray") is not None:
        return True
    return isinstance(layer.data, np.ndarray)


def _get_source_dataarray(layer: "Layer") -> xr.DataArray:
    """Return the stable source DataArray for a napari layer.

    Parameters
    ----------
    layer : napari.layers.Layer
        Napari layer to convert.

    Returns
    -------
    xarray.DataArray
        Original ConfUSIus DataArray when present in `layer.metadata`, otherwise a
        cached reconstruction captured before later manual napari transforms mutate the
        layer pose.

    Raises
    ------
    TypeError
        If the layer is backed by a lazy non-NumPy array that the registration
        panel should ignore.
    """
    existing = layer.metadata.get("xarray")
    if existing is not None:
        return cast("xr.DataArray", existing)

    cached = layer.metadata.get("confusius_cached_registration_xarray")
    if cached is not None:
        return cast("xr.DataArray", cached)

    if not isinstance(layer.data, np.ndarray):
        raise TypeError(
            f"Layer {layer.name!r} is not backed by eager NumPy data and cannot be used "
            "for registration."
        )

    reconstructed = _reconstruct_layer_dataarray(layer)
    layer.metadata["confusius_cached_registration_xarray"] = reconstructed
    return reconstructed


def _prepare_between_scan_data(data: xr.DataArray) -> xr.DataArray:
    """Return a spatial-only DataArray for between-scan registration.

    Parameters
    ----------
    data : xarray.DataArray
        Input layer data.

    Returns
    -------
    xarray.DataArray
        Spatial-only data. If the input has a time dimension, it is averaged over time
        with attributes preserved.
    """
    if TIME_DIM not in data.dims:
        return data
    averaged = data.mean(dim=TIME_DIM, keep_attrs=True)
    averaged.attrs = data.attrs.copy()
    return averaged


def _apply_registration_scale(
    data: xr.DataArray, scale_mode: Literal["off", "dB", "sqrt"]
) -> xr.DataArray:
    """Apply optional intensity preprocessing for registration.

    Parameters
    ----------
    data : xarray.DataArray
        Input data.
    scale_mode : {"off", "dB", "sqrt"}
        Intensity scaling mode used before registration.

    Returns
    -------
    xarray.DataArray
        Preprocessed data.

    Raises
    ------
    ValueError
        If `scale_mode` is not recognized.
    """
    if scale_mode == "off":
        return data
    if scale_mode == "dB":
        return db_scale(data)
    if scale_mode == "sqrt":
        return power_scale(data, exponent=0.5)
    raise ValueError(f"Unknown registration scale mode: {scale_mode}.")


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


def _should_reset_gamma(scale_mode: str) -> bool:
    """Return whether registration preview/result gamma should be reset.

    When using intensity scaling, the gamma of the preview and result layers is forced
    to 1.0 to avoid double scaling. When scaling is off, the original layer gamma is
    preserved.

    Parameters
    ----------
    scale_mode : str
        Registration intensity scaling mode.

    Returns
    -------
    bool
        Whether preview/result layers should force `gamma=1.0`.
    """
    return scale_mode != "off"


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

    def validate(
        self, input: str | None, pos: int
    ) -> tuple[QValidator.State, str, int]:
        """Validate decimals and scientific notation while the user types.

        Parameters
        ----------
        input : str, optional
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
        normalized = input or ""
        if normalized in {"", "+", "-", ".", "+.", "-."}:
            return (QValidator.State.Intermediate, normalized, pos)
        if self._ACCEPTABLE_RE.match(normalized).hasMatch():
            return (QValidator.State.Acceptable, normalized, pos)
        if self._INTERMEDIATE_RE.match(normalized).hasMatch():
            return (QValidator.State.Intermediate, normalized, pos)
        return (QValidator.State.Invalid, normalized, pos)

    def valueFromText(self, text: str | None) -> float:
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

    def textFromValue(self, v: float) -> str:
        """Format values compactly, using scientific notation when helpful.

        Parameters
        ----------
        v : float
            Value to format.

        Returns
        -------
        str
            Formatted text.
        """
        return f"{v:.12g}"

    def stepBy(self, steps: int) -> None:
        """Apply additive stepping using the configured single-step size.

        Parameters
        ----------
        steps : int
            Number of steps to apply.
        """
        self.setValue(self.value() + (steps * self.singleStep()))
