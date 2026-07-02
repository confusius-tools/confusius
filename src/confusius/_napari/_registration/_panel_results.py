"""Result-handling helpers for the napari registration panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr
from napari.layers.utils.layer_utils import calc_data_range
from napari.utils.notifications import show_info

from confusius._napari._registration._panel_progress import teardown_volumewise_progress
from confusius._napari._registration._panel_transforms import refresh_transform_controls
from confusius._napari._registration._transform_payloads import (
    make_affine_transform_payload,
    make_bspline_transform_payload,
)
from confusius._napari._registration._panel_utils import (
    _gamma_needs_reset,
    _get_image_display_kwargs_from_layer,
)
from confusius.plotting.napari import plot_napari

if TYPE_CHECKING:
    import numpy.typing as npt

    from confusius._napari._registration._panel import (
        RegistrationPanel,
        VolumeRegistrationRunPayload,
        VolumewiseRegistrationRunPayload,
    )
    from confusius.registration import RegistrationDiagnostics


def coerce_volume_registration_payload(
    payload: dict[str, Any] | "VolumeRegistrationRunPayload",
) -> "VolumeRegistrationRunPayload":
    """Return a typed between-scan registration payload.

    Parameters
    ----------
    payload : dict[str, Any] or VolumeRegistrationRunPayload
        Untyped or typed payload captured when the worker started.

    Returns
    -------
    VolumeRegistrationRunPayload
        Typed payload for a between-scan registration run.

    Raises
    ------
    ValueError
        If `payload["operation"]` is not `"register_volume"`.
    """
    if payload.get("operation") != "register_volume":
        raise ValueError("Expected a register_volume payload.")
    return cast("VolumeRegistrationRunPayload", payload)


def coerce_volumewise_registration_payload(
    payload: dict[str, Any] | "VolumewiseRegistrationRunPayload",
) -> "VolumewiseRegistrationRunPayload":
    """Return a typed within-scan registration payload.

    Parameters
    ----------
    payload : dict[str, Any] or VolumewiseRegistrationRunPayload
        Untyped or typed payload captured when the worker started.

    Returns
    -------
    VolumewiseRegistrationRunPayload
        Typed payload for a within-scan registration run.

    Raises
    ------
    ValueError
        If `payload["operation"]` is not `"register_volumewise"`.
    """
    if payload.get("operation") != "register_volumewise":
        raise ValueError("Expected a register_volumewise payload.")
    return cast("VolumewiseRegistrationRunPayload", payload)


def finalize_registration_layer(
    panel: "RegistrationPanel",
    *,
    payload: "VolumeRegistrationRunPayload | VolumewiseRegistrationRunPayload",
    registered: xr.DataArray,
    layer_name: str,
    metadata: dict[str, Any],
    registration_status: Literal["completed", "aborted"],
) -> None:
    """Attach registration metadata and add or update the result layer.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose viewer should receive the final layer.
    payload : VolumeRegistrationRunPayload or VolumewiseRegistrationRunPayload
        Typed UI snapshot captured before the worker started.
    registered : xarray.DataArray
        Registered output returned by the worker.
    layer_name : str
        Name to use when creating a new result layer.
    metadata : dict[str, Any]
        Extra layer metadata to attach.
    registration_status : {"completed", "aborted"}
        Final run status used for user feedback and layer naming.
    """
    metadata["registration_operation"] = payload["operation"]
    metadata["registration_parameters"] = payload.copy()

    source_layer = panel._get_layer_by_name(payload["moving_layer_name"])
    display_kwargs = (
        _get_image_display_kwargs_from_layer(source_layer)
        if source_layer is not None
        else {}
    )
    if _gamma_needs_reset(payload.get("scale", "off")):
        display_kwargs["gamma"] = 1.0
    if payload["operation"] == "register_volume":
        display_kwargs["colormap"] = "cyan"
        display_kwargs["blending"] = "additive"
    contrast_limits = tuple(calc_data_range(registered.data))

    if payload["operation"] == "register_volume" and panel._progress_layer is not None:
        layer = panel._progress_layer
        panel._set_image_layer_data(layer, np.asarray(registered.data))
        if hasattr(layer, "contrast_limits"):
            layer.contrast_limits = contrast_limits
        panel._progress_bridge = None
        panel._progress_layer = None
    elif (
        payload["operation"] == "register_volumewise"
        and panel._volumewise_progress_layer is not None
    ):
        layer = panel._volumewise_progress_layer
        panel._set_image_layer_data(layer, np.asarray(registered.data))
        if hasattr(layer, "contrast_limits"):
            layer.contrast_limits = contrast_limits
        teardown_volumewise_progress(panel, remove_layer=False)
    else:
        _, layer = plot_napari(
            registered,
            viewer=panel.viewer,
            name=layer_name,
            show_colorbar=False,
            contrast_limits=contrast_limits,
            **display_kwargs,
        )
    layer.metadata.update(metadata)
    layer.metadata["xarray"] = registered
    panel.viewer.layers.selection.active = layer
    refresh_transform_controls(panel)

    if payload["operation"] == "register_volumewise":
        panel._progress.setValue(panel._progress.maximum())

    if registration_status == "aborted":
        layer.name = f"{layer.name} (aborted)"
        panel._set_error("Registration aborted; added partial result.")
        show_info(f"Registration aborted; added partial layer: {layer.name}")
    else:
        show_info(f"Added registered layer: {layer.name}")


def on_registration_finished(
    panel: "RegistrationPanel",
    payload: dict[str, Any],
    result: object,
) -> None:
    """Dispatch a finished registration callback to the typed handler.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel that started the worker.
    payload : dict[str, Any]
        Untyped compatibility payload captured when the worker started.
    result : object
        Worker result to forward to the operation-specific handler.

    Raises
    ------
    ValueError
        If `payload["operation"]` is not recognized.
    """
    if payload.get("operation") == "register_volume":
        on_volume_registration_finished(
            panel,
            coerce_volume_registration_payload(payload),
            cast(
                "tuple[xr.DataArray, npt.NDArray[np.floating] | xr.DataArray, RegistrationDiagnostics]",
                result,
            ),
        )
        return
    if payload.get("operation") == "register_volumewise":
        on_volumewise_registration_finished(
            panel,
            coerce_volumewise_registration_payload(payload),
            cast("xr.DataArray", result),
        )
        return
    raise ValueError(f"Unknown registration operation: {payload.get('operation')!r}.")


def on_volume_registration_finished(
    panel: "RegistrationPanel",
    payload: "VolumeRegistrationRunPayload",
    result: tuple[
        xr.DataArray,
        "npt.NDArray[np.floating] | xr.DataArray",
        "RegistrationDiagnostics",
    ],
) -> None:
    """Add a between-scan registration result back to the viewer.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel that started the worker.
    payload : VolumeRegistrationRunPayload
        Typed UI snapshot captured before the worker started.
    result : tuple
        Registered volume, estimated transform, and diagnostics.
    """
    registered, transform, diagnostics = result
    registered = registered.copy(deep=False)
    registered.attrs = registered.attrs.copy()
    registered.attrs["registration_transform"] = transform
    registered.attrs["registration_diagnostics"] = diagnostics
    registered.attrs["registration_operation"] = payload["operation"]
    registered.attrs["registration_status"] = diagnostics.status
    metadata: dict[str, Any] = {
        "registration_transform": transform,
        "registration_diagnostics": diagnostics,
        "registration_status": diagnostics.status,
    }
    transform_name = panel._make_unique_transform_name(
        f"{payload['moving_layer_name']} → {payload['fixed_layer_name']} ({payload['transform']})"
    )
    if isinstance(transform, np.ndarray):
        metadata["confusius_transform"] = make_affine_transform_payload(
            np.asarray(transform, dtype=float),
            reference=registered,
            source_layer_name=payload["moving_layer_name"],
            target_layer_name=payload["fixed_layer_name"],
            operation=payload["operation"],
            transform_model=payload["transform"],
            metric=payload["metric"],
            diagnostics=diagnostics,
            name=transform_name,
        )
    else:
        metadata["confusius_transform"] = make_bspline_transform_payload(
            transform,
            reference=registered,
            source_layer_name=payload["moving_layer_name"],
            target_layer_name=payload["fixed_layer_name"],
            operation=payload["operation"],
            transform_model=payload["transform"],
            metric=payload["metric"],
            diagnostics=diagnostics,
            name=transform_name,
        )
    finalize_registration_layer(
        panel,
        payload=payload,
        registered=registered,
        layer_name=panel._volume_result_layer_name(
            payload["moving_layer_name"],
            payload["fixed_layer_name"],
            transform_model=payload["transform"],
        ),
        metadata=metadata,
        registration_status=diagnostics.status,
    )


def on_volumewise_registration_finished(
    panel: "RegistrationPanel",
    payload: "VolumewiseRegistrationRunPayload",
    result: xr.DataArray,
) -> None:
    """Add a within-scan registration result back to the viewer.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel that started the worker.
    payload : VolumewiseRegistrationRunPayload
        Typed UI snapshot captured before the worker started.
    result : xarray.DataArray
        Motion-corrected time series returned by the worker.
    """
    registered = result.copy(deep=False)
    registered.attrs = registered.attrs.copy()
    registered.attrs["registration_operation"] = payload["operation"]
    motion_params = registered.attrs.get("motion_params")
    registration_status = "completed"
    if motion_params is not None:
        try:
            statuses = motion_params["status"]
        except Exception:  # noqa: BLE001
            statuses = None
        if statuses is not None and bool((statuses == "aborted").any()):
            registration_status = "aborted"
    finalize_registration_layer(
        panel,
        payload=payload,
        registered=registered,
        layer_name=panel._volumewise_result_layer_name(payload["moving_layer_name"]),
        metadata={
            "motion_params": motion_params,
            "reference_time": payload["reference_time"],
        },
        registration_status=registration_status,
    )
