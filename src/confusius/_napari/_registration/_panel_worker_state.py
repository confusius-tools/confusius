"""Worker-state cleanup helpers for the napari registration panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from napari.utils.notifications import show_error

from confusius._napari._registration._panel_progress import (
    teardown_volume_progress,
    teardown_volumewise_progress,
)

if TYPE_CHECKING:
    from confusius._napari._registration._panel import RegistrationPanel


def on_registration_failed(panel: RegistrationPanel, exc: BaseException) -> None:
    """Handle a failed worker execution.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose in-flight state should be cleaned up.
    exc : BaseException
        Exception raised by the worker.
    """
    teardown_volume_progress(panel)
    teardown_volumewise_progress(panel, remove_layer=True)
    panel._set_error(str(exc))
    show_error(str(exc))
