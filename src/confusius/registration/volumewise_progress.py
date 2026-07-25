"""Progress reporting protocol for `register_volumewise`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import xarray as xr

    from confusius.registration import RegistrationDiagnostics


class VolumewiseProgressReporter(Protocol):
    """Duck-typed contract for `register_volumewise` progress reporting.

    Implementations may be called from worker threads when volumewise
    registration runs in parallel. Any GUI updates must therefore be marshalled
    via thread-safe mechanisms such as Qt signals.
    """

    def frame_completed(
        self,
        frame_index: int,
        registered_frame: xr.DataArray,
        diagnostics: RegistrationDiagnostics,
    ) -> None:
        """Report that one frame finished and provide its registered output.

        Parameters
        ----------
        frame_index : int
            Index of the completed frame.
        registered_frame : xarray.DataArray
            Registered frame output.
        diagnostics : confusius.registration.RegistrationDiagnostics
            Diagnostics collected for the completed frame.
        """
        ...

    def close(self) -> None:
        """Report that the full volumewise run has ended."""
        ...
