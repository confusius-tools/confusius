"""Exceptions raised by registration workflows."""


class RegistrationAbortedError(RuntimeError):
    """Raised when a registration run is cancelled before completion."""
