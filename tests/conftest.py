"""Global fixtures shared across all tests."""

import os
import sys

import numpy as np
import pytest

# Configure the Qt/OpenGL backend for headless testing.
# Must be set before any Qt import so that QApplication does not try to
# connect to a display server and abort.
if sys.platform == "win32":
    # Windows runners (desktop and CI) always have a display, so we keep the
    # native "windows" platform plugin. Force the Mesa llvmpipe software
    # renderer so that Qt can create OpenGL 3.0+ contexts without a GPU.
    os.environ.setdefault("QT_OPENGL", "software")
else:
    # Linux/macOS CI usually has no display; use the offscreen Qt backend.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _patch_vispy_gl_lib() -> None:
    """Replace vispy's OpenGL library with Mesa on Windows.

    The napari pytest plugin imports vispy before conftest.py runs, so
    ``VISPY_GL_LIB`` set here is too late. Instead we hot-swap the ctypes
    library object that vispy already loaded (the system ``opengl32.dll``,
    which only provides OpenGL 1.1) with PyQt5's bundled Mesa llvmpipe
    ``opengl32sw.dll`` (OpenGL 3.0+). This allows GL extension functions like
    ``glBindFramebuffer`` to be resolved without ``wglGetProcAddress``.
    """
    if sys.platform != "win32":
        return

    try:
        import ctypes
        import importlib.util
        from pathlib import Path

        spec = importlib.util.find_spec("PyQt5")
        if spec is None or spec.origin is None:
            return
        mesa = Path(spec.origin).resolve().parent / "Qt5" / "bin" / "opengl32sw.dll"
        if not mesa.exists():
            return
        import vispy.gloo.gl.gl2 as _gl2

        _gl2._lib = ctypes.windll.LoadLibrary(str(mesa))
        # Disable the wglGetProcAddress fallback — Mesa exports all GL 3.0+
        # functions directly, and the old wglGetProcAddress still points at the
        # system opengl32.dll.
        _gl2._have_get_proc_address = False
    except Exception:
        pass


_patch_vispy_gl_lib()


@pytest.fixture(scope="session")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
