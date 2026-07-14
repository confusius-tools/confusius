"""Generate documentation images for the Atlases user guide.

The Allen Mouse Brain atlas (`allen_mouse_100um`) is fetched via
`confusius.datasets.fetch_brainglobe_atlas`, which downloads it through the
BrainGlobe Atlas API on first run and caches it in BrainGlobe's own atlas cache
(`~/.brainglobe`, shared with other BrainGlobe tools); subsequent runs use the cache.

Usage
-----
Run from the project root::

    uv run docs/images/atlas/generate.py

All images are saved to docs/images/atlas/.

Notes
-----
- The napari mesh screenshot is taken programmatically. Review it after the script
  finishes and retake manually (File > Save Screenshot in napari) if the canvas
  renders poorly (e.g., all-black).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import napari
from napari.qt import get_qapp
from qtpy.QtCore import Qt
from rich.console import Console

import confusius as cf

HERE = Path(__file__).parent

_ATLAS_NAME = "allen_mouse_100um"
_SLICE_Z = 6.0  # Coronal slice (mm) used for the annotation-contour figure.
_MESH_REGION = "root"  # Whole-brain surface mesh for the napari screenshot.

# Camera picked interactively in napari for the Allen atlas mesh screenshot.
_MESH_CAMERA_ANGLES = (168.6206378127608, -54.99924316597373, 203.34564635933614)
_MESH_UP_DIRECTION = (
    0.22729948558043406,
    -0.9641291993119706,
    0.13707600405953269,
)
_MESH_CAMERA_CENTER: tuple[float, float, float] | None = None
# Screenshot framing lands a bit tighter than the interactive napari view, so back
# the picked zoom off slightly here.
_MESH_CAMERA_ZOOM = 52.0

_SAVEFIG_KWARGS = {"dpi": 150, "bbox_inches": "tight", "transparent": True}

console = Console()


def _section(title: str) -> None:
    console.rule(f"[bold]{title}[/bold]")


def _ok(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def _warn(message: str) -> None:
    console.print(f"[yellow]![/yellow] {message}")


def _napari_screenshot(viewer: "napari.Viewer", path: str) -> None:
    """Take a full-window napari screenshot without displaying the window.

    QWidget.grab() (used by canvas_only=False) requires the widget to have been
    shown at least once for its layout to be initialised. Setting
    WA_DontShowOnScreen before show() runs the full Qt layout pipeline without
    actually mapping the window on screen, so the tiling WM never sees or resizes it.
    """
    win = viewer.window._qt_window
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    win.show()
    win.resize(1100, 750)
    get_qapp().processEvents()
    try:
        viewer.camera.up_direction = _MESH_UP_DIRECTION
    except Exception:
        pass
    viewer.camera.angles = _MESH_CAMERA_ANGLES
    if _MESH_CAMERA_CENTER is not None:
        viewer.camera.center = _MESH_CAMERA_CENTER
    viewer.camera.zoom = _MESH_CAMERA_ZOOM
    get_qapp().processEvents()
    viewer.screenshot(path=path, canvas_only=False)


# ---------------------------------------------------------------------------
# Fetch the atlas
# ---------------------------------------------------------------------------

_section("Load Atlas")
console.print(f"Fetching BrainGlobe atlas {_ATLAS_NAME!r}")
atlas = cf.datasets.fetch_brainglobe_atlas(_ATLAS_NAME)
console.print(
    f"  {atlas.atlas.reference.dims}, shape {dict(atlas.atlas.reference.sizes)}"
)

# ---------------------------------------------------------------------------
# 1. Reference slice with region annotation contours
# ---------------------------------------------------------------------------

_section("Annotation contours")

for bg_color, suffix in [("white", "light"), ("black", "dark")]:
    plotter = cf.plotting.plot_volume(
        atlas.atlas.reference.sel(z=slice(_SLICE_Z, _SLICE_Z)),
        show_colorbar=False,
        bg_color=bg_color,
    )
    plotter.add_contours(atlas.atlas.annotation.sel(z=slice(_SLICE_Z, _SLICE_Z)))
    plotter.savefig(str(HERE / f"atlas-annotation-{suffix}.png"), **_SAVEFIG_KWARGS)
    plotter.close()

plt.close("all")
_ok("Saved atlas-annotation-light.png and atlas-annotation-dark.png")

# ---------------------------------------------------------------------------
# 2. napari: whole-brain surface mesh
# ---------------------------------------------------------------------------

_section("napari mesh")

try:
    surface = atlas.atlas.get_mesh(_MESH_REGION)
    viewer = napari.Viewer(ndisplay=3, show=False)
    viewer.add_surface(surface, colormap="gray", name=f"{_MESH_REGION} mesh")
    _napari_screenshot(viewer, str(HERE / "atlas-mesh-root.png"))
    viewer.close()
    _ok("Saved atlas-mesh-root.png")
except Exception as exc:
    _warn(f"atlas-mesh-root.png failed: {exc}")

# ---------------------------------------------------------------------------

_ok("Done! Rebuild docs with `just docs` to preview changes")
