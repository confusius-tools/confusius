"""Unit tests for the napari plugin manifest."""

from npe2 import PluginManifest


def test_manifest_python_names_resolve() -> None:
    """Every command in the manifest points at a real function."""
    PluginManifest.from_file("src/confusius/_napari/napari.yaml").validate_imports()
