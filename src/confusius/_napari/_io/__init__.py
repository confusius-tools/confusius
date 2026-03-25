"""Napari reader and writer entry points."""

from confusius._napari._io._readers import read_nifti, read_scan, read_zarr
from confusius._napari._io._writers import write_nifti, write_zarr

__all__ = ["read_nifti", "read_scan", "read_zarr", "write_nifti", "write_zarr"]
