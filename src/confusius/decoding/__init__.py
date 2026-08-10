"""Pattern decoding for fUSI DataArrays.

This package provides decoders in the spirit of
[`nilearn.decoding`](https://nilearn.github.io/dev/decoding/):

- [`SearchLight`][confusius.decoding.SearchLight]: per-voxel local decoding, also
  known as "spotlight" decoding.

Portions of this package are inspired by `nilearn.decoding`, which is licensed under
the BSD-3-Clause License. See `NOTICE` for details.
"""

from confusius.decoding.searchlight import SearchLight

__all__ = ["SearchLight"]
