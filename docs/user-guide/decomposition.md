---
icon: lucide/layers
---

# Decomposition

!!! info "Coming soon"
    This page is currently under construction. The decomposition module currently
    provides:

    - [`PCA`][confusius.decomposition.PCA]: Principal component analysis for
      `(time, ...)` `xarray.DataArray` inputs while preserving metadata through
      `transform` and `inverse_transform`.
    - [`NMF`][confusius.decomposition.NMF]: Non-negative matrix factorization
      for `(time, ...)` `xarray.DataArray` inputs while preserving metadata
      through `transform` and `inverse_transform`.
    - [`FastICA`][confusius.decomposition.FastICA]: Fast independent component
      analysis for `(time, ...)` `xarray.DataArray` inputs while preserving
      metadata through `transform` and `inverse_transform`.

    Please refer to the [API Reference](../api/decomposition.md) for more information.
