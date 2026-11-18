---
icon: lucide/funnel
---

# `confusius.extract`

Mask extraction, label aggregation, and reconstruction are direct re-exports from
the standalone VoxelData package. Use the public functions or the flat accessor
methods `.vd.extract_with_mask`, `.vd.extract_with_labels`, and `.vd.unmask`; `.fusi.extract` is removed. They preserve existing timing metadata without adding acquisition
defaults or validating fUSI timing semantics. ConfUSIus constructors, loaders, and
explicit acquisition-aware validation retain that responsibility.

Extraction works out of the box in VoxelData using Xarray. Flox acceleration and
Dask-backed lazy arrays are optional backends there; ConfUSIus continues to install
its existing analysis dependencies. VoxelData offers independent `accel` (flox)
and `lazy` (Dask) extras, or `all` for both; no extra is needed for extraction.

Binary masks flatten all their dimensions into `space`, including extra feature
axes used by decomposition and multipose GLM. Labels are spatial-only, optionally
stacked along `mask`. A shared spatial map requires identical physical grids
across geometry dependencies not included in the mask.
Label aggregation supports overlapping layers and negative or repeated layer ids.
`unmask` reconstructs eagerly and positionally, preserving the mask's native labels,
units, geometry, and named frames. With flox, median may need explicit spatial
rechunking.

::: confusius.extract
