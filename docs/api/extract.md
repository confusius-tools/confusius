---
icon: lucide/funnel
---

# `confusius.extract`

Mask extraction, label aggregation, and reconstruction delegate to the standalone
VoxelData package. The public functions and `.fusi.extract` methods remain
available; ConfUSIus applies acquisition defaults and validation before delegation.

Extraction works out of the box in VoxelData using Xarray. Flox acceleration and
Dask-backed lazy arrays are optional backends there; ConfUSIus continues to install
its existing analysis dependencies. No extraction extra is required.

Binary masks flatten all their dimensions into `space`, including extra feature
axes used by decomposition and multipose GLM. Labels are spatial-only, optionally
stacked along `mask`. A shared spatial map requires identical physical grids
across geometry dependencies not included in the mask.
Label aggregation supports overlapping layers and negative or repeated layer ids.
`unmask` reconstructs eagerly and positionally, preserving the mask's native labels,
units, geometry, and named frames. With flox, median may need explicit spatial
rechunking.

::: confusius.extract
