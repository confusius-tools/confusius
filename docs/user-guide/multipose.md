---
icon: lucide/waypoints
---

# Multi-Pose Data

!!! info "Coming soon"
    This page is currently under construction. The `multipose` module provides tools for
    processing multi-pose fUSI data, including consolidating multiple poses into a single
    volume, slice timing correction, and other multi-pose specific operations:

    **Pose consolidation:**

    - [`consolidate_poses`][confusius.multipose.consolidate_poses]: Merge `pose` and
      sweep dimensions into a single axis ordered by physical position. Computes
      positions for each `(pose, sweep)` voxel using per-pose affine transformations and
      reindexes the data along a consolidated sweep axis.

    Please refer to the [API Reference](../api/multipose.md) for more information.
