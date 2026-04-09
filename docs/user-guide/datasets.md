---
icon: lucide/database
---

# Datasets

!!! info "Coming soon"
    This page is currently under construction. The `datasets` module provides
    functions for fetching publicly available fUSI datasets, with automatic
    caching and offline support.

    The default cache location is platform-specific (e.g.
    `~/.cache/confusius` on Linux), and can be overridden via the
    `CONFUSIUS_DATA` environment variable or the `data_dir` argument of each
    fetcher.

    **Available datasets:**

    - [`fetch_nunez_elizalde_2022`][confusius.datasets.fetch_nunez_elizalde_2022]:
      Fetch the Nunez-Elizalde et al. (2022) dataset, containing simultaneous
      neural activity and cerebral blood volume recordings in awake mice,
      converted to fUSI-BIDS format. Supports filtering by subject, session,
      and task. Returns the path to the local BIDS root directory.

    **Utilities:**

    - [`list`][confusius.datasets.list]: Print a table of all available datasets with
      their download sizes.
    - [`get_datasets_dir`][confusius.datasets.get_datasets_dir]: Resolve the confusius
      data directory using the priority chain: function argument →
      `CONFUSIUS_DATA` environment variable → platform cache directory.

    Please refer to the [API Reference](../api/datasets.md) and
    [Roadmap](../roadmap.md) for more information.
