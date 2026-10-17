"""Allow `python -m zallery` alongside the `zallery` console script."""

from zallery.cli import main

raise SystemExit(main())
