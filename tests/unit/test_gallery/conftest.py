"""Shared fixtures for gallery unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def gallery_tree(tmp_path: Path) -> Path:
    """Create a minimal docs/examples/-style tree under ``tmp_path``."""
    root = tmp_path / "examples"
    (root / "io").mkdir(parents=True)
    (root / "io" / "_section.md").write_text("# Input/Output\n\nIO examples.\n")
    (root / "io" / "load_autc.py").write_text(
        "# %% [markdown]\n# # Load AUTC\n\n# %%\nprint('hi')\n"
    )
    (root / "io" / "_skipped.py").write_text("# leading underscore should be skipped\n")
    (root / "glm").mkdir()
    (root / "glm" / "first_level.py").write_text(
        "# %% [markdown]\n# # First level GLM\n\n# %%\npass\n"
    )
    return root
