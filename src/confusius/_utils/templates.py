"""Registry of the fUSI templates shipped in `confusius.datasets`.

Kept free of heavy imports so the CLI can list and validate template names at
parse time; the fetcher is resolved only when a template is actually fetched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import xarray as xr


class TemplateSpec(TypedDict):
    """One fUSI template entry."""

    fetcher: str
    """Name of the `confusius.datasets.fetch_template_*` function returning it."""

    reference_atlas: str
    """Human-readable name of the atlas space the template is aligned to."""


FUSI_TEMPLATES: dict[str, TemplateSpec] = {
    "huang_2025": {
        "fetcher": "fetch_template_huang_2025",
        "reference_atlas": "Allen Mouse Brain Atlas (CCFv3)",
    },
    "pepe_mariani_2026": {
        "fetcher": "fetch_template_pepe_mariani_2026",
        "reference_atlas": "Allen Mouse Brain Atlas (CCFv3)",
    },
}
"""fUSI templates by name, named after their `confusius.datasets` fetcher."""


def fetch_template(name: str, **kwargs: object) -> xr.DataArray:
    """Fetch a fUSI template by name.

    Parameters
    ----------
    name : str
        A key of [`FUSI_TEMPLATES`][confusius._utils.templates.FUSI_TEMPLATES], e.g.
        `"huang_2025"`.
    **kwargs
        Forwarded to the template's `confusius.datasets.fetch_template_*` function
        (`data_dir`, `refresh`, `print_citation`).

    Returns
    -------
    xarray.DataArray
        The template VoxelData array.

    Raises
    ------
    KeyError
        If `name` is not a known template.
    """
    from confusius import datasets

    if name not in FUSI_TEMPLATES:
        raise KeyError(
            f"Unknown fUSI template {name!r}; expected one of {sorted(FUSI_TEMPLATES)}."
        )
    fetcher = getattr(datasets, FUSI_TEMPLATES[name]["fetcher"])
    return fetcher(**kwargs)
