"""Griffe extension that keeps `xarray.*` docstring types pointing at the xarray package.

Griffe resolves names in docstring type lines (``data : xarray.DataArray``) by
walking up the scope chain. For any module under `confusius`, that walk reaches the
`confusius` package, whose `xarray` member is the `confusius.xarray` subpackage, so
`xarray.DataArray` resolves to the non-existent `confusius.xarray.DataArray`.
Binding the name `xarray` to the real package in every submodule stops the walk
before it reaches `confusius`. The alias is an "imported" (non-exported) member, so
mkdocstrings never renders it.
"""

import griffe


class ExternalXarray(griffe.Extension):
    """Bind `xarray` to the external package in every `confusius` submodule."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: object) -> None:
        """Add an `xarray` alias to each submodule lacking one.

        Parameters
        ----------
        pkg : griffe.Module
            The loaded top-level package.
        **kwargs : object
            Other hook arguments, unused.
        """
        if pkg.name != "confusius":
            return
        stack = list(pkg.modules.values())
        while stack:
            mod = stack.pop()
            stack.extend(mod.modules.values())
            if "xarray" not in mod.members:
                mod.set_member("xarray", griffe.Alias("xarray", "xarray", parent=mod))
