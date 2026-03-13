"""Reconstruction of fUSI DataArrays from N-D signals using masks."""

import numpy as np
import xarray as xr


def unmask(
    signals: np.ndarray | xr.DataArray,
    mask: xr.DataArray,
    new_dims: list[str] | None = None,
    new_dims_coords: dict[str, np.ndarray] | None = None,
    attrs: dict | None = None,
    fill_value: float = 0.0,
) -> xr.DataArray:
    """Reconstruct a fUSI DataArray from N-D signals using a mask.

    Parameters
    ----------
    signals : numpy.ndarray or xarray.DataArray
        Array with shape `(..., space)` where `...` can be any number of
        dimensions. The last dimension must correspond to masked voxels.

        - If `signals` is a DataArray, it must have a `space` dimension as the
          last dimension. All other dimensions and their coordinates are preserved.
        - If `signals` is a Numpy array, you can specify names and coordinates for
          the leading dimensions using `new_dims` and `new_dims_coords`. If not
          provided, dimensions are named `["dim_0", "dim_1", ...]` with integer
          coordinates.

    mask : xarray.DataArray
        Boolean mask used for the original extraction. Provides spatial dimensions
        and coordinates for reconstruction. Must have the same spatial dimensions
        and coordinates as the original data.
    new_dims : list of str, optional
        Names for leading dimensions when `signals` is a Numpy array. Must match the
        number of leading dimensions `(ndim - 1)`. If not provided, uses `["dim_0",
        "dim_1", ...]`. Ignored if `signals` is a DataArray.
    new_dims_coords : dict[str, numpy.ndarray], optional
        Coordinates for leading dimensions when `signals` is a Numpy array. Keys must
        match dimension names in `new_dims`. If not provided, uses integer indices for
        all dimensions. Ignored if `signals` is a DataArray.
    attrs : dict, optional
        Attributes to attach to the output DataArray.
    fill_value : float, default: 0.0
        Value to fill in non-masked voxels.

    Returns
    -------
    xarray.DataArray
        Reconstructed DataArray with shape `(..., z, y, x)` where spatial
        coordinates come from the mask.

    Raises
    ------
    ValueError
        If `signals` shape doesn't match `mask`, or if `new_dims`/`new_dims_coords`
        are inconsistent with `signals` shape.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius.extract import extract_with_mask, unmask
    >>> from sklearn.decomposition import PCA
    >>>
    >>> # Load data and mask
    >>> data = xr.open_zarr("recording.zarr")["power_doppler"]
    >>> mask = xr.open_zarr("brain_mask.zarr")["mask"]
    >>>
    >>> # Extract signals
    >>> signals = extract_with_mask(data, mask)
    >>>
    >>> # Apply PCA
    >>> pca = PCA(n_components=5)
    >>> components = pca.fit_transform(signals.values)  # (time, 5)
    >>>
    >>> # Unmask - 2D case
    >>> spatial_pca = unmask(
    ...     components.T,  # (5, n_voxels)
    ...     mask,
    ...     new_dims=["component"],
    ... )
    >>> spatial_pca.dims
    ("component", "z", "y", "x")
    >>>
    >>> # Unmask - 3D case with custom coords
    >>> pose_data = np.random.randn(5, 3, n_voxels)  # (component, pose, space)
    >>> spatial_pose = unmask(
    ...     pose_data,
    ...     mask,
    ...     new_dims=["component", "pose"],
    ...     new_dims_coords={"component": [1, 2, 3, 4, 5], "pose": [0, 1, 2]},
    ... )
    >>> spatial_pose.dims
    ("component", "pose", "z", "y", "x")
    """
    if isinstance(signals, np.ndarray):
        n_voxels_mask = int(mask.sum().values)

        if signals.shape[-1] != n_voxels_mask:
            raise ValueError(
                f"Last dimension of signals ({signals.shape[-1]}) doesn't match "
                f"number of masked voxels ({n_voxels_mask})"
            )

        n_leading_dims = signals.ndim - 1

        if new_dims is not None:
            if len(new_dims) != n_leading_dims:
                raise ValueError(
                    f"Length of new_dims ({len(new_dims)}) doesn't match "
                    f"number of leading dimensions ({n_leading_dims})"
                )
            dim_names = new_dims
        else:
            dim_names = [f"dim_{i}" for i in range(n_leading_dims)]

        coords = {"space": np.arange(n_voxels_mask)}
        if new_dims_coords is not None:
            for dim in dim_names:
                if dim in new_dims_coords:
                    coord_array = new_dims_coords[dim]
                    dim_idx = dim_names.index(dim)
                    if len(coord_array) != signals.shape[dim_idx]:
                        raise ValueError(
                            f"Coordinate array for '{dim}' has length {len(coord_array)} "
                            f"but dimension has size {signals.shape[dim_idx]}"
                        )
                    coords[dim] = coord_array
                else:
                    dim_idx = dim_names.index(dim)
                    coords[dim] = np.arange(signals.shape[dim_idx])
        else:
            for i, dim in enumerate(dim_names):
                coords[dim] = np.arange(signals.shape[i])

        all_dims = dim_names + ["space"]
        signals = xr.DataArray(signals, dims=all_dims, coords=coords)
    elif isinstance(signals, xr.DataArray):
        if "space" not in signals.dims:
            raise ValueError(
                f"DataArray signals must have 'space' dimension, got {signals.dims}"
            )
        if signals.dims[-1] != "space":
            raise ValueError(
                f"'space' must be the last dimension, got dims={signals.dims}"
            )
    else:
        raise TypeError(
            f"'signals' must be Numpy array or DataArray, got {type(signals)}"
        )

    spatial_dims = list(mask.dims)
    spatial_shape = tuple(mask.sizes[d] for d in spatial_dims)
    extra_dims = [d for d in signals.dims if d != "space"]

    if extra_dims:
        output_shape = tuple(signals.sizes[d] for d in extra_dims) + spatial_shape
        output_dims = extra_dims + spatial_dims

        output_data = np.full(output_shape, fill_value, dtype=signals.dtype)

        mask_flat = mask.values.flatten()
        n_extra = int(np.prod([signals.sizes[d] for d in extra_dims]))
        output_flat = output_data.reshape(n_extra, -1)
        signals_flat = signals.values.reshape(n_extra, -1)
        output_flat[:, mask_flat] = signals_flat

        coords = {d: mask.coords[d] for d in spatial_dims}
        for d in extra_dims:
            coords[d] = signals.coords[d]
    else:
        output_shape = spatial_shape
        output_dims = spatial_dims

        output_data = np.full(output_shape, fill_value, dtype=signals.dtype)

        mask_flat = mask.values.flatten()
        output_data.flat[mask_flat] = signals.values

        coords = {d: mask.coords[d] for d in spatial_dims}

    return xr.DataArray(
        output_data,
        dims=output_dims,
        coords=coords,
        attrs=attrs if attrs is not None else {},
    )
