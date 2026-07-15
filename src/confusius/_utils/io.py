"""IO-related cross-module helpers."""

import dask.array as da
import h5py
import xarray as xr


def is_h5py_backed(data: xr.DataArray) -> bool:
    """Return True if `data` is backed by an h5py dataset.

    h5py datasets cannot be pickled, so DataArrays backed by them cannot be
    serialized for parallel processing with joblib.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    bool
        True if `data.data` is a `dask.array.Array` whose graph contains at least
        one `h5py.Dataset`; False otherwise (including for in-memory arrays).
    """
    if not isinstance(data.data, da.Array):
        return False
    graph = data.data.__dask_graph__()
    for layer in graph.layers.values():
        for v in layer.values():
            if isinstance(v, h5py.Dataset):
                return True
    return False
