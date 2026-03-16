"""Seed-based functional connectivity maps for fUSI data."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from sklearn.base import BaseEstimator

from confusius.extract.labels import extract_with_labels
from confusius.signal import clean
from confusius.validation import validate_labels, validate_time_series


def _validate_seed_signals(seed_signals: xr.DataArray, data: xr.DataArray) -> None:
    """Validate pre-computed seed signals against a reference data array.

    Checks performed (in order):

    1. `seed_signals` has a `time` dimension and more than 1 timepoint (via
       [`validate_time_series`][confusius.validation.validate_time_series]).
    2. `seed_signals` has no unexpected dimensions — only `time` and an optional
       `region` dimension are allowed.
    3. The number of timepoints in `seed_signals` matches `data`.
    4. When both arrays carry a `time` coordinate, those coordinates are equal
       element-wise.

    Parameters
    ----------
    seed_signals : xarray.DataArray
        Pre-computed seed region signal(s). Must be 1-D ``(time,)`` or 2-D
        ``(time, region)``.
    data : xarray.DataArray
        Reference fUSI data array that `seed_signals` will be correlated against.

    Raises
    ------
    ValueError
        If any of the above checks fail.
    """
    validate_time_series(seed_signals, operation_name="SeedBasedMaps.fit")

    unexpected_dims = [str(d) for d in seed_signals.dims if d not in ("time", "region")]
    if unexpected_dims:
        raise ValueError(
            f"seed_signals has unexpected dimensions {unexpected_dims}. "
            f"Only 'time' and an optional 'region' dimension are allowed."
        )

    if seed_signals.sizes["time"] != data.sizes["time"]:
        raise ValueError(
            f"seed_signals has {seed_signals.sizes['time']} timepoints but data has "
            f"{data.sizes['time']}. They must have the same number of timepoints."
        )

    if "time" in seed_signals.coords and "time" in data.coords:
        if not np.array_equal(
            seed_signals.coords["time"].values, data.coords["time"].values
        ):
            raise ValueError(
                "seed_signals time coordinates do not match data time coordinates."
            )


class SeedBasedMaps(BaseEstimator):
    """Seed-based functional connectivity maps from fUSI data.

    Computes voxel-wise Pearson correlation maps between one or more seed region signals
    and every voxel in a fUSI DataArray.

    Two ways to supply the seed signal are supported:

    - **Mask-based** (`seed_masks`): integer label maps are passed and the seed signals
      are extracted from the (optionally cleaned) data via
      [`extract_with_labels`][confusius.extract.extract_with_labels]. Signal cleaning
      via [`clean`][confusius.signal.clean] is applied to the full data array *before*
      seed extraction so that both the seed signal and the per-voxel signals are
      preprocessed consistently.
    - **Signal-based** (`seed_signals`): pre-computed `(time, region)` seed signals are
      provided directly. In this case extraction is skipped entirely and the supplied
      signals are correlated against the (optionally cleaned) data.  This is useful when
      the seed signal has been computed externally or originates from a different
      modality.

    Exactly one of `seed_masks` or `seed_signals` must be provided.

    Parameters
    ----------
    seed_masks : xarray.DataArray, optional
        Integer label maps defining the seed region(s). Two formats are accepted (same
        as [`extract_with_labels`][confusius.extract.extract_with_labels]):

        - **Flat label map**: spatial dims only, e.g. `(z, y, x)`.
          Background voxels are `0`; each unique non-zero integer is a
          separate seed region.
        - **Stacked mask format**: leading `mask` dim followed by spatial
          dims, e.g. `(mask, z, y, x)`.  Each layer has values in `{0,
          region_id}` and regions may overlap.

        A boolean mask can be used by converting it first: `mask.astype(int)`. Mutually
        exclusive with `seed_signals`.
    seed_signals : xarray.DataArray, optional
        Pre-computed seed signals with a `time` dimension and an optional `region`
        dimension.  When provided, seed extraction from the data is skipped and these
        signals are used directly to compute Pearson correlations. `clean_kwargs` is
        still applied to the *data* array before computing correlations, but the seed
        signals themselves are used as-is. Mutually exclusive with `seed_masks`.
    labels_reduction : {"mean", "sum", "median", "min", "max", "var", "std"}, \
            default: "mean"
        Aggregation function applied across voxels within each seed region when
        extracting seed signals from `seed_masks`.  Ignored when `seed_signals` is
        provided.
    clean_kwargs : dict, optional
        Keyword arguments forwarded to [`clean`][confusius.signal.clean]. Cleaning is
        applied to the full data array before computing correlations. If not provided,
        no cleaning is applied.

        !!! warning "Chunking along time"
            Any operation in `clean_kwargs` that involves detrending or
            filtering requires the `time` dimension to be un-chunked.
            Rechunk your data before calling `fit`: `data.chunk({'time': -1})`.

    Attributes
    ----------
    seed_signals_ : (time, region) xarray.DataArray
        Extracted (and cleaned) seed region signals when `seed_masks` is used, or the
        supplied signals (possibly transposed to `(time, region)` order) when
        `seed_signals` is used. Set after
        [`fit`][confusius.connectivity.SeedBasedMaps.fit].
    maps_ : (region, ...) xarray.DataArray
        Voxel-wise Pearson r maps, one per seed region, set after
        [`fit`][confusius.connectivity.SeedBasedMaps.fit]. `region` is the leading
        dimension; the remaining dimensions match the spatial dimensions of the data
        passed to [`fit`][confusius.connectivity.SeedBasedMaps.fit]. If a single region
        is present the `region` dimension is squeezed out. `attrs["cmap"]` is set to
        `"coolwarm"`, `attrs["norm"]` to `Normalize(vmin=-1, vmax=1)`, and
        `attrs["long_name"]` to `"Pearson r"` so that plotting functions pick up
        sensible defaults automatically.

    Examples
    --------
    Mask-based usage: two seed regions from a flat integer label map.

    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.connectivity import SeedBasedMaps
    >>>
    >>> rng = np.random.default_rng(0)
    >>> data = xr.DataArray(
    ...     rng.standard_normal((200, 10, 20)),
    ...     dims=["time", "y", "x"],
    ...     coords={"time": np.arange(200) * 0.1},
    ... )
    >>>
    >>> labels = xr.DataArray(
    ...     np.zeros((10, 20), dtype=int),
    ...     dims=["y", "x"],
    ... )
    >>> labels[:3, :] = 1   # Region 1: first 3 y-slices.
    >>> labels[3:6, :] = 2  # Region 2: next 3 y-slices.
    >>>
    >>> mapper = SeedBasedMaps(seed_masks=labels)
    >>> mapper.fit(data)
    SeedBasedMaps(seed_masks=...)
    >>> mapper.maps_.dims
    ('region', 'y', 'x')
    >>> mapper.maps_.coords["region"].values
    array([1, 2])
    >>>
    >>> # Single seed from a boolean mask converted to integer.
    >>> mask = xr.DataArray(
    ...     np.zeros((10, 20), dtype=bool),
    ...     dims=["y", "x"],
    ... )
    >>> mask[:3, :] = True
    >>> mapper_single = SeedBasedMaps(seed_masks=mask.astype(int))
    >>> mapper_single.fit(data)
    SeedBasedMaps(seed_masks=...)
    >>> mapper_single.maps_.dims  # region dim is squeezed for a single seed
    ('y', 'x')

    Signal-based usage: provide seed signals directly.

    >>> seed_signal = xr.DataArray(
    ...     rng.standard_normal(200),
    ...     dims=["time"],
    ...     coords={"time": np.arange(200) * 0.1},
    ... )
    >>> mapper_sig = SeedBasedMaps(seed_signals=seed_signal)
    >>> mapper_sig.fit(data)
    SeedBasedMaps(seed_signals=...)
    >>> mapper_sig.maps_.dims  # single signal, region dim squeezed
    ('y', 'x')
    """

    def __init__(
        self,
        *,
        seed_masks: xr.DataArray | None = None,
        seed_signals: xr.DataArray | None = None,
        labels_reduction: Literal[
            "mean", "sum", "median", "min", "max", "var", "std"
        ] = "mean",
        clean_kwargs: dict | None = None,
    ) -> None:
        self.seed_masks = seed_masks
        self.seed_signals = seed_signals
        self.labels_reduction = labels_reduction
        self.clean_kwargs = clean_kwargs

    def fit(self, X: xr.DataArray, y: None = None) -> "SeedBasedMaps":
        """Compute the seed-based correlation maps.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            A fUSI DataArray to estimate seed-based maps from.  Must have a `time`
            dimension.  The spatial dimensions must be compatible with `seed_masks` when
            using mask-based seeding.

            !!! warning "Chunking along time"
                The `time` dimension must NOT be chunked when `clean_kwargs` includes
                detrending or filtering steps. Rechunk first: `X.chunk({'time': -1})`.

        y : None, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        SeedBasedMaps
            Fitted estimator.

        Raises
        ------
        ValueError
            If neither or both of `seed_masks` and `seed_signals` are provided, if `X`
            has no `time` dimension, fewer than 2 timepoints, or if the `time` dimension
            is chunked when required. When `seed_signals` is provided: also raised if it
            has unexpected dimensions, a `time` size that differs from `X`, or `time`
            coordinates that do not match `X`.
        TypeError
            If `seed_masks` is not an integer-dtype DataArray.
        """
        if self.seed_masks is None and self.seed_signals is None:
            raise ValueError(
                "Exactly one of 'seed_masks' or 'seed_signals' must be provided, "
                "but neither was given."
            )
        if self.seed_masks is not None and self.seed_signals is not None:
            raise ValueError(
                "Exactly one of 'seed_masks' or 'seed_signals' must be provided, "
                "but both were given."
            )

        # Validate time dimension *before* cleaning so the error message points to the
        # right cause. check_time_chunks=False here because we re-validate inside
        # signal.clean if needed.
        validate_time_series(
            X, operation_name="SeedBasedMaps.fit", check_time_chunks=False
        )

        if self.seed_masks is not None:
            validate_labels(self.seed_masks, X, "seed_masks")
        else:
            # self.seed_signals is not None, guaranteed by the mutual-exclusivity check
            # above.
            assert self.seed_signals is not None
            _validate_seed_signals(self.seed_signals, X)

        if self.clean_kwargs is not None:
            X_clean = clean(X, **self.clean_kwargs)
        else:
            X_clean = X

        if self.seed_masks is not None:
            reduction = cast(
                Literal["mean", "sum", "median", "min", "max", "var", "std"],
                self.labels_reduction,
            )
            extracted: xr.DataArray = extract_with_labels(
                X_clean,
                self.seed_masks,
                reduction=reduction,
            )
        else:
            # self.seed_signals is not None, guaranteed by the mutual-exclusivity check
            # above.
            assert self.seed_signals is not None
            extracted = self.seed_signals

        if extracted.dims[0] != "time":
            extracted = extracted.transpose("time", ...)

        # _compute_correlation_maps must always receive a 2-D input.
        if extracted.ndim == 1:
            extracted = extracted.expand_dims("region", axis=1)

        self.seed_signals_: xr.DataArray = extracted

        maps = _compute_correlation_maps(X_clean, extracted)

        if maps.sizes.get("region", 0) == 1:
            maps = maps.isel(region=0, drop=False)

        # Annotate with display defaults so plotting functions pick them up without
        # requiring the caller to pass cmap/vmin/vmax explicitly.
        maps.attrs.update(
            {
                "long_name": "Pearson r",
                "cmap": "coolwarm",
                "norm": Normalize(vmin=-1.0, vmax=1.0),
            }
        )

        self.maps_: xr.DataArray = maps
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Check whether the estimator has been fitted."""
        return hasattr(self, "maps_")


def _compute_correlation_maps(
    data: xr.DataArray,
    seed_signals: xr.DataArray,
) -> xr.DataArray:
    """Compute one Pearson r map per seed region.

    Uses native xarray/Dask operations (mean, dot, sum) over the `time`
    dimension only; no stacking of spatial dims required.

    Parameters
    ----------
    data : (time, ...) xarray.DataArray
        Clean fUSI data.
    seed_signals : (time, region) xarray.DataArray
        Extracted seed signals.

    Returns
    -------
    (region, ...) xarray.DataArray
        Pearson r maps with `region` as the leading dimension.
    """
    spatial_dims = [str(d) for d in data.dims if d != "time"]

    data_c = data - data.mean("time")
    seeds_c = seed_signals - seed_signals.mean("time")

    # Dot product over time gives (*spatial_dims, region).
    numerator = xr.dot(data_c, seeds_c, dim="time")

    data_norm = np.sqrt((data_c**2).sum("time"))  # (*spatial_dims,)
    seeds_norm = np.sqrt((seeds_c**2).sum("time"))  # (region,)

    r = xr.where(
        data_norm * seeds_norm == 0,
        0.0,
        numerator / (data_norm * seeds_norm),
    )

    return r.transpose("region", *spatial_dims)
