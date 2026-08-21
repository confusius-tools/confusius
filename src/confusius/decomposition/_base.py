"""Shared base class for xarray-aware scikit-learn decomposers."""

from abc import abstractmethod
from collections.abc import Hashable
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from confusius.extract import extract_with_mask, unmask
from confusius.validation import ensure_mask, ensure_voxeldata, validate_time_series


class _BaseFUSIDecomposer(BaseEstimator, TransformerMixin):
    """Base class for xarray-aware fUSI decomposers.

    Subclasses must:

    - Define `_signals_long_name` as a class attribute.
    - Assign `self._estimator` to the fitted sklearn estimator in `fit`.
    - Assign `self.maps_` and `self.n_components_` in `fit`.

    All shared xarray bookkeeping (data preparation, spatial reshaping, `fit_transform`,
    `transform`, and `inverse_transform`) is handled here.

    `X` must be a VoxelData array (native voxel dims `k`/`j`/`i` and a
    `VoxelToWorldIndex`). `mask` selects voxels, and reconstruction
    (`inverse_transform`, `maps_`) restores the full `(k, j, i)` grid. Decomposing an
    already-reduced signals table (e.g. the `(time, region)` output of
    [`extract_with_labels`][confusius.extract.extract_with_labels]) is regular
    tabular PCA/ICA/NMF with no spatial structure to track, so it offers nothing
    over calling scikit-learn directly on the array's values.
    """

    _signals_long_name: str
    mask: xr.DataArray | None

    # Fitted attributes, set by subclasses in fit.
    _estimator: Any
    maps_: xr.DataArray
    n_components_: int
    mode: str
    _spatial_components_flat_: npt.NDArray[np.floating]
    _spatial_feature_mean_: npt.NDArray[np.floating]

    # Set by _store_fit_metadata.
    spatial_dims_: tuple[str, ...]
    _spatial_sizes_: dict[str, int]
    _reconstruction_mask_: xr.DataArray
    _fit_attrs_: dict[str, Any]
    _fit_name_: Hashable | None
    n_features_in_: int

    @abstractmethod
    def fit(self, X: xr.DataArray, y: None = None) -> "_BaseFUSIDecomposer":
        """Fit the decomposer on `(time, ...)` fUSI data.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            VoxelData array.
        y : None, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        _BaseFUSIDecomposer
            Fitted estimator.
        """
        ...

    def fit_transform(
        self, X: xr.DataArray, y: None = None, **fit_params: object
    ) -> xr.DataArray:
        """Fit on `X` and return transformed signals.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            VoxelData array.
        y : None, optional
            Ignored. Present for scikit-learn API compatibility.
        **fit_params : object
            Additional fit parameters. Unsupported for this estimator.

        Returns
        -------
        (time, component) xarray.DataArray
            Decomposed signals in component space.

        Raises
        ------
        TypeError
            If additional fit parameters are provided.
        """
        del y

        if fit_params:
            keys = ", ".join(sorted(fit_params))
            raise TypeError(
                f"Unexpected fit parameters for "
                f"{type(self).__name__}.fit_transform: {keys}."
            )

        return self.fit(X).transform(X)

    def transform(self, X: xr.DataArray) -> xr.DataArray:
        """Project data into component space.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            VoxelData array with the same spatial dimensions and sizes as the data used
            during fit.

        Returns
        -------
        (time, component) xarray.DataArray
            Decomposed signals in component space.
        """
        check_is_fitted(self)
        X_proc, _, _ = self._prepare_data(
            X,
            check_layout=True,
            operation_name=f"{type(self).__name__}.transform",
        )
        if self._uses_spatial_projection():
            signals = self._spatial_transform(X_proc)
        else:
            signals = self._estimator.transform(X_proc)

        transformed = xr.DataArray(
            signals,
            dims=["time", "component"],
            coords={
                "time": X.coords["time"],
                "component": self.maps_.coords["component"],
            },
        )
        transformed.attrs.update({"long_name": self._signals_long_name})
        return transformed

    def inverse_transform(
        self, X: xr.DataArray | npt.NDArray[np.floating]
    ) -> xr.DataArray:
        """Reconstruct data from component signals.

        Parameters
        ----------
        X : (time, component) xarray.DataArray or (time, component) numpy.ndarray
            Signals in component space.

        Returns
        -------
        (time, ...) xarray.DataArray
            Reconstructed data in the fitted spatial geometry.

        Raises
        ------
        ValueError
            If `X` has invalid shape or component count.
        TypeError
            If `X` is neither `xarray.DataArray` nor `numpy.ndarray`.
        """
        check_is_fitted(self)

        if isinstance(X, xr.DataArray):
            if set(X.dims) != {"time", "component"}:
                raise ValueError(
                    "X must have exactly the dimensions {'time', 'component'} for "
                    "inverse_transform."
                )
            X_ordered = X.transpose("time", "component")
            signals = np.asarray(X_ordered.values)
            time_coord: npt.NDArray[np.generic] | xr.DataArray = (
                X_ordered.coords["time"]
                if "time" in X_ordered.coords
                else np.arange(X_ordered.sizes["time"], dtype=np.intp)
            )
        elif isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    f"X must be 2D with shape (time, component), got {X.shape}."
                )
            signals = np.asarray(X)
            time_coord = np.arange(signals.shape[0], dtype=np.intp)
        else:
            raise TypeError(f"X must be DataArray or ndarray, got {type(X)}")

        if signals.shape[1] != self.n_components_:
            raise ValueError(
                f"X has {signals.shape[1]} components, but "
                f"{type(self).__name__} was fitted with {self.n_components_}."
            )

        if self._uses_spatial_projection():
            reconstructed = self._spatial_inverse_transform(signals)
        else:
            reconstructed = self._estimator.inverse_transform(signals)

        reconstructed_da = unmask(
            reconstructed,
            self._reconstruction_mask_,
            new_dims=["time"],
            new_dims_coords={"time": np.asarray(time_coord)},
            attrs=self._fit_attrs_,
        )
        reconstructed_da.name = self._fit_name_
        return reconstructed_da

    def _prepare_data(
        self,
        X: xr.DataArray,
        check_layout: bool,
        operation_name: str,
    ) -> tuple[npt.NDArray[np.floating], tuple[str, ...], xr.DataArray]:
        """Validate and stack time series data into a 2D feature matrix.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            VoxelData array.
        check_layout : bool
            Whether to check that the spatial dimensions and sizes match the fitted
            estimator state.
        operation_name : str
            Name of the calling operation, used in validation error messages.

        Returns
        -------
        X_proc : (time, feature) numpy.ndarray
            Input data stacked over selected spatial features. Dtype is preserved from
            `X`; sklearn handles any required type promotion internally.
        spatial_dims : tuple[str, ...]
            Spatial dimensions used to order and stack the input data.
        mask : xarray.DataArray
            Boolean VoxelData mask selecting features used for fitting/projection,
            with `X`'s spatial dims, coords, and `VoxelToWorldIndex`.

        Raises
        ------
        ValueError
            If `X` is not a valid VoxelData time series, or has a spatial layout
            inconsistent with the fitted estimator when `check_layout` is `True`.
        """
        X = ensure_voxeldata(X, allow_extra_dims=True)
        validate_time_series(X, operation_name=operation_name)

        input_spatial_dims = tuple(str(dim) for dim in X.dims if dim != "time")

        if check_layout:
            spatial_dims = self.spatial_dims_
            if set(input_spatial_dims) != set(spatial_dims):
                raise ValueError(
                    "X spatial dimensions do not match fitted dimensions. "
                    f"Expected {spatial_dims}, got {input_spatial_dims}."
                )
            for dim in spatial_dims:
                if X.sizes[dim] != self._spatial_sizes_[dim]:
                    raise ValueError(
                        f"Spatial dimension '{dim}' has size {X.sizes[dim]} in X, "
                        f"but expected {self._spatial_sizes_[dim]} from fit."
                    )
        else:
            spatial_dims = input_spatial_dims

        X_ordered = X.transpose("time", *spatial_dims)
        mask = self.mask
        if mask is None:
            mask_template = X_ordered.isel(time=0, drop=True)
            mask = ensure_voxeldata(
                mask_template.copy(data=np.ones(mask_template.shape, dtype=bool)),
                allow_extra_dims=True,
            )
        else:
            mask = ensure_mask(mask, X_ordered, "mask", require_exact_dims=True)

        X_proc = extract_with_mask(X_ordered, mask).values
        return X_proc, spatial_dims, mask

    def _reshape_component_matrix(
        self,
        matrix: npt.NDArray[np.floating],
        component_coord: npt.NDArray[np.intp],
        *,
        long_name: str,
    ) -> xr.DataArray:
        """Reshape a `(component, feature)` matrix to the fitted spatial geometry.

        Parameters
        ----------
        matrix : (n_components, n_features) numpy.ndarray
            Flat matrix to reshape.
        component_coord : (n_components,) numpy.ndarray
            Coordinate values for the component dimension.
        long_name : str
            Value stored in `attrs["long_name"]` on the output DataArray.

        Returns
        -------
        (n_components, ...) xarray.DataArray
            Matrix unstacked to the original spatial dimensions, with
            `attrs["long_name"]` and `attrs["cmap"]` set.
        """
        reshaped = unmask(
            matrix,
            self._reconstruction_mask_,
            new_dims=["component"],
            new_dims_coords={"component": component_coord},
            attrs={"long_name": long_name, "cmap": "coolwarm"},
            fill_value=0.0,
        )
        return reshaped

    def _reshape_mean(self, mean: npt.NDArray[np.floating]) -> xr.DataArray:
        """Reshape a per-feature mean vector to the fitted spatial geometry.

        Parameters
        ----------
        mean : (n_features,) numpy.ndarray
            Per-feature mean values.

        Returns
        -------
        (...) xarray.DataArray
            Mean unstacked to the original spatial dimensions.
        """
        return unmask(mean, self._reconstruction_mask_, attrs={}, fill_value=0.0)

    def _store_fit_metadata(
        self,
        X: xr.DataArray,
        X_proc: npt.NDArray[np.floating],
        spatial_dims: tuple[str, ...],
        mask: xr.DataArray,
    ) -> None:
        """Store spatial and array metadata from a fit call.

        Sets `spatial_dims_`, `_spatial_sizes_`, `_reconstruction_mask_`, `_fit_attrs_`,
        `_fit_name_`, and `n_features_in_` on the estimator.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Original input passed to `fit`.
        X_proc : (time, feature) numpy.ndarray
            Stacked data over selected features returned by `_prepare_data`.
        spatial_dims : tuple[str, ...]
            Spatial dimensions returned by `_prepare_data`.
        mask : xarray.DataArray
            Mask returned by `_prepare_data`, already carrying `X`'s spatial dims,
            coords, and `VoxelToWorldIndex`.
        """
        self.spatial_dims_ = spatial_dims
        self._spatial_sizes_ = {
            dim: np.int64(X.sizes[dim]).item() for dim in self.spatial_dims_
        }
        self._reconstruction_mask_ = mask
        self._fit_attrs_ = dict(X.attrs)
        self._fit_name_ = X.name
        self.n_features_in_ = np.int64(X_proc.shape[1]).item()

    def _uses_spatial_projection(self) -> bool:
        """Whether transform/inverse use shared spatial projection logic."""
        return (
            getattr(self, "mode", None) == "spatial"
            and hasattr(self, "_spatial_components_flat_")
            and hasattr(self, "_spatial_feature_mean_")
        )

    def _spatial_transform(
        self, X_proc: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Project `(time, feature)` data onto spatial components."""
        return (X_proc - self._spatial_feature_mean_) @ self._spatial_components_flat_.T

    def _spatial_inverse_transform(
        self, signals: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Reconstruct `(time, feature)` data from spatial component signals."""
        return signals @ self._spatial_components_flat_ + self._spatial_feature_mean_

    def __sklearn_is_fitted__(self) -> bool:
        """Check whether the estimator has been fitted.

        Returns
        -------
        bool
            `True` if the estimator has been fitted, `False` otherwise.
        """
        return hasattr(self, "maps_") and hasattr(self, "n_components_")
