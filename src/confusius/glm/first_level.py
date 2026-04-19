"""First-level GLM for single-subject fUSI analysis.

This module implements a sklearn-style estimator for fitting voxel-wise
General Linear Models to fUSI time-series data stored as xarray DataArrays.

Portions of this file are derived from Nilearn, which is licensed under the
BSD-3-Clause License. See `NOTICE` file for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator

from confusius.glm._contrasts import Contrast
from confusius.glm._design import (
    make_first_level_design_matrix,
)
from confusius.glm._models import ARModel, OLSModel, RegressionResults
from confusius.glm._utils import estimate_ar_coeffs, expression_to_contrast_vector
from confusius.validation.time_series import validate_time_series

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


def _flatten_spatial(
    data: xr.DataArray,
) -> tuple[npt.NDArray[np.float64], tuple[str, ...], tuple[int, ...]]:
    """Flatten spatial dimensions of a DataArray for regression.

    Parameters
    ----------
    data : (time, ...) xarray.DataArray
        Input fUSI data.

    Returns
    -------
    flat : (n_timepoints, n_voxels) numpy.ndarray
        Flattened 2D array.
    spatial_dims : tuple of str
        Names of the spatial dimensions.
    spatial_shape : tuple of int
        Original shape of the spatial dimensions.

    """
    spatial_dims = tuple(str(d) for d in data.dims if d != "time")
    spatial_shape = tuple(data.sizes[d] for d in spatial_dims)
    values = data.stack(voxels=spatial_dims).transpose("time", "voxels").values
    return values, spatial_dims, spatial_shape


class FirstLevelModel(BaseEstimator):
    """First-level GLM estimator for voxel-wise fUSI analysis.

    Fits a General Linear Model to `(time, ...)` DataArrays and computes
    statistical contrasts. Supports multiple runs (fixed-effects combination),
    autoregressive noise modelling, and automatic sampling-interval inference
    from the `time` coordinate.

    This implementation is adapted from
    [`nilearn.glm.first_level.FirstLevelModel`][nilearn.glm.first_level.FirstLevelModel].

    Parameters
    ----------
    hrf_model : {"glover", "spm", "fir"} or None, default: "glover"
        Hemodynamic response function model.
    drift_model : {"cosine", "polynomial"} or None, default: "cosine"
        Drift model for low-frequency confounds.
    high_pass : float, default: 0.01
        High-pass filter cutoff in Hz (used with `drift_model="cosine"`).
    drift_order : int, default: 1
        Polynomial order when `drift_model="polynomial"`.
    fir_delays : list of int, optional
        FIR delays in volumes (required when `hrf_model="fir"`).
    noise_model : {"ols"} or "arN", default: "ar1"
        Noise model. `"arN"` estimates AR(N) coefficients from OLS residuals and refits
        with a whitened model.
    minimize_memory : bool, default: True
        Whether to keep only the statistics needed for contrast computation (per-run
        `RegressionResults` are discarded after extracting the contrast-relevant
        fields).
    oversampling : int, default: 50
        Oversampling factor for HRF convolution.
    min_onset : float, default: -24.0
        Minimum onset time in seconds for event regressors.

    Attributes
    ----------
    design_matrices_ : list of pandas.DataFrame
        Design matrix for each run (set after
        [`fit`][confusius.glm.first_level.FirstLevelModel.fit]).
    results_ : list of RegressionResults
        Per-run regression results (set after
        [`fit`][confusius.glm.first_level.FirstLevelModel.fit]).

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> data = xr.DataArray(
    ...     np.random.randn(200, 2, 3, 4),
    ...     dims=["time", "z", "y", "x"],
    ...     coords={"time": np.arange(200) * 0.5},
    ... )
    >>> events = pd.DataFrame(
    ...     {"trial_type": ["A", "B"] * 5,
    ...      "onset": np.arange(10) * 10.0,
    ...      "duration": [2.0] * 10}
    ... )
    >>> model = FirstLevelModel(noise_model="ols")
    >>> model.fit(data, events=events)  # doctest: +SKIP
    >>> z_map = model.compute_contrast("A - B")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        hrf_model: str | None = "glover",
        drift_model: str | None = "cosine",
        high_pass: float = 0.01,
        drift_order: int = 1,
        fir_delays: list[int] | None = None,
        noise_model: str = "ar1",
        minimize_memory: bool = True,
        oversampling: int = 50,
        min_onset: float = -24.0,
    ) -> None:
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.high_pass = high_pass
        self.drift_order = drift_order
        self.fir_delays = fir_delays
        self.noise_model = noise_model
        self.minimize_memory = minimize_memory
        self.oversampling = oversampling
        self.min_onset = min_onset

    def fit(
        self,
        run_data: xr.DataArray | list[xr.DataArray],
        events: pd.DataFrame | list[pd.DataFrame] | None = None,
        confounds: (
            pd.DataFrame
            | npt.NDArray[np.floating]
            | list[pd.DataFrame | npt.NDArray[np.floating] | None]
            | None
        ) = None,
        design_matrices: pd.DataFrame | list[pd.DataFrame] | None = None,
    ) -> FirstLevelModel:
        """Fit the GLM to fUSI data.

        Either `events` or `design_matrices` must be provided. When `design_matrices` is
        given, `events`, `confounds`, and the design-related constructor parameters
        (`hrf_model`, etc.) are ignored.

        Parameters
        ----------
        run_data : xarray.DataArray or list of xarray.DataArray
            Single-run or multi-run fUSI data. Must have a `time` dimension; all other
            dimensions are treated as spatial (e.g. `(time, z, y, x)` or `(time, pose,
            z, y, x)`).
        events : pandas.DataFrame or list of pandas.DataFrame, optional
            Events table(s) with `onset`, `duration`, and `trial_type` columns. Onsets
            are in the same physical time units as the `time` coordinate of `run_data`.
        confounds : pandas.DataFrame, numpy.ndarray, or list, optional
            Confound regressors per run.
        design_matrices : pandas.DataFrame or list of pandas.DataFrame, optional
            Pre-built design matrices. Overrides `events` / `confounds`.

        Returns
        -------
        FirstLevelModel
            Fitted estimator (`self`).

        Raises
        ------
        ValueError
            If neither `events` nor `design_matrices` is provided, if list lengths
            are inconsistent across arguments, or if runs have different spatial
            shapes.
        """
        if isinstance(run_data, xr.DataArray):
            run_data = [run_data]
        n_runs = len(run_data)

        for i, run in enumerate(run_data):
            validate_time_series(run, f"FirstLevelModel fit (run {i})")

        if n_runs > 1:
            ref_spatial = {d: s for d, s in run_data[0].sizes.items() if d != "time"}
            for i, run in enumerate(run_data[1:], start=1):
                spatial = {d: s for d, s in run.sizes.items() if d != "time"}
                if spatial != ref_spatial:
                    raise ValueError(
                        f"All runs must have the same spatial shape. "
                        f"Run 0 has {ref_spatial}, run {i} has {spatial}."
                    )

        design_matrices_list = self._resolve_design_matrices(
            run_data, events, confounds, design_matrices
        )

        ar_order = self._parse_noise_model()

        _, spatial_dims, _ = _flatten_spatial(run_data[0])

        self.design_matrices_: list[pd.DataFrame] = design_matrices_list
        self.results_: list[RegressionResults] = []
        self._spatial_dims: tuple[str, ...] = spatial_dims
        self._spatial_shapes: list[tuple[int, ...]] = []
        self._run_coords: list[dict[str, xr.Variable]] = []

        for run_index in range(n_runs):
            data_2d, s_dims, s_shape = _flatten_spatial(run_data[run_index])
            self._spatial_shapes.append(s_shape)
            self._run_coords.append(
                {
                    str(d): run_data[run_index].coords[d]
                    for d in s_dims
                    if d in run_data[run_index].coords
                }
            )

            dm = design_matrices_list[run_index]
            design_array = dm.to_numpy()

            if ar_order == 0:
                # Pure OLS.
                model = OLSModel(design_array)
                results = model.fit(data_2d)
            else:
                # Two-pass: OLS → estimate AR → refit with AR whitening.
                ols_model = OLSModel(design_array)
                ols_results = ols_model.fit(data_2d)
                rho_per_voxel, _ = estimate_ar_coeffs(
                    ols_results.residuals, order=ar_order
                )
                ar_model = ARModel(design_array, rho_per_voxel)
                results = ar_model.fit(data_2d)

            self.results_.append(results)

        return self

    # ------------------------------------------------------------------
    # Contrasts
    # ------------------------------------------------------------------

    def compute_contrast(
        self,
        contrast_def: str | npt.NDArray[np.floating],
        stat_type: Literal["t", "F"] | None = None,
        output_type: Literal[
            "z_score", "stat", "p_value", "effect_size", "effect_variance"
        ] = "z_score",
    ) -> xr.DataArray:
        """Compute a contrast and return a statistical map.

        Parameters
        ----------
        contrast_def : str or numpy.ndarray
            Contrast definition. A string is parsed as an expression over
            design-matrix column names (e.g. `"A - B"`). A 1-D array
            specifies a t-contrast vector; a 2-D array specifies an
            F-contrast matrix.
        stat_type : {"t", "F"}, optional
            Force the contrast type. By default inferred from the shape of
            the contrast vector (1-D → t, 2-D → F).
        output_type : {"z_score", "stat", "p_value", "effect_size", "effect_variance"}, default: "z_score"
            Which statistical map to return.

        Returns
        -------
        xarray.DataArray
            Statistical map with the spatial dimensions of the input data.

        Raises
        ------
        ValueError
            If the model has not been fitted or the contrast definition is
            invalid.
        """
        self._check_is_fitted()

        contrast_obj = self._compute_contrast_across_runs(contrast_def, stat_type)

        output_map = self._contrast_output(contrast_obj, output_type)

        return self._to_dataarray(output_map, output_type)

    def _resolve_design_matrices(
        self,
        run_data: list[xr.DataArray],
        events: pd.DataFrame | list[pd.DataFrame] | None,
        confounds: (
            pd.DataFrame
            | npt.NDArray[np.floating]
            | list[pd.DataFrame | npt.NDArray[np.floating] | None]
            | None
        ),
        design_matrices: pd.DataFrame | list[pd.DataFrame] | None,
    ) -> list[pd.DataFrame]:
        """Build or validate design matrices for all runs.

        If `design_matrices` is provided, validates list length and returns it.
        Otherwise builds design matrices from `events` and `confounds` using
        [`make_first_level_design_matrix`][confusius.glm.make_first_level_design_matrix].

        Parameters
        ----------
        run_data : list of xarray.DataArray
            Per-run fUSI data, used to extract volume times.
        events : pandas.DataFrame or list of pandas.DataFrame, optional
            Per-run event tables.
        confounds : pandas.DataFrame, numpy.ndarray, or list, optional
            Per-run confound regressors.
        design_matrices : pandas.DataFrame or list of pandas.DataFrame, optional
            Pre-built design matrices. If provided, `events` and `confounds`
            are ignored.

        Returns
        -------
        list of pandas.DataFrame
            One design matrix per run.

        Raises
        ------
        ValueError
            If list lengths are inconsistent or neither `events` nor
            `design_matrices` is provided.
        """
        import pandas as pd

        n_runs = len(run_data)

        if design_matrices is not None:
            if isinstance(design_matrices, pd.DataFrame):
                design_matrices = [design_matrices]
            if len(design_matrices) != n_runs:
                raise ValueError(
                    f"Got {len(design_matrices)} design matrices for {n_runs} runs."
                )
            return list(design_matrices)

        if events is None:
            raise ValueError("Either 'events' or 'design_matrices' must be provided.")

        if isinstance(events, pd.DataFrame):
            events_list: list[pd.DataFrame] = [events] * n_runs
        else:
            events_list = list(events)

        if len(events_list) != n_runs:
            raise ValueError(
                f"Got {len(events_list)} events DataFrames for {n_runs} runs."
            )

        if confounds is None:
            confounds_list = [None] * n_runs
        elif not isinstance(confounds, list):
            confounds_list = [confounds] * n_runs
        else:
            confounds_list = list(confounds)

        if len(confounds_list) != n_runs:
            raise ValueError(
                f"Got {len(confounds_list)} confound entries for {n_runs} runs."
            )

        dm_list: list[pd.DataFrame] = []
        for run_idx in range(n_runs):
            volume_times = run_data[run_idx].coords["time"].values
            dm = make_first_level_design_matrix(
                volume_times,
                events=events_list[run_idx],
                hrf_model=self.hrf_model,
                drift_model=self.drift_model,
                high_pass=self.high_pass,
                drift_order=self.drift_order,
                fir_delays=self.fir_delays,
                confounds=confounds_list[run_idx],
                oversampling=self.oversampling,
                min_onset=self.min_onset,
            )
            dm_list.append(dm)
        return dm_list

    def _parse_noise_model(self) -> int:
        """Parse `self.noise_model` and return the AR order.

        Returns
        -------
        int
            0 for OLS, or the AR order for AR noise models.

        Raises
        ------
        ValueError
            If `noise_model` is not `"ols"` or a valid `"arN"` string (e.g. `"ar1"`,
            `"ar2"`).
        """
        nm = self.noise_model.lower()
        if nm == "ols":
            return 0
        if nm.startswith("ar"):
            try:
                order = int(nm[2:])
            except ValueError:
                raise ValueError(
                    f"Invalid noise_model '{self.noise_model}'. "
                    "Expected 'ols' or 'arN' (e.g. 'ar1', 'ar2')."
                ) from None
            if order < 1:
                raise ValueError("AR order must be >= 1.")
            return order
        raise ValueError(
            f"Invalid noise_model '{self.noise_model}'. "
            "Expected 'ols' or 'arN' (e.g. 'ar1', 'ar2')."
        )

    def _compute_contrast_across_runs(
        self,
        contrast_def: str | npt.NDArray[np.floating],
        stat_type: Literal["t", "F"] | None,
    ) -> Contrast:
        """Compute and combine contrasts across all runs using fixed effects.

        Fits a per-run contrast and accumulates effects and variances by summation
        (fixed-effects combination). Degrees of freedom are also summed across runs.

        Parameters
        ----------
        contrast_def : str or numpy.ndarray
            Contrast definition (expression string or numeric vector/matrix).
        stat_type : {"t", "F"} or None
            Statistic type. Inferred from contrast shape if `None` (1D → `t`, 2D → `F`).

        Returns
        -------
        Contrast
            Combined contrast across all runs.
        """
        combined: Contrast | None = None

        for run_idx, (dm, results) in enumerate(
            zip(self.design_matrices_, self.results_, strict=True)
        ):
            contrast_vec = self._resolve_contrast_vector(contrast_def, dm, run_idx)

            c = np.atleast_2d(contrast_vec)
            if stat_type is None:
                run_stat_type: Literal["t", "F"] = "F" if c.shape[0] > 1 else "t"
            else:
                run_stat_type = stat_type

            if run_stat_type == "t":
                t_res = results.t_contrast(contrast_vec)
                run_contrast = Contrast(
                    effect=np.atleast_1d(t_res["effect"]),
                    variance=np.atleast_1d(t_res["sd"]) ** 2,
                    dof=float(t_res["df_den"]),
                    stat_type="t",
                )
            else:
                f_res = results.f_contrast(contrast_vec)
                q = int(f_res["df_num"])
                # Per-voxel variance: mean of diagonal of each voxel's contrast
                # covariance matrix. covariance is always (V, q, q).
                variance = f_res["covariance"][:, np.arange(q), np.arange(q)].mean(
                    axis=1
                )  # (V,)
                run_contrast = Contrast(
                    effect=f_res["effect"],
                    variance=variance,
                    dof=float(f_res["df_den"]),
                    stat_type="F",
                    dim=int(q),
                )

            combined = run_contrast if combined is None else combined + run_contrast

        assert combined is not None  # At least one run guaranteed.
        return combined

    def _resolve_contrast_vector(
        self,
        contrast_def: str | npt.NDArray[np.floating],
        dm: pd.DataFrame,
        run_idx: int,
    ) -> npt.NDArray[np.floating]:
        """Resolve a contrast definition to a numeric vector or matrix.

        Parses string expressions via `expression_to_contrast_vector`, and zero-pads
        numeric arrays shorter than the number of design columns.

        Parameters
        ----------
        contrast_def : str or numpy.ndarray
            Contrast definition.
        dm : pandas.DataFrame
            Design matrix for the current run.
        run_idx : int
            Run index, used in error messages.

        Returns
        -------
        (n_columns,) or (q, n_columns) numpy.ndarray
            Numeric contrast vector or matrix.

        Raises
        ------
        ValueError
            If the contrast exceeds the number of design columns or has invalid
            dimensionality.
        """
        columns = list(dm.columns)

        if isinstance(contrast_def, str):
            return expression_to_contrast_vector(contrast_def, columns)

        # Pad or validate length.
        if contrast_def.ndim == 1:
            if contrast_def.shape[0] > len(columns):
                raise ValueError(
                    f"Contrast vector length ({contrast_def.shape[0]}) exceeds "
                    f"number of design columns ({len(columns)}) for run {run_idx}."
                )
            if contrast_def.shape[0] < len(columns):
                # Zero-padding needed e.g. when user only specifies condition
                # regressors.
                padded = np.zeros(len(columns))
                padded[: contrast_def.shape[0]] = contrast_def
                return padded
            return contrast_def

        if contrast_def.ndim == 2:
            if contrast_def.shape[1] > len(columns):
                raise ValueError(
                    f"Contrast matrix width ({contrast_def.shape[1]}) exceeds "
                    f"number of design columns ({len(columns)}) for run {run_idx}."
                )
            if contrast_def.shape[1] < len(columns):
                padded = np.zeros((contrast_def.shape[0], len(columns)))
                padded[:, : contrast_def.shape[1]] = contrast_def
                return padded
            return contrast_def

        raise ValueError("Contrast must be a string, 1-D, or 2-D array.")

    @staticmethod
    def _contrast_output(
        contrast: Contrast,
        output_type: str,
    ) -> npt.NDArray[np.floating]:
        """Extract the requested statistical map from a Contrast object.

        Parameters
        ----------
        contrast : Contrast
            Fitted contrast object.
        output_type : {"z_score", "stat", "p_value", "effect_size", "effect_variance"}
            Requested output.

        Returns
        -------
        (n_voxels,) numpy.ndarray
            Flat statistical map.

        Raises
        ------
        ValueError
            If `output_type` is not recognized.
        """
        if output_type == "z_score":
            return contrast.z_score()
        if output_type == "stat":
            return contrast.stat()
        if output_type == "p_value":
            return contrast.p_value()
        if output_type == "effect_size":
            return contrast.effect_size()
        if output_type == "effect_variance":
            return contrast.effect_variance()
        raise ValueError(
            f"output_type must be one of 'z_score', 'stat', 'p_value', "
            f"'effect_size', 'effect_variance', got '{output_type}'."
        )

    def _to_dataarray(self, flat: npt.NDArray[np.floating], name: str) -> xr.DataArray:
        """Reshape a flat voxel array into a spatial DataArray.

        Parameters
        ----------
        flat : (n_voxels,) or (contrast_dim, n_voxels) numpy.ndarray
            Flat statistical map.
        name : str
            Value for the `long_name` DataArray attribute.

        Returns
        -------
        xarray.DataArray
            Map reshaped to the original spatial dimensions of the input data.
        """
        spatial_shape = self._spatial_shapes[0]
        spatial_coords = self._run_coords[0]

        # For effect_size with F-contrast, first dim is contrast_dim.
        if flat.ndim == 2:
            # (contrast_dim, n_voxels) → (contrast_dim, *spatial_shape)
            volume = flat.reshape((-1, *spatial_shape))
            dims = ("contrast_dim", *self._spatial_dims)
        else:
            volume = flat.reshape(spatial_shape)
            dims = self._spatial_dims

        return xr.DataArray(
            volume,
            dims=dims,
            coords={
                d: spatial_coords[d] for d in self._spatial_dims if d in spatial_coords
            },
            attrs={"long_name": name, "cmap": "coolwarm"},
        )

    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not hasattr(self, "results_"):
            raise ValueError(
                "This FirstLevelModel instance is not fitted yet. "
                "Call 'fit' before using 'compute_contrast'."
            )

    def __sklearn_is_fitted__(self) -> bool:
        """Check whether the estimator has been fitted."""
        return hasattr(self, "results_")
