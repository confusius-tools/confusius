"""Second-level GLM for group-level fUSI analysis.

This module implements a sklearn-style estimator for fitting voxel-wise General Linear
Models to collections of first-level contrast maps, enabling group-level statistical
inference.

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator

from confusius.glm._contrasts import Contrast
from confusius.glm._models import OLSModel, RegressionResults
from confusius.glm._utils import (
    consensus_attrs,
    resolve_contrast_vector,
    select_contrast_map,
    to_spatial_dataarray,
)
from confusius.glm.first_level import FirstLevelModel
from confusius.validation.coordinates import validate_matching_coordinates

if TYPE_CHECKING:
    import numpy.typing as npt


def make_second_level_design_matrix(
    n_subjects: int,
    confounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a second-level design matrix.

    Creates a design matrix for group-level analysis. By default, includes only an
    intercept column (one-sample *t*-test / group mean). If confounds are provided, they
    are prepended as additional regressors.

    Parameters
    ----------
    n_subjects : int
        Number of subjects (rows in the design matrix).
    confounds : (n_subjects, n_confounds) pandas.DataFrame, optional
        Subject-level confound regressors. Columns are added before the intercept.

    Returns
    -------
    (n_subjects, n_confounds + 1) pandas.DataFrame
        Design matrix.

    Raises
    ------
    ValueError
        If `confounds` has a number of rows different from `n_subjects`.

    Examples
    --------
    >>> import pandas as pd
    >>> dm = make_second_level_design_matrix(5)
    >>> list(dm.columns)
    ['intercept']
    >>> dm.shape
    (5, 1)

    >>> confounds = pd.DataFrame({"age": [25, 30, 35, 40, 45]})
    >>> dm = make_second_level_design_matrix(5, confounds=confounds)
    >>> list(dm.columns)
    ['age', 'intercept']
    """
    intercept = pd.DataFrame({"intercept": np.ones(n_subjects)}, dtype=np.float64)

    if confounds is not None:
        if len(confounds) != n_subjects:
            raise ValueError(
                f"confounds has {len(confounds)} rows but n_subjects={n_subjects}."
            )
        # Reject duplicates inside `confounds` and any collision with the
        # auto-added intercept.
        confound_columns = list(confounds.columns)
        duplicates = sorted(
            {c for c in confound_columns if confound_columns.count(c) > 1}
        )
        if duplicates:
            raise ValueError(f"confounds has duplicate column names: {duplicates}.")
        if "intercept" in confound_columns:
            raise ValueError(
                "confounds contains an 'intercept' column, which collides "
                "with the auto-added intercept regressor. Rename it before "
                "passing it in."
            )
        return pd.concat([confounds.reset_index(drop=True), intercept], axis=1)

    return intercept


def _validate_second_level_maps(maps: list[xr.DataArray]) -> None:
    """Reject second-level inputs that aren't purely spatial contrast maps.

    A stray `time` dimension would be flattened together with the spatial axes, and an
    F-contrast effect map (extracted from a `FirstLevelModel` via
    `output_type="effect"`) carries a leading `contrast_dim` axis whose components are
    not interchangeable subjects.

    Parameters
    ----------
    maps : list of xarray.DataArray
        Per-subject contrast maps to validate.

    Raises
    ------
    ValueError
        If any map carries a `time` or `contrast_dim` axis. *F*-contrasts at the group
        level should be requested through `SecondLevelModel.compute_contrast(...,
        stat_type="F")` on scalar-effect maps (e.g. first-level *t*-contrasts), not by
        feeding first-level F-contrast effect maps as input.
    """
    for i, m in enumerate(maps):
        if "time" in m.dims:
            raise ValueError(
                f"second_level_input[{i}] has a 'time' dimension. "
                "SecondLevelModel expects spatial contrast maps "
                "(e.g. FirstLevelModel.compute_contrast(..., "
                'output_type="effect")), not raw time series.'
            )
        if "contrast_dim" in m.dims:
            raise ValueError(
                f"second_level_input[{i}] has a 'contrast_dim' axis, which "
                "indicates a multi-row first-level F-contrast effect map. "
                "SecondLevelModel expects scalar-effect maps; pass "
                "first-level *t*-contrasts and request F at the group "
                'level via `compute_contrast(..., stat_type="F")`.'
            )


class SecondLevelModel(BaseEstimator):
    """Second-level GLM estimator for group-level fUSI analysis.

    Fits a voxel-wise OLS model to a collection of first-level contrast maps (one per
    subject or session). This is equivalent to a mass-univariate one-sample *t*-test by
    default, but any linear group-level model can be specified via a design matrix.

    `second_level_input` accepts either a list of fitted
    [`FirstLevelModel`][confusius.glm.first_level.FirstLevelModel] objects or a list of
    spatial `xarray.DataArray` maps (e.g. output of
    [`compute_contrast`][confusius.glm.first_level.FirstLevelModel.compute_contrast]
    with `output_type="effect"`). When `FirstLevelModel` objects are passed,
    `first_level_contrast` must be provided so the effect map can be extracted
    automatically from each model.

    This implementation is adapted from [`nilearn.glm.second_level.SecondLevelModel`][
    nilearn.glm.second_level.SecondLevelModel].

    Attributes
    ----------
    design_matrix_ : pandas.DataFrame
        Design matrix used for fitting (set after
        [`fit`][confusius.glm.second_level.SecondLevelModel.fit]).
    results_ : RegressionResults
        Regression results (set after
        [`fit`][confusius.glm.second_level.SecondLevelModel.fit]).

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> maps = [
    ...     xr.DataArray(np.random.randn(2, 3, 4), dims=["z", "y", "x"])
    ...     for _ in range(10)
    ... ]
    >>> model = SecondLevelModel()
    >>> model.fit(maps)  # doctest: +SKIP
    >>> z_map = model.compute_contrast("intercept")  # doctest: +SKIP
    """

    def fit(
        self,
        second_level_input: list[xr.DataArray] | list[FirstLevelModel],
        first_level_contrast: str | npt.NDArray[np.floating] | None = None,
        confounds: pd.DataFrame | None = None,
        design_matrix: pd.DataFrame | None = None,
    ) -> SecondLevelModel:
        """Fit the group-level GLM.

        Accepts either a list of fitted
        [`FirstLevelModel`][confusius.glm.first_level.FirstLevelModel] objects or a list
        of spatial contrast maps (`xarray.DataArray` with no `time` dimension). All maps
        must share the same spatial dimensions and shape.

        Parameters
        ----------
        second_level_input : list of FirstLevelModel or list of xarray.DataArray
            Input data. When a list of
            [`FirstLevelModel`][confusius.glm.first_level.FirstLevelModel] is provided,
            `first_level_contrast` must be given and effect maps are extracted
            automatically via
            [`compute_contrast`][confusius.glm.first_level.FirstLevelModel.compute_contrast].
            When a list of `xarray.DataArray` is provided, maps are used directly.
        first_level_contrast : str or numpy.ndarray, optional
            Contrast passed to each `FirstLevelModel.compute_contrast` call when
            `second_level_input` is a list of `FirstLevelModel`. Ignored otherwise.
        confounds : (n_subjects, n_confounds) pandas.DataFrame, optional
            Subject-level confound regressors in the same order as
            `second_level_input`. Ignored when `design_matrix` is provided.
        design_matrix : (n_subjects, n_regressors) pandas.DataFrame, optional
            Pre-built group-level design matrix. When provided, `confounds` is ignored.

        Returns
        -------
        SecondLevelModel
            Fitted estimator (`self`).

        Raises
        ------
        ValueError
            If `second_level_input` is empty, if `FirstLevelModel` objects are passed
            without `first_level_contrast`, if maps have inconsistent spatial shapes
            or mismatched spatial coordinates, or if `design_matrix` row count does
            not match the number of inputs.
        TypeError
            If `second_level_input` is not a list of `FirstLevelModel` or
            `xarray.DataArray`.
        """
        if not isinstance(second_level_input, list) or len(second_level_input) == 0:
            raise ValueError("second_level_input must be a non-empty list.")

        maps = self._resolve_input(second_level_input, first_level_contrast)

        n_subjects = len(maps)
        ref = maps[0]
        ref_dims = tuple(str(d) for d in ref.dims)
        ref_shape = ref.shape

        for i, da in enumerate(maps[1:], start=1):
            if tuple(str(d) for d in da.dims) != ref_dims:
                raise ValueError(
                    f"All maps must have the same dimensions. "
                    f"Map 0 has dims {ref_dims}, map {i} has "
                    f"{tuple(str(d) for d in da.dims)}."
                )
            if da.shape != ref_shape:
                raise ValueError(
                    f"All maps must have the same shape. "
                    f"Map 0 has shape {ref_shape}, map {i} has {da.shape}."
                )
            # Include dims where at least one side carries a coord:
            # validate_matching_coordinates raises if it's missing on one
            # side, catching asymmetric coord drops.
            checkable = [d for d in ref_dims if d in ref.coords or d in da.coords]
            if checkable:
                validate_matching_coordinates(
                    ref,
                    da,
                    checkable,
                    left_name="map 0",
                    right_name=f"map {i}",
                )

        if design_matrix is not None:
            if len(design_matrix) != n_subjects:
                raise ValueError(
                    f"design_matrix has {len(design_matrix)} rows but "
                    f"second_level_input has {n_subjects} maps."
                )
            dm = design_matrix
        else:
            dm = make_second_level_design_matrix(n_subjects, confounds)

        Y = np.stack([da.values.ravel() for da in maps], axis=0)

        model = OLSModel(dm.to_numpy(dtype=np.float64))
        self.results_: RegressionResults = model.fit(Y)
        self.design_matrix_: pd.DataFrame = dm
        self._spatial_dims: tuple[str, ...] = ref_dims
        self._spatial_shape: tuple[int, ...] = ref_shape
        self._coords: dict[str, xr.Variable] = {
            str(d): ref.coords[d] for d in ref_dims if d in ref.coords
        }
        self._input_attrs: dict[str, object] = consensus_attrs(maps)

        return self

    @staticmethod
    def _resolve_input(
        second_level_input: list[xr.DataArray] | list[FirstLevelModel],
        first_level_contrast: str | npt.NDArray[np.floating] | None,
    ) -> list[xr.DataArray]:
        """Resolve second_level_input to a list of spatial DataArrays.

        Parameters
        ----------
        second_level_input : list of FirstLevelModel or list of xarray.DataArray
            Input data.
        first_level_contrast : str or numpy.ndarray or None
            Contrast for extracting effect maps from `FirstLevelModel` objects.

        Returns
        -------
        list of xarray.DataArray
            Spatial contrast maps ready for fitting.

        Raises
        ------
        ValueError
            If `FirstLevelModel` objects are passed without `first_level_contrast`.
        TypeError
            If elements are neither `FirstLevelModel` nor `xarray.DataArray`.
        """
        if all(isinstance(m, FirstLevelModel) for m in second_level_input):
            if first_level_contrast is None:
                raise ValueError(
                    "first_level_contrast must be provided when second_level_input "
                    "is a list of FirstLevelModel."
                )
            maps = [
                m.compute_contrast(first_level_contrast, output_type="effect")
                for m in second_level_input
            ]
            _validate_second_level_maps(maps)
            return maps

        if all(isinstance(m, xr.DataArray) for m in second_level_input):
            maps = [m for m in second_level_input if isinstance(m, xr.DataArray)]
            _validate_second_level_maps(maps)
            return maps

        raise TypeError(
            "second_level_input must be a list of FirstLevelModel or "
            "a list of xarray.DataArray, not a mix of both."
        )

    def compute_contrast(
        self,
        second_level_contrast: str | npt.NDArray[np.floating] = "intercept",
        stat_type: Literal["t", "F"] | None = None,
        output_type: Literal[
            "zscore", "statistic", "pvalue", "effect", "variance"
        ] = "zscore",
        baseline: float = 0.0,
    ) -> xr.DataArray:
        """Compute a group-level contrast and return a statistical map.

        Parameters
        ----------
        second_level_contrast : str or numpy.ndarray, default: "intercept"
            Contrast definition. A string is parsed as an expression over design-matrix
            column names (e.g. `"group_A - group_B"`). A 1D array specifies a
            *t*-contrast vector; a 2D array specifies an *F*-contrast matrix. Defaults
            to `"intercept"` for a one-sample *t*-test.
        stat_type : {"t", "F"}, optional
            Force the contrast type. By default inferred from the shape of the contrast
            (1D → *t*, 2D → *F*).
        output_type : {"zscore", "statistic", "pvalue", "effect", "variance"}, default: "zscore"
            Which statistical map to return.
        baseline : float, default: 0.0
            Null-hypothesis value tested against. The statistic is
            `(effect - baseline) / sqrt(variance)` for *t*-contrasts and
            `sum((effect - baseline)**2) / dim / variance` for *F*-contrasts.

        Returns
        -------
        xarray.DataArray
            Statistical map with the spatial dimensions of the input maps.

        Raises
        ------
        ValueError
            If the model has not been fitted or the contrast definition is invalid.
        """
        self._check_is_fitted()

        contrast_vec = resolve_contrast_vector(
            second_level_contrast, list(self.design_matrix_.columns)
        )
        contrast_obj = Contrast.from_results(
            self.results_, contrast_vec, stat_type=stat_type, baseline=baseline
        )
        flat = select_contrast_map(contrast_obj, output_type)
        return to_spatial_dataarray(
            flat,
            spatial_dims=self._spatial_dims,
            spatial_shape=self._spatial_shape,
            coords=self._coords,
            attrs=self._input_attrs,
            name=output_type,
        )

    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not hasattr(self, "results_"):
            raise ValueError("This SecondLevelModel is not fitted. Call fit() first.")

    def __sklearn_is_fitted__(self) -> bool:
        """Return True if the model has been fitted."""
        return hasattr(self, "results_")
