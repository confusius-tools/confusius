"""Tests for confusius.glm.second_level."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.glm import (
    FirstLevelModel,
    SecondLevelModel,
    make_second_level_design_matrix,
)
from confusius.glm._models import OLSModel


# -----------------------------------------------------------------------------
# make_second_level_design_matrix tests
# -----------------------------------------------------------------------------


class TestMakeSecondLevelDesignMatrix:
    """Tests for make_second_level_design_matrix."""

    def test_intercept_only(self):
        dm = make_second_level_design_matrix(5)
        assert list(dm.columns) == ["intercept"]
        assert dm.shape == (5, 1)
        assert_allclose(dm["intercept"].values, np.ones(5))

    def test_with_confounds(self):
        confounds = pd.DataFrame({"age": [25.0, 30.0, 35.0]})
        dm = make_second_level_design_matrix(3, confounds=confounds)
        assert list(dm.columns) == ["age", "intercept"]
        assert dm.shape == (3, 2)

    def test_confounds_row_mismatch_raises(self):
        confounds = pd.DataFrame({"age": [25.0, 30.0]})
        with pytest.raises(ValueError, match="rows"):
            make_second_level_design_matrix(5, confounds=confounds)


# -----------------------------------------------------------------------------
# SecondLevelModel: fitting
# -----------------------------------------------------------------------------


class TestSecondLevelModelFit:
    """Tests for SecondLevelModel.fit."""

    def test_fit_2d_spatial(self, spatial_maps_2d):
        """Fitting `(y, x)` maps yields a contrast map with the same dims/shape."""
        model = SecondLevelModel()
        model.fit(spatial_maps_2d)
        z_map = model.compute_contrast("intercept")
        assert z_map.dims == ("y", "x")
        assert z_map.shape == (5, 6)

    def test_fit_confounds_pass_through_to_design_matrix(self, spatial_maps, rng):
        """User confounds appear in the design matrix with the user's values."""
        confounds = pd.DataFrame({"age": rng.standard_normal(10)})
        model = SecondLevelModel()
        model.fit(spatial_maps, confounds=confounds)
        assert_allclose(
            model.design_matrix_["age"].to_numpy(), confounds["age"].to_numpy()
        )
        assert_allclose(model.design_matrix_["intercept"].to_numpy(), np.ones(10))

    def test_fit_with_prebuilt_design_matrix(self, spatial_maps):
        """A pre-built design matrix is used as-is."""
        dm = make_second_level_design_matrix(10)
        model = SecondLevelModel()
        model.fit(spatial_maps, design_matrix=dm)
        assert model.design_matrix_ is dm

    def test_fit_preserves_spatial_coords(self, spatial_maps):
        """Spatial coords are propagated from the inputs to the contrast map."""
        model = SecondLevelModel()
        model.fit(spatial_maps)
        z_map = model.compute_contrast("intercept")
        assert_allclose(z_map.coords["z"].values, spatial_maps[0].coords["z"].values)
        assert_allclose(z_map.coords["y"].values, spatial_maps[0].coords["y"].values)
        assert_allclose(z_map.coords["x"].values, spatial_maps[0].coords["x"].values)

    def test_consensus_attrs_propagated_across_maps(self, spatial_maps):
        """Attributes equal across all maps propagate; conflicting ones are dropped."""
        for i, da in enumerate(spatial_maps):
            da.attrs.update(
                {"task": "stim", "contrast": "A-B", "subject_id": f"s{i:02d}"}
            )
        model = SecondLevelModel()
        model.fit(spatial_maps)
        z_map = model.compute_contrast("intercept")
        assert z_map.attrs["task"] == "stim"
        assert z_map.attrs["contrast"] == "A-B"
        assert "subject_id" not in z_map.attrs
        assert z_map.attrs["long_name"] == "zscore"
        assert z_map.attrs["cmap"] == "coolwarm"

    def test_sklearn_is_fitted(self, spatial_maps):
        model = SecondLevelModel()
        assert not model.__sklearn_is_fitted__()
        model.fit(spatial_maps)
        assert model.__sklearn_is_fitted__()

    def test_fit_from_first_level_models(self, rng):
        """Fitting from a list of FirstLevelModel objects extracts effect maps."""
        frame_times = np.arange(100) * 0.5
        events = pd.DataFrame(
            {
                "trial_type": ["A"] * 5 + ["B"] * 5,
                "onset": np.arange(10) * 5.0,
                "duration": [1.0] * 10,
            }
        )
        first_level_models = []
        for _ in range(5):
            data = xr.DataArray(
                rng.standard_normal((100, 2, 3)),
                dims=["time", "y", "x"],
                coords={"time": frame_times},
            )
            m = FirstLevelModel(noise_model="ols")
            m.fit(data, events=events)
            first_level_models.append(m)

        group_model = SecondLevelModel()
        group_model.fit(first_level_models, first_level_contrast="A - B")

        assert group_model.__sklearn_is_fitted__()
        z_map = group_model.compute_contrast("intercept")
        assert z_map.dims == ("y", "x")
        assert z_map.shape == (2, 3)


# -----------------------------------------------------------------------------
# SecondLevelModel: contrasts
# -----------------------------------------------------------------------------


class TestSecondLevelModelContrast:
    """Tests for SecondLevelModel.compute_contrast."""

    @pytest.fixture(autouse=True)
    def _fitted_model(self, spatial_maps):
        self.model = SecondLevelModel()
        self.model.fit(spatial_maps)

    def test_default_and_explicit_intercept_match(self):
        """Default contrast (`compute_contrast()`) is `"intercept"`, and
        passing `"intercept"` explicitly must yield identical maps."""
        z_default = self.model.compute_contrast()
        z_explicit = self.model.compute_contrast("intercept")

        assert_allclose(z_default.values, z_explicit.values, rtol=1e-12)
        assert z_default.dims == ("z", "y", "x")

    def test_string_and_array_contrast_agree(self):
        """The string `"intercept"` resolves to `[1.0]`; both must produce
        identical maps."""
        z_string = self.model.compute_contrast("intercept")
        z_array = self.model.compute_contrast(np.array([1.0]))

        assert_allclose(z_string.values, z_array.values, rtol=1e-12)

    def test_output_type_pvalue_in_unit_interval(self):
        p_map = self.model.compute_contrast(output_type="pvalue")
        assert np.all((p_map.values >= 0) & (p_map.values <= 1))

    def test_output_type_variance_non_negative(self):
        v_map = self.model.compute_contrast(output_type="variance")
        assert np.all(v_map.values >= 0)

    def test_invalid_output_type_raises(self):
        with pytest.raises(ValueError, match="output_type"):
            self.model.compute_contrast(output_type="invalid")  # ty: ignore[invalid-argument-type]


class TestSecondLevelModelFContrast:
    """Test F-contrast path."""

    def test_f_contrast_effect_has_contrast_dim_axis(self, spatial_maps):
        """F-contrast `output_type="effect"` returns a `(contrast_dim, *spatial)`
        array whose components remain accessible for downstream inspection."""
        confounds = pd.DataFrame({"group": [0.0] * 5 + [1.0] * 5})
        model = SecondLevelModel()
        model.fit(spatial_maps, confounds=confounds)
        c = np.eye(2)
        e_map = model.compute_contrast(c, stat_type="F", output_type="effect")
        assert e_map.dims == ("contrast_dim", "z", "y", "x")
        assert e_map.shape == (2, 2, 3, 4)

    def test_short_contrast_vector_is_zero_padded(self, spatial_maps):
        """A 1D contrast shorter than the design is zero-padded on the right.

        The result must equal the corresponding manually-padded contrast,
        not just match in shape.
        """
        confounds = pd.DataFrame({"group": [0.0] * 5 + [1.0] * 5})
        model = SecondLevelModel()
        model.fit(spatial_maps, confounds=confounds)
        # Short vector targets only the first column (group).
        short = np.array([1.0])
        padded = np.array([1.0, 0.0])
        z_short = model.compute_contrast(short).values
        z_padded = model.compute_contrast(padded).values
        assert_allclose(z_short, z_padded, rtol=1e-12, atol=1e-12)

    def test_short_contrast_matrix_is_zero_padded(self, spatial_maps):
        """A 2D contrast narrower than the design is zero-padded on the right."""
        confounds = pd.DataFrame({"group": [0.0] * 5 + [1.0] * 5})
        model = SecondLevelModel()
        model.fit(spatial_maps, confounds=confounds)
        short = np.array([[1.0]])
        padded = np.array([[1.0, 0.0]])
        z_short = model.compute_contrast(short, stat_type="F").values
        z_padded = model.compute_contrast(padded, stat_type="F").values
        assert_allclose(z_short, z_padded, rtol=1e-12, atol=1e-12)


# -----------------------------------------------------------------------------
# SecondLevelModel: reference check against manual computation
# -----------------------------------------------------------------------------


class TestSecondLevelModelReference:
    """Verify SecondLevelModel matches manual low-level computation."""

    def test_matches_manual_ols(self, spatial_maps):
        model = SecondLevelModel()
        model.fit(spatial_maps)
        z_map_auto = model.compute_contrast("intercept")

        # Manual computation.
        Y = np.stack([da.values.ravel() for da in spatial_maps], axis=0)
        X = np.ones((10, 1))
        ols = OLSModel(X)
        results = ols.fit(Y)
        t_res = results.compute_t_contrast(np.array([1.0]))

        from confusius.glm._contrasts import Contrast

        contrast = Contrast.from_estimate(
            effect=np.atleast_1d(t_res["effect"]),
            variance=np.atleast_1d(t_res["sd"]) ** 2,
            dof=float(t_res["df_den"]),
            stat_type="t",
        )
        z_manual = contrast.zscore.reshape(2, 3, 4)

        assert_allclose(z_map_auto.values, z_manual, rtol=1e-10)


# -----------------------------------------------------------------------------
# SecondLevelModel: error handling
# -----------------------------------------------------------------------------


class TestSecondLevelModelErrors:
    """Tests for error handling."""

    def test_not_list_raises(self):
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="non-empty list"):
            model.fit(xr.DataArray(np.ones((5, 3))))  # ty: ignore[invalid-argument-type]

    def test_list_of_non_dataarrays_raises(self):
        model = SecondLevelModel()
        with pytest.raises(TypeError, match="FirstLevelModel or"):
            model.fit([np.ones((3, 4)), np.ones((3, 4))])  # ty: ignore[invalid-argument-type]

    def test_empty_input_raises(self):
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="non-empty list"):
            model.fit([])

    def test_first_level_models_without_contrast_raises(self, rng):
        """Passing FirstLevelModel list without first_level_contrast raises."""
        frame_times = np.arange(50) * 0.5
        events = pd.DataFrame({"trial_type": ["A"], "onset": [5.0], "duration": [1.0]})
        data = xr.DataArray(
            rng.standard_normal((50, 2, 3)),
            dims=["time", "y", "x"],
            coords={"time": frame_times},
        )
        m = FirstLevelModel(noise_model="ols")
        m.fit(data, events=events)
        group_model = SecondLevelModel()
        with pytest.raises(ValueError, match="first_level_contrast"):
            group_model.fit([m, m])

    def test_mixed_input_types_raises(self, spatial_maps):
        """Mix of FirstLevelModel and DataArray raises TypeError."""
        frame_times = np.arange(50) * 0.5
        events = pd.DataFrame({"trial_type": ["A"], "onset": [5.0], "duration": [1.0]})
        data = xr.DataArray(
            np.zeros((50, 2, 3)),
            dims=["time", "y", "x"],
            coords={"time": frame_times},
        )
        m = FirstLevelModel(noise_model="ols")
        m.fit(data, events=events)
        group_model = SecondLevelModel()
        with pytest.raises(TypeError, match="mix"):
            group_model.fit([m, spatial_maps[0]])

    def test_shape_mismatch_raises(self):
        maps = [
            xr.DataArray(np.ones((2, 3)), dims=["y", "x"]),
            xr.DataArray(np.ones((4, 3)), dims=["y", "x"]),
        ]
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="shape"):
            model.fit(maps)

    def test_dims_mismatch_raises(self):
        maps = [
            xr.DataArray(np.ones((2, 3)), dims=["y", "x"]),
            xr.DataArray(np.ones((2, 3)), dims=["z", "x"]),
        ]
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="dimensions"):
            model.fit(maps)

    def test_spatial_coord_mismatch_raises(self, spatial_maps):
        """Mismatched spatial coordinates between maps raise an informative error."""
        bad = spatial_maps[1].assign_coords(z=spatial_maps[0].coords["z"] + 10.0)
        maps = [spatial_maps[0], bad]
        model = SecondLevelModel()
        with pytest.raises(
            ValueError,
            match=r"Coordinate 'z' does not match between map 0 and map 1",
        ):
            model.fit(maps)

    def test_design_matrix_row_mismatch_raises(self):
        maps = [xr.DataArray(np.ones((2, 3)), dims=["y", "x"]) for _ in range(5)]
        dm = make_second_level_design_matrix(3)
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="design_matrix"):
            model.fit(maps, design_matrix=dm)

    def test_dataarray_with_time_dim_raises(self):
        """Raw DataArray inputs with a `time` dim are rejected."""
        maps = [
            xr.DataArray(np.zeros((10, 2, 3)), dims=["time", "y", "x"])
            for _ in range(5)
        ]
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="time' dimension"):
            model.fit(maps)

    def test_first_level_f_contrast_effect_rejected(self):
        """First-level F-contrast effect maps (with leading `contrast_dim`)
        are rejected at the second-level boundary.

        `FirstLevelModel.compute_contrast(F_matrix, output_type="effect")`
        returns `(contrast_dim, *spatial)` arrays whose components are not
        interchangeable subjects. Passing them as group-level inputs would
        silently flatten contrast components into the spatial axes.
        """
        maps = [
            xr.DataArray(
                np.zeros((2, 3, 4)),
                dims=["contrast_dim", "y", "x"],
            )
            for _ in range(5)
        ]
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="contrast_dim"):
            model.fit(maps)

    def test_dropped_spatial_coord_raises(self, spatial_maps):
        """A subject map missing a spatial coord present on the reference is rejected."""
        maps = [spatial_maps[0], spatial_maps[1].drop_vars("z")]
        model = SecondLevelModel()
        with pytest.raises(ValueError, match=r"Coordinate 'z' is missing from map 1"):
            model.fit(maps)

    def test_duplicate_confound_columns_raises(self, spatial_maps):
        """Duplicate column names in confounds are rejected."""
        confounds = pd.DataFrame(
            np.zeros((10, 2)),
            columns=["age", "age"],  # duplicate
        )
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="duplicate column"):
            model.fit(spatial_maps, confounds=confounds)

    def test_intercept_in_confounds_raises(self, spatial_maps):
        """A confounds 'intercept' column collides with the auto-added one."""
        confounds = pd.DataFrame({"intercept": np.ones(10)})
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="intercept"):
            model.fit(spatial_maps, confounds=confounds)

    def test_contrast_before_fit_raises(self):
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.compute_contrast()

    def test_contrast_vector_too_long_raises(self):
        maps = [xr.DataArray(np.ones((2, 3)), dims=["y", "x"]) for _ in range(5)]
        model = SecondLevelModel()
        model.fit(maps)
        with pytest.raises(ValueError, match="exceeds"):
            model.compute_contrast(np.ones(5))

    def test_2d_contrast_too_wide_raises(self):
        maps = [xr.DataArray(np.ones((2, 3)), dims=["y", "x"]) for _ in range(5)]
        model = SecondLevelModel()
        model.fit(maps)
        with pytest.raises(ValueError, match="exceeds"):
            model.compute_contrast(np.ones((2, 5)), stat_type="F")

    def test_3d_contrast_raises(self):
        maps = [xr.DataArray(np.ones((2, 3)), dims=["y", "x"]) for _ in range(5)]
        model = SecondLevelModel()
        model.fit(maps)
        with pytest.raises(ValueError, match="string, 1D, or 2D"):
            model.compute_contrast(np.ones((2, 3, 4)))
