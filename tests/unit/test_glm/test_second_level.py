"""Tests for confusius.glm.second_level."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.glm import FirstLevelModel, SecondLevelModel, make_second_level_design_matrix
from confusius.glm._models import OLSModel


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def spatial_maps(rng):
    """10 spatial maps of shape (2, 3, 4) for group-level testing."""
    return [
        xr.DataArray(
            rng.standard_normal((2, 3, 4)),
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(2) * 0.5,
                "y": np.arange(3) * 0.1,
                "x": np.arange(4) * 0.1,
            },
        )
        for _ in range(10)
    ]


@pytest.fixture
def spatial_maps_2d(rng):
    """8 spatial maps of shape (5, 6) for group-level testing."""
    return [
        xr.DataArray(
            rng.standard_normal((5, 6)),
            dims=["y", "x"],
        )
        for _ in range(8)
    ]


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

    def test_basic_fit(self, spatial_maps):
        model = SecondLevelModel()
        result = model.fit(spatial_maps)
        assert result is model
        assert hasattr(model, "results_")
        assert hasattr(model, "design_matrix_")
        assert list(model.design_matrix_.columns) == ["intercept"]
        assert len(model.design_matrix_) == 10

    def test_fit_2d_spatial(self, spatial_maps_2d):
        model = SecondLevelModel()
        model.fit(spatial_maps_2d)
        z_map = model.compute_contrast("intercept")
        assert z_map.dims == ("y", "x")
        assert z_map.shape == (5, 6)

    def test_fit_with_confounds(self, spatial_maps, rng):
        confounds = pd.DataFrame({"age": rng.standard_normal(10)})
        model = SecondLevelModel()
        model.fit(spatial_maps, confounds=confounds)
        assert "age" in model.design_matrix_.columns
        assert "intercept" in model.design_matrix_.columns

    def test_fit_with_prebuilt_design_matrix(self, spatial_maps):
        dm = make_second_level_design_matrix(10)
        model = SecondLevelModel()
        model.fit(spatial_maps, design_matrix=dm)
        assert model.design_matrix_ is dm

    def test_fit_preserves_spatial_coords(self, spatial_maps):
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

    def test_default_intercept_contrast(self):
        z_map = self.model.compute_contrast()
        assert z_map.dims == ("z", "y", "x")
        assert z_map.shape == (2, 3, 4)
        assert np.all(np.isfinite(z_map.values))

    def test_explicit_string_contrast(self):
        z_map = self.model.compute_contrast("intercept")
        assert z_map.shape == (2, 3, 4)

    def test_array_contrast(self):
        vec = np.array([1.0])
        z_map = self.model.compute_contrast(vec)
        assert z_map.shape == (2, 3, 4)

    def test_output_type_statistic(self):
        t_map = self.model.compute_contrast(output_type="statistic")
        assert t_map.attrs["long_name"] == "statistic"

    def test_output_type_pvalue(self):
        p_map = self.model.compute_contrast(output_type="pvalue")
        assert np.all(p_map.values >= 0)
        assert np.all(p_map.values <= 1)

    def test_output_type_effect(self):
        e_map = self.model.compute_contrast(output_type="effect")
        assert e_map.shape == (2, 3, 4)

    def test_output_type_variance(self):
        v_map = self.model.compute_contrast(output_type="variance")
        assert np.all(v_map.values >= 0)

    def test_explicit_stat_type_t(self):
        z_map = self.model.compute_contrast("intercept", stat_type="t")
        assert z_map.shape == (2, 3, 4)

    def test_invalid_output_type_raises(self):
        with pytest.raises(ValueError, match="output_type"):
            self.model.compute_contrast(output_type="invalid")


class TestSecondLevelModelFContrast:
    """Test F-contrast path."""

    def test_f_contrast_2d_array(self, spatial_maps, rng):
        """F-contrast with a 2D matrix produces a spatial z-map."""
        confounds = pd.DataFrame(
            {"group": [0.0] * 5 + [1.0] * 5}
        )
        model = SecondLevelModel()
        model.fit(spatial_maps, confounds=confounds)
        # 2-row contrast over group and intercept.
        c = np.eye(2)
        z_map = model.compute_contrast(c, stat_type="F")
        assert z_map.shape == (2, 3, 4)
        assert np.all(np.isfinite(z_map.values))

    def test_f_contrast_effect_has_contrast_dim(self, spatial_maps, rng):
        """F-contrast effect output has a contrast_dim dimension."""
        confounds = pd.DataFrame({"group": [0.0] * 5 + [1.0] * 5})
        model = SecondLevelModel()
        model.fit(spatial_maps, confounds=confounds)
        c = np.eye(2)
        e_map = model.compute_contrast(c, stat_type="F", output_type="effect")
        assert e_map.dims[0] == "contrast_dim"
        assert e_map.shape[0] == 2


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
        t_res = results.t_contrast(np.array([1.0]))

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
            model.fit(xr.DataArray(np.ones((5, 3))))

    def test_list_of_non_dataarrays_raises(self):
        model = SecondLevelModel()
        with pytest.raises(TypeError, match="FirstLevelModel or"):
            model.fit([np.ones((3, 4)), np.ones((3, 4))])

    def test_empty_input_raises(self):
        model = SecondLevelModel()
        with pytest.raises(ValueError, match="non-empty list"):
            model.fit([])

    def test_first_level_models_without_contrast_raises(self, rng):
        """Passing FirstLevelModel list without first_level_contrast raises."""
        frame_times = np.arange(50) * 0.5
        events = pd.DataFrame(
            {"trial_type": ["A"], "onset": [5.0], "duration": [1.0]}
        )
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
        events = pd.DataFrame(
            {"trial_type": ["A"], "onset": [5.0], "duration": [1.0]}
        )
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
        with pytest.raises(ValueError, match="string, 1-D, or 2-D"):
            model.compute_contrast(np.ones((2, 3, 4)))
