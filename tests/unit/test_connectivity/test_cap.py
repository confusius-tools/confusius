"""Tests for confusius.connectivity.CAP."""

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from sklearn.exceptions import NotFittedError

from confusius.connectivity import CAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recordings(rng):
    """Three recordings of different lengths sharing the same spatial grid."""
    ny, nx = 5, 8
    out = []
    for n_time in (50, 60, 40):
        values = rng.standard_normal((n_time, ny, nx))
        out.append(
            xr.DataArray(
                values,
                dims=["time", "y", "x"],
                coords={
                    "time": np.arange(n_time) * 0.1,
                    "y": np.arange(ny) * 0.1,
                    "x": np.arange(nx) * 0.05,
                },
            )
        )
    return out


@pytest.fixture
def fitted_cap(recordings):
    """CAP fitted on three recordings with correlation metric."""
    cap = CAP(n_clusters=4, random_state=0)
    cap.fit(recordings)
    return cap


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFit:
    def test_labels_time_coords_preserved(self, fitted_cap, recordings):
        for lbl, rec in zip(fitted_cap.labels_, recordings):
            npt.assert_array_equal(lbl.coords["time"].values, rec.coords["time"].values)

    def test_labels_within_range(self, fitted_cap):
        n_caps = fitted_cap.caps_.sizes["cap"]
        for lbl in fitted_cap.labels_:
            assert lbl.min() >= 0
            assert lbl.max() < n_caps

    def test_empty_list_raises(self):
        cap = CAP(n_clusters=3)
        with pytest.raises(ValueError, match="at least one recording"):
            cap.fit([])

    def test_invalid_metric_raises(self, sample_4d_volume):
        cap = CAP(n_clusters=3, metric="manhattan")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="metric"):
            cap.fit([sample_4d_volume])

    def test_reproducibility(self, sample_4d_volume):
        cap1 = CAP(n_clusters=4, random_state=42).fit([sample_4d_volume])
        cap2 = CAP(n_clusters=4, random_state=42).fit([sample_4d_volume])
        npt.assert_array_equal(cap1.labels_[0].values, cap2.labels_[0].values)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_same_data_matches_fit_labels(self, fitted_cap, recordings):
        predicted = fitted_cap.predict(recordings)
        for lbl_fit, lbl_pred in zip(fitted_cap.labels_, predicted):
            npt.assert_array_equal(lbl_fit.values, lbl_pred.values)

    def test_predict_within_range(self, fitted_cap, recordings):
        n_caps = fitted_cap.caps_.sizes["cap"]
        for lbl in fitted_cap.predict(recordings):
            assert lbl.min() >= 0
            assert lbl.max() < n_caps

    def test_predict_unfitted_raises(self, recordings):
        cap = CAP(n_clusters=4)
        with pytest.raises(NotFittedError):
            cap.predict(recordings)


# ---------------------------------------------------------------------------
# compute_temporal_metrics()
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_temporal_fraction_sums_to_one(self, fitted_cap):
        ds = fitted_cap.compute_temporal_metrics()
        totals = ds["temporal_fraction"].sum("cap").values
        npt.assert_allclose(totals, 1.0, atol=1e-10)

    def test_temporal_fraction_non_negative(self, fitted_cap):
        ds = fitted_cap.compute_temporal_metrics()
        assert (ds["temporal_fraction"].values >= 0).all()

    def test_transition_matrix_row_sums(self, fitted_cap):
        ds = fitted_cap.compute_temporal_metrics()
        tm = ds["transition_matrix"].values  # (recording, cap_from, cap_to)
        row_sums = tm.sum(axis=-1)  # (recording, cap_from)
        # Rows can be 0 (CAP never appears as origin) or 1.
        assert ((row_sums == 0.0) | np.isclose(row_sums, 1.0)).all()

    def test_persistence_uses_time_coords(self):
        """Persistence is derived from time coordinate differences."""
        rng = np.random.default_rng(7)
        ny, nx = 2, 2
        center_0 = np.ones((ny, nx))
        center_1 = -np.ones((ny, nx))
        noise = 0.01

        # [0, 0, 1, 1] at 0.5 s intervals → each frame duration = 0.5 s.
        seq = [0, 0, 1, 1]
        frames = [
            (center_0 if s == 0 else center_1) + rng.standard_normal((ny, nx)) * noise
            for s in seq
        ]
        rec = xr.DataArray(
            np.stack(frames),
            dims=["time", "y", "x"],
            coords={"time": np.array([0.0, 0.5, 1.0, 1.5])},
        )
        cap = CAP(n_clusters=2, metric="euclidean", random_state=0)
        cap.fit([rec])

        caps_flat = cap.caps_.stack(space=["y", "x"]).values
        i0 = int(np.argmin(np.linalg.norm(caps_flat - center_0.ravel(), axis=1)))
        i1 = 1 - i0

        ds = cap.compute_temporal_metrics()
        pers = ds["persistence"].values[0]

        # Each CAP: 2 frames × 0.5 s = 1.0 s total, 1 episode → persistence = 1.0 s.
        npt.assert_allclose(pers[i0], 1.0, atol=1e-10)
        npt.assert_allclose(pers[i1], 1.0, atol=1e-10)

    def test_persistence_attrs_units_with_time_coord(self, fitted_cap):
        ds = fitted_cap.compute_temporal_metrics()
        # Fixture recordings have time coords but no units attr → "time".
        assert ds["persistence"].attrs["units"] == "time"

    def test_persistence_attrs_units_with_units_attr(self, rng):
        ny, nx = 3, 3
        rec = xr.DataArray(
            rng.standard_normal((20, ny, nx)),
            dims=["time", "y", "x"],
            coords={
                "time": xr.DataArray(
                    np.arange(20) * 0.1, dims=["time"], attrs={"units": "s"}
                )
            },
        )
        cap = CAP(n_clusters=2, random_state=0).fit([rec])
        assert cap.compute_temporal_metrics()["persistence"].attrs["units"] == "s"

    def test_persistence_attrs_units_no_time_coord(self, rng):
        ny, nx = 3, 3
        rec = xr.DataArray(
            rng.standard_normal((20, ny, nx)), dims=["time", "y", "x"]
        )
        cap = CAP(n_clusters=2, random_state=0).fit([rec])
        assert cap.compute_temporal_metrics()["persistence"].attrs["units"] == "frames"

    def test_transition_frequency_non_negative(self, fitted_cap):
        ds = fitted_cap.compute_temporal_metrics()
        assert (ds["transition_frequency"].values >= 0).all()

    def test_metrics_unfitted_raises(self):
        cap = CAP(n_clusters=4)
        with pytest.raises(NotFittedError):
            cap.compute_temporal_metrics()

    def test_metrics_correctness(self):
        """Verify metrics against a hand-crafted label sequence."""
        # Two recordings, 2 CAPs.
        # rec0: [0, 0, 1, 1, 1, 0] → tf=[3/6, 3/6], counts=[2,1], persistence=[1.5,3]
        # rec1: [1, 1, 1, 1]       → tf=[0/4, 4/4], counts=[0,1], persistence=[0, 4]
        rng = np.random.default_rng(0)
        ny, nx = 3, 3

        center_0 = np.ones((ny, nx))
        center_1 = -np.ones((ny, nx))
        seq0 = [0, 0, 1, 1, 1, 0]
        seq1 = [1, 1, 1, 1]
        noise_scale = 0.01

        def _make_rec(seq):
            frames = []
            for s in seq:
                center = center_0 if s == 0 else center_1
                frames.append(center + rng.standard_normal((ny, nx)) * noise_scale)
            return xr.DataArray(
                np.stack(frames),
                dims=["time", "y", "x"],
                coords={"time": np.arange(len(seq), dtype=float)},
            )

        rec0, rec1 = _make_rec(seq0), _make_rec(seq1)
        cap = CAP(n_clusters=2, random_state=0, metric="euclidean")
        cap.fit([rec0, rec1])

        # Re-map labels so CAP 0 = center_0, CAP 1 = center_1.
        # (cluster ordering may differ from input ordering)
        caps_flat = cap.caps_.stack(space=["y", "x"]).values  # (2, 9)
        center_0_flat = center_0.ravel()
        dist_to_0 = np.linalg.norm(caps_flat - center_0_flat, axis=1)
        label_for_center0 = int(np.argmin(dist_to_0))
        label_for_center1 = 1 - label_for_center0

        lbl0 = cap.labels_[0].values
        lbl1 = cap.labels_[1].values

        # Verify that each frame is assigned to the correct CAP.
        expected_raw0 = [label_for_center0 if s == 0 else label_for_center1 for s in seq0]
        expected_raw1 = [label_for_center1] * 4
        npt.assert_array_equal(lbl0, expected_raw0)
        npt.assert_array_equal(lbl1, expected_raw1)

        ds = cap.compute_temporal_metrics()
        tf = ds["temporal_fraction"].values  # (recording, cap)
        cnt = ds["counts"].values
        pers = ds["persistence"].values

        i0, i1 = label_for_center0, label_for_center1

        # Temporal fraction
        npt.assert_allclose(tf[0, i0], 3 / 6, atol=1e-10)
        npt.assert_allclose(tf[0, i1], 3 / 6, atol=1e-10)
        npt.assert_allclose(tf[1, i0], 0.0, atol=1e-10)
        npt.assert_allclose(tf[1, i1], 1.0, atol=1e-10)

        # Counts
        assert cnt[0, i0] == 2
        assert cnt[0, i1] == 1
        assert cnt[1, i0] == 0
        assert cnt[1, i1] == 1

        # Persistence (in frames)
        npt.assert_allclose(pers[0, i0], 1.5, atol=1e-10)
        npt.assert_allclose(pers[0, i1], 3.0, atol=1e-10)
        npt.assert_allclose(pers[1, i0], 0.0, atol=1e-10)
        npt.assert_allclose(pers[1, i1], 4.0, atol=1e-10)
