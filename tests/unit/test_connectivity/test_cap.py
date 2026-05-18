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
def clustered_recording():
    """Recording with 2 well-separated clusters alternating every volume.

    Even volumes cluster around +1, odd volumes around -1.
    Noise is small (0.01) so cluster assignment is deterministic.
    """
    rng = np.random.default_rng(0)
    ny, nx = 3, 3
    center_0 = np.ones((ny, nx))
    center_1 = -np.ones((ny, nx))
    noise = 0.01
    seq = [0, 1, 0, 1, 0, 1, 0, 1]
    frames = [
        (center_0 if s == 0 else center_1) + rng.standard_normal((ny, nx)) * noise
        for s in seq
    ]
    return xr.DataArray(np.stack(frames), dims=["time", "y", "x"])


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

    def test_mean_update_rule_correctness(self, clustered_recording):
        """update_rule='mean' correctly partitions well-separated clusters."""
        cap = CAP(n_clusters=2, metric="euclidean", update_rule="mean", random_state=0)
        cap.fit([clustered_recording])
        labels = cap.labels_[0].values
        # Even volumes → one CAP, odd volumes → the other.
        assert labels[0] == labels[2] == labels[4] == labels[6]
        assert labels[1] == labels[3] == labels[5] == labels[7]
        assert labels[0] != labels[1]

    def test_cosine_metric_correctness(self, clustered_recording):
        """metric='cosine' correctly partitions volumes by orientation."""
        cap = CAP(n_clusters=2, metric="cosine", random_state=0)
        cap.fit([clustered_recording])
        labels = cap.labels_[0].values
        assert labels[0] == labels[2] == labels[4] == labels[6]
        assert labels[1] == labels[3] == labels[5] == labels[7]
        assert labels[0] != labels[1]

    def test_empty_list_raises(self):
        cap = CAP(n_clusters=3)
        with pytest.raises(ValueError, match="at least one recording"):
            cap.fit([])

    def test_invalid_metric_raises(self, sample_4d_volume):
        cap = CAP(n_clusters=3, metric="manhattan")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="metric"):
            cap.fit([sample_4d_volume])

    def test_invalid_update_rule_raises(self, sample_4d_volume):
        cap = CAP(n_clusters=3, update_rule="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="update_rule"):
            cap.fit([sample_4d_volume])

    def test_nan_input_raises(self, rng):
        data = rng.standard_normal((20, 3, 3))
        data[5, 1, 1] = np.nan
        rec = xr.DataArray(data, dims=["time", "y", "x"])
        with pytest.raises(ValueError, match="NaN"):
            CAP(n_clusters=2).fit([rec])

    def test_mean_update_rule_cosine(self, clustered_recording):
        """update_rule='mean' in the cosine k-means loop correctly partitions volumes."""
        cap = CAP(n_clusters=2, metric="cosine", update_rule="mean", random_state=0)
        cap.fit([clustered_recording])
        labels = cap.labels_[0].values
        assert labels[0] == labels[2] == labels[4] == labels[6]
        assert labels[1] == labels[3] == labels[5] == labels[7]
        assert labels[0] != labels[1]

    def test_empty_cluster_warns(self):
        """More clusters than separable directions triggers an empty-cluster warning.

        With 2 samples and 3 clusters, k-means++ must reuse an existing center
        for the third seed (pot==0), leaving one empty cluster after convergence.
        update_rule='mean' uses argmax assignment, so the duplicate center gets
        zero samples → zero norm → warning. This exercises the coincident-center
        branch in _cosine_kmeans_init, the empty-mask branch in the Lloyd loop,
        and the empty-cluster cleanup path.
        """
        ny, nx = 2, 2
        data = np.stack([np.ones((ny, nx)), -np.ones((ny, nx))])
        rec = xr.DataArray(data, dims=["time", "y", "x"])
        cap = CAP(n_clusters=3, metric="cosine", update_rule="mean", random_state=0)
        with pytest.warns(UserWarning, match="empty cluster"):
            cap.fit([rec])
        assert cap.caps_.sizes["cap"] < 3

    def test_best_restart_selected(self):
        """n_init=2 uses the better of two restarts when they differ.

        With n_clusters=10 and random_state=0, restart 1 has lower cosine
        inertia than restart 0, so the two-restart result differs from the
        single-restart result. This test would fail if the best-restart
        update were dropped. Data is created locally so the RNG state is
        independent of the session-scoped rng fixture.
        """
        local_rng = np.random.default_rng(42)
        ny, nx = 5, 8
        recs = [
            xr.DataArray(local_rng.standard_normal((n_t, ny, nx)), dims=["time", "y", "x"])
            for n_t in (50, 60, 40)
        ]
        cap_1 = CAP(n_clusters=10, n_init=1, random_state=0).fit(recs)
        cap_2 = CAP(n_clusters=10, n_init=2, random_state=0).fit(recs)
        assert not np.array_equal(cap_1.labels_[0].values, cap_2.labels_[0].values)

    def test_n_init_integer(self, sample_4d_volume):
        """Integer n_init runs multiple restarts without error."""
        cap = CAP(n_clusters=4, metric="cosine", n_init=2, random_state=0)
        cap.fit([sample_4d_volume])
        assert cap.caps_.sizes["cap"] == 4

    def test_invalid_n_init_raises(self, sample_4d_volume):
        cap = CAP(n_clusters=2, metric="cosine", n_init=0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_init"):
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

    def test_predict_euclidean(self, clustered_recording):
        """Euclidean predict assigns same-data recordings to the fitted CAPs."""
        cap = CAP(n_clusters=2, metric="euclidean", random_state=0)
        cap.fit([clustered_recording])
        predicted = cap.predict([clustered_recording])
        npt.assert_array_equal(predicted[0].values, cap.labels_[0].values)

    def test_predict_within_range(self, fitted_cap, recordings):
        n_caps = fitted_cap.caps_.sizes["cap"]
        for lbl in fitted_cap.predict(recordings):
            assert lbl.min() >= 0
            assert lbl.max() < n_caps

    def test_predict_cosine_preserves_labels(self, clustered_recording):
        """Cosine predict on the same data reproduces fit labels."""
        cap = CAP(n_clusters=2, metric="cosine", random_state=0)
        cap.fit([clustered_recording])
        predicted = cap.predict([clustered_recording])
        npt.assert_array_equal(predicted[0].values, cap.labels_[0].values)

    def test_predict_nan_raises(self, fitted_cap, rng):
        data = rng.standard_normal((20, 5, 8))
        data[0, 2, 3] = np.nan
        bad_rec = xr.DataArray(data, dims=["time", "y", "x"])
        with pytest.raises(ValueError, match="NaN"):
            fitted_cap.predict([bad_rec])

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

        # [0, 0, 1, 1] at 0.5 s intervals → each volume duration = 0.5 s.
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

        # Each CAP: 2 volumes × 0.5 s = 1.0 s total, 1 episode → persistence = 1.0 s.
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
        assert cap.compute_temporal_metrics()["persistence"].attrs["units"] == "volumes"

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

        # Verify that each volume is assigned to the correct CAP.
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

        # Persistence (in volumes)
        npt.assert_allclose(pers[0, i0], 1.5, atol=1e-10)
        npt.assert_allclose(pers[0, i1], 3.0, atol=1e-10)
        npt.assert_allclose(pers[1, i0], 0.0, atol=1e-10)
        npt.assert_allclose(pers[1, i1], 4.0, atol=1e-10)


# ---------------------------------------------------------------------------
# select_n_clusters()
# ---------------------------------------------------------------------------


class TestSelectNClusters:
    def test_silhouette_returns_value_in_range(self, recordings):
        cap = CAP(random_state=0)
        best_k = cap.select_n_clusters(recordings, range(2, 5), show_progress=False)
        assert best_k in range(2, 5)

    def test_elbow_method(self, recordings):
        cap = CAP(random_state=0)
        best_k = cap.select_n_clusters(
            recordings, range(2, 5), method="elbow", show_progress=False
        )
        assert best_k in range(2, 5)

    def test_davies_bouldin_method(self, recordings):
        cap = CAP(random_state=0)
        best_k = cap.select_n_clusters(
            recordings, range(2, 5), method="davies_bouldin", show_progress=False
        )
        assert best_k in range(2, 5)

    def test_variance_ratio_method(self, recordings):
        cap = CAP(random_state=0)
        best_k = cap.select_n_clusters(
            recordings, range(2, 5), method="variance_ratio", show_progress=False
        )
        assert best_k in range(2, 5)

    def test_cluster_range_too_short_raises(self, recordings):
        with pytest.raises(ValueError, match="at least 2"):
            CAP().select_n_clusters(recordings, [3], show_progress=False)

    def test_cluster_range_below_2_raises(self, recordings):
        with pytest.raises(ValueError, match=">= 2"):
            CAP().select_n_clusters(recordings, [1, 2], show_progress=False)

    def test_invalid_method_raises(self, recordings):
        with pytest.raises(ValueError, match="method"):
            CAP().select_n_clusters(
                recordings, range(2, 4), method="invalid", show_progress=False  # type: ignore[arg-type]
            )

    def test_euclidean_elbow(self, recordings):
        """Euclidean metric + elbow method exercises the KMeans path."""
        cap = CAP(random_state=0, metric="euclidean")
        best_k = cap.select_n_clusters(
            recordings, range(2, 5), method="elbow", show_progress=False
        )
        assert best_k in range(2, 5)

    def test_invalid_metric_raises(self, recordings):
        cap = CAP(metric="manhattan")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="metric"):
            cap.select_n_clusters(recordings, range(2, 4), show_progress=False)

    def test_invalid_update_rule_raises(self, recordings):
        cap = CAP(update_rule="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="update_rule"):
            cap.select_n_clusters(recordings, range(2, 4), show_progress=False)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one recording"):
            CAP().select_n_clusters([], range(2, 4), show_progress=False)

    def test_nan_raises(self, rng):
        data = rng.standard_normal((20, 3, 3))
        data[5, 1, 1] = np.nan
        rec = xr.DataArray(data, dims=["time", "y", "x"])
        with pytest.raises(ValueError, match="NaN"):
            CAP(n_clusters=2).select_n_clusters([rec], range(2, 4), show_progress=False)


# ---------------------------------------------------------------------------
# scores_ and score_samples()
# ---------------------------------------------------------------------------


class TestScoreSamples:
    def test_scores_shape(self, fitted_cap, recordings):
        """scores_ is parallel to labels_ in length and time dimension."""
        assert len(fitted_cap.scores_) == len(recordings)
        for scr, rec in zip(fitted_cap.scores_, recordings):
            assert scr.dims == ("time",)
            assert scr.sizes["time"] == rec.sizes["time"]

    def test_scores_time_coords_preserved(self, fitted_cap, recordings):
        for scr, rec in zip(fitted_cap.scores_, recordings):
            npt.assert_array_equal(
                scr.coords["time"].values, rec.coords["time"].values
            )

    def test_scores_cosine_range(self, clustered_recording):
        """Cosine/correlation scores lie in [-1, 1]."""
        cap = CAP(n_clusters=2, metric="cosine", random_state=0)
        cap.fit([clustered_recording])
        scores = cap.scores_[0].values
        assert scores.min() >= -1.0 - 1e-6
        assert scores.max() <= 1.0 + 1e-6

    def test_scores_well_separated_high(self, clustered_recording):
        """Well-separated clusters yield cosine scores near 1."""
        cap = CAP(n_clusters=2, metric="cosine", random_state=0)
        cap.fit([clustered_recording])
        assert cap.scores_[0].values.min() > 0.9

    def test_scores_euclidean_nonpositive(self, clustered_recording):
        """Euclidean scores (negative L2 distance) are ≤ 0."""
        cap = CAP(n_clusters=2, metric="euclidean", random_state=0)
        cap.fit([clustered_recording])
        assert (cap.scores_[0].values <= 1e-10).all()

    def test_score_samples_matches_scores_(self, fitted_cap, recordings):
        """score_samples on training data matches scores_ from fit."""
        scores = fitted_cap.score_samples(recordings)
        for scr_fit, scr_pred in zip(fitted_cap.scores_, scores):
            npt.assert_allclose(scr_fit.values, scr_pred.values, rtol=1e-5)

    def test_score_samples_euclidean(self, clustered_recording):
        """Euclidean score_samples on same data reproduces scores_."""
        cap = CAP(n_clusters=2, metric="euclidean", random_state=0)
        cap.fit([clustered_recording])
        scores = cap.score_samples([clustered_recording])
        npt.assert_allclose(scores[0].values, cap.scores_[0].values, rtol=1e-5)

    def test_score_samples_unfitted_raises(self, recordings):
        cap = CAP(n_clusters=4)
        with pytest.raises(NotFittedError):
            cap.score_samples(recordings)


# ---------------------------------------------------------------------------
# compute_temporal_metrics(score_threshold=...)
# ---------------------------------------------------------------------------


class TestScoreThreshold:
    def test_none_matches_default(self, fitted_cap):
        """score_threshold=None is identical to calling with no argument."""
        ds_none = fitted_cap.compute_temporal_metrics(score_threshold=None)
        ds_default = fitted_cap.compute_temporal_metrics()
        npt.assert_array_equal(
            ds_none["temporal_fraction"].values, ds_default["temporal_fraction"].values
        )
        npt.assert_array_equal(
            ds_none["transition_frequency"].values,
            ds_default["transition_frequency"].values,
        )

    def test_impossible_threshold_censors_all(self, fitted_cap):
        """A threshold above the maximum possible score excludes all volumes."""
        # Cosine similarity ≤ 1.0, so 2.0 is always above every score.
        ds = fitted_cap.compute_temporal_metrics(score_threshold=2.0)
        npt.assert_array_equal(ds["temporal_fraction"].values, 0.0)
        npt.assert_array_equal(ds["counts"].values, 0)
        npt.assert_array_equal(ds["transition_frequency"].values, 0)

    def test_fixed_denominator_convention(self):
        """Excluded volumes reduce the tf numerator but not the denominator."""
        rng = np.random.default_rng(0)
        ny, nx = 3, 3
        c0, c1 = np.ones((ny, nx)), -np.ones((ny, nx))
        frames = [
            (c0 if s == 0 else c1) + rng.standard_normal((ny, nx)) * 0.01
            for s in [0, 0, 1, 1]
        ]
        rec = xr.DataArray(np.stack(frames), dims=["time", "y", "x"])
        cap = CAP(n_clusters=2, metric="euclidean", random_state=0)
        cap.fit([rec])

        # Force exactly one volume to have an artificially low score.
        scores = cap.scores_[0].values.copy()
        scores[0] = -1e9
        cap.scores_[0] = xr.DataArray(scores, dims=["time"])

        # Threshold that excludes only the injected outlier.
        ds = cap.compute_temporal_metrics(score_threshold=-1.0)
        # 3 of 4 volumes pass: tf sum over all CAPs = 3/4.
        npt.assert_allclose(ds["temporal_fraction"].values[0].sum(), 3 / 4, atol=1e-10)

    def test_censored_volumes_break_episodes(self):
        """A censored volume between two same-CAP runs creates two episodes."""
        rng = np.random.default_rng(0)
        ny, nx = 3, 3
        c0 = np.ones((ny, nx))
        # Four volumes all assigned to CAP 0, one will be censored in the middle.
        frames = [c0 + rng.standard_normal((ny, nx)) * 0.01 for _ in range(4)]
        rec = xr.DataArray(np.stack(frames), dims=["time", "y", "x"])
        cap = CAP(n_clusters=1, metric="euclidean", random_state=0)
        cap.fit([rec])

        # Without threshold: one episode of 4 volumes.
        ds_all = cap.compute_temporal_metrics()
        assert int(ds_all["counts"].values[0, 0]) == 1

        # Censor volume 1 (middle of the run) → should split into 2 episodes.
        scores = cap.scores_[0].values.copy()
        scores[1] = -1e9
        cap.scores_[0] = xr.DataArray(scores, dims=["time"])
        ds = cap.compute_temporal_metrics(score_threshold=-1.0)
        assert int(ds["counts"].values[0, 0]) == 2

    def test_transitions_across_censored_gap_not_counted(self):
        """Transitions that cross a censored volume are not counted."""
        rng = np.random.default_rng(0)
        ny, nx = 3, 3
        c0, c1 = np.ones((ny, nx)), -np.ones((ny, nx))
        # Sequence: CAP0, CAP1, then one volume that will be censored, then CAP0.
        # Real transitions: CAP0→CAP1 (1). The CAP1→CAP0 crossing the censored
        # volume should be dropped, leaving transition_frequency=1.
        frames = [
            c0 + rng.standard_normal((ny, nx)) * 0.01,
            c1 + rng.standard_normal((ny, nx)) * 0.01,
            c0 + rng.standard_normal((ny, nx)) * 0.01,
            c0 + rng.standard_normal((ny, nx)) * 0.01,
        ]
        rec = xr.DataArray(np.stack(frames), dims=["time", "y", "x"])
        cap = CAP(n_clusters=2, metric="euclidean", random_state=0)
        cap.fit([rec])

        # Without threshold: CAP0→CAP1 and CAP1→CAP0 are both counted.
        ds_all = cap.compute_temporal_metrics()
        assert int(ds_all["transition_frequency"].values[0]) == 2

        # Censor volume 2: the CAP1→CAP0 pair is dropped.
        scores = cap.scores_[0].values.copy()
        scores[2] = -1e9
        cap.scores_[0] = xr.DataArray(scores, dims=["time"])
        ds = cap.compute_temporal_metrics(score_threshold=-1.0)
        assert int(ds["transition_frequency"].values[0]) == 1
