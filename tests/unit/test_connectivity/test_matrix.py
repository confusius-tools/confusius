"""Tests for confusius.connectivity.ConnectivityMatrix.

Helper function tests (symmetric_matrix_to_vector, vector_to_symmetric_matrix,
precision_to_partial_correlation) verify the public math utilities. ConnectivityMatrix
tests cover correctness, vectorization, inverse transforms, and the sklearn interface.

Adapted from nilearn's test_connectivity_matrices.py (BSD-3-Clause License;
see NOTICE for details).
"""

from math import sqrt

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

from confusius.connectivity import (
    ConnectivityMatrix,
    precision_to_partial_correlation,
    symmetric_matrix_to_vector,
    vector_to_symmetric_matrix,
)

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

N_FEATURES = 20
N_SUBJECTS = 5

CONNECTIVITY_KINDS = (
    "covariance",
    "correlation",
    "partial correlation",
    "tangent",
    "precision",
)


def _make_signals(
    n_subjects: int = N_SUBJECTS,
    n_features: int = N_FEATURES,
    seed: int = 0,
) -> list[xr.DataArray]:
    """Generate zero-mean random (time, regions) DataArrays for each subject."""
    rng = np.random.default_rng(seed)
    regions = [f"r{i}" for i in range(n_features)]
    signals = []
    for k in range(n_subjects):
        n_samples = 100 + k
        values = rng.standard_normal((n_samples, n_features))
        values -= values.mean(axis=0)
        da = xr.DataArray(
            values,
            dims=["time", "regions"],
            coords={
                "time": np.arange(n_samples) * 0.1,
                "regions": regions,
            },
        )
        signals.append(da)
    return signals


def _random_spd(
    p: int, eig_min: float = 1.0, cond: float = 10.0, seed: int = 0
) -> np.ndarray:
    """Generate a random symmetric positive definite matrix."""
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((p, p))
    unitary, _ = linalg.qr(mat)
    diag = np.diag(rng.random(p) * (cond * eig_min - eig_min) + eig_min)
    diag[0, 0] = eig_min
    diag[-1, -1] = cond * eig_min
    return unitary.dot(diag).dot(unitary.T)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def signals() -> list[xr.DataArray]:
    """Zero-mean (time, regions) DataArrays for N_SUBJECTS subjects."""
    return _make_signals()


# ---------------------------------------------------------------------------
# Public math utility tests
# ---------------------------------------------------------------------------


class TestSymMatrixToVec:
    """Tests for symmetric_matrix_to_vector and vector_to_symmetric_matrix round-trip."""

    def test_known_values(self):
        """Vectorization of an all-ones matrix matches analytical formula."""
        sym = np.ones((3, 3))
        sqrt2 = 1.0 / sqrt(2.0)
        expected = np.array([sqrt2, 1.0, sqrt2, 1.0, 1.0, sqrt2])
        assert_array_almost_equal(symmetric_matrix_to_vector(sym), expected)

    def test_discard_diagonal(self):
        """Off-diagonal vectorization of all-ones matrix gives all ones."""
        sym = np.ones((3, 3))
        assert_array_almost_equal(
            symmetric_matrix_to_vector(sym, discard_diagonal=True), np.ones(3)
        )

    def test_round_trip_with_diagonal(self, rng):
        """vector_to_symmetric_matrix inverts symmetric_matrix_to_vector (diagonal included)."""
        n = 5
        p = n * (n + 1) // 2
        vec = rng.random(p)
        sym = vector_to_symmetric_matrix(vec)
        assert_array_almost_equal(symmetric_matrix_to_vector(sym), vec)

    def test_round_trip_without_diagonal(self, rng):
        """vector_to_symmetric_matrix inverts symmetric_matrix_to_vector (diagonal separate)."""
        n = 5
        p = n * (n + 1) // 2
        vec = rng.random(p)
        diagonal = rng.random(n + 1)
        sym = vector_to_symmetric_matrix(vec, diagonal=diagonal)
        assert_array_almost_equal(symmetric_matrix_to_vector(sym, discard_diagonal=True), vec)

    def test_round_trip_batch(self, rng):
        """Batch (multiple matrices) round-trip."""
        n = 5
        p = n * (n + 1) // 2
        vec = rng.random(p)
        vecs = np.asarray([vec, 2.0 * vec, 0.5 * vec])
        syms = vector_to_symmetric_matrix(vecs)
        assert_array_almost_equal(symmetric_matrix_to_vector(syms), vecs)

    def test_unsuitable_shape_raises(self):
        """vector_to_symmetric_matrix raises ValueError for a vector with bad length."""
        with pytest.raises(ValueError, match="Vector of unsuitable shape"):
            vector_to_symmetric_matrix(np.ones(31))

    def test_incompatible_diagonal_raises(self):
        """vector_to_symmetric_matrix raises ValueError when diagonal shape is incompatible."""
        vec = np.ones(3)
        diagonal = np.zeros(4)
        with pytest.raises(ValueError, match="incompatible with vector"):
            vector_to_symmetric_matrix(vec, diagonal)


class TestPrecToPartial:
    """Tests for precision_to_partial_correlation."""

    def test_known_values(self):
        """Partial correlation matches analytic result for a small precision matrix."""
        precision = np.array([[2.0, -1.0, 1.0], [-1.0, 2.0, -1.0], [1.0, -1.0, 1.0]])
        expected = np.array(
            [
                [1.0, 0.5, -sqrt(2.0) / 2.0],
                [0.5, 1.0, sqrt(2.0) / 2.0],
                [-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 1.0],
            ]
        )
        assert_array_almost_equal(precision_to_partial_correlation(precision), expected)

    def test_diagonal_is_one(self):
        """Partial correlation diagonal is exactly 1."""
        n = 6
        spd = _random_spd(n, seed=1)
        precision = linalg.inv(spd)
        partial = precision_to_partial_correlation(precision)
        assert_array_almost_equal(np.diag(partial), np.ones(n))


# ---------------------------------------------------------------------------
# ConnectivityMatrix: validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for ConnectivityMatrix input validation."""

    def test_non_iterable_input_raises(self):
        """fit raises TypeError for non-iterable input."""
        measure = ConnectivityMatrix()
        with pytest.raises(TypeError, match="DataArray"):
            measure.fit(1.0)

    def test_numpy_array_input_raises(self):
        """fit raises TypeError for numpy array input (DataArrays required)."""
        measure = ConnectivityMatrix()
        with pytest.raises(TypeError, match="DataArray"):
            measure.fit([np.ones((100, N_FEATURES))])

    def test_no_time_dim_raises(self):
        """fit raises ValueError when a subject is missing the time dimension."""
        measure = ConnectivityMatrix()
        da = xr.DataArray(np.ones((100, N_FEATURES)), dims=["samples", "regions"])
        with pytest.raises(ValueError, match="time"):
            measure.fit([da])

    def test_extra_dims_raises(self):
        """fit raises ValueError when a subject has more than two dimensions."""
        measure = ConnectivityMatrix()
        da = xr.DataArray(np.ones((100, 5, N_FEATURES)), dims=["time", "z", "regions"])
        with pytest.raises(ValueError, match="exactly one non-time dimension"):
            measure.fit([da])

    def test_inconsistent_features_dim_name_raises(self):
        """fit raises ValueError when subjects have different features dim names."""
        measure = ConnectivityMatrix()
        da1 = xr.DataArray(np.ones((100, N_FEATURES)), dims=["time", "regions"])
        da2 = xr.DataArray(np.ones((100, N_FEATURES)), dims=["time", "voxels"])
        with pytest.raises(ValueError, match="features dimension name"):
            measure.fit([da1, da2])

    def test_inconsistent_features_size_raises(self):
        """fit raises ValueError when subjects have different numbers of features."""
        measure = ConnectivityMatrix()
        da1 = xr.DataArray(np.ones((100, N_FEATURES)), dims=["time", "regions"])
        da2 = xr.DataArray(np.ones((100, N_FEATURES + 1)), dims=["time", "regions"])
        with pytest.raises(ValueError, match="same number of features"):
            measure.fit([da1, da2])

    def test_invalid_kind_raises(self, signals):
        """fit raises ValueError for an unknown kind."""
        measure = ConnectivityMatrix(kind="unknown")
        with pytest.raises(ValueError, match="kind must be one of"):
            measure.fit(signals)

    def test_tangent_single_subject_raises(self):
        """fit_transform raises ValueError for tangent kind with one subject."""
        measure = ConnectivityMatrix(kind="tangent")
        da = xr.DataArray(np.ones((100, N_FEATURES)), dims=["time", "regions"])
        with pytest.raises(ValueError, match="group of subjects"):
            measure.fit_transform([da])

    def test_transform_features_dim_mismatch_raises(self, signals):
        """transform raises ValueError when features dim name differs from fit."""
        measure = ConnectivityMatrix().fit(signals)
        new_signals = [
            xr.DataArray(np.ones((100, N_FEATURES)), dims=["time", "voxels"])
        ]
        with pytest.raises(ValueError, match="features dimension"):
            measure.transform(new_signals)

    def test_transform_features_size_mismatch_raises(self, signals):
        """transform raises ValueError when feature count differs from fit."""
        measure = ConnectivityMatrix().fit(signals)
        new_signals = [
            xr.DataArray(np.ones((100, N_FEATURES + 5)), dims=["time", "regions"])
        ]
        with pytest.raises(ValueError, match="features"):
            measure.transform(new_signals)


# ---------------------------------------------------------------------------
# ConnectivityMatrix: single DataArray input
# ---------------------------------------------------------------------------


class TestSingleDataArrayInput:
    """Tests for convenience of passing a single DataArray (no list needed)."""

    def test_single_da_equals_list_input(self):
        """A single DataArray produces the same result as a one-element list."""
        da = xr.DataArray(
            np.random.default_rng(0).standard_normal((100, N_FEATURES)),
            dims=["time", "regions"],
        )
        measure = ConnectivityMatrix(kind="covariance")
        out_single = measure.fit_transform(da)
        out_list = ConnectivityMatrix(kind="covariance").fit_transform([da])
        assert_array_almost_equal(out_single, out_list)

    def test_single_da_fit_then_transform_consistent(self):
        """fit then transform a single DataArray matches fit_transform."""
        da = xr.DataArray(
            np.random.default_rng(1).standard_normal((100, N_FEATURES)),
            dims=["time", "regions"],
        )
        measure = ConnectivityMatrix(kind="covariance")
        out_fit_transform = ConnectivityMatrix(kind="covariance").fit_transform([da])
        measure.fit(da)
        out_transform = measure.transform(da)
        assert_array_almost_equal(out_transform, out_fit_transform)


# ---------------------------------------------------------------------------
# ConnectivityMatrix: output correctness
# ---------------------------------------------------------------------------


class TestCorrectness:
    """Tests that ConnectivityMatrix outputs have the expected shape and properties."""

    def test_correlation_diagonal_is_one(self, signals):
        """Correlation matrices have diagonal elements of exactly 1."""
        measure = ConnectivityMatrix(kind="correlation")
        conns = measure.fit_transform(signals)
        for conn in conns:
            assert_array_almost_equal(np.diag(conn), np.ones(N_FEATURES))

    def test_correlation_is_symmetric_spd(self, signals):
        """Each correlation matrix is symmetric and positive definite."""
        measure = ConnectivityMatrix(kind="correlation")
        conns = measure.fit_transform(signals)
        for conn in conns:
            assert_array_almost_equal(conn, conn.T)
            eigenvalues = np.linalg.eigvalsh(conn)
            assert np.all(eigenvalues > 0), (
                "correlation matrix is not positive definite"
            )

    def test_precision_times_cov_is_identity(self, signals):
        """Precision matrix multiplied by the empirical covariance is close to I."""
        measure_prec = ConnectivityMatrix(
            kind="precision", cov_estimator=EmpiricalCovariance()
        )
        prec_conns = measure_prec.fit_transform(signals)

        measure_cov = ConnectivityMatrix(
            kind="covariance", cov_estimator=EmpiricalCovariance()
        )
        cov_conns = measure_cov.fit_transform(signals)

        for prec, cov in zip(prec_conns, cov_conns):
            assert_array_almost_equal(prec.dot(cov), np.eye(N_FEATURES))

    def test_partial_correlation_diagonal_is_one(self, signals):
        """Partial correlation matrices have diagonal elements of exactly 1."""
        measure = ConnectivityMatrix(kind="partial correlation")
        conns = measure.fit_transform(signals)
        for conn in conns:
            assert_array_almost_equal(np.diag(conn), np.ones(N_FEATURES))

    def test_tangent_output_is_symmetric(self, signals):
        """Tangent matrices are symmetric."""
        measure = ConnectivityMatrix(kind="tangent")
        conns = measure.fit_transform(signals)
        for conn in conns:
            assert_array_almost_equal(conn, conn.T)

    def test_tangent_whitening_properties(self, signals):
        """whitening_ is positive definite and inverts the square root of mean_."""
        measure = ConnectivityMatrix(kind="tangent")
        measure.fit_transform(signals)

        assert measure.whitening_ is not None
        eigenvalues = np.linalg.eigvalsh(measure.whitening_)
        assert np.all(eigenvalues > 0)

        gmean_sqrt = np.linalg.cholesky(measure.mean_)
        assert_array_almost_equal(
            measure.whitening_.dot(gmean_sqrt.dot(gmean_sqrt.T)).dot(
                measure.whitening_
            ),
            np.eye(N_FEATURES),
            decimal=5,
        )

    def test_whitening_is_none_for_non_tangent(self, signals):
        """whitening_ is None for all non-tangent kinds."""
        for kind in CONNECTIVITY_KINDS:
            if kind == "tangent":
                continue
            measure = ConnectivityMatrix(kind=kind)
            measure.fit_transform(signals)
            assert measure.whitening_ is None

    def test_features_dim_tracked(self, signals):
        """features_dim_in_ records the features dimension name."""
        measure = ConnectivityMatrix().fit(signals)
        assert measure.features_dim_in_ == "regions"

    def test_dim_name_preserved_across_different_feature_dims(self):
        """features_dim_in_ matches the actual dim name from the input DataArrays."""
        das = [
            xr.DataArray(
                np.random.default_rng(k).standard_normal((100, 5)),
                dims=["time", "voxels"],
            )
            for k in range(3)
        ]
        measure = ConnectivityMatrix().fit(das)
        assert measure.features_dim_in_ == "voxels"


class TestMean:
    """Tests for the mean_ attribute."""

    @pytest.mark.parametrize(
        "kind",
        ["covariance", "correlation", "partial correlation", "precision"],
    )
    def test_mean_equals_mean_of_transform(self, signals, kind):
        """For non-tangent kinds, mean_ == np.mean(transform(signals), axis=0)."""
        measure = ConnectivityMatrix(kind=kind)
        measure.fit_transform(signals)
        per_subject = measure.transform(signals)
        assert_array_almost_equal(measure.mean_, np.mean(per_subject, axis=0))

    def test_mean_not_modified_by_transform(self, signals):
        """transform does not overwrite mean_."""
        measure = ConnectivityMatrix(kind="covariance")
        measure.fit(signals[:1])
        mean_before = measure.mean_.copy()
        measure.transform(signals[1:])
        assert_array_equal(mean_before, measure.mean_)


# ---------------------------------------------------------------------------
# ConnectivityMatrix: vectorization
# ---------------------------------------------------------------------------


class TestVectorization:
    """Tests for the vectorize and discard_diagonal options."""

    @pytest.mark.parametrize("kind", CONNECTIVITY_KINDS)
    def test_vectorized_equals_manual(self, signals, kind):
        """Vectorized output matches symmetric_matrix_to_vector applied to matrices."""
        measure_mat = ConnectivityMatrix(kind=kind)
        matrices = measure_mat.fit_transform(signals)

        measure_vec = ConnectivityMatrix(kind=kind, vectorize=True)
        vecs = measure_vec.fit_transform(signals)

        assert_array_almost_equal(vecs, symmetric_matrix_to_vector(matrices))


# ---------------------------------------------------------------------------
# ConnectivityMatrix: inverse transform
# ---------------------------------------------------------------------------


class TestInverseTransform:
    """Tests for ConnectivityMatrix.inverse_transform."""

    @pytest.mark.parametrize(
        "kind",
        ["covariance", "correlation", "partial correlation", "precision"],
    )
    def test_non_vectorized_roundtrip(self, signals, kind):
        """Without vectorization, inverse_transform returns the input unchanged."""
        measure = ConnectivityMatrix(kind=kind)
        conns = measure.fit_transform(signals)
        assert_array_almost_equal(measure.inverse_transform(conns), conns)

    @pytest.mark.parametrize(
        "kind",
        ["covariance", "correlation", "partial correlation", "precision"],
    )
    def test_vectorized_roundtrip(self, signals, kind):
        """With vectorization, inverse_transform reconstructs the matrices."""
        measure = ConnectivityMatrix(kind=kind)
        conns = measure.fit_transform(signals)

        measure_vec = ConnectivityMatrix(kind=kind, vectorize=True)
        vecs = measure_vec.fit_transform(signals)

        assert_array_almost_equal(measure_vec.inverse_transform(vecs), conns)

    @pytest.mark.parametrize("kind", ["correlation", "partial correlation"])
    def test_discard_diagonal_roundtrip_corr_kinds(self, signals, kind):
        """Diagonal-discarded correlation/partial correlation can be reconstructed."""
        conns = ConnectivityMatrix(kind=kind).fit_transform(signals)

        measure = ConnectivityMatrix(kind=kind, vectorize=True, discard_diagonal=True)
        vecs = measure.fit_transform(signals)
        assert_array_almost_equal(measure.inverse_transform(vecs), conns)

    @pytest.mark.parametrize("kind", ["covariance", "precision"])
    def test_discard_diagonal_no_diagonal_raises(self, signals, kind):
        """inverse_transform raises for cov/precision when diagonal was discarded."""
        measure = ConnectivityMatrix(kind=kind, vectorize=True, discard_diagonal=True)
        vecs = measure.fit_transform(signals)
        with pytest.raises(ValueError, match="cannot reconstruct"):
            measure.inverse_transform(vecs)

    def test_tangent_inverse_transform_recovers_covariances(self, signals):
        """Tangent inverse_transform reconstructs the original covariance matrices."""
        tangent = ConnectivityMatrix(kind="tangent")
        displacements = tangent.fit_transform(signals)

        covariances = ConnectivityMatrix(kind="covariance").fit_transform(signals)
        assert_array_almost_equal(tangent.inverse_transform(displacements), covariances)

    def test_tangent_vectorized_inverse_transform(self, signals):
        """Vectorized tangent inverse_transform also reconstructs covariances."""
        covariances = ConnectivityMatrix(kind="covariance").fit_transform(signals)

        tangent = ConnectivityMatrix(kind="tangent", vectorize=True)
        vecs = tangent.fit_transform(signals)
        assert_array_almost_equal(tangent.inverse_transform(vecs), covariances)


# ---------------------------------------------------------------------------
# ConnectivityMatrix: sklearn interface
# ---------------------------------------------------------------------------


class TestSklearnInterface:
    """Tests for sklearn BaseEstimator compatibility."""

    def test_get_params(self):
        """get_params returns all constructor parameters."""
        measure = ConnectivityMatrix(kind="correlation", vectorize=True)
        params = measure.get_params()
        assert params["kind"] == "correlation"
        assert params["vectorize"] is True
        assert "cov_estimator" in params
        assert "discard_diagonal" in params

    def test_set_params(self):
        """set_params correctly updates parameters."""
        measure = ConnectivityMatrix()
        measure.set_params(kind="precision", vectorize=True)
        assert measure.kind == "precision"
        assert measure.vectorize is True

    def test_not_fitted_raises(self, signals):
        """transform raises NotFittedError before fit."""
        from sklearn.exceptions import NotFittedError

        measure = ConnectivityMatrix()
        with pytest.raises(NotFittedError):
            measure.transform(signals)

    def test_is_fitted_after_fit(self, signals):
        """__sklearn_is_fitted__ returns True after fit."""
        from sklearn.utils.validation import check_is_fitted

        measure = ConnectivityMatrix().fit(signals)
        check_is_fitted(measure)  # Should not raise.

    def test_fit_returns_self(self, signals):
        """fit returns the estimator itself for method chaining."""
        measure = ConnectivityMatrix()
        result = measure.fit(signals)
        assert result is measure

    def test_default_cov_estimator_is_ledoit_wolf(self, signals):
        """cov_estimator_ defaults to LedoitWolf when cov_estimator=None."""
        measure = ConnectivityMatrix().fit(signals)
        assert isinstance(measure.cov_estimator_, LedoitWolf)

    def test_custom_cov_estimator(self, signals):
        """A custom cov_estimator is cloned and stored in cov_estimator_."""
        measure = ConnectivityMatrix(cov_estimator=EmpiricalCovariance()).fit(signals)
        assert isinstance(measure.cov_estimator_, EmpiricalCovariance)

    def test_n_features_in(self, signals):
        """n_features_in_ is set correctly after fit."""
        measure = ConnectivityMatrix().fit(signals)
        assert measure.n_features_in_ == N_FEATURES
