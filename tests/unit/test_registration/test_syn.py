"""Public-API checks for symmetric diffeomorphic registration."""

from threading import Event
from types import SimpleNamespace

import itk
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from confusius.io import load, save
from confusius.registration import register_volume, resample_like
from confusius.registration.progress import resample_intermediate
from confusius.validation import validate_displacement_field
from confusius.xarray import create_voxeldata


@pytest.fixture
def syn_volume():
    """An asymmetric single-slice image in oblique, anisotropic world geometry."""
    j, i = np.mgrid[:32, :32]
    values = np.exp(-((j - 13) ** 2 + (i - 12) ** 2) / 30) + 0.7 * np.exp(
        -((j - 20) ** 2 + (i - 21) ** 2) / 12
    )
    return create_voxeldata(
        values[None].astype(np.float32),
        dims=["k", "j", "i"],
        spacing=[0.4, 0.08, 0.1],
        origin=[2, -3, 4],
        direction=Rotation.from_euler("xyz", [0.12, -0.23, 0.34]).as_matrix(),
    )


@pytest.mark.parametrize("metric", ["correlation", "mattes_mi", "ants_cc"])
def test_syn_identity_and_reusable_field(syn_volume, metric):
    """Identity registration retains signal and exports a reusable fixed-grid field."""
    result, field, diagnostics = register_volume(
        syn_volume,
        syn_volume,
        transform_type="syn",
        metric=metric,
        number_of_iterations=2,
        itk_threads=1,
        # Mattes estimates a nonzero finite-sample gradient even for identical images.
        learning_rate=1e-6 if metric == "mattes_mi" else 0.1,
    )
    validate_displacement_field(field)
    assert_allclose(field.values, 0, atol=1e-5)
    assert_allclose(result.values, syn_volume.values, atol=1e-5)
    reapplied = resample_like(syn_volume, syn_volume, field, itk_threads=1)
    assert_allclose(reapplied.values, result.values, atol=1e-6)
    assert diagnostics.metric == metric
    assert diagnostics.n_iterations == 2
    assert len(diagnostics.metric_values) == 2
    assert np.isfinite(diagnostics.metric_values).all()
    assert diagnostics.final_metric_value == diagnostics.metric_values[-1]
    assert diagnostics.status == "completed"
    for coordinate in ("z", "y", "x"):
        assert_allclose(field.coords[coordinate], syn_volume.coords[coordinate])
        assert field.coords[coordinate].attrs["units"] == "mm"


def test_syn_recovers_smooth_known_warp(syn_volume):
    """Recover a known pull displacement, not merely a changed output image."""
    k, j, i = np.indices(syn_volume.shape, dtype=float)
    offset = 0.7 * np.sin(np.pi * j / 31)
    moving = syn_volume.copy(
        data=map_coordinates(
            syn_volume.values,
            [k, j, i - offset],
            order=3,
            mode="nearest",
        )
    )
    result, field, _ = register_volume(
        moving,
        syn_volume,
        transform_type="syn",
        metric="ants_cc",
        initialization=None,
        total_field_variance=0.2,
        number_of_iterations=40,
        itk_threads=1,
    )
    interior = (syn_volume.values > 0.1) & (j > 6) & (j < 25) & (i > 6) & (i < 25)
    before = np.mean((moving.values[interior] - syn_volume.values[interior]) ** 2)
    after = np.mean((result.values[interior] - syn_volume.values[interior]) ** 2)
    assert after < before * 0.5
    direction = syn_volume.fusi.direction[:, 2]
    expected = direction[:, None, None, None] * offset * 0.1
    error = np.mean((field.values[:, interior] - expected[:, interior]) ** 2)
    assert error < np.mean(expected[:, interior] ** 2) * 0.5
    native = np.einsum("ij,j...->i...", syn_volume.fusi.direction.T, field.values)[:, 0]
    du_j_dj, du_j_di = np.gradient(native[1], 0.08, 0.1)
    du_i_dj, du_i_di = np.gradient(native[2], 0.08, 0.1)
    jacobian = (1 + du_j_dj) * (1 + du_i_di) - du_j_di * du_i_dj
    assert np.min(jacobian[interior[0]]) > 0
    reapplied = resample_like(moving, syn_volume, field, itk_threads=1)
    assert_allclose(reapplied.values, result.values, atol=2e-5)


@pytest.mark.parametrize("abort", [False, True])
def test_syn_affine_initialization_is_included_in_field(syn_volume, abort):
    """Serialized fields include pre-affine scaling and translation in pull order."""
    affine = np.eye(4)
    affine[:3, :3] = np.diag([1.01, 0.99, 1.02])
    affine[:3, 3] = [0.01, -0.02, 0.03]
    event = Event()
    if abort:
        event.set()
    _, field, diagnostics = register_volume(
        syn_volume,
        syn_volume,
        transform_type="syn",
        initialization=affine,
        number_of_iterations=1,
        learning_rate=1e-9,
        abort_event=event,
        resample=False,
        itk_threads=1,
    )
    world = np.stack([syn_volume.coords[c].values for c in ("z", "y", "x")])
    expected = (
        np.einsum("ij,j...->i...", affine[:3, :3], world)
        + affine[:3, 3, None, None, None]
        - world
    )
    assert_allclose(field.values, expected, atol=1e-8)
    assert diagnostics.status == ("aborted" if abort else "completed")
    assert diagnostics.n_iterations == (0 if abort else 1)


def test_syn_live_progress_and_cancellation(syn_volume):
    """Live half-transforms produce an overlay and cancellation stops native SyN."""
    moving = syn_volume.copy(data=np.roll(syn_volume.values, 1, axis=2))
    event = Event()
    observed = []
    closed = []

    def factory(method, fixed, moving, **kwargs):
        """Collect live frames through the same hook used by GUI reporters."""

        def update():
            """Read the current pull transform and request cancellation."""
            preview = resample_intermediate(
                method,
                moving,
                fixed,
                fill_value=0,
                itk_threads=1,
            )
            observed.append(itk.array_from_image(preview))
            if len(observed) == 2:
                event.set()

        return SimpleNamespace(update=update, close=lambda: closed.append(True))

    _, field, diagnostics = register_volume(
        moving,
        syn_volume,
        transform_type="syn",
        metric="ants_cc",
        number_of_iterations=20,
        show_progress=True,
        progress_plotter=factory,
        abort_event=event,
        itk_threads=1,
    )
    assert diagnostics.status == "aborted"
    assert diagnostics.n_iterations == len(observed) == 2
    assert closed == [True]
    assert np.isfinite(observed).all()
    assert not np.allclose(observed[-1], observed[0])
    assert np.max(np.abs(field.values)) > 0


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"use_multi_resolution": True}, "single-resolution"),
        ({"optimizer_weights": [1, 1, 1]}, "optimizer_weights"),
        ({"update_field_variance": -1}, "update_field_variance"),
        ({"update_field_variance": np.nan}, "update_field_variance"),
        ({"total_field_variance": np.inf}, "total_field_variance"),
        ({"total_field_variance": -1}, "total_field_variance"),
        ({"metric": "ants_cc", "correlation_radius": 0}, "correlation_radius"),
        ({"metric": "ants_cc", "correlation_radius": 1.5}, "correlation_radius"),
    ],
)
def test_syn_rejects_invalid_or_unsupported_settings(syn_volume, kwargs, message):
    """Unsupported native options must fail explicitly rather than being ignored."""
    with pytest.raises(ValueError, match=message):
        register_volume(syn_volume, syn_volume, transform_type="syn", **kwargs)


@pytest.mark.parametrize("transform_type", ["rigid", "affine", "bspline"])
def test_neighborhood_metric_with_existing_transforms(syn_volume, transform_type):
    """The added metric also retains identity with existing transform models."""
    result, _, diagnostics = register_volume(
        syn_volume,
        syn_volume,
        transform_type=transform_type,
        metric="ants_cc",
        mesh_size=(1, 3, 3),
        number_of_iterations=1,
        itk_threads=1,
    )
    assert_allclose(result.values, syn_volume.values, atol=1e-6)
    assert_allclose(diagnostics.metric_values, -1, atol=1e-6)


def test_syn_field_round_trip_and_masked_sampling(syn_volume, tmp_path):
    """Saved fields retain oblique geometry and reproduce masked, sampled registration."""
    source = syn_volume.isel(j=slice(1, 29, 2), i=slice(0, 32, 2)).astype(np.float64)
    mask = source > 0.01
    result, field, diagnostics = register_volume(
        source,
        source,
        transform_type="syn",
        metric="correlation",
        fixed_mask=mask,
        moving_mask=mask,
        metric_sampling_percentage=0.8,
        metric_sampling_seed=42,
        number_of_iterations=2,
        itk_threads=1,
    )
    path = tmp_path / "syn_field.zarr"
    save(field, path, zarr_format=2)
    restored = load(path)
    validate_displacement_field(restored)
    assert_allclose(restored.values, field.values)
    reapplied = resample_like(source, source, restored, itk_threads=1)
    assert_allclose(reapplied.values, result.values, atol=1e-10)
    assert np.isfinite(diagnostics.metric_values).all()


def test_syn_matplotlib_reports_native_metric(syn_volume):
    """The standard progress reporter reads SyN's metric, not its unused optimizer."""
    import matplotlib.pyplot as plt

    from confusius.registration import MatplotlibRegistrationProgressPlotter

    figures = []

    def factory(*args, **kwargs):
        """Retain the public figure to inspect the metric curve after closure."""
        plotter = MatplotlibRegistrationProgressPlotter(*args, **kwargs)
        figures.append(plt.gcf())
        return plotter

    _, _, diagnostics = register_volume(
        syn_volume,
        syn_volume,
        transform_type="syn",
        metric="ants_cc",
        number_of_iterations=2,
        itk_threads=1,
        show_progress=True,
        plot_composite=False,
        progress_plotter=factory,
    )
    assert_allclose(figures[0].axes[0].lines[0].get_ydata(), diagnostics.metric_values)
    assert_allclose(diagnostics.metric_values, -1, atol=1e-6)


def test_syn_accessor_and_volumetric_data(syn_volume):
    """The accessor forwards SyN controls and supports genuinely volumetric data."""
    volume = create_voxeldata(
        np.repeat(syn_volume.values, 8, axis=0),
        dims=["k", "j", "i"],
        spacing=[0.4, 0.08, 0.1],
    )
    result, field, diagnostics = volume.fusi.register.to_volume(
        volume,
        transform="syn",
        metric="ants_cc",
        correlation_radius=2,
        update_field_variance=0,
        total_field_variance=0,
        number_of_iterations=1,
        resample=True,
        itk_threads=1,
    )
    validate_displacement_field(field)
    assert_allclose(field.values, 0, atol=1e-12)
    assert_allclose(result.values, volume.values, atol=1e-6)
    assert diagnostics.n_iterations == 1
