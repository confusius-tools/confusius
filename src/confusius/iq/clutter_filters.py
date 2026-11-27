"""Utilities for clutter filtering of beamformed IQ data."""

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from confusius._utils.filtering import design_butterworth_filter
from confusius.iq._utils import (
    ensure_iq_voxeldata,
    resolve_iq_window,
    validate_energy_cutoffs,
    validate_svd_index_cutoffs,
)
from confusius.validation import ensure_voxeldata, validate_mask

if TYPE_CHECKING:
    import dask.array as da
    import xarray as xr


def _validate_block_and_clutter_mask(
    block: npt.NDArray, clutter_mask: npt.NDArray | None = None
) -> tuple[npt.NDArray, npt.NDArray]:
    """Validate IQ block dimensions and prepare signals for clutter filtering.

    Reshapes 4D IQ data to a 2D matrix for clutter filtering and applies spatial masking
    if provided.

    Parameters
    ----------
    block : (time, k, j, i) numpy.ndarray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(k, j, i)` are spatial dimensions.
    clutter_mask : (k, j, i) numpy.ndarray, optional
        Boolean spatial mask. If provided, only voxels where mask is `True` are
        included in the masked signals. Otherwise, all voxels are included.

    Returns
    -------
    signals : (time, k * j * i) numpy.ndarray
        Reshaped IQ data where all spatial dimensions are flattened into the second
        axis.
    masked_signals : (time, mask.sum()) numpy.ndarray
        IQ data for masked voxels only.

    Raises
    ------
    ValueError
        If `block` is not 4D, or if `clutter_mask` shape doesn't match spatial
        dimensions `(k, j, i)` of `block`, or selects no voxels.
    """
    if block.ndim != 4:
        raise ValueError(f"'block' must be 4D, got {block.ndim}D")

    volumes, z, y, x = block.shape
    signals = block.reshape(volumes, -1)

    if clutter_mask is not None:
        if clutter_mask.shape != (z, y, x):
            raise ValueError(
                f"Mask shape {clutter_mask.shape} doesn't match spatial dimensions "
                f"({z}, {y}, {x})"
            )
        # An empty mask must not silently bypass requested clutter suppression.
        if not np.any(clutter_mask):
            raise ValueError("clutter_mask must select at least one voxel.")
        masked_signals = signals[:, clutter_mask.ravel()]
    else:
        masked_signals = signals

    return signals, masked_signals


def _compute_gram_eigendecomposition(
    signals: npt.NDArray,
    ascending: bool = False,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute eigendecomposition of the temporal Gram matrix.

    Computes the eigenvalues and eigenvectors of the Gram matrix formed from beamformed
    IQ signals.

    !!! warning

        The computation of the Gram matrix can lead to numerical instability when using
        single-precision data types. It is recommended to cast `signals` to
        double-precision before calling this function.

    Parameters
    ----------
    signals : (time, space) numpy.ndarray
        Beamformed IQ signals.
    ascending : bool, default: False
        Whether results should be sorted in ascending order of eigenvalues (lowest
        energy first), which is convenient for energy-based filtering. In contrast,
        index-based filtering typically expects descending order (highest energy first).

    Returns
    -------
    eigenvalues : (min(time, space),) numpy.ndarray
        Eigenvalues of the Gram matrix, sorted according to `ascending`.
    eigenvectors : (min(time, space), min(time, space)) numpy.ndarray
        Eigenvectors of the Gram matrix, sorted according to `ascending`.

    Notes
    -----
    The Gram matrix is computed as `signals @ signals.conj().T`. This is more
    efficient than full SVD when only the temporal correlations are needed, avoiding
    computation of the large spatial covariance matrix.

    !!! note "TODO"

        Consider implementing dimension-based optimization. The eigendecomposition
        of the Gram matrix is only more efficient when the number of spatial voxels
        is much larger than the number of time points. When time > voxels, it would
        be more efficient to compute the SVD directly on the data matrix instead of
        forming the temporal Gram matrix. Add logic to choose the optimal approach
        based on the data dimensions.
    """
    from scipy import linalg as sp_linalg

    gram_matrix = signals @ signals.conj().T

    # eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = sp_linalg.eigh(gram_matrix)

    if not ascending:
        # Reverse to descending order for index-based filtering (highest energy first).
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

    return eigenvalues, eigenvectors


def _apply_clutter_filter(
    signals: npt.NDArray, clutter_vectors: npt.NDArray
) -> npt.NDArray:
    """Apply clutter filtering by regressing out clutter vectors.

    Performs orthogonal projection to remove clutter components from IQ signals.
    The clutter vectors (typically eigenvectors corresponding to clutter) are
    projected out of the signal space using the formula:
    `filtered = signals - clutter_vectors @ clutter_vectors.conj().T @ signals`

    Parameters
    ----------
    signals : (time, space) numpy.ndarray
        IQ signals to filter.
    clutter_vectors : (time, components) numpy.ndarray
        Clutter vectors to remove.

    Returns
    -------
    (time, space) numpy.ndarray
        Filtered signals with clutter components removed.
    """
    if clutter_vectors.size > 0:
        # Parentheses enforce projection through component space first, avoiding
        # construction of a dense (time, time) projector matrix.
        filtered_signals = signals - clutter_vectors @ (
            clutter_vectors.conj().T @ signals
        )
    else:
        # Adaptive selection must not change output precision between windows.
        filtered_signals = signals.astype(
            np.result_type(signals.dtype, clutter_vectors.dtype), copy=False
        )

    return filtered_signals


def clutter_filter_svd_from_indices(
    block: npt.NDArray,
    mask: npt.NDArray | None = None,
    low_cutoff: int | None = None,
    high_cutoff: int | None = None,
) -> npt.NDArray:
    """Filter IQ data using SVD clutter filtering based on component indices.

    This function performs singular value decomposition (SVD) on masked IQ signals
    and removes clutter by regressing out singular vectors outside the provided
    index range. Singular vectors are ordered by decreasing energy, so lower indices
    correspond to higher-energy components (typically tissue clutter).

    !!! warning
        `block` will be cast to double-precision floating point to avoid numerical
        instabilities.

    Parameters
    ----------
    block : (time, k, j, i) numpy.ndarray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(k, j, i)` are spatial dimensions.
    mask : (k, j, i) numpy.ndarray, optional
        Boolean mask selecting at least one voxel. SVD is computed only from masked
        voxels. If not provided, all voxels are used.
    low_cutoff : int, optional
        Lower bound for singular vector indices to retain (inclusive), counted from the
        highest-energy component (index 0). Vectors with indices less than `low_cutoff`
        are treated as clutter and removed. If not provided, defaults to `0` (no
        high-energy removal).
    high_cutoff : int, optional
        Upper bound for singular vector indices to retain (exclusive), counted from the
        highest-energy component (index 0). Vectors with indices greater than or equal
        to `high_cutoff` are treated as clutter and removed. `high_cutoff` must be at
        most `min(time, mask.sum())`. If not provided, defaults to the maximum number
        of components (no low-energy removal).

    Returns
    -------
    (time, k, j, i) numpy.ndarray
        Filtered IQ data.

    Raises
    ------
    ValueError
        If `block` is not 4D, or if cutoff values are invalid.

    Notes
    -----
    For efficiency, this function computes the eigendecomposition of the temporal
    Gram matrix rather than full SVD of the data matrix, avoiding computation of
    the large spatial covariance matrix.

    References
    ----------
    [^1]:
        Demene, Charlie, et al. "Spatiotemporal Clutter Filtering of Ultrafast
        Ultrasound Data Highly Increases Doppler and fUltrasound Sensitivity." IEEE
        Transactions on Medical Imaging, vol. 34, no. 11, Nov. 2015, pp. 2271–85.
        DOI.org (Crossref), <https://doi.org/10.1109/TMI.2015.2428634>.

    [^2]:
        Baranger, Jerome, et al. "Adaptive Spatiotemporal SVD Clutter Filtering for
        Ultrafast Doppler Imaging Using Similarity of Spatial Singular Vectors." IEEE
        Transactions on Medical Imaging, vol. 37, no. 7, July 2018, pp. 1574–86. DOI.org
        (Crossref), <https://doi.org/10.1109/TMI.2018.2789499>.

    [^3]:
        Le Meur-Diebolt, Samuel, et al. "Robust Functional Ultrasound Imaging in the
        Awake and Behaving Brain: A Systematic Framework for Motion Artifact Removal."
        17 June 2025. Neuroscience, <https://doi.org/10.1101/2025.06.16.659882>.
    """
    signals, masked_signals = _validate_block_and_clutter_mask(block, mask)

    if masked_signals.size == 0:
        return block

    max_components = cast(int, min(masked_signals.shape))
    low_cutoff, high_cutoff = validate_svd_index_cutoffs(
        low_cutoff, high_cutoff, max_components
    )

    if low_cutoff == 0 and high_cutoff == max_components:
        return block

    # Computing the eigendecomposition of the temporal Gram matrix is more efficient
    # than computing the SVD of the data matrix directly, as we avoid computing the full
    # spatial covariance matrix, which can be very large. Here, the eigenvalues
    # correspond to the squared singular values, that is, the energies of the singular
    # vectors.
    _eigenvalues, eigenvectors = _compute_gram_eigendecomposition(
        # Casting to double-precision for stability.
        masked_signals.astype(np.cdouble, copy=False)
    )

    clutter_components = np.concatenate(
        [eigenvectors[:, :low_cutoff], eigenvectors[:, high_cutoff:]], axis=1
    )

    filtered_signals = _apply_clutter_filter(signals, clutter_components)

    return filtered_signals.reshape(block.shape)


def clutter_filter_svd_from_energy(
    block: npt.NDArray,
    mask: npt.NDArray | None = None,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
) -> npt.NDArray:
    """Filter IQ data using SVD clutter filtering based on component energies.

    This function performs singular value decomposition (SVD) on masked IQ signals and
    removes clutter by regressing out singular vectors whose energies fall outside the
    specified range. This is an adaptive filtering approach that identifies clutter
    based on signal energy rather than component indices.

    !!! warning
        `block` will be cast to double-precision floating point to avoid numerical
        instabilities.

    Parameters
    ----------
    block : (time, k, j, i) numpy.ndarray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(k, j, i)` are spatial dimensions.
    mask : (k, j, i) numpy.ndarray, optional
        Boolean mask selecting at least one voxel. SVD is computed only from masked
        voxels. If not provided, all voxels are used.
    low_cutoff : int or float, optional
        Lower bound for singular vector energy to retain (inclusive). Vectors with
        energy less than `low_cutoff` are treated as clutter and removed. `low_cutoff`
        must be positive. If not provided, defaults to `0.0` (no low-energy removal).
    high_cutoff : int or float, optional
        Upper bound for singular vector energy to retain (inclusive). Vectors with
        energy greater than `high_cutoff` are treated as clutter and removed. If not
        provided, defaults to `numpy.inf` (no high-energy removal).

    Returns
    -------
    (time, k, j, i) numpy.ndarray
        Filtered IQ data.

    Raises
    ------
    ValueError
        If `block` is not 4D, or if cutoff values are invalid.

    Notes
    -----
    For efficiency, this function computes the eigendecomposition of the temporal
    Gram matrix rather than full SVD of the data matrix, avoiding computation of
    the large spatial covariance matrix.

    References
    ----------
    [^1]:
        Demene, Charlie, et al. "Spatiotemporal Clutter Filtering of Ultrafast
        Ultrasound Data Highly Increases Doppler and fUltrasound Sensitivity." IEEE
        Transactions on Medical Imaging, vol. 34, no. 11, Nov. 2015, pp. 2271–85.
        DOI.org (Crossref), <https://doi.org/10.1109/TMI.2015.2428634>.

    [^2]:
        Baranger, Jerome, et al. "Adaptive Spatiotemporal SVD Clutter Filtering for
        Ultrafast Doppler Imaging Using Similarity of Spatial Singular Vectors." IEEE
        Transactions on Medical Imaging, vol. 37, no. 7, July 2018, pp. 1574–86. DOI.org
        (Crossref), <https://doi.org/10.1109/TMI.2018.2789499>.

    [^3]:
        Le Meur-Diebolt, Samuel, et al. "Robust Functional Ultrasound Imaging in the
        Awake and Behaving Brain: A Systematic Framework for Motion Artifact Removal."
        17 June 2025. Neuroscience, <https://doi.org/10.1101/2025.06.16.659882>.
    """
    signals, masked_signals = _validate_block_and_clutter_mask(block, mask)

    if masked_signals.size == 0:
        return block

    low_cutoff, high_cutoff = validate_energy_cutoffs(low_cutoff, high_cutoff)

    if low_cutoff == 0.0 and high_cutoff == np.inf:
        return block

    # Computing the eigendecomposition of the temporal Gram matrix is more efficient
    # than computing the SVD of the data matrix directly, as we avoid computing the full
    # spatial covariance matrix, which can be very large. Here, the eigenvalues
    # correspond to the squared singular values, that is, the energies of the singular
    # vectors.
    eigenvalues, eigenvectors = _compute_gram_eigendecomposition(
        # Casting to double-precision for stability.
        masked_signals.astype(np.cdouble, copy=False)
    )

    clutter_vectors = eigenvectors[
        :, (eigenvalues < low_cutoff) | (eigenvalues > high_cutoff)
    ]

    filtered_signals = _apply_clutter_filter(signals, clutter_vectors)

    return filtered_signals.reshape(block.shape)


def clutter_filter_svd_from_cumulative_energy(
    block: npt.NDArray,
    mask: npt.NDArray | None = None,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
) -> npt.NDArray:
    """Filter IQ data using SVD clutter filtering based on cumulative component energies.

    This function performs singular value decomposition (SVD) on masked IQ signals
    and removes clutter by regressing out singular vectors whose cumulative energies
    fall outside the specified range. This approach allows filtering based on the
    total energy contribution of components, useful for retaining a specific
    percentage of total signal energy.

    !!! warning
        `block` will be cast to double-precision floating point to avoid numerical
        instabilities.

    Parameters
    ----------
    block : (time, k, j, i) numpy.ndarray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(k, j, i)` are spatial dimensions.
    mask : (k, j, i) numpy.ndarray, optional
        Boolean mask selecting at least one voxel. SVD is computed only from masked
        voxels. If not provided, all voxels are used.
    low_cutoff : int or float, optional
        Lower bound for cumulative energy to retain (inclusive). Components with
        cumulative energy lower than `low_cutoff` are treated as clutter and removed.
        `low_cutoff` must be positive. If not provided, defaults to `0.0` (no
        low-energy removal).
    high_cutoff : int or float, optional
        Upper bound for cumulative energy to retain (inclusive). Components with
        cumulative energy greater than `high_cutoff` are treated as clutter and removed.
        If not provided, defaults to `numpy.inf` (no high-energy removal).

    Returns
    -------
    (time, k, j, i) numpy.ndarray
        Filtered IQ data.

    Raises
    ------
    ValueError
        If `block` is not 4D, or if cutoff values are invalid.

    Notes
    -----
    Cumulative energy is computed as the running sum of singular values squared.
    This allows filtering based on the total energy contribution rather than
    individual component energies.

    For efficiency, this function computes the eigendecomposition of the temporal
    Gram matrix rather than full SVD of the data matrix, avoiding computation of
    the large spatial covariance matrix.

    References
    ----------
    [^1]:
        Demene, Charlie, et al. "Spatiotemporal Clutter Filtering of Ultrafast
        Ultrasound Data Highly Increases Doppler and fUltrasound Sensitivity." IEEE
        Transactions on Medical Imaging, vol. 34, no. 11, Nov. 2015, pp. 2271–85.
        DOI.org (Crossref), <https://doi.org/10.1109/TMI.2015.2428634>.

    [^2]:
        Baranger, Jerome, et al. "Adaptive Spatiotemporal SVD Clutter Filtering for
        Ultrafast Doppler Imaging Using Similarity of Spatial Singular Vectors." IEEE
        Transactions on Medical Imaging, vol. 37, no. 7, July 2018, pp. 1574–86. DOI.org
        (Crossref), <https://doi.org/10.1109/TMI.2018.2789499>.

    [^3]:
        Le Meur-Diebolt, Samuel, et al. "Robust Functional Ultrasound Imaging in the
        Awake and Behaving Brain: A Systematic Framework for Motion Artifact Removal."
        17 June 2025. Neuroscience, <https://doi.org/10.1101/2025.06.16.659882>.
    """
    signals, masked_signals = _validate_block_and_clutter_mask(block, mask)

    if masked_signals.size == 0:
        return block

    low_cutoff, high_cutoff = validate_energy_cutoffs(low_cutoff, high_cutoff)

    if low_cutoff == 0.0 and high_cutoff == np.inf:
        return block

    # Computing the eigendecomposition of the temporal Gram matrix is more efficient
    # than computing the SVD of the data matrix directly, as we avoid computing the full
    # spatial covariance matrix, which can be very large. Here, the eigenvalues
    # correspond to the squared singular values, that is, the energies of the singular
    # vectors.
    eigenvalues, eigenvectors = _compute_gram_eigendecomposition(
        # Casting to double-precision for stability.
        masked_signals.astype(np.cdouble, copy=False),
        # Computing the eigendecomposition in ascending order so that np.cumsum
        # accumulates from the lowest-energy (noise/blood) component to the highest-energy
        # (tissue/clutter) component, matching the semantics of low_cutoff and
        # high_cutoff.
        ascending=True,
    )

    cumsum_energy = np.cumsum(eigenvalues)
    clutter_vectors = eigenvectors[
        :, np.logical_or(cumsum_energy < low_cutoff, cumsum_energy > high_cutoff)
    ]

    filtered_signals = _apply_clutter_filter(signals, clutter_vectors)

    return filtered_signals.reshape(block.shape)


def clutter_filter_sosfiltfilt(
    block: npt.NDArray,
    sos: npt.NDArray,
    *,
    padtype: str | None = "odd",
    padlen: int | None = None,
) -> npt.NDArray:
    """Filter IQ data using second-order sections (SOS) digital filter.

    Applies a digital filter defined by second-order sections using forward-backward
    filtering to eliminate phase distortion. This is a general-purpose filtering
    function that accepts pre-computed SOS coefficients from any SciPy filter design.

    Parameters
    ----------
    block : (time, k, j, i) numpy.ndarray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(k, j, i)` are spatial dimensions.
    sos : (sections, 6) numpy.ndarray
        Second-order sections filter coefficients, typically obtained from SciPy
        functions like `scipy.signal.butter`, `scipy.signal.cheby1`, etc.
    padtype : {"odd", "even", "constant", None}, default: "odd"
        Padding extension type. Pass `None` to disable padding.
    padlen : int, optional
        Padding length in samples. If not provided, uses SciPy's default.
        Ignored when padding is disabled.

    Returns
    -------
    (time, k, j, i) numpy.ndarray
        Filtered IQ data.

    Raises
    ------
    ValueError
        If `block` is not 4D.

    Notes
    -----
    Forward-backward filtering (`scipy.signal.sosfiltfilt`) ensures zero phase delay by
    filtering the signal twice: once forward and once backward.
    """
    import scipy.signal as sp_signal

    if block.ndim != 4:
        raise ValueError(f"'block' must be 4D, got {block.ndim}D")

    return sp_signal.sosfiltfilt(sos, block, axis=0, padtype=padtype, padlen=padlen)


def clutter_filter_butterworth(
    block: npt.NDArray,
    fs: float,
    *,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    order: int = 4,
    padtype: str | None = "odd",
    padlen: int | None = None,
) -> npt.NDArray:
    """Filter IQ data using a Butterworth digital filter.

    Applies a Butterworth infinite impulse response (IIR) filter using forward-backward
    filtering (`scipy.signal.sosfiltfilt`) to eliminate phase distortion. Supports
    low-pass, high-pass, and band-pass filtering based on the cutoff frequency
    parameters.

    Parameters
    ----------
    block : (time, k, j, i) numpy.ndarray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(k, j, i)` are spatial dimensions.
    fs : float
        Sampling frequency in hertz.
    low_cutoff : float, optional
        Low cutoff frequency in hertz, in range `(0, fs/2)`. If provided, applies
        high-pass filtering above this frequency.
    high_cutoff : float, optional
        High cutoff frequency in hertz, in range `(0, fs/2)`. If provided, applies
        low-pass filtering below this frequency.
    order : int, default: 4
        Filter order. Due to forward-backward filtering, the effective order is
        doubled.
    padtype : {"odd", "even", "constant", None}, default: "odd"
        Padding extension type. Pass `None` to disable padding.
    padlen : int, optional
        Padding length in samples. If not provided, uses SciPy's default.
        Ignored when padding is disabled.

    Returns
    -------
    (time, k, j, i) numpy.ndarray
        Filtered IQ data.

    Raises
    ------
    ValueError
        If `block` is not 4D or if cutoff frequencies are invalid.
    """
    if block.ndim != 4:
        raise ValueError(f"'block' must be 4D, got {block.ndim}D")

    if low_cutoff is None and high_cutoff is None:
        return block

    sos = design_butterworth_filter(fs, low_cutoff, high_cutoff, order)
    return clutter_filter_sosfiltfilt(block, sos=sos, padtype=padtype, padlen=padlen)


def compute_svd_cumulative_energy_threshold(
    iq: "xr.DataArray",
    singular_value_index: int,
    clutter_mask: "xr.DataArray | None" = None,
    window_width: int | np.integer | None = None,
    window_stride: int | np.integer | None = None,
) -> "da.Array":
    """Compute the high cutoff threshold for the cumulative-energy SVD clutter filter.

    Iterates over sliding temporal windows of an IQ acquisition and computes the
    cumulative eigenvalue spectrum for each window. The threshold is defined as the
    minimum cumulative energy at a given singular-value index across all windows,
    ensuring that the chosen cutoff does not over-filter any window.

    The result is returned as a lazy scalar [`dask.array.Array`][dask.array.Array]. Call
    [`.compute`][dask.array.Array.compute] on it to obtain the concrete `float` value
    when needed.

    Parameters
    ----------
    iq : (time, k, j, i) xarray.DataArray
        Complex beamformed IQ data, where `time` is the temporal dimension and
        `(k, j, i)` are native voxel spatial dimensions.
    singular_value_index : int
        Number of high-energy components to remove. Must satisfy
        `1 <= singular_value_index <= window_width - 1`.
    clutter_mask : (k, j, i) xarray.DataArray, optional
        Boolean spatial mask. Eigendecomposition is computed only from masked voxels.
        If not provided, all voxels are used.
    window_width : int, optional
        Width of the sliding temporal window, in volumes. If not provided, defaults
        to the total number of volumes.
    window_stride : int, optional
        Stride of the sliding temporal window, in volumes. If not provided, defaults
        to `window_width`.

    Returns
    -------
    dask.array.Array
        Lazy scalar array containing the minimum cumulative energy at
        `singular_value_index` across all sliding windows. Call `.compute()` to
        materialise the value. It can be passed directly as `high_cutoff` to
        [`clutter_filter_svd_from_cumulative_energy`][confusius.iq.clutter_filters.clutter_filter_svd_from_cumulative_energy]
        after computing.

    Raises
    ------
    ValueError
        If `singular_value_index` is less than 1 or greater than `window_width - 1`,
        or if `clutter_mask` selects no voxels.

    Notes
    -----
    For efficiency, this function computes the eigendecomposition of the temporal Gram
    matrix rather than full SVD of the data matrix, avoiding computation of the large
    spatial covariance matrix.

    Eigenvalues are sorted in ascending order, so the cumulative sum accumulates from
    the lowest-energy (noise/blood) component to the highest-energy (tissue/clutter)
    component. The threshold is set to the cumulative energy just *below* the top
    `singular_value_index` components, so that when used as `high_cutoff` in
    [`clutter_filter_svd_from_cumulative_energy`][confusius.iq.clutter_filters.clutter_filter_svd_from_cumulative_energy],
    exactly those `singular_value_index` high-energy (tissue/clutter) components have
    cumulative energy that exceeds the threshold and are removed.

    References
    ----------
    [^1]:
        Demene, Charlie, et al. "Spatiotemporal Clutter Filtering of Ultrafast
        Ultrasound Data Highly Increases Doppler and fUltrasound Sensitivity." IEEE
        Transactions on Medical Imaging, vol. 34, no. 11, Nov. 2015, pp. 2271–85.
        DOI.org (Crossref), <https://doi.org/10.1109/TMI.2015.2428634>.

    [^2]:
        Baranger, Jerome, et al. "Adaptive Spatiotemporal SVD Clutter Filtering for
        Ultrafast Doppler Imaging Using Similarity of Spatial Singular Vectors." IEEE
        Transactions on Medical Imaging, vol. 37, no. 7, July 2018, pp. 1574–86. DOI.org
        (Crossref), <https://doi.org/10.1109/TMI.2018.2789499>.

    [^3]:
        Le Meur-Diebolt, Samuel, et al. "Robust Functional Ultrasound Imaging in the
        Awake and Behaving Brain: A Systematic Framework for Motion Artifact Removal."
        17 June 2025. Neuroscience, <https://doi.org/10.1101/2025.06.16.659882>.
    """
    import dask.array as da
    from scipy import linalg as sp_linalg

    # Deferred to avoid circular import: clutter_filters <- process <- clutter_filters.
    from confusius.iq.process import process_iq_blocks

    iq = ensure_iq_voxeldata(iq)

    mask_array: npt.NDArray[np.bool_] | None = None
    if clutter_mask is not None:
        clutter_mask = ensure_voxeldata(
            clutter_mask, allow_pose=False, allow_extra_dims=False
        )
        validate_mask(clutter_mask, iq, "clutter_mask")
        mask_array = clutter_mask.values.astype(bool)

    dask_iq = iq.data
    if not isinstance(dask_iq, da.Array):
        dask_iq = da.from_array(dask_iq)

    window_width, window_stride, _ = resolve_iq_window(
        dask_iq.shape[0],
        window_width,
        window_stride,
        default_width=max(dask_iq.chunks[0]),
    )

    if not 1 <= singular_value_index <= window_width - 1:
        raise ValueError(
            f"singular_value_index ({singular_value_index}) must be between 1 and "
            f"window_width - 1 ({window_width - 1})."
        )

    def _window_energy(block: npt.NDArray) -> npt.NDArray:
        """Return shape-(1,) array with cumulative energy for one window."""
        _, signals = _validate_block_and_clutter_mask(block, mask_array)
        signals = signals.astype(np.cdouble, copy=False)

        # Eigenvalues are in ascending order; cumsum accumulates from lowest to highest
        # energy. Indexing at -(singular_value_index + 1) gives the cumulative energy
        # just below the top singular_value_index components, so that components with
        # cumsum strictly greater than this threshold are exactly those top components.
        gram_matrix = signals @ signals.conj().T
        eigenvalues = sp_linalg.eigvalsh(gram_matrix)
        return np.array([np.cumsum(eigenvalues)[-(singular_value_index + 1)].real])

    energies = process_iq_blocks(
        dask_iq,
        _window_energy,
        window_width=window_width,
        window_stride=window_stride,
        output_shape=(1,),
        output_dtype=np.float64,
    )

    return energies.min()
