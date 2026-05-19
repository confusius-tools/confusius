"""Red/cyan composite blend helpers shared by static and live registration views."""

import numpy as np
from numpy.typing import NDArray


def scale_min_max(arr: NDArray[np.floating]) -> NDArray[np.floating]:
    """Linearly scale `arr` to [0, 1], handling flat arrays gracefully.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Float array with the same shape as `arr`, rescaled to `[0, 1]`. Returns an
        all-zero array when `arr` is flat (`arr.min() == arr.max()`).
    """
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def blend_red_cyan(
    fixed: NDArray[np.floating], moving: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Blend two 2D arrays as red (fixed) and cyan (moving) channels.

    Parameters
    ----------
    fixed : numpy.ndarray
        2D reference array, normalized to [0, 1].
    moving : numpy.ndarray
        2D moving array, normalized to [0, 1].

    Returns
    -------
    numpy.ndarray
        RGB image of shape `(*fixed.shape, 3)`.
    """
    h, w = fixed.shape
    rgb = np.zeros((h, w, 3))
    # Red channel: fixed only.
    rgb[..., 0] = fixed
    # Green + blue channels: cyan = moving.
    rgb[..., 1] = moving
    rgb[..., 2] = moving
    return rgb


def make_mosaic(
    fixed_vol: NDArray[np.floating], moving_vol: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Assemble a mosaic of per-slice red/cyan blends along the first axis.

    Parameters
    ----------
    fixed_vol : numpy.ndarray
        3D reference volume `(n_slices, H, W)`.
    moving_vol : numpy.ndarray
        3D moving volume `(n_slices, H, W)`.

    Returns
    -------
    numpy.ndarray
        RGB mosaic image.
    """
    n = fixed_vol.shape[0]
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    h, w = fixed_vol.shape[1], fixed_vol.shape[2]

    mosaic = np.zeros((n_rows * h, n_cols * w, 3))
    for i in range(n):
        r, c = divmod(i, n_cols)
        blend = blend_red_cyan(
            scale_min_max(fixed_vol[i]),
            scale_min_max(moving_vol[i]),
        )
        mosaic[r * h : (r + 1) * h, c * w : (c + 1) * w] = blend
    return mosaic
