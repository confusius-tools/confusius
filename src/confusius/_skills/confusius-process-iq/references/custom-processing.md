# Custom IQ block processing

Use `cf.iq.process_iq_blocks` when built-in IQ processing does not fit. It calls
`process_func` on NumPy `(time, k, j, i)` windows and returns a bare Dask array.
Callbacks must return `(output_time, ...)` with consistent shape/dtype and no side
effects: they run during eager inference and again during execution.

## Example: directional power Doppler

For validated complex `iq` VoxelData with `(time, k, j, i)` dimensions, compute both
directions from one clutter-filtered block and FFT. Use contiguous, regularly sampled
windows; the frame count and SVD cutoff below are illustrative, not defaults.

```python
import numpy as np

import confusius as cf


def directional_power(block):
    """Compute both directional powers from one IQ window.

    Parameters
    ----------
    block : (time, k, j, i) numpy.ndarray
        Complex IQ samples.

    Returns
    -------
    (1, 2, k, j, i) numpy.ndarray
        Positive- and negative-frequency mean powers.
    """
    block = cf.iq.clutter_filter_svd_from_indices(block, low_cutoff=40)
    n = block.shape[0]
    frequencies = np.fft.fftfreq(n)
    power = np.abs(np.fft.fft(block, axis=0) / n) ** 2
    positive = power[frequencies > 0].sum(axis=0)
    # Exclude the direction-ambiguous Nyquist bin at -0.5 cycles/frame (even n).
    negative = power[(frequencies < 0) & (frequencies > -0.5)].sum(axis=0)
    return np.stack([positive, negative])[None, ...]


width = 300
result = cf.iq.process_iq_blocks(
    iq.chunk({"time": width}).data,
    process_func=directional_power,
    window_width=width,
    window_stride=width,
    new_axis=1,  # Separate direction from output time.
)
```

Both maps are nonnegative powers, not velocities. Their sum equals conventional power
Doppler minus excluded DC/Nyquist power. Calibrate scanner IQ sign before assigning
toward/away flow; noise, clutter leakage, and aliasing can obscure direction. Keep both
maps for analysis: subtraction is useful for signed display but cancels opposing flows.

This multi-map callback uses a public clutter filter directly. For single-map custom
measurements with nested integration windows, reuse
`process_iq_block_with_clutter_filter`: its inner callback returns `(k, j, i)` and
accepts forwarded `fs`/keywords. Bind inner `window_width` / `window_stride` using
`functools.partial` so they do not collide with the outer arguments.

## Reconstruct VoxelData

Continue the example with the input geometry and actual window timings, not guessed
spacing or `dt`. Here each outer block produces one measurement per direction.
Use IQ timestamps in seconds; the constructor labels raw time arrays in seconds.

```python
times, durations = cf.iq.compute_processed_volume_timings(
    iq,
    clutter_window_width=width,
    clutter_window_stride=width,
    inner_window_width=width,
    inner_window_stride=width,
)
if not np.allclose(durations, durations[0]):
    raise ValueError("Split into uniform-duration segments before reconstruction.")

directional = cf.create_voxeldata(
    result,
    dims=("time", "direction", "k", "j", "i"),
    extra_coords={"direction": ["positive", "negative"]},
    time=times,
    volume_acquisition_reference=iq.time.volume_acquisition_reference,
    volume_acquisition_duration=float(durations[0]),
    k=iq.k,
    j=iq.j,
    i=iq.i,
    voxel_to_world=iq.fusi.affine.voxel_to_world,
    units=iq.z.attrs["units"],
    name="directional_power_doppler",
    attrs={"units": "a.u."},
)
```

`directional` is lazy VoxelData with dimensions `(direction, time, k, j, i)`; select
either map with `.sel(direction="positive")` or `.sel(direction="negative")`. Power has
squared IQ units, conventionally labeled arbitrary units here.

Copy geometry only for an unchanged grid. Match timing to actual sample support;
preserve varying durations separately or split segments. Check values, geometry, and
timing against small synthetic inputs. See [temporal windows](temporal-windows.md),
[clutter filtering](clutter-filtering.md), and [performance](performance.md).
