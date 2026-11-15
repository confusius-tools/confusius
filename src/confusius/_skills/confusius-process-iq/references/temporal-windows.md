# Choosing IQ temporal windows

Use this reference to select widths/strides and interpret output timing. All window
arguments count **compound IQ frames**, not transmitted pulses or seconds.

## Outer and inner windows

- **B-mode:** `bmode_window_width` / `bmode_window_stride` integrate one window without
  clutter filtering.
- **Doppler/velocity:** filter each outer `clutter_window_width` /
  `clutter_window_stride` block, then measure inner `doppler_window_*` or
  `velocity_window_*` windows. Outer width controls clutter estimation; inner width
  controls integration. Shorter inner windows can give more frequent output without
  re-estimating the clutter subspace.
- Outer/B-mode widths default to Dask time-chunk size; inner widths default to outer
  width; strides default to widths. **Set widths explicitly** to avoid chunk-dependent
  results. This is even more important after loading IQ into memory, where the whole
  recording may be one chunk.

Use at least one complete outer window, inner widths that fit within it, and strides no
larger than widths. Incomplete tails are dropped. Overlap can be useful for smoother
time courses but will create autocorrelation in the output signals.

## Choose a physical duration first

For contiguous regular sampling, `frames ≈ round(duration_s * compound_rate_hz)`. Verify
timestamps: compound rate is not transmit PRF, and angle counts alone may not capture
repeated firings or acquisition delays.

**Illustrative preclinical settings:** 200–300 compound frames appear in conventional
2D brain fUSI protocols. At 500 Hz this spans 400–600 ms; at 1000 Hz, 200–300 ms.
These are pilot candidates, not optimized defaults. Longer outer windows supply more
samples but may decrease clutter filter performance due to brain motion. Longer
integration reduces temporal resolution, but may improve SNR. Reassess cutoffs when
window length changes; see [clutter filtering](clutter-filtering.md).

Split at acquisition gaps or align blocks to ensemble boundaries: windows **do not
detect dead time**. Butterworth wrappers require regular sampling within every clutter
window and derive its sampling frequency locally, allowing gaps between windows. Other
filter paths use a global representative time step for velocity; split changes in
sampling rate before SVD-based velocity estimation rather than assuming local rates are
used there too. Scanner `compound_sampling_frequency` metadata does not override the
frequency derived from timestamps.

## Verify output timing

Output spacing differs from integration duration, especially for overlap or nested
windows. Check returned `time`, `volume_acquisition_reference`, and
`volume_acquisition_duration`. If integration durations vary, built-in wrappers warn and
store their **median**. Use `cf.iq.compute_processed_volume_timings` for individual
spans when needed; it returns timestamps/durations in the input time units. Do not
interpret the median as every window's duration or invent uniform timing across gaps.
