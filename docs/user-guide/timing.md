---
icon: lucide/clock
---

# Temporal Resampling

!!! info "Coming soon"
    This page is currently under construction. The timing module provides:

    - [`get_time_coord_to_seconds_factor`][confusius.timing.get_time_coord_to_seconds_factor]: Get the
      conversion factor from time coordinate units to seconds.
    - [`convert_time_units`][confusius.timing.convert_time_units]: Convert time values between units.
    - [`get_representative_time_step`][confusius.timing.get_representative_time_step]: Get a representative time step
      from non-uniform time coordinates.
    - [`convert_time_reference`][confusius.timing.convert_time_reference]: Convert between volume
      timing references (start, center, end).
    - [`resample_time`][confusius.timing.resample_time]: Resample fUSI time series to new time
      coordinates.
    - [`resample_to_uniform_time`][confusius.timing.resample_to_uniform_time]: Resample fUSI time
      series onto a uniform time grid, fixing jitter, dropped frames, or non-uniform sampling.

    Please refer to the [API Reference](../api/timing.md) for more information.