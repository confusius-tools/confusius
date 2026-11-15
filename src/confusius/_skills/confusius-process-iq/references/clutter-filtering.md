# Choosing IQ clutter filters

Use this reference to select filters, SVD cutoffs, and masks. See [temporal
windows](temporal-windows.md) for durations and acquisition timing. Spatiotemporal SVD
uses space and time to separate coherent tissue from blood signals; their temporal
frequencies can overlap, especially during motion ([Demené et al.,
2015](https://doi.org/10.1109/TMI.2015.2428634)).

## Select a method

| Method | Retained components / cutoff units | Use and trade-off |
| --- | --- | --- |
| `svd_indices` | Indices `[low_cutoff, high_cutoff)`, strongest first at index 0. | Fixed-count starting approach; assess residual motion clutter. |
| `svd_energy` | Individual energies between absolute bounds (squared singular values). | Adaptive count; thresholds depend on amplitude and acquisition. |
| `svd_cumulative_energy` | Absolute cumulative energies between bounds, summed weakest to strongest. | Variable-motion alternative; can distort global hemodynamics. |
| `butterworth` | `low_cutoff` / `high_cutoff` are frequency bounds in Hz. | Specialized filtering; generally insufficient alone to separate tissue and slow blood. |

For `svd_indices`, `low_cutoff=40` discards the first 40 components (typically tissue);
`high_cutoff=40` retains only those components (potentially useful for computing axial
tissue velcity). Both bounds retain their interval; neither means no filtering. Integer
bounds must satisfy `0 <= low_cutoff < high_cutoff <= min(window_frames,
selected_voxels)`; omitted bounds use the interval ends. The basis is recomputed per
block; only the component count is fixed. Energy-based methods can instead reject
different numbers of components per window. Their thresholds are not percentages or
component indices.

For conventional 2D preclinical brain fUSI, removing 30–60 leading components in windows
of size 200–300 is an initial range to investigate, but not a general rule. Adjust for
window length, field of view/anatomy, mask, and motion. Published settings outside these
ranges exist.

## Cumulative-energy SVD for variable motion

Constant-energy filtering varies the rejected component count to constrain retained
energy. In the awake-mouse datasets and functional-connectivity benchmark evaluated by
[Le Meur-Diebolt et al. (2026)](https://doi.org/10.1162/IMAG.a.1191), this improved
motion robustness over fixed-count filtering. It is not universally superior.

For validated `iq` and boolean VoxelData `brain_mask`, calibrate a `high_cutoff` using a
reference component count (settings are illustrative):

```python
import confusius as cf

threshold = cf.iq.compute_svd_cumulative_energy_threshold(
    iq,
    singular_value_index=40,
    clutter_mask=brain_mask,
    window_width=200,
    window_stride=200,
).compute()

pwd = cf.iq.process_iq_to_power_doppler(
    iq,
    clutter_window_width=200,
    clutter_window_stride=200,
    clutter_mask=brain_mask,
    filter_method="svd_cumulative_energy",
    high_cutoff=float(threshold),
)
```

The helper returns the **minimum retained energy across windows** after excluding the
specified strongest components. Match its windows and mask to processing. However,
constant-energy filtering suppresses genuine global changes; validate signal
preservation and avoid this method when global blood dynamics are central.

[Baranger et al. (2018)](https://doi.org/10.1109/TMI.2018.2789499) uses a different
adaptive method based on spatial singular-vector similarity; this method is not yet
available in ConfUSIus but can be implemented with `process_iq_blocks` and a custom
callback.

## Masking and checking artifacts

A boolean VoxelData `clutter_mask` matching the IQ grid selects voxels for estimating
SVD components; **filtering still covers the full grid**. A brain mask can reduce bias
from strong extracerebral signals. Omit the mask to estimate components from all voxels.

Compare a few settings with the user on quiet and motion-rich blocks:

- **Underfiltering:** widespread power surges, tissue/muscle contamination, motion
  spikes.
- **Overfiltering:** lost small vessels, apparent global signal drops, reduced
  functional connectivity.
- **Time courses:** compare vessel signals and available motion measures (axial
  velocity, rDVARS), remembering that delayed vascular responses can be physiological.

Do not select thresholds solely from cleaner-looking images: compare high- and
low-motion frames for apparent artifacts. Record method, cutoffs, mask, and windows.
Power Doppler measures filtered blood-signal power, not absolute blood volume.
