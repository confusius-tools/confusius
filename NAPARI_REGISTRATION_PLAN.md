# Napari registration panel plan

## Goal

Add a first **Registration** tab to the ConfUSIus napari plugin so users can run the main registration workflows directly from the viewer.

## Phase 1 scope

Deliver a thin but usable panel focused on running registrations and adding the resampled output back to napari.

### Included

- New **Registration** accordion tab in the napari plugin.
- Use the same Lucide icon as the docs registration page: `images`.
- Support:
  - `register_volume`
  - `register_volumewise`
- Always resample in the GUI.
  - No `resample=True/False` toggle.
  - The resampled result is always added as a **new layer**.
- Background-thread execution so the napari UI stays responsive.
- Result-layer metadata carries the original `xarray.DataArray`, parameters, diagnostics, and transform provenance.
- Save / load / apply affine transform UI.
- Cancellation / abort support with partial-result return semantics.
- In-napari live progress for both workflows.
  - `register_volume`: live red/cyan overlay + metric plot.
  - `register_volumewise`: determinate progress bar + progressively filled output layer.
- Per-frame / per-iteration progress callbacks for `register_volumewise`.

### Not yet included

- Manual initialization transforms from direct napari interaction.
- Standalone `resample_like` / `resample_volume` actions in the panel.
- Registration masks in the panel.
- Unified payload support for arbitrary manual napari-created transforms.
- Non-affine transform payload support.

## UX decisions

### `register_volume`

- Requires a moving layer and a fixed layer.
- Both must be spatial-only volumes.
- Result layer name should clearly indicate the fixed target.
- Keep the estimated transform and diagnostics in metadata for later reuse.

### `register_volumewise`

- Operates on one time-series layer.
- Uses a selected `reference_time`.
- Adds the registered time series as a new layer.
- Preserve motion metadata already returned by `register_volumewise`.

## Implementation notes

### Layer → DataArray conversion

The panel should prefer `layer.metadata["xarray"]` when available.
For generic napari layers without ConfUSIus metadata, reconstruct a simple `xarray.DataArray` from:

- `layer.data`
- `layer.scale`
- `layer.translate`
- `layer.axis_labels`
- `layer.units`

This keeps manual/foreign napari layers usable in the panel.

### Provenance

Store a small provenance payload on the result layer metadata, including:

- operation name
- moving layer name
- fixed layer name when applicable
- transform model
- metric
- interpolation
- transform object for `register_volume`
- diagnostics

## Follow-up phases

### Phase 2

Transform management.

#### Implemented

- Save/load/apply affine transforms from the registration panel.
- Stable ConfUSIus-owned JSON payload for affine transforms.
- Human-friendly transform names in the payload.
- Output-grid metadata stored with the transform so a saved transform can be
  reapplied later without reloading the original fixed/reference layer.
- Affine registration results store a reusable transform payload in layer metadata.

#### Remaining polish

- Better internal layout for the registration tab as it grows.
- Unified payload support for manual napari-created transforms.
- Optional support for non-affine transform payloads in the future.
- Decide whether volumewise should also hide / retint the source layer after
  completion, mirroring the single-volume workflow more closely.

### Phase 3

Manual initialization:

- capture napari layer transforms as initialization affines
- apply saved affines back onto layers
- reset/apply current transform actions

### Phase 4

Progress integration.

#### Implemented

- `progress_plotter` factory argument on `register_volume`; defaults to the
  matplotlib plotter, the napari plugin injects a Qt-signal bridge.
- Napari-side bridge + `NapariVolumeProgress` reporter resamples the moving
  image at every SimpleITK iteration and streams the array into a live
  `Image` layer (the "resampled" overlay).
- The fixed layer is tinted red, the moving layer is tinted cyan + additive
  and hidden during the run; the preview is seeded with the moving image
  resampled onto the fixed grid (identity transform) so the first frame is
  a meaningful "unaligned moving on fixed" view.
- Bottom-dock `RegistrationMetricPlotter` widget renders the per-iteration
  optimizer metric curve. Coalesces redraws through a 16 ms `QTimer` so
  rapid iteration events don't flood the GUI thread.
- `register_volumewise` exposes a public progress-reporter hook.
- Napari volumewise registration uses a determinate progress bar.
- Napari volumewise registration pre-creates the output layer, fills it with
  the moving-layer minimum value, then writes frames in as they finish.
- During volumewise progress, the original layer is tinted red and the
  in-progress output layer is tinted cyan + additive for visual comparison.

### Phase 5

Panel polish.

#### Implemented

- Sidebar widened so the "Moving layer" label and dropdown align with the
  rest of the form rows.
- Run button is disabled (greyed out, visibly non-clickable) when the
  current layer selection is invalid (no moving layer, missing fixed
  layer, moving == fixed, time-dim mismatch, etc.). Re-evaluated on every
  selection / mode / param change.
- Learning rate spinbox lower bound lowered below `1e-6` so bspline and
  fine-scale transforms can use the small rates they need.
- Missing `register_volume` / `register_volumewise` parameters exposed in
  the panel: `number_of_histogram_bins` (mattes MI), convergence
  (`convergence_minimum_value`, `convergence_window_size`),
  `centering_initialization`, `shrink_factors` / `smoothing_sigmas`,
  `fill_value`, `keep_diagnostics`, and `n_jobs` / **Parallel jobs**.
  Grouped into a basic section plus a foldable in-panel "Advanced" section.
- Advanced-row visibility is context-sensitive:
  - histogram bins only show for `mattes_mi`
  - shrink factors / smoothing sigmas only show when multi-resolution is on
  - parallel jobs only show for within-scan registration
- The whole "Advanced" header is clickable, not just the disclosure triangle.
- Volumewise mode defaults to a fixed learning rate of `0.01` with `Auto`
  unticked; between-scan mode keeps `Auto` on.
- Mode-specific parameter state is preserved while the panel stays open:
  switching between between-scan and within-scan restores the last values
  used in that mode instead of resetting them.
- Thicker determinate progress bar so the percentage text remains visible.
- **Abort button** for both `register_volume` and `register_volumewise`.
  Aborting stops at the next cooperative checkpoint and returns the current
  partial result instead of failing the worker.

#### Remaining polish

- Better internal layout for the registration tab as it grows.
- Unified payload support for manual napari-created transforms.
- Optional support for non-affine transform payloads in the future.

### Phase 6

CLI / Python UX polish.

#### Implemented

- `register_volume` is now Ctrl+C-aware in Python usage: on the main thread,
  the first Ctrl+C is converted into cooperative cancellation via the shared
  abort event, and the current partial result is returned with
  `diagnostics.status="aborted"`.

#### Remaining polish

- Consider extending the same Ctrl+C wrapper to higher-level workflows beyond
  direct `register_volume` calls if needed.
