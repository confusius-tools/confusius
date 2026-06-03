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
- Minimal parameter surface:
  - operation
  - moving layer
  - fixed layer for `register_volume`
  - reference time for `register_volumewise`
  - transform model
  - metric
  - resampling interpolation
  - optional multi-resolution toggle
  - learning rate
  - number of iterations
- Run work in a background thread so the napari UI stays responsive.
- Attach the resulting `xarray.DataArray` to layer metadata, plus transform/diagnostic provenance.

### Not yet included

- Manual initialization transforms from direct napari interaction.
- Save/load/apply transform UI.
- In-napari live registration progress plots.
- Per-frame progress callbacks for `register_volumewise` or resampling utilities.
- Standalone `resample_like` / `resample_volume` actions.
- Registration masks.
- Cancellation.

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

Transform management:

- save/load/apply affine transforms
- stable serialized transform payload owned by ConfUSIus
- better provenance for manual vs optimized transforms

### Phase 3

Manual initialization:

- capture napari layer transforms as initialization affines
- apply saved affines back onto layers
- reset/apply current transform actions

### Phase 4

Progress integration:

- custom progress hooks for `register_volume`
- napari-native metric/composite viewer
- per-frame progress callback for `register_volumewise`
