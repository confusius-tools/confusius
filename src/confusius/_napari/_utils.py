"""Shared helpers for the ConfUSIus napari plugin."""

CATEGORICAL_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]
"""Qualitative palette cycled to assign distinct colors to categories.

Shared by the signal store (per imported signal) and the event store (per trial
type) so the plugin uses one consistent set of distinguishable colors.
"""
