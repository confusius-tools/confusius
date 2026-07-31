"""Tests for shared motion plotting helpers."""

from confusius._utils.motion_plotting import (
    FD_LABELS,
    get_dark_motion_color_cycle,
    get_motion_diagnostic_label,
    get_motion_diagnostic_linewidth,
)


def test_motion_diagnostic_labels() -> None:
    """Column labels match motion diagnostics plots."""
    assert get_motion_diagnostic_label("rotation") is None
    assert get_motion_diagnostic_label("rot_x") == "x"
    assert get_motion_diagnostic_label("trans_y") == "y"
    assert get_motion_diagnostic_label("mean_fd") == "Mean FD"
    assert get_motion_diagnostic_label("custom") == "custom"


def test_motion_diagnostic_linewidths() -> None:
    """Line widths match motion diagnostics plots."""
    assert get_motion_diagnostic_linewidth("mean_fd") == 1.8
    assert get_motion_diagnostic_linewidth("max_fd") == 1.2
    assert get_motion_diagnostic_linewidth("trans_x") == 1.6


def test_dark_motion_color_cycle_uses_dark_background_style() -> None:
    """Dark color cycle is taken from Matplotlib's dark background style."""
    cycle = get_dark_motion_color_cycle()
    assert len(list(cycle)) == 10
    assert set(FD_LABELS) == {"mean_fd", "max_fd", "rms_fd"}
