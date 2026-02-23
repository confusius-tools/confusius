"""Tests for confusius._utils."""

from confusius._utils import find_stack_level


def test_find_stack_level():
    """Test find_stack_level."""
    assert find_stack_level() == 1
