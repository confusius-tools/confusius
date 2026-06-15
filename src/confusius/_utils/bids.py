"""Shared BIDS event helpers used by the events I/O and the GLM design tools.

Both `confusius.bids.events` (reading/writing BIDS events files) and
`confusius.glm._design` (validating events before building a design matrix) need
the same notion of a default trial type and the same normalization of a candidate
trial-type value. Keeping them here avoids two sources of truth for the default.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_TRIAL_TYPE = "event"
"""Trial type assigned to events whose `trial_type` is missing or empty."""


def normalize_trial_type(trial_type: object) -> str:
    """Return a non-empty trial type, falling back to the default.

    Parameters
    ----------
    trial_type : object
        Candidate trial type. Missing values (`None`, NaN) and blank strings
        are replaced by the default trial type.

    Returns
    -------
    str
        The stripped trial type, or `DEFAULT_TRIAL_TYPE` when the input is
        missing or empty.
    """
    if trial_type is None or pd.isna(trial_type):
        return DEFAULT_TRIAL_TYPE
    text = str(trial_type).strip()
    return text or DEFAULT_TRIAL_TYPE
