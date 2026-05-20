"""Diagnostics returned alongside a registration result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RegistrationDiagnostics:
    """Per-call diagnostics for a registration optimization.

    Returned as the third element of
    [`register_volume`][confusius.registration.register_volume]. Useful for
    plotting metric convergence curves, comparing runs, and detecting frames
    that failed to converge.

    Attributes
    ----------
    metric : {"correlation", "mattes_mi"}
        Similarity metric used during optimization, echoed from
        `register_volume`'s `metric` argument. Determines the sign and scale
        of `metric_values` (SimpleITK always minimizes, so both supported
        metrics are recorded as negative values).
    metric_values : (n_iterations,) numpy.ndarray
        Similarity metric value recorded at each optimizer iteration, in
        chronological order.
    final_metric_value : float
        Last value of the metric recorded by the optimizer. Equal to
        `metric_values[-1]` when at least one iteration ran.
    n_iterations : int
        Number of iterations actually performed by the optimizer. May be
        smaller than `register_volume`'s `number_of_iterations` if the
        optimizer converged early.
    stop_condition : str
        Human-readable description of the optimizer stop condition, as
        returned by SimpleITK's `GetOptimizerStopConditionDescription`.
    """

    metric: Literal["correlation", "mattes_mi"]
    metric_values: npt.NDArray[np.floating]
    final_metric_value: float
    n_iterations: int
    stop_condition: str
