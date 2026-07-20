"""Signal processing module for fUSI time series."""

from confusius.signal.censor import censor_samples, interpolate_samples
from confusius.signal.confounds import compute_compcor_confounds, regress_confounds
from confusius.signal.core import clean
from confusius.signal.detrending import detrend
from confusius.signal.filters import filter_butterworth, filter_cosine
from confusius.signal.standardization import standardize

__all__ = [
    "censor_samples",
    "clean",
    "compute_compcor_confounds",
    "detrend",
    "filter_butterworth",
    "filter_cosine",
    "interpolate_samples",
    "regress_confounds",
    "standardize",
]
