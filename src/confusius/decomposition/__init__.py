"""Decomposition techniques for fUSI data."""

from confusius.decomposition.fastica import FastICA
from confusius.decomposition.nmf import NMF
from confusius.decomposition.pca import PCA

__all__ = [
    "FastICA",
    "NMF",
    "PCA",
]
