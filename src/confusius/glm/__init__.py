"""GLM analysis for fUSI data.

This package provides General Linear Model tools for voxel-wise statistical analysis of
functional ultrasound imaging data. The code is adapted from [Nilearn's GLM
module](https://nilearn.github.io/stable/modules/glm.html) under the BSD-3-Clause
license.
"""

from confusius.glm._contrasts import Contrast
from confusius.glm._design import make_first_level_design_matrix
from confusius.glm._hrf_models import (
    claron2021_hrf,
    gamma_difference_hrf,
    gamma_hrf,
    glover_hrf,
    inverse_gamma_hrf,
    spm_hrf,
    verhoef2025_hrf,
)
from confusius.glm._models import (
    RegressionResults,
)  # exported for user type annotations
from confusius.glm.first_level import FirstLevelModel
from confusius.glm.second_level import SecondLevelModel, make_second_level_design_matrix

__all__ = [
    "Contrast",
    "FirstLevelModel",
    "RegressionResults",
    "SecondLevelModel",
    "claron2021_hrf",
    "gamma_difference_hrf",
    "gamma_hrf",
    "glover_hrf",
    "inverse_gamma_hrf",
    "make_first_level_design_matrix",
    "make_second_level_design_matrix",
    "spm_hrf",
    "verhoef2025_hrf",
]
