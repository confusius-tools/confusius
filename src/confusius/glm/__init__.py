"""GLM analysis for fUSI data.

This package provides General Linear Model tools for voxel-wise statistical analysis of
functional ultrasound imaging data. The code is adapted from [Nilearn's GLM
module](https://nilearn.github.io/stable/modules/glm.html) under the BSD-3-Clause
license.
"""

from confusius.glm._contrasts import Contrast
from confusius.glm._design import make_first_level_design_matrix
from confusius.glm._models import (
    RegressionResults,
)  # exported for user type annotations
from confusius.glm.first_level import FirstLevelModel

__all__ = [
    "FirstLevelModel",
    "make_first_level_design_matrix",
    "RegressionResults",
    "Contrast",
]
