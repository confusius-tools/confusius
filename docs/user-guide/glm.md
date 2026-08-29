---
icon: lucide/square-function
---

# General Linear Model

!!! info "Coming soon"
    This page is currently under construction. The `glm` module provides a general
    linear model implementation for stimulus-evoked and task-based fUSI analysis.

    **First-level analysis:**

    - [`FirstLevelModel`][confusius.glm.FirstLevelModel]: Estimate a first-level GLM from fUSI data.
    - [`make_first_level_design_matrix`][confusius.glm.make_first_level_design_matrix]: Create a design matrix from stimulus onset times and timing parameters.

    **Second-level analysis:**

    - [`SecondLevelModel`][confusius.glm.SecondLevelModel]: Estimate a second-level (group) GLM.
    - [`make_second_level_design_matrix`][confusius.glm.make_second_level_design_matrix]: Create a design matrix for group analysis.

    **Contrasts:**

    - [`Contrast`][confusius.glm.Contrast]: Define and compute contrasts for statistical inference.

    **HRF models:**

    - [`gamma_difference_hrf`][confusius.glm.gamma_difference_hrf]: Base double-gamma HRF model with customizable parameters.
    - [`glover_hrf`][confusius.glm.glover_hrf]: Glover canonical HRF.
    - [`spm_hrf`][confusius.glm.spm_hrf]: SPM canonical HRF.
    - [`gamma_hrf`][confusius.glm.gamma_hrf]: Single-gamma HRF without undershoot (suitable for fUSI).
    - [`inverse_gamma_hrf`][confusius.glm.inverse_gamma_hrf]: Inverse gamma HRF model.
    - [`verhoef2025_hrf`][confusius.glm.verhoef2025_hrf]: Human fUSI HRF preset (Verhoef et al., 2025).
    - [`claron2021_hrf`][confusius.glm.claron2021_hrf]: Rodent spinal cord fUSI HRF preset (Claron et al., 2021).

    Please refer to the [API Reference](../api/glm.md) for more information.

!!! tip "Parallel multi-run fits"
    Multi-run fits can be parallelized over runs with the `n_jobs` parameter of
    [`FirstLevelModel`][confusius.glm.FirstLevelModel], e.g.
    `FirstLevelModel(n_jobs=-1)` to use all CPUs.
    Each run is sent to a joblib worker process, so lazy h5py-backed data (e.g.
    straight from `load_scan`) must be materialized with `.compute()` first. joblib
    caps each worker's BLAS thread pool automatically, so the workers do not
    oversubscribe the CPU (unless you set `OMP_NUM_THREADS`-style variables
    yourself).
