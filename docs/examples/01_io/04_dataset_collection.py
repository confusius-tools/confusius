# %% [markdown]
# # The ConfUSIus Dataset Collection
#
# ConfUSIus provides access to public fUSI datasets and brain templates from several
# studies. The [datasets user guide](../../../user-guide/datasets.md) describes the
# available data, fetchers, and source publications; the
# [collection repository](https://github.com/confusius-tools/confusius-datasets)
# documents preparation recipes, provenance, and per-source licenses.
#
# This example introduces the collection's organization and access conventions.
# The other [gallery examples](../../index.md) demonstrate how to use these data for
# registration, decomposition, connectivity, and statistical analysis.

# %% [markdown]
# ## Browse the collection
#
# The public S3 bucket contains two prefixes, `datasets/` and `templates/`, with
# immutable releases under `<identifier>/<version>/`. The `last_versions.conf`
# catalog lists the latest published versions; each release's `manifest.json`
# records its file paths, sizes, and SHA-256 checksums.
#
# Fetchers resolve the latest published release on the first download, then reuse
# the cached version. Pass `refresh=True` to check for a newer release without
# overwriting an earlier one. Set `data_dir` for a particular fetch, or the
# `CONFUSIUS_DATA` environment variable to choose the download-cache directory for
# all fetchers (for example, `export CONFUSIUS_DATA=/path/to/cache`). `data_dir`
# takes priority; otherwise the platform cache directory is used by default.
#
# [`list_datasets`][confusius.datasets.list_datasets] prints the available fetchers,
# full dataset/template sizes, and whether their metadata is cached locally,
# without downloading data.

# %%
from pathlib import Path

import matplotlib as mpl
import xarray as xr

import confusius as cf

xr.set_options(display_expand_data=False)
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])
cf.datasets.list_datasets()

# %% [markdown]
# ## Data formats
#
# Datasets follow fUSI-BIDS, the functional ultrasound extension of
# [BIDS](https://bids.neuroimaging.io/); see the
# [current specification draft](https://docs.google.com/document/d/1W3z01mf1E8cfg_OY7ZGqeUeOKv659jCHQBXavtmT-T8/edit?usp=sharing).
# Images are stored as [NIfTI](https://nifti.nimh.nih.gov/) files with acquisition,
# processing, and timing metadata in JSON sidecars. [`confusius.load`][confusius.load]
# loads both automatically, merging matching sidecars according to the BIDS
# inheritance principle: more specific metadata overrides less specific metadata.
#
# Templates are reference NIfTI volumes for spatial alignment; their fetchers
# return VoxelData arrays directly. The [gallery](../../index.md) shows how to use
# collection datasets and templates in analysis workflows.
#
# Each dataset and template retains its own license. Fetchers print source citations
# by default (`print_citation=False` disables this); citations and per-source licenses
# are also listed on the [citing page](../../../citing.md).

# %% [markdown]
# ## A small structural preview
#
# As a quick view of the collection, fetch one structural ultrasound image from
# [Cybis Pereira et al. (2026)](https://doi.org/10.1016/j.celrep.2025.116791).
# Dataset fetchers accept study-specific filters such as subject, session,
# acquisition, and datatype. Matching JSON sidecars and supporting BIDS metadata
# are included, and downloaded files are verified against the release manifest.
#
# For a fuller introduction to downloading, loading, and manipulating recordings,
# see [ConfUSIus and Xarray 101](confusius_xarray_101.md).

# %%
bids_root = Path(
    cf.datasets.fetch_cybis_pereira_2026(
        datasets="rawdata",
        subjects="rat75",
        sessions="20220523",
        acqs="slice32",
        datatypes="susi",
    )
)
image_path = (
    bids_root
    / "sub-rat75"
    / "ses-20220523"
    / "susi"
    / "sub-rat75_ses-20220523_acq-slice32_rec-minframe2d_pwd.nii.gz"
)
assert image_path.with_suffix("").with_suffix(".json").is_file()
image = cf.load(image_path).compute()
assert {"k", "j", "i"}.issubset(image.dims)
print("Release:", bids_root.name, "\nVoxel dimensions:", dict(image.sizes))

# %% [markdown]
# This is a static anatomical reference, with vessels that can serve as alignment
# landmarks. Display it in the source's metric `qform` space, as in
# [two-session registration](../registration/register_volume_same_subject.md).

# %% tags=["thumbnail"]
image_qform = image.fusi.affine.apply(image.attrs["affines"]["world_to_qform"])
plotter = image_qform.fusi.scale.db().fusi.plot.volume(
    cmap="gray",
    cbar_label="Power Doppler (dB)",
    bg_color=bg_color,
)

# %% [markdown]
# ## Explore the data further
#
# The worked analysis examples show how to answer scientific questions with these
# recordings, including the preprocessing, models, and visualizations:
#
# - [Lagged GLM](../glm/first_level_continuous.md): how does vascular activity relate
#   to locomotion speed? Reproduce the Cybis Pereira analysis with a continuous
#   behavioral regressor at different temporal lags.
# - [PCA](../decomposition/pca_single_recording.md): which spatial patterns explain
#   the dominant variance in a recording? Extract component maps and time courses.
# - [Task GLM](../glm/first_level.md): which regions respond to olfactory stimulation
#   in the Khallaf recordings? Fit and visualize a stimulus contrast.
# - [Regional connectivity](../connectivity/atlas_correlation_matrix.md): how do
#   atlas-defined regional signals covary in the Nunez-Elizalde recordings?
#
# Beyond these examples, the ConfUSIus dataset collection can support further
# analyses and cross-study meta-analyses, taking each study's acquisition and
# processing differences into account. See the [citing page](../../../citing.md)
# for source references and licenses.
