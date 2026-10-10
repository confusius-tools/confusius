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
# overwriting an earlier one. Use `data_dir` or `CONFUSIUS_DATA` to choose the cache
# location. No AWS account or credentials are needed.
#
# Read the public catalog to see the dataset and template prefixes currently
# available. This access workflow requires ConfUSIus 0.8.0 or its development version.

# %%
import configparser
from pathlib import Path

import matplotlib as mpl
import requests
import xarray as xr

import confusius as cf

xr.set_options(display_expand_data=False)
bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])
base_url = "https://confusius-datasets.s3.us-west-2.amazonaws.com"
response = requests.get(f"{base_url}/last_versions.conf", timeout=30)
response.raise_for_status()
catalog = configparser.ConfigParser(interpolation=None)
catalog.read_string(response.text)
for category in ("datasets", "templates"):
    print(f"{category}/")
    for identifier, version in sorted(catalog[category].items()):
        print(f"  {identifier}/{version}/")

# %% [markdown]
# ## Data formats
#
# Recording datasets follow [BIDS](https://bids.neuroimaging.io/) and the
# [fUSI-BIDS specification draft](https://docs.google.com/document/d/1W3z01mf1E8cfg_OY7ZGqeUeOKv659jCHQBXavtmT-T8/edit?usp=sharing).
# Subject/session directories and filename entities such as `task-` and `acq-`
# organize recordings. `fusi/` contains functional images and `susi/` contains
# structural ultrasound images where available. Study-specific protocols and
# metadata still differ; consistent names do not imply equivalent measurements.
#
# Imaging arrays and spatial transforms are stored in
# [NIfTI](https://nifti.nimh.nih.gov/) files, supported by ConfUSIus and NiBabel.
# Download their JSON sidecars too: they describe acquisition, processing, and
# exact timing that the image header alone cannot always represent. TSV tables
# hold participant, scan, event, or behavioral information and can be read with
# pandas or Python's `csv` module. Some studies also provide videos and derivatives.
#
# Templates are reference NIfTI volumes with spatial metadata for registration and
# anatomical alignment; their fetchers return VoxelData arrays directly. BrainGlobe
# atlases are fetched separately and are not hosted in this collection.
#
# The [gallery](../../index.md) uses these datasets and templates throughout, except
# for the MAT-file example, which downloads a separate Rabut archive. See
# [multi-pose loading](../io/load_multipose_recordings.md) for fUSI-BIDS file assembly
# and [template registration](../registration/register_to_allen_fusi_template.md)
# for working with anatomical reference volumes.
#
# Licenses remain source-specific, including CC BY 4.0, CC0 1.0, CC BY-NC 4.0, and
# CC BY-NC-SA 4.0. Check the user guide and source terms before reuse, and cite the
# original dataset/publication and ConfUSIus when using its tooling. The code's
# license does not replace data licenses. Public S3 access is anonymous; any Amazon
# EC2 or SageMaker compute used for analysis is billed separately.

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
# see [ConfUSIus and Xarray 101](../io/confusius_xarray_101.md).

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
    cmap="magma",
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
# - [Searchlight decoding](../decoding/searchlight_speed.md): can local patterns of
#   vascular activity predict speed? Compare a multivariate model with a GLM.
# - [Task GLM](../glm/first_level.md): which regions respond to olfactory stimulation
#   in the Khallaf recordings? Fit and visualize a stimulus contrast.
# - [Regional connectivity](../connectivity/atlas_correlation_matrix.md): how do
#   atlas-defined regional signals covary in the Nunez-Elizalde recordings?
#
# Looking across studies, an open question is whether a representation of fUSI
# vascular dynamics learned on one dataset can transfer to another dataset or task.
# A useful starting point would be small subsets from two studies, a within-study
# baseline, and evaluation held out by subject or study. Differences in species,
# geometry, sampling, and preprocessing should be inspected rather than pooled
# away; report results separately and record release versions and source licenses.
# This can be explored using the existing public recordings without collecting new
# data.
