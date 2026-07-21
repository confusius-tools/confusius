"""Pre-fetch every dataset that the docs pipeline loads.

This is the single source of truth for which fUSI-BIDS data the documentation
build depends on. It covers both:

- the userguide image generators in `docs/images/*/generate.py`, and
- the gallery examples in `docs/examples/`.

CI uses this file's hash as the dataset cache key, and runs the script before
the image generators and the gallery builder so every downstream call hits a
warm cache. That matters for the gallery in particular: the renderer requires
the light and dark passes to produce identical output, and download progress
on a cold cache would otherwise show up in one but not the other.

Add an entry here whenever a new image generator or example pulls data. The
args must match the call site exactly so the OSF index resolves to the same
files.
"""

from __future__ import annotations

from confusius.datasets import (
    fetch_brainglobe_atlas,
    fetch_cybis_pereira_2026,
    fetch_khallaf_2026,
    fetch_nunez_elizalde_2022,
    fetch_template_pepe_mariani_2026,
)


def _prefetch_nunez_elizalde() -> None:
    # docs/images/home/generate.py
    fetch_nunez_elizalde_2022(
        subjects=["CR022", "CR024"],
        sessions=["20201011", "20201029"],
        tasks="spontaneous",
        acqs="slice03",
    )

    # docs/images/qc/generate.py
    fetch_nunez_elizalde_2022(
        subjects="CR024",
        sessions="20201029",
        tasks="spontaneous",
        acqs="slice03",
    )

    # docs/images/gui/generate.py
    fetch_nunez_elizalde_2022(
        subjects="CR022",
        sessions=["20201007", "20201011"],
        tasks="spontaneous",
        acqs="slice04",
    )

    # docs/images/visualization/generate.py
    fetch_nunez_elizalde_2022(
        subjects="CR022",
        sessions=["20201011", "20201007"],
        tasks="spontaneous",
        acqs="slice04",
    )

    # docs/examples/01_io/01_confusius_xarray_101.py
    fetch_nunez_elizalde_2022(
        subjects="CR022",
        sessions="20201011",
        tasks="spontaneous",
        acqs="slice03",
    )

    # docs/examples/04_connectivity/01_atlas_correlation_matrix.py,
    # docs/examples/04_connectivity/02_atlas_seed_map.py,
    # docs/examples/05_atlases_and_templates/01_saving_resampled_atlas.py
    fetch_nunez_elizalde_2022(
        subjects="CR022",
        sessions="20201007",
        tasks="spontaneous",
        acqs="slice02",
    )


def _prefetch_pepe_mariani_template() -> None:
    # docs/examples/04_connectivity/01_atlas_correlation_matrix.py,
    # docs/examples/04_connectivity/02_atlas_seed_map.py,
    # docs/examples/05_atlases_and_templates/01_saving_resampled_atlas.py
    fetch_template_pepe_mariani_2026()


def _prefetch_allen_atlas() -> None:
    # docs/examples/04_connectivity/01_atlas_correlation_matrix.py,
    # docs/examples/04_connectivity/02_atlas_seed_map.py,
    # docs/examples/05_atlases_and_templates/01_saving_resampled_atlas.py
    fetch_brainglobe_atlas("allen_mouse_100um")


def _prefetch_cybis_pereira() -> None:
    # docs/images/gui/generate.py (openfield video panel)
    fetch_cybis_pereira_2026(
        datasets=["rawdata", "dlc-videos"],
        subjects="rat75",
        sessions="20220525",
        acqs="slice37",
    )

    # docs/images/gui/generate.py (within-scan registration GIF)
    fetch_cybis_pereira_2026(
        datasets="rawdata",
        subjects="rat75",
        sessions="20220523",
        acqs="slice32",
    )

    # docs/examples/02_registration/01_register_volume_same_subject.py
    fetch_cybis_pereira_2026(
        datasets="rawdata",
        subjects="rat75",
        sessions=["20220523", "20220524"],
        datatypes="angio",
        acqs="slice32",
    )

    # docs/examples/02_registration/02_volumewise_motion_correction.py
    fetch_cybis_pereira_2026(
        datasets="rawdata",
        subjects="rat75",
        sessions="20220523",
        datatypes="fusi",
        acqs="slice32",
    )

    # docs/examples/05_glm/02_first_level_continuous.py
    fetch_cybis_pereira_2026(
        datasets="rawdata",
        subjects="rat75",
        sessions="20220524",
        acqs="slice32",
    )


def _prefetch_khallaf() -> None:
    # docs/examples/glm/first_level.py
    fetch_khallaf_2026(
        datasets="rawdata",
        subjects="5622",
        sessions="IPM",
        reconstruction="resampled",
    )
    # Mouse template and Allen atlas used for registration and masks in the same
    # example. Warming the brainglobe atlas here keeps its download progress out of
    # the parity-sensitive gallery render.
    fetch_template_pepe_mariani_2026()
    fetch_brainglobe_atlas("allen_mouse_100um", check_latest=False)


def main() -> None:
    _prefetch_nunez_elizalde()
    _prefetch_cybis_pereira()
    _prefetch_pepe_mariani_template()
    _prefetch_allen_atlas()
    _prefetch_khallaf()


if __name__ == "__main__":
    main()
