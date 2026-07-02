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

from confusius.datasets import fetch_cybis_pereira_2026, fetch_nunez_elizalde_2022


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

    # docs/examples/io/confusius_xarray_101.py
    fetch_nunez_elizalde_2022(
        subjects="CR022",
        sessions="20201011",
        tasks="spontaneous",
        acqs="slice03",
    )


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

    # docs/examples/registration/register_volume_same_subject.py
    fetch_cybis_pereira_2026(
        datasets="rawdata",
        subjects="rat75",
        sessions=["20220523", "20220524"],
        datatypes="angio",
        acqs="slice32",
    )


def main() -> None:
    _prefetch_nunez_elizalde()
    _prefetch_cybis_pereira()


if __name__ == "__main__":
    main()
