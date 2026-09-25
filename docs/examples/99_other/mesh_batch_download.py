"""Benchmark: BrainGlobe lazy per-mesh S3 download vs one batched s3fs.get call.

BrainGlobe 3.x downloads each region mesh on first access (`fs.exists` + `fs.get`,
one file at a time). s3fs/fsspec `get` accepts *lists* of remote/local paths and
fetches them concurrently, so prefetching all wanted meshes in one call is much
faster on a cold cache. Run with `uv run python docs/examples/99_other/mesh_batch_download.py`.
"""

import os
import time
from pathlib import Path

import s3fs
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.descriptors import remote_url_s3
from fsspec.callbacks import TqdmCallback

N_PER_ARM = 100
KEEP_DOWNLOADS = False  # Delete what this script fetched so re-runs stay cold.


def remote_mesh_path(mesh_filename) -> str:
    """Mirror `Structure._download_mesh`: last 6 path parts under the S3 atlas root."""
    return remote_url_s3.format("/".join(str(mesh_filename).split(os.sep)[-6:]))


def prefetch_meshes(
    atlas: BrainGlobeAtlas, ids: list[int], batch_size: int = 16
) -> None:
    """Download all missing mesh files for `ids` in one concurrent s3fs call."""
    missing = [Path(atlas.structures[i]["mesh_filename"]) for i in ids]
    missing = [p for p in missing if not p.exists()]
    if not missing:
        return
    fs = s3fs.S3FileSystem(anon=True)
    # One LIST call replaces BrainGlobe's per-mesh `fs.exists`, and names the exact
    # keys that are absent remotely: a batched `fs.get` only raises a bare
    # "The specified key does not exist." with no path in it.
    remote_dir = remote_mesh_path(missing[0]).rsplit("/", 1)[0]
    remote_names = {Path(k).name for k in fs.ls(remote_dir)}
    absent = [remote_mesh_path(p) for p in missing if p.name not in remote_names]
    if absent:
        raise FileNotFoundError(f"No remote mesh for: {absent}")
    try:
        fs.get(
            [remote_mesh_path(p) for p in missing],
            [str(p) for p in missing],
            callback=TqdmCallback(),
            batch_size=batch_size,
        )
    except BaseException:
        # Mirror `Structure._download_mesh`: never leave a partial mesh in the cache,
        # since BrainGlobe treats any existing file as a valid cached mesh.
        for p in missing:
            p.unlink(missing_ok=True)
        raise


def main() -> None:
    atlas = BrainGlobeAtlas("allen_mouse_25um")

    import numpy as np

    rois = set(np.unique(atlas.annotation)) - {0, 545}
    rois = {s["id"] for s in atlas.structures_list} - {545}

    uncached = [
        s["id"]
        for s in atlas.structures_list
        if s["mesh_filename"] is not None and not Path(s["mesh_filename"]).exists()
    ]
    seq_ids, batch_ids = uncached[:N_PER_ARM], uncached[N_PER_ARM : 2 * N_PER_ARM]
    downloaded = [
        Path(atlas.structures[i]["mesh_filename"]) for i in seq_ids + batch_ids
    ]

    t0 = time.perf_counter()
    for i in seq_ids:
        atlas.structures[i]["mesh"]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    prefetch_meshes(atlas, rois)
    t_batch_dl = time.perf_counter() - t0
    for i in batch_ids:
        atlas.structures[i]["mesh"]
    t_batch = time.perf_counter() - t0

    print(f"\nsequential (brainglobe lazy) {N_PER_ARM} meshes: {t_seq:.2f}s")
    print(
        f"batched s3fs.get            {N_PER_ARM} meshes: {t_batch_dl:.2f}s download, {t_batch:.2f}s incl. decode"
    )
    print(f"speedup: {t_seq / t_batch:.1f}x")

    assert all(p.exists() for p in downloaded)
    if not KEEP_DOWNLOADS:
        for p in downloaded:
            p.unlink()


if __name__ == "__main__":
    main()
