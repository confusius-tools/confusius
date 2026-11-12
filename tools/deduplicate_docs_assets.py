"""Deduplicate identical raster image assets in a generated documentation tree."""

from __future__ import annotations

import argparse
import hashlib
import posixpath
import re
import shutil
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ASSET_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
TEXT_EXTENSIONS = {".css", ".html", ".js", ".json", ".md", ".mjs", ".txt", ".xml"}
SHARED_DIRECTORY = Path("shared-assets")
ASSET_REFERENCE_PATTERN = re.compile(
    r"(?P<url>(?:https?://[^\"'()\s]+|/[^\"'()\s]+|[^\"'()\s]+)"
    r"\.(?:avif|bmp|gif|jpeg|jpg|png|tiff|tif|webp)(?:[?#][^\"'()\s]*)?)",
    re.IGNORECASE,
)


def get_asset_hash(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> list[Path]:
    """Return regular files below a documentation tree.

    Parameters
    ----------
    root : pathlib.Path
        Documentation tree to scan.

    Returns
    -------
    list[pathlib.Path]
        Regular files below ``root``, excluding Git metadata and shared assets.
    """
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(root).parts
        and SHARED_DIRECTORY not in path.relative_to(root).parents
    ]


def get_relative_url(source: Path, target: Path) -> str:
    """Return a POSIX relative URL from one file to another.

    Parameters
    ----------
    source : pathlib.Path
        Source file whose containing directory is the URL base.
    target : pathlib.Path
        Target file.

    Returns
    -------
    str
        Relative POSIX URL from ``source`` to ``target``.
    """
    start = source.parent.as_posix() or "."
    return posixpath.relpath(target.as_posix(), start=start)


def rewrite_references(
    root: Path, replacements: dict[Path, Path], dry_run: bool
) -> tuple[int, set[Path]]:
    """Rewrite text references to deduplicated assets.

    Parameters
    ----------
    root : pathlib.Path
        Documentation tree to update.
    replacements : dict[pathlib.Path, pathlib.Path]
        Mapping from original asset paths to shared asset paths.
    dry_run : bool
        Whether to report changes without writing files.

    Returns
    -------
    changed_file_count : int
        Number of text files changed.
    referenced_assets : set[pathlib.Path]
        Original assets referenced by at least one text file.
    """
    changed_files = 0
    referenced_assets: set[Path] = set()
    text_files = [
        path for path in iter_files(root) if path.suffix.lower() in TEXT_EXTENSIONS
    ]
    for text_file in text_files:
        content = text_file.read_text(encoding="utf-8")

        def replace_reference(match: re.Match[str], text_file: Path = text_file) -> str:
            url = match.group("url")
            split_url = urlsplit(url)
            if (split_url.scheme or split_url.netloc) and not (
                split_url.scheme in ("", "http", "https")
                and split_url.netloc == "confusius.tools"
            ):
                return url
            asset_url = split_url.path
            if asset_url.startswith("/"):
                asset_rel = Path(asset_url.lstrip("/"))
            else:
                source = text_file.relative_to(root)
                asset_rel = Path(
                    posixpath.normpath(
                        posixpath.join(source.parent.as_posix(), asset_url)
                    )
                )
            original = (root / asset_rel).resolve()
            shared = replacements.get(original)
            if shared is None:
                return url
            referenced_assets.add(original)
            shared_rel = shared.relative_to(root).as_posix()
            if asset_url.startswith("/"):
                replacement_path = f"/{shared_rel}"
            else:
                replacement_path = get_relative_url(
                    text_file.relative_to(root), Path(shared_rel)
                )
            return urlunsplit(
                (
                    split_url.scheme,
                    split_url.netloc,
                    replacement_path,
                    split_url.query,
                    split_url.fragment,
                )
            )

        updated = ASSET_REFERENCE_PATTERN.sub(replace_reference, content)
        if updated != content:
            changed_files += 1
            if not dry_run:
                text_file.write_text(updated, encoding="utf-8")
    return changed_files, referenced_assets


def deduplicate_assets(root: Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Deduplicate identical raster image files in a documentation tree.

    Parameters
    ----------
    root : pathlib.Path
        Documentation tree to update.
    dry_run : bool, default: False
        Whether to report the planned changes without modifying files.

    Returns
    -------
    duplicate_count : int
        Number of duplicate asset files removed.
    changed_file_count : int
        Number of text files whose references changed.
    pruned_count : int
        Number of unreferenced shared assets removed.
    """
    root = root.resolve()
    shared_root = root / SHARED_DIRECTORY
    if not dry_run:
        shared_root.mkdir(exist_ok=True)

    grouped_assets: defaultdict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in iter_files(root):
        if path.suffix.lower() in ASSET_EXTENSIONS:
            grouped_assets[(get_asset_hash(path), path.suffix.lower())].append(path)

    existing_shared: dict[tuple[str, str], Path] = {}
    if shared_root.exists():
        for path in shared_root.iterdir():
            if path.is_file() and path.suffix.lower() in ASSET_EXTENSIONS:
                existing_shared[(get_asset_hash(path), path.suffix.lower())] = path

    replacements: dict[Path, Path] = {}
    files_to_remove: list[Path] = []
    created_shared: list[Path] = []

    for key, assets in grouped_assets.items():
        if len(assets) < 2 and key not in existing_shared:
            continue
        digest, extension = key
        shared = existing_shared.get(key, shared_root / f"{digest}{extension}")
        if not shared.exists():
            if not dry_run:
                shutil.copy2(assets[0], shared)
            created_shared.append(shared)
        for asset in assets:
            replacements[asset] = shared
            files_to_remove.append(asset)

    changed_file_count, referenced_assets = rewrite_references(
        root, replacements, dry_run
    )
    referenced_shared = set()
    if shared_root.exists():
        for path in iter_files(root):
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            content = path.read_text(encoding="utf-8")
            for shared in shared_root.glob("*"):
                if shared.name in content:
                    referenced_shared.add(shared)

    files_to_remove = [
        path
        for path in files_to_remove
        if path in referenced_assets
        and (dry_run or replacements[path] in referenced_shared)
    ]
    duplicate_count = len(files_to_remove)
    if not dry_run:
        for path in files_to_remove:
            path.unlink()

    pruned_count = 0
    if shared_root.exists():
        for path in shared_root.iterdir():
            if path.is_file() and path not in referenced_shared:
                pruned_count += 1
                if not dry_run:
                    path.unlink()

    if not dry_run and shared_root.exists() and not any(shared_root.iterdir()):
        shared_root.rmdir()

    return duplicate_count, changed_file_count, pruned_count


def main() -> int:
    """Run documentation asset deduplication from the command line.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Generated documentation tree")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without modifying the documentation tree",
    )
    args = parser.parse_args()
    duplicate_count, changed_file_count, pruned_count = deduplicate_assets(
        args.root, dry_run=args.dry_run
    )
    print(
        f"deduplicated={duplicate_count} "
        f"rewritten={changed_file_count} "
        f"pruned={pruned_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
