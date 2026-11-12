from pathlib import Path

import pytest

from tools.deduplicate_docs_assets import deduplicate_assets


def test_deduplicate_assets_rewrites_references(tmp_path: Path) -> None:
    first_asset = tmp_path / "dev" / "images" / "example.png"
    second_asset = tmp_path / "0.8.0" / "images" / "example.png"
    first_page = tmp_path / "dev" / "index.html"
    second_page = tmp_path / "0.8.0" / "index.html"
    for path in (first_asset, second_asset, first_page, second_page):
        path.parent.mkdir(parents=True, exist_ok=True)
    first_asset.write_bytes(b"identical image")
    second_asset.write_bytes(b"identical image")
    first_page.write_text('<img src="images/example.png">', encoding="utf-8")
    second_page.write_text('<img src="images/example.png">', encoding="utf-8")

    result = deduplicate_assets(tmp_path)

    assert result == (2, 2, 0)
    shared_assets = list((tmp_path / "shared-assets").iterdir())
    assert len(shared_assets) == 1
    assert not first_asset.exists()
    assert not second_asset.exists()
    assert shared_assets[0].name in first_page.read_text(encoding="utf-8")
    assert shared_assets[0].name in second_page.read_text(encoding="utf-8")
    assert deduplicate_assets(tmp_path) == (0, 0, 0)


@pytest.mark.parametrize(
    "suffix", [".avif", ".bmp", ".jpeg", ".jpg", ".tif", ".tiff", ".webp"]
)
def test_deduplicate_assets_supports_common_image_formats(
    tmp_path: Path, suffix: str
) -> None:
    first_asset = tmp_path / "dev" / f"example{suffix}"
    second_asset = tmp_path / "0.8.0" / f"example{suffix}"
    first_page = tmp_path / "dev" / "index.html"
    second_page = tmp_path / "0.8.0" / "index.html"
    for path in (first_asset, second_asset, first_page, second_page):
        path.parent.mkdir(parents=True, exist_ok=True)
    first_asset.write_bytes(b"identical image")
    second_asset.write_bytes(b"identical image")
    first_page.write_text(f'<img src="example{suffix}">', encoding="utf-8")
    second_page.write_text(f'<img src="example{suffix}">', encoding="utf-8")

    assert deduplicate_assets(tmp_path) == (2, 2, 0)
    assert len(list((tmp_path / "shared-assets").iterdir())) == 1
    assert not first_asset.exists()
    assert not second_asset.exists()


def test_deduplicate_assets_does_not_rewrite_external_urls(tmp_path: Path) -> None:
    first_asset = tmp_path / "dev" / "example.png"
    second_asset = tmp_path / "0.8.0" / "example.png"
    page = tmp_path / "dev" / "index.html"
    for path in (first_asset, second_asset, page):
        path.parent.mkdir(parents=True, exist_ok=True)
    first_asset.write_bytes(b"identical image")
    second_asset.write_bytes(b"identical image")
    page.write_text('<img src="https://example.com/example.png">', encoding="utf-8")

    assert deduplicate_assets(tmp_path) == (0, 0, 1)
    assert first_asset.exists()
    assert second_asset.exists()


def test_deduplicate_assets_does_not_remove_unreferenced_files(tmp_path: Path) -> None:
    first_asset = tmp_path / "dev" / "images" / "example.png"
    second_asset = tmp_path / "0.8.0" / "images" / "example.png"
    for path in (first_asset, second_asset):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"identical image")

    assert deduplicate_assets(tmp_path) == (0, 0, 1)
    assert first_asset.exists()
    assert second_asset.exists()
    assert not (tmp_path / "shared-assets").exists()
