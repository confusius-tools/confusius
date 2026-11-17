"""Build artifacts must contain usable skills, including future nested resources."""

import runpy
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.slow
def test_skills_survive_sdist_and_wheel(tmp_path: Path) -> None:
    """Build from an sdist and install resources without accessing the checkout."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("The project's uv build tool is required for packaging checks.")
    repository = Path(__file__).resolve().parents[2]
    project = tmp_path / "source"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repository / name, project / name)
    for name in ("src", "licenses"):
        shutil.copytree(
            repository / name,
            project / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    # Future nested resources must ship without additions to a packaging manifest.
    future = (
        project / "src/confusius/_skills/confusius-load-data/references/nested/new.txt"
    )
    future.parent.mkdir()
    future.write_text("future resource", encoding="utf-8")
    artifacts = tmp_path / "dist"
    subprocess.run(
        [uv, "build", str(project), "--sdist", "--out-dir", str(artifacts)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (sdist,) = artifacts.glob("*.tar.gz")
    subprocess.run(
        [uv, "build", str(sdist), "--wheel", "--out-dir", str(artifacts)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (wheel,) = artifacts.glob("*.whl")
    expected = {
        file.relative_to(project / "src").as_posix(): file.read_bytes()
        for file in (project / "src/confusius/_skills").rglob("*")
        if file.is_file()
    }
    with ZipFile(wheel) as archive:
        actual = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("confusius/_skills/") and not name.endswith("/")
        }
        assert actual == expected
        assert not any("skills/release/" in name for name in archive.namelist())

    # A zip-backed package exercises Traversable resources, not just pathlib paths.
    # -I removes checkout/PYTHONPATH influence; normal environment dependencies remain.
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import confusius
from confusius._cli import main
assert str(Path(sys.argv[1])) in confusius.__file__
sys.argv = ['confusius', 'skills', 'list']
main()
sys.argv = ['confusius', 'skills', 'install']
main()
"""
    installed = tmp_path / "installed"
    installed.mkdir()
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(wheel)],
        cwd=installed,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "Available Agent Skills" in result.stdout
    assert "confusius-load-data" in result.stdout
    destination = installed / ".agents/skills/confusius-load-data"
    for name, content in expected.items():
        relative = Path(name).relative_to("confusius/_skills/confusius-load-data")
        assert (destination / relative).read_bytes() == content


def test_bundled_examples_run() -> None:
    """The bundled custom-data examples execute offline."""
    script = resources.files("confusius").joinpath(
        "_skills", "confusius-load-data", "scripts", "examples.py"
    )
    with resources.as_file(script) as path:
        runpy.run_path(str(path), run_name="__main__")
