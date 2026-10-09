"""Local release server for public dataset-fetcher tests."""

from __future__ import annotations

import hashlib
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def release_server(tmp_path, monkeypatch):
    """Serve a release collection through ordinary HTTP without external requests."""
    root = tmp_path / "server"
    root.mkdir()
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            super().do_GET()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(root))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        "confusius.datasets._s3.BASE_URL",
        f"http://127.0.0.1:{server.server_port}",
    )
    try:
        yield root, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def publish_release(release_server):
    """Publish local immutable files and their SHA-256 release inventory."""
    root, _ = release_server

    def publish(category, name, version, files):
        destination = root / category / name / version
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for relative, content in files.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            manifest[relative] = {
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        (destination / "manifest.json").write_text(json.dumps(manifest))
        (root / "last_versions.conf").write_text(f"[{category}]\n{name} = {version}\n")
        return destination

    return publish
