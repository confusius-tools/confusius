"""Local release server for public dataset-fetcher tests."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import pytest


@pytest.fixture
def s3_requests() -> list[tuple[str, str, str | None, str | None, int]]:
    """Record S3 methods, paths, ranges, authorization, and connection ports."""
    return []


@pytest.fixture
def s3_failures() -> dict[str, list[int | Literal["disconnect"]]]:
    """Inject HTTP errors or interrupted streams for selected object paths."""
    return {}


@pytest.fixture
def s3_barrier() -> list[threading.Barrier]:
    """Optionally require concurrent GET requests before serving their bodies."""
    return []


@pytest.fixture
def release_server(tmp_path, monkeypatch, s3_requests, s3_failures, s3_barrier):
    """Serve a release collection through ordinary HTTP without external requests."""
    root = tmp_path / "server"
    root.mkdir()
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def record_s3_request(self):
            if self.path.startswith("/confusius-datasets/"):
                self.path = self.path.removeprefix("/confusius-datasets")
                s3_requests.append(
                    (
                        self.command,
                        self.path,
                        self.headers.get("Range"),
                        self.headers.get("Authorization"),
                        self.client_address[1],
                    )
                )

        def do_HEAD(self):
            self.record_s3_request()
            super().do_HEAD()

        def do_GET(self):
            self.record_s3_request()
            requests.append(self.path)
            actions = s3_failures.get(self.path, [])
            action = actions.pop(0) if actions else None
            if isinstance(action, int):
                self.send_error(action)
                return
            if s3_barrier and self.path != "/last_versions.conf" and not self.path.endswith("/manifest.json"):
                s3_barrier[0].wait(timeout=5)
            if self.headers.get("Range") or action == "disconnect":
                content = Path(self.translate_path(self.path)).read_bytes()
                start, end = 0, len(content) - 1
                range_header = self.headers.get("Range")
                if range_header:
                    first, last = range_header.removeprefix("bytes=").split("-")
                    start = int(first)
                    end = int(last) if last else len(content) - 1
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                if action == "disconnect":
                    self.wfile.write(content[:512 * 1024])
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                else:
                    self.wfile.write(content[start:end + 1])
                return
            super().do_GET()

        def log_message(self, format: str, *args: object) -> None:
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
