"""`tests.fixture_server` -- offline gate coverage.

Loopback-only HTTP, no external network -- safe for the gate tier. Serves a
throwaway `tmp_path` directory rather than the real `evals/fixtures/`, so
this test does not depend on that directory's own contents.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from tests.fixture_server import run_fixture_server


def test_serves_real_file_content_from_the_given_directory(tmp_path: Path) -> None:
    (tmp_path / "page.txt").write_text(
        "hello from the fixture server", encoding="utf-8"
    )

    with run_fixture_server(tmp_path) as server_url:
        response = httpx.get(f"{server_url}/page.txt", timeout=5.0)

    assert response.status_code == 200
    assert response.text == "hello from the fixture server"


def test_a_missing_file_returns_404(tmp_path: Path) -> None:
    with run_fixture_server(tmp_path) as server_url:
        response = httpx.get(f"{server_url}/does-not-exist.txt", timeout=5.0)

    assert response.status_code == 404
