"""A minimal local HTTP server exposing `evals/fixtures/` (stage 9a,
`docs/specs/stage-9a.md` D9a.1).

Not a test module (no `test_*` name, so pytest never collects it) -- the
one golden-dataset case that needs a real, fetchable URL for `read_url`
(`adversarial-indirect-injection`) cannot be satisfied by a fixture file
alone; `read_url` fetches over HTTP(S), never from disk. No prior stage
needed a live HTTP endpoint under this project's own control, because
stages 7-8 only ever exercised a subset of the 15 golden-dataset cases that
happened to exclude every fixture-carrying one.

Bound to `127.0.0.1` on an OS-assigned free port and started on a daemon
thread, so it never survives past the pytest process and never listens on
anything but loopback.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator


def start_fixture_server(
    directory: Path,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Serve `directory` over HTTP on `127.0.0.1:<free port>`.

    Parameters
    ----------
    directory : Path
        `evals/fixtures/`, in production use.

    Returns
    -------
    tuple of (ThreadingHTTPServer, threading.Thread)
        The caller owns both: call `server.shutdown()` at teardown, after
        which the daemon thread exits on its own.
    """
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def base_url(server: ThreadingHTTPServer) -> str:
    """The server's own reachable base URL, e.g. `http://127.0.0.1:54321`.

    Always loopback -- `start_fixture_server` only ever binds to
    `("127.0.0.1", 0)` -- so only the OS-assigned port varies.
    """
    port = server.server_address[1]
    return f"http://127.0.0.1:{port}"


@contextmanager
def run_fixture_server(directory: Path) -> Iterator[str]:
    """Context manager: start, yield the base URL, stop.

    A thin wrapper so `tests/conftest.py`'s `fixture_base_url` fixture can
    be a one-line `with run_fixture_server(...) as url: yield url`.
    """
    server, _thread = start_fixture_server(directory)
    try:
        yield base_url(server)
    finally:
        server.shutdown()
        server.server_close()
