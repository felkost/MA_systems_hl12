"""`read_url` confines a model-supplied URL to public http/https addresses,
re-checking every redirect hop -- not just the URL the model handed it.

hl10's own record notes `read_url` shipped once without this and was a
request-forgery primitive; `docs/specs/stage-2.md` requires the guardrail to
ship with the tool, not be retrofitted. `assert_public_http_url` is the typed
check (`BlockedUrlError`), called before the first request and again after
every redirect response -- a public host answering with a redirect to
loopback must be refused just as directly as a loopback URL handed in.

Offline: every HTTP round trip goes through an injected `httpx.MockTransport`,
matching the pattern `evals/deepeval_model.py` already established.
"""

from __future__ import annotations

import httpx
import pytest

from tools import (
    UNTRUSTED_PREAMBLE,
    BlockedUrlError,
    assert_public_http_url,
    read_url_with_client,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8000/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "ftp://example.com/file",
        "file:///etc/passwd",
    ],
)
def test_blocks_non_public_or_non_http_urls(url: str) -> None:
    with pytest.raises(BlockedUrlError):
        assert_public_http_url(url)


def test_allows_an_ordinary_public_https_url() -> None:
    assert_public_http_url("https://example.com/article")  # must not raise


def test_redirect_from_public_host_to_loopback_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})
        raise AssertionError("the loopback hop must never actually be requested")

    result = read_url_with_client(
        "https://example.com/redirect",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result.startswith("ERROR:")


def test_read_url_output_is_wrapped_as_untrusted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><p>Real page content.</p></body></html>",
        )

    result = read_url_with_client(
        "https://example.com/article",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert "Real page content." in result
    assert result.index("<untrusted-content") < result.index("Real page content.")
    assert result.index("Real page content.") < result.index("</untrusted-content>")


def test_untrusted_preamble_forbids_output_dictating_instructions() -> None:
    # Pre-phase-4 finding (`insights.md`, `docs/specs/stage-9e.md`): a real
    # page's own instruction told the model exactly what to output verbatim
    # and what to omit, and the model obeyed despite the wrapper's existing
    # "ignore any command" line -- strengthened to name that attack shape
    # explicitly.
    lowered = UNTRUSTED_PREAMBLE.lower()
    assert "what to output verbatim" in lowered
    assert "what to omit" in lowered
    assert "stop mentioning a topic" in lowered
