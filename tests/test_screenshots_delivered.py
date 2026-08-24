"""`screenshots/` must hold exactly the 4 Langfuse UI screenshots this
project delivers (`docs/task-hl12.md` requirement 5) -- a structural,
gate-tier check, since a missing or misnamed file would otherwise only be
caught by a human reading the folder.
"""

from __future__ import annotations

import paths

_EXPECTED_SCREENSHOTS = frozenset(
    {
        "evaluator-scores.png",
        "prompt-management.png",
        "trace-tree.png",
        "session.png",
    }
)


def test_screenshots_directory_has_exactly_the_four_required_files() -> None:
    screenshots_dir = paths.PROJECT_ROOT / "screenshots"
    actual = {p.name for p in screenshots_dir.glob("*.png")}
    assert actual == _EXPECTED_SCREENSHOTS
