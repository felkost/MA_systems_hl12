"""`config.py` must put the cache-location variables into `os.environ` itself.

Stage 0 measured that pydantic-settings parses `.env` into `Settings` and
exports nothing: Hugging Face and pip read `os.environ` and nothing else, so
without an explicit write the model cache lands wherever the ambient
environment says -- which on a fresh machine can be `C:\\Users\\...`, the one
place CLAUDE.md's Forbidden list bans it from.

This runs the check in a **child process** with a deliberately hostile
ambient environment, not in-process: stage 0's own finding was that the rule
looked satisfied only because this machine's ambient `HF_HOME` already
happened to point off the system drive. Asserting from the current process
would repeat exactly that mistake.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CHILD_SCRIPT = (
    "import config\n"
    "import os\n"
    "print(os.environ['HF_HOME'])\n"
    "print(os.environ['PIP_CACHE_DIR'])\n"
    "print(os.environ['DEEPEVAL_HOME'])\n"
)


def test_config_exports_cache_paths_to_environ(tmp_path: Path) -> None:
    # Start from the real environment (Windows subprocess startup needs
    # PATH/SYSTEMROOT/etc.) and override only the two variables under test,
    # plus PYTHONPATH so a bare `import config` resolves from the child's
    # unrelated `tmp_path` cwd.
    hostile_env = dict(os.environ)
    hostile_env["PYTHONPATH"] = str(PROJECT_ROOT)
    hostile_env["HF_HOME"] = "C:\\Users\\hostile\\hf-cache"
    hostile_env["PIP_CACHE_DIR"] = "C:\\Users\\hostile\\pip-cache"
    hostile_env["DEEPEVAL_HOME"] = "C:\\Users\\hostile\\.deepeval"

    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        cwd=tmp_path,  # no .env here -- nothing this test relies on can leak in
        env=hostile_env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    hf_home, pip_cache_dir, deepeval_home = result.stdout.strip().splitlines()
    assert hf_home == str(PROJECT_ROOT / ".cache" / "hf")
    assert pip_cache_dir == str(PROJECT_ROOT / ".cache" / "pip")
    assert deepeval_home == str(PROJECT_ROOT / ".cache" / "deepeval")
    assert "hostile" not in hf_home
    assert "hostile" not in pip_cache_dir
    assert "hostile" not in deepeval_home


# -- Stage 9e, D9e.1 -- the judge's per-task timeout, a DELIBERATE exception
# to the pattern above. Two tests asserting OPPOSITE things, not the same
# pattern the three cache variables above share: those three exist
# specifically so that an ambient value can never outrank the project's own
# (`_CacheSettings.settings_customise_sources` drops the ambient-env
# source). A timeout is different -- a shell override is a legitimate thing
# for a user to want, so `deepeval_per_task_timeout_seconds` is a plain
# `Settings` field with ordinary pydantic-settings precedence, where the
# ambient environment DOES win over the project default. This is the only
# field in this project's `Settings` where that direction is intentional.

_TIMEOUT_FIELD_CHILD_SCRIPT = (
    "import config\n"
    "settings = config.load_settings()\n"
    "print(settings.deepeval_per_task_timeout_seconds)\n"
)

_TIMEOUT_EXPORT_CHILD_SCRIPT = (
    "import config\n"
    "from deepeval.config.settings import get_settings\n"
    "from deepeval.utils import get_per_task_timeout_seconds\n"
    "settings = config.load_settings()\n"
    "config.export_deepeval_timeout_override(settings)\n"
    "print(get_settings().DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE)\n"
    "print(get_per_task_timeout_seconds())\n"
)


def _hostile_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["OPENROUTER_API_KEY"] = "test-key"
    return env


def test_deepeval_timeout_field_lets_an_ambient_override_win(tmp_path: Path) -> None:
    """Unlike HF_HOME/PIP_CACHE_DIR/DEEPEVAL_HOME above, the ambient
    environment WINS here on purpose -- ordinary pydantic-settings
    precedence (env > .env > default), no source-list surgery."""
    env = _hostile_env(tmp_path)
    env["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS"] = "123"

    result = subprocess.run(
        [sys.executable, "-c", _TIMEOUT_FIELD_CHILD_SCRIPT],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "123.0"


def test_export_deepeval_timeout_override_reaches_deepevals_own_settings(
    tmp_path: Path,
) -> None:
    """The artefact D9e.1 names: the exported key is DeepEval's own
    suffixed override variable, and DeepEval's effective per-task timeout
    (not just the raw field) reflects it."""
    env = _hostile_env(tmp_path)
    env["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS"] = "123"

    result = subprocess.run(
        [sys.executable, "-c", _TIMEOUT_EXPORT_CHILD_SCRIPT],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    override, effective = result.stdout.strip().splitlines()
    assert override == "123.0"
    assert effective == "123.0"
