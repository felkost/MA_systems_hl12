"""Filesystem locations, anchored to the checkout rather than to the caller.

`Settings` stores its paths as relative strings (`data`, `index`,
`.cache/models`) so that no machine-specific path reaches `.env`. Resolving
them here means `python ingest.py` writes the same index from any working
directory, and the model cache stays inside the checkout.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve(path: str | Path) -> Path:
    """Anchor a configured path to the project root.

    Parameters
    ----------
    path : str or Path
        Value of a `Settings` path field. An absolute path is returned
        unchanged, so tests can point at a temporary directory.

    Returns
    -------
    Path
        Absolute path.

    Examples
    --------
    >>> resolve("index") == PROJECT_ROOT / "index"
    True
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def manifest_path(index_dir: str | Path) -> Path:
    """Locate the manifest describing how the index was built.

    See Also
    --------
    corpus_path : The chunk dump BM25 reads.
    """
    return resolve(index_dir) / "manifest.json"


def corpus_path(index_dir: str | Path) -> Path:
    """Locate the chunk dump.

    Notes
    -----
    BM25 scores raw text and cannot read Chroma's vectors, so the chunks are
    also stored as plain JSON.
    """
    return resolve(index_dir) / "chunks.json"


def log_path(service: str, log_dir: str | Path = "logs") -> Path:
    """Locate the rotating log file for one long-lived process.

    Parameters
    ----------
    service : str
        Process name, e.g. `"supervisor"` -- the file is `<service>.log`.
    log_dir : str or Path, default "logs"
        Explicit so a test can redirect it under `tmp_path`, the same shape
        `manifest_path`/`corpus_path` already use for `index_dir`.
    """
    return resolve(log_dir) / f"{service}.log"


def run_dir(run_id: str, runs_dir: str | Path = "runs") -> Path:
    """Directory for one run's artefacts, created if missing.

    Parameters
    ----------
    run_id : str
        Fresh per question/turn (stage 5, `docs/specs/stage-5.md` D5.7) --
        not the REPL session's `thread_id`.
    runs_dir : str or Path, default "runs"
        Explicit so a test can redirect it under `tmp_path`, the same shape
        `log_path`'s `log_dir` already uses.
    """
    result = resolve(runs_dir) / run_id
    result.mkdir(parents=True, exist_ok=True)
    return result


def span_dump_path(run_id: str, runs_dir: str | Path = "runs") -> Path:
    """Locate the offline span dump for one run.

    See Also
    --------
    run_dir : Creates the parent directory this path lives in.
    """
    return run_dir(run_id, runs_dir) / "spans.json"


def checkpoint_path(db: str | Path) -> Path:
    """Locate the checkpoint database, creating its parent directory if
    missing.

    Has no caller: stage 4 checkpoints in memory, so nothing in the shipped
    system is crash-safe. Kept for the stage that adds a durable backend.

    Parameters
    ----------
    db : str or Path
        Value of `Settings.checkpoint_db`.

    Returns
    -------
    Path

    Notes
    -----
    `AsyncSqliteSaver.from_conn_string` does not create missing parent
    directories itself -- a fresh checkout's gitignored `runtime/` must exist
    before the first connection, or the very first session fails.
    """
    result = resolve(db)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result
