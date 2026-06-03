"""
Shared helpers: logging setup, GPU detection, link-or-copy, marker files.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Collection

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


class _NoopTqdm:
    """Small fallback so lightweight helpers work before tqdm is installed."""

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable or ())

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover - exercised only in minimal envs
    tqdm = _NoopTqdm


def _normalise_suffixes(suffixes: Collection[str]) -> frozenset[str]:
    return frozenset(s.lower() for s in suffixes)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger once. Safe to call repeatedly."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        root.addHandler(handler)
    root.setLevel(level)
    return logging.getLogger("docbank")


def detect_device() -> str:
    """Return 'cuda', 'mps' or 'cpu' based on what is actually usable."""
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link `src` to `dst` when possible (fast, no disk doubling),
    otherwise fall back to a copy. Idempotent: skips if dst already exists."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def is_image_file(
    path: str | Path,
    *,
    suffixes: Collection[str] = IMAGE_SUFFIXES,
) -> bool:
    """Return True for regular files with a supported image suffix."""
    p = Path(path)
    return p.is_file() and p.suffix.lower() in _normalise_suffixes(suffixes)


def iter_image_files(
    folder: str | Path,
    *,
    suffixes: Collection[str] = IMAGE_SUFFIXES,
    recursive: bool = False,
) -> list[Path]:
    """Return image files in stable order."""
    root = Path(folder)
    if not root.is_dir():
        return []
    supported = _normalise_suffixes(suffixes)
    paths = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in supported)


# --- marker / sentinel files ------------------------------------------------
#
# Every long-running stage drops a `.done` file with optional payload. Subsequent
# runs check for this and skip the work. To force a re-run, delete the marker.

def marker_path(folder: Path, name: str) -> Path:
    return folder / f".{name}.done"


def stage_done(folder: Path, name: str) -> bool:
    return marker_path(folder, name).is_file()


def mark_stage_done(folder: Path, name: str, payload: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    marker_path(folder, name).write_text(payload, encoding="utf-8")


def clear_stage(folder: Path, name: str) -> None:
    marker_path(folder, name).unlink(missing_ok=True)


@contextmanager
def cwd(path: Path):
    """Temporarily chdir, restoring previous cwd on exit."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev)


def which_seven_zip() -> str | None:
    """Locate a usable 7-Zip executable, returning its name or None."""
    for candidate in ("7z", "7z.exe", "7za", "7za.exe", "7zz"):
        if shutil.which(candidate):
            return candidate
    # Common Windows install locations not always on PATH.
    for candidate in (
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}"
        f /= 1024
    return f"{n} B"
