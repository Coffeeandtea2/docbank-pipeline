"""
Stage 1 — Download + extract DocBank.

Public API:
    download_docbank_parts(cfg)   -> downloads N split-zip parts + annotation zip
    verify_downloads(cfg)         -> sanity check on file sizes / part count
    extract_archives(cfg)         -> annotations.zip + split-zip image archive

All operations are resumable. Each stage writes a `.<stage>.done` marker so a
second run is a no-op. Delete the marker to force a redo.

The split image archive is handled in two ways, in order of preference:
  1. 7-Zip executable (`7z x part.001 -odest`) — handles partial subsets too.
  2. Pure-Python concat-then-zipfile — requires ALL `cfg.num_image_parts_total`
     parts to be downloaded so the central directory is intact.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from tqdm.auto import tqdm

from .config import PipelineConfig
from .utils import (
    human_bytes,
    mark_stage_done,
    stage_done,
    which_seven_zip,
)

log = logging.getLogger("docbank.download")


# ----------------------------------------------------------------- download

def _hf_download(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: str | None,
) -> Path:
    """Wrapper around huggingface_hub's `hf_hub_download` with resume on."""
    from huggingface_hub import hf_hub_download  # lazy import

    out = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
        # NOTE: `resume_download` is on by default and `force_download=False`
        # makes hf_hub_download skip when the SHA-tagged file already exists.
        token=token,
    )
    return Path(out)


def download_docbank_parts(cfg: PipelineConfig) -> dict:
    """Download the annotation zip and the first `cfg.dataset_parts` image parts.

    Already-present files are skipped automatically by the HF cache layer."""
    cfg.ensure_dirs()

    n = cfg.dataset_parts
    if not 1 <= n <= cfg.num_image_parts_total:
        raise ValueError(
            f"dataset_parts must be in [1, {cfg.num_image_parts_total}], got {n}"
        )

    raw = cfg.raw_data_dir
    log.info("Downloading annotation archive into %s", raw)
    ann_path = _hf_download(
        cfg.repo_id, cfg.annotation_zip_name, raw, cfg.hf_token
    )

    image_parts: list[Path] = []
    log.info("Downloading %d image archive part(s) into %s", n, raw)
    for i in tqdm(range(1, n + 1), desc="image parts", unit="part"):
        fname = cfg.part_filename(i)
        p = _hf_download(cfg.repo_id, fname, raw, cfg.hf_token)
        image_parts.append(p)

    mark_stage_done(
        raw,
        "download",
        payload=f"annotation={ann_path.name}\nparts={n}\n",
    )
    return {
        "annotation_zip": ann_path,
        "image_parts": image_parts,
    }


# ----------------------------------------------------------------- verify

def verify_downloads(cfg: PipelineConfig) -> dict:
    """Quick file-size sanity check. Does NOT validate ZIP structure."""
    raw = cfg.raw_data_dir
    report: dict = {"annotation_zip": None, "parts": [], "missing": []}

    ann = raw / cfg.annotation_zip_name
    if ann.is_file():
        report["annotation_zip"] = (ann, ann.stat().st_size)
    else:
        report["missing"].append(cfg.annotation_zip_name)

    for i in range(1, cfg.dataset_parts + 1):
        p = raw / cfg.part_filename(i)
        if p.is_file():
            report["parts"].append((p, p.stat().st_size))
        else:
            report["missing"].append(p.name)

    log.info(
        "Verify: annotation=%s, parts=%d, missing=%d",
        "ok" if report["annotation_zip"] else "MISSING",
        len(report["parts"]),
        len(report["missing"]),
    )
    for path, size in report["parts"]:
        log.debug("  %s : %s", path.name, human_bytes(size))
    if report["missing"]:
        for m in report["missing"]:
            log.warning("  MISSING: %s", m)
    return report


# ----------------------------------------------------------------- extract

def _extract_zip(zip_path: Path, dest: Path) -> int:
    """Extract a regular .zip with the stdlib, returning extracted-file count."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        n = 0
        for m in tqdm(members, desc=f"extract {zip_path.name}", unit="file"):
            zf.extract(m, dest)
            n += 1
    return n


def _extract_with_7z(part001: Path, dest: Path, exe: str) -> int:
    """Use the `7z` CLI to extract a multi-volume zip starting from `.001`.

    7z exit codes:
        0 = OK, 1 = non-fatal warning, 2 = fatal error.

    For a multi-part archive where the user only downloaded a subset of the
    parts, 7z will report rc=2 with `Unexpected end of archive` because the
    last file is truncated and the central directory lives in the final part.
    That is EXPECTED for partial extraction — every other file in the
    available parts has already been written to disk by the time 7z gives up.
    So we only raise if NOTHING was extracted.
    """
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "x", str(part001), f"-o{dest}", "-y", "-aos"]
    log.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    extracted = sum(1 for _ in dest.rglob("*.jpg"))

    if proc.returncode == 0:
        return extracted
    if proc.returncode == 1:
        log.warning("7z reported non-fatal warnings. Output may be partial.")
        return extracted
    # rc >= 2: fatal in 7z's eyes, but acceptable for partial archives so long
    # as we got SOMETHING out.
    if extracted > 0:
        log.warning(
            "7z exited with rc=%d (likely partial archive: missing parts or "
            "truncated last file). Continuing with %d extracted image(s).",
            proc.returncode, extracted,
        )
        return extracted
    raise RuntimeError(
        f"7z failed (rc={proc.returncode}) and produced no images:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def _concat_parts_to_zip(parts: list[Path], out_zip: Path) -> None:
    """Pure-Python combine: concatenate `.001..NNN` byte-for-byte."""
    log.info("Concatenating %d part(s) -> %s", len(parts), out_zip)
    with open(out_zip, "wb") as out_f:
        for part in tqdm(parts, desc="concat", unit="part"):
            with open(part, "rb") as p:
                shutil.copyfileobj(p, out_f, length=4 * 1024 * 1024)


def extract_archives(cfg: PipelineConfig, *, force: bool = False) -> dict:
    """Extract annotations.zip and the multi-part image archive.

    Skipped automatically if the `extract.done` marker is present, unless
    `force=True`.
    """
    cfg.ensure_dirs()
    raw = cfg.raw_data_dir
    out_imgs = cfg.extracted_dir
    out_anns = cfg.annotations_dir

    if not force and stage_done(raw, "extract"):
        log.info("Extraction already done (marker present). Skipping.")
        return {"skipped": True}

    # ---- annotations -------------------------------------------------------
    ann_zip = raw / cfg.annotation_zip_name
    if not ann_zip.is_file():
        raise FileNotFoundError(
            f"Annotation zip not found at {ann_zip}. "
            "Run `download_docbank_parts` first."
        )
    ann_marker = out_anns / ".extract.done"
    if force or not ann_marker.is_file():
        log.info("Extracting annotations -> %s", out_anns)
        _extract_zip(ann_zip, out_anns)
        mark_stage_done(out_anns, "extract")
    else:
        log.info("Annotations already extracted. Skipping.")

    # ---- images ------------------------------------------------------------
    parts = sorted(
        p for p in raw.glob(f"{cfg.image_archive_basename}.*")
        if p.suffix[1:].isdigit()
    )
    if not parts:
        raise FileNotFoundError(
            f"No image archive parts found in {raw}. "
            "Run `download_docbank_parts` first."
        )

    seven_zip = which_seven_zip()
    if seven_zip:
        log.info("Using 7-Zip at %s for image extraction.", seven_zip)
        n_extracted = _extract_with_7z(parts[0], out_imgs, seven_zip)
    elif len(parts) == cfg.num_image_parts_total:
        # We have ALL parts -> we can safely concat and use stdlib zipfile.
        combined = raw / cfg.image_archive_basename
        if not combined.is_file() or combined.stat().st_size <= parts[0].stat().st_size:
            _concat_parts_to_zip(parts, combined)
        log.info("Extracting combined zip with Python's zipfile module.")
        n_extracted = _extract_zip(combined, out_imgs)
    else:
        raise RuntimeError(
            "7-Zip executable not found AND only a subset of parts is "
            "downloaded. Either:\n"
            "  - install 7-Zip (https://www.7-zip.org/, "
            "`brew install p7zip`, or `apt install p7zip-full`), or\n"
            "  - set cfg.dataset_parts = "
            f"{cfg.num_image_parts_total} to download every part."
        )

    log.info("Extracted %d image file(s) into %s", n_extracted, out_imgs)
    mark_stage_done(
        raw, "extract",
        payload=f"images={n_extracted}\nparts={len(parts)}\n7z={bool(seven_zip)}\n",
    )
    return {
        "skipped": False,
        "image_count": n_extracted,
        "used_7z": bool(seven_zip),
    }
