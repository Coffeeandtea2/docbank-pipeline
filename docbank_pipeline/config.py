"""
Central configuration for the DocBank layout-extraction pipeline.

Every module in `docbank_pipeline` accepts a `PipelineConfig` instance.
The defaults are safe for a local run; override them by:

  1. Constructing `PipelineConfig(...)` with explicit values, OR
  2. Setting environment variables (DATA_ROOT, MAX_PAGES, ...), OR
  3. Passing CLI flags to `python -m docbank_pipeline ...`.

No path inside the pipeline is hard-coded; everything routes through here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val).expanduser().resolve() if val else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val else default


def _env_opt_int(name: str, default: int | None) -> int | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return int(val)


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default


# DocBank's 12 fine-grained labels collapsed to the 3 classes we want to detect.
DEFAULT_CLASS_MAPPING: Mapping[str, str] = {
    "title": "text", "plain text": "text",
    "figure_caption": "text", "table_caption": "text",
    "table_footnote": "text", "formula_caption": "text",
    "isolate_formula": "formula",
    "figure": "image", "table": "image",   # or handle "table" specially
    "abandon": None,
    # "abstract": "text",
    # "author": "text",
    # "caption": "text",
    # "date": "text",
    # "footer": "text",
    # "list": "text",
    # "paragraph": "text",
    # "reference": "text",
    # "section": "text",
    # "title": "text",
    # "equation": "formula",
    # "figure": "image",
    # "table": "image",
}

DEFAULT_YOLO_CLASSES: tuple[str, ...] = ("text", "formula", "image")


@dataclass
class PipelineConfig:
    """All paths and knobs in one place. Resolve with `cfg.ensure_dirs()`."""

    # ------------------------------------------------------------------ paths
    data_root: Path = field(
        default_factory=lambda: _env_path("DATA_ROOT", Path.cwd() / "DocBank")
    )

    # ---------------------------------------------------------- HF dataset ids
    repo_id: str = "liminghao1630/DocBank"
    annotation_zip_name: str = "MSCOCO_Format_Annotation.zip"
    image_archive_basename: str = "DocBank_500K_ori_img.zip"
    num_image_parts_total: int = 10  # the dataset is split into .001 ... .010

    # -------------------------------------------------------- subset / limits
    # Number of `.zip.NNN` parts to actually download (1..10).
    dataset_parts: int = field(
        default_factory=lambda: _env_int("DATASET_PARTS", 1)
    )
    # Optional cap on the number of pages converted per split.
    max_pages: int | None = field(
        default_factory=lambda: _env_opt_int("MAX_PAGES", None)
    )

    # ------------------------------------------------------------- training
    train_val_split: float = field(
        default_factory=lambda: _env_float("TRAIN_VAL_SPLIT", 0.9)
    )
    num_workers: int = field(
        default_factory=lambda: _env_int("NUM_WORKERS", 4)
    )

    yolo_model: str = "yolo8n.pt"
    yolo_imgsz: int = 1024
    yolo_epochs: int = 100
    yolo_batch: int = 16
    yolo_patience: int = 10

    # Path to the DocLayout-YOLO checkpoint used at INFERENCE time on diverse
    # documents. If left None, run_yolo_inference falls back to a best.pt under
    # runs/ — which is your OLD DocBank model — so set this for the new detector.
    layout_weights: str | None = field(
        default_factory=lambda: os.environ.get("LAYOUT_WEIGHTS")
    )
    # -------------------------------------------------------------- classes
    class_mapping: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_MAPPING)
    )
    yolo_classes: tuple[str, ...] = DEFAULT_YOLO_CLASSES

    # --------------------------------------------------------- HF auth token
    hf_token: str | None = field(
        default_factory=lambda: os.environ.get("HF_TOKEN")
    )

    # -------------------------------------------------------- derived paths
    @property
    def raw_data_dir(self) -> Path:
        """Where the downloaded `.zip.NNN` archive parts live."""
        return self.data_root / "raw"

    @property
    def annotations_dir(self) -> Path:
        return self.data_root / "annotations"

    @property
    def extracted_dir(self) -> Path:
        """Where the `.jpg` pages live after extraction."""
        return self.data_root / "images"

    @property
    def yolo_dataset_dir(self) -> Path:
        return self.data_root / "yolo_dataset"

    @property
    def output_dir(self) -> Path:
        return self.data_root / "outputs"

    @property
    def crops_dir(self) -> Path:
        return self.output_dir / "crops"

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"

    @property
    def class_id(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.yolo_classes)}

    # ---------------------------------------------------------------- helpers
    def ensure_dirs(self) -> None:
        for d in [
            self.data_root,
            self.raw_data_dir,
            self.annotations_dir,
            self.extracted_dir,
            self.yolo_dataset_dir,
            self.output_dir,
            self.crops_dir,
            self.runs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def part_filename(self, idx: int) -> str:
        return f"{self.image_archive_basename}.{idx:03d}"

    def describe(self) -> str:
        return (
            f"DATA_ROOT       = {self.data_root}\n"
            f"RAW_DATA_DIR    = {self.raw_data_dir}\n"
            f"EXTRACTED_DIR   = {self.extracted_dir}\n"
            f"YOLO_DATASET_DIR= {self.yolo_dataset_dir}\n"
            f"OUTPUT_DIR      = {self.output_dir}\n"
            f"NUM_WORKERS     = {self.num_workers}\n"
            f"MAX_PAGES       = {self.max_pages}\n"
            f"DATASET_PARTS   = {self.dataset_parts}\n"
            f"TRAIN_VAL_SPLIT = {self.train_val_split}\n"
        )
