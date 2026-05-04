"""
Stage 3 — YOLO training and validation.

Public API:
    train_yolo(cfg, ...)     -> trains a YOLOv8 model on cfg.yolo_dataset_dir
    validate_yolo(cfg, ...)  -> runs val on the best checkpoint

Both functions auto-detect CUDA / MPS / CPU. Training is light-weight by
default so it finishes on a laptop; bump epochs/imgsz/batch via cfg or kwargs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .utils import detect_device

log = logging.getLogger("docbank.train")


def train_yolo(
    cfg: PipelineConfig,
    *,
    model: str | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    run_name: str = "yolov8_docbank",
    extra: dict[str, Any] | None = None,
):
    """Train a YOLOv8 model on the converted dataset.

    Returns the Ultralytics `results` object (which has `.save_dir`).
    """
    from ultralytics import YOLO  # lazy import

    yaml_path = cfg.yolo_dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"{yaml_path} not found. Run dataset conversion first "
            "(`split_dataset` or CLI `convert`)."
        )

    dev = device or detect_device()
    log.info("Training device resolved to: %s", dev)

    weights = model or cfg.yolo_model
    log.info("Loading base weights: %s", weights)
    yolo = YOLO(weights)

    train_kwargs: dict[str, Any] = dict(
        data=str(yaml_path),
        epochs=epochs or cfg.yolo_epochs,
        imgsz=imgsz or cfg.yolo_imgsz,
        batch=batch or cfg.yolo_batch,
        workers=cfg.num_workers,
        patience=cfg.yolo_patience,
        device=dev,
        project=str(cfg.runs_dir),
        name=run_name,
        save=True,
        exist_ok=True,
        # Reasonable defaults for document-layout images.
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.2,
        degrees=0.0, translate=0.05, scale=0.2,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,  # never flip text
        mosaic=0.5, mixup=0.0,
    )
    if extra:
        train_kwargs.update(extra)

    log.info("Starting training: %s", train_kwargs)
    results = yolo.train(**train_kwargs)
    log.info("Training complete. Best weights: %s/weights/best.pt", results.save_dir)
    return results


def validate_yolo(
    cfg: PipelineConfig,
    *,
    weights: str | Path | None = None,
    split: str = "val",
    device: str | None = None,
):
    """Run validation. By default uses the most recent run's `best.pt`."""
    from ultralytics import YOLO  # lazy import

    if weights is None:
        weights = _find_latest_best(cfg)
    if weights is None or not Path(weights).is_file():
        raise FileNotFoundError(
            "Could not locate a trained checkpoint. Pass `weights=...` "
            "or run `train_yolo` first."
        )

    log.info("Validating with weights: %s", weights)
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(cfg.yolo_dataset_dir / "data.yaml"),
        split=split,
        device=device or detect_device(),
        project=str(cfg.runs_dir),
        name="val",
        exist_ok=True,
    )
    log.info("mAP50=%.3f  mAP50-95=%.3f", metrics.box.map50, metrics.box.map)
    return metrics


def _find_latest_best(cfg: PipelineConfig) -> Path | None:
    runs = list(cfg.runs_dir.glob("**/weights/best.pt"))
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)
