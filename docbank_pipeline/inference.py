"""
Stage 4 — Run YOLO inference on page images and crop detections.

Public API:
    run_yolo_inference(cfg, source, weights=...) -> list of detections per image

Each detection dict contains:
    {
      "image": Path,       # source page image
      "class_id": int,
      "class_name": str,
      "bbox": [x1,y1,x2,y2],        # absolute pixel coords
      "bbox_norm": [cx,cy,w,h],     # YOLO-normalised
      "confidence": float,
      "crop_path": Path,            # saved crop on disk (or None if save=False)
    }
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

from .config import PipelineConfig
from .utils import detect_device

log = logging.getLogger("docbank.inference")


def _resolve_weights(cfg: PipelineConfig, weights: str | Path | None) -> Path:
    if weights:
        p = Path(weights)
        if p.is_file():
            return p
    # Deployment override: point at a baked-in weights file via env var
    # (used by the Docker image / Render / HF Spaces, where there is no
    # `runs/.../best.pt` from a local training run).
    env_w = os.environ.get("YOLO_WEIGHTS") or os.environ.get("WEIGHTS")
    if env_w:
        ep = Path(env_w).expanduser()
        if ep.is_file():
            return ep
        log.warning("YOLO_WEIGHTS/WEIGHTS set to %s but file not found", ep)
    candidates = list(cfg.runs_dir.glob("**/weights/best.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(
        "No trained YOLO weights found. Pass weights=... or run `train_yolo`."
    )


def _iter_sources(source: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            return sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        if p.is_file():
            return [p]
        raise FileNotFoundError(f"Source not found: {p}")
    return [Path(s) for s in source]


def run_yolo_inference(
    cfg: PipelineConfig,
    source: str | Path | Sequence[str | Path],
    *,
    weights: str | Path | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    save_crops: bool = True,
    device: str | None = None,
) -> list[dict]:
    """Run YOLO over `source` (file/dir/list) and return detection dicts.

    If `save_crops` is True, each detection's crop is written under
    `cfg.crops_dir/<class_name>/<imgstem>_<idx>.jpg`.
    """
    from ultralytics import YOLO  # lazy import
    from PIL import Image  # lazy import

    weights_path = _resolve_weights(cfg, weights)
    log.info("Inference with weights: %s", weights_path)

    images = _iter_sources(source)
    if not images:
        log.warning("No input images found.")
        return []

    model = YOLO(str(weights_path))
    dev = device or detect_device()
    log.info("Inference device: %s, %d image(s)", dev, len(images))

    if save_crops:
        for cname in cfg.yolo_classes:
            (cfg.crops_dir / cname).mkdir(parents=True, exist_ok=True)

    detections: list[dict] = []
    # `stream=True` keeps memory bounded on large folders.
    results_iter: Iterable = model.predict(
        source=[str(p) for p in images],
        conf=conf,
        iou=iou,
        device=dev,
        stream=True,
        verbose=False,
    )
    for img_path, r in zip(images, results_iter):
        h, w = r.orig_shape
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        try:
            page_img = Image.open(img_path).convert("RGB") if save_crops else None
        except Exception as e:
            log.warning("Could not open %s for cropping: %s", img_path, e)
            page_img = None

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            xywh = boxes.xywhn[i].tolist()
            cls_id = int(boxes.cls[i].item())
            conf_score = float(boxes.conf[i].item())
            cls_name = (
                cfg.yolo_classes[cls_id] if 0 <= cls_id < len(cfg.yolo_classes)
                else f"cls_{cls_id}"
            )

            det = {
                "image": img_path,
                "class_id": cls_id,
                "class_name": cls_name,
                "bbox": [round(v, 2) for v in xyxy],
                "bbox_norm": [round(v, 6) for v in xywh],
                "confidence": round(conf_score, 4),
                "image_width": int(w),
                "image_height": int(h),
                "crop_path": None,
            }

            if save_crops and page_img is not None:
                x1, y1, x2, y2 = (max(0, int(v)) for v in xyxy)
                x2 = min(x2, w)
                y2 = min(y2, h)
                if x2 > x1 and y2 > y1:
                    crop = page_img.crop((x1, y1, x2, y2))
                    crop_path = (
                        cfg.crops_dir / cls_name
                        / f"{img_path.stem}_{i:03d}.jpg"
                    )
                    crop.save(crop_path, "JPEG", quality=92)
                    det["crop_path"] = crop_path
            detections.append(det)

    log.info("Got %d detection(s) across %d image(s)", len(detections), len(images))
    return detections
