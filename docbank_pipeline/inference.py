"""
Stage 4 — Run DocLayout-YOLO inference on page images and crop detections.

Public API:
    run_yolo_inference(cfg, source, weights=...) -> list of detections per image

Each detection dict contains:
    {
      "image": Path,       # source page image
      "class_id": int,     # PIPELINE class id (0=text,1=formula,2=image)
      "class_name": str,   # PIPELINE class name (text/formula/image)
      "raw_class": str,    # original DocLayout class (e.g. 'isolate_formula')
      "bbox": [x1,y1,x2,y2],        # absolute pixel coords
      "bbox_norm": [cx,cy,w,h],     # YOLO-normalised
      "confidence": float,
      "crop_path": Path,            # saved crop on disk (or None if save=False)
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from .config import PipelineConfig
from .utils import detect_device, iter_image_files

log = logging.getLogger("docbank.inference")


def _resolve_weights(cfg: PipelineConfig, weights: str | Path | None) -> Path:
    # 1) explicit argument wins
    if weights:
        p = Path(weights)
        if p.is_file():
            return p
    # 2) a path configured on the cfg (add `layout_weights` to PipelineConfig)
    cfg_w = getattr(cfg, "layout_weights", None)
    if cfg_w and Path(cfg_w).is_file():
        return Path(cfg_w)
    # 3) fall back to a trained best.pt under runs/  -- WARNING: with the
    #    DocLayout-YOLO swap this would load your OLD DocBank model, which is
    #    NOT what you want. Prefer passing weights=<doclayout checkpoint>.
    candidates = list(cfg.runs_dir.glob("**/weights/best.pt"))
    if candidates:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        log.warning(
            "No explicit weights given; falling back to %s. If you intended to "
            "use DocLayout-YOLO, pass weights=<doclayout_yolo_*.pt> instead.",
            chosen,
        )
        return chosen
    raise FileNotFoundError(
        "No YOLO weights found. Pass weights=<doclayout checkpoint> "
        "or set cfg.layout_weights."
    )


def _iter_sources(source: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            return iter_image_files(p)
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
    imgsz: int = 1024,
) -> list[dict]:
    """Run DocLayout-YOLO over `source` (file/dir/list) and return detection
    dicts, with DocLayout classes remapped to the pipeline's
    {text, formula, image} via ``cfg.class_mapping``.

    If `save_crops` is True, each detection's crop is written under
    `cfg.crops_dir/<pipeline_class>/<imgstem>_<idx>.jpg`.
    """
    from doclayout_yolo import YOLOv10  # lazy import
    from PIL import Image  # lazy import

    weights_path = _resolve_weights(cfg, weights)
    log.info("Inference with weights: %s", weights_path)

    images = _iter_sources(source)
    if not images:
        log.warning("No input images found.")
        return []

    model = YOLOv10(str(weights_path))
    model_names: dict[int, str] = dict(model.names)  # DocLayout index -> name
    log.info("Model classes: %s", model_names)

    dev = device or detect_device()
    log.info("Inference device: %s, %d image(s)", dev, len(images))

    if save_crops:
        for cname in cfg.yolo_classes:
            (cfg.crops_dir / cname).mkdir(parents=True, exist_ok=True)

    detections: list[dict] = []
    results_iter: Iterable = model.predict(
        source=[str(p) for p in images],
        conf=conf,
        iou=iou,
        batch=12,
        device=dev,
        stream=True,
        verbose=False,
        imgsz=imgsz,
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
            raw_id = int(boxes.cls[i].item())
            raw_name = model_names.get(raw_id, str(raw_id))

            # --- DocLayout class -> pipeline class via cfg.class_mapping ---
            if raw_name not in cfg.class_mapping:
                log.warning(
                    "Unmapped layout class %r (id %d) — skipping. "
                    "Add it to cfg.class_mapping.", raw_name, raw_id,
                )
                continue
            mapped = cfg.class_mapping[raw_name]
            if mapped is None:
                continue  # intentionally dropped (e.g. 'abandon')
            cls_name = mapped
            cls_id = cfg.class_id.get(mapped, -1)
            # ---------------------------------------------------------------

            xyxy = boxes.xyxy[i].tolist()
            xywh = boxes.xywhn[i].tolist()
            conf_score = float(boxes.conf[i].item())

            det = {
                "image": img_path,
                "class_id": cls_id,
                "class_name": cls_name,
                "raw_class": raw_name,
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
