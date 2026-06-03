"""
Stage 2 — Convert DocBank's COCO-format annotations to YOLO format.

Public API:
    convert_docbank_to_yolo(cfg, split=...)  -> writes labels + image links
    split_dataset(cfg)                       -> all splits
    create_yolo_yaml(cfg)                    -> writes data.yaml

Key fix vs the original notebook
--------------------------------
The original code located the source image with `Path(image_dir).rglob(filename)`
*inside* the per-image loop, i.e. an O(N) directory walk per annotation. That
made conversion run at ~27 s/page on a regular dataset.

Here we build a single `{filename: full_path}` map upfront (one walk), so the
inner loop is O(1) per image. This is the change that takes conversion from
hours-and-hangs to minutes.

Resumability: if a label `.txt` already exists for a given image, the image is
skipped. Delete `cfg.yolo_dataset_dir` to force a clean reconvert.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

from .config import PipelineConfig
from .utils import iter_image_files, link_or_copy, mark_stage_done, stage_done, tqdm

log = logging.getLogger("docbank.convert")

# DocBank's COCO json file names (relative to cfg.annotations_dir).
COCO_FILES: dict[str, str] = {
    "train": "500K_train.json",
    "val": "500K_valid.json",
    "test": "500K_test.json",
}


def _build_image_index(image_dir: Path) -> dict[str, Path]:
    """Walk `image_dir` once and return {basename -> absolute path}.

    DocBank's annotations refer to images by basename only, but the archives
    extract them under a `DocBank_500K_ori_img/` subdir, so we need this
    mapping.
    """
    log.info("Indexing images under %s ...", image_dir)
    index: dict[str, Path] = {}
    files = iter_image_files(image_dir, suffixes={".jpg"}, recursive=True)
    for f in tqdm(files, desc="index", unit="img"):
        # Last write wins; basenames are unique in DocBank.
        index[f.name] = f
    log.info("Indexed %d image(s).", len(index))
    return index


def _yolo_line(
    bbox: list[float],
    img_w: int,
    img_h: int,
    class_id: int,
) -> str:
    x_min, y_min, w, h = bbox
    cx = (x_min + w / 2) / img_w
    cy = (y_min + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to valid YOLO range [0, 1].
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def convert_docbank_to_yolo(
    cfg: PipelineConfig,
    split: str,
    *,
    image_index: Mapping[str, Path] | None = None,
    max_pages: int | None = None,
) -> dict:
    """Convert one split (train/val/test) of DocBank COCO -> YOLO.

    Returns a small report dict {processed, skipped_missing, skipped_existing}.
    """
    if split not in COCO_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(COCO_FILES)}")

    coco_json = cfg.annotations_dir / COCO_FILES[split]
    if not coco_json.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {coco_json}. "
            "Did `extract_archives` finish?"
        )

    img_out = cfg.yolo_dataset_dir / "images" / split
    lbl_out = cfg.yolo_dataset_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", coco_json.name)
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_info = {
        im["id"]: (im["file_name"], im["width"], im["height"])
        for im in coco["images"]
    }
    coco_categories = {c["id"]: c["name"].lower() for c in coco["categories"]}

    # COCO category name -> YOLO class id (via cfg.class_mapping).
    cls_id = cfg.class_id  # {"text":0, "formula":1, "image":2}
    cat_to_yolo: dict[int, int] = {}
    skipped_cats: set[str] = set()
    for cid, cname in coco_categories.items():
        target = cfg.class_mapping.get(cname)
        if target is None:
            skipped_cats.add(cname)
            continue
        if target not in cls_id:
            raise ValueError(
                f"Class mapping references unknown YOLO class {target!r}"
            )
        cat_to_yolo[cid] = cls_id[target]
    if skipped_cats:
        log.info("Categories without mapping (will be ignored): %s",
                 sorted(skipped_cats))

    # Group annotations by image id.
    img_to_anns: dict[int, list] = {}
    for a in coco["annotations"]:
        img_to_anns.setdefault(a["image_id"], []).append(a)

    items: Iterable = list(img_to_anns.items())
    cap = max_pages if max_pages is not None else cfg.max_pages
    if cap is not None:
        items = items[:cap]
        log.info("Capping %s to first %d page(s) (max_pages).", split, cap)

    if image_index is None:
        image_index = _build_image_index(cfg.extracted_dir)

    processed = skipped_existing = skipped_missing = 0
    for img_id, anns in tqdm(items, desc=f"convert/{split}", unit="img"):
        fname, w, h = images_info[img_id]
        stem = Path(fname).stem
        lbl_path = lbl_out / f"{stem}.txt"
        img_path = img_out / fname

        if lbl_path.is_file() and img_path.is_file():
            skipped_existing += 1
            continue

        src = image_index.get(fname)
        if src is None:
            skipped_missing += 1
            continue

        link_or_copy(src, img_path)

        lines: list[str] = []
        for a in anns:
            yolo_cls = cat_to_yolo.get(a["category_id"])
            if yolo_cls is None:
                continue
            lines.append(_yolo_line(a["bbox"], w, h, yolo_cls))
        # Write even when empty; YOLO accepts empty label files.
        lbl_path.write_text("\n".join(lines), encoding="utf-8")
        processed += 1

    log.info(
        "[%s] processed=%d  skipped_existing=%d  skipped_missing=%d",
        split, processed, skipped_existing, skipped_missing,
    )
    return {
        "split": split,
        "processed": processed,
        "skipped_existing": skipped_existing,
        "skipped_missing": skipped_missing,
    }


def split_dataset(cfg: PipelineConfig) -> list[dict]:
    """Convert all three splits, sharing one image index."""
    if stage_done(cfg.yolo_dataset_dir, "convert"):
        log.info("Conversion already done (marker present). Skipping.")
        return []

    image_index = _build_image_index(cfg.extracted_dir)
    if not image_index:
        raise RuntimeError(
            f"No images found under {cfg.extracted_dir}. "
            "Run `extract_archives` first."
        )

    reports = []
    for split in ("train", "val", "test"):
        reports.append(
            convert_docbank_to_yolo(cfg, split, image_index=image_index)
        )

    create_yolo_yaml(cfg)
    mark_stage_done(
        cfg.yolo_dataset_dir, "convert",
        payload=json.dumps(reports, indent=2),
    )
    return reports


def create_yolo_yaml(cfg: PipelineConfig) -> Path:
    """Write Ultralytics-style data.yaml at cfg.yolo_dataset_dir/data.yaml."""
    yaml_path = cfg.yolo_dataset_dir / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(cfg.yolo_classes))
    content = (
        f"# Auto-generated by docbank_pipeline.convert.create_yolo_yaml\n"
        f"path: {cfg.yolo_dataset_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(cfg.yolo_classes)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    log.info("Wrote %s", yaml_path)
    return yaml_path


def dataset_statistics(cfg: PipelineConfig) -> dict:
    """Return per-split image / label / per-class instance counts."""
    stats: dict = {}
    for split in ("train", "val", "test"):
        imgs = list((cfg.yolo_dataset_dir / "images" / split).glob("*"))
        lbls = list((cfg.yolo_dataset_dir / "labels" / split).glob("*.txt"))
        per_class = {n: 0 for n in cfg.yolo_classes}
        for lf in lbls:
            for line in lf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                cid = int(line.split()[0])
                if 0 <= cid < len(cfg.yolo_classes):
                    per_class[cfg.yolo_classes[cid]] += 1
        stats[split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "instances": per_class,
        }
    return stats
