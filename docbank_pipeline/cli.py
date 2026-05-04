"""
Command-line entry point.

Run:  `python -m docbank_pipeline <subcommand> [opts]`

Subcommands:
    download   Download annotations + N image archive parts from HuggingFace
    extract    Extract downloaded archives (resumable)
    convert    Build the YOLO dataset (COCO -> YOLO labels + data.yaml)
    train      Train YOLOv8 on the converted dataset
    val        Validate the latest (or given) checkpoint
    infer      Run inference + OCR + LaTeX-OCR on a folder of page images
    serve      Start the upload-and-extract Flask web app
    smoketest  End-to-end run on 5-20 images for sanity check
    info       Print resolved config paths
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .utils import setup_logging


def _add_common_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", type=Path, default=None,
                   help="Project root (overrides DATA_ROOT). All paths derive from this.")
    p.add_argument("--max-pages", type=int, default=None,
                   help="Cap pages per split during conversion / inference.")
    p.add_argument("--dataset-parts", type=int, default=None,
                   help="How many of the 10 image archive parts to use (1-10).")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--verbose", "-v", action="store_true")


def _build_cfg(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    if args.data_root:
        cfg.data_root = Path(args.data_root).expanduser().resolve()
    if args.max_pages is not None:
        cfg.max_pages = args.max_pages
    if args.dataset_parts is not None:
        cfg.dataset_parts = args.dataset_parts
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    cfg.ensure_dirs()
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docbank_pipeline",
        description="DocBank → YOLO → OCR pipeline (resumable).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # download
    p_dl = sub.add_parser("download", help="Download annotations + N image parts.")
    _add_common_overrides(p_dl)

    # extract
    p_ex = sub.add_parser("extract", help="Extract downloaded archives.")
    _add_common_overrides(p_ex)
    p_ex.add_argument("--force", action="store_true",
                      help="Re-extract even if marker file exists.")

    # convert
    p_cv = sub.add_parser("convert", help="Convert annotations to YOLO format.")
    _add_common_overrides(p_cv)
    p_cv.add_argument("--split", choices=["all", "train", "val", "test"],
                      default="all")

    # train
    p_tr = sub.add_parser("train", help="Train YOLO.")
    _add_common_overrides(p_tr)
    p_tr.add_argument("--epochs", type=int, default=None)
    p_tr.add_argument("--imgsz", type=int, default=None)
    p_tr.add_argument("--batch", type=int, default=None)
    p_tr.add_argument("--model", type=str, default=None,
                      help="Base weights (e.g. yolov8n.pt, yolov8s.pt).")
    p_tr.add_argument("--device", type=str, default=None,
                      help="cuda | mps | cpu | <gpu-index>. Auto-detect if omitted.")
    p_tr.add_argument("--name", default="yolov8_docbank")

    # val
    p_va = sub.add_parser("val", help="Run validation.")
    _add_common_overrides(p_va)
    p_va.add_argument("--weights", type=str, default=None)
    p_va.add_argument("--split", choices=["val", "test"], default="val")

    # infer
    p_in = sub.add_parser("infer", help="Run inference + OCR + LaTeX-OCR.")
    _add_common_overrides(p_in)
    p_in.add_argument("--source", required=True, type=Path,
                      help="Path to an image, or folder of images.")
    p_in.add_argument("--weights", type=str, default=None)
    p_in.add_argument("--conf", type=float, default=0.25)
    p_in.add_argument("--no-text-ocr", action="store_true")
    p_in.add_argument("--no-formula-ocr", action="store_true")
    p_in.add_argument("--output", type=Path, default=None,
                      help="JSON output path (defaults to <output_dir>/results.json).")
    p_in.add_argument("--shuffle", action="store_true",
                      help="Pick pages at random instead of in sorted order.")
    p_in.add_argument("--seed", type=int, default=None,
                      help="Random seed for --shuffle (omit for non-deterministic).")
    p_in.add_argument("--min-ocr-conf", type=float, default=0.30,
                      help="Skip OCR for detections below this confidence (default 0.30).")
    p_in.add_argument("--min-ocr-area", type=int, default=600,
                      help="Skip OCR for crops smaller than this many pixels² (default 600).")
    p_in.add_argument("--min-formula-area", type=int, default=None,
                      help="Per-class override for formula crops only. "
                           "pix2tex is the bottleneck on CPU; a 2000-3000 "
                           "threshold cuts noise crops aggressively.")
    p_in.add_argument("--no-cache", action="store_true",
                      help="Disable the persistent OCR result cache.")
    p_in.add_argument("--max-formulas", type=int, default=None,
                      help="Hard cap on formula crops sent to pix2tex. "
                           "Useful on CPU where each formula takes ~80 s.")
    p_in.add_argument("--max-text-crops", type=int, default=None,
                      help="Hard cap on text/image crops sent to PaddleOCR.")

    # serve (web app)
    p_sv = sub.add_parser(
        "serve",
        help="Run the upload-and-extract web app (Flask).",
    )
    _add_common_overrides(p_sv)
    p_sv.add_argument("--host", default="127.0.0.1",
                      help='Host to bind. Use "0.0.0.0" to expose on LAN.')
    p_sv.add_argument("--port", type=int, default=5000)
    p_sv.add_argument("--debug", action="store_true")

    # smoketest
    p_st = sub.add_parser("smoketest", help="Tiny end-to-end run (5-20 images).")
    _add_common_overrides(p_st)
    p_st.add_argument("--n", type=int, default=10,
                      help="Number of pages to use (5-20).")
    p_st.add_argument("--skip-train", action="store_true",
                      help="Skip the (slow) training step.")
    p_st.add_argument("--epochs", type=int, default=2)

    # info
    sub.add_parser("info", help="Print resolved configuration.")

    args = parser.parse_args(argv)

    setup_logging(logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)
    cfg = _build_cfg(args)

    if args.cmd == "info":
        print(cfg.describe())
        return 0

    if args.cmd == "download":
        from . import download
        download.download_docbank_parts(cfg)
        download.verify_downloads(cfg)
        return 0

    if args.cmd == "extract":
        from . import download
        download.extract_archives(cfg, force=args.force)
        return 0

    if args.cmd == "convert":
        from . import convert
        if args.split == "all":
            convert.split_dataset(cfg)
        else:
            idx = convert._build_image_index(cfg.extracted_dir)
            convert.convert_docbank_to_yolo(cfg, args.split, image_index=idx)
            convert.create_yolo_yaml(cfg)
        stats = convert.dataset_statistics(cfg)
        for k, v in stats.items():
            print(f"{k}: {v}")
        return 0

    if args.cmd == "train":
        from . import train
        train.train_yolo(
            cfg,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            run_name=args.name,
        )
        return 0

    if args.cmd == "val":
        from . import train
        train.validate_yolo(cfg, weights=args.weights, split=args.split)
        return 0

    if args.cmd == "infer":
        from . import ocr
        out_json = args.output or cfg.output_dir / "results.json"
        ocr.process_folder(
            cfg,
            args.source,
            weights=args.weights,
            conf=args.conf,
            do_text=not args.no_text_ocr,
            do_formula=not args.no_formula_ocr,
            output_json=out_json,
            shuffle=args.shuffle,
            seed=args.seed,
            min_ocr_conf=args.min_ocr_conf,
            min_ocr_area=args.min_ocr_area,
            min_formula_area=args.min_formula_area,
            use_cache=not args.no_cache,
            max_text_crops=args.max_text_crops,
            max_formulas=args.max_formulas,
        )
        print(f"Results written to: {out_json}")
        return 0

    if args.cmd == "serve":
        from . import webapp
        webapp.serve(cfg, host=args.host, port=args.port, debug=args.debug)
        return 0

    if args.cmd == "smoketest":
        return _run_smoketest(cfg, args)

    parser.error(f"Unknown subcommand {args.cmd!r}")
    return 2


# ----------------------------------------------------------------- smoketest

def _run_smoketest(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    """Tiny end-to-end run for sanity checking on a fresh machine."""
    from . import convert, download, ocr, train

    n = max(5, min(args.n, 20))
    log = logging.getLogger("docbank.smoketest")
    log.info("=== SMOKETEST: %d pages, skip_train=%s ===", n, args.skip_train)

    # Pin a small subset.
    cfg.dataset_parts = max(cfg.dataset_parts, 1)
    cfg.max_pages = n

    log.info("[1/5] download")
    download.download_docbank_parts(cfg)

    log.info("[2/5] extract")
    download.extract_archives(cfg)

    log.info("[3/5] convert")
    convert.split_dataset(cfg)

    if not args.skip_train:
        log.info("[4/5] train (epochs=%d, tiny model)", args.epochs)
        train.train_yolo(
            cfg, model="yolov8n.pt", epochs=args.epochs,
            imgsz=320, batch=2, run_name="smoketest",
        )
    else:
        log.info("[4/5] training skipped (--skip-train).")

    log.info("[5/5] inference + OCR")
    val_imgs = list((cfg.yolo_dataset_dir / "images" / "val").glob("*.jpg"))[:n]
    if not val_imgs:
        log.error("No val images found; smoketest cannot finish inference.")
        return 1
    out_json = cfg.output_dir / "smoketest_results.json"
    ocr.process_folder(
        cfg, val_imgs[0].parent,
        conf=0.10,  # low threshold so detections appear even with few epochs
        output_json=out_json,
    )
    log.info("=== SMOKETEST OK -> %s ===", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
