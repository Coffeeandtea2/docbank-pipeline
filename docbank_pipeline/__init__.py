"""DocBank -> YOLO -> OCR pipeline (refactored, resumable)."""

# IMPORTANT: these env vars MUST be set before `paddle` (and therefore
# `paddleocr`) is imported anywhere in the process. Paddle 3.x's new IR
# executor + oneDNN combo crashes on Windows CPU with:
#   (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>]  (onednn_instruction.cc:118)
# Disabling oneDNN AND falling back to the legacy executor avoids the path.
import os as _os

for _k, _v in {
    "FLAGS_use_mkldnn": "0",
    "FLAGS_enable_mkldnn": "0",
    "FLAGS_enable_pir_in_executor": "0",
    "FLAGS_enable_new_ir_in_executor": "0",
}.items():
    _os.environ.setdefault(_k, _v)

# CPU-core split for parallel OCR — without this, PaddleOCR (OpenMP-based)
# and pix2tex (PyTorch) both try to grab every core, so the two threads in
# `_enrich` end up serialised and even slower than a single-thread run. We
# pin each numerical library to ~half the cores; text + formula then truly
# run in parallel on disjoint CPUs. Override with e.g. OMP_NUM_THREADS=8 if
# you want a single thread to use everything.
_cpu_total = _os.cpu_count() or 4
_per_model = max(1, _cpu_total // 2)
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_k, str(_per_model))

__all__ = [
    "PipelineConfig",
    "convert_docbank_to_yolo",
    "create_yolo_yaml",
    "dataset_statistics",
    "detect_device",
    "download_docbank_parts",
    "extract_archives",
    "preprocess_crop",
    "process_folder",
    "process_page_image",
    "recognize_formula_with_latexocr",
    "recognize_text_with_paddleocr",
    "run_yolo_inference",
    "save_results_json",
    "setup_logging",
    "split_dataset",
    "train_yolo",
    "validate_yolo",
    "verify_downloads",
]

_EXPORTS = {
    "PipelineConfig": ".config",
    "convert_docbank_to_yolo": ".convert",
    "create_yolo_yaml": ".convert",
    "dataset_statistics": ".convert",
    "detect_device": ".utils",
    "download_docbank_parts": ".download",
    "extract_archives": ".download",
    "preprocess_crop": ".ocr",
    "process_folder": ".ocr",
    "process_page_image": ".ocr",
    "recognize_formula_with_latexocr": ".ocr",
    "recognize_text_with_paddleocr": ".ocr",
    "run_yolo_inference": ".inference",
    "save_results_json": ".ocr",
    "setup_logging": ".utils",
    "split_dataset": ".convert",
    "train_yolo": ".train",
    "validate_yolo": ".train",
    "verify_downloads": ".download",
}


def __getattr__(name: str):
    """Resolve public exports lazily so importing config/utilities stays light."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
