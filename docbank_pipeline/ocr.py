"""
Stage 5 — Recognition (OCR + LaTeXOCR) and structured JSON output.

Public API:
    preprocess_crop(image, ...)              -> np.ndarray for OCR
    recognize_text_with_paddleocr(image)     -> str
    recognize_formula_with_latexocr(image)   -> str  (LaTeX source)
    process_page_image(cfg, image, weights)  -> list[dict] of detections+text
    process_folder(cfg, folder, ...)         -> list[dict]
    save_results_json(results, out_path)     -> writes JSON

Heavy ML dependencies (`paddleocr`, `pix2tex`, `cv2`) are imported lazily.
If a dependency is missing, the recognition step for that class is skipped
with a clear warning rather than crashing the whole pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from .config import PipelineConfig
from .inference import run_yolo_inference

log = logging.getLogger("docbank.ocr")


# ---------------------------------------------------------------- preprocess

def preprocess_crop(
    image: "Any",
    *,
    grayscale: bool = True,
    blur_ksize: int = 3,
    threshold: bool = True,
    upscale: float = 1.0,
):
    """Cleanup pass before handing a crop to an OCR model.

    Steps (any of which can be disabled):
      1. grayscale,
      2. small Gaussian blur,
      3. adaptive threshold / Otsu binarisation,
      4. optional upscale for tiny crops.

    Accepts either a PIL Image or a numpy array; returns numpy uint8.
    """
    import numpy as np  # lazy

    try:
        import cv2  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required for preprocessing. "
            "Install with: pip install opencv-python-headless"
        ) from e

    if hasattr(image, "convert"):  # PIL.Image
        arr = np.array(image)
    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        gray = arr
    elif arr.shape[-1] == 4:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if grayscale else arr

    if upscale and upscale != 1.0:
        h, w = gray.shape[:2]
        gray = cv2.resize(
            gray,
            (int(w * upscale), int(h * upscale)),
            interpolation=cv2.INTER_CUBIC,
        )

    if blur_ksize and blur_ksize >= 3:
        # Must be odd.
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if threshold:
        # Otsu picks the threshold automatically; very robust on text.
        _, gray = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return gray


# ----------------------------------------------------------- model singletons

# At module level, change this line:
#     _paddle_singleton: Any = None
# to a per-language dict:
_paddle_singletons: dict[str, Any] = {}
_latex_singleton: Any = None


def _get_paddleocr(lang: str = "korean"):
    """Lazy-load and cache a PaddleOCR instance *per language*.
    Returns None on import failure."""
    if lang in _paddle_singletons:
        return _paddle_singletons[lang]

    # Disable oneDNN — Paddle 3.x's IR runtime crashes converting
    # pir::ArrayAttribute<DoubleAttribute> on Windows CPU.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_mkldnn", "0")

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError:
        log.warning(
            "paddleocr is not installed. Text recognition will be skipped. "
            "Install with: pip install paddleocr paddlepaddle"
        )
        return None

    log.info("Loading PaddleOCR for lang=%s (first run downloads weights)...", lang)
    last_err: Exception | None = None
    for kwargs in (
        {"use_doc_orientation_classify": False,
         "use_doc_unwarping": False,
         "use_textline_orientation": False,
         "enable_mkldnn": False,
         "lang": lang},
        {"use_angle_cls": False, "lang": lang, "show_log": False},
        {"use_angle_cls": False, "lang": lang},
        {"lang": lang},
        {},
    ):
        try:
            ocr = PaddleOCR(**kwargs)
            _paddle_singletons[lang] = ocr      # mutate dict; no `global` needed
            return ocr
        except (TypeError, ValueError) as e:
            last_err = e
            continue

    log.warning(
        "Could not initialise PaddleOCR (lang=%s) with any known signature: %s. "
        "Text recognition will be skipped.", lang, last_err,
    )
    return None

# _paddle_singleton: Any = None
# _latex_singleton: Any = None


# def _get_paddleocr():
#     """Lazy-load and cache a PaddleOCR instance. Returns None on import failure.

#     The PaddleOCR constructor has changed across releases — older versions
#     expect `use_angle_cls` and `show_log`; v3 dropped them. We attempt the
#     rich call first and fall back to progressively simpler signatures so this
#     works on whichever paddleocr the user happened to install.
#     """
#     global _paddle_singleton
#     if _paddle_singleton is not None:
#         return _paddle_singleton

#     # Disable oneDNN — Paddle 3.x's new IR runtime hits an "Unimplemented"
#     # path when oneDNN tries to convert pir::ArrayAttribute<DoubleAttribute>,
#     # which kills almost every PaddleOCR call on Windows CPU.
#     #
#     # Equivalent to `set FLAGS_use_mkldnn=0` before running. We set both the
#     # bool flag and `FLAGS_enable_mkldnn` for older builds.
#     os.environ.setdefault("FLAGS_use_mkldnn", "0")
#     os.environ.setdefault("FLAGS_enable_mkldnn", "0")

#     try:
#         from paddleocr import PaddleOCR  # type: ignore
#     except ImportError:
#         log.warning(
#             "paddleocr is not installed. Text recognition will be skipped. "
#             "Install with: pip install paddleocr paddlepaddle"
#         )
#         return None
#     log.info("Loading PaddleOCR (first run downloads weights, ~300MB)...")
#     last_err: Exception | None = None
#     # We pass `use_angle_cls=False` because our YOLO crops are already
#     # axis-aligned. Angle classification adds ~30% per call for no gain.
#     for kwargs in (
#         {"use_doc_orientation_classify": False,
#         "use_doc_unwarping": False,
#         "use_textline_orientation": False,
#         "enable_mkldnn": False,        # <-- the fix
#         "lang": "korean"},
#         # v2 fallbacks for older installs:
#         {"use_angle_cls": False, "lang": "korean", "show_log": False},
#         {"use_angle_cls": False, "lang": "korean"},
#         {"lang": "korean"},
#         {},
#         # {"use_angle_cls": False, "lang": "en", "show_log": False},
#         # {"use_angle_cls": False, "lang": "en"},
#         # {"lang": "en"},
#         # {},
#     ):
#         try:
#             _paddle_singleton = PaddleOCR(**kwargs)
#             return _paddle_singleton
#         except (TypeError, ValueError) as e:
#             last_err = e
#             continue
#     log.warning(
#         "Could not initialise PaddleOCR with any known signature: %s. "
#         "Text recognition will be skipped.", last_err,
#     )
#     return None


def _get_latexocr():
    """Lazy-load and cache a pix2tex instance. Returns None on import failure.

    pix2tex defaults are tuned for GPU and over-generate on CPU:
      * `max_seq_len = 1024` — autoregressive decoding to 1024 tokens. 99% of
        real formulas are < 100 tokens; long sequences only happen when the
        crop is junk and the decoder never emits EOS. Capping at 256 cuts
        worst-case wall-clock by 3-4x.
      * `temperature = 0.25` — sampling, slightly slower than greedy and
        non-deterministic. We force greedy by setting it to 0.

    Both knobs are read by `decoder.generate(...)` inside pix2tex, so changing
    them on the singleton is safe.
    """
    global _latex_singleton
    if _latex_singleton is not None:
        return _latex_singleton
    try:
        from pix2tex.cli import LatexOCR  # type: ignore
    except ImportError:
        log.warning(
            "pix2tex is not installed. Formula recognition will be skipped. "
            "Install with: pip install pix2tex"
        )
        return None
    log.info("Loading LaTeXOCR (first run downloads weights)...")
    _latex_singleton = LatexOCR()
# Cap decode length only. Do NOT touch temperature — pix2tex divides
# logits by it, so 0.0 produces nan and empty output. The default (0.25)
# is effectively near-greedy already and works fine.
    for attr, value in (("max_seq_len", 256),):
        try:
            setattr(_latex_singleton.args, attr, value)
        except (AttributeError, TypeError):
            pass
    return _latex_singleton
    # _latex_singleton = LatexOCR()
    # # Speed knobs — see the docstring above.
    # for attr, value in (("max_seq_len", 256), ("temperature", 0.0)):
    #     try:
    #         setattr(_latex_singleton.args, attr, value)
    #     except (AttributeError, TypeError):
    #         pass
    # return _latex_singleton


# ----------------------------------------------------------------- recognise

def _maybe_upscale(image_path: Path | str, min_h: int = 40, factor: float = 2.0):
    """Return either the path (no resize needed) or a numpy array upscaled
    via Lanczos. Tiny crops (height < `min_h` px) are upscaled by `factor`
    before OCR — PaddleOCR struggles with sub-30px text rows.

    Falls back to returning the original path if cv2/PIL are unavailable.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return str(image_path)
    try:
        img = Image.open(image_path)
    except Exception:
        return str(image_path)
    if img.height >= min_h:
        return str(image_path)
    new_size = (int(img.width * factor), int(img.height * factor))
    img = img.convert("RGB").resize(new_size, Image.LANCZOS)
    return np.array(img)


def recognize_text_with_paddleocr(image_path: Path | str) -> str:
    """Run PaddleOCR on a single line/paragraph crop and return its text.

    Speed knobs vs the default:
      * <b>det=False</b> — skip PaddleOCR's internal text-box detector. Our
        YOLO crops are already line/paragraph-level, so the inner detector
        is just re-finding what we already cut. Removing it is roughly a
        3x speedup per call.
      * <b>cls=False</b> — skip angle classification. Document pages
        aren't rotated; this saves another ~30% per call.
      * <b>auto-upscale</b> — tiny crops are 2x'd via Lanczos before OCR.

    Handles both v2 (<code>ocr.ocr</code>) and v3 (<code>ocr.predict</code>)
    output shapes.
    """
    ocr = _get_paddleocr()
    if ocr is None:
        return ""

    src = _maybe_upscale(image_path)
    result = None
    # ---- v3 path ----------------------------------------------------------
    if hasattr(ocr, "predict"):
        try:
            result = ocr.predict(src)
        except Exception as e:
            log.debug("PaddleOCR.predict failed on %s: %s", src, e)
            result = None
    # ---- v2 fast path: rec-only -----------------------------------------
    if result is None:
        try:
            # Skip det+cls; Paddle treats input as a single already-cropped
            # text line and only runs the recognition net.
            result = ocr.ocr(src, det=False, cls=False, rec=True)
        except TypeError:
            # Older Paddle versions don't accept all three kwargs.
            for kw in (dict(det=False, cls=False), dict(cls=False), {}):
                try:
                    result = ocr.ocr(src, **kw)
                    break
                except TypeError:
                    continue
        except Exception as e:
            log.warning("PaddleOCR failed on %s: %s", image_path, e)
            return ""

    if not result:
        return ""

    lines: list[str] = []
    for page in result:
        if page is None:
            continue
        # v3: dict-like with 'rec_texts'
        rec_texts = None
        if isinstance(page, dict):
            rec_texts = page.get("rec_texts")
        elif hasattr(page, "get"):
            try:
                rec_texts = page.get("rec_texts")
            except Exception:
                rec_texts = None
        elif hasattr(page, "rec_texts"):
            rec_texts = page.rec_texts
        if rec_texts:
            lines.extend(t for t in rec_texts if t)
            continue
        # v2 with det=True: list of [box, (txt, conf)]
        # v2 with det=False (our fast path): list of (txt, conf) tuples,
        # OR a single (txt, conf) tuple.
        if isinstance(page, tuple) and len(page) == 2 \
                and isinstance(page[0], str):
            if page[0]:
                lines.append(page[0])
            continue
        try:
            iterator = iter(page)
        except TypeError:
            continue
        for det in iterator:
            # det=False shape: (txt, conf)
            if isinstance(det, tuple) and len(det) == 2 \
                    and isinstance(det[0], str):
                if det[0]:
                    lines.append(det[0])
                continue
            # det=True shape: [box, (txt, conf)]
            try:
                _box, (txt, _conf) = det
            except (ValueError, TypeError):
                continue
            if txt:
                lines.append(txt)
    return "\n".join(lines)

_formula_model = None

def _get_formula_model(model_name: str = "PP-FormulaNet_plus-M"):
    """Lazy-load PaddleX's formula recognizer. Same oneDNN workaround as PaddleOCR."""
    global _formula_model
    if _formula_model is not None:
        return _formula_model
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    try:
        from paddlex import create_model
    except ImportError:
        log.warning("paddlex not available; falling back to pix2tex for formulas.")
        return None
    for kwargs in (
        {"model_name": model_name, "enable_mkldnn": False},  # mirrors the OCR fix
        {"model_name": model_name},
    ):
        try:
            _formula_model = create_model(**kwargs)
            return _formula_model
        except TypeError:
            continue
    return None


def recognize_formula_with_ppformulanet(image_path, *, pad: int = 12) -> str:
    """Run PaddleX PP-FormulaNet on a formula crop, returning LaTeX source.

    The crop is padded with a white margin first: YOLO formula boxes are
    tight, and these models expect some whitespace around the equation, so a
    small border noticeably improves recognition of edge glyphs (subscripts,
    brackets, integral signs).
    """
    model = _get_formula_model()
    if model is None:
        return ""
    tmp_path = None
    try:
        import tempfile
        from PIL import Image, ImageOps  # lazy
        img = Image.open(image_path).convert("RGB")
        if pad > 0:
            img = ImageOps.expand(img, border=pad, fill="white")
        # Hand PP-FormulaNet a path to the padded crop rather than a NumPy
        # array — this sidesteps any RGB/BGR channel-order assumptions inside
        # PaddleX and keeps the call identical to the unpadded case.
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(tmp_path)
        for res in model.predict(tmp_path, batch_size=1):
            for key in ("rec_formula", "rec_text", "formula", "text"):
                try:
                    val = res[key] if not isinstance(res, dict) else res.get(key)
                except Exception:
                    val = None
                if val:
                    return val
        return ""
    except Exception as e:
        log.warning("PP-FormulaNet failed on %s: %s", image_path, e)
        return ""
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def recognize_formula_with_latexocr(image_path: Path | str) -> str:
    """Run pix2tex (LaTeXOCR) on a formula crop, returning LaTeX source.

    Tiny crops are auto-upscaled to give pix2tex something it can read.
    """
    model = _get_latexocr()
    if model is None:
        return ""
    try:
        from PIL import Image  # lazy
        img = Image.open(image_path).convert("RGB")
        if img.height < 40:
            img = img.resize(
                (img.width * 2, img.height * 2), Image.LANCZOS
            )
        return model(img) or ""
    except Exception as e:
        log.warning("LaTeXOCR failed on %s: %s", image_path, e)
        return ""


def recognize_formula(image_path: Path | str) -> str:
    """Primary formula recognizer: PP-FormulaNet first, pix2tex as fallback.

    PP-FormulaNet is the stronger model and handles the bulk of formulas
    (including the complex / multi-line cases pix2tex mangled). pix2tex is
    only consulted when PP-FormulaNet returns nothing — e.g. a crop it can't
    read, or when paddlex / its weights aren't available and
    `recognize_formula_with_ppformulanet` short-circuits to "". Both share the
    same (image_path) -> str signature, so this drops straight into the
    dispatcher and still produces LaTeX (recognition_kind stays "latex").
    """
    out = recognize_formula_with_ppformulanet(image_path)
    if out and out.strip():
        return out
    return recognize_formula_with_latexocr(image_path)


# ------------------------------------------------------------------- pipeline

def _detection_to_record(det: dict) -> dict:
    """Make a detection dict JSON-serialisable (Paths -> str)."""
    rec = dict(det)
    for k, v in list(rec.items()):
        if isinstance(v, Path):
            rec[k] = str(v)
    return rec


def process_page_image(
    cfg: PipelineConfig,
    image: str | Path,
    *,
    weights: str | Path | None = None,
    conf: float = 0.25,
    do_text: bool = True,
    do_formula: bool = True,
    min_ocr_conf: float = 0.30,
    min_ocr_area: int = 600,
    use_cache: bool = True,
) -> list[dict]:
    """Detect → crop → recognise on a single page. Returns enriched detections."""
    detections = run_yolo_inference(
        cfg, image, weights=weights, conf=conf, save_crops=True
    )
    return _enrich(
        detections,
        do_text=do_text, do_formula=do_formula,
        min_ocr_conf=min_ocr_conf, min_ocr_area=min_ocr_area,
        use_cache=use_cache, cache_path=cfg.output_dir / "ocr_cache.json",
    )


def process_folder(
    cfg: PipelineConfig,
    folder: str | Path,
    *,
    weights: str | Path | None = None,
    conf: float = 0.25,
    do_text: bool = True,
    do_formula: bool = True,
    output_json: str | Path | None = None,
    max_pages: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    min_ocr_conf: float = 0.30,
    min_ocr_area: int = 600,
    min_formula_area: int | None = None,
    use_cache: bool = True,
    max_text_crops: int | None = None,
    max_formulas: int | None = None,
) -> list[dict]:
    """Run the full pipeline on every image in `folder`.

    `max_pages` (or `cfg.max_pages` when None) caps the number of source pages
    so a quick inference smoke-test doesn't have to wait for OCR over thousands
    of crops. `shuffle=True` picks them at random instead of in sorted order
    (useful for demo runs that should be representative). Pass `seed` for a
    reproducible random pick.

    `max_text_crops` / `max_formulas` hard-cap the number of OCR calls per
    class — pix2tex on CPU runs at ~80 s per formula crop, so a 1000-page run
    can otherwise take 20+ hours on the formula thread alone.
    """
    import random  # stdlib, lazy

    cap = max_pages if max_pages is not None else cfg.max_pages
    src: str | Path | list = folder
    if cap is not None or shuffle:
        p = Path(folder)
        if p.is_dir():
            imgs = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if shuffle:
                rng = random.Random(seed)
                rng.shuffle(imgs)
            if cap is not None:
                imgs = imgs[:cap]
            src = imgs
            mode = "random" if shuffle else "sequential"
            log.info(
                "Capping inference to %d page(s) (%s%s).",
                len(imgs),
                mode,
                f", seed={seed}" if shuffle and seed is not None else "",
            )

    detections = run_yolo_inference(
        cfg, src, weights=weights, conf=conf, save_crops=True, device=None
    )
    enriched = _enrich(
        detections,
        do_text=do_text, do_formula=do_formula,
        min_ocr_conf=min_ocr_conf, min_ocr_area=min_ocr_area,
        min_formula_area=min_formula_area,
        use_cache=use_cache, cache_path=cfg.output_dir / "ocr_cache.json",
        max_text_crops=max_text_crops,
        max_formulas=max_formulas,
    )
    if output_json:
        save_results_json(enriched, output_json)
    return enriched


def _crop_hash(path: str | Path) -> str:
    """Cheap, stable id for a crop file. Uses size + first-4KB md5 so we
    don't read 100 KB JPEGs end-to-end."""
    import hashlib
    p = Path(path)
    try:
        st = p.stat()
        with open(p, "rb") as f:
            head = f.read(4096)
        h = hashlib.md5(head).hexdigest()
        return f"{st.st_size}-{h}"
    except OSError:
        return ""


def _load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    """Atomic JSON write so a Ctrl+C halfway through never leaves a
    half-written cache file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        log.warning("Could not persist OCR cache to %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _enrich(
    detections: Sequence[dict],
    *,
    do_text: bool,
    do_formula: bool,
    min_ocr_conf: float = 0.30,
    min_ocr_area: int = 600,
    min_formula_area: int | None = None,
    use_cache: bool = True,
    cache_path: Path | None = None,
    max_text_crops: int | None = None,
    max_formulas: int | None = None,
) -> list[dict]:
    """Run OCR / LaTeX-OCR over `detections` with three big speedups:

    1. Filter — skip crops that fall below `min_ocr_conf` (low-confidence
       boxes) or `min_ocr_area` (tiny boxes that PaddleOCR can't read anyway).
       Each skipped detection drops a 0.3-2 s OCR call.
    2. Parallelise — text and formula OCR use different models, so we run
       them on two threads. Both ML libs release the GIL during inference,
       so this is real parallelism on CPU.
    3. Cache — keyed by (file size + 4KB md5) of the crop, persisted to
       `<output_dir>/ocr_cache.json`. Reruns of the same set are near-instant.
    """
    from concurrent.futures import ThreadPoolExecutor

    cache: dict = _load_cache(cache_path) if use_cache and cache_path else {}
    cache_hits = 0

    # Decide what to actually run on each detection.
    text_jobs: list[tuple[int, dict, str, str]] = []
    formula_jobs: list[tuple[int, dict, str, str]] = []
    skipped_filter = 0

    for idx, det in enumerate(detections):
        crop = det.get("crop_path")
        det.setdefault("recognized", "")
        det.setdefault("recognition_kind", None)
        if not crop:
            continue
        if det.get("confidence", 0.0) < min_ocr_conf:
            skipped_filter += 1
            continue
        bbox = det.get("bbox") or [0, 0, 0, 0]
        area = max(0, int(bbox[2]) - int(bbox[0])) * \
               max(0, int(bbox[3]) - int(bbox[1]))
        cls_name = det["class_name"]
        # Formula crops can use a stricter threshold because pix2tex on CPU
        # is the bottleneck — small noisy formula crops produce nothing useful
        # but still consume ~80 s each.
        threshold = (
            min_formula_area
            if cls_name == "formula" and min_formula_area is not None
            else min_ocr_area
        )
        if area < threshold:
            skipped_filter += 1
            continue
        ckey = _crop_hash(crop) if use_cache else ""
        if ckey and ckey in cache:
            det["recognized"] = cache[ckey]
            det["recognition_kind"] = "cached"
            cache_hits += 1
            continue

        if cls_name == "formula" and do_formula:
            formula_jobs.append((idx, det, str(crop), ckey))
        elif cls_name in {"text", "image"} and do_text:
            text_jobs.append((idx, det, str(crop), ckey))

    capped_text = capped_formula = 0
    if max_text_crops is not None and len(text_jobs) > max_text_crops:
        capped_text = len(text_jobs) - max_text_crops
        text_jobs = text_jobs[:max_text_crops]
    if max_formulas is not None and len(formula_jobs) > max_formulas:
        capped_formula = len(formula_jobs) - max_formulas
        formula_jobs = formula_jobs[:max_formulas]

    log.info(
        "Recognition plan: text=%d, formula=%d, cached=%d, filtered_out=%d, "
        "capped(text=%d, formula=%d), total=%d",
        len(text_jobs), len(formula_jobs), cache_hits, skipped_filter,
        capped_text, capped_formula, len(detections),
    )

    import threading
    cache_lock = threading.Lock()
    SAVE_EVERY = 25  # persist cache to disk every N items per worker

    def _worker(jobs, recognise_fn, kind, desc):
        bar = tqdm(
            total=len(jobs), desc=desc, unit="det",
            position=0 if kind == "text" else 1,
        )
        since_save = 0
        for _idx, det, crop, ckey in jobs:
            try:
                recog = recognise_fn(crop)
            except Exception as e:
                log.debug("%s OCR failed on %s: %s", kind, crop, e)
                recog = ""
            det["recognized"] = recog or ""
            det["recognition_kind"] = kind
            if ckey and recog:
                with cache_lock:
                    cache[ckey] = recog
            since_save += 1
            # Persist the cache periodically. Saves cost is negligible
            # (small JSON), and a Ctrl+C now keeps every result we've
            # already produced — no more "4 hours of work, nothing on disk".
            if since_save >= SAVE_EVERY and cache_path is not None:
                with cache_lock:
                    _save_cache(cache_path, cache)
                since_save = 0
            bar.update(1)
        bar.close()

    # Run text and formula passes truly in parallel — they use disjoint models
    # so neither is blocked on the other.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = []
        if text_jobs:
            futs.append(pool.submit(
                _worker, text_jobs, recognize_text_with_paddleocr,
                "text", "recognise/text",
            ))
        if formula_jobs:
            futs.append(pool.submit(
                _worker, formula_jobs, recognize_formula,
                "latex", "recognise/formula",
            ))
        for f in futs:
            f.result()

    if use_cache and cache_path:
        _save_cache(cache_path, cache)

    # Re-emit the original list in order, with mutations applied in place.
    return list(detections)


def save_results_json(results: Sequence[dict], out_path: str | Path) -> Path:
    """Group detections by source page and write a structured JSON file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    pages: dict[str, dict] = {}
    for det in results:
        rec = _detection_to_record(det)
        img_str = rec.pop("image", "")
        page_entry = pages.setdefault(
            img_str,
            {
                "image": img_str,
                "image_width": rec.get("image_width"),
                "image_height": rec.get("image_height"),
                "detections": [],
            },
        )
        page_entry["detections"].append(rec)

    payload = {"pages": list(pages.values())}
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d page record(s) to %s", len(pages), p)
    return p

# """
# Stage 5 — Recognition (OCR + LaTeXOCR) and structured JSON output.

# Public API:
#     preprocess_crop(image, ...)              -> np.ndarray for OCR
#     recognize_text_with_paddleocr(image)     -> str
#     recognize_formula_with_latexocr(image)   -> str  (LaTeX source)
#     process_page_image(cfg, image, weights)  -> list[dict] of detections+text
#     process_folder(cfg, folder, ...)         -> list[dict]
#     save_results_json(results, out_path)     -> writes JSON

# Heavy ML dependencies (`paddleocr`, `pix2tex`, `cv2`) are imported lazily.
# If a dependency is missing, the recognition step for that class is skipped
# with a clear warning rather than crashing the whole pipeline.
# """

# from __future__ import annotations

# import json
# import logging
# import os
# from pathlib import Path
# from typing import Any, Sequence

# from tqdm.auto import tqdm

# from .config import PipelineConfig
# from .inference import run_yolo_inference

# log = logging.getLogger("docbank.ocr")


# # ---------------------------------------------------------------- preprocess

# def preprocess_crop(
#     image: "Any",
#     *,
#     grayscale: bool = True,
#     blur_ksize: int = 3,
#     threshold: bool = True,
#     upscale: float = 1.0,
# ):
#     """Cleanup pass before handing a crop to an OCR model.

#     Steps (any of which can be disabled):
#       1. grayscale,
#       2. small Gaussian blur,
#       3. adaptive threshold / Otsu binarisation,
#       4. optional upscale for tiny crops.

#     Accepts either a PIL Image or a numpy array; returns numpy uint8.
#     """
#     import numpy as np  # lazy

#     try:
#         import cv2  # type: ignore
#     except ImportError as e:  # pragma: no cover
#         raise ImportError(
#             "OpenCV (cv2) is required for preprocessing. "
#             "Install with: pip install opencv-python-headless"
#         ) from e

#     if hasattr(image, "convert"):  # PIL.Image
#         arr = np.array(image)
#     else:
#         arr = np.asarray(image)

#     if arr.ndim == 2:
#         gray = arr
#     elif arr.shape[-1] == 4:
#         gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
#     else:
#         gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if grayscale else arr

#     if upscale and upscale != 1.0:
#         h, w = gray.shape[:2]
#         gray = cv2.resize(
#             gray,
#             (int(w * upscale), int(h * upscale)),
#             interpolation=cv2.INTER_CUBIC,
#         )

#     if blur_ksize and blur_ksize >= 3:
#         # Must be odd.
#         k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
#         gray = cv2.GaussianBlur(gray, (k, k), 0)

#     if threshold:
#         # Otsu picks the threshold automatically; very robust on text.
#         _, gray = cv2.threshold(
#             gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#         )

#     return gray


# # ----------------------------------------------------------- model singletons

# # At module level, change this line:
# #     _paddle_singleton: Any = None
# # to a per-language dict:
# _paddle_singletons: dict[str, Any] = {}
# _latex_singleton: Any = None


# def _get_paddleocr(lang: str = "korean"):
#     """Lazy-load and cache a PaddleOCR instance *per language*.
#     Returns None on import failure."""
#     if lang in _paddle_singletons:
#         return _paddle_singletons[lang]

#     # Disable oneDNN — Paddle 3.x's IR runtime crashes converting
#     # pir::ArrayAttribute<DoubleAttribute> on Windows CPU.
#     os.environ.setdefault("FLAGS_use_mkldnn", "0")
#     os.environ.setdefault("FLAGS_enable_mkldnn", "0")

#     try:
#         from paddleocr import PaddleOCR  # type: ignore
#     except ImportError:
#         log.warning(
#             "paddleocr is not installed. Text recognition will be skipped. "
#             "Install with: pip install paddleocr paddlepaddle"
#         )
#         return None

#     log.info("Loading PaddleOCR for lang=%s (first run downloads weights)...", lang)
#     last_err: Exception | None = None
#     for kwargs in (
#         {"use_doc_orientation_classify": False,
#          "use_doc_unwarping": False,
#          "use_textline_orientation": False,
#          "enable_mkldnn": False,
#          "lang": lang},
#         {"use_angle_cls": False, "lang": lang, "show_log": False},
#         {"use_angle_cls": False, "lang": lang},
#         {"lang": lang},
#         {},
#     ):
#         try:
#             ocr = PaddleOCR(**kwargs)
#             _paddle_singletons[lang] = ocr      # mutate dict; no `global` needed
#             return ocr
#         except (TypeError, ValueError) as e:
#             last_err = e
#             continue

#     log.warning(
#         "Could not initialise PaddleOCR (lang=%s) with any known signature: %s. "
#         "Text recognition will be skipped.", lang, last_err,
#     )
#     return None

# # _paddle_singleton: Any = None
# # _latex_singleton: Any = None


# # def _get_paddleocr():
# #     """Lazy-load and cache a PaddleOCR instance. Returns None on import failure.

# #     The PaddleOCR constructor has changed across releases — older versions
# #     expect `use_angle_cls` and `show_log`; v3 dropped them. We attempt the
# #     rich call first and fall back to progressively simpler signatures so this
# #     works on whichever paddleocr the user happened to install.
# #     """
# #     global _paddle_singleton
# #     if _paddle_singleton is not None:
# #         return _paddle_singleton

# #     # Disable oneDNN — Paddle 3.x's new IR runtime hits an "Unimplemented"
# #     # path when oneDNN tries to convert pir::ArrayAttribute<DoubleAttribute>,
# #     # which kills almost every PaddleOCR call on Windows CPU.
# #     #
# #     # Equivalent to `set FLAGS_use_mkldnn=0` before running. We set both the
# #     # bool flag and `FLAGS_enable_mkldnn` for older builds.
# #     os.environ.setdefault("FLAGS_use_mkldnn", "0")
# #     os.environ.setdefault("FLAGS_enable_mkldnn", "0")

# #     try:
# #         from paddleocr import PaddleOCR  # type: ignore
# #     except ImportError:
# #         log.warning(
# #             "paddleocr is not installed. Text recognition will be skipped. "
# #             "Install with: pip install paddleocr paddlepaddle"
# #         )
# #         return None
# #     log.info("Loading PaddleOCR (first run downloads weights, ~300MB)...")
# #     last_err: Exception | None = None
# #     # We pass `use_angle_cls=False` because our YOLO crops are already
# #     # axis-aligned. Angle classification adds ~30% per call for no gain.
# #     for kwargs in (
# #         {"use_doc_orientation_classify": False,
# #         "use_doc_unwarping": False,
# #         "use_textline_orientation": False,
# #         "enable_mkldnn": False,        # <-- the fix
# #         "lang": "korean"},
# #         # v2 fallbacks for older installs:
# #         {"use_angle_cls": False, "lang": "korean", "show_log": False},
# #         {"use_angle_cls": False, "lang": "korean"},
# #         {"lang": "korean"},
# #         {},
# #         # {"use_angle_cls": False, "lang": "en", "show_log": False},
# #         # {"use_angle_cls": False, "lang": "en"},
# #         # {"lang": "en"},
# #         # {},
# #     ):
# #         try:
# #             _paddle_singleton = PaddleOCR(**kwargs)
# #             return _paddle_singleton
# #         except (TypeError, ValueError) as e:
# #             last_err = e
# #             continue
# #     log.warning(
# #         "Could not initialise PaddleOCR with any known signature: %s. "
# #         "Text recognition will be skipped.", last_err,
# #     )
# #     return None


# def _get_latexocr():
#     """Lazy-load and cache a pix2tex instance. Returns None on import failure.

#     pix2tex defaults are tuned for GPU and over-generate on CPU:
#       * `max_seq_len = 1024` — autoregressive decoding to 1024 tokens. 99% of
#         real formulas are < 100 tokens; long sequences only happen when the
#         crop is junk and the decoder never emits EOS. Capping at 256 cuts
#         worst-case wall-clock by 3-4x.
#       * `temperature = 0.25` — sampling, slightly slower than greedy and
#         non-deterministic. We force greedy by setting it to 0.

#     Both knobs are read by `decoder.generate(...)` inside pix2tex, so changing
#     them on the singleton is safe.
#     """
#     global _latex_singleton
#     if _latex_singleton is not None:
#         return _latex_singleton
#     try:
#         from pix2tex.cli import LatexOCR  # type: ignore
#     except ImportError:
#         log.warning(
#             "pix2tex is not installed. Formula recognition will be skipped. "
#             "Install with: pip install pix2tex"
#         )
#         return None
#     log.info("Loading LaTeXOCR (first run downloads weights)...")
#     _latex_singleton = LatexOCR()
# # Cap decode length only. Do NOT touch temperature — pix2tex divides
# # logits by it, so 0.0 produces nan and empty output. The default (0.25)
# # is effectively near-greedy already and works fine.
#     for attr, value in (("max_seq_len", 256),):
#         try:
#             setattr(_latex_singleton.args, attr, value)
#         except (AttributeError, TypeError):
#             pass
#     return _latex_singleton
#     # _latex_singleton = LatexOCR()
#     # # Speed knobs — see the docstring above.
#     # for attr, value in (("max_seq_len", 256), ("temperature", 0.0)):
#     #     try:
#     #         setattr(_latex_singleton.args, attr, value)
#     #     except (AttributeError, TypeError):
#     #         pass
#     # return _latex_singleton


# # ----------------------------------------------------------------- recognise

# def _maybe_upscale(image_path: Path | str, min_h: int = 40, factor: float = 2.0):
#     """Return either the path (no resize needed) or a numpy array upscaled
#     via Lanczos. Tiny crops (height < `min_h` px) are upscaled by `factor`
#     before OCR — PaddleOCR struggles with sub-30px text rows.

#     Falls back to returning the original path if cv2/PIL are unavailable.
#     """
#     try:
#         from PIL import Image
#         import numpy as np
#     except ImportError:
#         return str(image_path)
#     try:
#         img = Image.open(image_path)
#     except Exception:
#         return str(image_path)
#     if img.height >= min_h:
#         return str(image_path)
#     new_size = (int(img.width * factor), int(img.height * factor))
#     img = img.convert("RGB").resize(new_size, Image.LANCZOS)
#     return np.array(img)


# def recognize_text_with_paddleocr(image_path: Path | str) -> str:
#     """Run PaddleOCR on a single line/paragraph crop and return its text.

#     Speed knobs vs the default:
#       * <b>det=False</b> — skip PaddleOCR's internal text-box detector. Our
#         YOLO crops are already line/paragraph-level, so the inner detector
#         is just re-finding what we already cut. Removing it is roughly a
#         3x speedup per call.
#       * <b>cls=False</b> — skip angle classification. Document pages
#         aren't rotated; this saves another ~30% per call.
#       * <b>auto-upscale</b> — tiny crops are 2x'd via Lanczos before OCR.

#     Handles both v2 (<code>ocr.ocr</code>) and v3 (<code>ocr.predict</code>)
#     output shapes.
#     """
#     ocr = _get_paddleocr()
#     if ocr is None:
#         return ""

#     src = _maybe_upscale(image_path)
#     result = None
#     # ---- v3 path ----------------------------------------------------------
#     if hasattr(ocr, "predict"):
#         try:
#             result = ocr.predict(src)
#         except Exception as e:
#             log.debug("PaddleOCR.predict failed on %s: %s", src, e)
#             result = None
#     # ---- v2 fast path: rec-only -----------------------------------------
#     if result is None:
#         try:
#             # Skip det+cls; Paddle treats input as a single already-cropped
#             # text line and only runs the recognition net.
#             result = ocr.ocr(src, det=False, cls=False, rec=True)
#         except TypeError:
#             # Older Paddle versions don't accept all three kwargs.
#             for kw in (dict(det=False, cls=False), dict(cls=False), {}):
#                 try:
#                     result = ocr.ocr(src, **kw)
#                     break
#                 except TypeError:
#                     continue
#         except Exception as e:
#             log.warning("PaddleOCR failed on %s: %s", image_path, e)
#             return ""

#     if not result:
#         return ""

#     lines: list[str] = []
#     for page in result:
#         if page is None:
#             continue
#         # v3: dict-like with 'rec_texts'
#         rec_texts = None
#         if isinstance(page, dict):
#             rec_texts = page.get("rec_texts")
#         elif hasattr(page, "get"):
#             try:
#                 rec_texts = page.get("rec_texts")
#             except Exception:
#                 rec_texts = None
#         elif hasattr(page, "rec_texts"):
#             rec_texts = page.rec_texts
#         if rec_texts:
#             lines.extend(t for t in rec_texts if t)
#             continue
#         # v2 with det=True: list of [box, (txt, conf)]
#         # v2 with det=False (our fast path): list of (txt, conf) tuples,
#         # OR a single (txt, conf) tuple.
#         if isinstance(page, tuple) and len(page) == 2 \
#                 and isinstance(page[0], str):
#             if page[0]:
#                 lines.append(page[0])
#             continue
#         try:
#             iterator = iter(page)
#         except TypeError:
#             continue
#         for det in iterator:
#             # det=False shape: (txt, conf)
#             if isinstance(det, tuple) and len(det) == 2 \
#                     and isinstance(det[0], str):
#                 if det[0]:
#                     lines.append(det[0])
#                 continue
#             # det=True shape: [box, (txt, conf)]
#             try:
#                 _box, (txt, _conf) = det
#             except (ValueError, TypeError):
#                 continue
#             if txt:
#                 lines.append(txt)
#     return "\n".join(lines)

# _formula_model = None

# def _get_formula_model(model_name: str = "PP-FormulaNet_plus-M"):
#     """Lazy-load PaddleX's formula recognizer. Same oneDNN workaround as PaddleOCR."""
#     global _formula_model
#     if _formula_model is not None:
#         return _formula_model
#     os.environ.setdefault("FLAGS_use_mkldnn", "0")
#     try:
#         from paddlex import create_model
#     except ImportError:
#         log.warning("paddlex not available; falling back to pix2tex for formulas.")
#         return None
#     for kwargs in (
#         {"model_name": model_name, "enable_mkldnn": False},  # mirrors the OCR fix
#         {"model_name": model_name},
#     ):
#         try:
#             _formula_model = create_model(**kwargs)
#             return _formula_model
#         except TypeError:
#             continue
#     return None


# def recognize_formula_with_ppformulanet(image_path) -> str:
#     model = _get_formula_model()
#     if model is None:
#         return ""
#     try:
#         for res in model.predict(str(image_path), batch_size=1):
#             for key in ("rec_formula", "rec_text", "formula", "text"):
#                 try:
#                     val = res[key] if not isinstance(res, dict) else res.get(key)
#                 except Exception:
#                     val = None
#                 if val:
#                     return val
#         return ""
#     except Exception as e:
#         log.warning("PP-FormulaNet failed on %s: %s", image_path, e)
#         return ""


# def recognize_formula_with_latexocr(image_path: Path | str) -> str:
#     """Run pix2tex (LaTeXOCR) on a formula crop, returning LaTeX source.

#     Tiny crops are auto-upscaled to give pix2tex something it can read.
#     """
#     model = _get_latexocr()
#     if model is None:
#         return ""
#     try:
#         from PIL import Image  # lazy
#         img = Image.open(image_path).convert("RGB")
#         if img.height < 40:
#             img = img.resize(
#                 (img.width * 2, img.height * 2), Image.LANCZOS
#             )
#         return model(img) or ""
#     except Exception as e:
#         log.warning("LaTeXOCR failed on %s: %s", image_path, e)
#         return ""


# # ------------------------------------------------------------------- pipeline

# def _detection_to_record(det: dict) -> dict:
#     """Make a detection dict JSON-serialisable (Paths -> str)."""
#     rec = dict(det)
#     for k, v in list(rec.items()):
#         if isinstance(v, Path):
#             rec[k] = str(v)
#     return rec


# def process_page_image(
#     cfg: PipelineConfig,
#     image: str | Path,
#     *,
#     weights: str | Path | None = None,
#     conf: float = 0.25,
#     do_text: bool = True,
#     do_formula: bool = True,
#     min_ocr_conf: float = 0.30,
#     min_ocr_area: int = 600,
#     use_cache: bool = True,
# ) -> list[dict]:
#     """Detect → crop → recognise on a single page. Returns enriched detections."""
#     detections = run_yolo_inference(
#         cfg, image, weights=weights, conf=conf, save_crops=True
#     )
#     return _enrich(
#         detections,
#         do_text=do_text, do_formula=do_formula,
#         min_ocr_conf=min_ocr_conf, min_ocr_area=min_ocr_area,
#         use_cache=use_cache, cache_path=cfg.output_dir / "ocr_cache.json",
#     )


# def process_folder(
#     cfg: PipelineConfig,
#     folder: str | Path,
#     *,
#     weights: str | Path | None = None,
#     conf: float = 0.25,
#     do_text: bool = True,
#     do_formula: bool = True,
#     output_json: str | Path | None = None,
#     max_pages: int | None = None,
#     shuffle: bool = False,
#     seed: int | None = None,
#     min_ocr_conf: float = 0.30,
#     min_ocr_area: int = 600,
#     min_formula_area: int | None = None,
#     use_cache: bool = True,
#     max_text_crops: int | None = None,
#     max_formulas: int | None = None,
# ) -> list[dict]:
#     """Run the full pipeline on every image in `folder`.

#     `max_pages` (or `cfg.max_pages` when None) caps the number of source pages
#     so a quick inference smoke-test doesn't have to wait for OCR over thousands
#     of crops. `shuffle=True` picks them at random instead of in sorted order
#     (useful for demo runs that should be representative). Pass `seed` for a
#     reproducible random pick.

#     `max_text_crops` / `max_formulas` hard-cap the number of OCR calls per
#     class — pix2tex on CPU runs at ~80 s per formula crop, so a 1000-page run
#     can otherwise take 20+ hours on the formula thread alone.
#     """
#     import random  # stdlib, lazy

#     cap = max_pages if max_pages is not None else cfg.max_pages
#     src: str | Path | list = folder
#     if cap is not None or shuffle:
#         p = Path(folder)
#         if p.is_dir():
#             imgs = sorted(
#                 f for f in p.iterdir()
#                 if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
#             )
#             if shuffle:
#                 rng = random.Random(seed)
#                 rng.shuffle(imgs)
#             if cap is not None:
#                 imgs = imgs[:cap]
#             src = imgs
#             mode = "random" if shuffle else "sequential"
#             log.info(
#                 "Capping inference to %d page(s) (%s%s).",
#                 len(imgs),
#                 mode,
#                 f", seed={seed}" if shuffle and seed is not None else "",
#             )

#     detections = run_yolo_inference(
#         cfg, src, weights=weights, conf=conf, save_crops=True, device=0
#     )
#     enriched = _enrich(
#         detections,
#         do_text=do_text, do_formula=do_formula,
#         min_ocr_conf=min_ocr_conf, min_ocr_area=min_ocr_area,
#         min_formula_area=min_formula_area,
#         use_cache=use_cache, cache_path=cfg.output_dir / "ocr_cache.json",
#         max_text_crops=max_text_crops,
#         max_formulas=max_formulas,
#     )
#     if output_json:
#         save_results_json(enriched, output_json)
#     return enriched


# def _crop_hash(path: str | Path) -> str:
#     """Cheap, stable id for a crop file. Uses size + first-4KB md5 so we
#     don't read 100 KB JPEGs end-to-end."""
#     import hashlib
#     p = Path(path)
#     try:
#         st = p.stat()
#         with open(p, "rb") as f:
#             head = f.read(4096)
#         h = hashlib.md5(head).hexdigest()
#         return f"{st.st_size}-{h}"
#     except OSError:
#         return ""


# def _load_cache(path: Path) -> dict:
#     if not path.is_file():
#         return {}
#     try:
#         return json.loads(path.read_text(encoding="utf-8"))
#     except Exception:
#         return {}


# def _save_cache(path: Path, cache: dict) -> None:
#     """Atomic JSON write so a Ctrl+C halfway through never leaves a
#     half-written cache file on disk."""
#     path.parent.mkdir(parents=True, exist_ok=True)
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     try:
#         tmp.write_text(json.dumps(cache), encoding="utf-8")
#         tmp.replace(path)
#     except Exception as e:
#         log.warning("Could not persist OCR cache to %s: %s", path, e)
#         try:
#             tmp.unlink(missing_ok=True)
#         except Exception:
#             pass


# def _enrich(
#     detections: Sequence[dict],
#     *,
#     do_text: bool,
#     do_formula: bool,
#     min_ocr_conf: float = 0.30,
#     min_ocr_area: int = 600,
#     min_formula_area: int | None = None,
#     use_cache: bool = True,
#     cache_path: Path | None = None,
#     max_text_crops: int | None = None,
#     max_formulas: int | None = None,
# ) -> list[dict]:
#     """Run OCR / LaTeX-OCR over `detections` with three big speedups:

#     1. Filter — skip crops that fall below `min_ocr_conf` (low-confidence
#        boxes) or `min_ocr_area` (tiny boxes that PaddleOCR can't read anyway).
#        Each skipped detection drops a 0.3-2 s OCR call.
#     2. Parallelise — text and formula OCR use different models, so we run
#        them on two threads. Both ML libs release the GIL during inference,
#        so this is real parallelism on CPU.
#     3. Cache — keyed by (file size + 4KB md5) of the crop, persisted to
#        `<output_dir>/ocr_cache.json`. Reruns of the same set are near-instant.
#     """
#     from concurrent.futures import ThreadPoolExecutor

#     cache: dict = _load_cache(cache_path) if use_cache and cache_path else {}
#     cache_hits = 0

#     # Decide what to actually run on each detection.
#     text_jobs: list[tuple[int, dict, str, str]] = []
#     formula_jobs: list[tuple[int, dict, str, str]] = []
#     skipped_filter = 0

#     for idx, det in enumerate(detections):
#         crop = det.get("crop_path")
#         det.setdefault("recognized", "")
#         det.setdefault("recognition_kind", None)
#         if not crop:
#             continue
#         if det.get("confidence", 0.0) < min_ocr_conf:
#             skipped_filter += 1
#             continue
#         bbox = det.get("bbox") or [0, 0, 0, 0]
#         area = max(0, int(bbox[2]) - int(bbox[0])) * \
#                max(0, int(bbox[3]) - int(bbox[1]))
#         cls_name = det["class_name"]
#         # Formula crops can use a stricter threshold because pix2tex on CPU
#         # is the bottleneck — small noisy formula crops produce nothing useful
#         # but still consume ~80 s each.
#         threshold = (
#             min_formula_area
#             if cls_name == "formula" and min_formula_area is not None
#             else min_ocr_area
#         )
#         if area < threshold:
#             skipped_filter += 1
#             continue
#         ckey = _crop_hash(crop) if use_cache else ""
#         if ckey and ckey in cache:
#             det["recognized"] = cache[ckey]
#             det["recognition_kind"] = "cached"
#             cache_hits += 1
#             continue

#         if cls_name == "formula" and do_formula:
#             formula_jobs.append((idx, det, str(crop), ckey))
#         elif cls_name in {"text", "image"} and do_text:
#             text_jobs.append((idx, det, str(crop), ckey))

#     capped_text = capped_formula = 0
#     if max_text_crops is not None and len(text_jobs) > max_text_crops:
#         capped_text = len(text_jobs) - max_text_crops
#         text_jobs = text_jobs[:max_text_crops]
#     if max_formulas is not None and len(formula_jobs) > max_formulas:
#         capped_formula = len(formula_jobs) - max_formulas
#         formula_jobs = formula_jobs[:max_formulas]

#     log.info(
#         "Recognition plan: text=%d, formula=%d, cached=%d, filtered_out=%d, "
#         "capped(text=%d, formula=%d), total=%d",
#         len(text_jobs), len(formula_jobs), cache_hits, skipped_filter,
#         capped_text, capped_formula, len(detections),
#     )

#     import threading
#     cache_lock = threading.Lock()
#     SAVE_EVERY = 25  # persist cache to disk every N items per worker

#     def _worker(jobs, recognise_fn, kind, desc):
#         bar = tqdm(
#             total=len(jobs), desc=desc, unit="det",
#             position=0 if kind == "text" else 1,
#         )
#         since_save = 0
#         for _idx, det, crop, ckey in jobs:
#             try:
#                 recog = recognise_fn(crop)
#             except Exception as e:
#                 log.debug("%s OCR failed on %s: %s", kind, crop, e)
#                 recog = ""
#             det["recognized"] = recog or ""
#             det["recognition_kind"] = kind
#             if ckey and recog:
#                 with cache_lock:
#                     cache[ckey] = recog
#             since_save += 1
#             # Persist the cache periodically. Saves cost is negligible
#             # (small JSON), and a Ctrl+C now keeps every result we've
#             # already produced — no more "4 hours of work, nothing on disk".
#             if since_save >= SAVE_EVERY and cache_path is not None:
#                 with cache_lock:
#                     _save_cache(cache_path, cache)
#                 since_save = 0
#             bar.update(1)
#         bar.close()

#     # Run text and formula passes truly in parallel — they use disjoint models
#     # so neither is blocked on the other.
#     with ThreadPoolExecutor(max_workers=2) as pool:
#         futs = []
#         if text_jobs:
#             futs.append(pool.submit(
#                 _worker, text_jobs, recognize_text_with_paddleocr,
#                 "text", "recognise/text",
#             ))
#         if formula_jobs:
#             futs.append(pool.submit(
#                 _worker, formula_jobs, recognize_formula_with_ppformulanet,
#                 "latex", "recognise/formula",
#             ))
#         for f in futs:
#             f.result()

#     if use_cache and cache_path:
#         _save_cache(cache_path, cache)

#     # Re-emit the original list in order, with mutations applied in place.
#     return list(detections)


# def save_results_json(results: Sequence[dict], out_path: str | Path) -> Path:
#     """Group detections by source page and write a structured JSON file."""
#     p = Path(out_path)
#     p.parent.mkdir(parents=True, exist_ok=True)

#     pages: dict[str, dict] = {}
#     for det in results:
#         rec = _detection_to_record(det)
#         img_str = rec.pop("image", "")
#         page_entry = pages.setdefault(
#             img_str,
#             {
#                 "image": img_str,
#                 "image_width": rec.get("image_width"),
#                 "image_height": rec.get("image_height"),
#                 "detections": [],
#             },
#         )
#         page_entry["detections"].append(rec)

#     payload = {"pages": list(pages.values())}
#     p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
#     log.info("Wrote %d page record(s) to %s", len(pages), p)
#     return p
