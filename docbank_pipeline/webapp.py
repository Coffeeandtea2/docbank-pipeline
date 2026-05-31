"""
Stage 7 — Standalone web app.

Flask app that lets users upload page images, runs the full
detection + OCR + LaTeX recognition pipeline, and shows the result inline
plus a downloadable JSON.

Run with:
    python -m docbank_pipeline serve
    # then open http://127.0.0.1:5000

Per-upload job state lives under  <cfg.output_dir>/web/<job_id>/  containing:
    inputs/        # the uploaded page images
    boxes/         # the same pages with detection rectangles drawn on top
    crops/         # cropped detection regions
    results.json   # the structured output the user can download

The OCR / LaTeXOCR / YOLO models are loaded lazily by the existing pipeline
modules and reused across requests (singletons in `ocr.py`).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .config import PipelineConfig

log = logging.getLogger("docbank.webapp")


# ----------------------------------------------------------- box drawing

_BOX_COLORS_BGR = {
    "text":    (255, 180,   0),   # blue-ish
    "formula": (  0, 200,   0),   # green
    "image":   ( 80,  80, 255),   # red-ish
}


def _draw_boxes(image_path: Path, detections: list[dict], out_path: Path) -> bool:
    """Draw labelled bounding boxes on a copy of the page image."""
    import cv2  # lazy

    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("Could not read %s", image_path)
        return False

    for d in detections:
        cls = d["class_name"]
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        color = _BOX_COLORS_BGR.get(cls, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}
ALLOWED_EXTS = IMAGE_EXTS | PDF_EXTS

# Render PDF pages at this DPI. 200 dpi is the sweet spot for OCR — high
# enough to keep small superscripts/subscripts legible, low enough to keep
# pages around 1.5-2 MB each.
PDF_RENDER_DPI = 200
# Per-PDF page cap so a 300-page paper doesn't melt the laptop.
PDF_MAX_PAGES = 25


def _explode_pdf(pdf_path: Path, out_dir: Path) -> list[Path]:
    """Render every page of `pdf_path` to a JPG inside `out_dir`.

    Uses PyMuPDF (`fitz`) — pure-Python wheel, no external poppler binary
    required. Returns the list of created page images, in order.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PDF support needs PyMuPDF. Install with: pip install pymupdf"
        ) from e

    doc = fitz.open(str(pdf_path))
    try:
        n_pages = min(len(doc), PDF_MAX_PAGES)
        if len(doc) > PDF_MAX_PAGES:
            log.warning(
                "PDF %s has %d pages; only the first %d will be processed.",
                pdf_path.name, len(doc), PDF_MAX_PAGES,
            )
        zoom = PDF_RENDER_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        stem = pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []
        for i in range(n_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out = out_dir / f"{stem}_p{i+1:03d}.jpg"
            pix.save(str(out), jpg_quality=92)
            produced.append(out)
        return produced
    finally:
        doc.close()

def _escape_latex_text(text: str) -> str:
    """Escape text for safe LaTeX rendering."""
    if not text:
        return ""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


def _results_to_tex(results: dict, tex_path: Path) -> Path:
    """
    Convert OCR/layout results to a simple LaTeX file.
    This version prioritizes correctness and readable output.
    """

    # XeLaTeX + xeCJK so Korean / Hanja (and any CJK) render. Plain pdflatex
    # with inputenc/T1 fontenc cannot typeset these glyphs at all, which is
    # what produced the near-empty output PDFs.
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{fontspec}",
        r"\usepackage{xeCJK}",
        r"\setCJKmainfont{Noto Sans CJK KR}",
        r"\usepackage{amsmath, amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{geometry}",
        r"\usepackage{xcolor}",
        r"\geometry{margin=1in}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.7em}",
        r"\begin{document}",
    ]

    pages = results.get("pages", [])

    for page_index, page in enumerate(pages, start=1):
        lines.append(rf"\section*{{Page {page_index}}}")

        detections = page.get("detections", [])

        # Sort top-to-bottom, left-to-right
        detections = sorted(
            detections,
            key=lambda d: (
                d.get("bbox", [0, 0, 0, 0])[1],
                d.get("bbox", [0, 0, 0, 0])[0],
            )
        )

        for det in detections:
            cls = det.get("class_name", "")
            recognized = det.get("recognized", "") or det.get("text", "") or ""

            if cls == "text":
                lines.append(_escape_latex_text(recognized))

            elif cls == "formula":
                formula = recognized.strip()

                if formula:
                    lines.append(r"\[")
                    lines.append(formula)
                    lines.append(r"\]")
                else:
                    lines.append(r"\textit{[Formula was detected but not recognized]}")

            elif cls == "image":
                crop_path = det.get("crop_path")
                if crop_path:
                    crop_path = str(crop_path).replace("\\", "/")
                    lines.append(r"\begin{center}")
                    lines.append(rf"\includegraphics[width=0.85\linewidth]{{{crop_path}}}")
                    lines.append(r"\end{center}")
                else:
                    lines.append(r"\textit{[Image region detected]}")

        if page_index != len(pages):
            lines.append(r"\newpage")

    lines.append(r"\end{document}")

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path

def _compile_tex_to_pdf(tex_path: Path) -> Path:
    """
    Compile the LaTeX file to PDF using XeLaTeX.

    XeLaTeX (not pdflatex) is required: documents may contain Korean / Hanja /
    other CJK text, rendered via xeCJK + a CJK font (Noto Sans CJK KR). Needs a
    TeX install with xetex + the CJK font (see the Dockerfile).
    """
    import subprocess

    out_dir = tex_path.parent
    pdf_path = tex_path.with_suffix(".pdf")
    log_path = tex_path.with_suffix(".log")

    # NOTE: no `-halt-on-error`. XeLaTeX frequently exits non-zero on
    # recoverable issues (e.g. a single malformed recognised formula) while
    # still producing a perfectly usable PDF. We therefore judge success by
    # whether the PDF was written, not by the exit code. Two passes settle any
    # layout/box references.
    cmd = [
        "xelatex",
        "-interaction=nonstopmode",
        "-output-directory",
        str(out_dir),
        str(tex_path),
    ]

    proc = None
    for _ in range(2):
        proc = subprocess.run(
            cmd,
            cwd=str(out_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if not pdf_path.exists():
            break  # hard failure — don't bother with a second pass

    if not pdf_path.exists():
        # Surface the real cause: tail of the .log (falls back to stdout).
        tail = ""
        try:
            log_txt = log_path.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(log_txt.splitlines()[-40:])
        except OSError:
            tail = ((proc.stdout if proc else "") or "")[-2000:]
        rc = proc.returncode if proc else "?"
        raise RuntimeError(
            f"xelatex produced no PDF (exit {rc}). Log tail:\n{tail}"
        )

    return pdf_path

# --------------------------------------------------------------- HTML

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DocBank AI PDF Converter</title>
<style>
  :root {
    --blue-900: #0f172a;
    --blue-800: #1e3a8a;
    --blue-700: #1d4ed8;
    --blue-600: #2563eb;
    --blue-500: #3b82f6;
    --blue-100: #dbeafe;
    --blue-50: #eff6ff;
    --white: #ffffff;
    --gray-500: #64748b;
    --gray-200: #e2e8f0;
    --success: #10b981;
  }

  * {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    min-height: 100vh;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background:
      radial-gradient(circle at top left, rgba(59, 130, 246, 0.35), transparent 35%),
      linear-gradient(135deg, #eff6ff 0%, #dbeafe 40%, #ffffff 100%);
    color: var(--blue-900);
  }

  .page {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
  }

  .shell {
    width: 100%;
    max-width: 1050px;
    display: grid;
    grid-template-columns: 1fr 420px;
    gap: 32px;
    align-items: center;
  }

  .hero {
    padding: 20px;
  }

  .badge-top {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(37, 99, 235, 0.1);
    color: var(--blue-700);
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 24px;
  }

  h1 {
    font-size: 54px;
    line-height: 1.05;
    margin: 0 0 20px;
    letter-spacing: -1.5px;
  }

  h1 span {
    color: var(--blue-600);
  }

  .lead {
    font-size: 18px;
    line-height: 1.7;
    color: #334155;
    max-width: 620px;
    margin-bottom: 28px;
  }

  .features {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-top: 30px;
  }

  .feature {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 18px;
    padding: 16px;
    backdrop-filter: blur(10px);
  }

  .feature strong {
    display: block;
    color: var(--blue-800);
    margin-bottom: 4px;
  }

  .feature small {
    color: var(--gray-500);
    line-height: 1.4;
  }

  .card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 28px;
    padding: 28px;
    box-shadow:
      0 25px 60px rgba(37, 99, 235, 0.18),
      0 8px 24px rgba(15, 23, 42, 0.06);
    backdrop-filter: blur(20px);
  }

  .card h2 {
    margin: 0 0 8px;
    font-size: 26px;
  }

  .card p {
    margin: 0 0 22px;
    color: var(--gray-500);
    font-size: 14px;
    line-height: 1.5;
  }

  .drop {
    display: block;
    border: 2px dashed #93c5fd;
    border-radius: 22px;
    padding: 34px 22px;
    text-align: center;
    background: linear-gradient(180deg, #eff6ff, #ffffff);
    cursor: pointer;
    transition: 0.2s ease;
  }

  .drop:hover,
  .drop.over {
    border-color: var(--blue-600);
    background: #dbeafe;
    transform: translateY(-2px);
  }

  .drop input {
    display: none;
  }

  .upload-icon {
    width: 58px;
    height: 58px;
    margin: 0 auto 14px;
    border-radius: 18px;
    background: linear-gradient(135deg, var(--blue-600), var(--blue-500));
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 28px;
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.3);
  }

  .drop-title {
    font-weight: 700;
    color: var(--blue-900);
    margin-bottom: 6px;
  }

  .drop-subtitle {
    color: var(--gray-500);
    font-size: 13px;
  }

  .files {
    margin: 16px 0 0;
    padding: 0;
    text-align: left;
    color: #334155;
    font-size: 13px;
  }

  .files li {
    list-style: none;
    padding: 8px 10px;
    background: var(--blue-50);
    border-radius: 10px;
    margin-top: 6px;
  }

  .opts {
    margin-top: 22px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .field label {
    font-size: 13px;
    font-weight: 600;
    color: #334155;
  }

  .field input[type=number] {
    width: 100%;
    border: 1px solid var(--gray-200);
    border-radius: 12px;
    padding: 10px 12px;
    outline: none;
    color: var(--blue-900);
    background: white;
  }

  .field input[type=number]:focus {
    border-color: var(--blue-500);
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.12);
  }

  .checks {
    margin-top: 18px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }

  .checks label {
    font-size: 14px;
    color: #334155;
    background: #f8fafc;
    border: 1px solid var(--gray-200);
    padding: 9px 12px;
    border-radius: 999px;
  }

  button {
    width: 100%;
    margin-top: 22px;
    border: none;
    border-radius: 16px;
    padding: 15px 20px;
    font-size: 16px;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, var(--blue-700), var(--blue-500));
    cursor: pointer;
    box-shadow: 0 14px 30px rgba(37, 99, 235, 0.35);
    transition: 0.2s ease;
  }

  button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 38px rgba(37, 99, 235, 0.42);
  }

  button:disabled {
    background: #94a3b8;
    cursor: not-allowed;
    box-shadow: none;
  }

  .spinner {
    display: none;
    margin-top: 16px;
    padding: 12px 14px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: var(--blue-700);
    border-radius: 14px;
    font-size: 14px;
  }

  .spinner.on {
    display: block;
  }

  @media (max-width: 900px) {
    .shell {
      grid-template-columns: 1fr;
    }

    h1 {
      font-size: 40px;
    }

    .features {
      grid-template-columns: 1fr;
    }
  }
</style>
</head>

<body>
<div class="page">
  <div class="shell">

    <section class="hero">
      <div class="badge-top">AI-powered document reconstruction</div>

      <h1>Convert document images into <span>LaTeX PDF</span></h1>

      <p class="lead">
        Upload a photo, scanned page or PDF. The platform detects text blocks,
        formulas and images, recognizes their content and recreates the page as
        a downloadable PDF.
      </p>

      <div class="features">
        <div class="feature">
          <strong>YOLOv8 Layout Detection</strong>
          <small>Detects text, formulas and image regions using the trained DocBank model.</small>
        </div>
        <div class="feature">
          <strong>OCR + LaTeX OCR</strong>
          <small>Uses OCR for text and pix2tex for mathematical formulas.</small>
        </div>
        <div class="feature">
          <strong>Fast Local Processing</strong>
          <small>The model is already trained, so the web app only runs inference.</small>
        </div>
        <div class="feature">
          <strong>PDF Output</strong>
          <small>Returns a reconstructed PDF instead of only JSON results.</small>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Upload file</h2>
      <p>Supported formats: JPG, PNG, TIFF and PDF.</p>

      <form id="uploadForm" method="post" action="/upload" enctype="multipart/form-data">
        <label class="drop" id="drop">
          <input type="file" name="images" id="filesInput" multiple
                 accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.pdf">

          <div class="upload-icon">↑</div>
          <div class="drop-title">Click or drag files here</div>
          <div class="drop-subtitle">Your reconstructed PDF will be generated automatically.</div>

          <ul class="files" id="fileList"></ul>
        </label>

        <div class="opts">
          <div class="field">
            <label>Detection confidence</label>
            <input type="number" name="conf" min="0.05" max="0.95" step="0.05" value="0.25">
          </div>

          <div class="field">
            <label>OCR confidence</label>
            <input type="number" name="min_ocr_conf" min="0" max="1" step="0.05" value="0.30">
          </div>

          <div class="field">
            <label>Min text area</label>
            <input type="number" name="min_ocr_area" min="0" max="100000" step="100" value="600">
          </div>

          <div class="field">
            <label>Min formula area</label>
            <input type="number" name="min_formula_area" min="0" max="100000" step="100" value="2000">
          </div>
        </div>

        <div class="checks">
          <label><input type="checkbox" name="text_ocr" checked> Text OCR</label>
          <label><input type="checkbox" name="formula_ocr" checked> Formula OCR</label>
          <label><input type="checkbox" name="use_cache" checked> Cache</label>
        </div>

        <button id="submitBtn" type="submit">Generate PDF</button>

        <div class="spinner" id="spinner">
          Processing file… If this is the first run, loading OCR models may take a little longer.
        </div>
      </form>
    </section>

  </div>
</div>

<script>
  const drop = document.getElementById('drop');
  const inp  = document.getElementById('filesInput');
  const list = document.getElementById('fileList');
  const form = document.getElementById('uploadForm');
  const btn  = document.getElementById('submitBtn');
  const spin = document.getElementById('spinner');

  function refresh() {
    list.innerHTML = '';
    for (const f of inp.files) {
      const li = document.createElement('li');
      li.textContent = '• ' + f.name + ' (' + Math.round(f.size / 1024) + ' KB)';
      list.appendChild(li);
    }
  }

  inp.addEventListener('change', refresh);

  ['dragenter', 'dragover'].forEach(ev =>
    drop.addEventListener(ev, e => {
      e.preventDefault();
      drop.classList.add('over');
    })
  );

  ['dragleave', 'drop'].forEach(ev =>
    drop.addEventListener(ev, e => {
      e.preventDefault();
      drop.classList.remove('over');
    })
  );

  drop.addEventListener('drop', e => {
    inp.files = e.dataTransfer.files;
    refresh();
  });

  form.addEventListener('submit', e => {
    if (!inp.files.length) {
      e.preventDefault();
      return false;
    }

    btn.disabled = true;
    btn.textContent = 'Generating PDF…';
    spin.classList.add('on');
  });
</script>
</body>
</html>
"""



_RESULTS_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extraction results — {job_id}</title>
<script>
window.MathJax = {{tex: {{inlineMath: [['$','$']], displayMath: [['$$','$$']]}} }};
</script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 24px; background: #f5f6fa; color: #222; }}
  .top {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px; }}
  h1 {{ margin: 0; }}
  .meta {{ color: #666; font-size: 14px; }}
  .actions a {{ background: #1e88e5; color: #fff; padding: 8px 14px;
              border-radius: 6px; text-decoration: none; font-size: 14px;
              margin-right: 8px; }}
  .actions a.secondary {{ background: #555; }}
  .summary {{ background: #fff; border: 1px solid #e2e6ee; border-radius: 8px;
             padding: 14px 18px; margin: 14px 0 24px; display: inline-block; }}
  .summary span {{ display: inline-block; margin-right: 24px; }}
  .page {{ background: #fff; border: 1px solid #e2e6ee; border-radius: 8px;
          padding: 18px; margin-bottom: 28px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
  .page-img img {{ max-width: 100%; height: auto; border: 1px solid #d4d8e0; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
          margin-top: 18px; }}
  @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .det {{ border: 1px solid #e2e6ee; border-radius: 6px; padding: 10px;
         background: #fafbfc; display: flex; gap: 12px; }}
  .det img {{ max-width: 240px; max-height: 120px; object-fit: contain;
             border: 1px solid #d4d8e0; background: #fff; }}
  .det-meta {{ flex: 1; min-width: 0; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
           color: #fff; font-size: 12px; font-weight: 600; }}
  .badge.text    {{ background: #1e88e5; }}
  .badge.formula {{ background: #2e7d32; }}
  .badge.image   {{ background: #d32f2f; }}
  .conf {{ color: #666; font-size: 12px; margin-left: 6px; }}
  .recog {{ margin-top: 6px; white-space: pre-wrap; word-break: break-word;
           font-family: ui-monospace, Consolas, monospace; font-size: 13px;
           background: #fff; border: 1px solid #e2e6ee; padding: 8px;
           border-radius: 4px; max-height: 220px; overflow: auto; }}
  .recog.empty {{ color: #999; font-style: italic; }}
  .latex {{ background: #f3faf3; }}
</style>
</head>
<body>
"""


_RESULTS_TAIL = "</body></html>\n"


# --------------------------------------------------------------- helpers

def _summary(detections: list[dict]) -> dict:
    counts: dict[str, list[int]] = {}
    for d in detections:
        c = d.get("class_name", "?")
        counts.setdefault(c, [0, 0])[1] += 1
        if d.get("recognized"):
            counts[c][0] += 1
    return counts


def _render_results(
    job_id: str,
    job_dir: Path,
    pages: list[dict],
    elapsed_s: float,
) -> str:
    import html

    all_dets = [d for p in pages for d in p["detections"]]
    summary = _summary(all_dets)

    head = _RESULTS_HEAD.format(job_id=html.escape(job_id))
    parts: list[str] = [head]
    parts.append(
        f'<div class="top"><h1>Results</h1>'
        f'<div class="meta">job <code>{html.escape(job_id)}</code> '
        f'· {len(pages)} page(s) · {len(all_dets)} detection(s) '
        f'· {elapsed_s:.1f} s</div></div>'
    )
    parts.append('<div class="actions">')
    parts.append(f'<a href="/jobs/{job_id}/results.json" download>'
                 f'Download JSON</a>')
    parts.append(f'<a class="secondary" href="/">New upload</a>')
    parts.append('</div>')

    parts.append('<div class="summary">')
    parts.append(f'<span><b>Detections:</b> {len(all_dets)}</span>')
    for cls, (ok, tot) in summary.items():
        pct = 100 * ok / tot if tot else 0
        parts.append(
            f'<span><b>{html.escape(cls)}:</b> {ok}/{tot} '
            f'({pct:.0f}% recognised)</span>'
        )
    cached_count = sum(
        1 for d in all_dets if d.get("recognition_kind") == "cached"
    )
    if cached_count:
        parts.append(
            f'<span style="color:#2e7d32;"><b>Cache hits:</b> '
            f'{cached_count}/{len(all_dets)}</span>'
        )
    parts.append('</div>')

    for i, page in enumerate(pages, 1):
        boxed_rel = f"/jobs/{job_id}/boxes/{Path(page['boxed_image']).name}"
        parts.append(f'<div class="page">')
        parts.append(
            f'<h3>Page {i}: {html.escape(Path(page["image"]).name)}</h3>'
        )
        parts.append(
            f'<div class="page-img"><img src="{boxed_rel}" alt="page"></div>'
        )
        parts.append('<div class="grid">')
        for d in page["detections"]:
            parts.append(_render_detection_html(job_id, d))
        parts.append('</div></div>')

    parts.append(_RESULTS_TAIL)
    return "".join(parts)


def _render_detection_html(job_id: str, det: dict) -> str:
    import html

    cls = det.get("class_name", "?")
    conf = det.get("confidence", 0.0)
    recog = det.get("recognized") or ""
    bbox = det.get("bbox") or [0, 0, 0, 0]
    crop_w = max(0, int(bbox[2]) - int(bbox[0])) if len(bbox) >= 4 else 0
    crop_h = max(0, int(bbox[3]) - int(bbox[1])) if len(bbox) >= 4 else 0

    crop_html = '<span style="color:#999">[no crop]</span>'
    crop_path = det.get("crop_path")
    if crop_path:
        crop_url = f"/jobs/{job_id}/crops/{cls}/{Path(crop_path).name}"
        crop_html = f'<img src="{crop_url}" alt="crop">'

    if cls == "formula" and recog:
        body = f'<div class="recog latex">$$ {html.escape(recog)} $$</div>'
    elif recog:
        body = f'<div class="recog">{html.escape(recog)}</div>'
    else:
        if crop_w * crop_h < 600:
            note = f"(crop too small for OCR: {crop_w}×{crop_h}px)"
        elif conf < 0.35:
            note = f"(low-confidence detection, conf={conf:.2f})"
        else:
            note = "(no recognition output)"
        body = f'<div class="recog empty">{html.escape(note)}</div>'

    return (
        f'<div class="det">'
        f'  <div class="det-crop">{crop_html}</div>'
        f'  <div class="det-meta">'
        f'    <span class="badge {cls}">{html.escape(cls)}</span>'
        f'    <span class="conf">conf {conf:.2f} | bbox {bbox}</span>'
        f'    {body}'
        f'  </div>'
        f'</div>'
    )


# ---------------------------------------------------------------- pipeline

def _process_uploads(
    cfg: PipelineConfig,
    job_dir: Path,
    inputs: list[Path],
    *,
    conf: float,
    do_text: bool,
    do_formula: bool,
    min_ocr_conf: float = 0.30,
    min_ocr_area: int = 600,
    min_formula_area: int | None = 2000,
    use_cache: bool = True,
    max_text_crops: int | None = None,
    max_formulas: int | None = None,
) -> tuple[list[dict], float]:
    """Run YOLO + OCR on the uploaded pages. All artefacts under job_dir.

    Uses the same fast OCR path as the CLI:
      * filter low-conf / tiny crops (`min_ocr_conf`, `min_ocr_area`),
      * run text + formula recognition on two threads (real CPU parallelism),
      * persist a hash-keyed result cache shared across uploads.

    Returns (page_records, elapsed_seconds).
    """
    from .inference import run_yolo_inference
    from .ocr import _enrich, ocr_full_page  # fast path + full-page OCR

    # The DocBank-trained detector misses large text blocks on document styles
    # it never saw (textbooks, scans). When FULLPAGE_TEXT_OCR is on (default),
    # we take TEXT from PaddleOCR's own full-page detector instead of relying on
    # YOLO "text" boxes, and use YOLO only for image/formula regions.
    fullpage_text = (
        do_text
        and os.environ.get("FULLPAGE_TEXT_OCR", "1").strip().lower()
        not in ("0", "false", "no", "off", "")
    )

    boxes_dir = job_dir / "boxes"
    crops_dir = job_dir / "crops"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # `run_yolo_inference` writes crops under cfg.crops_dir, which derives
    # from cfg.output_dir. We redirect both at the per-job folder via a tiny
    # property-overriding shim so different uploads stay isolated.
    class _JobCfg:
        def __init__(self, base, jdir):
            self._b = base
            self._j = jdir
        def __getattr__(self, name):
            return getattr(self._b, name)
        @property
        def crops_dir(self):
            return self._j / "crops"
        @property
        def output_dir(self):
            return self._j

    job_cfg = _JobCfg(cfg, job_dir)

    t0 = time.time()
    detections = run_yolo_inference(
        job_cfg, inputs, conf=conf, save_crops=True
    )
    log.info("YOLO produced %d detection(s)", len(detections))

    # Shared OCR cache across uploads — saves running OCR on identical crops
    # someone has uploaded before. Lives under the package-wide outputs dir
    # (NOT per-job) so it persists across users.
    cache_path = cfg.output_dir / "web_ocr_cache.json"

    # In full-page mode YOLO text boxes are discarded, so don't waste time
    # OCR-ing them; _enrich still recognises formulas.
    _enrich(
        detections,
        do_text=do_text and not fullpage_text,
        do_formula=do_formula,
        min_ocr_conf=min_ocr_conf,
        min_ocr_area=min_ocr_area,
        min_formula_area=min_formula_area,
        use_cache=use_cache,
        cache_path=cache_path,
        max_text_crops=max_text_crops,
        max_formulas=max_formulas,
    )

    # Group per page + draw boxes
    by_image: dict[str, list[dict]] = {}
    for det in detections:
        by_image.setdefault(str(det["image"]), []).append(det)

    page_records: list[dict] = []
    for img_path in inputs:
        dets = by_image.get(str(img_path), [])

        if fullpage_text:
            # Keep YOLO's non-text regions (image/formula), drop its text boxes,
            # and rebuild text from a full-page OCR pass.
            non_text = [d for d in dets if d.get("class_name") != "text"]
            block_boxes = [
                d["bbox"] for d in non_text
                if d.get("class_name") in ("image", "formula") and d.get("bbox")
            ]
            text_dets: list[dict] = []
            for ln in ocr_full_page(img_path, min_conf=min_ocr_conf or 0.3):
                if _center_inside(ln["bbox"], block_boxes):
                    continue  # text sitting inside a figure/formula box
                text_dets.append({
                    "image": img_path,
                    "class_name": "text",
                    "confidence": ln["confidence"],
                    "bbox": ln["bbox"],
                    "recognized": ln["text"],
                    "recognition_kind": "fullpage_ocr",
                    "crop_path": None,
                })
            dets = text_dets + non_text
            log.info(
                "Page %s: %d full-page text line(s) + %d non-text region(s)",
                img_path.name, len(text_dets), len(non_text),
            )

        boxed = boxes_dir / img_path.name
        _draw_boxes(img_path, dets, boxed)
        page_records.append({
            "image": str(img_path),
            "boxed_image": str(boxed),
            "detections": [_serialise_det(d) for d in dets],
        })
    elapsed = time.time() - t0
    return page_records, elapsed


def _center_inside(inner: list, boxes: list[list]) -> bool:
    """True if the centre of `inner` bbox falls inside any box in `boxes`."""
    try:
        cx = (inner[0] + inner[2]) / 2.0
        cy = (inner[1] + inner[3]) / 2.0
    except (TypeError, IndexError):
        return False
    for b in boxes:
        try:
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                return True
        except (TypeError, IndexError):
            continue
    return False


def _serialise_det(det: dict) -> dict:
    """Convert Path objects to strings so json.dump works."""
    out = {}
    for k, v in det.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------- app

def create_app(cfg: PipelineConfig | None = None):
    """Build a Flask app bound to `cfg`. Lazy import so flask is optional."""
    try:
        from flask import (
            Flask, abort, request, send_file, send_from_directory,
        )
        from werkzeug.utils import secure_filename
    except ImportError as e:
        raise ImportError(
            "Flask is required for the web app. Install with: pip install flask"
        ) from e

    if cfg is None:
        cfg = PipelineConfig()

    cfg.ensure_dirs()

    web_root = cfg.output_dir / "web"
    web_root.mkdir(parents=True, exist_ok=True)

    app = Flask("docbank.webapp")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB / request

    @app.route("/", methods=["GET"])
    def index():
        return _INDEX_HTML

    @app.route("/upload", methods=["POST"])
    def upload():
        files = request.files.getlist("images")
        files = [f for f in files if f and f.filename]

        if not files:
            return ("No files uploaded.", 400)

        def _ffloat(name, default):
            try:
                return float(request.form.get(name, default))
            except (TypeError, ValueError):
                return default

        def _fint(name, default):
            try:
                return int(float(request.form.get(name, default)))
            except (TypeError, ValueError):
                return default

        def _opt_int(name):
            raw = request.form.get(name, "")
            if raw is None or str(raw).strip() == "":
                return None
            try:
                v = int(float(raw))
                return v if v > 0 else None
            except (TypeError, ValueError):
                return None

        conf_v = max(0.05, min(0.95, _ffloat("conf", 0.25)))
        min_ocr_conf = max(0.0, min(1.0, _ffloat("min_ocr_conf", 0.30)))
        min_ocr_area = max(0, _fint("min_ocr_area", 600))
        min_formula_area = max(0, _fint("min_formula_area", 2000)) or None
        max_formulas = _opt_int("max_formulas")

        do_text = "text_ocr" in request.form
        do_formula = "formula_ocr" in request.form
        use_cache = "use_cache" in request.form

        job_id = uuid.uuid4().hex[:12]
        job_dir = web_root / job_id
        inputs_dir = job_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        # Save uploads to disk; expand PDFs into images.
        saved: list[Path] = []

        for f in files:
            name = secure_filename(f.filename or "")

            if not name:
                continue

            ext = Path(name).suffix.lower()

            if ext not in ALLOWED_EXTS:
                continue

            target = inputs_dir / name
            f.save(target)

            if ext in PDF_EXTS:
                try:
                    pages_from_pdf = _explode_pdf(target, inputs_dir)
                except ImportError as e:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    return (str(e), 500)
                except Exception as e:
                    log.exception("PDF rendering failed for %s", name)
                    shutil.rmtree(job_dir, ignore_errors=True)
                    return (f"Failed to render PDF {name}: {e}", 400)

                saved.extend(pages_from_pdf)
            else:
                saved.append(target)

        if not saved:
            shutil.rmtree(job_dir, ignore_errors=True)
            return ("No valid image or PDF files.", 400)

        log.info(
            "Job %s: %d file(s), conf=%.2f, text=%s, formula=%s",
            job_id,
            len(saved),
            conf_v,
            do_text,
            do_formula,
        )

        try:
            pages, elapsed = _process_uploads(
                cfg,
                job_dir,
                saved,
                conf=conf_v,
                do_text=do_text,
                do_formula=do_formula,
                min_ocr_conf=min_ocr_conf,
                min_ocr_area=min_ocr_area,
                min_formula_area=min_formula_area,
                max_formulas=max_formulas,
                use_cache=use_cache,
            )
        except Exception as e:
            log.exception("Job %s failed", job_id)
            return (f"Pipeline error: {e}", 500)

        # This is the full structured result from the model.
        results_payload = {
            "job_id": job_id,
            "elapsed_seconds": round(elapsed, 2),
            "pages": pages,
        }

        # Save results.json as backup/debug output.
        results_json_path = job_dir / "results.json"
        results_json_path.write_text(
            json.dumps(results_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Convert model results to LaTeX.
        tex_path = job_dir / "result.tex"

        try:
            _results_to_tex(results_payload, tex_path)
        except Exception as e:
            log.exception("Failed to create LaTeX for job %s", job_id)
            return (f"Failed to create LaTeX file: {e}", 500)

        # Compile LaTeX to PDF.
        try:
            pdf_path = _compile_tex_to_pdf(tex_path)
        except Exception as e:
            log.exception("Failed to compile PDF for job %s", job_id)
            return (
                "The model processed the file, but PDF compilation failed. "
                f"Error: {e}. "
                "Make sure TeX Live with XeLaTeX (xelatex) and a CJK font "
                "(Noto Sans CJK KR) is installed.",
                500,
            )

        log.info(
            "Job %s finished in %.2fs. PDF: %s",
            job_id,
            elapsed,
            pdf_path,
        )

        # Return final PDF directly to user.
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name="reconstructed_document.pdf",
            mimetype="application/pdf",
        )

    @app.route("/jobs/<job_id>/results", methods=["GET"])
    @app.route("/jobs/<job_id>", methods=["GET"])
    def results(job_id: str):
        job_dir = web_root / secure_filename(job_id)
        rj = job_dir / "results.json"

        if not rj.is_file():
            abort(404)

        payload = json.loads(rj.read_text(encoding="utf-8"))

        html_doc = _render_results(
            job_id,
            job_dir,
            payload["pages"],
            payload.get("elapsed_seconds", 0),
        )

        return html_doc

    @app.route("/jobs/<job_id>/results.json")
    def results_json(job_id: str):
        job_dir = web_root / secure_filename(job_id)
        rj = job_dir / "results.json"

        if not rj.is_file():
            abort(404)

        return send_file(
            rj,
            mimetype="application/json",
            as_attachment=True,
            download_name=f"docbank_{job_id}.json",
        )

    @app.route("/jobs/<job_id>/result.pdf")
    def result_pdf(job_id: str):
        job_dir = web_root / secure_filename(job_id)
        pdf_path = job_dir / "result.pdf"

        if not pdf_path.is_file():
            abort(404)

        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="reconstructed_document.pdf",
        )

    @app.route("/jobs/<job_id>/<path:rel>")
    def job_file(job_id: str, rel: str):
        job_dir = web_root / secure_filename(job_id)

        if not job_dir.is_dir():
            abort(404)

        return send_from_directory(job_dir, rel)

    return app


def serve(
    cfg: PipelineConfig | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
) -> None:
    """Convenience entry-point for the CLI.

    For production (concurrent classmates uploading at once) we prefer
    `waitress`, which is a real WSGI server with a thread pool. If it isn't
    installed we fall back to Flask's dev server with `threaded=True`, which
    is enough for a small group on a LAN.
    """
    app = create_app(cfg)
    log.info("Starting webapp on http://%s:%d (data_root=%s)",
             host, port, (cfg or PipelineConfig()).data_root)
    if not debug:
        try:
            from waitress import serve as waitress_serve  # type: ignore
            log.info("Using waitress with 8 worker threads.")
            waitress_serve(app, host=host, port=port, threads=8)
            return
        except ImportError:
            log.info("waitress not installed (pip install waitress for "
                     "better concurrency); using Flask dev server in threaded mode.")
    app.run(host=host, port=port, debug=debug, threaded=True)
