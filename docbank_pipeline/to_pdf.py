"""
docbank_pipeline/to_pdf.py

Turn an uploaded page image into a searchable PDF using the full
detection + OCR + formula-recognition pipeline.

The output PDF is visually identical to the input image (the image is
drawn as the page background); the recognized text and formula LaTeX
are overlaid as invisible, selectable / Ctrl-F-searchable text.

Public API:
    image_to_pdf(cfg, image_path, output_pdf)            -> Path
    detections_to_searchable_pdf(detections, output_pdf) -> Path

Requires: pip install reportlab
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "reportlab is required to build PDFs. Install with: pip install reportlab"
    ) from e

from .config import PipelineConfig
from .ocr import process_page_image


def _draw_page(c, img_path: str, detections: Sequence[dict], *, invisible_text: bool) -> None:
    """Draw a single PDF page: the source image as background, with each
    detection's recognized text overlaid at its bbox."""
    with Image.open(img_path) as im:
        page_w, page_h = im.size

    # Use the image's pixel size as the page size so bboxes map 1:1.
    c.setPageSize((page_w, page_h))
    c.drawImage(ImageReader(img_path), 0, 0, page_w, page_h)

    for det in detections:
        text = (det.get("recognized") or "").strip()
        if not text:
            continue

        x1, y1, x2, y2 = (float(v) for v in det.get("bbox", [0, 0, 0, 0]))
        box_h = max(1.0, y2 - y1)

        lines = text.splitlines() or [text]
        line_h = box_h / max(1, len(lines))
        # 0.85 of the line height is a decent visual match for most fonts.
        font_size = max(4.0, line_h * 0.85)

        # Image coords are top-down; PDF coords are bottom-up.
        pdf_y_top = page_h - y1

        t = c.beginText()
        t.setFont("Helvetica", font_size)
        if invisible_text:
            t.setTextRenderMode(3)  # invisible, but still indexed/selectable
        t.setLeading(font_size * 1.15)
        t.setTextOrigin(x1, pdf_y_top - font_size)
        for line in lines:
            # Helvetica is latin-1 only; the visible content comes from the
            # background image anyway, so drop glyphs it can't encode rather
            # than crash. Register a Unicode TTF (see module docstring note
            # below) if you need exact Greek/math in the *searchable* layer.
            t.textLine(line.encode("latin-1", "replace").decode("latin-1"))
        c.drawText(t)


def detections_to_searchable_pdf(
    detections: Sequence[dict],
    output_pdf: str | Path,
    *,
    invisible_text: bool = True,
) -> Path:
    """Render a flat list of enriched detections into a searchable PDF.

    Each detection must carry an ``image`` key (its source page path), a
    ``bbox`` ``[x1, y1, x2, y2]`` in image pixels, and a ``recognized``
    string. One PDF page is produced per distinct source image, in
    first-seen order.
    """
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Group detections by source image, preserving first-seen order.
    pages: dict[str, list[dict]] = {}
    for det in detections:
        img = det.get("image") or det.get("image_path") or ""
        pages.setdefault(str(img), []).append(det)

    c = canvas.Canvas(str(output_pdf))
    for img_path, dets in pages.items():
        if not img_path or not Path(img_path).is_file():
            continue
        _draw_page(c, img_path, dets, invisible_text=invisible_text)
        c.showPage()
    c.save()
    return output_pdf


def image_to_pdf(
    cfg: PipelineConfig,
    image_path: str | Path,
    output_pdf: str | Path,
    *,
    conf: float = 0.25,
    do_text: bool = True,
    do_formula: bool = True,
    invisible_text: bool = True,
) -> Path:
    """End-to-end: detect + OCR + formula-recognise a single uploaded image,
    then write a searchable PDF version of it.

    Parameters
    ----------
    cfg          : your PipelineConfig.
    image_path   : the uploaded page image.
    output_pdf   : where to write the PDF.
    conf         : YOLO detection confidence threshold.
    invisible_text : True  -> overlay is invisible (PDF looks like the
                              original image but is searchable).
                     False -> overlay is drawn visibly on top of the image,
                              useful for checking OCR alignment.

    Returns the path to the written PDF.
    """
    detections = process_page_image(
        cfg, image_path,
        conf=conf, do_text=do_text, do_formula=do_formula,
    )
    # Ensure every detection knows its source image so the renderer can
    # group correctly (process_page_image runs on exactly one page).
    for det in detections:
        det.setdefault("image", str(image_path))

    return detections_to_searchable_pdf(
        detections, output_pdf, invisible_text=invisible_text,
    )
