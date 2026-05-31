# DocBank pipeline web app — portable container.
# Runs on any Docker host: Hugging Face Spaces (Docker), Render, Fly.io,
# Google Cloud Run, Railway, etc.
#
# Full stack: YOLOv8 detection + PaddleOCR text + pix2tex formulas + LaTeX->PDF.
# Needs ~3-4 GB RAM. The first upload downloads OCR model weights (~300 MB+).

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Where the app writes per-job artefacts + caches (must be writable).
    DATA_ROOT=/tmp/docbank \
    # Baked-in trained detection weights (read by inference._resolve_weights).
    WEIGHTS=/app/best.pt \
    # Match the detector's training resolution (args.yaml: imgsz=960).
    YOLO_IMGSZ=960 \
    # PaddleOCR recognition language (Korean docs with Hanja). Override as needed.
    OCR_LANG=korean \
    # Keep every model/cache download under writable /tmp (HF Spaces runs as
    # a non-root user; /app is read-only at runtime there).
    HF_HOME=/tmp/cache/hf \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/cache/mpl \
    # HF Spaces runs as a non-root user; give PaddleOCR (~/.paddleocr) a
    # writable HOME so its model download doesn't fail at runtime.
    HOME=/tmp/home \
    # Default port; platforms that inject $PORT (Render) override at runtime.
    PORT=7860

# ---- system deps: pdflatex (TeX Live) + runtime libs for opencv/paddle ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra \
        texlive-fonts-recommended lmodern \
        texlive-xetex texlive-lang-cjk fonts-noto-cjk \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- python deps ----
# Install CPU-only torch first so ultralytics + pix2tex reuse it instead of
# pulling the multi-GB CUDA wheel.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

# ---- app code + baked model weights ----
COPY . .

# Make runtime cache/output dirs world-writable (non-root hosts like HF Spaces).
RUN mkdir -p /tmp/docbank /tmp/cache/hf /tmp/cache/mpl /tmp/home \
    && chmod -R 777 /tmp/docbank /tmp/cache /tmp/home

EXPOSE 7860

# Bind to 0.0.0.0 and the platform-provided $PORT. serve() uses waitress
# (a production WSGI server with a thread pool) since it's in requirements.txt.
CMD ["sh", "-c", "python -m docbank_pipeline serve --host 0.0.0.0 --port ${PORT:-7860}"]
