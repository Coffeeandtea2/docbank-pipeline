# Deploying the DocBank web app

The app is a Flask service (`python -m docbank_pipeline serve`) packaged as a
**single portable Docker image** (`Dockerfile`). It runs the full pipeline:

- **YOLOv8** layout detection (weights `best.pt` are baked into the image)
- **PaddleOCR** text recognition
- **pix2tex** formula → LaTeX
- **pdflatex** (TeX Live) → reconstructed PDF

### Resource reality check
The full stack needs **~3–4 GB RAM**. The first upload downloads OCR model
weights (~300 MB+) to `/tmp`, so the *first* request is slow; later ones are
fast (singletons are cached in-process). Anything with <2 GB RAM will OOM.

| Host | RAM | Cost | Notes |
|------|-----|------|-------|
| **Hugging Face Spaces (Docker)** | 16 GB | **Free** | ✅ Recommended. ML-native, plenty of RAM, sleeps when idle. |
| Render (Pro) | 4 GB | Paid (~$85/mo) | Works via `render.yaml`. Free/Starter will OOM. |
| Fly.io | configurable | Pay-as-you-go | Set VM to 4 GB. |
| Google Cloud Run | up to 32 GB | Pay-per-request | Scales to zero; heavy cold starts. |

---

## Option A — Hugging Face Spaces (recommended, free, 16 GB)

1. Create a Space: https://huggingface.co/new-space → **SDK: Docker**, **Blank**.
2. Add this YAML block to the **top** of the Space's `README.md` (HF reads it to
   know the SDK and port):

   ```yaml
   ---
   title: DocBank Pipeline
   emoji: 📄
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   ---
   ```
3. Push this repo's contents to the Space's git remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
   (Use an HF access token with *write* scope when prompted for a password.)
4. HF builds the Dockerfile and serves on port `7860`. Done.

> `best.pt` (6 MB) is committed, so the Space has the detection model with no
> extra setup.

---

## Option B — Render (paid Pro)

This repo includes `render.yaml`.

1. Push to GitHub (already done).
2. Render dashboard → **New + → Blueprint** → select this repo → **Apply**.
3. It provisions a Docker web service on the **Pro** plan (4 GB). Render injects
   `$PORT`; the app binds to it automatically.

If you want a smaller/cheaper plan, drop the OCR libs from `requirements.txt`
(remove `paddleocr`, `paddlepaddle`, `pix2tex`) — detection + PDF will fit ~2 GB,
but text/formula recognition is disabled.

---

## Option C — Fly.io

```bash
fly launch --no-deploy          # generates fly.toml; pick a name/region
fly scale memory 4096           # 4 GB RAM
fly deploy
```
The `Dockerfile` is used as-is. Set the internal port to `7860` in `fly.toml`.

---

## Environment variables the image understands

| Var | Default (in image) | Purpose |
|-----|--------------------|---------|
| `PORT` | `7860` | Port to bind (Render injects its own). |
| `WEIGHTS` | `/app/best.pt` | Trained YOLO detection weights. |
| `YOLO_IMGSZ` | `960` | Inference image size. Matches the model's training resolution (`args.yaml`); 640 hurts small-text recall. |
| `DATA_ROOT` | `/tmp/docbank` | Writable dir for per-job artefacts + caches. |
| `HF_TOKEN` | _(unset)_ | Only needed for dataset/training, not inference. |

## Local Docker test

```bash
docker build -t docbank .
docker run --rm -p 7860:7860 docbank
# open http://localhost:7860
```
