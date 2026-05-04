# DocBank → YOLO → OCR pipeline

A small, resumable pipeline that:

1. Downloads the [DocBank](https://huggingface.co/datasets/liminghao1630/DocBank) dataset.
2. Converts its COCO-format annotations into YOLO format (collapsing the
   12 fine-grained classes into **`text` / `formula` / `image`**).
3. Trains YOLOv8 to detect those layout regions.
4. Runs detection + crop + preprocessing + recognition on new pages:
   - **PaddleOCR** for `text` and figure captions
   - **LaTeXOCR / pix2tex** for `formula`
5. Saves a structured JSON with bounding boxes, confidences and recognised
   text/LaTeX.

Every long-running stage is **resumable**: each writes a `.<stage>.done`
marker, and the next run skips work that's already finished. Delete the
marker (or pass `--force` where supported) to redo a stage.

---

## Installation

```bash
# 1. Python 3.10+ recommended
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 2. Core deps
pip install -r requirements.txt
```

### 7-Zip (recommended)

The DocBank image archive is a 10-part split zip
(`DocBank_500K_ori_img.zip.001` … `.010`). The pipeline uses **7-Zip** to
extract it because that's the only tool that handles partial subsets
gracefully.

| OS       | Install                                      |
|----------|----------------------------------------------|
| Windows  | <https://www.7-zip.org/> (or `winget install 7zip.7zip`) |
| macOS    | `brew install p7zip`                         |
| Ubuntu   | `sudo apt install p7zip-full`                |

If 7-Zip is missing **and** you've downloaded all 10 parts, the pipeline
falls back to a pure-Python concat-and-extract, which works but is slower.

---

## Configuration

All paths and knobs live in `docbank_pipeline/config.py` and can be
overridden via env vars or CLI flags:

| Env var          | Default                  | Meaning                         |
|------------------|--------------------------|----------------------------------|
| `DATA_ROOT`      | `./DocBank`              | Project root                     |
| `MAX_PAGES`      | _(unset)_                | Cap pages per split during convert / inference |
| `DATASET_PARTS`  | `1`                      | How many of the 10 image parts to download (1–10) |
| `NUM_WORKERS`    | `4`                      | Worker threads for YOLO          |
| `TRAIN_VAL_SPLIT`| `0.9`                    | Used only if you re-split        |
| `HF_TOKEN`       | _(unset)_                | HuggingFace auth (faster + higher rate limits) |

Folders derived from `DATA_ROOT`:

```
DocBank/
├── raw/             # downloaded .zip parts
├── annotations/     # extracted COCO json files
├── images/          # extracted .jpg pages
├── yolo_dataset/    # YOLO labels + symlinked images + data.yaml
├── outputs/
│   ├── crops/       # cropped detections per class
│   └── results.json
└── runs/            # ultralytics training runs
```

---

## How to run

### Option 1 — Notebook
Open `docbank_local_setup.ipynb` in Jupyter / Colab / VS Code and run the
cells top-to-bottom.

### Option 2 — CLI
```bash
python -m docbank_pipeline <subcommand> [opts]
```

Show resolved paths:
```bash
python -m docbank_pipeline info
```

---

### Small test run (5–20 pages, end-to-end)
```bash
python -m docbank_pipeline smoketest --n 10 --epochs 2
```
Downloads 1 archive part (~1 GB), converts a tiny subset, trains for 2
epochs on a tiny model, then runs detection + OCR + LaTeX-OCR. Use
`--skip-train` to skip training entirely (uses any pre-existing weights
under `runs/`).

### Dataset conversion only
```bash
python -m docbank_pipeline download --dataset-parts 1
python -m docbank_pipeline extract
python -m docbank_pipeline convert --max-pages 1000
```
Each step is idempotent: rerun freely.

### YOLO training only
```bash
python -m docbank_pipeline train --epochs 50 --batch 8 --imgsz 640
```
Add `--device cpu` to force CPU. With no `--device`, the pipeline picks
CUDA / MPS / CPU automatically.

### Inference + OCR only
```bash
python -m docbank_pipeline infer \
    --source path/to/page_images/ \
    --weights DocBank/runs/yolov8_docbank/weights/best.pt \
    --output DocBank/outputs/my_results.json
```
Add `--no-text-ocr` or `--no-formula-ocr` to skip a recognition stage.

---

## What changed and why

| Issue in the old notebook | Fix |
|---|---|
| `Path(image_dir).rglob(filename)` called inside the per-image loop made conversion O(N×M). The notebook log showed ~27 s per page. | Build a single `{filename → path}` index up front in `convert._build_image_index`. Inner loop is O(1). |
| Extraction shelled out to `zip -F` / `7z` / `brew install p7zip`. Failed on Windows; no skip-existing logic. | `download.extract_archives` detects 7-Zip on PATH (or the standard install dirs), falls back to a pure-Python concat-and-`zipfile` path when all parts are present, and writes a `.extract.done` marker. |
| No resume: every cell re-ran from scratch. | Each stage drops a `.done` sentinel; `link_or_copy` skips already-copied images; conversion skips images whose label already exists. |
| Hardcoded Google-Drive paths in the Colab notebook. | All paths derive from `PipelineConfig.data_root` (env: `DATA_ROOT`). The Colab notebook can still mount Drive, but only sets `DATA_ROOT` to a Drive path — no other change needed. |
| `device=0` hardcoded in training. | `utils.detect_device()` returns `cuda` / `mps` / `cpu`. CLI takes `--device` to override. |
| Subset extraction silently produced 5 % coverage because the central directory was missing. | The pipeline either uses 7-Zip (handles partial archives) or refuses politely with a clear message when 7-Zip is missing and only some parts are present. |
| Annotation file names drifted between cells (`train.json` vs `500K_train.json`) and crashed. | Single source of truth: `convert.COCO_FILES`. |
| OCR / LaTeX recognition stages didn't exist. | New `ocr.py` with `preprocess_crop`, `recognize_text_with_paddleocr`, `recognize_formula_with_latexocr`, `process_page_image`, `process_folder`, `save_results_json`. Heavy deps are lazy imports — missing models warn and skip rather than crash. |
| One huge cell mixed download, extract, convert. | Split into `download.py`, `convert.py`, `train.py`, `inference.py`, `ocr.py` with a thin CLI. |

---

## Known assumptions / things to double-check

These are the places where you may need to adjust based on your environment:

1. **Disk space.** Each archive part is ~5 GB compressed; the whole 10-part
   archive is 55 GB and unpacks to ~75 GB of JPGs. Aim for ≥120 GB free if
   you plan to use everything.
2. **HF rate limits.** Without `HF_TOKEN`, large downloads frequently get
   throttled. Set `export HF_TOKEN=hf_xxx` (or pass it via Colab secrets)
   before downloading.
3. **PaddleOCR + paddlepaddle.** First call downloads ~300 MB of weights.
   GPU users should replace `paddlepaddle` with `paddlepaddle-gpu` in
   `requirements.txt`.
4. **pix2tex.** Requires a recent PyTorch. On Apple Silicon use
   `pip install torch torchvision` (CPU/MPS) before `pip install pix2tex`.
5. **Windows hardlinks.** `link_or_copy` falls back to `shutil.copy2` if
   hardlinking fails (e.g. on FAT32 / different drives).
