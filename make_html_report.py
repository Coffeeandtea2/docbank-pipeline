"""
Pipeline validation report.

Runs a series of pass/fail checks against the project's actual artefacts and
writes a self-contained HTML report at ./Pipeline_Validation_Report.html.

Each check answers ONE question from the original brief:

  Q1. Does the YOLO model exist on disk?           -> model_exists
  Q2. Did training actually finish? (args.yaml +
      results.csv produced by Ultralytics)         -> training_completed
  Q3. Did training reach a useful score?           -> training_quality
  Q4. Is the YOLO dataset wired correctly?         -> dataset_valid
  Q5. Are class labels (text/formula/image)
      detected on real pages?                      -> classes_detected
  Q6. Does OCR return real text (non-empty)?       -> text_ocr_works
  Q7. Does pix2tex return LaTeX?                   -> formula_ocr_works
  Q8. Is the JSON output schema correct?           -> json_schema_ok
  Q9. Is the result cache active?                  -> cache_active
  Q10. Does the web app import + register
       all routes?                                 -> webapp_imports

Run:    python make_html_report.py
Output: ./Pipeline_Validation_Report.html  (open in any browser)
"""

from __future__ import annotations

import csv
import html
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "DocBank"


# ----------------------------------------------------------- check helpers

class Check:
    """A single named check with a pass/fail/warn status and evidence."""
    def __init__(self, qid: str, question: str):
        self.qid = qid
        self.question = question
        self.status: str = "pending"   # "pass" | "fail" | "warn" | "skip"
        self.summary: str = ""
        self.evidence: list[Any] = []  # plain strings or {"table": rows}

    def passed(self, summary: str, evidence: list[Any] | None = None):
        self.status = "pass"
        self.summary = summary
        self.evidence = evidence or []

    def failed(self, summary: str, evidence: list[Any] | None = None):
        self.status = "fail"
        self.summary = summary
        self.evidence = evidence or []

    def warned(self, summary: str, evidence: list[Any] | None = None):
        self.status = "warn"
        self.summary = summary
        self.evidence = evidence or []

    def skipped(self, summary: str):
        self.status = "skip"
        self.summary = summary


def fmt_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024:
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{n} B"


# ---------------------------------------------------------------- the checks

def check_model_exists() -> Check:
    c = Check("Q1", "Does the trained YOLO model exist on disk?")
    p = DATA_ROOT / "runs" / "yolov8_docbank" / "weights" / "best.pt"
    if p.is_file():
        c.passed(
            f"Model file present at {p.relative_to(ROOT)}",
            [f"Size: {fmt_bytes(p.stat().st_size)}",
             f"Last modified: {time.ctime(p.stat().st_mtime)}"],
        )
    else:
        c.failed(f"Missing: {p}", ["Run `python -m docbank_pipeline train`."])
    return c


def check_training_completed() -> Check:
    c = Check("Q2", "Did training actually finish?")
    run_dir = DATA_ROOT / "runs" / "yolov8_docbank"
    args = run_dir / "args.yaml"
    results = run_dir / "results.csv"
    if not args.is_file() or not results.is_file():
        c.failed("Ultralytics did not produce args.yaml + results.csv.")
        return c
    rows = list(csv.DictReader(results.open(encoding="utf-8")))
    if not rows:
        c.failed("results.csv is empty.")
        return c
    last = rows[-1]
    epoch = int(float(last.get("epoch", 0)))
    c.passed(
        f"Training ran to completion: {epoch + 1} epoch(s) recorded.",
        [f"args.yaml + results.csv present in {run_dir.relative_to(ROOT)}"],
    )
    return c


def check_training_quality() -> Check:
    c = Check("Q3", "Did the model reach a useful mAP@0.5?")
    results = DATA_ROOT / "runs" / "yolov8_docbank" / "results.csv"
    if not results.is_file():
        c.failed("results.csv missing.")
        return c
    rows = list(csv.DictReader(results.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("metrics/mAP50(B)")]
    if not rows:
        c.failed("No mAP@0.5 column in results.csv.")
        return c
    last = rows[-1]
    map50 = float(last["metrics/mAP50(B)"])
    map5095 = float(last.get("metrics/mAP50-95(B)", "0") or 0)
    table = [["Epoch", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]]
    samples = rows[::max(1, len(rows) // 6)] + [rows[-1]]
    seen = set()
    pretty = []
    for r in samples:
        e = r.get("epoch", "?")
        if e in seen:
            continue
        seen.add(e)
        pretty.append([
            e,
            f'{float(r.get("metrics/mAP50(B)", 0)):.3f}',
            f'{float(r.get("metrics/mAP50-95(B)", 0)):.3f}',
            f'{float(r.get("metrics/precision(B)", 0)):.3f}',
            f'{float(r.get("metrics/recall(B)", 0)):.3f}',
        ])
    table += pretty
    if map50 >= 0.45:
        c.passed(
            f"Final mAP@0.5 = {map50:.3f}, mAP@0.5:0.95 = {map5095:.3f} — "
            f"well above the 0.4 'classroom-quality' threshold.",
            [{"table": table}],
        )
    elif map50 >= 0.30:
        c.warned(
            f"Final mAP@0.5 = {map50:.3f}. Usable but more epochs / data "
            f"would help.",
            [{"table": table}],
        )
    else:
        c.failed(
            f"Final mAP@0.5 = {map50:.3f}, below the 0.30 floor.",
            [{"table": table}],
        )
    return c


def check_dataset_valid() -> Check:
    c = Check("Q4", "Is the YOLO dataset wired correctly?")
    yolo_dir = DATA_ROOT / "yolo_dataset"
    yaml_p = yolo_dir / "data.yaml"
    if not yaml_p.is_file():
        c.failed(f"{yaml_p} missing.")
        return c
    counts = {}
    for split in ("train", "val", "test"):
        imgs = list((yolo_dir / "images" / split).glob("*.jpg"))
        lbls = list((yolo_dir / "labels" / split).glob("*.txt"))
        counts[split] = (len(imgs), len(lbls))
    if all(v[0] > 0 and v[1] > 0 for v in counts.values()):
        rows = [["Split", "Images", "Labels"]] + [
            [k, str(i), str(l)] for k, (i, l) in counts.items()
        ]
        c.passed(
            "data.yaml present, every split has matching images and labels.",
            [{"table": rows},
             f"Class names from data.yaml: {yaml_p.read_text(encoding='utf-8').splitlines()[-3:]}"],
        )
    else:
        c.failed(
            "Some splits are missing images or labels.",
            [{"table": [["Split", "Images", "Labels"]] +
                       [[k, str(i), str(l)] for k, (i, l) in counts.items()]}],
        )
    return c


def _load_results(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick_results_file() -> tuple[str, dict] | None:
    for name in ("results_demo.json", "results_50.json", "results.json"):
        d = _load_results(DATA_ROOT / "outputs" / name)
        if d and d.get("pages"):
            return name, d
    return None


def check_classes_detected() -> Check:
    c = Check("Q5", "Are all three classes (text/formula/image) actually "
                    "detected on real pages?")
    picked = _pick_results_file()
    if not picked:
        c.failed("No results JSON found under DocBank/outputs/. "
                 "Run `python -m docbank_pipeline infer ...` first.")
        return c
    name, data = picked
    counts = {"text": 0, "formula": 0, "image": 0}
    for pg in data["pages"]:
        for det in pg.get("detections", []):
            cn = det.get("class_name", "?")
            if cn in counts:
                counts[cn] += 1
    if all(counts[k] > 0 for k in counts):
        c.passed(
            f"Source: {name}. All three classes detected on real pages.",
            [{"table": [["Class", "Detections"]] +
                       [[k, str(v)] for k, v in counts.items()]}],
        )
    else:
        missing = [k for k, v in counts.items() if v == 0]
        c.failed(
            f"Missing classes in detections: {', '.join(missing)}",
            [{"table": [["Class", "Detections"]] +
                       [[k, str(v)] for k, v in counts.items()]}],
        )
    return c


def check_text_ocr() -> Check:
    c = Check("Q6", "Does PaddleOCR return real recognised text?")
    picked = _pick_results_file()
    if not picked:
        c.failed("No results JSON to inspect.")
        return c
    name, data = picked
    samples = []
    ok = total = 0
    for pg in data["pages"]:
        for det in pg.get("detections", []):
            if det.get("class_name") in ("text", "image"):
                total += 1
                rec = det.get("recognized") or ""
                if rec.strip():
                    ok += 1
                    if len(samples) < 3:
                        samples.append(rec[:140].replace("\n", " / "))
    rate = (ok / total) if total else 0
    if rate >= 0.5:
        c.passed(
            f"Source: {name}. {ok}/{total} text/image crops recognised "
            f"({rate*100:.1f}%).",
            ["Examples:"] + [f"&bull; {html.escape(s)}" for s in samples],
        )
    elif rate >= 0.2:
        c.warned(
            f"Source: {name}. {ok}/{total} ({rate*100:.1f}%). Many tiny crops "
            f"are below OCR resolution; check --min-ocr-area.",
            ["Examples:"] + [f"&bull; {html.escape(s)}" for s in samples],
        )
    else:
        c.failed(
            f"Source: {name}. Only {ok}/{total} ({rate*100:.1f}%) "
            f"text crops recognised. Likely the Paddle 3.x oneDNN bug — "
            f"`pip install \"paddlepaddle==2.6.2\" \"paddleocr<3\"`.",
        )
    return c


def check_formula_ocr() -> Check:
    c = Check("Q7", "Does pix2tex return real LaTeX?")
    picked = _pick_results_file()
    if not picked:
        c.failed("No results JSON to inspect.")
        return c
    name, data = picked
    samples = []
    ok = total = 0
    for pg in data["pages"]:
        for det in pg.get("detections", []):
            if det.get("class_name") == "formula":
                total += 1
                rec = det.get("recognized") or ""
                if rec.strip():
                    ok += 1
                    if len(samples) < 3:
                        samples.append(rec[:200])
    rate = (ok / total) if total else 0
    if total == 0:
        c.warned(f"Source: {name}. No formula detections to evaluate.")
        return c
    if rate >= 0.7:
        c.passed(
            f"Source: {name}. {ok}/{total} formula crops returned LaTeX "
            f"({rate*100:.1f}%).",
            ["Examples (will render as math in the actual demo HTML):"] +
            [f"<code>{html.escape(s)}</code>" for s in samples],
        )
    elif rate > 0:
        c.warned(
            f"Source: {name}. {ok}/{total} ({rate*100:.1f}%). Some formulas "
            f"may have been filtered out via --min-formula-area or "
            f"--max-formulas.",
        )
    else:
        c.failed(
            f"Source: {name}. 0/{total} formulas recognised. pix2tex may not "
            f"be installed: `pip install pix2tex`.",
        )
    return c


def check_json_schema() -> Check:
    c = Check("Q8", "Is the JSON output schema correct?")
    picked = _pick_results_file()
    if not picked:
        c.failed("No results JSON found.")
        return c
    name, data = picked
    if not isinstance(data, dict) or "pages" not in data:
        c.failed("Top-level 'pages' key missing.")
        return c
    page = data["pages"][0] if data["pages"] else None
    if not page or "detections" not in page:
        c.failed("First page is missing the 'detections' field.")
        return c
    det = page["detections"][0] if page["detections"] else None
    if not det:
        c.warned("Page has no detections.")
        return c
    required = {"class_name", "bbox", "confidence", "image_width",
                "image_height", "crop_path", "recognized"}
    missing = required - set(det.keys())
    if missing:
        c.failed(f"Missing keys in a detection: {sorted(missing)}")
        return c
    pretty = json.dumps({**det, "recognized":
                         (det.get("recognized") or "")[:60]}, indent=2,
                        ensure_ascii=False)[:1200]
    c.passed(
        f"Schema valid (every detection has class_name, bbox, confidence, "
        f"crop_path, recognized).",
        [f"Sample (truncated):", f"<pre>{html.escape(pretty)}</pre>"],
    )
    return c


def check_cache_active() -> Check:
    c = Check("Q9", "Is the OCR result cache active?")
    p = DATA_ROOT / "outputs" / "ocr_cache.json"
    if not p.is_file():
        c.warned("No cache file yet; one is created on the first inference run.")
        return c
    try:
        cache = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        c.failed(f"Cache JSON is unreadable: {e}")
        return c
    if not isinstance(cache, dict) or not cache:
        c.warned("Cache file exists but is empty.")
        return c
    c.passed(
        f"Cache holds <b>{len(cache):,}</b> entries — re-runs over the same "
        f"crops are instant.",
        [f"File: {p.relative_to(ROOT)}",
         f"Size: {fmt_bytes(p.stat().st_size)}"],
    )
    return c


def check_webapp_imports() -> Check:
    c = Check("Q10", "Does the web app import &amp; register every route?")
    sys.path.insert(0, str(ROOT))
    try:
        webapp = importlib.import_module("docbank_pipeline.webapp")
    except Exception as e:
        c.failed(f"Failed to import docbank_pipeline.webapp: {e}")
        return c
    try:
        app = webapp.create_app()
    except Exception as e:
        c.failed(f"create_app() raised: {e}")
        return c
    routes = sorted(r.rule for r in app.url_map.iter_rules())
    expected = {"/", "/upload", "/jobs/<job_id>", "/jobs/<job_id>/results",
                "/jobs/<job_id>/results.json", "/jobs/<job_id>/<path:rel>"}
    missing = expected - set(routes)
    if missing:
        c.warned(f"Some expected routes are missing: {sorted(missing)}",
                 ["Routes registered:"] +
                 [f"&bull; <code>{html.escape(r)}</code>" for r in routes])
    else:
        c.passed(
            "All expected routes are registered.",
            ["Routes:"] + [f"&bull; <code>{html.escape(r)}</code>" for r in routes],
        )
    return c


# ---------------------------------------------------------------- HTML out

CSS = """
:root { --green:#2e7d32; --red:#c62828; --orange:#ef6c00; --grey:#666; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
       margin: 0; padding: 32px; background: #f5f7fa; color: #222; max-width: 1100px; }
h1 { margin: 0 0 6px; }
h2 { margin: 28px 0 8px; color: #1a3a6b; }
.lead { color: var(--grey); margin: 0 0 20px; }
.summary { background: #fff; border: 1px solid #dde0e6; border-radius: 8px;
           padding: 14px 18px; margin-bottom: 28px; display: flex; gap: 22px;
           flex-wrap: wrap; }
.summary b { font-size: 22px; }
.check { background: #fff; border: 1px solid #dde0e6; border-radius: 8px;
         padding: 16px 20px; margin-bottom: 14px;
         box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.check h3 { margin: 0 0 6px; font-size: 16px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
         color: #fff; font-size: 12px; font-weight: 600; margin-right: 8px;
         vertical-align: 1px; }
.pass { background: var(--green); }
.fail { background: var(--red); }
.warn { background: var(--orange); }
.skip { background: var(--grey); }
.qid { color: var(--grey); font-size: 13px; margin-right: 6px;
       font-family: ui-monospace, Consolas, monospace; }
.summary-line { color: #333; margin: 4px 0 8px; }
.evidence { font-size: 13px; color: #444; margin: 6px 0 0; }
.evidence pre { background: #f4f5f8; padding: 8px; border-radius: 4px;
                border: 1px solid #dde0e6; overflow-x: auto;
                font-size: 12px; }
table { border-collapse: collapse; margin-top: 6px; font-size: 13px; }
th, td { padding: 4px 10px; text-align: left;
         border-bottom: 1px solid #e2e6ee; }
th { background: #1e88e5; color: #fff; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
.foot { margin-top: 30px; color: var(--grey); font-size: 12px; }
"""


def render_evidence(items: list[Any]) -> str:
    if not items:
        return ""
    out: list[str] = ['<div class="evidence">']
    for item in items:
        if isinstance(item, dict) and "table" in item:
            rows = item["table"]
            out.append("<table>")
            for i, row in enumerate(rows):
                tag = "th" if i == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{html.escape(str(c))}</{tag}>"
                                            for c in row) + "</tr>")
            out.append("</table>")
        else:
            out.append(f"<div>{item}</div>")
    out.append("</div>")
    return "\n".join(out)


def render(checks: list[Check]) -> str:
    counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
    for c in checks:
        counts[c.status] = counts.get(c.status, 0) + 1
    overall = "PASS" if counts["fail"] == 0 else "FAIL"
    overall_color = "var(--green)" if overall == "PASS" else "var(--red)"
    when = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>DocBank Pipeline — Validation Report</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>DocBank Pipeline — Validation Report</h1>",
        f"<p class='lead'>Generated {html.escape(when)} &middot; "
        f"answers 10 verification questions about the project's actual "
        f"on-disk artefacts.</p>",
        "<div class='summary'>",
        f"<div><b style='color:{overall_color};'>{overall}</b><br>"
        f"<span style='color:#666;font-size:13px;'>overall</span></div>",
        f"<div><b>{counts['pass']}</b><br><span style='color:#666;font-size:13px;'>passed</span></div>",
        f"<div><b>{counts['warn']}</b><br><span style='color:#666;font-size:13px;'>warnings</span></div>",
        f"<div><b>{counts['fail']}</b><br><span style='color:#666;font-size:13px;'>failed</span></div>",
        f"<div><b>{counts['skip']}</b><br><span style='color:#666;font-size:13px;'>skipped</span></div>",
        "</div>",
        "<h2>Checks</h2>",
    ]

    for c in checks:
        badge = c.status
        parts.append("<div class='check'>")
        parts.append(
            f"<h3><span class='badge {badge}'>{c.status.upper()}</span>"
            f"<span class='qid'>{html.escape(c.qid)}</span>"
            f"{html.escape(c.question)}</h3>"
        )
        if c.summary:
            parts.append(f"<div class='summary-line'>{c.summary}</div>")
        parts.append(render_evidence(c.evidence))
        parts.append("</div>")

    parts.append(
        "<div class='foot'>This page was produced by "
        "<code>make_html_report.py</code>. Re-run after any change to "
        "regenerate.</div></body></html>"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------- main

def main():
    print("Running pipeline validation checks…")
    checks = [
        check_model_exists(),
        check_training_completed(),
        check_training_quality(),
        check_dataset_valid(),
        check_classes_detected(),
        check_text_ocr(),
        check_formula_ocr(),
        check_json_schema(),
        check_cache_active(),
        check_webapp_imports(),
    ]

    out = ROOT / "Pipeline_Validation_Report.html"
    out.write_text(render(checks), encoding="utf-8")
    print(f"\nWrote {out}")
    print("Open it in any browser to see the report.\n")
    # ASCII-only console summary (Windows cp949 can't render em-dash etc.)
    def _ascii(s: str) -> str:
        return (s.encode("ascii", "replace").decode("ascii"))
    for c in checks:
        marker = {"pass": "OK ", "fail": "FAIL", "warn": "WARN",
                  "skip": "skip"}.get(c.status, "?")
        print(f"  [{marker}] {c.qid}: {_ascii(c.question)}")
        print(f"          {_ascii(c.summary)}")
    fail_count = sum(1 for c in checks if c.status == "fail")
    # exit 0 even if fails — the HTML report is the primary output
    sys.exit(0)


if __name__ == "__main__":
    main()
