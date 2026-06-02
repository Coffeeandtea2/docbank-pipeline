"""
Stage 7 -- Telegram Bot.

Telegram interface for the same upload -> detection -> OCR -> searchable-PDF
workflow exposed by `webapp.py`.

Run with:
    export TELEGRAM_BOT_TOKEN=123456:abcdef...
    python -m docbank_pipeline.tgbot

Optional environment / CLI configuration:
    DATA_ROOT=/path/to/DocBank              # same PipelineConfig root
    LAYOUT_WEIGHTS=/path/to/checkpoint.pt   # DocLayout-YOLO weights
    TELEGRAM_BOT_TOKEN=123456:abcdef...     # BotFather token
    python -m docbank_pipeline.tgbot --data-root /path/to/DocBank

Per-chat job state lives under:
    <cfg.output_dir>/telegram/<chat_id>/<job_id>/

with the same per-job artefact layout as the Flask app:
    inputs/        # uploaded images and original PDFs
    boxes/         # source pages with detection rectangles drawn on top
    crops/         # cropped detection regions
    results.json   # structured output sent back to the user
    result.pdf     # searchable reconstructed PDF sent back to the user

Telegram differs from a browser form in one important way: files arrive as
individual messages, not one multi-file POST. The bot therefore keeps a small
in-memory session per chat:

    1. The user sends one or more page images / PDFs.
    2. The bot stores them in the current job folder and expands PDFs to JPGs.
    3. The user optionally changes settings with /set.
    4. The user runs /run.
    5. The bot executes the exact same pipeline path as the webapp and sends
       back `reconstructed_document.pdf` plus `results.json`.

The heavy ML models are still loaded lazily by the existing pipeline modules
and reused across requests. The Telegram dependency is also lazy: importing
this module does not require `python-telegram-bot`; creating/running the bot
does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .webapp import (
    ALLOWED_EXTS,
    IMAGE_EXTS,
    PDF_EXTS,
    _explode_pdf,
    _process_uploads,
)

log = logging.getLogger("docbank.tgbot")


# --------------------------------------------------------------- settings

@dataclass
class BotSettings:
    """User-tunable OCR and detection knobs.

    These defaults mirror the Flask form in `webapp.py`. Keeping them here as
    a dataclass makes it easy to show, reset, validate, and pass the settings
    into the shared `_process_uploads` implementation.
    """

    conf: float = 0.25
    min_ocr_conf: float = 0.30
    min_ocr_area: int = 600
    min_formula_area: int | None = 2000
    max_formulas: int | None = None
    text_ocr: bool = True
    formula_ocr: bool = True
    use_cache: bool = True


@dataclass
class UploadSession:
    """One queued Telegram upload batch for one chat."""

    chat_id: int
    job_id: str
    job_dir: Path
    inputs_dir: Path
    inputs: list[Path] = field(default_factory=list)
    source_notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    processing: bool = False
    finished: bool = False


SETTING_HELP = {
    "conf": "YOLO detection confidence, 0.05..0.95.",
    "min_ocr_conf": "Skip OCR for detections below this confidence, 0..1.",
    "min_ocr_area": "Skip OCR for crops smaller than this many pixels squared.",
    "min_formula_area": "Formula-only area threshold; use off/none to disable.",
    "max_formulas": "Hard cap on formula OCR calls; use off/none to disable.",
    "text_ocr": "on/off for PaddleOCR text recognition.",
    "formula_ocr": "on/off for formula recognition.",
    "use_cache": "on/off for the persistent crop OCR cache.",
}

BOOL_KEYS = {"text_ocr", "formula_ocr", "use_cache"}
INT_KEYS = {"min_ocr_area", "min_formula_area", "max_formulas"}
FLOAT_KEYS = {"conf", "min_ocr_conf"}


def _format_settings(settings: BotSettings) -> str:
    """Return the current settings as compact plain text."""
    max_formulas = (
        "none" if settings.max_formulas is None else str(settings.max_formulas)
    )
    min_formula_area = (
        "none"
        if settings.min_formula_area is None
        else str(settings.min_formula_area)
    )
    return (
        "Current settings:\n"
        f"conf = {settings.conf:.2f}\n"
        f"min_ocr_conf = {settings.min_ocr_conf:.2f}\n"
        f"min_ocr_area = {settings.min_ocr_area}\n"
        f"min_formula_area = {min_formula_area}\n"
        f"max_formulas = {max_formulas}\n"
        f"text_ocr = {_onoff(settings.text_ocr)}\n"
        f"formula_ocr = {_onoff(settings.formula_ocr)}\n"
        f"use_cache = {_onoff(settings.use_cache)}"
    )


def _onoff(value: bool) -> str:
    return "on" if value else "off"


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if value in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    raise ValueError("expected on/off")


def _parse_optional_int(raw: str) -> int | None:
    value = raw.strip().lower()
    if value in {"", "none", "null", "off", "disable", "disabled"}:
        return None
    parsed = int(float(value))
    return parsed if parsed > 0 else None


def _set_setting(settings: BotSettings, key: str, raw_value: str) -> str:
    """Validate and assign one `/set key value` pair."""
    key = key.strip().lower().replace("-", "_")
    if key not in SETTING_HELP:
        raise KeyError(key)

    if key in BOOL_KEYS:
        value: Any = _parse_bool(raw_value)
    elif key in INT_KEYS:
        value = _parse_optional_int(raw_value)
        if key == "min_ocr_area":
            value = max(0, int(value or 0))
        elif key == "min_formula_area" and value is not None:
            value = max(0, int(value)) or None
    elif key in FLOAT_KEYS:
        value = float(raw_value)
        if key == "conf":
            value = max(0.05, min(0.95, value))
        else:
            value = max(0.0, min(1.0, value))
    else:  # pragma: no cover - guarded by SETTING_HELP
        raise KeyError(key)

    setattr(settings, key, value)
    return key


# ------------------------------------------------------------ file helpers

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(name: str, fallback_suffix: str = ".jpg") -> str:
    """Small stdlib equivalent of Werkzeug's secure_filename.

    The webapp can lean on Werkzeug because Flask already pulls it in. The bot
    keeps its own sanitizer so Telegram support stays independent from Flask's
    request stack.
    """
    original = Path(name or "").name.strip()
    if not original:
        original = f"upload_{uuid.uuid4().hex[:8]}{fallback_suffix}"
    original = original.replace(" ", "_")
    safe = _SAFE_NAME_RE.sub("_", original).strip("._")
    if not safe:
        safe = f"upload_{uuid.uuid4().hex[:8]}{fallback_suffix}"
    return safe[:180]


def _unique_target(directory: Path, filename: str) -> Path:
    """Return a non-existing path under `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    target = directory / filename
    i = 1
    while target.exists():
        target = directory / f"{stem}_{i:03d}{suffix}"
        i += 1
    return target


def _telegram_root(cfg: PipelineConfig) -> Path:
    root = cfg.output_dir / "telegram"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _get_settings(context: Any, chat_id: int) -> BotSettings:
    settings_by_chat: dict[int, BotSettings] = context.application.bot_data[
        "settings"
    ]
    return settings_by_chat.setdefault(chat_id, BotSettings())


def _get_session(context: Any, chat_id: int) -> UploadSession:
    """Create or return the active upload session for a chat."""
    sessions: dict[int, UploadSession] = context.application.bot_data["sessions"]
    current = sessions.get(chat_id)
    if current is not None and not current.finished:
        return current

    root: Path = context.application.bot_data["telegram_root"]
    job_id = uuid.uuid4().hex[:12]
    job_dir = root / str(chat_id) / job_id
    inputs_dir = job_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    session = UploadSession(
        chat_id=chat_id,
        job_id=job_id,
        job_dir=job_dir,
        inputs_dir=inputs_dir,
    )
    sessions[chat_id] = session
    return session


def _clear_session(context: Any, chat_id: int, *, delete_files: bool) -> None:
    sessions: dict[int, UploadSession] = context.application.bot_data["sessions"]
    session = sessions.pop(chat_id, None)
    if delete_files and session is not None and not session.processing:
        shutil.rmtree(session.job_dir, ignore_errors=True)


def _job_payload(job_id: str, elapsed: float, pages: list[dict]) -> dict:
    return {
        "job_id": job_id,
        "source": "telegram",
        "elapsed_seconds": round(elapsed, 2),
        "pages": pages,
    }


def _write_results_json(session: UploadSession, elapsed: float, pages: list[dict]) -> Path:
    result_path = session.job_dir / "results.json"
    result_path.write_text(
        json.dumps(
            _job_payload(session.job_id, elapsed, pages),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result_path


def _build_pdf(session: UploadSession, pages: list[dict]) -> Path:
    """Build the same searchable PDF produced by the webapp upload route."""
    from .to_pdf import detections_to_searchable_pdf

    all_dets = [d for page in pages for d in page["detections"]]
    pdf_path = session.job_dir / "result.pdf"
    detections_to_searchable_pdf(all_dets, pdf_path)
    return pdf_path


def _run_pipeline_sync(
    cfg: PipelineConfig,
    session: UploadSession,
    settings: BotSettings,
) -> tuple[list[dict], float, Path, Path]:
    """Blocking pipeline body, intended to run inside `asyncio.to_thread`."""
    pages, elapsed = _process_uploads(
        cfg,
        session.job_dir,
        session.inputs,
        conf=settings.conf,
        do_text=settings.text_ocr,
        do_formula=settings.formula_ocr,
        min_ocr_conf=settings.min_ocr_conf,
        min_ocr_area=settings.min_ocr_area,
        min_formula_area=settings.min_formula_area,
        max_formulas=settings.max_formulas,
        use_cache=settings.use_cache,
    )
    json_path = _write_results_json(session, elapsed, pages)
    pdf_path = _build_pdf(session, pages)
    return pages, elapsed, json_path, pdf_path


def _summarize_pages(pages: list[dict], elapsed: float) -> str:
    detections = [d for page in pages for d in page["detections"]]
    recognized = sum(1 for d in detections if d.get("recognized"))
    by_class: dict[str, int] = {}
    for det in detections:
        cls = det.get("class_name", "?")
        by_class[cls] = by_class.get(cls, 0) + 1
    class_bits = ", ".join(f"{k}: {v}" for k, v in sorted(by_class.items()))
    if not class_bits:
        class_bits = "none"
    return (
        "Done.\n"
        f"Pages: {len(pages)}\n"
        f"Detections: {len(detections)} ({class_bits})\n"
        f"Recognized: {recognized}/{len(detections)}\n"
        f"Elapsed: {elapsed:.1f} s"
    )


# --------------------------------------------------------------- commands

async def start(update: Any, context: Any) -> None:
    """Introduce the bot and the upload/run workflow."""
    await update.effective_message.reply_text(
        "DocBank Layout Extractor is ready.\n\n"
        "Send JPG/PNG/TIFF page images or PDFs, then run /run. "
        "I will return a searchable PDF and results.json.\n\n"
        "Commands:\n"
        "/settings - show detection/OCR options\n"
        "/set conf 0.30 - change one option\n"
        "/new - clear the current upload batch\n"
        "/status - show queued files\n"
        "/help - show detailed command help"
    )


async def help_command(update: Any, context: Any) -> None:
    """Show command help without using Telegram parse modes."""
    keys = "\n".join(f"{key}: {desc}" for key, desc in SETTING_HELP.items())
    await update.effective_message.reply_text(
        "Workflow:\n"
        "1. Send one or more image/PDF files.\n"
        "2. Optionally tune settings with /set key value.\n"
        "3. Run /run.\n\n"
        "Examples:\n"
        "/set conf 0.25\n"
        "/set min_ocr_conf 0.30\n"
        "/set min_formula_area 2000\n"
        "/set max_formulas 30\n"
        "/set formula_ocr off\n"
        "/set use_cache on\n\n"
        "Supported settings:\n"
        f"{keys}"
    )


async def settings_command(update: Any, context: Any) -> None:
    chat_id = update.effective_chat.id
    await update.effective_message.reply_text(
        _format_settings(_get_settings(context, chat_id))
    )


async def reset_settings_command(update: Any, context: Any) -> None:
    chat_id = update.effective_chat.id
    context.application.bot_data["settings"][chat_id] = BotSettings()
    await update.effective_message.reply_text(
        "Settings reset.\n" + _format_settings(BotSettings())
    )


async def set_command(update: Any, context: Any) -> None:
    chat_id = update.effective_chat.id
    if len(context.args) < 2:
        await update.effective_message.reply_text(
            "Usage: /set key value\nExample: /set conf 0.30\n"
            "Run /help for the full settings list."
        )
        return

    key = context.args[0]
    value = " ".join(context.args[1:])
    settings = _get_settings(context, chat_id)
    try:
        normalized = _set_setting(settings, key, value)
    except KeyError:
        await update.effective_message.reply_text(
            f"Unknown setting: {key}\nRun /help for supported keys."
        )
        return
    except ValueError as e:
        await update.effective_message.reply_text(f"Invalid value: {e}")
        return

    await update.effective_message.reply_text(
        f"Updated {normalized}.\n" + _format_settings(settings)
    )


async def new_command(update: Any, context: Any) -> None:
    """Clear the queued upload batch for this chat."""
    chat_id = update.effective_chat.id
    session = context.application.bot_data["sessions"].get(chat_id)
    if session is not None and session.processing:
        await update.effective_message.reply_text(
            "A job is already processing. Wait for it to finish before /new."
        )
        return
    _clear_session(context, chat_id, delete_files=True)
    await update.effective_message.reply_text("Cleared. Send new files when ready.")


async def status_command(update: Any, context: Any) -> None:
    chat_id = update.effective_chat.id
    session = context.application.bot_data["sessions"].get(chat_id)
    if session is None or not session.inputs:
        await update.effective_message.reply_text("No files queued. Send an image or PDF.")
        return
    notes = "\n".join(f"- {note}" for note in session.source_notes[-10:])
    if len(session.source_notes) > 10:
        notes += f"\n- ... {len(session.source_notes) - 10} more"
    await update.effective_message.reply_text(
        f"Job {session.job_id}\n"
        f"Queued page image(s): {len(session.inputs)}\n"
        f"Recent uploads:\n{notes}\n\n"
        "Run /run to process or /new to clear."
    )


async def handle_upload(update: Any, context: Any) -> None:
    """Download one Telegram image/PDF and add it to the active session."""
    message = update.effective_message
    chat_id = update.effective_chat.id
    session = _get_session(context, chat_id)

    if session.processing:
        await message.reply_text(
            "The current job is processing. Please wait, or start a new batch after it finishes."
        )
        return

    file_obj = None
    filename = ""
    fallback_suffix = ".jpg"

    if message.document:
        file_obj = message.document
        filename = message.document.file_name or f"document_{message.document.file_unique_id}"
        if message.document.mime_type == "application/pdf" and not filename.lower().endswith(".pdf"):
            filename += ".pdf"
    elif message.photo:
        file_obj = message.photo[-1]
        filename = f"photo_{file_obj.file_unique_id}.jpg"
    else:
        await message.reply_text("Please send a supported image or PDF file.")
        return

    safe_name = _safe_filename(filename, fallback_suffix=fallback_suffix)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        await message.reply_text(
            "Unsupported file type. Send JPG, PNG, BMP, TIFF, or PDF."
        )
        return

    target = _unique_target(session.inputs_dir, safe_name)
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        tg_file = await file_obj.get_file()
        await tg_file.download_to_drive(custom_path=target)
    except Exception as e:
        log.exception("Telegram download failed")
        await message.reply_text(f"Download failed: {e}")
        return

    if ext in PDF_EXTS:
        try:
            pages = _explode_pdf(target, session.inputs_dir)
        except ImportError as e:
            target.unlink(missing_ok=True)
            await message.reply_text(str(e))
            return
        except Exception as e:
            log.exception("PDF rendering failed for %s", target)
            target.unlink(missing_ok=True)
            await message.reply_text(f"Failed to render PDF: {e}")
            return
        session.inputs.extend(pages)
        session.source_notes.append(f"{target.name} -> {len(pages)} page(s)")
        await message.reply_text(
            f"Queued {target.name}: {len(pages)} rendered page(s).\n"
            f"Total queued page image(s): {len(session.inputs)}\n"
            "Send more files or run /run."
        )
        return

    if ext in IMAGE_EXTS:
        session.inputs.append(target)
        session.source_notes.append(target.name)
        await message.reply_text(
            f"Queued {target.name}.\n"
            f"Total queued page image(s): {len(session.inputs)}\n"
            "Send more files or run /run."
        )
        return

    # Guarded by ALLOWED_EXTS, but keep the branch explicit for future edits.
    target.unlink(missing_ok=True)
    await message.reply_text("Unsupported file type.")


async def run_command(update: Any, context: Any) -> None:
    """Run the queued batch through the shared webapp pipeline."""
    chat_id = update.effective_chat.id
    message = update.effective_message
    session: UploadSession | None = context.application.bot_data["sessions"].get(chat_id)
    if session is None or not session.inputs:
        await message.reply_text("No files queued. Send an image or PDF first.")
        return
    if session.processing:
        await message.reply_text("This job is already processing.")
        return

    cfg: PipelineConfig = context.application.bot_data["cfg"]
    settings = _get_settings(context, chat_id)
    session.processing = True

    status = await message.reply_text(
        "Processing started.\n"
        f"Job: {session.job_id}\n"
        f"Pages: {len(session.inputs)}\n"
        "First run may spend extra time loading OCR models."
    )
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        pages, elapsed, json_path, pdf_path = await asyncio.to_thread(
            _run_pipeline_sync, cfg, session, settings
        )
    except Exception as e:
        session.processing = False
        log.exception("Telegram job %s failed", session.job_id)
        await status.edit_text(f"Pipeline error: {e}")
        return

    session.finished = True
    session.processing = False

    await status.edit_text(_summarize_pages(pages, elapsed))

    # Sending is intentionally sequential: Telegram file upload is I/O-bound,
    # and sequential sends produce a more predictable chat transcript.
    try:
        with pdf_path.open("rb") as fh:
            await message.reply_document(
                document=fh,
                filename="reconstructed_document.pdf",
                caption="Searchable reconstructed PDF",
            )
        with json_path.open("rb") as fh:
            await message.reply_document(
                document=fh,
                filename=f"docbank_{session.job_id}.json",
                caption="Structured JSON results",
            )
    except Exception as e:
        log.exception("Could not send Telegram result files for %s", session.job_id)
        await message.reply_text(
            f"Recognition succeeded, but Telegram upload failed: {e}\n"
            f"Local files are in: {session.job_dir}"
        )
        return

    # Keep files on disk for debugging/download parity with the webapp, but
    # remove the in-memory queue so the next upload starts a fresh job.
    context.application.bot_data["sessions"].pop(chat_id, None)


async def unknown_text(update: Any, context: Any) -> None:
    await update.effective_message.reply_text(
        "Send an image/PDF, or run /help for commands."
    )


# ----------------------------------------------------------------- app

def create_application(cfg: PipelineConfig | None = None, *, token: str | None = None):
    """Build a python-telegram-bot Application bound to `cfg`.

    The token is read from `cfg.telegram_bot_token`, which defaults to the
    `TELEGRAM_BOT_TOKEN` environment variable. The optional `token` argument is
    an explicit override for local experiments. This function registers all
    handlers but does not start polling; use
    `run_bot(...)` for the CLI-style entry point.
    """
    try:
        from telegram.ext import (
            ApplicationBuilder,
            CommandHandler,
            MessageHandler,
            filters,
        )
    except ImportError as e:
        raise ImportError(
            "Telegram bot support requires python-telegram-bot>=20. "
            "Install it with: pip install python-telegram-bot"
        ) from e

    if cfg is None:
        cfg = PipelineConfig()
    cfg.ensure_dirs()

    token = token or cfg.telegram_bot_token
    if not token:
        raise RuntimeError(
            "Missing Telegram bot token. Set TELEGRAM_BOT_TOKEN, "
            "set PipelineConfig.telegram_bot_token, or pass --token."
        )

    app = ApplicationBuilder().token(token).build()
    app.bot_data["cfg"] = cfg
    app.bot_data["telegram_root"] = _telegram_root(cfg)
    app.bot_data["sessions"] = {}
    app.bot_data["settings"] = {}

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("reset_settings", reset_settings_command))
    app.add_handler(CommandHandler("set", set_command))
    app.add_handler(CommandHandler("new", new_command))
    app.add_handler(CommandHandler("cancel", new_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("run", run_command))
    app.add_handler(
        MessageHandler((filters.Document.ALL | filters.PHOTO) & ~filters.COMMAND, handle_upload)
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_text))
    return app


def run_bot(
    cfg: PipelineConfig | None = None,
    *,
    token: str | None = None,
    drop_pending_updates: bool = False,
) -> None:
    """Start Telegram long polling and block until interrupted."""
    app = create_application(cfg, token=token)
    resolved_cfg = cfg or app.bot_data["cfg"]
    log.info("Starting Telegram bot (data_root=%s)", resolved_cfg.data_root)
    app.run_polling(drop_pending_updates=drop_pending_updates)


def _build_cfg(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    if args.data_root:
        cfg.data_root = Path(args.data_root).expanduser().resolve()
    cfg.ensure_dirs()
    return cfg


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point: `python -m docbank_pipeline.tgbot`."""
    parser = argparse.ArgumentParser(
        prog="python -m docbank_pipeline.tgbot",
        description="Run the Stage 7 Telegram bot for DocBank extraction.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--token", default=None,
                        help="Telegram bot token; overrides PipelineConfig/env.")
    parser.add_argument("--drop-pending-updates", action="store_true",
                        help="Ignore old queued Telegram updates on startup.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    from .utils import setup_logging

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    cfg = _build_cfg(args)
    run_bot(
        cfg,
        token=args.token,
        drop_pending_updates=args.drop_pending_updates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
