import os
import tempfile
import unittest
from pathlib import Path

from docbank_pipeline.config import PipelineConfig
from docbank_pipeline.inference import _iter_sources, _resolve_weights


class InferenceHelperTests(unittest.TestCase):
    def test_iter_sources_accepts_file_folder_and_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_b = root / "b.png"
            img_a = root / "a.jpg"
            txt = root / "notes.txt"
            img_b.write_bytes(b"png")
            img_a.write_bytes(b"jpg")
            txt.write_text("no", encoding="utf-8")

            self.assertEqual(_iter_sources(img_a), [img_a])
            self.assertEqual(_iter_sources(root), [img_a, img_b])
            self.assertEqual(_iter_sources([img_b, "plain.jpg"]), [img_b, Path("plain.jpg")])
            with self.assertRaises(FileNotFoundError):
                _iter_sources(root / "missing")

    def test_resolve_weights_precedence_and_latest_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(data_root=Path(tmp))
            cfg.ensure_dirs()

            explicit = Path(tmp) / "explicit.pt"
            configured = Path(tmp) / "configured.pt"
            older = cfg.runs_dir / "old" / "weights" / "best.pt"
            newer = cfg.runs_dir / "new" / "weights" / "best.pt"
            for p in (explicit, configured, older, newer):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"weights")

            os.utime(older, (1_000_000, 1_000_000))
            os.utime(newer, (2_000_000, 2_000_000))
            cfg.layout_weights = str(configured)

            self.assertEqual(_resolve_weights(cfg, explicit), explicit)
            self.assertEqual(_resolve_weights(cfg, "missing.pt"), configured)

            cfg.layout_weights = None
            self.assertEqual(_resolve_weights(cfg, None), newer)

    def test_resolve_weights_raises_when_none_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(data_root=Path(tmp))
            cfg.ensure_dirs()

            with self.assertRaises(FileNotFoundError):
                _resolve_weights(cfg, None)


if __name__ == "__main__":
    unittest.main()
