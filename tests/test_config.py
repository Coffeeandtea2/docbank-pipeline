import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docbank_pipeline.config import PipelineConfig


class PipelineConfigTests(unittest.TestCase):
    def test_environment_overrides_and_derived_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            env = {
                "DATA_ROOT": str(root),
                "DATASET_PARTS": "3",
                "MAX_PAGES": "17",
                "NUM_WORKERS": "8",
                "TRAIN_VAL_SPLIT": "0.75",
                "LAYOUT_WEIGHTS": "/tmp/layout.pt",
                "HF_TOKEN": "hf_secret",
                "TELEGRAM_BOT_TOKEN": "tg_secret",
            }
            with patch.dict(os.environ, env, clear=True):
                cfg = PipelineConfig()

        self.assertEqual(cfg.data_root, root.resolve())
        self.assertEqual(cfg.raw_data_dir, root.resolve() / "raw")
        self.assertEqual(cfg.annotations_dir, root.resolve() / "annotations")
        self.assertEqual(cfg.extracted_dir, root.resolve() / "images")
        self.assertEqual(cfg.yolo_dataset_dir, root.resolve() / "yolo_dataset")
        self.assertEqual(cfg.output_dir, root.resolve() / "outputs")
        self.assertEqual(cfg.crops_dir, root.resolve() / "outputs" / "crops")
        self.assertEqual(cfg.runs_dir, root.resolve() / "runs")
        self.assertEqual(cfg.dataset_parts, 3)
        self.assertEqual(cfg.max_pages, 17)
        self.assertEqual(cfg.num_workers, 8)
        self.assertEqual(cfg.train_val_split, 0.75)
        self.assertEqual(cfg.layout_weights, "/tmp/layout.pt")
        self.assertEqual(cfg.hf_token, "hf_secret")
        self.assertEqual(cfg.telegram_bot_token, "tg_secret")

    def test_empty_optional_int_uses_default(self):
        with patch.dict(os.environ, {"MAX_PAGES": ""}, clear=True):
            self.assertIsNone(PipelineConfig().max_pages)

    def test_helpers_create_expected_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(data_root=Path(tmp))
            cfg.ensure_dirs()

            for folder in (
                cfg.data_root,
                cfg.raw_data_dir,
                cfg.annotations_dir,
                cfg.extracted_dir,
                cfg.yolo_dataset_dir,
                cfg.output_dir,
                cfg.crops_dir,
                cfg.runs_dir,
            ):
                self.assertTrue(folder.is_dir(), folder)

            self.assertEqual(cfg.part_filename(7), "DocBank_500K_ori_img.zip.007")
            self.assertEqual(cfg.class_id, {"text": 0, "formula": 1, "image": 2})

            described = cfg.describe()
            self.assertIn(f"DATA_ROOT       = {cfg.data_root}", described)
            self.assertIn("HF_TOKEN        = <unset>", described)
            self.assertIn("TELEGRAM_BOT_TOKEN = <unset>", described)


if __name__ == "__main__":
    unittest.main()
