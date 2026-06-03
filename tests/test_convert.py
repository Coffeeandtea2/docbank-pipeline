import json
import tempfile
import unittest
from pathlib import Path

from docbank_pipeline.config import PipelineConfig
from docbank_pipeline.convert import (
    COCO_FILES,
    _yolo_line,
    convert_docbank_to_yolo,
    create_yolo_yaml,
    dataset_statistics,
)


class ConvertTests(unittest.TestCase):
    def _cfg(self, tmp: str) -> PipelineConfig:
        cfg = PipelineConfig(data_root=Path(tmp))
        cfg.ensure_dirs()
        return cfg

    def test_yolo_line_normalises_and_clamps_values(self):
        self.assertEqual(
            _yolo_line([10, 20, 30, 40], img_w=100, img_h=200, class_id=2),
            "2 0.250000 0.200000 0.300000 0.200000",
        )
        self.assertEqual(
            _yolo_line([-50, -50, 300, 300], img_w=100, img_h=100, class_id=1),
            "1 1.000000 1.000000 1.000000 1.000000",
        )

    def test_create_yolo_yaml_and_dataset_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp)
            yaml_path = create_yolo_yaml(cfg)

            content = yaml_path.read_text(encoding="utf-8")
            self.assertIn(f"path: {cfg.yolo_dataset_dir.resolve().as_posix()}", content)
            self.assertIn("train: images/train", content)
            self.assertIn("nc: 3", content)
            self.assertIn("  0: text", content)
            self.assertIn("  1: formula", content)
            self.assertIn("  2: image", content)

            img_dir = cfg.yolo_dataset_dir / "images" / "train"
            lbl_dir = cfg.yolo_dataset_dir / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            (img_dir / "page.jpg").write_bytes(b"jpg")
            (lbl_dir / "page.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )

            stats = dataset_statistics(cfg)
            self.assertEqual(stats["train"]["images"], 1)
            self.assertEqual(stats["train"]["labels"], 1)
            self.assertEqual(stats["train"]["instances"], {
                "text": 1,
                "formula": 1,
                "image": 0,
            })
            self.assertEqual(stats["val"]["images"], 0)

    def test_convert_docbank_to_yolo_writes_labels_and_reports_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp)
            src = cfg.extracted_dir / "nested" / "page1.jpg"
            src.parent.mkdir(parents=True)
            src.write_bytes(b"fake image")

            coco = {
                "images": [
                    {"id": 1, "file_name": "page1.jpg", "width": 100, "height": 200},
                    {"id": 2, "file_name": "missing.jpg", "width": 100, "height": 100},
                ],
                "categories": [
                    {"id": 10, "name": "title"},
                    {"id": 11, "name": "isolate_formula"},
                    {"id": 12, "name": "abandon"},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 10, "bbox": [10, 20, 30, 40]},
                    {"image_id": 1, "category_id": 11, "bbox": [0, 0, 20, 20]},
                    {"image_id": 1, "category_id": 12, "bbox": [0, 0, 90, 90]},
                    {"image_id": 2, "category_id": 10, "bbox": [0, 0, 10, 10]},
                ],
            }
            ann_path = cfg.annotations_dir / COCO_FILES["train"]
            ann_path.write_text(json.dumps(coco), encoding="utf-8")

            report = convert_docbank_to_yolo(
                cfg,
                "train",
                image_index={"page1.jpg": src},
            )

            self.assertEqual(report, {
                "split": "train",
                "processed": 1,
                "skipped_existing": 0,
                "skipped_missing": 1,
            })
            self.assertTrue((cfg.yolo_dataset_dir / "images" / "train" / "page1.jpg").is_file())
            label = cfg.yolo_dataset_dir / "labels" / "train" / "page1.txt"
            self.assertEqual(
                label.read_text(encoding="utf-8").splitlines(),
                [
                    "0 0.250000 0.200000 0.300000 0.200000",
                    "1 0.100000 0.050000 0.200000 0.100000",
                ],
            )

            rerun = convert_docbank_to_yolo(
                cfg,
                "train",
                image_index={"page1.jpg": src},
            )
            self.assertEqual(rerun["processed"], 0)
            self.assertEqual(rerun["skipped_existing"], 1)
            self.assertEqual(rerun["skipped_missing"], 1)

    def test_convert_docbank_to_yolo_validates_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp)

            with self.assertRaisesRegex(ValueError, "Unknown split"):
                convert_docbank_to_yolo(cfg, "dev", image_index={})

            with self.assertRaises(FileNotFoundError):
                convert_docbank_to_yolo(cfg, "train", image_index={})

    def test_convert_docbank_to_yolo_rejects_unknown_target_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp)
            cfg.class_mapping = {"title": "heading"}
            coco = {
                "images": [
                    {"id": 1, "file_name": "page1.jpg", "width": 100, "height": 100},
                ],
                "categories": [{"id": 10, "name": "title"}],
                "annotations": [
                    {"image_id": 1, "category_id": 10, "bbox": [0, 0, 10, 10]},
                ],
            }
            (cfg.annotations_dir / COCO_FILES["train"]).write_text(
                json.dumps(coco),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown YOLO class"):
                convert_docbank_to_yolo(cfg, "train", image_index={})


if __name__ == "__main__":
    unittest.main()
