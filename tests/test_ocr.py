import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docbank_pipeline.ocr import (
    _detection_to_record,
    _enrich,
    _load_cache,
    _save_cache,
    save_results_json,
)


class OcrHelperTests(unittest.TestCase):
    def test_detection_to_record_converts_paths_without_mutating_input(self):
        original = {"image": Path("page.jpg"), "crop_path": Path("crop.jpg")}

        record = _detection_to_record(original)

        self.assertEqual(record, {"image": "page.jpg", "crop_path": "crop.jpg"})
        self.assertIsInstance(original["image"], Path)
        self.assertIsInstance(original["crop_path"], Path)

    def test_save_results_json_groups_detections_by_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "results.json"
            page1 = Path(tmp) / "page1.jpg"
            page2 = Path(tmp) / "page2.jpg"
            results = [
                {
                    "image": page1,
                    "image_width": 100,
                    "image_height": 200,
                    "bbox": [1, 2, 3, 4],
                    "recognized": "안녕하세요",
                },
                {
                    "image": page1,
                    "image_width": 100,
                    "image_height": 200,
                    "bbox": [5, 6, 7, 8],
                    "recognized": "text",
                    "crop_path": Path(tmp) / "crop.jpg",
                },
                {
                    "image": page2,
                    "image_width": 50,
                    "image_height": 60,
                    "bbox": [0, 0, 1, 1],
                    "recognized": "",
                },
            ]

            saved = save_results_json(results, out)

            self.assertEqual(saved, out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["pages"]), 2)
            self.assertEqual(payload["pages"][0]["image"], str(page1))
            self.assertEqual(payload["pages"][0]["image_width"], 100)
            self.assertEqual(len(payload["pages"][0]["detections"]), 2)
            self.assertNotIn("image", payload["pages"][0]["detections"][0])
            self.assertEqual(payload["pages"][0]["detections"][1]["crop_path"], str(Path(tmp) / "crop.jpg"))
            self.assertEqual(payload["pages"][1]["image"], str(page2))
            self.assertIn("안녕하세요", out.read_text(encoding="utf-8"))

    def test_cache_load_and_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache" / "ocr.json"

            self.assertEqual(_load_cache(path), {})
            _save_cache(path, {"a": "b"})
            self.assertEqual(_load_cache(path), {"a": "b"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

            path.write_text("{bad", encoding="utf-8")
            self.assertEqual(_load_cache(path), {})

    def test_enrich_filters_caps_and_uses_patched_recognizers(self):
        with tempfile.TemporaryDirectory() as tmp:
            crop1 = Path(tmp) / "crop1.jpg"
            crop2 = Path(tmp) / "crop2.jpg"
            crop3 = Path(tmp) / "crop3.jpg"
            crop4 = Path(tmp) / "crop4.jpg"
            for crop in (crop1, crop2, crop3, crop4):
                crop.write_bytes(b"crop")

            detections = [
                {
                    "class_name": "text",
                    "confidence": 0.9,
                    "bbox": [0, 0, 40, 40],
                    "crop_path": crop1,
                },
                {
                    "class_name": "text",
                    "confidence": 0.9,
                    "bbox": [0, 0, 40, 40],
                    "crop_path": crop2,
                },
                {
                    "class_name": "text",
                    "confidence": 0.1,
                    "bbox": [0, 0, 40, 40],
                    "crop_path": crop3,
                },
                {
                    "class_name": "formula",
                    "confidence": 0.9,
                    "bbox": [0, 0, 50, 50],
                    "crop_path": crop4,
                },
            ]

            with patch("docbank_pipeline.ocr.recognize_text_with_paddleocr", return_value="text-out") as text_rec, \
                    patch("docbank_pipeline.ocr.recognize_formula", return_value="formula-out") as formula_rec:
                enriched = _enrich(
                    detections,
                    do_text=True,
                    do_formula=True,
                    min_ocr_conf=0.3,
                    min_ocr_area=600,
                    use_cache=False,
                    max_text_crops=1,
                    max_formulas=1,
                )

            self.assertIs(enriched[0], detections[0])
            self.assertEqual(enriched[0]["recognized"], "text-out")
            self.assertEqual(enriched[0]["recognition_kind"], "text")
            self.assertEqual(enriched[1]["recognized"], "")
            self.assertIsNone(enriched[1]["recognition_kind"])
            self.assertEqual(enriched[2]["recognized"], "")
            self.assertIsNone(enriched[2]["recognition_kind"])
            self.assertEqual(enriched[3]["recognized"], "formula-out")
            self.assertEqual(enriched[3]["recognition_kind"], "latex")
            text_rec.assert_called_once_with(str(crop1))
            formula_rec.assert_called_once_with(str(crop4))


if __name__ == "__main__":
    unittest.main()
