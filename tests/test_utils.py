import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docbank_pipeline import utils


class UtilsTests(unittest.TestCase):
    def test_image_file_helpers_are_case_insensitive_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.PNG").write_bytes(b"b")
            (root / "a.JPG").write_bytes(b"a")
            (root / "notes.txt").write_text("not an image", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.jpeg").write_bytes(b"c")

            self.assertTrue(utils.is_image_file(root / "a.JPG"))
            self.assertTrue(utils.is_image_file(root / "b.PNG"))
            self.assertFalse(utils.is_image_file(root / "notes.txt"))
            self.assertFalse(utils.is_image_file(root / "missing.jpg"))

            self.assertEqual(
                [p.name for p in utils.iter_image_files(root)],
                ["a.JPG", "b.PNG"],
            )
            self.assertEqual(
                [p.name for p in utils.iter_image_files(root, recursive=True)],
                ["a.JPG", "b.PNG", "c.jpeg"],
            )
            self.assertEqual(
                [p.name for p in utils.iter_image_files(root, suffixes={".png"})],
                ["b.PNG"],
            )
            self.assertEqual(utils.iter_image_files(root / "missing"), [])

    def test_marker_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self.assertFalse(utils.stage_done(root, "convert"))
            utils.mark_stage_done(root, "convert", "ok")

            marker = root / ".convert.done"
            self.assertEqual(utils.marker_path(root, "convert"), marker)
            self.assertTrue(utils.stage_done(root, "convert"))
            self.assertEqual(marker.read_text(encoding="utf-8"), "ok")

            utils.clear_stage(root, "convert")
            self.assertFalse(utils.stage_done(root, "convert"))

    def test_link_or_copy_falls_back_to_copy_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src.txt"
            dst = root / "out" / "dst.txt"
            src.write_text("original", encoding="utf-8")

            with patch("docbank_pipeline.utils.os.link", side_effect=OSError):
                utils.link_or_copy(src, dst)
            self.assertEqual(dst.read_text(encoding="utf-8"), "original")

            src.write_text("changed", encoding="utf-8")
            utils.link_or_copy(src, dst)
            self.assertEqual(dst.read_text(encoding="utf-8"), "original")

    def test_cwd_context_restores_previous_directory(self):
        before = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            with utils.cwd(Path(tmp)) as entered:
                self.assertEqual(Path.cwd(), Path(tmp))
                self.assertEqual(entered, Path(tmp))
        self.assertEqual(Path.cwd(), before)

    def test_human_bytes(self):
        self.assertEqual(utils.human_bytes(0), "0.0 B")
        self.assertEqual(utils.human_bytes(1023), "1023.0 B")
        self.assertEqual(utils.human_bytes(1024), "1.0 KB")
        self.assertEqual(utils.human_bytes(1024 * 1024), "1.0 MB")


if __name__ == "__main__":
    unittest.main()
