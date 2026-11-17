"""Compatibilidad entre vídeo y tracking y lectura ligera de biblioteca."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from tracking_catalog import read_tracking_header, inspect_tracking, deduplicate_library


class TrackingCatalogTests(unittest.TestCase):
    def test_header_with_large_samples(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "partido_tracking.json"
            data = {"version": 1, "pipeline_version": 2,
                    "video_filename": "partido.mp4",
                    "player_stats": {}, "tactical_analysis": {},
                    "quality": {}, "samples_count": 1,
                    "samples": [{"p": [1] * 500_000}]}
            path.write_text(json.dumps(data), encoding="utf-8")
            header = read_tracking_header(path, max_header_bytes=100_000)
            self.assertNotIn("samples", header)
            self.assertEqual(1, header["samples_count"])
            self.assertEqual("actual", inspect_tracking(path, "partido.mp4")["state"])
            self.assertEqual("otro_video", inspect_tracking(path, "otro.mp4")["state"])

    def test_legacy_tracking_rejected_by_default(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "viejo.json"
            path.write_text(json.dumps({"version": 1, "samples": []}),
                            encoding="utf-8")
            self.assertEqual("antiguo", inspect_tracking(path)["state"])

    def test_duplicate_copy_preferred_newer(self):
        first = {"archivo": "tracking_a.json", "partido": "A vs B",
                 "duracion": "45:00", "muestras": 3000,
                 "video_filename": "a.mp4", "mtime": 1}
        second = dict(first, archivo="tracking_a_copy.json", mtime=3)
        result = deduplicate_library([first, second])
        self.assertEqual(1, len(result))
        self.assertEqual(3, result[0]["mtime"])


if __name__ == "__main__":
    unittest.main()
