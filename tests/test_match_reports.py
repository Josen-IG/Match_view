"""Pruebas del generador de PDF, Excel y CSV con datos sintéticos."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from match_reports import generate_match_reports


def sample_data():
    return {
        "video_filename": "prueba.mp4",
        "teams": {"local": "Atlético", "visitante": "Rival"},
        "match_duration_s": 90 * 60,
        "stats": {"posesion": {"local_pct": 54, "visitante_pct": 46}},
        "samples": [{"pp": [[20, 30, 0, 11], [80, 32, 1, 12]]}],
        "player_stats": {"players": [
            {"track_id": 11, "team": "local", "visible_s": 1300,
             "distance_m": 2300, "avg_speed_kmh": 6.4,
             "max_speed_kmh": 25, "sprints": 4, "accelerations": 3,
             "field_coverage_pct": 85},
        ]},
        "tactical_analysis": {"periods": [
            {"period_start_s": 0, "period_end_s": 300,
             "teams": {"local": {"usable_frames": 30, "visible_players": 6,
                                  "width_m": 29, "depth_m": 36, "area_m2": 700},
                       "visitante": {}}},
        ]},
        "events": [],
    }


class ReportTests(unittest.TestCase):
    def test_generate_all_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "partido_tracking.json"
            path.write_text(json.dumps(sample_data()), encoding="utf-8")
            outputs = generate_match_reports(path)
            self.assertEqual(4, len(outputs))
            for filename in outputs.values():
                self.assertTrue(Path(filename).is_file())
                self.assertGreater(Path(filename).stat().st_size, 100)
            self.assertEqual(b"%PDF", Path(outputs["report_pdf"]).read_bytes()[:4])
            with ZipFile(outputs["report_xlsx"]) as archive:
                self.assertIn("xl/workbook.xml", archive.namelist())
                self.assertIn("xl/charts/chart1.xml", archive.namelist())
            with open(outputs["players_csv"], newline="", encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
            self.assertEqual(2, len(rows))
            self.assertEqual("11", rows[1][0])

    def test_invalid_tracking_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text('{"teams": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "samples"):
                generate_match_reports(path)

    def test_old_tracking_without_metrics_still_generates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old_tracking.json"
            path.write_text('{"samples": [], "events": []}', encoding="utf-8")
            output = generate_match_reports(path)
            self.assertTrue(Path(output["report_pdf"]).is_file())
            self.assertTrue(Path(output["report_xlsx"]).is_file())


if __name__ == "__main__":
    unittest.main()
