"""Pruebas de la interfaz de jugadas IA y sintaxis JavaScript embebida."""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "analizador_video.html"


class TimelineUITests(unittest.TestCase):
    def test_timeline_is_one_ai_lane_not_editing_lanes(self):
        code = HTML.read_text(encoding="utf-8")
        self.assertIn('id="laneTracking"', code)
        self.assertIn('id="aiEventsList"', code)
        self.assertIn('id="aiCount"', code)
        self.assertNotIn('id="laneAnnos"', code)
        self.assertNotIn('id="laneFx"', code)
        self.assertNotIn('id="laneCuts"', code)
        self.assertNotIn('id="laneEvents"', code)

    def test_suggestions_have_review_and_clip_actions(self):
        code = HTML.read_text(encoding="utf-8")
        for expected in ("suggested_events", "renderSuggestedMoments",
                         "restoreSuggestedReviews", "persistSuggestionReviews",
                         "Guardar corte", "Confirmar", "Descartar",
                         "Posible córner", "Posible gol", "Posible tiro"):
            self.assertIn(expected, code)

    @unittest.skipUnless(shutil.which("node"), "Node.js no está instalado")
    def test_all_inline_javascript_passes_node_check(self):
        html = HTML.read_text(encoding="utf-8")
        scripts = re.findall(
            r'<script(?:\\s[^>]*)?>(.*?)</script>', html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        self.assertGreaterEqual(len(scripts), 2)
        with tempfile.TemporaryDirectory() as directory:
            for index, script in enumerate(scripts):
                path = Path(directory) / f"inline_{index}.js"
                path.write_text(script, encoding="utf-8")
                process = subprocess.run(
                    ["node", "--check", str(path)],
                    capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(0, process.returncode, process.stderr)


if __name__ == "__main__":
    unittest.main()
