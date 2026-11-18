"""Regresión estructural del flujo de vídeo de Match View.

Verifica que la entrada siga siendo simple y que el análisis quede
separado del reproductor, sin lanzar modelos ni necesitar un vídeo.
"""
import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "match.py"


class MinimalVideoFlowTests(unittest.TestCase):
    def setUp(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.video = text.split("# --- 7. PANTALLA 4: VÍDEO (flujo guiado) ---", 1)[1]
        ast.parse(text)

    def test_only_three_form_fields(self):
        section = self.video.split('if iniciar:', 1)[0]
        self.assertIn('with st.form("mv_video_start_form"', section)
        self.assertEqual(2, section.count("st.text_input("))
        self.assertEqual(1, section.count("st.selectbox("))
        self.assertIn('st.form_submit_button(', section)
        self.assertNotIn('st.expander(', section)
        self.assertNotIn('st.file_uploader(', section)

    def test_processing_has_progress_and_distinct_ready_view(self):
        for expected in ('@st.fragment(run_every=2)',
                         'st.progress(', 'mv_video_ready_path',
                         'components.html(html_pantalla',
                         'st.button("Analizar otro vídeo"'):
            self.assertIn(expected, self.video)

    def test_no_extra_results_ui_in_video_page(self):
        for forbidden in ("st.radio(", "video_show_manual_editor",
                          "Explorar resultados", "Partido y seguimiento",
                          "st.dataframe(", "Previsualizar límites del campo"):
            self.assertNotIn(forbidden, self.video)

    def test_video_is_passed_to_html_automatically(self):
        self.assertIn('servir_video(playback_path)', self.video)
        self.assertIn('prepare_playback_clip(ruta_gen, start_s, end_s)', self.video)
        self.assertIn('window.VIDEO_OFFSET_S', self.video)
        self.assertIn('window.VIDEO_URL', self.video)
        self.assertIn('window.TRACKING_DATA', self.video)


if __name__ == "__main__":
    unittest.main()

