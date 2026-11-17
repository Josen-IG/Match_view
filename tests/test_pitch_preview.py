"""Previsualización de la máscara del campo para Streamlit."""
import unittest

import numpy as np

from pitch_filter import render_pitch_preview


class PreviewTests(unittest.TestCase):
    def test_green_field_mask_preview(self):
        image = np.full((300, 480, 3), (130, 130, 130), dtype=np.uint8)
        image[95:, :] = (42, 83, 38)
        rgb, valid, message = render_pitch_preview(image)
        self.assertTrue(valid)
        self.assertEqual((300, 480, 3), rgb.shape)
        self.assertIn("Región", message)

    def test_no_grass_warns_without_exception(self):
        image = np.full((260, 460, 3), (125, 125, 125), dtype=np.uint8)
        rgb, valid, message = render_pitch_preview(image)
        self.assertFalse(valid)
        self.assertEqual((260, 460, 3), rgb.shape)
        self.assertIn("No se pudo", message)

    def test_bad_frame(self):
        with self.assertRaises(ValueError):
            render_pitch_preview(None)


if __name__ == "__main__":
    unittest.main()
