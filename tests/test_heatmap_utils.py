"""Mapas de calor con memoria limitada y compatibilidad antigua."""
import unittest
from heatmap_utils import build_heatmap_histogram


class HeatmapTests(unittest.TestCase):
    def test_real_pitch_team_and_player(self):
        samples = [
            {"t": 2, "pp": [[50.0, 30.0, 0, 7], [70.0, 40.0, 1, 8]]},
            {"t": 3, "pp": [[52.0, 31.0, 0, 7]]},
        ]
        tracking = {"samples": samples}
        heatmap, field, count = build_heatmap_histogram(tracking, "local", 0, 5)
        self.assertTrue(field)
        self.assertEqual((35, 23), heatmap.shape)
        self.assertEqual(2, count)
        self.assertEqual(2, int(heatmap.sum()))
        _, _, selected = build_heatmap_histogram(tracking, "jugador:8", 0, 5)
        self.assertEqual(1, selected)

    def test_legacy_json_and_time_range(self):
        tracking = {"samples": [
            {"t": 5, "p": [[20, 40, 4, 9, 0, 4]]},
            {"t": 6, "p": [[25, 50, 4, 9, 1, 5]]},
        ]}
        heatmap, field, count = build_heatmap_histogram(tracking, "visitante", 6, 7)
        self.assertFalse(field)
        self.assertEqual((50, 28), heatmap.shape)
        self.assertEqual(1, count)

    def test_large_sample_count_without_extra_point_arrays(self):
        sample = {"t": 1, "pp": [[10, 10, 0, 1]]}
        tracking = {"samples": [sample] * 120_000}
        heatmap, _, count = build_heatmap_histogram(tracking, "local", 0, 2)
        self.assertEqual(120_000, count)
        self.assertEqual(120_000, int(heatmap.sum()))


if __name__ == "__main__":
    unittest.main()
