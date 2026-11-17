"""Checks de cobertura, homografía y consistencia del tracking."""
import unittest

from quality_metrics import assess_tracking_quality


class QualityTests(unittest.TestCase):
    def test_empty_tracking(self):
        result = assess_tracking_quality([])
        self.assertEqual(0, result["total_samples"])
        self.assertTrue(result["warnings"])

    def test_good_samples_and_coverage(self):
        frames = [{"t": i / 10, "pitch_ok": True, "pitch_calidad": "alta",
                   "ball": {"x": 25, "y": 44, "fuente": "yolo"},
                   "people": [
                       {"track_id": 4, "team": "local", "pitch_x": 20, "pitch_y": 30},
                       {"track_id": 5, "team": "visitante", "pitch_x": 65, "pitch_y": 32},
                   ]} for i in range(5)]
        q = assess_tracking_quality(frames)
        self.assertEqual(100, q["pitch_coverage_pct"])
        self.assertEqual(100, q["ball_yolo_pct"])
        self.assertEqual(2, q["unique_track_ids"])
        self.assertEqual(0, q["duplicate_track_id_occurrences"])
        self.assertEqual([], q["warnings"])

    def test_duplicate_ids_and_nonmonotonic(self):
        item = {"t": 1.0, "people": [
            {"track_id": 7, "team": "local"},
            {"track_id": 7, "team": "local"},
        ]}
        result = assess_tracking_quality([item, item])
        self.assertEqual(2, result["duplicate_track_id_occurrences"])
        self.assertEqual(1, result["nonmonotonic_samples"])
        self.assertTrue(any("duplicados" in w for w in result["warnings"]))

    def test_invalid_pitch_coordinates(self):
        row = {"t": 1, "people": [
            {"track_id": 1, "team": "local", "pitch_x": 400, "pitch_y": 40},
            {"track_id": 2, "team": "local", "pitch_x": float("nan"), "pitch_y": 40},
        ]}
        result = assess_tracking_quality([row])
        self.assertEqual(2, result["invalid_pitch_coordinates"])

    def test_kalman_predictions_not_counted_as_yolo(self):
        frames = [{"t": .1, "ball": None, "ball_s": {"fuente": "kalman"},
                   "people": []}, {"t": .2, "ball": None, "people": []}]
        q = assess_tracking_quality(frames)
        self.assertEqual(0.0, q["ball_yolo_pct"])
        self.assertEqual(50.0, q["ball_predicted_pct"])

    def test_interpolated_ball_not_counted_as_yolo(self):
        frames = [
            {"t": 0.0, "ball": {"x": 20, "y": 50},
             "ball_yolo_accepted": True,
             "ball_s": {"fuente": "yolo"}, "people": []},
            {"t": 0.1, "ball": {"x": 95, "y": 50},
             "ball_yolo_accepted": False,
             "ball_s": {"fuente": "interpolado", "sigma": 2.,
                        "missing_s": 0.1},
             "predicted_people": [{"track_id": 1}], "people": []},
            {"t": 0.2, "ball": None,
             "ball_s": {"fuente": "kalman", "sigma": 3.,
                        "missing_s": 0.2}, "people": []},
        ]
        quality = assess_tracking_quality(frames)
        self.assertAlmostEqual(33.3, quality["ball_yolo_pct"])
        self.assertAlmostEqual(33.3, quality["ball_predicted_pct"])
        self.assertAlmostEqual(33.3, quality["ball_interpolated_pct"])
        self.assertEqual(1, quality["hidden_player_hypotheses"])
        self.assertAlmostEqual(2.5, quality["ball_prediction_uncertainty_u_mean"])
        self.assertAlmostEqual(0.15, quality["ball_prediction_missing_s_mean"])

    def test_unknown_team_warning(self):
        q = assess_tracking_quality([{"t": 0.0, "people": [
            {"track_id": 1, "team": "desconocido"},
            {"track_id": 2, "team": "desconocido"},
        ]}])
        self.assertEqual(100.0, q["unknown_team_pct"])
        self.assertTrue(any("equipo" in w for w in q["warnings"]))


if __name__ == "__main__":
    unittest.main()
