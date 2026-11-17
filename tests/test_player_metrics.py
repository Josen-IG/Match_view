"""Regresión para las estadísticas individuales sobre muestras sintéticas."""
import unittest
from player_metrics import compute_player_metrics


def sample(i, x=None, team="local", pitch_ok=True, ball=None):
    p = {"track_id": 4, "team": team, "x": 20.0, "y": 20.0,
         "h": 5.0, "pitch_x": x, "pitch_y": 30.0 if x is not None else None}
    return {"t": i * .1, "pitch_calidad": "alta", "pitch_ok": pitch_ok,
            "people": [p], "ball": ball}


class MetricTests(unittest.TestCase):
    def test_linear_motion_distance_and_speed(self):
        data = [sample(i, 20 + i * .1) for i in range(40)]
        p = compute_player_metrics(data)["players"][0]
        self.assertEqual(40, p["visible_samples"])
        self.assertAlmostEqual(p["distance_m"], 3.5, delta=.15)
        self.assertAlmostEqual(p["avg_speed_kmh"], 3.6, delta=.2)
        self.assertEqual(100, p["field_coverage_pct"])

    def test_no_homography_means_no_distance_not_zero(self):
        p = compute_player_metrics([sample(i) for i in range(30)])["players"][0]
        self.assertIsNone(p["distance_m"])
        self.assertIsNone(p["max_speed_kmh"])
        self.assertEqual(0, p["field_coverage_pct"])

    def test_spike_is_not_counted_as_speed(self):
        data = [sample(i, 20 + i * .1) for i in range(40)]
        for point in data[10:15]:
            point["people"][0]["pitch_x"] = 100
        p = compute_player_metrics(data)["players"][0]
        self.assertLess(p["distance_m"], 10)
        self.assertLess(p["max_speed_kmh"], 43.3)

    def test_explicit_invalid_pitch_rejected(self):
        data = [sample(i, 25 + i * .1, pitch_ok=False) for i in range(20)]
        p = compute_player_metrics(data)["players"][0]
        self.assertEqual(0, p["field_samples"])
        self.assertIsNone(p["distance_m"])

    def test_ball_proximity_is_not_touches(self):
        data = [sample(i, 20, ball={"x": 20.2, "y": 20.1}) for i in range(10)]
        p = compute_player_metrics(data)["players"][0]
        self.assertAlmostEqual(.9, p["ball_proximity_s"])

    def test_track_stays_separate_and_team_votes(self):
        data = [sample(i, 20) for i in range(10)]
        data.append({"t": 2.0, "people": [{"track_id": 50,
                     "team": "visitante", "pitch_x": None, "pitch_y": None}]})
        result = compute_player_metrics(data)
        self.assertEqual([4, 50], [p["track_id"] for p in result["players"]])
        self.assertEqual("local", result["players"][0]["team"])


if __name__ == "__main__":
    unittest.main()
