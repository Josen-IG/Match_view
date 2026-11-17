"""Fiabilidad de métricas y comparativas sin datos GPS."""
import unittest
from player_metrics import compute_player_metrics


def generate_track(count, pitch_quality="alta", with_pitch=True, track_id=4):
    result = []
    for i in range(count):
        player = {"track_id": track_id, "team": "local",
                  "x": 50, "y": 50, "h": 5,
                  "pitch_x": 20 + .05 * i if with_pitch else None,
                  "pitch_y": 30 if with_pitch else None}
        result.append({"t": i / 10, "pitch_ok": with_pitch,
                       "pitch_calidad": pitch_quality,
                       "people": [player]})
    return result


class ReliabilityTests(unittest.TestCase):
    def test_adequate_tracking_gets_medium_confidence(self):
        result = compute_player_metrics(generate_track(300))
        player = result["players"][0]
        self.assertEqual("media", player["reliability"])
        self.assertEqual(1, result["team_comparison"]["local"]["reliable_track_ids"])

    def test_long_stable_tracking_gets_high_confidence(self):
        result = compute_player_metrics(generate_track(820))
        self.assertEqual("alta", result["players"][0]["reliability"])

    def test_no_pitch_cannot_enter_team_comparison(self):
        result = compute_player_metrics(generate_track(200, with_pitch=False))
        self.assertEqual("insuficiente", result["players"][0]["reliability"])
        self.assertIsNone(result["team_comparison"]["local"]["measured_distance_m"])

    def test_short_fragment_is_not_treated_as_reliable(self):
        result = compute_player_metrics(generate_track(80))
        self.assertEqual("baja", result["players"][0]["reliability"])
        self.assertEqual(0, result["team_comparison"]["local"]["reliable_track_ids"])


if __name__ == "__main__":
    unittest.main()
