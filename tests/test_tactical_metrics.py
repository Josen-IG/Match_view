"""Pruebas de geometría visible, líneas y presión aproximada."""
import unittest
from tactical_metrics import compute_tactical_metrics


def frame(t=0, local=True, visitors=False, gk=False,
          pitch_ok=True, possession="local"):
    people = []
    if local:
        people += [{"team": "local", "pitch_x": x, "pitch_y": y} for x, y in
                   [(10, 20), (20, 20), (20, 40), (10, 40)]]
    if visitors:
        people += [{"team": "visitante", "pitch_x": x, "pitch_y": y} for x, y in
                   [(22, 20), (32, 20), (32, 40), (22, 40)]]
    if gk:
        people += [{"team": "gk_local", "pitch_x": 3, "pitch_y": 34}]
    return {"t": t, "people": people, "ball": {"pitch_x": 25, "pitch_y": 33},
            "pitch_ok": pitch_ok, "pitch_calidad": "alta",
            "possession_suave": possession}


class TacticalTests(unittest.TestCase):
    def test_field_axis_width_depth_and_area(self):
        row = compute_tactical_metrics([frame(gk=True)], sample_every=1)["periods"][0]
        local = row["teams"]["local"]
        self.assertEqual(20, local["width_m"])
        self.assertEqual(10, local["depth_m"])
        self.assertEqual(200, local["area_m2"])
        self.assertEqual(10, local["defensive_line_m"])

    def test_missing_goalkeeper_no_defensive_line(self):
        local = compute_tactical_metrics([frame()], sample_every=1)["periods"][0]["teams"]["local"]
        self.assertNotIn("defensive_line_m", local)

    def test_fewer_than_four_visible_no_shape(self):
        f = frame()
        f["people"] = f["people"][:3]
        local = compute_tactical_metrics([f], sample_every=1)["periods"][0]["teams"]["local"]
        self.assertEqual(0, local["usable_frames"])
        self.assertNotIn("width_m", local)

    def test_invalid_homography_skipped(self):
        row = compute_tactical_metrics([frame(pitch_ok=False)], sample_every=1)["periods"][0]
        self.assertEqual(0, row["coverage"].get("some_team", 0))

    def test_pressure_requires_opponent_possession(self):
        row = compute_tactical_metrics([
            frame(gk=True, visitors=True, possession="visitante")
        ], sample_every=1)["periods"][0]
        self.assertEqual(1, row["teams"]["local"]["defenders_near_ball_10m"])
        self.assertIsNone(row["teams"]["visitante"]["defenders_near_ball_10m"])
        self.assertIsNotNone(row["local_players_minus_visitors_near_ball_median"])

    def test_possession_change_counts_as_proxy(self):
        frames = [frame(t=0, possession="local"),
                  frame(t=1, possession="visitante"),
                  frame(t=2, possession="local")]
        row = compute_tactical_metrics(frames, sample_every=1)["periods"][0]
        self.assertEqual(1, row["possession_recoveries_proxy"]["visitante"])
        self.assertEqual(1, row["possession_recoveries_proxy"]["local"])


if __name__ == "__main__":
    unittest.main()
