"""Pruebas sintéticas para candidatos NO confirmados del tracking."""
import json
import tempfile
import unittest
from pathlib import Path

from event_candidates import suggest_key_moments


def point(t, x, y=34.0, source="yolo"):
    return {"t": float(t), "bp": [float(x), float(y)], "bf": source}


class CandidateTests(unittest.TestCase):
    def test_no_false_shots_by_default(self):
        samples = [point(i * .1, 68 + 2.2 * i) for i in range(17)]
        self.assertEqual([], suggest_key_moments(samples))

    def test_fast_ball_towards_goal_suggests_shot(self):
        samples = [point(i * .1, 68 + 2.2 * i) for i in range(17)]
        result = suggest_key_moments(samples, experimental=True)
        shots = [c for c in result if c["type"] == "shot"]
        self.assertEqual(1, len(shots))
        self.assertEqual("suggested", shots[0]["status"])
        self.assertLess(shots[0]["confidence"], .85)
        self.assertLess(shots[0]["start_s"], shots[0]["end_s"])

    def test_stationary_corner_and_restarting_ball(self):
        samples = [point(i * .1, .8, .9) for i in range(9)]
        samples += [point(1.0, 5.0, 5.0), point(1.1, 9.0, 8.0)]
        result = suggest_key_moments(samples, experimental=True)
        self.assertTrue(any(c["type"] == "corner" for c in result))

    def test_goal_requires_measured_goal_line_crossing(self):
        samples = [
            point(0, 102.0), point(.1, 103.2),
            point(.2, 104.7), point(.3, 105.7),
        ]
        result = suggest_key_moments(samples, experimental=True)
        self.assertTrue(any(c["type"] == "goal" for c in result))
        self.assertTrue(all(c["label"].startswith("Posible") for c in result))

    def test_approaching_goal_is_not_automatically_a_goal(self):
        samples = [point(i*.1, 70 + 1.9*i) for i in range(18)]
        result = suggest_key_moments(samples, experimental=True)
        self.assertFalse(any(c["type"] == "goal" for c in result))

    def test_no_pitch_data_no_guessed_events(self):
        samples = [{"t": i*.1, "b": [40, 40], "bf": "yolo"} for i in range(25)]
        self.assertEqual([], suggest_key_moments(samples, experimental=True))

    def test_kalman_does_not_produce_events(self):
        samples = [point(i*.1, 66+i*2, source="kalman") for i in range(20)]
        self.assertEqual([], suggest_key_moments(samples, experimental=True))

    def test_unphysical_jump_rejected(self):
        samples = [point(i*.1, 60) for i in range(6)] + [point(.6, 103)]
        self.assertFalse(any(c["type"] == "shot" for c in suggest_key_moments(samples, experimental=True)))

    def test_video_kickoff_bounds(self):
        samples = [point(900+i*.1, 68+2.2*i) for i in range(17)]
        result = suggest_key_moments(samples, kickoff_s=900, experimental=True)
        self.assertTrue(result)
        self.assertTrue(all(c["start_s"] >= 900 for c in result))
        self.assertTrue(all(c["t"] >= 900 for c in result))

    def test_json_export_has_candidates_and_no_verified_event(self):
        from video_track import export_tracking_json

        samples = []
        for i in range(17):
            t = 900 + i*.1
            x = 68+2.2*i
            samples.append({
                "t_video": t, "ball": {"x": 50.0, "y": 50.0, "fuente": "yolo"},
                "ball_pitch_s": {"x": x, "y": 34.0},
                "people": [], "possession_suave": "desconocido",
            })
        info = {"kickoff_offset_s": 900, "match_duration_s": 20}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test_tracking.json"
            export_tracking_json(samples, info, None, dest)
            data = json.loads(dest.read_text(encoding="utf-8"))
        self.assertEqual([], data["events"])
        # Las hipótesis geométricas de disparo ya no se publican en
        # producción porque no verifican contacto ni resultado.
        self.assertEqual([], data["suggested_events"])


if __name__ == "__main__":
    unittest.main()
