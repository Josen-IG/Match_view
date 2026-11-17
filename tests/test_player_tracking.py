"""Pruebas del tracking: identidad, cámara, oclusiones y compatibilidad JSON."""
import json
import tempfile
import unittest
from pathlib import Path

from player_tracking import (
    PlayerIDTracker, estimate_camera_transform, flip_camera_transform,
)


def det(x, y=40, h=10, team="local", hue=110):
    return {"cx": float(x), "cy": float(y), "h": float(h),
            "team_hint": team, "hsv": (hue, 180, 160)}


class TrackingTests(unittest.TestCase):
    def test_stable_ids_with_motion(self):
        tracker = PlayerIDTracker()
        first = tracker.update([det(10), det(70, team="visitante", hue=10)], 1)
        self.assertEqual(first, tracker.update([
            det(12), det(69, team="visitante", hue=10)], 2))

    def test_pan_beyond_matching_radius(self):
        tracker = PlayerIDTracker()
        first = tracker.update([det(10), det(40)], 1)
        self.assertEqual(first, tracker.update([det(26), det(56)], 2,
            camera_transform=(1, 0, 16, 0, 1, 0)))

    def test_crossing_with_distinct_shirts(self):
        tracker = PlayerIDTracker()
        first = tracker.update([det(42, team="local", hue=108),
                                det(58, team="visitante", hue=12)], 1)
        tracker.update([det(47, team="local", hue=108),
                        det(53, team="visitante", hue=12)], 2)
        last = tracker.update([det(52, team="local", hue=108),
                               det(48, team="visitante", hue=12)], 3)
        self.assertEqual(first, last)

    def test_occlusion_and_short_reid(self):
        tracker = PlayerIDTracker(sample_fps=10, max_gap_seconds=1, reid_seconds=6)
        original = tracker.update([det(10)], 1)[0]
        tracker.update([det(10.4)], 2)
        tracker.update([det(10.8)], 3)
        tracker.update([], 4)
        self.assertEqual(original, tracker.update([det(11)], 5)[0])
        for i in range(6, 24):
            tracker.update([], i)
        self.assertEqual(original, tracker.update([det(12)], 24)[0])
        self.assertEqual(1, tracker.stats["reidentified"])

    def test_team_prevents_switch(self):
        tracker = PlayerIDTracker()
        initial = tracker.update([det(33, team="local", hue=105),
                                  det(38, team="visitante", hue=10)], 1)
        for i in (2, 3):
            tracker.update([det(33, team="local", hue=105),
                            det(38, team="visitante", hue=10)], i)
        self.assertEqual(initial, tracker.update([
            det(38, team="local", hue=105),
            det(33, team="visitante", hue=10)], 4))

    def test_stale_track_gets_new_id(self):
        tracker = PlayerIDTracker(sample_fps=10, reid_seconds=3)
        original = tracker.update([det(10)], 1)[0]
        self.assertNotEqual(original, tracker.update([det(10)], 40)[0])

    def test_invalid_order(self):
        tracker = PlayerIDTracker()
        tracker.update([], 1)
        with self.assertRaises(ValueError):
            tracker.update([], 1)

    def test_flip_camera_transform(self):
        original = (1, 0, 16, 0, 1, -5)
        inverted = flip_camera_transform(original, flip_x=True, flip_y=True)
        self.assertAlmostEqual(inverted[2], -16)
        self.assertAlmostEqual(inverted[5], 5)
        self.assertEqual(original, flip_camera_transform(original))

    def test_optical_flow_background_pan(self):
        import cv2
        import numpy as np
        rng = np.random.default_rng(42)
        image = rng.integers(20, 230, (240, 400), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        moved = cv2.warpAffine(
            image, np.float32([[1, 0, 21], [0, 1, -8]]), (400, 240))
        transform = estimate_camera_transform(image, moved)
        self.assertIsNotNone(transform)
        self.assertAlmostEqual(transform[2], 21 * 100 / 400, delta=0.5)
        self.assertAlmostEqual(transform[5], -8 * 100 / 240, delta=0.5)

    def test_export_preserves_legacy_fields_and_adds_pitch_id(self):
        from video_track import export_tracking_json
        sample = {"t_video": 5.0, "people": [
            {"x": 10, "y": 20, "w": 3, "h": 6, "team": "local",
             "track_id": 7, "pitch_x": 40, "pitch_y": 31}],
            "ball": None, "possession_suave": "local"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tracking.json"
            export_tracking_json([sample], {}, None, path)
            exported = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(7, exported["samples"][0]["p"][0][5])
        self.assertEqual([40.0, 31.0, 0, 7], exported["samples"][0]["pp"][0])
        self.assertEqual(1, exported["version"])


if __name__ == "__main__":
    unittest.main()
