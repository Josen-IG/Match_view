"""Radar homography quality gate: only grounded locations may reach the UI."""
import json
import tempfile
import unittest
from pathlib import Path

from radar_quality import radar_calibration_valid


def calibration(**overrides):
    sample = {
        "pitch_ok": True,
        "pitch_calidad": "alta",
        "pitch_age_s": 0.3,
        "pitch_reprojection_error_m": 0.6,
        "pitch_inlier_ratio": 0.9,
    }
    sample.update(overrides)
    return sample


class RadarQualityTests(unittest.TestCase):
    def test_fresh_reliable_homography(self):
        self.assertTrue(radar_calibration_valid(calibration()))

    def test_stale_homography_rejected(self):
        self.assertFalse(radar_calibration_valid(calibration(pitch_age_s=60)))
        self.assertFalse(radar_calibration_valid(calibration(pitch_age_s=3.1)))

    def test_bad_geometry_rejected(self):
        self.assertFalse(radar_calibration_valid(calibration(pitch_calidad="baja")))
        self.assertFalse(radar_calibration_valid(calibration(pitch_reprojection_error_m=3)))
        self.assertFalse(radar_calibration_valid(calibration(pitch_inlier_ratio=.4)))
        self.assertFalse(radar_calibration_valid(calibration(pitch_ok=False)))

    def test_missing_quality_never_assumed_valid(self):
        self.assertFalse(radar_calibration_valid({"pitch_ok": True}))
        self.assertFalse(radar_calibration_valid(calibration(pitch_age_s=None)))

    def test_export_omits_stale_field_positions_but_keeps_detections(self):
        from video_track import export_tracking_json
        sample = calibration(
            pitch_age_s=8.0, t_video=12.0,
            people=[{
                "x": 42, "y": 50, "w": 3, "h": 8,
                "team": "local", "track_id": 10,
                "pitch_x": 41, "pitch_y": 32,
            }],
            ball={"x": 55, "y": 45, "pitch_x": 58, "pitch_y": 30},
            possession_suave="local",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracking.json"
            export_tracking_json([sample], {}, None, path)
            exported = json.loads(path.read_text(encoding="utf8"))
        output = exported["samples"][0]
        self.assertFalse(output["rk"])
        self.assertEqual(8.0, output["ra"])
        self.assertEqual(1, len(output["p"]))
        self.assertIsNotNone(output["b"])
        self.assertNotIn("pp", output)
        self.assertNotIn("bp", output)


if __name__ == "__main__":
    unittest.main()
