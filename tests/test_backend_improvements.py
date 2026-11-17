"""Regression tests for six backend improvement workstreams.

Synthetic examples verify contracts, not real-match model accuracy.
"""
import json
import tempfile
import unittest
from pathlib import Path

from model_evaluation import evaluate_tracking, evaluate_events
from player_tracking import appearance_distance, PlayerIDTracker
from radar_quality import reject_collective_pitch_jumps
from temporal_event_evidence import experimental_shot_sequences
from training_pipeline import audit_dataset


def calibration(t, xoffset=0):
    return {
        "t": t, "t_video": t, "pitch_ok": True,
        "pitch_calidad": "alta", "pitch_age_s": .1,
        "pitch_inlier_ratio": .95, "pitch_reprojection_error_m": .5,
        "people": [
            {"track_id": i, "pitch_x": x+xoffset, "pitch_y": 30}
            for i, x in enumerate((20, 35, 50, 65), 1)
        ], "ball_pitch_s": {"x": 47+xoffset, "y": 34},
        "ball": {"x": 50, "y": 50, "pitch_x": 47+xoffset,
                 "pitch_y": 34},
    }


class TrainingAuditTests(unittest.TestCase):
    def test_training_refuses_missing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\n")
            for split in ("train", "val"):
                labels = root / "labels" / split
                labels.mkdir(parents=True)
                (labels / "frame.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            result = audit_dataset(root)
            self.assertFalse(result["ready"])
            self.assertTrue(any("no images" in x for x in result["issues"]))

    def test_label_integrity_and_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.yaml").write_text("path: .\n")
            for split in ("train", "val"):
                images = root / "images" / split
                labels = root / "labels" / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                (images / "frame1.jpg").write_bytes(b"test")
                (labels / "frame1.txt").write_text("1 .5 .5 .2 .5\n")
            report = audit_dataset(root)
            self.assertFalse(report["ready"])
            self.assertEqual(1, report["duplicate_stems_across_splits"])


class AppearanceTests(unittest.TestCase):
    def test_descriptor_distance(self):
        self.assertAlmostEqual(0, appearance_distance([1, 0], [2, 0]))
        self.assertAlmostEqual(1, appearance_distance([1, 0], [0, 1]))
        self.assertIsNone(appearance_distance(None, [1, 0]))
        self.assertIsNone(appearance_distance([1, -1], [0, 1]))

    def test_reid_rejects_wrong_appearance_after_gap(self):
        tracker = PlayerIDTracker(sample_fps=10, max_gap_seconds=.2,
                                  reid_seconds=5)
        def det(x, desc):
            return {"cx": x, "cy": 40., "h": 10., "team_hint": "local",
                    "hsv": (110, 180, 160), "appearance": desc}
        original = tracker.update([det(30., [1, 0])], 1)[0]
        tracker.update([det(30.2, [1, 0])], 2)
        tracker.update([det(30.4, [1, 0])], 3)
        tracker.update([], 4)
        replacement = tracker.update([det(30.5, [0, 1])], 10)[0]
        self.assertNotEqual(original, replacement)


class PitchConsistencyTests(unittest.TestCase):
    def test_collective_jump_is_rejected_not_players(self):
        first = calibration(0)
        jumped = calibration(.1, 12)
        items = [first, jumped]
        self.assertEqual(1, reject_collective_pitch_jumps(items))
        self.assertFalse(jumped["pitch_ok"])
        self.assertTrue(jumped["pitch_jump_rejected"])
        self.assertEqual(4, len(jumped["people"]))
        self.assertIsNone(jumped["people"][0]["pitch_x"])
        self.assertIsNone(jumped["ball"]["pitch_x"])

    def test_one_fast_player_does_not_invalidate_everyone(self):
        first = calibration(0)
        second = calibration(.1)
        second["people"][0]["pitch_x"] += 15
        self.assertEqual(0, reject_collective_pitch_jumps([first, second]))
        self.assertTrue(second["pitch_ok"])


class TemporalEventTests(unittest.TestCase):
    def test_requires_visible_player_and_direct_ball_trajectory(self):
        samples = []
        for i in range(6):
            x = 70 + 4*i
            samples.append({
                "t": i*.12, "bp": [x, 34], "bf": "yolo",
                "pp": [[70, 34, 0, 7]] if i == 0 else [],
            })
        proposed = experimental_shot_sequences(samples)
        self.assertTrue(proposed)
        self.assertEqual("experimental_unverified", proposed[0]["status"])
        self.assertNotEqual("shot", proposed[0]["type"])
        for sample in samples:
            sample["pp"] = []
        self.assertEqual([], experimental_shot_sequences(samples))
        samples[0]["pp"] = [[70, 34, 0, 7]]
        samples[2]["bf"] = "kalman"
        self.assertEqual([], experimental_shot_sequences(samples))


class LabelledEvaluationTests(unittest.TestCase):
    def test_reports_observed_id_switch_and_ball_accuracy(self):
        gt = {"frames": [
            {"t":0., "players":[{"id":"p7","x":20.,"y":30.}],
             "ball":[50.,50.], "pitch":{"p7":[20.,30.]}},
            {"t":1., "players":[{"id":"p7","x":21.,"y":30.}],
             "ball":[51.,50.]},
        ], "events":[{"type":"shot","t":3.}]}
        pred = {"samples":[
            {"t":0., "p":[[20,30,3,6,0,4]], "b":[50,50],
             "bf":"yolo","pp":[[20,30,0,4]]},
            {"t":1., "p":[[21,30,3,6,0,9]], "b":[50,50],
             "bf":"kalman"},
        ], "events":[]}
        metrics = evaluate_tracking(gt, pred)
        self.assertEqual(1, metrics["id_switches_observed"])
        self.assertEqual(2, metrics["players"]["tp"])
        self.assertEqual(1, metrics["ball"]["tp"])
        self.assertEqual(1, metrics["ball"]["fn"])
        self.assertEqual(1, metrics["pitch_pairs"])
        self.assertIsNone(evaluate_events(gt, pred)["shot"]["precision"])


if __name__ == "__main__":
    unittest.main()
