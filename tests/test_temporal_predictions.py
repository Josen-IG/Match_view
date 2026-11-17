"""Temporal tracking regression: measured vs predicted ball and hidden IDs."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from player_tracking import PlayerIDTracker
from video_track import _suavizar_balon, _filtro_kalman_balon, export_tracking_json


def make_frame(t, x=None):
    return {"t": float(t), "t_video": float(t),
            "ball": {"x": float(x), "y": 50.} if x is not None else None,
            "people": [], "possession_suave": "desconocido"}


class BallTemporalTests(unittest.TestCase):
    def test_short_gap_is_not_mislabeled_as_yolo(self):
        samples = [make_frame(0,30), make_frame(.1,31),
                   make_frame(.2), make_frame(.3), make_frame(.4,34),
                   make_frame(.5), make_frame(.6), make_frame(1.7)]
        _suavizar_balon(samples)
        self.assertTrue(samples[0]["ball_yolo_accepted"])
        self.assertFalse(samples[2]["ball_yolo_accepted"])
        self.assertEqual("interpolado",samples[2]["ball_s"]["fuente"])
        _filtro_kalman_balon(samples)
        self.assertEqual("yolo",samples[1]["ball_s"]["fuente"])
        self.assertEqual("interpolado",samples[2]["ball_s"]["fuente"])
        self.assertEqual("kalman",samples[5]["ball_s"]["fuente"])
        self.assertIsNone(samples[-1]["ball_s"])
        self.assertGreater(samples[5]["ball_s"]["sigma"],0)
        self.assertGreater(samples[5]["ball_s"]["missing_s"],0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracking.json"
            export_tracking_json(samples, {}, None, path)
            data = json.loads(path.read_text(encoding="utf-8"))
        s = data["samples"]
        self.assertEqual("yolo",s[0]["bf"])
        self.assertEqual("interpolado",s[2]["bf"])
        self.assertEqual("kalman",s[5]["bf"])
        self.assertGreater(s[5]["bg"],0)
        self.assertGreater(s[5]["bu"],0)
        self.assertIsNone(s[-1]["b"])

    def test_outlier_does_not_export_as_direct_yolo(self):
        samples = [make_frame(0,10), make_frame(.1,90),
                   make_frame(.2,10.5)]
        _suavizar_balon(samples)
        self.assertFalse(samples[1]["ball_yolo_accepted"])
        _filtro_kalman_balon(samples)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp) / "tracking.json"
            export_tracking_json(samples, {}, None, path)
            result=json.loads(path.read_text(encoding="utf-8"))
        self.assertNotEqual(result["samples"][1].get("bf"),"yolo")


class PlayerTemporalTests(unittest.TestCase):
    @staticmethod
    def det(x,team="local"):
        return {"cx":float(x),"cy":40.,"h":10.,
                "hsv":(110,180,160),"team_hint":team}

    def test_prediction_keeps_id_and_is_separate(self):
        tracker=PlayerIDTracker(sample_fps=10)
        tid=tracker.update([self.det(30)],1)[0]
        tracker.update([self.det(31)],2)
        tracker.update([],3)
        predicted=tracker.predict_missing(3)
        self.assertEqual(1,len(predicted))
        self.assertEqual(tid,predicted[0]["track_id"])
        self.assertGreater(predicted[0]["x"],31)
        self.assertEqual("predicted",predicted[0]["source"])
        self.assertGreater(predicted[0]["uncertainty_u"],0)
        recovered=tracker.update([self.det(32)],4)[0]
        self.assertEqual(tid,recovered)
        self.assertEqual([],tracker.predict_missing(4))

    def test_unknown_location_after_long_absence(self):
        tracker=PlayerIDTracker(sample_fps=10)
        tracker.update([self.det(20)],1)
        for i in range(2,15):
            tracker.update([],i)
        self.assertEqual([],tracker.predict_missing(14))

    def test_predictions_do_not_inflate_player_detections(self):
        sample=make_frame(1,None)
        sample["predicted_people"]=[{"x":20,"y":30,"track_id":4,
              "age_s":0.2,"uncertainty_u":3.0,"source":"predicted"}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"tracking.json"
            export_tracking_json([sample],{},None,path)
            s=json.loads(path.read_text(encoding="utf-8"))["samples"][0]
        self.assertEqual([],s["p"])
        self.assertEqual(4,s["pe"][0][2])
        self.assertNotIn("pp",s)


if __name__ == "__main__":
    unittest.main()
