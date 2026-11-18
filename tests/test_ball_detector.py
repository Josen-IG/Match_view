import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from ball_detector import SoccerBallDetector
from test_video_pipeline import _CpuTensor, _FakeCapture, _FakeYOLO


class BallDetectorTests(unittest.TestCase):
    def test_boxes_coordinates_flips_and_rejected_invalid_values(self):
        fake = types.SimpleNamespace(names={0: "ball"})
        fake.predict = lambda *a, **kw: [types.SimpleNamespace(boxes=types.SimpleNamespace(
            cls=_CpuTensor([0, 0, 0]),
            xyxy=_CpuTensor([[20, 10, 40, 30], [0, 0, 0, 4], [float('nan'), 1, 2, 3]]),
            conf=_CpuTensor([.3, .9, .9]))) ]
        with patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=lambda _: fake)}), \
             patch("ball_detector.Path.is_file", return_value=True):
            detector = SoccerBallDetector("local.pt")
        candidates = detector.predict(np.zeros((100, 200, 3)), True, True)
        self.assertEqual(1, len(candidates))
        self.assertEqual((85, 80), (candidates[0]['x'], candidates[0]['y']))
        self.assertEqual((30, 30), (candidates[0]['px'], candidates[0]['py']))

    def test_public_download_ignores_expired_credentials(self):
        from ball_detector import BALL_REPO, BALL_REVISION
        with patch.dict(sys.modules, {
                "ultralytics": types.SimpleNamespace(YOLO=lambda _: types.SimpleNamespace(names={0: "ball"})),
                "huggingface_hub": types.SimpleNamespace(hf_hub_download=__import__("unittest.mock").mock.Mock(return_value="cached.pt")),
        }), patch("ball_detector.Path.is_file", return_value=False):
            import huggingface_hub
            SoccerBallDetector("hf")
            huggingface_hub.hf_hub_download.assert_called_once_with(
                BALL_REPO, "v9b_best.pt", revision=BALL_REVISION, token=False)

    def test_local_weights_next_to_app_skip_download(self):
        with patch.dict(sys.modules, {
                "ultralytics": types.SimpleNamespace(YOLO=lambda _: types.SimpleNamespace(names={0: "ball"})),
            "huggingface_hub": types.SimpleNamespace(hf_hub_download=__import__("unittest.mock").mock.Mock()),
        }), patch("ball_detector.Path.is_file", return_value=True), \
             patch("huggingface_hub.hf_hub_download") as download:
            SoccerBallDetector("hf")
            download.assert_not_called()

    def test_wrong_classes_fail_explicitly(self):
        with patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(
                YOLO=lambda _: types.SimpleNamespace(names={0: "player"}))}), \
             patch("ball_detector.Path.is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "ball class"):
                SoccerBallDetector("local.pt")

    def test_disabled_does_not_load_specialist(self):
        self.run_pipeline(base_ball=True, enabled=False)

    def test_base_ball_is_kept_without_specialist_inference(self):
        self.run_pipeline(base_ball=True, enabled=True)

    def test_missing_ball_recovered_without_changing_people(self):
        self.run_pipeline(base_ball=False, enabled=True)

    def run_pipeline(self, base_ball, enabled):
        import video_track
        class Detector(_FakeYOLO):
            def predict(self, *a, **kw):
                results = super().predict(*a, **kw)
                if not base_ball:
                    boxes = results[0].boxes
                    boxes.cls.values = boxes.cls.values[:1]
                    boxes.xyxy.values = boxes.xyxy.values[:1]
                    boxes.conf.values = boxes.conf.values[:1]
                return results
        class Tracker:
            def __init__(self, **kwargs): pass
            def update(self, detections, *args, **kwargs): return [0] * len(detections)
            def predict_missing(self, *args): return []
        candidate = dict(x=42., y=52., px=67.2, py=66., w=3., h=5.,
                         conf=.25, pitch_x=None, pitch_y=None,
                         detector="yolov9_soccer_ball")
        with patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=Detector)}), \
             patch("cv2.VideoCapture", return_value=_FakeCapture()), \
             patch.object(video_track, "video_duration", return_value=1.), \
             patch.object(video_track, "PlayerIDTracker", Tracker), \
             patch("ball_detector.SoccerBallDetector") as specialist:
            specialist.return_value.predict.return_value = [candidate]
            samples, _, _ = video_track._track_video(
                "fake.mp4", "base.pt", 10, 320, "#FF0000", "#0000FF",
                pitch_model="", pitch_only=False,
                ball_model="hf" if enabled else None)
        self.assertEqual(10, len(samples))
        self.assertTrue(all(len(s['people']) == 1 and s['ball'] for s in samples))
        if not enabled:
            specialist.assert_not_called()
        elif base_ball:
            specialist.return_value.predict.assert_not_called()
            self.assertTrue(all(s['ball_detector'] == 'base' for s in samples))
        else:
            self.assertEqual(10, specialist.return_value.predict.call_count)
            self.assertTrue(all(s['ball_detector'] == 'yolov9_soccer_ball' for s in samples))


if __name__ == '__main__':
    unittest.main()
