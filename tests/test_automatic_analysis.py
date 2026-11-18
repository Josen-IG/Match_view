import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from test_video_pipeline import _FakeCapture, _FakeYOLO
import video_track


class StartRangeTests(unittest.TestCase):
    def test_minutes_and_mmss(self):
        for text, seconds in [('15', 900), ('15:00', 900), ('15:30', 930), ('1,5', 90)]:
            self.assertEqual(seconds, video_track.parse_mmss(text))
        for text in ('-1', 'nan', 'inf', '1:60', '1:-1', '1:2:3', ''):
            with self.assertRaises(ValueError): video_track.parse_mmss(text)

    def test_failed_seek_reopens_and_skips_exact_frames(self):
        first, reopened = _FakeCapture(), _FakeCapture()
        # FakeCapture deliberately ignores CAP_PROP_POS_FRAMES.
        with patch('cv2.VideoCapture', return_value=reopened):
            cap, index = video_track._seek_analysis_start(first, 'video.mp4', .225, 20)
        self.assertIs(cap, reopened)
        self.assertEqual(5, index)
        self.assertEqual(5, reopened.position)
        self.assertEqual(5, reopened.grab_calls)

    def test_frame_seek_reads_actual_video_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frames.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(20): writer.write(np.full((48, 64, 3), i*10, dtype=np.uint8))
            writer.release()
            cap = cv2.VideoCapture(str(path))
            try:
                cap, index = video_track._seek_analysis_start(cap, path, .55, 10)
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertEqual(6, index)
                self.assertAlmostEqual(60, float(frame.mean()), delta=3)
            finally:
                cap.release()

    def test_pipeline_never_detects_frames_before_start_or_at_end(self):
        observed = []
        class Capture(_FakeCapture):
            def get(self, key):
                if key == cv2.CAP_PROP_POS_FRAMES: return self.position
                return super().get(key)
            def set(self, key, value):
                if key == cv2.CAP_PROP_POS_FRAMES: self.position = int(value)
                else: super().set(key, value)
            def read(self):
                value = self.position
                ok, frame = super().read()
                if ok: frame[:] = value
                return ok, frame
        class Detector(_FakeYOLO):
            def predict(self, frame, **kwargs):
                observed.append(int(frame[0, 0, 0]))
                return super().predict(frame, **kwargs)
        class Tracker:
            def __init__(self, **kw): pass
            def update(self, detections, *a, **kw): return [0]*len(detections)
            def predict_missing(self, *a): return []
        with patch('cv2.VideoCapture', return_value=Capture()), \
             patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace(YOLO=Detector)}), \
             patch.object(video_track, 'video_duration', return_value=1), \
             patch.object(video_track, 'PlayerIDTracker', Tracker):
            samples, duration, _ = video_track._track_video(
                'fake.mp4', 'base.pt', 10, 320, '#FF0000', '#0000FF',
                start_seconds=.3, end_seconds=.6, pitch_model='', pitch_only=False)
        self.assertEqual([6, 8, 10], observed)  # actual frames at .3/.4/.5 seconds
        self.assertEqual([.3, .4, .5], [s['t_video'] for s in samples])
        self.assertAlmostEqual(.3, duration)


class AutomaticModelTests(unittest.TestCase):
    def test_availability_is_not_an_accuracy_percentage(self):
        from auto_models import automatic_plan
        with patch('auto_models.available', side_effect=lambda name: name == 'huggingface_hub'), \
             patch('model_jobs.validate_model_environment', side_effect=RuntimeError('Missing environment')):
            plan, status = automatic_plan()
        self.assertEqual('hf', plan['ball_model'])
        self.assertEqual('hf32', plan['pitch_model'])
        self.assertIsNone(plan['rfdetr_model'])
        self.assertFalse(plan['actions_enabled'])
        self.assertEqual('unavailable', status['actions']['state'])
        self.assertEqual('rules', status['policy']['state'])

    def test_available_video_workers_are_selected(self):
        from auto_models import automatic_plan
        with patch('auto_models.available', return_value=True), \
             patch('model_jobs.validate_model_environment'):
            plan, status = automatic_plan()
        self.assertTrue(plan['actions_enabled'])
        self.assertTrue(plan['gamestate_enabled'])
        self.assertEqual('hf', plan['rfdetr_model'])

    def test_failed_optional_checkpoint_keeps_primary_tracking(self):
        class Tracker:
            def __init__(self, **kw): pass
            def update(self, detections, *a, **kw): return [0]*len(detections)
            def predict_missing(self, *a): return []
        status = {}
        with patch('cv2.VideoCapture', return_value=_FakeCapture()), \
             patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace(YOLO=_FakeYOLO)}), \
             patch.object(video_track, 'video_duration', return_value=1), \
             patch.object(video_track, 'PlayerIDTracker', Tracker), \
             patch('ball_detector.SoccerBallDetector', side_effect=RuntimeError('download failed')):
            samples, _, _ = video_track._track_video(
                'fake.mp4', 'base.pt', 10, 320, '#FF0000', '#0000FF',
                ball_model='hf', pitch_model='', pitch_only=False, auto_status=status)
        self.assertTrue(samples)
        self.assertTrue(all(s['ball'] for s in samples))
        self.assertEqual('error', status['yolov9']['state'])
        self.assertEqual('active', status['base']['state'])

    def test_simple_form_has_no_model_checkboxes(self):
        source = (Path(__file__).resolve().parents[1]/'match.py').read_text(encoding='utf-8')
        form = source.split('with st.form("mv_video_start_form"',1)[1].split('if iniciar:',1)[0]
        self.assertNotIn('st.checkbox(', form)
        self.assertIn('Iniciar análisis en', form)
        self.assertIn('10 minutos desde el inicio indicado', form)
        self.assertIn('"auto_models": True', source)


if __name__ == '__main__':
    unittest.main()
