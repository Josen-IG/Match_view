"""Prueba de integración de la lectura de vídeo con detector YOLO simulado.

No descarga pesos ni requiere vídeo real: verifica los fotogramas que se
decodifican, el orden del tiempo y la continuidad de los IDs.
"""
import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np


class _CpuTensor:
    def __init__(self, values):
        self.values = values
    def cpu(self):
        return self
    def tolist(self):
        return self.values


class _FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
    def predict(self, frame, **kwargs):
        return [types.SimpleNamespace(boxes=types.SimpleNamespace(
            cls=_CpuTensor([1.0, 0.0]),
            xyxy=_CpuTensor([[25, 35, 50, 95], [65, 60, 71, 66]]),
            conf=_CpuTensor([.98, .88]),
        ))]


class _FakeCapture:
    def __init__(self):
        self.position = 0
        self.read_calls = 0
        self.grab_calls = 0
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.frame[35:95, 25:50] = (0, 0, 255)  # camiseta roja BGR
    def isOpened(self):
        return True
    def set(self, key, value):
        if key == cv2.CAP_PROP_POS_MSEC:
            self.position = int(value * 20 / 1000)
    def get(self, key):
        return {cv2.CAP_PROP_FPS: 20, cv2.CAP_PROP_FRAME_WIDTH: 160,
                cv2.CAP_PROP_FRAME_HEIGHT: 120}.get(key, 0)
    def read(self):
        self.read_calls += 1
        if self.position >= 20:
            return False, None
        self.position += 1
        return True, self.frame.copy()
    def grab(self):
        self.grab_calls += 1
        if self.position >= 20:
            return False
        self.position += 1
        return True
    def release(self):
        pass


class VideoPipelineTests(unittest.TestCase):
    def test_synthetic_video_sampling_and_stable_identity(self):
        import video_track
        cap = _FakeCapture()
        with patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=_FakeYOLO)}), \
             patch("cv2.VideoCapture", return_value=cap), \
             patch.object(video_track, "video_duration", return_value=1.0):
            samples, duration, teams = video_track._track_video(
                "simulado.mp4", "modelo_propio.pt", sample_fps=10,
                imgsz=320, color_local_hex="#FF0000",
                color_visitante_hex="#0000FF", pitch_model="")
        self.assertEqual(10, len(samples))
        self.assertEqual(1.0, duration)
        self.assertEqual([round(i / 10, 1) for i in range(10)],
                         [round(s["t"], 1) for s in samples])
        self.assertEqual({0}, {s["people"][0]["track_id"] for s in samples})
        self.assertEqual("local", teams.get(0))
        self.assertEqual(10, cap.grab_calls)
        self.assertLessEqual(cap.read_calls, 11)


if __name__ == "__main__":
    unittest.main()
