"""Regresión de calibración con keypoints sintéticos del campo."""
import unittest

import numpy as np
import video_track


class PitchCalibrationTests(unittest.TestCase):
    def test_keypoints_recover_world_coordinates(self):
        mapper = video_track.PitchMapper()
        points = np.zeros((29, 3), dtype=np.float32)
        for index, (x, y) in video_track.PITCH_KEYPOINTS_M.items():
            points[index] = [x * 3 + 100, y * 2 + 70, .99]
        self.assertTrue(mapper.update(points, 1.0))
        self.assertIn(mapper.calidad, ("media", "alta"))
        self.assertGreaterEqual(mapper.inlier_ratio, .65)
        pos = mapper.to_pitch(130, 90)
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(10, pos[0], delta=.1)
        self.assertAlmostEqual(10, pos[1], delta=.1)

    def test_unreliable_points_not_accepted(self):
        mapper = video_track.PitchMapper()
        all_zero = np.zeros((29, 3), dtype=np.float32)
        self.assertFalse(mapper.update(all_zero, 0))
        self.assertIsNone(mapper.to_pitch(150, 250))

    def test_no_cross_frame_matrix_mixing(self):
        mapper = video_track.PitchMapper()
        pts = np.zeros((29, 3), dtype=np.float32)
        for index, (x, y) in video_track.PITCH_KEYPOINTS_M.items():
            pts[index] = [x * 2 + 50, y * 2 + 60, .99]
        self.assertTrue(mapper.update(pts, 1))
        first = mapper.to_pitch(70, 80)
        pts[:, 0] += 60  # cámara se desplaza 60px
        self.assertTrue(mapper.update(pts, 2))
        second = mapper.to_pitch(130, 80)
        self.assertAlmostEqual(first[0], second[0], delta=.1)


if __name__ == "__main__":
    unittest.main()
