"""Pruebas para excluir espectadores sin perjudicar el balón."""
import unittest

import cv2
import numpy as np

from pitch_filter import field_mask_from_frame, filter_pitch_people


class _Mapper:
    def to_pitch(self, x, y):
        return (50.0, 34.0) if x < 150 else (50.0, -4.0)


class FieldFilterTests(unittest.TestCase):
    def test_green_turf_accepted_and_spectator_excluded(self):
        frame = np.full((240, 360, 3), (190, 190, 190), np.uint8)
        frame[65:, :] = (45, 115, 50)  # césped verde oscuro
        # Grada/edificio, arriba del borde del campo.
        frame[:65, :] = (120, 100, 105)
        mask = field_mask_from_frame(frame, target_width=360)
        self.assertTrue(mask.reliable, mask.reason)
        self.assertTrue(mask.contains(120, 180))
        self.assertFalse(mask.contains(120, 20))
        detections = [
            {"caja": (105, 100, 135, 180)},   # jugador real
            {"caja": (105, 5, 135, 40)},      # grada
        ]
        allowed, counts = filter_pitch_people(
            detections, _Mapper(), pitch_ok=False, field_mask=mask)
        self.assertEqual(1, len(allowed))
        self.assertEqual(1, counts["rejected"])

    def test_pitch_model_rejects_touchline_staff(self):
        frame = np.full((240, 360, 3), (45, 115, 50), np.uint8)
        mask = field_mask_from_frame(frame, target_width=360)
        dets = [{"caja": (30, 40, 50, 125)},
                {"caja": (260, 40, 280, 125)}]
        allowed, counts = filter_pitch_people(dets, _Mapper(), True, mask)
        self.assertEqual(1, len(allowed))
        self.assertEqual(1, counts["rejected"])

    def test_no_field_data_is_conservative(self):
        frame = np.zeros((200, 320, 3), np.uint8)
        mask = field_mask_from_frame(frame)
        self.assertFalse(mask.reliable)
        dets = [{"caja": (10, 20, 30, 70)}]
        allowed, counts = filter_pitch_people(dets, _Mapper(), False, mask)
        self.assertEqual([], allowed)
        self.assertEqual(1, counts["unverified"])

    def test_detected_ball_is_not_a_filter_input(self):
        frame = np.full((240, 360, 3), (45, 115, 50), np.uint8)
        mask = field_mask_from_frame(frame)
        allowed, _ = filter_pitch_people([], _Mapper(), False, mask)
        self.assertEqual([], allowed)


if __name__ == "__main__":
    unittest.main()
