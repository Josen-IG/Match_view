import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from retrain_experiment import box_iou, clipped_label, match_id, fixed_threshold_counts


class ExperimentTests(unittest.TestCase):
    def test_clip_visible_part(self):
        line, changed = clipped_label("1 0.01 0.5 0.1 0.2")
        self.assertTrue(changed)
        _, x, y, w, h = map(float, line.split())
        self.assertAlmostEqual(x-w/2, 0)
        self.assertAlmostEqual(x+w/2, .06)
        self.assertAlmostEqual(y, .5)
        self.assertAlmostEqual(h, .2)

    def test_invalid_labels_fail(self):
        for line in ("0 nan .5 .1 .1", "0 .5 .5 0 .1", "2 .5 .5 .1 .1", "0 .5 .5 .1"):
            with self.assertRaises(ValueError):
                clipped_label(line)

    def test_unchanged_box(self):
        self.assertEqual(clipped_label("0 .5 .5 .1 .1"), ("0 .5 .5 .1 .1", False))

    def test_match_groups(self):
        for name in ("f_12_30", "f2_04650", "cirbonero_r2_0000"):
            self.assertEqual(match_id(name), "cirbonero_milagres")
        self.assertEqual(match_id("alcala_fuenlabrada_0001"), "alcala_fuenlabrada")

    def test_iou(self):
        self.assertEqual(box_iou([0,0,1,1], [0,0,1,1]), 1)
        self.assertEqual(box_iou([0,0,1,1], [2,2,3,3]), 0)
        self.assertAlmostEqual(box_iou([0,0,2,2], [1,1,3,3]), 1/7)

    def test_prediction_memory_bounded_by_one_image(self):
        empty = MagicMock()
        empty.cpu.return_value.tolist.return_value = []
        model = MagicMock()
        model.predict.return_value = [SimpleNamespace(
            orig_shape=(100, 100), boxes=SimpleNamespace(conf=empty, cls=empty, xyxy=empty))]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'images'/'val').mkdir(parents=True)
            (root/'labels'/'val').mkdir(parents=True)
            for name in ('a', 'b'):
                (root/'images'/'val'/(name+'.jpg')).touch()
                (root/'labels'/'val'/(name+'.txt')).write_text('')
            with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda _: model)}):
                report = fixed_threshold_counts('weights.pt', root, root, 'mock')
            self.assertEqual(report['frames'], 2)
            self.assertEqual(model.predict.call_count, 2)
            for call in model.predict.call_args_list:
                self.assertIsInstance(call.args[0], str)
