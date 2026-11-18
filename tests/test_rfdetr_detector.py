from contextlib import nullcontext
import sys
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np
from rfdetr_detector import SoccerNetDetector, review_people, normalize_device
from test_video_pipeline import _FakeYOLO, _FakeCapture, _CpuTensor


class SoccerNetAdapterTests(unittest.TestCase):
    def load_detector(self, weights="local.pth", strict_error=False):
        network = Mock()
        if strict_error:
            network.load_state_dict.side_effect = RuntimeError("shape mismatch")
        model = Mock()
        model.model.model = network
        factory = Mock(return_value=model)
        torch = types.SimpleNamespace(
            serialization=types.SimpleNamespace(safe_globals=lambda _: nullcontext()),
            load=Mock(return_value={"model": {"class_embed.weight": "weights"}}))
        download = Mock(return_value="cached.pth")
        with patch.dict(sys.modules, {
                "rfdetr": types.SimpleNamespace(RFDETRBase=factory), "torch": torch,
                "huggingface_hub": types.SimpleNamespace(hf_hub_download=download),
        }), patch("rfdetr_detector.Path.is_file", side_effect=lambda: True):
            detector = SoccerNetDetector(weights, device="0")
        return detector, network, factory, torch, download

    def test_strict_loading_four_class_head_and_gpu_index(self):
        _, network, factory, torch, _ = self.load_detector()
        factory.assert_called_once_with(pretrain_weights=None, device="cuda:0")
        network.reinitialize_detection_head.assert_called_once_with(4)
        network.load_state_dict.assert_called_once_with({"class_embed.weight": "weights"}, strict=True)
        self.assertTrue(torch.load.call_args.kwargs['weights_only'])
        network.to.assert_called_once_with("cuda:0")
        network.eval.assert_called_once()

    def test_mismatched_checkpoint_is_not_partially_loaded(self):
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            self.load_detector(strict_error=True)

    def test_local_file_avoids_download(self):
        _, _, _, _, download = self.load_detector("hf")
        download.assert_not_called()

    def test_public_download_without_credentials(self):
        from rfdetr_detector import RFDETR_REPO, RFDETR_REVISION
        with patch.dict(sys.modules, {
                "rfdetr": types.SimpleNamespace(RFDETRBase=Mock()),
                "torch": types.SimpleNamespace(serialization=types.SimpleNamespace(
                    safe_globals=lambda _: nullcontext()), load=Mock(return_value={"model": {"x": 1}})),
                "huggingface_hub": types.SimpleNamespace(hf_hub_download=Mock(return_value="cache.pth")),
        }), patch("rfdetr_detector.Path.is_file", side_effect=[False, False, True]):
            import huggingface_hub
            SoccerNetDetector("hf")
            huggingface_hub.hf_hub_download.assert_called_once_with(
                RFDETR_REPO, "weights/checkpoint_best_regular.pth",
                revision=RFDETR_REVISION, token=False)

    def test_four_classes_rgb_conversion_and_flipped_coordinates(self):
        detector, _, _, _, _ = self.load_detector()
        detector.model.predict.return_value = types.SimpleNamespace(
            xyxy=np.array([[20, 10, 40, 30]] * 4),
            class_id=np.array([0, 1, 2, 3]), confidence=np.array([.8] * 4))
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        result = detector.predict(frame, True, True)
        rgb = detector.model.predict.call_args.args[0]
        self.assertEqual([0, 0, 255], rgb[0, 0].tolist())
        self.assertEqual(1, len(result['ball']))
        self.assertEqual(['player', 'referee', 'goalkeeper'], [p['class_name'] for p in result['people']])
        self.assertEqual((85., 80.), (result['ball'][0]['x'], result['ball'][0]['y']))
        self.assertEqual((30., 30.), (result['ball'][0]['px'], result['ball'][0]['py']))

    def test_invalid_boxes_scores_and_unknown_classes_rejected(self):
        detector, _, _, _, _ = self.load_detector()
        detector.model.predict.return_value = types.SimpleNamespace(
            xyxy=np.array([[0, 0, 5, 5], [1, 1, 0, 0], [float('nan'), 0, 5, 5], [0, 0, 5, 5]]),
            class_id=np.array([0, 0, 0, 50]), confidence=np.array([.1, .8, .9, .9]))
        self.assertEqual({'ball': [], 'people': []}, detector.predict(np.zeros((10, 10, 3))))

    def test_review_keeps_team_and_matches_each_box_only_once(self):
        people = [{'caja': (0, 0, 10, 10), 'team_hint': 'local'},
                  {'caja': (0, 0, 10, 10), 'team_hint': 'visitante'}]
        review_people(people, [{'caja': (0, 0, 10, 10), 'class_name': 'referee', 'conf': .8}])
        self.assertEqual('local', people[0]['team_hint'])
        self.assertEqual('referee', people[0]['rfdetr_review']['class_name'])
        self.assertNotIn('rfdetr_review', people[1])

    def test_device_and_threshold_validation(self):
        self.assertEqual('cuda:1', normalize_device(1))
        with self.assertRaises(ValueError): normalize_device('0,1')
        with self.assertRaises(ValueError): SoccerNetDetector(confidence=0)


class SoccerNetPipelineTests(unittest.TestCase):
    def run_pipeline(self, base_ball=False, specialist_ball=False, enabled=True, fail=False):
        import video_track
        class Detector(_FakeYOLO):
            def predict(self, *args, **kwargs):
                result = super().predict(*args, **kwargs)
                if not base_ball:
                    result[0].boxes.cls = _CpuTensor([1.])
                    result[0].boxes.xyxy = _CpuTensor([[25, 35, 50, 95]])
                    result[0].boxes.conf = _CpuTensor([.98])
                return result
        class Tracker:
            def __init__(self, **kwargs): pass
            def update(self, detections, *a, **kw): return [0] * len(detections)
            def predict_missing(self, *a): return []
        ball = dict(x=42., y=52., px=67., py=66., w=3., h=5., conf=.8,
                    pitch_x=None, pitch_y=None, detector='rfdetr_soccernet')
        with patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace(YOLO=Detector)}), \
             patch('cv2.VideoCapture', return_value=_FakeCapture()), \
             patch.object(video_track, 'video_duration', return_value=1.), \
             patch.object(video_track, 'PlayerIDTracker', Tracker), \
             patch('ball_detector.SoccerBallDetector') as specialist, \
             patch('rfdetr_detector.SoccerNetDetector') as reviewer:
            specialist.return_value.predict.return_value = [dict(ball, detector='yolov9_soccer_ball')] if specialist_ball else []
            reviewer.return_value.predict.return_value = {
                'ball': [ball], 'people': [{'caja': (25, 35, 50, 95), 'class_name': 'player', 'conf': .9}]}
            if fail:
                reviewer.return_value.predict.side_effect = RuntimeError('inference failed')
            samples, _, _ = video_track._track_video(
                'fake.mp4', 'base.pt', 10, 320, '#FF0000', '#0000FF',
                pitch_model='', pitch_only=False, ball_model='hf',
                rfdetr_model='hf' if enabled else None)
        return samples, reviewer

    def test_disabled_does_not_load_rfdetr(self):
        _, reviewer = self.run_pipeline(base_ball=True, enabled=False)
        reviewer.assert_not_called()

    def test_base_and_specialist_have_priority(self):
        for kwargs in ({'base_ball': True}, {'specialist_ball': True}):
            samples, reviewer = self.run_pipeline(**kwargs)
            reviewer.return_value.predict.assert_not_called()
            self.assertTrue(all(not s['rfdetr_reviewed'] for s in samples))

    def test_last_fallback_recovers_ball_and_reviews_existing_people(self):
        samples, reviewer = self.run_pipeline()
        self.assertEqual(10, reviewer.return_value.predict.call_count)
        self.assertTrue(all(s['ball_detector'] == 'rfdetr_soccernet' for s in samples))
        self.assertTrue(all(len(s['people']) == 1 and s['rfdetr_reviewed'] for s in samples))
        self.assertTrue(all(s['people'][0]['rfdetr_review']['class_name'] == 'player' for s in samples))

    def test_inference_error_is_not_silently_ignored(self):
        import video_track
        with self.assertRaisesRegex(video_track.VideoProcessingError, 'inference failed'):
            self.run_pipeline(fail=True)

    def test_public_apis_forward_rfdetr_settings(self):
        import video_track
        for api in (video_track.analyze_video, video_track.analyze_video_full):
            with patch.object(video_track, '_analizar_impl', return_value={'results': []}) as impl:
                api('video.mp4', rfdetr_model='local.pth', rfdetr_conf=.6)
                self.assertEqual('local.pth', impl.call_args.kwargs['rfdetr_model'])
                self.assertEqual(.6, impl.call_args.kwargs['rfdetr_conf'])


if __name__ == '__main__':
    unittest.main()
