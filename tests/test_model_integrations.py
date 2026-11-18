import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import numpy as np
from model_jobs import normalize_actions, attach_gamestate, worker_python, run_optional_model
from pitch_detector import pitch_points_32, load_pitch_detector


def predictions(events):
    return {'version': '2.0', 'data': [{'events': events}]}


class Pitch32Tests(unittest.TestCase):
    def test_topology_has_distinct_corners_and_penalty_spots(self):
        points = pitch_points_32()
        self.assertEqual(32, len(points))
        self.assertEqual((0, 0), points[0])
        self.assertEqual((0, 68), points[5])
        self.assertEqual((105, 68), points[29])
        self.assertEqual((11, 34), points[8])
        self.assertEqual((94, 34), points[21])

    def test_32_keypoints_recover_pitch_without_changing_legacy_map(self):
        from video_track import PitchMapper
        points = pitch_points_32()
        mapper = PitchMapper(points)
        data = np.array([[x*3+100, y*2+70, .99] for x, y in points.values()], dtype=np.float32)
        self.assertTrue(mapper.update(data, 1.))
        pos = mapper.to_pitch(130, 90)
        self.assertAlmostEqual(10, pos[0], places=2)
        self.assertAlmostEqual(10, pos[1], places=2)
        with self.assertRaises(ValueError):
            PitchMapper().update(data, 1.)

    def test_anonymous_download_and_shape_check(self):
        fake = types.SimpleNamespace(model=types.SimpleNamespace(kpt_shape=[32, 3]))
        download = Mock(return_value='cache.pt')
        with patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace(YOLO=Mock(return_value=fake)),
                     'huggingface_hub': types.SimpleNamespace(hf_hub_download=download)}), \
             patch('pitch_detector.Path.is_file', return_value=False):
            self.assertIs(fake, load_pitch_detector())
            self.assertFalse(download.call_args.kwargs['token'])
            fake.model.kpt_shape = [29, 3]
            with self.assertRaisesRegex(ValueError, '32 keypoints'):
                load_pitch_detector()

    def test_ultralytics_yaml_topology_metadata(self):
        fake = types.SimpleNamespace(model=types.SimpleNamespace(yaml={'kpt_shape': [32, 3]}))
        with patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace(YOLO=Mock(return_value=fake))}), \
             patch('pitch_detector.Path.is_file', return_value=True):
            self.assertIs(fake, load_pitch_detector())


class ActionTests(unittest.TestCase):
    def test_absolute_time_offset_nms_and_unverified_status(self):
        data = predictions([
            {'label': 'SHOT', 'position_ms': 2000, 'confidence': .7},
            {'label': 'SHOT', 'position_ms': 2500, 'confidence': .9},
            {'label': 'PASS', 'position_ms': 2500, 'confidence': .8}])
        events = normalize_actions(data, 900, 10)
        self.assertEqual(2, len(events))
        self.assertEqual({'shot', 'pass'}, {e['type'] for e in events})
        self.assertTrue(all(e['t'] == 902.5 and e['seconds'] == 2.5 for e in events))
        self.assertTrue(all(e['start_s'] >= 900 and e['end_s'] <= 910 for e in events))
        self.assertTrue(all(e['status'] == 'suggested' for e in events))

    def test_all_twelve_labels_and_invalid_predictions(self):
        from model_jobs import ACTION_LABELS
        data = predictions([{'label': label, 'position_ms': 1000, 'confidence': .8}
                            for label in ACTION_LABELS])
        data['data'][0]['events'] += [
            {'label': 'GOAL', 'position_ms': 20000, 'confidence': .99},
            {'label': 'GOAL', 'position_ms': -1, 'confidence': .99},
            {'label': 'SHOT', 'position_ms': 2000, 'confidence': float('nan')},
            {'label': 'UNKNOWN', 'position_ms': 1000, 'confidence': .99},
            {'label': 'PASS', 'position_ms': 1000, 'confidence': .1}]
        self.assertEqual(12, len(normalize_actions(data, 0, 10)))

    def test_invalid_schema_and_multiple_video_outputs_fail(self):
        for data in ({}, {'data': []}, {'data': [{}, {}]}):
            with self.assertRaises(ValueError): normalize_actions(data, 0, 10)
        with self.assertRaises(ValueError): normalize_actions(predictions([]), 0, 10, nms_seconds=0)


class GameStateTests(unittest.TestCase):
    def sample(self):
        return [{'t_video': 100.2, 'people': [dict(x=20., y=30., w=10., h=20.,
                                                  track_id=4, team='local')]}]

    def payload(self):
        return {'source': 'soccernet_gamestate', 'fps': 25, 'detections': [
            {'frame': 5, 'box': [20., 30., 10., 20.], 'external_track_id': 88,
             'jersey_number': 9, 'role': 'player', 'team': 'right'}]}

    def test_association_keeps_original_id_and_team(self):
        samples = self.sample()
        result = attach_gamestate(samples, self.payload(), 100)
        self.assertEqual(1, result['matched_observations'])
        person = samples[0]['people'][0]
        self.assertEqual((4, 'local'), (person['track_id'], person['team']))
        self.assertEqual(88, person['gamestate']['external_track_id'])
        self.assertEqual(9, person['gamestate']['jersey_number'])

    def test_wrong_time_or_nonoverlapping_box_is_not_attached(self):
        for row_change in ({'frame': 6}, {'box': [70., 60., 10., 20.]}):
            payload = self.payload(); payload['detections'][0].update(row_change)
            samples = self.sample()
            self.assertEqual(0, attach_gamestate(samples, payload, 100)['matched_observations'])
            self.assertNotIn('gamestate', samples[0]['people'][0])

    def test_flip_is_applied_only_to_matching_coordinates(self):
        samples = self.sample(); samples[0]['people'][0]['x'] = 80
        self.assertEqual(1, attach_gamestate(samples, self.payload(), 100, flip_x=True)['matched_observations'])

    def test_json_export_keeps_suggestions_separate_from_confirmed_events(self):
        from video_track import export_tracking_json
        samples = self.sample(); attach_gamestate(samples, self.payload(), 100)
        suggested = normalize_actions(predictions([
            {'label': 'GOAL', 'position_ms': 1000, 'confidence': .9}]), 100, 10)
        info = {'kickoff_offset_s': 100, 'match_duration_s': 10,
                'suggested_events': suggested, 'integrations': {'gamestate': True}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tracking.json'
            export_tracking_json(samples, info, None, path)
            data = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual([], data['events'])
        self.assertEqual('goal', data['suggested_events'][0]['type'])
        self.assertEqual(88, data['samples'][0]['gi'][0]['external_track_id'])


class WorkerTests(unittest.TestCase):
    def test_missing_external_python_is_actionable(self):
        with patch.dict(os.environ, {'MATCHVIEW_GSR_PYTHON': '/does/not/exist'}):
            with self.assertRaisesRegex(FileNotFoundError, 'MATCHVIEW_GSR_PYTHON'):
                worker_python('gamestate')

    def test_crop_and_worker_use_argument_lists_and_preserve_absolute_metadata(self):
        payload = predictions([])
        calls = []
        def run(command, **kwargs):
            calls.append((command, kwargs))
            if '--output' in command:
                Path(command[command.index('--output')+1]).write_text(json.dumps(payload), encoding='utf-8')
            return types.SimpleNamespace(returncode=0, stderr='')
        with tempfile.TemporaryDirectory() as tmp, \
             patch('model_jobs.worker_python', return_value=sys.executable), \
             patch('model_jobs.subprocess.run', side_effect=run), \
             patch('video_track._tool_path', return_value='ffmpeg'):
            run_optional_model('actions', Path(tmp)/'video with spaces.mp4', 30, 10, tmp)
            saved = json.loads((Path(tmp)/'actions_predictions.json').read_text(encoding='utf-8'))
        self.assertEqual(2, len(calls))
        self.assertEqual('30', calls[0][0][calls[0][0].index('-ss')+1])
        self.assertFalse(any(kw.get('shell') for _, kw in calls))
        self.assertEqual(30, saved['offset_s'])
        self.assertEqual('1', calls[1][1]['env']['HF_HUB_DISABLE_IMPLICIT_TOKEN'])

    def test_environment_check_fails_before_models_run(self):
        from model_jobs import validate_model_environment
        with patch('model_jobs.worker_python', return_value=sys.executable), \
             patch('model_jobs.subprocess.run', return_value=types.SimpleNamespace(returncode=1, stderr='Missing model')):
            with self.assertRaisesRegex(RuntimeError, 'Missing model'):
                validate_model_environment('actions')

    def test_public_apis_forward_video_level_model_options(self):
        import video_track
        for api in (video_track.analyze_video, video_track.analyze_video_full):
            with patch.object(video_track, '_analizar_impl', return_value={'results': []}) as impl:
                api('video.mp4', actions_enabled=True, gamestate_enabled=True, action_conf=.7)
                self.assertTrue(impl.call_args.kwargs['actions_enabled'])
                self.assertTrue(impl.call_args.kwargs['gamestate_enabled'])
                self.assertEqual(.7, impl.call_args.kwargs['action_conf'])

    def test_analysis_exports_actions_without_turning_them_into_verified_events(self):
        import video_track
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace()}))
            stack.enter_context(patch.object(video_track, 'video_duration', return_value=100.))
            stack.enter_context(patch.object(video_track, '_track_video', return_value=([], 10., {})))
            for name in ('_suavizar_balon', '_filtro_kalman_balon', '_compensar_camara',
                         '_anotar_posesion', '_anotar_porteros'):
                stack.enter_context(patch.object(video_track, name, side_effect=lambda samples, *a: samples))
            stack.enter_context(patch.object(video_track, '_corregir_eje_x_campo'))
            stack.enter_context(patch.object(video_track, 'reject_collective_pitch_jumps'))
            stack.enter_context(patch.object(video_track, 'posesion_por_tramos', return_value=[]))
            stats = stack.enter_context(patch.object(video_track, '_calcular_estadisticas', return_value={}))
            stack.enter_context(patch.object(video_track, '_exportar_informe', return_value={}))
            exporter = stack.enter_context(patch.object(video_track, 'export_tracking_json', return_value=None))
            stack.enter_context(patch('model_jobs.validate_model_environment'))
            worker = stack.enter_context(patch('model_jobs.run_optional_model', return_value=predictions([
                {'label': 'SHOT', 'position_ms': 1000, 'confidence': .8}])))
            result = video_track.analyze_video_full('video.mp4', kickoff_offset=30,
                                                   max_minutes=1/6, actions_enabled=True)
        self.assertEqual(31, result['suggested_events'][0]['t'])
        self.assertEqual([], result['results'])
        self.assertEqual([], stats.call_args.args[1])
        self.assertEqual('actions', worker.call_args.args[0])
        self.assertEqual((30, 10), worker.call_args.args[2:4])
        self.assertEqual('shot', exporter.call_args.args[1]['suggested_events'][0]['type'])


if __name__ == '__main__':
    unittest.main()
