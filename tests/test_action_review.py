import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from action_review import analyze_saved_actions
from model_jobs import worker_python


class SavedActionTests(unittest.TestCase):
    def prepare(self, folder):
        video = Path(folder) / 'video.mp4'
        video.touch()
        path = Path(folder) / 'tracking.json'
        data = {'video_filename': video.name, 'analysis_start_s': 900,
                'analysis_end_s': 1500, 'samples': [{'t': 1380, 'ball': None}],
                'stats': {'goals': 0}, 'events': [], 'suggested_events': [
                    {'id': 'before', 'source': 'opensportslab_ball_action',
                     't': 950, 'type': 'shot', 'status': 'suggested'}]}
        path.write_text(json.dumps(data), encoding='utf-8')
        return path, video, data

    def test_model_goal_at_23_minutes_keeps_tracking_and_statistics(self):
        payload = {'data': [{'events': [
            {'label': 'GOAL', 'position_ms': 20000, 'confidence': .85}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path, video, before = self.prepare(tmp)
            with patch('action_review.validate_model_environment'), \
                 patch('action_review.run_optional_model', return_value=payload) as run:
                events = analyze_saved_actions(path, video, 1360, 1400)
            after = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(('goal', 1380, 'suggested'),
                         (events[0]['type'], events[0]['t'], events[0]['status']))
        self.assertEqual((1360, 40), run.call_args.args[2:4])
        for key in ('samples', 'stats', 'events'):
            self.assertEqual(before[key], after[key])
        self.assertEqual(2, len(after['suggested_events']))

    def test_failed_model_does_not_modify_saved_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, video, _ = self.prepare(tmp)
            before = path.read_bytes()
            with patch('action_review.validate_model_environment', side_effect=RuntimeError('missing')):
                with self.assertRaises(RuntimeError):
                    analyze_saved_actions(path, video, 1360, 1400)
            self.assertEqual(before, path.read_bytes())

    def test_rerunning_replaces_proposals_instead_of_duplicating(self):
        payload = {'data': [{'events': [
            {'label': 'GOAL', 'position_ms': 20000, 'confidence': .85}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path, video, _ = self.prepare(tmp)
            with patch('action_review.validate_model_environment'), \
                 patch('action_review.run_optional_model', return_value=payload):
                analyze_saved_actions(path, video, 1360, 1400)
                analyze_saved_actions(path, video, 1360, 1400)
            self.assertEqual(2, len(json.loads(path.read_text())['suggested_events']))

    def test_wrong_interval_or_video_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, video, _ = self.prepare(tmp)
            with patch('action_review.run_optional_model') as run:
                for start, end in ((899, 1400), (1400, 1360), (1360, 1501)):
                    with self.assertRaises(ValueError):
                        analyze_saved_actions(path, video, start, end)
                other = Path(tmp) / 'other.mp4'; other.touch()
                with self.assertRaises(ValueError):
                    analyze_saved_actions(path, other)
                run.assert_not_called()

    def test_isolated_environment_is_discovered_without_shell_variables(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch('model_jobs.Path.is_file', return_value=True):
            path = worker_python('actions')
        self.assertIn('.model-envs', path)
        self.assertIn('actions', path)
