from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from playback_media import prepare_playback_clip, media_version


class PlaybackMediaTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_real_cut_starts_at_requested_frame_and_has_segment_duration(self):
        from video_track import video_duration
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'frames.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64,48))
            self.assertTrue(writer.isOpened())
            for i in range(20): writer.write(np.full((48,64,3), i*10, dtype=np.uint8))
            writer.release()
            clip, offset = prepare_playback_clip(source, .5, 1.3)
            self.assertEqual(.5, offset)
            self.assertNotEqual(source, clip)
            self.assertAlmostEqual(.8, video_duration(clip), delta=.12)
            cap = cv2.VideoCapture(str(clip))
            try:
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertAlmostEqual(50, float(frame.mean()), delta=4)
            finally:
                cap.release()
            with patch('video_track.video_duration', return_value=2), patch('video_track._run', side_effect=AssertionError('cache should be reused')):
                cached, offset2 = prepare_playback_clip(source, .5, 1.3)
            self.assertEqual(clip, cached)
            self.assertEqual(offset, offset2)

    def test_url_identity_changes_when_served_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/'a.mp4', Path(tmp)/'b.mp4'
            first.write_bytes(b'123'); second.write_bytes(b'123')
            self.assertNotEqual(media_version(first), media_version(second))

    def test_invalid_ranges_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.mp4'; source.write_bytes(b'video')
            for start, end in ((-1, 2), (2, 1), (float('nan'), 2)):
                with self.assertRaises(ValueError): prepare_playback_clip(source, start, end)

    @unittest.skipUnless(shutil.which('node'), 'Node required')
    def test_clip_clock_preserves_original_tracking_coordinates(self):
        html = (Path(__file__).resolve().parents[1]/'analizador_video.html').read_text(encoding='utf-8')
        begin = html.index('    function playbackOffset(){')
        end = html.index('    var trackCanvas', begin)
        clock = html[begin:end]
        program = '''const assert=require('assert');
const window={VIDEO_OFFSET_S:900};
const state={videoFromServer:true};
const video={currentTime:0,duration:600};
''' + clock + '''
assert.strictEqual(mediaClock.currentTime,900);
mediaClock.currentTime=915;
assert.strictEqual(video.currentTime,15);
assert.strictEqual(mediaClock.duration,1500);
state.videoFromServer=false;
mediaClock.currentTime=915;
assert.strictEqual(video.currentTime,915);
assert.strictEqual(mediaClock.currentTime,915);
'''
        result = subprocess.run(['node','-e',program],capture_output=True,text=True)
        self.assertEqual(0,result.returncode,result.stderr)
