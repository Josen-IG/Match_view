"""Cola CPU de análisis en segundo plano: estado, progreso y errores."""
import threading
import time
import unittest
from video_jobs import submit_video_job, get_video_job


def wait_for_state(job_id, expected, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = get_video_job(job_id)
        if result["state"] == expected:
            return result
        time.sleep(.01)
    raise AssertionError(f"{job_id}: no llegó al estado {expected}: {result}")


class VideoJobsTests(unittest.TestCase):
    def test_job_returns_output_and_progress(self):
        def analyzer(video_path, progress):
            progress("Analizando", .65)
            return {"tracking_json": video_path}
        job_id = submit_video_job(analyzer, {"video_path": "prueba.json"})
        result = wait_for_state(job_id, "completed")
        self.assertEqual("prueba.json", result["result"]["tracking_json"])
        self.assertEqual(1.0, result["progress"])

    def test_errors_are_visible_without_crashing(self):
        def analyzer(progress):
            raise RuntimeError("YOLO unavailable")
        result = wait_for_state(submit_video_job(analyzer, {}), "failed")
        self.assertIn("YOLO unavailable", result["error"])

    def test_jobs_are_serialized(self):
        start = threading.Event()
        unblock = threading.Event()
        def first(progress):
            start.set()
            self.assertTrue(unblock.wait(3))
            return {"ok": 1}
        first_id = submit_video_job(first, {})
        self.assertTrue(start.wait(3))
        second_id = submit_video_job(lambda progress: {"ok": 2}, {})
        self.assertEqual("queued", get_video_job(second_id)["state"])
        unblock.set()
        self.assertEqual("completed", wait_for_state(first_id, "completed")["state"])
        self.assertEqual(2, wait_for_state(second_id, "completed")["result"]["ok"])

    def test_missing_job(self):
        self.assertIsNone(get_video_job("no-existe"))


if __name__ == "__main__":
    unittest.main()
