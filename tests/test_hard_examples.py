"""Hard-example mining and confirmed event-time evaluation tests."""
import unittest

from hard_example_mining import mine_hard_moments
from model_evaluation import evaluate_events


class HardExampleTests(unittest.TestCase):
    def test_ball_gap_suggests_a_correction_frame(self):
        s = [
            {"t": 0.0, "bf":"yolo", "b":[20,50]},
            {"t": .2, "bf":"kalman", "b":[21,50]},
            {"t": .5, "bf":"kalman", "b":[24,50]},
            {"t": 1.0, "bf":"yolo", "b":[30,50]},
        ]
        proposed = mine_hard_moments(s)
        self.assertEqual(1, len(proposed))
        self.assertEqual("ball_missing", proposed[0]["reason"])
        self.assertEqual(.5, proposed[0]["t"])

    def test_nearby_events_are_deduplicated(self):
        s = [{"t":x/10, "rk":False, "p":[[]]*9} for x in range(20)]
        proposed = mine_hard_moments(s, min_spacing_s=3.)
        self.assertEqual(1, len(proposed))

    def test_unverified_ids_are_separate(self):
        s = [{"t":1, "pe":[[1]]*4, "p":[]}]
        proposed = mine_hard_moments(s)
        self.assertEqual("player_occlusion", proposed[0]["reason"])


class ConfirmedEventTests(unittest.TestCase):
    def test_match_view_relative_event_times(self):
        truth = {"events":[{"type":"shot","t":902.0}]}
        pred = {"kickoff_s":900, "events":[{"event":"shot","seconds":2.0}]}
        result = evaluate_events(truth,pred)
        self.assertEqual(1,result["shot"]["tp"])
        self.assertEqual(1.,result["shot"]["precision"])

    def test_hypotheses_are_not_confirmed_events(self):
        truth={"events":[{"type":"shot","t":90.}]}
        pred={"events":[],"suggested_events":[{"type":"shot","t":90.}]}
        result=evaluate_events(truth,pred)
        self.assertEqual(1,result["shot"]["fn"])
        self.assertIsNone(result["shot"]["precision"])


if __name__ == "__main__":
    unittest.main()
