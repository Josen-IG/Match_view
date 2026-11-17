"""Regresión de interfaz y analítica observada de Match View."""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "analizador_video.html"


class ReliableVideoUITests(unittest.TestCase):
    def setUp(self):
        self.content = HTML.read_text(encoding="utf-8")

    def test_video_is_only_primary_view(self):
        self.assertNotIn('id="tabBoard"', self.content)
        self.assertNotIn('id="tabAnalytics"', self.content)
        self.assertNotIn('id="tabVideo"', self.content)
        self.assertIn('id="stageWrap"', self.content)

    def test_right_sidebar_and_observed_analytics(self):
        for marker in ('mv-radar-panel','mv-live-panel',
                       'mv-possession-panel','mv-context-panel',
                       'mv-measured-analysis','mvMeasuredBall',
                       'mvMeasuredPitch','mvMeasuredField'):
            self.assertIn(marker,self.content)
        self.assertIn('class="mv-side-details"', self.content)
        self.assertIn('ev.source !== \'ball_pitch_heuristic\'',self.content)

    @unittest.skipUnless(shutil.which("node"), "Node.js no disponible")
    def test_analytics_counts_only_observations(self):
        block = re.search(
            r"/\* MV-PURE-LOGIC-START \*/(.*?)/\* MV-PURE-LOGIC-END \*/",
            self.content, re.DOTALL,
        )
        self.assertIsNotNone(block)
        script = block.group(1) + r"""
const assert=require('assert');
let samples=[
 {t:0,bf:'yolo',b:[40,50],bp:[20,30],p:[[1],[2]],
  pp:[[20,30,0,4],[60,40,1,8]]},
 {t:.1,bf:'kalman',b:[45,50],p:[[1]],
  pp:[[300,60,0,4]]},
 {t:.2,bf:'yolo',b:[55,50],bp:[35,25],p:[],pp:[]},
];
const stats=measuredTrackingStats(samples,0,1);
assert.strictEqual(stats.samples,3);
assert.strictEqual(stats.ballYolo,2);
assert.strictEqual(stats.ballPitch,2);
assert.strictEqual(stats.positions,2);
assert.strictEqual(stats.byTeam.local,1);
assert.strictEqual(stats.byTeam.visitante,1);
assert.strictEqual(stats.peopleMean,1);
const empty=measuredTrackingStats(samples,1,2);
assert.strictEqual(empty.samples,0);
assert.strictEqual(empty.positions,0);
console.log('measured tracking OK');
"""
        p=subprocess.run(["node","-e",script],capture_output=True,text=True,timeout=15)
        self.assertEqual(0,p.returncode,p.stderr)
        self.assertIn("measured tracking OK",p.stdout)


if __name__ == "__main__":
    unittest.main()
