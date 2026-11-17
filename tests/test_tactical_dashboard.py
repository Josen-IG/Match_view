"""Pruebas de radar y posesiones del visor sin lanzar Streamlit ni YOLO."""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "analizador_video.html"


class TacticalDashboardTests(unittest.TestCase):
    def test_panels_exist_and_ai_timeline_is_unchanged(self):
        html = HTML.read_text(encoding="utf-8")
        for panel in ("radarCanvas", "radarPlayers", "mvPossList",
                      "mvPossTeam", "mvPossNow", "mvTacticalInsights",
                      "laneTracking", "aiEventsList"):
            self.assertIn(f'id="{panel}"', html)
        for method in ("renderLiveRadar", "renderPossessionPanel",
                       "renderTacticalContext", "updateTacticalDashboard"):
            self.assertIn(f"function {method}", html)

    @unittest.skipUnless(shutil.which("node"), "Node.js no disponible")
    def test_radar_rejects_legacy_and_unstable_calibration(self):
        html = HTML.read_text(encoding="utf-8")
        match = re.search(
            r"/\\* MV-PURE-LOGIC-START \\*/(.*?)/\\* MV-PURE-LOGIC-END \\*/",
            html, re.DOTALL,
        )
        self.assertIsNotNone(match)
        script = match.group(1) + r"""
const assert = require('assert');
const player = (x,y,id) => [x,y,0,id];
let sample={t:1,pp:[player(30,30,1),player(50,30,2),player(70,30,3)],bp:[52,34]};
assert.strictEqual(reliableRadarSnapshot(sample,null).players.length,0);
assert.strictEqual(reliableRadarSnapshot({...sample,rk:false},null).players.length,0);
assert.strictEqual(reliableRadarSnapshot({...sample,rk:true},null).players.length,3);
let old={t:1,rk:true,pp:[player(30,30,1),player(50,30,2),player(70,30,3)]};
let jump={t:1.1,rk:true,pp:[player(38,30,1),player(58,30,2),player(78,30,3)]};
let result=reliableRadarSnapshot(jump,old);
assert.strictEqual(result.verified,false);
assert.strictEqual(result.players.length,0);
let normal={t:1.1,rk:true,pp:[player(30.3,30,1),player(50.4,30,2),player(70.2,30,3)]};
assert.strictEqual(reliableRadarSnapshot(normal,old).players.length,3);
console.log('radar quality OK');
"""
        result = subprocess.run(["node", "-e", script], text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("radar quality OK", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js no disponible")
    def test_pure_analytics_without_invented_pitch_positions(self):
        html = HTML.read_text(encoding="utf-8")
        match = re.search(
            r"/\* MV-PURE-LOGIC-START \*/(.*?)/\* MV-PURE-LOGIC-END \*/",
            html, re.DOTALL,
        )
        self.assertIsNotNone(match)
        script = match.group(1) + r"""
const assert = require('assert');
const inputs = [
  {t:0,o:0,bp:[10,20]}, {t:.5,o:0,bp:[17,30]},
  {t:1,o:0,bp:[25,32]}, {t:2,o:0,bp:[30,32]},
  {t:2.2,o:-1}, {t:3,o:1,bp:[72,34]},
  {t:3.6,o:1,bp:[75,34]}, {t:4.4,o:1,bp:[80,34]},
];
const possessions = derivePossessions(inputs);
assert.strictEqual(possessions.length,2);
assert.strictEqual(possessions[0].team,'local');
assert.strictEqual(possessions[1].team,'visitante');
assert.strictEqual(possessions[0].duration_s,2);
assert.strictEqual(possessions[1].duration_s,1.4);
assert.strictEqual(possessions[0].start_zone,'Tercio izquierdo');
assert.strictEqual(possessions[1].start_zone,'Tercio derecho');
const empty = radarSnapshot({p:[[50,50,4,9,0,2]],b:[50,50]});
assert.strictEqual(empty.players.length,0);
assert.strictEqual(empty.ball,null);
const sample = radarSnapshot({
 pp:[[30,20,0,7],[76,35,1,8],[200,40,0,11],[20,10,5,9]],
 bp:[52,31], bf:'yolo'
});
assert.strictEqual(sample.players.length,2);
assert.strictEqual(sample.ball.x,52);
assert.strictEqual(derivePossessions([{t:0,o:-1},{t:1,o:-1}]).length,0);
assert.strictEqual(possessionThird({b:[50,50]}),null);
console.log('tactical logic OK');
"""
        result = subprocess.run(["node", "-e", script], text=True,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tactical logic OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
