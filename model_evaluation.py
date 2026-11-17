"""Reproducible evaluation against HUMAN-ANNOTATED match frames.

Ground truth JSON:
{
 "frames": [{"t": 0.0, "players":[{"id":"home_7","x":41,"y":35}],
             "ball":[52,50], "pitch":{"home_7":[40,25]}}],
 "events":[{"type":"shot","t":12.2}]
}
Coordinates x/y for objects are normalized video pixels in 0..100;
pitch uses metres (105x68), when independently annotated.
Predicted JSON is Match_view's exported tracking (samples, events).
No labels => no claimed precision.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _xy(value):
    try:
        if isinstance(value, dict):
            x, y = float(value["x"]), float(value["y"])
        else:
            x, y = float(value[0]), float(value[1])
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _score_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1}


def _greedy_pairs(truth, predicted, max_distance):
    distances = []
    for i, a in enumerate(truth):
        for j, b in enumerate(predicted):
            distance = math.dist(a, b)
            if distance <= max_distance:
                distances.append((distance, i, j))
    used_a, used_b, pairs = set(), set(), []
    for distance, i, j in sorted(distances):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, distance))
    return pairs


def evaluate_tracking(ground_truth, exported, time_tolerance=.12,
                      player_gate=3.0, ball_gate=2.5):
    frames = ground_truth.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Human-annotated ground truth frames required")
    samples = sorted(exported.get("samples") or [], key=lambda s: s.get("t", 0))
    player_tp = player_fp = player_fn = ball_tp = ball_fp = ball_fn = 0
    id_switches = 0
    ball_errors = []
    pitch_errors = []
    prior_id_assignment = {}
    matched_frames = 0
    for truth in frames:
        t = float(truth["t"])
        candidates = [p for p in samples if abs(float(p.get("t", -1e9)) - t) <= time_tolerance]
        sample = min(candidates, key=lambda s: abs(s["t"] - t)) if candidates else {}
        if sample:
            matched_frames += 1
        truth_players = [(str(p["id"]), _xy(p)) for p in truth.get("players", [])]
        truth_players = [(tid, p) for tid, p in truth_players if p is not None]
        detected = [p for p in (sample.get("p") or [])
                    if len(p) >= 6 and _xy(p) is not None]
        pairs = _greedy_pairs([p for _, p in truth_players],
                              [_xy(p) for p in detected], player_gate)
        player_tp += len(pairs)
        player_fp += len(detected) - len(pairs)
        player_fn += len(truth_players) - len(pairs)
        pitch_by_tid = {str(row[3]): _xy(row) for row in sample.get("pp") or []
                        if len(row) >= 4}
        for i, j, _ in pairs:
            human_id = truth_players[i][0]
            model_id = str(detected[j][5])
            if human_id in prior_id_assignment and prior_id_assignment[human_id] != model_id:
                id_switches += 1
            prior_id_assignment[human_id] = model_id
            gt_pitch = _xy((truth.get("pitch") or {}).get(human_id))
            predicted_pitch = pitch_by_tid.get(model_id)
            if gt_pitch and predicted_pitch:
                pitch_errors.append(math.dist(gt_pitch, predicted_pitch))
        truth_ball = _xy(truth.get("ball"))
        measured_ball = (_xy(sample.get("b")) if sample.get("bf") == "yolo" else None)
        if truth_ball and measured_ball:
            distance = math.dist(truth_ball, measured_ball)
            if distance <= ball_gate:
                ball_tp += 1
            else:
                ball_fn += 1
                ball_fp += 1
            ball_errors.append(distance)
        elif truth_ball:
            ball_fn += 1
        elif measured_ball:
            ball_fp += 1
    return {
        "annotated_frames": len(frames),
        "matched_tracking_frames": matched_frames,
        "players": _score_counts(player_tp, player_fp, player_fn),
        "ball": _score_counts(ball_tp, ball_fp, ball_fn),
        "id_switches_observed": id_switches,
        "ball_position_mae_image_u": (
            sum(ball_errors) / len(ball_errors) if ball_errors else None),
        "pitch_position_mae_m": (
            sum(pitch_errors) / len(pitch_errors) if pitch_errors else None),
        "pitch_pairs": len(pitch_errors),
        "note": ("Frame matching uses nearest timestamps and greedy player matching. "
                 "ID switches are only observable at labelled times; "
                 "these are not official MOT/HOTA metrics."),
    }


def evaluate_events(ground_truth, exported, tolerance_s=2.0):
    true = [e for e in ground_truth.get("events", [])
            if e.get("type") and isinstance(e.get("t"), (int, float))]
    # Confirmed legacy Match_view events use 'event' and 'seconds'
    # relative to the kickoff. Experimental hypotheses are deliberately
    # excluded from this validation.
    kickoff = float(exported.get("kickoff_s") or 0.)
    predicted = []
    for event in exported.get("events", []):
        kind = event.get("type") or event.get("event")
        moment = event.get("t")
        if not isinstance(moment, (int, float)):
            seconds = event.get("seconds")
            moment = (kickoff + seconds
                      if isinstance(seconds, (int, float)) else None)
        if kind and isinstance(moment, (int, float)):
            predicted.append({"type": kind, "t": moment})
    output = {}
    for kind in sorted({e["type"] for e in true + predicted}):
        truth_times = [[float(x["t"]), False] for x in true if x["type"] == kind]
        pred_times = [float(x["t"]) for x in predicted if x["type"] == kind]
        tp = 0
        for t in pred_times:
            available = [(abs(t-v[0]), i) for i, v in enumerate(truth_times)
                         if not v[1] and abs(t-v[0]) <= tolerance_s]
            if available:
                _, idx = min(available)
                truth_times[idx][1] = True
                tp += 1
        output[kind] = _score_counts(tp, len(pred_times)-tp, len(truth_times)-tp)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--output", help="Optional result JSON filename")
    args = parser.parse_args()
    gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    pred = json.loads(Path(args.tracking).read_text(encoding="utf-8"))
    report = {"tracking": evaluate_tracking(gt, pred),
              "events": evaluate_events(gt, pred)}
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result)
    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
