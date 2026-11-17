"""Experimental temporal football action evidence, NOT an event classifier.

A fast ball moving goalward is often a pass, cross or clearance. These
hypotheses additionally require a visible nearby player before release,
several direct YOLO ball observations and a coherent trajectory. They
MUST NOT be shown as confirmed shots until validated on human-labelled
matches. The production 'suggested_events' remains disabled.
"""
from __future__ import annotations

import math


def _ball(sample):
    point = sample.get("bp")
    if (sample.get("bf") != "yolo" or not isinstance(point, (list, tuple))
            or len(point) < 2):
        return None
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 105 and 0 <= y <= 68):
        return None
    return x, y


def _nearest_player(sample, ball):
    distances = []
    for p in sample.get("pp") or []:
        if len(p) < 3 or p[2] not in (0, 1, 2, 3):
            continue
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            distances.append((math.hypot(x-ball[0], y-ball[1]), p[2]))
    return min(distances) if distances else (float("inf"), None)


def experimental_shot_sequences(samples, min_release_m=10.0):
    """Return labelled-as-UNVERIFIED hypotheses, with supporting evidence.

    Requires at least 4 *directly detected* ball samples in a coherent
    trajectory after contact. Excludes speed-only detections and provides
    interpretable evidence for manual annotation and later model training.
    """
    points = []
    for sample in samples:
        position = _ball(sample)
        if position is None:
            continue
        try:
            t = float(sample["t"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(t):
            points.append((t, position, sample))
    points.sort(key=lambda p: p[0])
    results = []
    for i in range(len(points)-3):
        t0, p0, first = points[i]
        player_distance, team = _nearest_player(first, p0)
        if player_distance > 2.5 or team is None:
            continue
        # Only contiguous direct measurements of a continuous flight.
        future = []
        for p in points[i+1:]:
            dt = p[0] - t0
            if dt <= 0:
                continue
            if dt > 1.3:
                break
            if future and p[0]-future[-1][0] > .35:
                break
            future.append(p)
        if len(future) < 3:
            continue
        last_t, last_ball, _ = future[-1]
        dx, dy = last_ball[0]-p0[0], last_ball[1]-p0[1]
        dt = last_t-t0
        if not (.3 <= dt <= 1.3 and abs(dx) >= min_release_m):
            continue
        speed = math.hypot(dx, dy)/dt
        if not (12 <= speed <= 42) or abs(dx) < math.hypot(dx, dy)*.7:
            continue
        goal_x = 105 if dx > 0 else 0
        if not ((dx > 0 and last_ball[0] >= 75)
                or (dx < 0 and last_ball[0] <= 30)):
            continue
        target_y = p0[1]+(goal_x-p0[0])*dy/dx
        if not (27 <= target_y <= 41):
            continue
        # Reject zig-zags and homography teleports.
        previous_t, previous_pos = t0, p0
        coherent = True
        for t, pos, _ in future:
            step = math.hypot(pos[0]-previous_pos[0],pos[1]-previous_pos[1])
            if (step / (t-previous_t) > 44
                    or (pos[0]-previous_pos[0])*dx < 0):
                coherent = False
                break
            previous_t, previous_pos = t, pos
        if not coherent:
            continue
        if results and t0-results[-1]["t"] < 6:
            continue
        results.append({
            "t": round(t0, 2),
            "type": "possible_shot_sequence",
            "status": "experimental_unverified",
            "source": "temporal_ball_player_geometry",
            "contact_distance_m": round(player_distance, 2),
            "measured_ball_points": len(future)+1,
            "travel_m": round(math.hypot(dx, dy), 2),
            "release_speed_m_s": round(speed, 2),
            "estimated_goal_line_y_m": round(target_y, 2),
            "team_code_near_release": team,
            "warning": "Not proof of a shot: may be a pass, cross or clearance.",
        })
    return results
