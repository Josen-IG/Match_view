"""Quality gate for pitch coordinates used by the live tactical radar.

A homography kept alive for tens of seconds during VEO pan/zoom can look
plausible numerically while mapping players to the wrong pitch locations.
Keep the player detector running, but do not publish that mapping as a
trusted 2D location when calibration is stale or lacks evidence.
"""
from __future__ import annotations

import math


MAX_RADAR_CALIBRATION_AGE_S = 3.0
MAX_RADAR_REPROJECTION_ERROR_M = 1.5
MIN_RADAR_INLIER_RATIO = 0.70


def radar_calibration_valid(sample: dict) -> bool:
    """True only when an explicitly assessed recent pitch calibration exists."""
    if not sample.get("pitch_ok"):
        return False
    if sample.get("pitch_calidad") not in ("alta", "media"):
        return False
    try:
        age = float(sample["pitch_age_s"])
        error = float(sample["pitch_reprojection_error_m"])
        ratio = float(sample["pitch_inlier_ratio"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(age) and 0 <= age <= MAX_RADAR_CALIBRATION_AGE_S
        and math.isfinite(error) and 0 <= error <= MAX_RADAR_REPROJECTION_ERROR_M
        and math.isfinite(ratio) and MIN_RADAR_INLIER_RATIO <= ratio <= 1
    )


def reject_collective_pitch_jumps(samples, max_speed_m_s=13.0):
    """Invalidate a suspicious collective coordinate jump after calibration.

    Uses >=3 repeatedly identified observed players, not extrapolated IDs.
    The gate is intentionally conservative: a single sprint or tracking
    identity swap must never invalidate an otherwise useful frame.
    Returns rejected frame count. Keeps original image-space YOLO detections.
    """
    rejected = 0
    previous = None
    for sample in samples:
        if not radar_calibration_valid(sample):
            previous = None
            continue
        if previous is not None:
            dt = float(sample.get("t_video", sample.get("t", 0))) - float(
                previous.get("t_video", previous.get("t", 0)))
            if 0.025 <= dt <= 0.35:
                old = {p.get("track_id"): (p["pitch_x"], p["pitch_y"])
                       for p in previous.get("people") or []
                       if isinstance(p.get("track_id"), int)
                       and p.get("pitch_x") is not None
                       and p.get("pitch_y") is not None}
                matched = []
                for p in sample.get("people") or []:
                    tid = p.get("track_id")
                    pos = old.get(tid)
                    if (pos is not None and p.get("pitch_x") is not None
                            and p.get("pitch_y") is not None):
                        distance = math.hypot(
                            float(pos[0]) - float(p["pitch_x"]),
                            float(pos[1]) - float(p["pitch_y"]))
                        matched.append(distance / dt)
                if len(matched) >= 3 and (
                        sum(speed > max_speed_m_s for speed in matched) /
                        len(matched) >= .67):
                    sample["pitch_ok"] = False
                    sample["pitch_jump_rejected"] = True
                    for p in sample.get("people") or []:
                        p["pitch_x"] = None
                        p["pitch_y"] = None
                    if sample.get("ball") is not None:
                        sample["ball"]["pitch_x"] = None
                        sample["ball"]["pitch_y"] = None
                    sample["ball_pitch_s"] = None
                    rejected += 1
                    previous = None
                    continue
        previous = sample
    return rejected
