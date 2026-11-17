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
