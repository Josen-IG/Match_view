"""Seguimiento de IDs de jugadores con compensación de movimiento de cámara.

La identidad es estimada: sin dorsales o embeddings, no se garantiza tras
ausencias largas o cruces completamente ambiguos.
"""
from __future__ import annotations
import math
from collections import Counter


def estimate_camera_transform(previous_gray, current_gray, previous_boxes=()):
    """Transformación afín entre fotogramas grises, en coordenadas 0-100."""
    import cv2
    import numpy as np
    if previous_gray is None or current_gray is None:
        return None
    if previous_gray.shape != current_gray.shape:
        return None
    height, width = previous_gray.shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in previous_boxes:
        x0, x1 = max(0, int(x0)), min(width, int(x1))
        y0, y1 = max(0, int(y0)), min(height, int(y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=120,
                                    qualityLevel=0.015, minDistance=7, mask=mask)
    if points is None or len(points) < 8:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None,
        winSize=(19, 19), maxLevel=3,
    )
    if tracked is None or status is None:
        return None
    good = status.ravel() == 1
    if int(good.sum()) < 8:
        return None
    before = points.reshape(-1, 2)[good]
    after = tracked.reshape(-1, 2)[good]
    affine, inliers = cv2.estimateAffinePartial2D(
        before, after, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=1000, confidence=0.98,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 6:
        return None
    if int(inliers.sum()) / len(before) < 0.45:
        return None
    a, b, tx = map(float, affine[0])
    c, d, ty = map(float, affine[1])
    scale = math.hypot(a, c)
    if not (0.85 <= scale <= 1.15):
        return None
    return (a, b * height / width, tx * 100.0 / width,
            c * width / height, d, ty * 100.0 / height)


def flip_camera_transform(transform, flip_x=False, flip_y=False):
    """Adapta una transformación de cámara a ejes invertidos."""
    if transform is None:
        return None
    sx, sy = (-1 if flip_x else 1), (-1 if flip_y else 1)
    ox, oy = (100.0 if flip_x else 0.0), (100.0 if flip_y else 0.0)
    a, b, tx, c, d, ty = transform
    b_new, c_new = sx * sy * b, sx * sy * c
    return (a, b_new, sx * tx + ox - a * ox - b_new * oy,
            c_new, d, sy * ty + oy - c_new * ox - d * oy)


def _hue_distance(color_a, color_b):
    if color_a is None or color_b is None:
        return None
    if min(color_a[1], color_b[1]) < 55 or min(color_a[2], color_b[2]) < 45:
        return None
    delta = abs(color_a[0] - color_b[0])
    return min(delta, 180.0 - delta)


class PlayerIDTracker:
    """Asocia detecciones cx/cy/h/team_hint/hsv a identificadores estables."""

    def __init__(self, sample_fps=10, match_radius=8.0,
                 max_gap_seconds=2.0, reid_seconds=8.0):
        if sample_fps <= 0:
            raise ValueError("sample_fps debe ser positivo")
        self.sample_fps = float(sample_fps)
        self.match_radius = float(match_radius)
        self.max_gap = max(1, int(round(max_gap_seconds * sample_fps)))
        self.reid_gap = max(self.max_gap, int(round(reid_seconds * sample_fps)))
        self.tracks = {}
        self.next_id = 0
        self.last_sample = None
        self.stats = Counter()

    def update(self, detections, sample_index, camera_transform=None):
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        if self.last_sample is not None and sample_index <= self.last_sample:
            raise ValueError("Las muestras deben venir en orden creciente")
        if self.last_sample is not None and camera_transform is not None:
            a, b, tx, c, d, ty = camera_transform
            scale = math.sqrt(abs(a * d - b * c))
            for track in self.tracks.values():
                x, y = track["x"], track["y"]
                vx, vy = track["vx"], track["vy"]
                track["x"] = a * x + b * y + tx
                track["y"] = c * x + d * y + ty
                track["vx"] = a * vx + b * vy
                track["vy"] = c * vx + d * vy
                track["h"] *= scale
            self.stats["camera_compensated"] += 1
        self.last_sample = sample_index
        self.tracks = {tid: tr for tid, tr in self.tracks.items()
                       if sample_index - tr["last"] <= self.reid_gap}
        ids = list(self.tracks)
        costs = np.full((len(ids), len(detections)), 1e6, dtype=float)
        for row, tid in enumerate(ids):
            tr = self.tracks[tid]
            gap = sample_index - tr["last"]
            if gap < 1:
                continue
            reid = gap > self.max_gap
            x = tr["x"] + tr["vx"] * min(gap, 4)
            y = tr["y"] + tr["vy"] * min(gap, 4)
            radius = (self.match_radius * (1 + 0.06 * min(gap - 1, 8))
                      if not reid else min(21.0, 7.0 + 0.18 * gap))
            votes = tr["teams"]
            known = [(k, v) for k, v in votes.items() if k != "desconocido"]
            best_team, count = max(known, key=lambda kv: kv[1]) if known else (None, 0)
            stable_team = best_team if count >= 3 and count >= sum(votes.values()) * 0.6 else None
            for col, det in enumerate(detections):
                team = det.get("team_hint")
                if stable_team and team not in (None, "desconocido", stable_team):
                    continue
                height = float(det.get("h", 0))
                if min(height, tr["h"]) <= 0:
                    continue
                height_ratio = max(height, tr["h"]) / min(height, tr["h"])
                if height_ratio > (1.7 if reid else 2.0):
                    continue
                dist = math.hypot(det["cx"] - x, det["cy"] - y)
                if dist > radius:
                    continue
                hue_delta = _hue_distance(det.get("hsv"), tr["hsv"])
                if reid and (hue_delta is None or hue_delta > 17 or stable_team is None):
                    continue
                if hue_delta is not None and hue_delta > 35:
                    continue
                score = dist / radius + 0.13 * abs(math.log(height_ratio))
                if hue_delta is not None:
                    score += hue_delta * 0.008
                if reid:
                    score += 0.17
                costs[row, col] = score
        matches = {}
        if len(ids) and len(detections):
            row_ids, col_ids = linear_sum_assignment(costs)
            for row, col in zip(row_ids, col_ids):
                if costs[row, col] <= 1.08:
                    matches[int(col)] = ids[int(row)]
        output_ids = []
        for col, det in enumerate(detections):
            tid = matches.get(col)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "x": float(det["cx"]), "y": float(det["cy"]),
                    "h": float(det["h"]), "vx": 0.0, "vy": 0.0,
                    "last": sample_index, "hsv": det.get("hsv"),
                    "teams": Counter(),
                }
                self.stats["new_tracks"] += 1
            else:
                tr = self.tracks[tid]
                gap = max(1, sample_index - tr["last"])
                reid = gap > self.max_gap
                tr["vx"] = 0.55 * tr["vx"] + 0.45 * (det["cx"] - tr["x"]) / gap
                tr["vy"] = 0.55 * tr["vy"] + 0.45 * (det["cy"] - tr["y"]) / gap
                tr["x"], tr["y"] = float(det["cx"]), float(det["cy"])
                tr["h"] = float(det["h"])
                tr["last"] = sample_index
                if det.get("hsv") is not None:
                    if tr["hsv"] is None:
                        tr["hsv"] = det["hsv"]
                    elif (_hue_distance(det["hsv"], tr["hsv"]) is None
                          or _hue_distance(det["hsv"], tr["hsv"]) <= 20):
                        tr["hsv"] = det["hsv"]
                self.stats["reidentified" if reid else "continued"] += 1
            team = det.get("team_hint") or "desconocido"
            self.tracks[tid]["teams"][team] += 1
            output_ids.append(tid)
        return output_ids

    def predict_missing(self, sample_index, max_gap_seconds=1.0):
        """Short-horizon image-space hypotheses for temporarily hidden IDs.

        These are NOT detections or estimated field positions. The
        hypotheses are stored separately, with age and uncertainty, so
        downstream statistics cannot accidentally count invisible players.
        """
        if self.last_sample is None or sample_index != self.last_sample:
            raise ValueError("Predicciones solo para la última muestra procesada")
        horizon = min(float(max_gap_seconds), self.max_gap/self.sample_fps)
        result = []
        for tid, track in self.tracks.items():
            gap = sample_index - track["last"]
            if gap < 1 or gap/self.sample_fps > horizon:
                continue
            # The velocity is estimated over observed samples. After a
            # longer occlusion it is less trustworthy: decay to avoid
            # projecting a player across the whole pitch.
            damp = min(gap, 5) / gap if gap > 5 else 1.0
            x = track["x"] + track["vx"]*gap*damp
            y = track["y"] + track["vy"]*gap*damp
            if not (math.isfinite(x) and math.isfinite(y)
                    and 0 <= x <= 100 and 0 <= y <= 100):
                continue
            votes = [(name, count) for name,count in track["teams"].items()
                     if name != "desconocido"]
            known_team = max(votes, key=lambda item:item[1])[0] if votes else "desconocido"
            age = gap/self.sample_fps
            uncertainty = min(25.0, 1.5 + 6.0*age
                              + 0.4*math.hypot(track["vx"],track["vy"])*gap)
            result.append({
                "track_id": tid,
                "x": round(x, 2), "y": round(y, 2),
                "team": known_team, "age_s": round(age, 3),
                "uncertainty_u": round(uncertainty, 2),
                "source": "predicted",
            })
        return sorted(result, key=lambda p:p["track_id"])
