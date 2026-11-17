"""Métricas individuales orientativas derivadas del tracking y la homografía.

No equiparar track_id a dorsal. No inferir pases, tiros o recuperaciones
sin eventos etiquetados. Las medidas en metros son aproximadas.
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict


def _num(value):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def compute_player_metrics(samples, pitch_length=105.0, pitch_width=68.0,
                           bin_seconds=0.5, max_speed_ms=12.0):
    """Métricas por ID, sin extrapolar posiciones no observadas.

    Agrega posiciones en ventanas de 0,5 s para reducir jitter. No suma
    saltos >12 m/s ni huecos >1,25 s. Muestra cobertura y calidad.
    """
    players = defaultdict(lambda: {
        "teams": Counter(), "samples": 0, "field_samples": 0,
        "quality": Counter(), "field_bins": defaultdict(list),
        "last_visible": None, "visible_s": 0.0,
        "last_near": None, "near_ball_s": 0.0,
    })
    for sample in samples:
        t = _num(sample.get("t", sample.get("t_video")))
        if t is None:
            continue
        ball = sample.get("ball")
        bx = _num(ball.get("x")) if isinstance(ball, dict) else None
        by = _num(ball.get("y")) if isinstance(ball, dict) else None
        seen = set()
        for person in sample.get("people") or []:
            tid = person.get("track_id")
            if not isinstance(tid, int) or tid < 0 or tid in seen:
                continue
            seen.add(tid)
            p = players[tid]
            p["samples"] += 1
            p["teams"][person.get("team") or "desconocido"] += 1
            prev = p["last_visible"]
            if prev is not None and 0 < t - prev <= 0.5:
                p["visible_s"] += t - prev
            p["last_visible"] = t

            px, py = _num(person.get("pitch_x")), _num(person.get("pitch_y"))
            if (sample.get("pitch_ok") is not False and px is not None
                    and py is not None and 0 <= px <= pitch_length
                    and 0 <= py <= pitch_width):
                p["field_samples"] += 1
                q = sample.get("pitch_calidad") or "desconocida"
                p["quality"][q] += 1
                p["field_bins"][int(math.floor(t / bin_seconds))].append((t, px, py))

            x, y = _num(person.get("x")), _num(person.get("y"))
            h = _num(person.get("h"))
            if None not in (bx, by, x, y, h) and h > 0:
                near = math.hypot(x - bx, y - by) <= 1.8 * h
                if near and p["last_near"] is not None and 0 < t - p["last_near"] <= 0.5:
                    p["near_ball_s"] += t - p["last_near"]
                p["last_near"] = t if near else None
            else:
                p["last_near"] = None

    result = []
    for tid, p in sorted(players.items()):
        known = {k: v for k, v in p["teams"].items() if k != "desconocido"}
        team = max(known, key=known.get) if known else "desconocido"
        bins = []
        for key, values in sorted(p["field_bins"].items()):
            n = len(values)
            bins.append((sum(v[0] for v in values) / n,
                         sum(v[1] for v in values) / n,
                         sum(v[2] for v in values) / n))
        zone_counts = Counter()
        for _t, x, y in bins:
            ix = min(2, int(x / pitch_length * 3))
            iy = min(2, int(y / pitch_width * 3))
            zone_counts[f"{ix}-{iy}"] += 1
        distance = 0.0
        measured_s = 0.0
        top_speed = 0.0
        accelerations = decelerations = sprints = 0
        previous_speed = None
        acc_active = dec_active = False
        sprint_duration = 0.0
        valid_steps = 0
        for i in range(1, len(bins)):
            t0, x0, y0 = bins[i - 1]
            t1, x1, y1 = bins[i]
            dt = t1 - t0
            if not (0.2 <= dt <= 1.25):
                if sprint_duration >= 1.0:
                    sprints += 1
                sprint_duration = 0.0
                previous_speed = None
                acc_active = dec_active = False
                continue
            speed = math.hypot(x1 - x0, y1 - y0) / dt
            if speed > max_speed_ms:
                if sprint_duration >= 1.0:
                    sprints += 1
                sprint_duration = 0.0
                previous_speed = None
                acc_active = dec_active = False
                continue
            valid_steps += 1
            distance += speed * dt
            measured_s += dt
            top_speed = max(top_speed, speed)
            if previous_speed is not None:
                acceleration = (speed - previous_speed) / dt
                is_acc = acceleration >= 2.5 and speed >= 3.0
                is_dec = acceleration <= -2.5 and previous_speed >= 3.0
                if is_acc and not acc_active:
                    accelerations += 1
                if is_dec and not dec_active:
                    decelerations += 1
                acc_active, dec_active = is_acc, is_dec
            if speed >= 7.0:
                sprint_duration += dt
            else:
                if sprint_duration >= 1.0:
                    sprints += 1
                sprint_duration = 0.0
            previous_speed = speed
        if sprint_duration >= 1.0:
            sprints += 1
        has_motion = valid_steps > 0
        result.append({
            "track_id": tid,
            "team": team,
            "visible_samples": p["samples"],
            "visible_s": round(p["visible_s"], 1),
            "field_samples": p["field_samples"],
            "field_coverage_pct": round(100 * p["field_samples"] / p["samples"], 1),
            "pitch_quality": dict(p["quality"]),
            "field_measured_s": round(measured_s, 1),
            "distance_m": round(distance, 1) if has_motion else None,
            "avg_speed_kmh": round(distance / measured_s * 3.6, 1) if measured_s else None,
            "max_speed_kmh": round(top_speed * 3.6, 1) if has_motion else None,
            "accelerations": accelerations if has_motion else None,
            "decelerations": decelerations if has_motion else None,
            "sprints": sprints if has_motion else None,
            "mean_position_m": [round(sum(b[1] for b in bins) / len(bins), 1),
                                round(sum(b[2] for b in bins) / len(bins), 1)] if bins else None,
            "zone_3x3_s": {key: round(count * bin_seconds, 1)
                           for key, count in sorted(zone_counts.items())},
            "ball_proximity_s": round(p["near_ball_s"], 1),
        })
    return {
        "version": 1,
        "method": "Posiciones de homografia agregadas cada 0,5 s; estimaciones, no GPS.",
        "caveats": [
            "track_id identifica un track, no un dorsal ni un jugador verificado.",
            "Tiempo visible no equivale a minutos disputados.",
            "Las distancias y velocidades dependen de la calidad de la homografia.",
            "Cercania al balon no es una estadistica de toques o pases.",
            "No se infieren eventos (pases, tiros, recuperaciones) sin etiquetado.",
        ],
        "players": result,
    }
