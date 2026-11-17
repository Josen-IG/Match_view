"""Geometría táctica descriptiva de jugadores y balón visibles.

No se infieren formaciones ni presión real con una cámara parcial.
Las coordenadas dependen de la homografía.
"""
from __future__ import annotations
import math
import statistics
from collections import Counter, defaultdict


def _median(values):
    return round(float(statistics.median(values)), 1) if values else None


def _hull_area(points):
    points = sorted(set(points))
    if len(points) < 3:
        return 0.0

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return abs(sum(hull[i][0] * hull[(i + 1) % len(hull)][1] -
                   hull[(i + 1) % len(hull)][0] * hull[i][1]
                   for i in range(len(hull)))) / 2.0


def _shape(points):
    xs, ys = zip(*points)
    nearest = [min(math.dist(p, q) for j, q in enumerate(points) if i != j)
               for i, p in enumerate(points)]
    return {
        "visible_players": len(points),
        "width_m": max(ys) - min(ys),
        "depth_m": max(xs) - min(xs),
        "area_m2": _hull_area(points),
        "centroid_x": sum(xs) / len(xs),
        "centroid_y": sum(ys) / len(ys),
        "nearest_teammate_m": statistics.median(nearest),
    }


def compute_tactical_metrics(samples, period_seconds=300.0, sample_every=5,
                             min_players=4, pitch_length=105.0, pitch_width=68.0):
    """Resumen por periodos; equipo visible, presión cercana y ocupación.

    Anchura corresponde al eje Y, profundidad al X. La línea defensiva
    requiere portero visible y al menos cuatro jugadores de campo.
    """
    periods = defaultdict(lambda: {
        "shape": {"local": defaultdict(list), "visitante": defaultdict(list)},
        "coverage": Counter(), "quality": Counter(),
        "zones": {"local": Counter(), "visitante": Counter()},
        "pressure": {"local": [], "visitante": []},
        "near_ball_advantage": [], "transitions": Counter(),
    })
    previous_possession = None
    previous_time = None
    for i, sample in enumerate(samples):
        t = sample.get("t")
        if t is None or not math.isfinite(float(t)):
            continue
        t = float(t)
        period = int(max(0.0, t) // period_seconds)
        possession = sample.get("possession_suave")
        if possession in ("local", "visitante"):
            if (previous_possession in ("local", "visitante")
                    and possession != previous_possession and previous_time is not None
                    and t - previous_time <= 3.0):
                periods[period]["transitions"][possession] += 1
            previous_possession, previous_time = possession, t
        if i % sample_every:
            continue
        row = periods[period]
        row["coverage"]["analyzed"] += 1
        if sample.get("pitch_ok") is False:
            continue
        positions = {"local": [], "visitante": []}
        goalkeepers = {}
        for player in sample.get("people") or []:
            try:
                x, y = float(player["pitch_x"]), float(player["pitch_y"])
            except (ValueError, TypeError, KeyError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)
                    and 0 <= x <= pitch_length and 0 <= y <= pitch_width):
                continue
            team = player.get("team")
            if team in ("local", "visitante"):
                positions[team].append((x, y))
                cell = (min(5, int(x / pitch_length * 6)),
                        min(3, int(y / pitch_width * 4)))
                row["zones"][team][f"{cell[0]}-{cell[1]}"] += 1
            elif team in ("gk_local", "gk_visitante"):
                goalkeepers["local" if team == "gk_local" else "visitante"] = x
        count_valid = 0
        for team in ("local", "visitante"):
            if len(positions[team]) < min_players:
                continue
            count_valid += 1
            shape = _shape(positions[team])
            for key, value in shape.items():
                row["shape"][team][key].append(value)
            if team in goalkeepers:
                xs = sorted(p[0] for p in positions[team])
                if goalkeepers[team] < pitch_length / 2:
                    line = xs[max(0, int(.2 * (len(xs) - 1)))]
                else:
                    line = pitch_length - xs[min(len(xs) - 1, int(.8 * (len(xs) - 1)))]
                row["shape"][team]["defensive_line_m"].append(line)
        row["coverage"]["some_team"] += int(count_valid > 0)
        row["coverage"]["both_teams"] += int(count_valid == 2)
        row["quality"][sample.get("pitch_calidad") or "desconocida"] += 1
        ball = sample.get("ball") or {}
        try:
            bx, by = float(ball["pitch_x"]), float(ball["pitch_y"])
        except (TypeError, KeyError, ValueError):
            continue
        if not (0 <= bx <= pitch_length and 0 <= by <= pitch_width):
            continue
        if possession in ("local", "visitante"):
            defense = "visitante" if possession == "local" else "local"
            if len(positions[defense]) >= min_players:
                row["pressure"][defense].append(
                    sum(math.dist((bx, by), p) <= 10.0 for p in positions[defense]))
        if all(len(positions[team]) >= min_players for team in positions):
            nearby = {team: sum(math.dist((bx, by), p) <= 12.0 for p in pts)
                      for team, pts in positions.items()}
            row["near_ball_advantage"].append(nearby["local"] - nearby["visitante"])

    result = []
    for index, row in sorted(periods.items()):
        shape_summary = {}
        for team in ("local", "visitante"):
            metrics = row["shape"][team]
            shape_summary[team] = {key: _median(values) for key, values in metrics.items()}
            shape_summary[team]["usable_frames"] = len(metrics["width_m"])
            shape_summary[team]["zones_6x4"] = dict(row["zones"][team])
            shape_summary[team]["defenders_near_ball_10m"] = _median(row["pressure"][team])
        result.append({
            "period_start_s": index * period_seconds,
            "period_end_s": (index + 1) * period_seconds,
            "coverage": dict(row["coverage"]),
            "pitch_quality": dict(row["quality"]),
            "possession_recoveries_proxy": dict(row["transitions"]),
            "local_players_minus_visitors_near_ball_median": _median(row["near_ball_advantage"]),
            "teams": shape_summary,
        })
    return {
        "version": 1, "period_seconds": period_seconds,
        "min_players_for_shape": min_players,
        "method": "Geometria parcial de jugadores visibles con homografia; no reconstruccion de todos los jugadores.",
        "caveats": [
            "Las anchuras/profundidades son del grupo visible, no necesariamente del equipo completo.",
            "La linea defensiva solo se estima si el portero es visible.",
            "Presion = numero de defensores visibles a <=10 m del balon; no intensidad real.",
            "Recuperaciones = cambios en posesion estimada, no eventos verificados.",
            "No se infieren formaciones ni superioridades cuando faltan jugadores visibles.",
        ],
        "periods": result,
    }
