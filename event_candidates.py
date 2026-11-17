"""Detección conservadora de jugadas *candidatas* desde tracking de balón.

No son goles/tiros/córneres confirmados. Sin visión de portería, jugadores y
resultado, las trayectorias del balón solo permiten generar hipótesis para
revisión humana. Usa EXCLUSIVAMENTE coordenadas calibradas de campo (bp) y
detecciones directas YOLO (bf=yolo), nunca imagen->campo aproximada ni Kalman.
"""
from __future__ import annotations

import math
from collections import defaultdict


def _valid_point(sample):
    try:
        t = float(sample["t"])
        xy = sample["bp"]
        x, y = float(xy[0]), float(xy[1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not all(math.isfinite(n) for n in (t, x, y)):
        return None
    if sample.get("bf") != "yolo":
        return None
    # Tolerar 2 m más allá de línea de gol para sugerir posible cruce.
    if not (-2 <= x <= 107 and -2 <= y <= 70):
        return None
    return (t, x, y)


def _candidate(kind, t, score, reason, kickoff_s, duration_s):
    start = max(float(kickoff_s), t - 8)
    end = min(float(kickoff_s) + max(duration_s, 0), t + 6)
    if end <= start:
        end = t + 1
    labels = {
        "shot": "Posible tiro",
        "corner": "Posible córner",
        "goal": "Posible gol",
    }
    return {
        "type": kind,
        "label": labels[kind],
        "t": round(t, 2),  # Tiempo de VÍDEO, no tiempo relativo al saque.
        "start_s": round(start, 2),
        "end_s": round(end, 2),
        "confidence": round(max(0.1, min(0.85, score)), 2),
        "status": "suggested",
        "reason": reason,
        "source": "ball_pitch_heuristic",
    }


def suggest_key_moments(samples, kickoff_s=0.0, max_events=120):
    """Genera candidatos de eventos por geometría, sin inventar eventos.

    Se evitan posiciones sin homografía o predichas. Debe existir un
    movimiento coherente hacia portería o una reanudación junto a esquina.
    Si solo hay balón en píxeles, devuelve [].
    """
    if not isinstance(samples, list) or len(samples) < 4:
        return []

    kickoff_s = max(0.0, float(kickoff_s or 0.0))
    points = [_valid_point(sample) for sample in samples]
    points = [p for p in points if p is not None and p[0] >= kickoff_s]
    points.sort(key=lambda p: p[0])
    if len(points) < 4:
        return []

    duration_s = max(points[-1][0] - kickoff_s, 1.0)
    raw = []

    # Candidatos a tiro: el balón progresa >=7 m hacia una línea de gol
    # en <=1,7 s, con velocidad plausible, cerca del carril de la portería.
    # El detector NO sabe si alguien ha disparado; este es un proxy.
    for i in range(3, len(points)):
        p = points[i]
        current_t, current_x, current_y = p
        j = i - 1
        while j >= 0 and current_t - points[j][0] < 0.9:
            j -= 1
        if j < 0 or current_t - points[j][0] > 1.7:
            continue
        old_t, old_x, old_y = points[j]
        dt = current_t - old_t
        if dt < 0.7:
            continue
        dx, dy = current_x - old_x, current_y - old_y
        distance = math.hypot(dx, dy)
        speed = distance / dt
        if not (7.0 <= abs(dx) and 11.0 <= speed <= 42.0):
            continue
        # Para un tiro hacia x=105 o x=0, el punto final debe estar ya
        # en el tercio ofensivo y moverse mayormente en X.
        towards_right = dx > 0 and 73 <= current_x <= 107
        towards_left = dx < 0 and -2 <= current_x <= 32
        if not (towards_right or towards_left):
            continue
        if abs(dx) < distance * 0.68:
            continue
        # Pista de dirección sin asumir orientación de cada equipo:
        # proyectar y cuando llegue a línea de gol; sin calibración de
        # profundidad no inferimos si entrará entre postes.
        goal_x = 105 if towards_right else 0
        target_y = old_y + (goal_x - old_x) * dy / dx
        if not (20 <= target_y <= 48):
            continue
        if not (10 <= old_y <= 58 and 10 <= current_y <= 58):
            continue
        score = .4 + min(.18, max(0, speed - 11) / 90)
        raw.append(_candidate(
            "shot", current_t, score,
            "Desplazamiento rápido del balón hacia una portería; tiro sin confirmar.",
            kickoff_s, duration_s,
        ))

    # Córner: balón estacionario junto a esquina >=0,5 s y salida clara
    # hacia interior del campo. No distinguir automáticamente saque de
    # esquina de otro balón parado sin imágenes del ejecutor.
    corners = [(0.0, 0.0), (0.0, 68.0), (105.0, 0.0), (105.0, 68.0)]
    i = 0
    while i < len(points) - 4:
        t0, x0, y0 = points[i]
        corner = min(corners, key=lambda xy: math.hypot(x0 - xy[0], y0 - xy[1]))
        if math.hypot(x0 - corner[0], y0 - corner[1]) > 2.2:
            i += 1
            continue
        stable_until = i
        while (stable_until + 1 < len(points)
               and points[stable_until + 1][0] - points[stable_until][0] <= .5
               and points[stable_until + 1][0] - t0 <= 3.0
               and math.dist(points[stable_until + 1][1:], corner) <= 2.5):
            stable_until += 1
        dwell = points[stable_until][0] - t0
        if dwell >= .5:
            for k in range(stable_until + 1, min(stable_until + 18, len(points))):
                tk, xk, yk = points[k]
                if tk - points[stable_until][0] > 2.5:
                    break
                if math.dist((xk, yk), corner) >= 5.0:
                    raw.append(_candidate(
                        "corner", points[stable_until][0], .54,
                        "Balón detenido cerca de la esquina y posterior reanudación.",
                        kickoff_s, duration_s,
                    ))
                    break
        i = max(i + 1, stable_until + 1)

    # Gol: NO inferir un gol por entrar en el área ni por un tiro.
    # Únicamente sugerir un posible cruce cuando hay observaciones
    # medidas en ambos lados de la línea, con balón cerca del centro de
    # la portería. No se asume gol confirmado (puede no ser válido).
    for prev, curr in zip(points, points[1:]):
        t0, x0, y0 = prev
        t1, x1, y1 = curr
        dt = t1 - t0
        if not (0 < dt <= .35 and 3.0 <= math.dist((x0, y0), (x1, y1))/dt <= 42):
            continue
        crossing = ((x0 >= 0 and x1 < 0 and x0 <= 3) or
                    (x0 <= 105 and x1 > 105 and x0 >= 102))
        if crossing and 30.34 <= y0 <= 37.66 and 30.34 <= y1 <= 37.66:
            raw.append(_candidate(
                "goal", t1, .5,
                "Posible cruce de la línea de gol: requiere revisión del vídeo.",
                kickoff_s, duration_s,
            ))

    # Evitar decenas de marcadores del mismo movimiento; priorizar gol
    # respecto a tiro en el mismo instante, conservando tipos distintos
    # cuando son realmente eventos diferentes.
    raw.sort(key=lambda c: (-{"goal": 3, "shot": 2, "corner": 1}[c["type"]],
                            -c["confidence"], c["t"]))
    selected = []
    for item in raw:
        if any(abs(item["t"] - e["t"]) < 7.0 and item["type"] == e["type"]
               for e in selected):
            continue
        if item["type"] == "shot" and any(
                e["type"] == "goal" and abs(item["t"] - e["t"]) <= 2
                for e in selected):
            continue
        selected.append(item)
        if len(selected) >= max_events:
            break
    selected.sort(key=lambda item: item["t"])
    for i, candidate in enumerate(selected, 1):
        candidate["id"] = f"{candidate['type']}:{int(round(candidate['t']*10))}:{i}"
    return selected
