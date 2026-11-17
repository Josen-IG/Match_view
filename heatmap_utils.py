"""Histogramas de tracking por acumulación: memoria O(número de celdas).

Acepta JSON antiguo y nuevo sin construir listas gigantes de puntos.
"""
from __future__ import annotations
import math
import numpy as np


def build_heatmap_histogram(tracking, subject, start_s, end_s):
    """Devuelve (histograma, coords_campo, nº puntos)."""
    samples = tracking.get("samples") or []
    code = tracking.get("team_code") or {}
    teams = {
        "local": {code.get("local", 0), code.get("gk_local", 2)},
        "visitante": {code.get("visitante", 1), code.get("gk_visitante", 3)},
    }
    codes = teams.get(subject)
    track_id = int(subject.partition(":")[2]) if subject.startswith("jugador:") else None
    use_pitch = any(("pp" in m or "bp" in m) for m in samples)
    nx, ny = (35, 23) if use_pitch else (50, 28)
    xrange, yrange = (105.0, 68.0) if use_pitch else (100.0, 100.0)
    histogram = np.zeros((nx, ny), dtype=np.uint32)
    count = 0
    for m in samples:
        t = m.get("t")
        if t is None or not start_s <= float(t) < end_s:
            continue
        if subject == "balon":
            ball = m.get("bp" if use_pitch else "b")
            entries = [ball] if ball else []
        else:
            entries = m.get("pp" if use_pitch else "p") or []
        for entry in entries:
            if not entry or len(entry) < 2:
                continue
            if subject != "balon":
                team_index = 2 if use_pitch else 4
                id_index = 3 if use_pitch else 5
                if track_id is not None:
                    if len(entry) <= id_index or entry[id_index] != track_id:
                        continue
                elif codes is None or len(entry) <= team_index or entry[team_index] not in codes:
                    continue
            try:
                x, y = float(entry[0]), float(entry[1])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)
                    and 0 <= x <= xrange and 0 <= y <= yrange):
                continue
            ix = min(nx - 1, int(x * nx / xrange))
            iy = min(ny - 1, int(y * ny / yrange))
            histogram[ix, iy] += 1
            count += 1
    return histogram, use_pitch, count
