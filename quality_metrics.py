"""Auditoría ligera de cobertura y consistencia del tracking.

Los porcentajes describen cobertura de los datos, NO precisión del modelo
ni exactitud de identificación. Sin etiquetas humanas no puede medirse IDF1.
"""
from __future__ import annotations

from collections import Counter
import math


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def assess_tracking_quality(samples, max_people_per_team=16):
    """Examina muestras completas sin conservar otro duplicado en memoria.

    Devuelve indicadores reproducibles y advertencias para la interfaz y
    los informes. Acepta listas vacías y tracking sin homografía.
    """
    total = 0
    n_ball_yolo = n_ball_pred = n_ball_interpolated = n_pitch = 0
    uncertainty_sum = uncertainty_n = age_sum = 0
    hidden_predictions = 0
    people_sum = total_people = n_unknown = 0
    duplicate_frames = nonmonotonic = invalid_coords = 0
    pitch_quality = Counter()
    field_filter = Counter()
    prev_t = None
    team_overflow_frames = 0
    observed_ids = set()
    for sample in samples:
        total += 1
        for key in ("detected", "accepted", "rejected", "unverified"):
            field_filter[key] += int((sample.get("field_filter") or {}).get(key, 0))
        field_filter["mask_frames"] += bool((sample.get("field_filter") or {}).get("mask_active"))
        field_filter["homography_frames"] += bool((sample.get("field_filter") or {}).get("homography_active"))
        t = sample.get("t", sample.get("t_video"))
        if _finite(t):
            t = float(t)
            if prev_t is not None and t <= prev_t:
                nonmonotonic += 1
            prev_t = t
        else:
            nonmonotonic += 1

        ball = sample.get("ball")
        ball_smoothed = sample.get("ball_s") or {}
        accepted = sample.get("ball_yolo_accepted", isinstance(ball, dict))
        source = ball_smoothed.get("fuente")
        if accepted and isinstance(ball, dict):
            n_ball_yolo += 1
        elif source == "kalman":
            n_ball_pred += 1
        elif source == "interpolado":
            n_ball_interpolated += 1
        if source in ("kalman", "interpolado"):
            if _finite(ball_smoothed.get("sigma")):
                uncertainty_sum += max(0.0, float(ball_smoothed["sigma"]))
                uncertainty_n += 1
            if _finite(ball_smoothed.get("missing_s")):
                age_sum += max(0.0, float(ball_smoothed["missing_s"]))
        hidden_predictions += len(sample.get("predicted_people") or [])

        if sample.get("pitch_ok"):
            n_pitch += 1
            pitch_quality[sample.get("pitch_calidad") or "desconocida"] += 1

        people = sample.get("people") or []
        people_sum += len(people)
        ids = set()
        teams = Counter()
        for person in people:
            total_people += 1
            team = person.get("team") or "desconocido"
            teams[team] += 1
            if team == "desconocido":
                n_unknown += 1
            tid = person.get("track_id")
            if isinstance(tid, int) and tid >= 0:
                if tid in ids:
                    duplicate_frames += 1
                ids.add(tid)
                observed_ids.add(tid)
            x, y = person.get("pitch_x"), person.get("pitch_y")
            if (x is not None or y is not None) and (
                    not _finite(x) or not _finite(y)
                    or not -30 <= float(x) <= 135
                    or not -30 <= float(y) <= 98):
                invalid_coords += 1
        for side in ("local", "visitante"):
            n = teams[side] + teams["gk_" + side]
            if n > max_people_per_team:
                team_overflow_frames += 1
                break

    def pct(n, denominator):
        return round(100.0 * n / denominator, 1) if denominator else 0.0

    warnings = []
    if not total:
        warnings.append("No hay muestras de seguimiento.")
    else:
        if pct(n_pitch, total) < 40:
            warnings.append("Cobertura de homografía baja: las distancias individuales pueden ser incompletas.")
        if pct(n_ball_yolo, total) < 20:
            warnings.append("Pocas detecciones directas del balón; revisar umbrales y modelo.")
        if pct(n_unknown, total_people) > 35:
            warnings.append("Muchos jugadores sin equipo asignado; revisar colores de camisetas.")
    if total and field_filter["unverified"]:
        warnings.append("Algunas personas se omitieron porque no se pudo verificar el campo.")
    if total and field_filter["accepted"] == 0 and field_filter["detected"] > 0:
        warnings.append("Ninguna persona superó el filtro del terreno: revisar la máscara y homografía.")
    if duplicate_frames:
        warnings.append("Se han detectado IDs duplicados en una misma muestra.")
    if nonmonotonic:
        warnings.append("Existen marcas temporales duplicadas, ausentes o desordenadas.")
    if invalid_coords:
        warnings.append("Hay coordenadas de campo fuera del margen aceptado o no numéricas.")
    if team_overflow_frames:
        warnings.append("Demasiadas detecciones de un mismo equipo en algunos fotogramas.")

    return {
        "version": 1,
        "total_samples": total,
        "unique_track_ids": len(observed_ids),
        "avg_people_visible": round(people_sum / total, 1) if total else 0.0,
        "ball_yolo_pct": pct(n_ball_yolo, total),
        "ball_predicted_pct": pct(n_ball_pred, total),
        "ball_interpolated_pct": pct(n_ball_interpolated, total),
        "ball_prediction_uncertainty_u_mean": (
            round(uncertainty_sum / uncertainty_n, 2) if uncertainty_n else None),
        "ball_prediction_missing_s_mean": (
            round(age_sum / (n_ball_pred + n_ball_interpolated), 2)
            if n_ball_pred + n_ball_interpolated else None),
        "hidden_player_hypotheses": hidden_predictions,
        "pitch_coverage_pct": pct(n_pitch, total),
        "pitch_quality_frames": dict(pitch_quality),
        "field_filter": dict(field_filter),
        "field_mask_coverage_pct": pct(field_filter["mask_frames"], total),
        "field_filter_rejection_pct": pct(field_filter["rejected"], field_filter["detected"]),
        "unknown_team_pct": pct(n_unknown, total_people),
        "duplicate_track_id_occurrences": duplicate_frames,
        "nonmonotonic_samples": nonmonotonic,
        "invalid_pitch_coordinates": invalid_coords,
        "team_overflow_frames": team_overflow_frames,
        "warnings": warnings,
        "note": "Cobertura no es precisión. La continuidad real de identidad requiere vídeo etiquetado.",
    }
