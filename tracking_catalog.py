"""Lectura ligera de cabeceras del tracking y compatibilidad con vídeo.

Evita cargar cientos de MB de muestras solo para listar la biblioteca.
El JSON de Match_view guarda la clave samples al final.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SAMPLE_START = re.compile(r',\s*"samples"\s*:')


def read_tracking_header(path, max_header_bytes=8_000_000):
    """Devuelve metadatos previos a samples sin materializar las muestras."""
    path = Path(path)
    buffer = ""
    with path.open("r", encoding="utf-8") as f:
        while len(buffer) < max_header_bytes:
            chunk = f.read(32_768)
            if not chunk:
                break
            buffer += chunk
            match = _SAMPLE_START.search(buffer)
            if match:
                return json.loads(buffer[:match.start()] + "}")
    raise ValueError("Formato no soportado o cabecera de tracking demasiado grande")


def inspect_tracking(path, video_filename=None):
    """Clasifica datos actuales, heredados e incompatibles sin cargar samples."""
    path = Path(path)
    data = read_tracking_header(path)
    filename = data.get("video_filename") or ""
    if video_filename and filename and filename.casefold() != Path(video_filename).name.casefold():
        state = "otro_video"
    elif data.get("pipeline_version", 0) >= 2 and all(
            key in data for key in ("player_stats", "tactical_analysis", "quality")):
        state = "actual"
    else:
        state = "antiguo"
    return {"state": state, "video_filename": filename,
            "pipeline_version": data.get("pipeline_version", 0),
            "team_names": data.get("teams") or {},
            "match_duration_s": data.get("match_duration_s") or 0,
            "quality": data.get("quality") or {},
            "path": str(path)}


def deduplicate_library(items):
    """Agrupa copias del mismo tracking sin colapsar partidos distintos."""
    chosen = {}
    for item in items:
        key = (item.get("video_filename") or item.get("archivo"),
               item.get("partido"), item.get("duracion"), item.get("muestras"))
        current = chosen.get(key)
        if current is None or item.get("mtime", 0) > current.get("mtime", 0):
            chosen[key] = item
    return sorted(chosen.values(), key=lambda item: item.get("mtime", 0), reverse=True)
