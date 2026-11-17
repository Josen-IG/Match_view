"""Análisis del partido completo por tramos, resistente a reinicios.

El VM puede reiniciarse en mitad de un análisis largo (4+ h) y los procesos
en fondo mueren. Por eso el partido se procesa en tramos de 15 min: cada
tramo guarda sus muestras en un pickle al terminar (escritura atómica) y
volver a lanzar este script salta los tramos ya hechos. La fase final
(clasificación, cortes, estadísticas, informe) es rápida y se hace de una
vez con todas las muestras juntas.

Uso:
  venv/bin/python chunked_full.py           # procesa tramos pendientes + mezcla
  venv/bin/python chunked_full.py --merge-only   # solo mezcla (tramos ya hechos)

Los parámetros (saque inicial, colores, equipos) se leen de params.json.
Tramos: desde el saque inicial, 15 min cada uno, hasta 105 min de partido.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
VIDEO = BASE / "video.mp4"
PARAMS = BASE / "params.json"
# Directorio de tramos. Por defecto "chunks" (corrida original, colores
# Rojo/Azul con porteros por defecto). Para re-trackear con los colores
# correctos (Cirbonero azul / Milagres rojo, porteros negro/verde) sin
# pisar los tramos viejos: --chunks-dir chunks_v2
CHUNKS = BASE / "chunks"
VIDEO_TRACK = Path("/home/hatch/workspace/match_view_video_v2")
sys.path.insert(0, str(VIDEO_TRACK))

TRAMO_S = 15 * 60
TOTAL_S = 105 * 60  # 105 min de partido desde el saque inicial


def _ajustes(p):
    # En esta VM lenta: 5 fps a 1280/conf 0,10 (~3 h el partido). En el
    # Ryzen 7 del usuario, 10 fps con los mismos imgsz/conf.
    return (int(p.get("sample_fps", 5)), int(p.get("imgsz", 960)),
            float(p.get("conf", 0.10)))


def cargar_params():
    return json.loads(PARAMS.read_text(encoding="utf-8"))


def procesar_tramo(idx, inicio_partido_s, fin_partido_s, p):
    import video_track
    destino = CHUNKS / f"chunk_{idx:02d}.pkl"
    if destino.exists():
        print(f"tramo {idx} ya hecho ({destino.name}), se salta", flush=True)
        return True
    inicio_video = p["kickoff_sugerido"] + inicio_partido_s
    fin_video = p["kickoff_sugerido"] + fin_partido_s
    sample_fps, imgsz, conf = _ajustes(p)
    print(f"tramo {idx}: partido {inicio_partido_s // 60}-{fin_partido_s // 60}' "
          f"(vídeo {inicio_video:.0f}-{fin_video:.0f} s, "
          f"{sample_fps} fps, imgsz {imgsz}, conf {conf})", flush=True)
    t0 = time.time()
    muestras, dur, _ = video_track._track_video(
        VIDEO, str(VIDEO_TRACK / video_track.DEFAULT_YOLO_MODEL),
        sample_fps, imgsz,
        p["color_local"], p["color_visitante"],
        p["color_gk_local"], p["color_gk_visitante"], p["color_arbitro"],
        False, False, inicio_video, fin_video, None, conf=conf)
    # t relativo al saque inicial (no al inicio del tramo).
    for m in muestras:
        m["t"] = round(m["t"] + inicio_partido_s, 2)
    tmp = destino.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(muestras, f)
    os.replace(tmp, destino)
    print(f"tramo {idx} listo: {len(muestras)} muestras en "
          f"{time.time() - t0:.0f} s -> {destino.name}", flush=True)
    return True


def mezclar():
    import video_track
    from video_track import (
        _anotar_campo_desde_video, _anotar_porteros, _anotar_posesion,
        _calcular_estadisticas, _compensar_camara,
        _corregir_eje_x_campo, _exportar_informe, _filtro_kalman_balon,
        _suavizar_balon, format_mmss, posesion_por_tramos, video_duration,
    )
    p = cargar_params()
    archivos = sorted(CHUNKS.glob("chunk_*.pkl"))
    if not archivos:
        print("no hay tramos procesados", flush=True)
        return None
    muestras = []
    for a in archivos:
        with open(a, "rb") as f:
            muestras.extend(pickle.load(f))
    muestras.sort(key=lambda m: m["t"])
    print(f"mezcla: {len(muestras)} muestras de {len(archivos)} tramos", flush=True)
    duracion_partido = muestras[-1]["t"] if muestras else 0.0
    duracion_video = video_duration(VIDEO)
    team_local = p.get("team_local", "Local")
    team_visitante = p.get("team_visitante", "Visitante")
    kickoff = float(p["kickoff_sugerido"])
    # Mapa del campo: se calcula sobre las muestras ya trackeadas (no hace
    # falta re-trackear) con el modelo de keypoints. Si los chunks son de
    # una corrida con colores viejos, los porteros no vendran etiquetados
    # como porteros y la correccion del eje x no podra aplicarse.
    pitch_model = str(BASE / "pitch_keypoints.pt")
    if not Path(pitch_model).exists():
        pitch_model = None  # auto-descarga en video_track
    muestras, n_con_campo = _anotar_campo_desde_video(
        VIDEO, muestras, pitch_model=pitch_model)
    print(f"campo: {n_con_campo}/{len(muestras)} muestras con mapa", flush=True)
    muestras = _suavizar_balon(muestras)
    muestras = _filtro_kalman_balon(muestras)
    muestras = _compensar_camara(muestras)
    muestras = _anotar_posesion(muestras, team_local, team_visitante)
    muestras = _anotar_porteros(muestras)
    n_corr = _corregir_eje_x_campo(muestras)
    print(f"correccion eje x: {n_corr} muestras", flush=True)
    posesion_tramos = posesion_por_tramos(muestras, team_local, team_visitante)
    # Solo tracking (decisión del usuario, 2026-11-13): sin candidatos,
    # sin clasificación de eventos ni cortes automáticos. Pases, secuencias
    # y resultados quedan vacíos, como en _analizar_impl.
    pases = []
    secuencias = []
    resultados = []
    output_dir = VIDEO.parent / f"{VIDEO.stem}_clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    estadisticas = _calcular_estadisticas(
        muestras, resultados, pases, secuencias,
        team_local, team_visitante, duracion_partido)
    informe = {
        "results": resultados, "possession_by_period": posesion_tramos,
        "kickoff_offset_s": kickoff, "match_duration_s": duracion_partido,
        "video_duration_s": duracion_video, "samples_analyzed": len(muestras),
        "team_local": team_local, "team_visitante": team_visitante,
        "team_colors": {"local": p["color_local"], "visitante": p["color_visitante"],
                        "gk_local": p["color_gk_local"],
                        "gk_visitante": p["color_gk_visitante"],
                        "arbitro": p["color_arbitro"]},
    }
    rutas = _exportar_informe(informe, estadisticas, pases, VIDEO, output_dir)
    # Tracking JSON para la pantalla de análisis manual (como en
    # _analizar_impl): si falla no se tira la mezcla entera.
    try:
        tracking_path = video_track.export_tracking_json(
            muestras, informe, estadisticas,
            output_dir / f"{VIDEO.stem}_tracking.json",
            formation_local=p.get("formation_local", ""),
            formation_visitante=p.get("formation_visitante", ""),
            camera_mode="seguimiento",
            video_filename=VIDEO.name,
        )
    except Exception as err:  # noqa: BLE001 - no romper la mezcla
        print(f"aviso: no se pudo exportar el tracking JSON: {err}", flush=True)
        tracking_path = None
    informe = {**informe, "estadisticas": estadisticas, "pases": pases,
               "secuencias": secuencias, **rutas,
               "tracking_json": tracking_path}
    (BASE / "informe_full.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    print("INFORME LISTO:", rutas["informe_md"], flush=True)
    print("TRACKING JSON:", tracking_path, flush=True)
    print("posesion:", posesion_tramos, flush=True)
    return informe


if __name__ == "__main__":
    CHUNKS.mkdir(parents=True, exist_ok=True)
    p = cargar_params()
    solo_mezcla = "--merge-only" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--chunks-dir="):
            CHUNKS = BASE / a.split("=", 1)[1]
            CHUNKS.mkdir(parents=True, exist_ok=True)
    if not solo_mezcla:
        n_tramos = TOTAL_S // TRAMO_S
        for idx in range(n_tramos):
            inicio = idx * TRAMO_S
            fin = min(TOTAL_S, (idx + 1) * TRAMO_S)
            procesar_tramo(idx, inicio, fin, p)
    mezclar()
