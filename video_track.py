"""video_track.py — Detección de vídeo v2 para Match_view (match.py).

Sustituye el enfoque de ``video_ai.py`` (muestrear una foto cada 5 s y hacer
``YOLO.track`` por lotes de imágenes fijas) por tracking continuo de verdad.

Diseño (respuestas del usuario, 2026-11-11):
* Vídeo del partido completo, pero los primeros minutos son calentamiento.
  El saque inicial se ajusta a mano en la UI (MM:SS) y todos los tiempos
  son relativos a ese inicio. ``suggest_kickoff()`` propone un inicio
  (primer momento con balón en el centro y formación 11+11), pero el manual
  manda siempre. El calentamiento no se analiza.
* Cámara fija pero QUE SIGUE AL BALÓN (principalmente VEO, a veces móvil).
  La posición en imagen NO es posición fija en el campo: hay pan/zoom y
  temblor. Por eso el tracking es RELATIVO: trayectoria del balón, proximidad,
  velocidad relativa a la escena, agrupación. La zona Z1-Z9 solo se infiere
  como aproximación en modo "fija_amplia"; en modo "seguimiento" (por defecto)
  la zona es "Sin especificar". No se asume vídeo panorámico de VEO.
* Equipos por color de camiseta, pero el usuario asigna MANUALMENTE qué color
  es cada equipo en la UI (``detect_shirt_colors()`` muestra la paleta
  dominante; los colores de ``match.py`` son solo el valor inicial).
* Jugadas: todas. Gol, tiro a puerta/fuera, pase, pase clave, centro, córner,
  falta, tarjeta, duelo, recuperación, pérdida, transición/contraataque,
  entrada al área, balón parado, distribución del portero y posesión por
  tramos. Prioridad a gol/tiro/córner.
* PC: Windows, Ryzen 7 serie 7000, CPU (sin GPU). YOLO11s por defecto,
  imgsz 640, muestreo ~10 fps, ``device="cpu"``. Un partido de 90 min tarda
  más que en tiempo real en CPU: probar primero con 10 minutos.
* Cortes solo como botones con resumen breve; guardado MANUAL a eventos.
  Nada se añade solo al CSV.

Tracking:
* ``YOLO.track(persist=True, tracker="bytetrack.yaml")`` fotograma a
  fotograma, con IDs persistentes en todo el vídeo, no por lotes.
* Personas = clase COCO 0. Balón = clase COCO 32, suavizado EMA e
  interpolación de huecos de hasta 1 s.
* Compensación del movimiento de cámara: se estima el desplazamiento común
  de los jugadores (mediana por muestra) y se resta al balón; las velocidades
  y distancias se expresan también en "alturas de jugador por segundo", que
  son robustas al zoom.
* Equipo por color: torso del bbox en HSV frente a los colores elegidos.
  Voto por mayoría por track ID. Si no se distingue: "desconocido".
* Posesión = jugador más cercano al balón, suavizada.
* Candidatos desde señales del tracking (pico de velocidad, parada brusca,
  agrupación, balón fuera de plano, balón parado, saque de portero,
  transición...), no de cambios de conteo. Groq clasifica cada candidato
  con 3 fotogramas (-2, 0, +2 s) + contexto estructurado. El modelo solo
  admite 3 imágenes por petición, por eso no se envían 5.
* Solo se corta clip si event != "uncertain" y confianza >= 0,65.

Modelo Groq por defecto: ``qwen/qwen3.8-27b``.
Verificado el 2026-11-11 en https://console.groq.com/docs/models y
https://console.groq.com/docs/deprecations: modelo multimodal (visión) de
Groq, sucesor de ``qwen/qwen3.6-27b``, en Preview (puede retirarse con poco
aviso; comprobar con ``GET https://api.groq.com/openai/v1/models`` con la
API key). NO usar ``meta-llama/llama-4-scout-17b-16e-instruct``: Groq lo
retiró el 17/07/2026. Los ``openai/gpt-oss-120b/20b`` son de producción pero
solo texto, no valen para fotogramas. El modelo es configurable.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constantes / defaults (todos configurables por parámetro)
# ---------------------------------------------------------------------------
DEFAULT_YOLO_MODEL = "yolo11s.pt"          # equilibrio CPU Ryzen 7; 11n más rápido, 11m más preciso
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"    # visión, verificado en docs de Groq 2026-11-11
DEFAULT_MIN_CONFIDENCE = 0.65
DEFAULT_SAMPLE_FPS = 10                     # ~10 fps (respuesta del usuario); bajar a 5 si va lento
DEFAULT_MAX_CANDIDATES = 15
DEFAULT_PRE_ROLL = 8                        # segundos antes del evento en el clip
DEFAULT_POST_ROLL = 7                       # segundos después del evento en el clip
MIN_GAP_SECONDS = 20                        # separación mínima entre candidatos
MERGE_WINDOW_SECONDS = 8                    # señales a menos de 8 s se fusionan en 1 candidato
BALL_GAP_MAX_SECONDS = 1.0                  # huecos de balón <= 1 s se interpolan
BALL_EMA_ALPHA = 0.4                        # suavizado del balón (0-1, mayor = más reactivo)
POSSESSION_RADIUS_HEIGHTS = 2.5             # alturas de jugador balón-jugador para posesión
COLOR_DISTANCE_MAX = 55.0                   # distancia HSV máxima para asignar equipo
COLOR_MARGIN_MIN = 12.0                     # margen mínimo entre local y visitante
SPEED_PEAK_BH = 5.0                         # alturas/s para "pico de velocidad"
SPEED_FAST_BH = 3.0                         # alturas/s para "balón rápido"
STOP_FROM_BH = 4.0                          # velocidad inicial para "parada brusca"
STOP_TO_BH = 0.8                            # velocidad final para "parada brusca"
CROWD_RADIUS_HEIGHTS = 3.5                  # radio (alturas) para "agrupación"
CROWD_MIN_PEOPLE = 5                        # personas en ese radio para "agrupación"
TRANSITION_ADVANCE_HEIGHTS = 8.0            # alturas de avance en 3 s para "transición"
SET_PIECE_STOP_SECONDS = 2.0                # balón parado para "balón parado / saque"
TRAMO_MINUTES = 15                          # tramos para la posesión por tramos

VALID_EVENTS = {
    "goal", "shot", "shot_on_target", "shot_off_target",
    "pass", "key_pass", "cross", "corner", "foul", "card",
    "duel", "recovery", "loss", "transition", "counterattack",
    "area_entry", "set_piece", "gk_distribution", "uncertain",
}

# Nombres en español para la UI (match.py los usa en los botones).
EVENT_NAMES_ES = {
    "goal": "Gol", "shot": "Tiro", "shot_on_target": "Tiro a puerta",
    "shot_off_target": "Tiro fuera", "pass": "Pase", "key_pass": "Pase clave",
    "cross": "Centro", "corner": "Córner", "foul": "Falta", "card": "Tarjeta",
    "duel": "Duelo", "recovery": "Recuperación", "loss": "Pérdida",
    "transition": "Transición", "counterattack": "Contraataque",
    "area_entry": "Entrada al área", "set_piece": "Balón parado",
    "gk_distribution": "Distribución portero", "uncertain": "Incierto",
}

ProgressCallback = Callable[[str, float], None]


class VideoProcessingError(RuntimeError):
    """Error al procesar el vídeo (archivo, ffmpeg, YOLO o Groq)."""


# ---------------------------------------------------------------------------
# Utilidades de sistema (ffmpeg / ffprobe)
# ---------------------------------------------------------------------------
def _tool_path(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise VideoProcessingError(
            f"No se encuentra {name}. Instala FFmpeg y asegúrate de que esté en PATH."
        )
    return executable


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip()[-2000:] or "Error desconocido."
        tool_name = Path(command[0]).name
        raise VideoProcessingError(f"Falló {tool_name}: {detail}")
    return result.stdout.strip()


def video_duration(video_path: Path | str) -> float:
    """Duración del vídeo en segundos (ffprobe)."""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"No existe el vídeo: {video_path}")
    output = _run([
        _tool_path("ffprobe"), "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ])
    try:
        duration = float(output)
    except ValueError as error:
        raise VideoProcessingError(
            "FFprobe no pudo determinar la duración del vídeo."
        ) from error
    if duration <= 0:
        raise VideoProcessingError("La duración del vídeo no es válida.")
    return duration


def export_clip(
    video_path: Path | str,
    start_seconds: float,
    end_seconds: float,
    output_dir: Path | None = None,
    label: str = "accion",
) -> Path:
    """Corta un clip con FFmpeg (H.264 + AAC). Nombre: MMSS_evento.mp4.

    Los tiempos son ABSOLUTOS del vídeo (no relativos al saque inicial).
    """
    video_path = Path(video_path).expanduser().resolve()
    duration = video_duration(video_path)
    start = max(0.0, min(float(start_seconds), duration))
    end = max(start, min(float(end_seconds), duration))
    if end <= start:
        raise VideoProcessingError("El intervalo del corte no contiene vídeo.")
    output_dir = output_dir or video_path.parent / f"{video_path.stem}_clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_") or "accion"
    output_path = output_dir / f"{int(start):06d}_{safe_label}.mp4"
    _run([
        _tool_path("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(video_path),
        "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(output_path),
    ])
    return output_path


def parse_mmss(texto: str) -> float:
    """'MM:SS' o 'MM' a segundos. Lanza ValueError si el formato no vale."""
    texto = (texto or "").strip()
    if ":" in texto:
        minutos, segundos = texto.split(":", 1)
        return float(minutos) * 60.0 + float(segundos)
    return float(texto) * 60.0


def format_mmss(seconds: float) -> str:
    """Segundos a 'MM:SS'."""
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


# ---------------------------------------------------------------------------
# Zonas Z1-Z9 (solo aproximación en cámara fija amplia; ver zona_para_candidato)
# ---------------------------------------------------------------------------
def zona_desde_xy(x: float, y: float) -> str:
    """Zona Z1-Z9 desde posición normalizada (0-100, 0-100).

    SOLO válido si la cámara es fija y amplia con todo el campo en plano
    (X = eje de ataque, 0 portería local, 100 visitante; Y = ancho, 100 banda
    izquierda, 0 banda derecha). Coherente con ZONA_A_XY de prueba.py:
    Z1 (17, 82), Z5 (50, 50), Z9 (83, 18).
    Con cámara que sigue al balón NO usar: la posición en imagen no es
    posición en el campo.
    """
    if x < 33.33:
        columna = "Def."
    elif x < 66.67:
        columna = "Medio"
    else:
        columna = "Ataque"
    if y >= 66.67:
        banda = "Izq"
    elif y >= 33.33:
        banda = "Centro"
    else:
        banda = "Der"
    numero = {"Def.": {"Izq": 1, "Centro": 2, "Der": 3},
              "Medio": {"Izq": 4, "Centro": 5, "Der": 6},
              "Ataque": {"Izq": 7, "Centro": 8, "Der": 9}}[columna][banda]
    return f"Z{numero} - {columna} {banda}"


def zona_para_candidato(ball_x: float, ball_y: float, camera_mode: str) -> str:
    """Zona de un candidato según el modo de cámara.

    - "fija_amplia": aproximación desde la posición del balón en el plano.
    - "seguimiento" (VEO/móvil, por defecto): "Sin especificar". La cámara
      sigue al balón, así que la posición en imagen no dice en qué zona del
      campo ocurrió la jugada. Mejor no inventarla que ponerla mal.
    """
    if camera_mode == "fija_amplia":
        return zona_desde_xy(ball_x, ball_y)
    return "Sin especificar"


# ---------------------------------------------------------------------------
# Colores de camiseta
# ---------------------------------------------------------------------------
def _hex_a_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise VideoProcessingError(f"Color no válido (se espera #RRGGBB): {hex_color}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_a_hsv_opencv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """RGB (0-255) a HSV en escala OpenCV: H 0-179, S 0-255, V 0-255."""
    import numpy as np
    import cv2
    pixel = np.uint8([[[b, g, r]]])  # OpenCV trabaja en BGR
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def _hsv_opencv_a_hex(h: float, s: float, v: float) -> str:
    """HSV OpenCV a #RRGGBB."""
    import numpy as np
    import cv2
    pixel = np.uint8([[[int(h), int(s), int(v)]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
    r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
    return f"#{r:02X}{g:02X}{b:02X}"


def _distancia_hsv(h1: float, s1: float, v1: float,
                   h2: float, s2: float, v2: float) -> float:
    """Distancia HSV con matiz circular. El matiz pesa más (es lo que distingue camisetas)."""
    dh = min(abs(h1 - h2), 180.0 - abs(h1 - h2))
    return math.sqrt((dh * 2.0) ** 2 + ((s1 - s2) * 0.5) ** 2 + ((v1 - v2) * 0.3) ** 2)


def _color_torso(frame, x0: int, y0: int, x1: int, y1: int):
    """Color mediano del torso (centro del bbox) en HSV OpenCV. None si no se puede."""
    import numpy as np
    import cv2
    h, w = frame.shape[:2]
    tx0 = max(0, int(x0 + 0.30 * (x1 - x0)))
    tx1 = min(w, int(x0 + 0.70 * (x1 - x0)))
    ty0 = max(0, int(y0 + 0.30 * (y1 - y0)))
    ty1 = min(h, int(y0 + 0.62 * (y1 - y0)))
    if tx1 - tx0 < 3 or ty1 - ty0 < 3:
        return None
    region = frame[ty0:ty1, tx0:tx1]
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mediana = np.median(hsv.reshape(-1, 3), axis=0)
    return float(mediana[0]), float(mediana[1]), float(mediana[2])


def _clasificar_color(hsv, ref_local, ref_visitante) -> str:
    """'local' / 'visitante' / 'desconocido' según distancia HSV a los colores de equipo."""
    if hsv is None:
        return "desconocido"
    d_local = _distancia_hsv(*hsv, *ref_local)
    d_visitante = _distancia_hsv(*hsv, *ref_visitante)
    mejor = min(d_local, d_visitante)
    if mejor > COLOR_DISTANCE_MAX:
        return "desconocido"
    if abs(d_local - d_visitante) < COLOR_MARGIN_MIN:
        return "desconocido"
    return "local" if d_local < d_visitante else "visitante"


def detect_shirt_colors(
    video_path: Path | str,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    imgsz: int = 640,
    max_frames: int = 20,
    progress: ProgressCallback | None = None,
) -> list[dict]:
    """Detecta los colores dominantes de camiseta en el vídeo.

    Muestrea fotogramas repartidos por todo el vídeo, detecta personas con
    YOLO y agrupa los colores de torso en 3 grupos (k-means simple con numpy).
    Devuelve [{"hex": "#RRGGBB", "share": 0.42, "count": 120}, ...] ordenados
    por frecuencia. El usuario asigna después qué color es local/visitante
    en la UI; esto solo muestra la paleta, no decide equipos.
    """
    import numpy as np
    import cv2
    from ultralytics import YOLO

    video_path = Path(video_path).expanduser().resolve()
    duration = video_duration(video_path)
    update = progress or (lambda _m, _v: None)
    try:
        detector = YOLO(yolo_model)
    except Exception as error:
        raise VideoProcessingError(f"No se pudo cargar YOLO ({yolo_model}): {error}") from error
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"No se pudo abrir el vídeo: {video_path}")
    muestras_hsv: list[tuple[float, float, float]] = []
    try:
        instantes = [duration * (i + 0.5) / max_frames for i in range(max_frames)]
        for n, t in enumerate(instantes):
            update(f"Detectando colores de camiseta ({n + 1}/{len(instantes)}).",
                   0.1 + 0.8 * n / max(1, len(instantes)))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            alto, ancho = frame.shape[:2]
            try:
                resultados = detector.predict(
                    frame, classes=[0], imgsz=imgsz, verbose=False, device="cpu")
            except Exception as error:
                raise VideoProcessingError(f"Falló YOLO al detectar colores: {error}") from error
            if not resultados or len(resultados) == 0:
                continue
            res = resultados[0]
            if res.boxes is None or res.boxes.xyxy is None:
                continue
            for caja in res.boxes.xyxy.cpu().tolist():
                x0, y0, x1, y1 = [float(v) for v in caja]
                hsv = _color_torso(frame, int(x0), int(y0), int(x1), int(y1))
                # Ignorar grises (césped/sombra): hace falta algo de saturación
                if hsv is not None and hsv[1] > 40:
                    muestras_hsv.append(hsv)
    finally:
        cap.release()
    if len(muestras_hsv) < 10:
        raise VideoProcessingError(
            "No se detectaron suficientes camisetas para hacer la paleta. "
            "Comprueba que el vídeo muestra jugadores."
        )
    # k-means simple (k=3) sobre HSV, con distancia de matiz circular aproximada
    # en el espacio (H*cos, H*sin, S, V) para que el matiz no se rompa en 0/179.
    datos = np.array(muestras_hsv, dtype=float)
    ang = np.deg2rad(datos[:, 0] * 2.0)  # H 0-179 -> 0-360
    feats = np.column_stack([
        np.cos(ang) * 2.0, np.sin(ang) * 2.0,
        datos[:, 1] / 128.0, datos[:, 2] / 128.0,
    ])
    rng = np.random.default_rng(42)
    centros = feats[rng.choice(len(feats), size=3, replace=False)]
    asignacion = np.zeros(len(feats), dtype=int)
    for _ in range(30):
        dists = ((feats[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
        nueva = dists.argmin(axis=1)
        if (nueva == asignacion).all():
            break
        asignacion = nueva
        for k in range(3):
            if (asignacion == k).any():
                centros[k] = feats[asignacion == k].mean(axis=0)
    paleta: list[dict] = []
    for k in range(3):
        idx = np.where(asignacion == k)[0]
        if len(idx) == 0:
            continue
        h_med = float(np.median(datos[idx, 0]))
        s_med = float(np.median(datos[idx, 1]))
        v_med = float(np.median(datos[idx, 2]))
        paleta.append({
            "hex": _hsv_opencv_a_hex(h_med, s_med, v_med),
            "share": round(len(idx) / len(datos), 2),
            "count": int(len(idx)),
        })
    paleta.sort(key=lambda c: c["share"], reverse=True)
    update("Paleta de colores lista.", 1.0)
    return paleta


# ---------------------------------------------------------------------------
# Sugerencia automática del saque inicial
# ---------------------------------------------------------------------------
def suggest_kickoff(
    video_path: Path | str,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    imgsz: int = 640,
    max_scan_minutes: float = 15,
    progress: ProgressCallback | None = None,
) -> float | None:
    """Propone el saque inicial (segundos de vídeo) o None si no lo encuentra.

    Heurística, no es fiable al 100%: primer momento (dentro de los primeros
    ``max_scan_minutes`` minutos) en que, durante 3 segundos seguidos, hay
    >= 15 personas visibles, el balón está cerca del centro del plano y los
    jugadores están repartidos por el ancho (formación 11+11). En calentamiento
    los jugadores están dispersos y el balón rara vez está quieto en el centro.
    El valor manual de la UI manda siempre sobre esta sugerencia.
    """
    import cv2
    from ultralytics import YOLO

    video_path = Path(video_path).expanduser().resolve()
    duration = video_duration(video_path)
    limite = min(duration, float(max_scan_minutes) * 60.0)
    update = progress or (lambda _m, _v: None)
    try:
        detector = YOLO(yolo_model)
    except Exception as error:
        raise VideoProcessingError(f"No se pudo cargar YOLO ({yolo_model}): {error}") from error
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"No se pudo abrir el vídeo: {video_path}")
    seguidos = 0
    try:
        t = 0.0
        while t <= limite:
            update(f"Buscando el saque inicial: {format_mmss(t)} de {format_mmss(limite)}.",
                   0.05 + 0.9 * t / max(1.0, limite))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += 1.0
                seguidos = 0
                continue
            alto, ancho = frame.shape[:2]
            try:
                resultados = detector.predict(
                    frame, classes=[0, 32], imgsz=imgsz, verbose=False, device="cpu")
            except Exception as error:
                raise VideoProcessingError(f"Falló YOLO al buscar el inicio: {error}") from error
            personas_x: list[float] = []
            balon_centro = False
            if resultados and len(resultados) > 0:
                res = resultados[0]
                if res.boxes is not None and res.boxes.cls is not None:
                    clases = res.boxes.cls.cpu().tolist()
                    cajas = res.boxes.xyxy.cpu().tolist()
                    for clase, caja in zip(clases, cajas):
                        x0, y0, x1, y1 = [float(v) for v in caja]
                        cx = ((x0 + x1) / 2.0) / ancho * 100.0
                        cy = ((y0 + y1) / 2.0) / alto * 100.0
                        if int(clase) == 0:
                            personas_x.append(cx)
                        elif int(clase) == 32:
                            if 40.0 <= cx <= 60.0 and 40.0 <= cy <= 60.0:
                                balon_centro = True
            import numpy as np
            repartidos = (len(personas_x) >= 2
                          and float(np.std(personas_x)) >= 12.0)
            if len(personas_x) >= 15 and balon_centro and repartidos:
                seguidos += 1
                if seguidos >= 3:
                    update("Saque inicial sugerido.", 1.0)
                    return max(0.0, t - 2.0)  # inicio del bloque de 3 s
            else:
                seguidos = 0
            t += 1.0
    finally:
        cap.release()
    update("No se encontró un saque inicial claro; ajústalo a mano.", 1.0)
    return None


# ---------------------------------------------------------------------------
# Tracking continuo
# ---------------------------------------------------------------------------
def _track_video(
    video_path: Path | str,
    yolo_model: str,
    sample_fps: int,
    imgsz: int,
    color_local_hex: str,
    color_visitante_hex: str,
    flip_x: bool,
    flip_y: bool,
    start_seconds: float,
    end_seconds: float,
    progress: ProgressCallback,
) -> tuple[list[dict], float, dict[int, str]]:
    """Pasa YOLO por el vídeo fotograma a fotograma con IDs persistentes.

    Analiza solo [start_seconds, end_seconds] (el calentamiento, antes del
    saque inicial, no se procesa). Devuelve (muestras, duración_del_partido,
    equipo_por_track_id). En cada muestra ``t`` es relativo al saque inicial
    y ``t_video`` absoluto. Las coordenadas x, y son DE PLANO (0-100 en la
    imagen), no de campo: con cámara que sigue al balón no hay correspondencia
    fija con el campo.
    """
    video_path = Path(video_path).expanduser().resolve()
    import numpy as np
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"No se pudo abrir el vídeo: {video_path}")
    fps_real = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_real <= 0:
        fps_real = 30.0
    total_duration = video_duration(video_path)
    end_seconds = min(float(end_seconds), total_duration)
    duracion_partido = max(0.0, end_seconds - start_seconds)

    paso = max(1, int(round(fps_real / max(1, sample_fps))))
    ref_local = _rgb_a_hsv_opencv(*_hex_a_rgb(color_local_hex))
    ref_visitante = _rgb_a_hsv_opencv(*_hex_a_rgb(color_visitante_hex))

    try:
        detector = YOLO(yolo_model)
    except Exception as error:
        cap.release()
        raise VideoProcessingError(f"No se pudo cargar YOLO ({yolo_model}): {error}") from error

    # Saltar hasta el saque inicial
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_seconds) * 1000.0)
    frame_idx = int(start_seconds * fps_real)
    muestras: list[dict] = []
    votos_color: dict[int, Counter] = defaultdict(Counter)
    procesados = 0
    total_a_procesar = max(1, int(duracion_partido * fps_real) // paso)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_video = frame_idx / fps_real
            if t_video > end_seconds:
                break
            t = t_video - start_seconds
            if frame_idx % paso == 0:
                procesados += 1
                if procesados % 100 == 0:
                    progress(f"YOLO siguiendo el vídeo: {format_mmss(t)} de partido.",
                             0.05 + 0.35 * min(1.0, procesados / total_a_procesar))
                alto, ancho = frame.shape[:2]
                try:
                    resultados = detector.track(
                        frame, persist=True, tracker="bytetrack.yaml",
                        classes=[0, 32], imgsz=imgsz, verbose=False,
                        device="cpu", half=False,
                    )
                except Exception as error:
                    raise VideoProcessingError(
                        f"Falló el seguimiento con YOLO: {error}"
                    ) from error
                personas: list[dict] = []
                balon: dict | None = None
                if resultados and len(resultados) > 0:
                    res = resultados[0]
                    if res.boxes is not None and res.boxes.cls is not None:
                        clases = res.boxes.cls.cpu().tolist()
                        ids = (res.boxes.id.cpu().tolist()
                               if res.boxes.id is not None
                               else [None] * len(clases))
                        cajas = res.boxes.xyxy.cpu().tolist()
                        confs = (res.boxes.conf.cpu().tolist()
                                 if res.boxes.conf is not None
                                 else [0.0] * len(clases))
                        for clase, track_id, caja, conf in zip(clases, ids, cajas, confs):
                            clase = int(clase)
                            x0, y0, x1, y1 = [float(v) for v in caja]
                            cx = ((x0 + x1) / 2.0) / ancho * 100.0
                            cy = ((y0 + y1) / 2.0) / alto * 100.0
                            if flip_x:
                                cx = 100.0 - cx
                            if flip_y:
                                cy = 100.0 - cy
                            if clase == 0:
                                personas.append({
                                    "track_id": int(track_id) if track_id is not None else -1,
                                    "x": cx, "y": cy,
                                    "w": (x1 - x0) / ancho * 100.0,
                                    "h": (y1 - y0) / alto * 100.0,
                                    "team": "desconocido",
                                })
                                if track_id is not None:
                                    hsv = _color_torso(frame, int(x0), int(y0), int(x1), int(y1))
                                    votos_color[int(track_id)][
                                        _clasificar_color(hsv, ref_local, ref_visitante)
                                    ] += 1
                            elif clase == 32:
                                if balon is None or conf > balon["conf"]:
                                    balon = {"x": cx, "y": cy, "conf": float(conf)}
                muestras.append({"t": t, "t_video": t_video,
                                 "people": personas, "ball": balon})
            frame_idx += 1
    finally:
        cap.release()

    # Equipo por track ID: voto por mayoría, con mínimo de votos y de proporción.
    equipo_por_track: dict[int, str] = {}
    for track_id, votos in votos_color.items():
        total = sum(votos.values())
        if total < 3:
            equipo_por_track[track_id] = "desconocido"
            continue
        equipo, n = votos.most_common(1)[0]
        equipo_por_track[track_id] = equipo if n / total >= 0.6 else "desconocido"
    for muestra in muestras:
        for persona in muestra["people"]:
            persona["team"] = equipo_por_track.get(persona["track_id"], "desconocido")
    return muestras, duracion_partido, equipo_por_track


def _suavizar_balon(muestras: list[dict]) -> list[dict]:
    """EMA + interpolación lineal de huecos de balón <= 1 s. Añade 'ball_s'.

    Un solo pase sobre las muestras (índices): un hueco <= 1 s son como
    mucho 10 muestras a 10 fps, así que el coste es lineal.
    """
    detecciones = [(i, m["t"], m["ball"]["x"], m["ball"]["y"])
                   for i, m in enumerate(muestras) if m["ball"] is not None]
    if not detecciones:
        for m in muestras:
            m["ball_s"] = None
        return muestras
    serie: dict[int, tuple[float, float]] = {}
    for k, (idx, t, x, y) in enumerate(detecciones):
        serie[idx] = (x, y)
        if k + 1 < len(detecciones):
            idx1, t1, x1, y1 = detecciones[k + 1]
            if 0 < t1 - t <= BALL_GAP_MAX_SECONDS:
                for j in range(idx + 1, idx1):
                    f = (muestras[j]["t"] - t) / (t1 - t)
                    serie[j] = (x + f * (x1 - x), y + f * (y1 - y))
    ema: tuple[float, float] | None = None
    ultimo_t: float | None = None
    for i, m in enumerate(muestras):
        t = m["t"]
        pos = serie.get(i)
        if pos is not None:
            ema = pos if ema is None else (
                BALL_EMA_ALPHA * pos[0] + (1 - BALL_EMA_ALPHA) * ema[0],
                BALL_EMA_ALPHA * pos[1] + (1 - BALL_EMA_ALPHA) * ema[1],
            )
            ultimo_t = t
            m["ball_s"] = {"x": ema[0], "y": ema[1]}
        else:
            if ultimo_t is not None and t - ultimo_t > BALL_GAP_MAX_SECONDS:
                ema = None
            m["ball_s"] = {"x": ema[0], "y": ema[1]} if ema is not None else None
    return muestras


def _compensar_camara(muestras: list[dict]) -> list[dict]:
    """Compensa pan/zoom: estima el movimiento común de los jugadores.

    Para cada par de muestras consecutivas, la mediana del desplazamiento de
    los track IDs comunes es el movimiento de cámara. Se acumula como offset
    y se resta al balón. Las velocidades se expresan en unidades de plano/s y
    en alturas de jugador/s (robusto al zoom: se divide por la altura mediana
    de los jugadores en ese instante). Añade por muestra: cam_off (x, y),
    median_height, ball_scene (x, y en alturas), ball_speed (u/s relativo),
    ball_speed_bh (alturas/s).
    """
    import numpy as np
    off_x, off_y = 0.0, 0.0
    altura_previa = 20.0
    for i, m in enumerate(muestras):
        if i > 0:
            anterior = muestras[i - 1]
            mapa_ant = {p["track_id"]: p for p in anterior["people"]
                        if p["track_id"] >= 0}
            dxs, dys = [], []
            for p in m["people"]:
                if p["track_id"] in mapa_ant:
                    q = mapa_ant[p["track_id"]]
                    dxs.append(p["x"] - q["x"])
                    dys.append(p["y"] - q["y"])
            if len(dxs) >= 2:
                off_x += float(np.median(dxs))
                off_y += float(np.median(dys))
        alturas = [p["h"] for p in m["people"] if p["h"] > 0]
        altura = float(np.median(alturas)) if alturas else altura_previa
        altura_previa = altura
        m["cam_off"] = (off_x, off_y)
        m["median_height"] = altura
        if m["ball_s"] is not None:
            m["ball_scene"] = ((m["ball_s"]["x"] - off_x) / max(1.0, altura),
                               (m["ball_s"]["y"] - off_y) / max(1.0, altura))
        else:
            m["ball_scene"] = None
    # Velocidad relativa (en escena, alturas/s y unidades/s)
    for i, m in enumerate(muestras):
        velocidad_bh = 0.0
        velocidad_u = 0.0
        if m["ball_scene"] is not None:
            j = i
            while j > 0 and muestras[j]["t"] > m["t"] - 0.5:
                j -= 1
            anterior = muestras[j]
            if anterior.get("ball_scene") is not None and anterior["t"] < m["t"]:
                dt = m["t"] - anterior["t"]
                if dt > 0:
                    velocidad_bh = math.hypot(
                        m["ball_scene"][0] - anterior["ball_scene"][0],
                        m["ball_scene"][1] - anterior["ball_scene"][1]) / dt
                    velocidad_u = math.hypot(
                        (m["ball_s"]["x"] - m["cam_off"][0])
                        - (anterior["ball_s"]["x"] - anterior["cam_off"][0]),
                        (m["ball_s"]["y"] - m["cam_off"][1])
                        - (anterior["ball_s"]["y"] - anterior["cam_off"][1])) / dt
        m["ball_speed"] = velocidad_u
        m["ball_speed_bh"] = velocidad_bh
    return muestras


def _anotar_posesion(muestras: list[dict], team_local: str, team_visitante: str) -> list[dict]:
    """Posesión por muestra: jugador más cercano al balón (en alturas).

    Se mantiene la última posesión cuando el balón va solo (pase en vuelo).
    """
    posesion_actual = "desconocido"
    for m in muestras:
        if m["ball_scene"] is not None and m["people"]:
            def dist_alturas(p):
                if p.get("team") is None:
                    return float("inf")
                # Posición de la persona también en escena (alturas)
                px = (p["x"] - m["cam_off"][0]) / max(1.0, m["median_height"])
                py = (p["y"] - m["cam_off"][1]) / max(1.0, m["median_height"])
                return math.hypot(px - m["ball_scene"][0], py - m["ball_scene"][1])
            cercana = min(m["people"], key=dist_alturas)
            if dist_alturas(cercana) <= POSSESSION_RADIUS_HEIGHTS:
                posesion_actual = cercana["team"]
        m["possession"] = posesion_actual
        m["possession_team_name"] = (
            team_local if posesion_actual == "local"
            else team_visitante if posesion_actual == "visitante"
            else "Desconocido"
        )
    return muestras


def posesion_por_tramos(muestras: list[dict], team_local: str, team_visitante: str,
                        tramo_minutes: int = TRAMO_MINUTES) -> list[dict]:
    """Posesión por tramos de 15' (o el tramo dado) a partir del tracking.

    Devuelve [{"tramo": "0-15'", "local_pct": 52, "visitante_pct": 41,
    "desconocido_pct": 7}, ...]. Es un resumen, no un evento.
    """
    if not muestras:
        return []
    duracion = muestras[-1]["t"]
    n_tramos = max(1, int(math.ceil(duracion / (tramo_minutes * 60.0))))
    out: list[dict] = []
    for k in range(n_tramos):
        inicio = k * tramo_minutes * 60.0
        fin = min(duracion, (k + 1) * tramo_minutes * 60.0)
        conteo = Counter(m["possession"] for m in muestras if inicio <= m["t"] < fin)
        total = sum(conteo.values()) or 1
        out.append({
            "tramo": f"{int(inicio // 60)}-{int(fin // 60)}'",
            "local_pct": round(100.0 * conteo.get("local", 0) / total),
            "visitante_pct": round(100.0 * conteo.get("visitante", 0) / total),
            "desconocido_pct": round(100.0 * conteo.get("desconocido", 0) / total),
        })
    return out


def _detectar_candidatos(muestras: list[dict], camera_mode: str) -> list[dict]:
    """Candidatos a evento desde señales relativas del tracking.

    Señales (robustas a pan/zoom porque usan velocidad relativa y distancias
    en alturas de jugador, no posiciones absolutas en el campo):
    pico de velocidad, parada brusca, agrupación junto al balón, balón fuera
    de plano, balón parado + agrupación (balón parado), saque del portero,
    cambio de posesión + avance rápido (transición), pase largo rápido.
    En modo fija_amplia se añaden: cruce de línea de gol y balón fuera del campo.
    """
    brutos: list[dict] = []
    n = len(muestras)
    if n < 2:
        return []
    # Las muestras son uniformes (1/sample_fps): los instantes cercanos se
    # buscan por índice, no recorriendo toda la lista (coste lineal total).
    dt = muestras[1]["t"] - muestras[0]["t"]
    fps_m = 1.0 / dt if dt > 0 else 10.0

    def cerca(i: int, offset_s: float) -> dict | None:
        """Muestra más cercana a muestras[i].t + offset_s (por índice)."""
        j = int(round(i + offset_s * fps_m))
        return muestras[j] if 0 <= j < n else None

    for i, m in enumerate(muestras):
        if m["ball_scene"] is None:
            continue
        t = m["t"]
        velocidad_bh = m["ball_speed_bh"]
        personas = m["people"]
        senales: list[str] = []
        score = 0.0
        # Pico de velocidad relativa (máximo local en ±1,5 s)
        if velocidad_bh >= SPEED_PEAK_BH:
            ventana = [mm["ball_speed_bh"] for mm in muestras[max(0, i - 15): i + 16]]
            if ventana and velocidad_bh >= max(ventana):
                senales.append(f"pico de velocidad del balón ({velocidad_bh:.1f} alturas/s)")
                score += 3.0 + min(2.0, velocidad_bh / 5.0)
        # Parada brusca: de rápido a casi parado en 0,6 s
        if velocidad_bh <= STOP_TO_BH:
            previas = [mm["ball_speed_bh"] for mm in muestras[max(0, i - 6): i]]
            if previas and max(previas) >= STOP_FROM_BH:
                senales.append("parada brusca del balón")
                score += 2.5
        # Agrupación junto al balón (en alturas)
        if m["ball_scene"] is not None:
            n_cerca = 0
            for p in personas:
                px = (p["x"] - m["cam_off"][0]) / max(1.0, m["median_height"])
                py = (p["y"] - m["cam_off"][1]) / max(1.0, m["median_height"])
                if math.hypot(px - m["ball_scene"][0], py - m["ball_scene"][1]) <= CROWD_RADIUS_HEIGHTS:
                    n_cerca += 1
            if n_cerca >= CROWD_MIN_PEOPLE:
                senales.append(f"agrupación de {n_cerca} jugadores junto al balón")
                score += 2.0
        # Balón fuera de plano: se pierde el balón con velocidad alta (gol, fuera, red)
        if m["ball_s"] is None:
            continue
        siguiente = cerca(i, 1.0)
        if siguiente is not None and siguiente.get("ball_scene") is None and velocidad_bh >= SPEED_FAST_BH:
            senales.append("balón fuera de plano con velocidad alta")
            score += 2.0
        # Balón parado + agrupación (balón parado) y saque del portero
        parada = cerca(i, -SET_PIECE_STOP_SECONDS)
        if (parada is not None and parada["ball_speed_bh"] <= STOP_TO_BH
                and velocidad_bh >= SPEED_FAST_BH):
            if len(personas) >= CROWD_MIN_PEOPLE:
                senales.append("balón parado con agrupación, reanudación brusca")
                score += 2.5
            else:
                senales.append("saque / distribución tras balón parado")
                score += 2.0
        # Pase largo rápido: balón rápido que recorre distancia en <1 s
        hace_1s = cerca(i, -1.0)
        if (hace_1s is not None and hace_1s.get("ball_scene") is not None
                and velocidad_bh >= SPEED_FAST_BH):
            distancia = math.hypot(m["ball_scene"][0] - hace_1s["ball_scene"][0],
                                   m["ball_scene"][1] - hace_1s["ball_scene"][1])
            if distancia >= 6.0:
                senales.append(f"balón rápido, recorre {distancia:.0f} alturas en 1 s")
                score += 1.5
        # Transición: cambio de posesión + avance rápido en escena
        hace_3s = cerca(i, -3.0)
        if (hace_3s is not None and hace_3s.get("ball_scene") is not None
                and hace_3s["possession"] != m["possession"]
                and m["possession"] in ("local", "visitante")):
            avance = math.hypot(m["ball_scene"][0] - hace_3s["ball_scene"][0],
                                m["ball_scene"][1] - hace_3s["ball_scene"][1])
            if avance >= TRANSITION_ADVANCE_HEIGHTS:
                senales.append("cambio de posesión + avance rápido")
                score += 3.0
        # Solo en cámara fija amplia: línea de gol y balón fuera del campo
        if camera_mode == "fija_amplia" and m["ball_s"] is not None:
            x, y = m["ball_s"]["x"], m["ball_s"]["y"]
            if (x >= 97.0 or x <= 3.0) and velocidad_bh >= SPEED_FAST_BH:
                senales.append("balón cruzando línea de gol")
                score += 5.0
            if x <= 1.0 or x >= 99.0 or y <= 1.0 or y >= 99.0:
                senales.append("balón fuera del campo")
                score += 2.0
        if senales:
            brutos.append({
                "seconds": t, "score": score, "signals": senales,
                "ball_x": m["ball_s"]["x"], "ball_y": m["ball_s"]["y"],
                "speed": m["ball_speed"], "speed_bh": velocidad_bh,
                "team": m["possession_team_name"],
                "possession": m["possession"],
                "people": len(personas),
                "ball": True,
            })
    if not brutos:
        return []
    # Fusionar señales a menos de 8 s: quedarse con la de mayor score
    brutos.sort(key=lambda c: c["seconds"])
    fusionados: list[dict] = [dict(brutos[0])]
    for cand in brutos[1:]:
        ultimo = fusionados[-1]
        if cand["seconds"] - ultimo["seconds"] <= MERGE_WINDOW_SECONDS:
            if cand["score"] > ultimo["score"]:
                senales = list(dict.fromkeys(ultimo["signals"] + cand["signals"]))
                fusionados[-1] = dict(cand, signals=senales)
            else:
                ultimo["signals"] = list(dict.fromkeys(ultimo["signals"] + cand["signals"]))
                ultimo["score"] = max(ultimo["score"], cand["score"])
        else:
            fusionados.append(dict(cand))
    return fusionados


def _dedup_y_limitar(candidatos: list[dict], max_candidates: int) -> list[dict]:
    """Ordena por score, impone 20 s entre candidatos, limita y reordena por tiempo."""
    ordenados = sorted(candidatos, key=lambda c: c["score"], reverse=True)
    elegidos: list[dict] = []
    for cand in ordenados:
        if all(abs(cand["seconds"] - e["seconds"]) >= MIN_GAP_SECONDS for e in elegidos):
            elegidos.append(cand)
        if len(elegidos) >= max(1, min(max_candidates, 20)):
            break
    return sorted(elegidos, key=lambda c: c["seconds"])


# ---------------------------------------------------------------------------
# Fotogramas para Groq
# ---------------------------------------------------------------------------
def _extraer_fotogramas(video_path: Path | str, timestamps_video: list[float]) -> list[tuple[float, str]]:
    """Extrae fotogramas en los instantes ABSOLUTOS dados. Devuelve [(t, base64_jpeg)]."""
    video_path = Path(video_path).expanduser().resolve()
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"No se pudo abrir el vídeo: {video_path}")
    out: list[tuple[float, str]] = []
    try:
        for t in timestamps_video:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            alto, ancho = frame.shape[:2]
            if ancho > 960:
                escala = 960.0 / ancho
                frame = cv2.resize(frame, (960, max(1, int(alto * escala))))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                continue
            out.append((t, base64.b64encode(buf.tobytes()).decode("ascii")))
    finally:
        cap.release()
    return out


def _clasificar_candidato(client, model: str, frames: list[tuple[float, str]],
                         contexto: dict) -> dict:
    """Clasifica un candidato con Groq (visión). Devuelve event/confidence/summary/team."""
    from groq import APIError
    # El modelo qwen/qwen3.8-27b solo admite 3 imágenes por petición (error 400
    # "Too many images provided" si se envían 5). Si llegan más, quedarse con
    # 3 repartidas: primera, central y última.
    if len(frames) > 3:
        frames = [frames[0], frames[len(frames) // 2], frames[-1]]
    eventos_txt = ", ".join(sorted(VALID_EVENTS - {"uncertain"}))
    prompt = (
        "Eres el analista de un partido de fútbol amateur (Regional Preferente, "
        "cámara que sigue al balón: VEO o móvil, plano que se mueve; la posición en "
        "la imagen NO es la posición fija en el campo). Clasifica con prudencia esta "
        "secuencia de fotogramas. No afirmes que hubo gol si no hay evidencia visual "
        "clara (balón dentro de la portería o celebración clara). Si no se distingue "
        "una acción concreta, usa uncertain. "
        "Devuelve SOLO JSON con las claves event, confidence, summary y team. "
        f"event debe ser uno de: {eventos_txt}, uncertain. "
        "Prioriza gol, tiro y córner cuando la evidencia lo permita, pero cubre "
        "también pase, pase clave, centro, falta, tarjeta, duelo, recuperación, "
        "pérdida, transición/contraataque, entrada al área, balón parado y "
        "distribución del portero. "
        "confidence es un número entre 0 y 1. "
        "summary es una frase MUY breve en español, máximo 12 palabras, que diga qué pasa. "
        "team es el nombre exacto del equipo protagonista, o \"Desconocido\"."
    )
    contexto_texto = (
        f"Contexto del tracking YOLO (medido, no lo inventes): minuto {contexto['minute']} "
        f"de partido, segundo {contexto['seconds']:.0f}, zona {contexto['zone']} "
        f"(solo aproximación; en cámara de seguimiento es 'Sin especificar'), "
        f"balón en el plano X={contexto['ball_x']:.0f}, Y={contexto['ball_y']:.0f} (0-100, "
        f"coordenadas de imagen, no de campo), velocidad relativa del balón "
        f"{contexto['speed_bh']:.1f} alturas/s, posesión: {contexto['team']}, "
        f"personas detectadas: {contexto['people']}, señales: {', '.join(contexto['signals'])}. "
        f"Equipos: local {contexto['team_local']}, visitante {contexto['team_visitante']}. "
        f"Trayectoria del balón en escena (t relativo, X, Y en alturas): {contexto['trayectoria']}."
    )
    content: list[dict] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": contexto_texto},
    ]
    for timestamp, b64 in frames:
        content.append({"type": "text", "text": f"Fotograma en {timestamp:.1f} segundos de partido."})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_completion_tokens=256,
            response_format={"type": "json_object"},
        )
    except APIError as error:
        raise VideoProcessingError(f"Error de la API de Groq: {error}") from error
    response_text = completion.choices[0].message.content
    try:
        result = json.loads(response_text)
        event = str(result["event"]).lower().strip()
        confidence = float(result["confidence"])
        summary = str(result["summary"]).strip()
        team = str(result.get("team", "Desconocido")).strip()
    except (TypeError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise VideoProcessingError(
            "Groq devolvió una clasificación con formato inválido."
        ) from error
    if event not in VALID_EVENTS or not 0.0 <= confidence <= 1.0:
        raise VideoProcessingError("Groq devolvió un tipo de jugada o confianza inválidos.")
    summary = " ".join(summary.split()[:12]) or "Jugada sin descripción clara."
    if team not in (contexto["team_local"], contexto["team_visitante"], "Desconocido"):
        team = contexto["team"] if contexto["team"] in (
            contexto["team_local"], contexto["team_visitante"]) else "Desconocido"
    return {"event": event, "confidence": confidence, "summary": summary, "team": team}


# ---------------------------------------------------------------------------
# Función principal (interna: devuelve resultados + resumen)
# ---------------------------------------------------------------------------
def _analizar_impl(
    video_path: Path | str,
    api_key: str,
    color_local: str,
    color_visitante: str,
    team_local: str,
    team_visitante: str,
    sample_fps: int,
    max_candidates: int,
    pre_roll: int,
    post_roll: int,
    min_confidence: float,
    yolo_model: str,
    groq_model: str,
    imgsz: int,
    flip_x: bool,
    flip_y: bool,
    max_minutes: float | None,
    kickoff_offset: float,
    camera_mode: str,
    progress: ProgressCallback | None,
) -> dict:
    video_path = Path(video_path).expanduser().resolve()
    if not 1 <= sample_fps <= 10:
        raise ValueError("El muestreo debe estar entre 1 y 10 fps.")
    if not 1 <= max_candidates <= 20:
        raise ValueError("El número de candidatos debe estar entre 1 y 20.")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("La confianza mínima debe estar entre 0 y 1.")
    if kickoff_offset < 0:
        raise ValueError("El saque inicial no puede ser negativo.")
    if camera_mode not in ("seguimiento", "fija_amplia"):
        raise ValueError("camera_mode debe ser 'seguimiento' o 'fija_amplia'.")
    if not api_key:
        raise VideoProcessingError(
            "No está configurada la variable de entorno GROQ_API_KEY. "
            "En PowerShell: $env:GROQ_API_KEY = \"tu_clave\" y abre Streamlit "
            "desde esa misma terminal."
        )
    update = progress or (lambda _message, _value: None)
    duracion_video = video_duration(video_path)
    if kickoff_offset >= duracion_video:
        raise VideoProcessingError(
            "El saque inicial está después del final del vídeo. Revísalo."
        )
    update("Cargando YOLO y siguiendo el vídeo fotograma a fotograma.", 0.02)
    try:
        from groq import Groq
    except ImportError as error:
        raise VideoProcessingError(
            "Falta el paquete groq. Instala con: py -m pip install groq"
        ) from error
    try:
        import ultralytics  # noqa: F401
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as error:
        raise VideoProcessingError(
            f"Falta {error.name or 'una dependencia'}. Instala con: "
            "py -m pip install ultralytics opencv-python-headless"
        ) from error

    fin = duracion_video
    if max_minutes:
        fin = min(fin, kickoff_offset + float(max_minutes) * 60.0)
    muestras, duracion_partido, _equipos = _track_video(
        video_path, yolo_model, sample_fps, imgsz,
        color_local, color_visitante, flip_x, flip_y,
        kickoff_offset, fin, update,
    )
    update("Compensando el movimiento de cámara y suavizando el balón.", 0.45)
    muestras = _suavizar_balon(muestras)
    muestras = _compensar_camara(muestras)
    muestras = _anotar_posesion(muestras, team_local, team_visitante)
    posesion_tramos = posesion_por_tramos(muestras, team_local, team_visitante)
    update("Buscando candidatos a evento en las señales del tracking.", 0.55)
    candidatos = _detectar_candidatos(muestras, camera_mode)
    candidatos = _dedup_y_limitar(candidatos, max_candidates)
    if not candidatos:
        update("YOLO no encontró señales claras de jugada en el vídeo.", 1.0)
        return {
            "results": [], "possession_by_period": posesion_tramos,
            "kickoff_offset_s": kickoff_offset,
            "match_duration_s": duracion_partido,
            "video_duration_s": duracion_video,
            "samples_analyzed": len(muestras),
            "team_colors": {"local": color_local, "visitante": color_visitante},
        }

    client = Groq(api_key=api_key)
    output_dir = video_path.parent / f"{video_path.stem}_clips"
    resultados: list[dict] = []
    for n_cand, cand in enumerate(candidatos, start=1):
        t = float(cand["seconds"])  # relativo al saque inicial
        t_video = t + kickoff_offset
        update(f"Groq clasifica candidato {n_cand} de {len(candidatos)}.",
               0.55 + 0.35 * n_cand / max(1, len(candidatos)))
        tray = []
        for m in muestras:
            if abs(m["t"] - t) <= 2.05 and m.get("ball_scene") is not None:
                tray.append(f"{m['t'] - t:+.1f}s:({m['ball_scene'][0]:.1f},{m['ball_scene'][1]:.1f})")
        frames = _extraer_fotogramas(
            video_path, [t_video - 2, t_video, t_video + 2])
        if len(frames) < 2:
            clasificacion = {
                "event": "uncertain", "confidence": 0.0,
                "summary": "No se pudieron extraer fotogramas.",
                "team": "Desconocido",
            }
        else:
            contexto = {
                "minute": int(t // 60), "seconds": t,
                "zone": zona_para_candidato(cand["ball_x"], cand["ball_y"], camera_mode),
                "ball_x": cand["ball_x"], "ball_y": cand["ball_y"],
                "speed_bh": cand["speed_bh"], "team": cand["team"],
                "people": cand["people"], "signals": cand["signals"],
                "team_local": team_local, "team_visitante": team_visitante,
                "trayectoria": "; ".join(tray) if tray else "sin trayectoria",
            }
            clasificacion = _clasificar_candidato(client, groq_model, frames, contexto)
        clip_path: str | None = None
        if (clasificacion["event"] != "uncertain"
                and clasificacion["confidence"] >= min_confidence):
            clip_start = max(0.0, t_video - pre_roll)
            clip_end = min(duracion_video, t_video + post_roll)
            label = f"{int(t // 60):02d}{int(t % 60):02d}_{clasificacion['event']}"
            clip_path = str(export_clip(video_path, clip_start, clip_end,
                                        output_dir, label))
        minute = int(t // 60)
        resultados.append({
            "seconds": round(t, 1),
            "minute": minute,
            "event": clasificacion["event"],
            "confidence": round(float(clasificacion["confidence"]), 2),
            "summary": clasificacion["summary"],
            "team": clasificacion["team"],
            "ball_x": round(float(cand["ball_x"]), 1),
            "ball_y": round(float(cand["ball_y"]), 1),
            "zone": zona_para_candidato(cand["ball_x"], cand["ball_y"], camera_mode),
            "clip_path": clip_path,
            "yolo_people": int(cand["people"]),
            "yolo_ball": 1 if cand["ball"] else 0,
            "reason": "; ".join(cand["signals"]),
            "signals": list(cand["signals"]),
        })
    update("Análisis y generación de cortes completados.", 1.0)
    return {
        "results": resultados,
        "possession_by_period": posesion_tramos,
        "kickoff_offset_s": kickoff_offset,
        "match_duration_s": duracion_partido,
        "video_duration_s": duracion_video,
        "samples_analyzed": len(muestras),
        "team_colors": {"local": color_local, "visitante": color_visitante},
    }


def analyze_video(
    video_path: Path | str,
    api_key: str,
    color_local: str = "#1E88E5",
    color_visitante: str = "#E53935",
    team_local: str = "Local",
    team_visitante: str = "Visitante",
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    pre_roll: int = DEFAULT_PRE_ROLL,
    post_roll: int = DEFAULT_POST_ROLL,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    groq_model: str = DEFAULT_GROQ_MODEL,
    imgsz: int = 640,
    flip_x: bool = False,
    flip_y: bool = False,
    max_minutes: float | None = None,
    kickoff_offset: float = 0.0,
    camera_mode: str = "seguimiento",
    progress: ProgressCallback | None = None,
) -> list[dict]:
    """Analiza un vídeo y devuelve la lista de cortes clasificados.

    Cada resultado: seconds (relativo al saque inicial), minute, event,
    confidence, summary, team, ball_x, ball_y (coordenadas DE PLANO, no de
    campo), zone, clip_path (str o None), yolo_people, yolo_ball, reason,
    signals. ``clip_path`` solo existe si event != "uncertain" y
    confidence >= min_confidence; el resto son candidatos descartados.
    Para el resumen completo (posesión por tramos, duración, etc.) usar
    ``analyze_video_full``.
    """
    return _analizar_impl(
        video_path, api_key, color_local, color_visitante, team_local,
        team_visitante, sample_fps, max_candidates, pre_roll, post_roll,
        min_confidence, yolo_model, groq_model, imgsz, flip_x, flip_y,
        max_minutes, kickoff_offset, camera_mode, progress,
    )["results"]


def analyze_video_full(
    video_path: Path | str,
    api_key: str,
    color_local: str = "#1E88E5",
    color_visitante: str = "#E53935",
    team_local: str = "Local",
    team_visitante: str = "Visitante",
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    pre_roll: int = DEFAULT_PRE_ROLL,
    post_roll: int = DEFAULT_POST_ROLL,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    groq_model: str = DEFAULT_GROQ_MODEL,
    imgsz: int = 640,
    flip_x: bool = False,
    flip_y: bool = False,
    max_minutes: float | None = None,
    kickoff_offset: float = 0.0,
    camera_mode: str = "seguimiento",
    progress: ProgressCallback | None = None,
) -> dict:
    """Como analyze_video, pero devuelve el informe completo.

    {"results": [...], "possession_by_period": [{"tramo", "local_pct",
    "visitante_pct", "desconocido_pct"}, ...], "kickoff_offset_s": ...,
    "match_duration_s": ..., "video_duration_s": ...,
    "samples_analyzed": ..., "team_colors": {"local": ..., "visitante": ...}}
    """
    return _analizar_impl(
        video_path, api_key, color_local, color_visitante, team_local,
        team_visitante, sample_fps, max_candidates, pre_roll, post_roll,
        min_confidence, yolo_model, groq_model, imgsz, flip_x, flip_y,
        max_minutes, kickoff_offset, camera_mode, progress,
    )


# ---------------------------------------------------------------------------
# Prueba rápida por terminal (sin Streamlit):
#   python video_track.py "C:\\Videos\\partido.mp4" --kickoff 04:32
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analizar un vídeo de partido con YOLO + Groq.")
    parser.add_argument("video", type=Path, help="Ruta local del vídeo")
    parser.add_argument("--color-local", default="#1E88E5")
    parser.add_argument("--color-visitante", default="#E53935")
    parser.add_argument("--team-local", default="Local")
    parser.add_argument("--team-visitante", default="Visitante")
    parser.add_argument("--sample-fps", type=int, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--kickoff", default="00:00",
                        help="Saque inicial en MM:SS (el calentamiento no se analiza)")
    parser.add_argument("--camera-mode", default="seguimiento",
                        choices=["seguimiento", "fija_amplia"])
    parser.add_argument("--suggest-kickoff", action="store_true",
                        help="Solo sugerir el saque inicial y salir")
    args = parser.parse_args()
    if args.suggest_kickoff:
        sugerido = suggest_kickoff(args.video, args.yolo_model)
        print("Saque inicial sugerido:",
              format_mmss(sugerido) if sugerido is not None else "no encontrado")
        raise SystemExit(0)
    key = os.environ.get("GROQ_API_KEY", "")
    informe = analyze_video_full(
        args.video, key,
        color_local=args.color_local, color_visitante=args.color_visitante,
        team_local=args.team_local, team_visitante=args.team_visitante,
        sample_fps=args.sample_fps, max_candidates=args.max_candidates,
        min_confidence=args.min_confidence, yolo_model=args.yolo_model,
        groq_model=args.groq_model, max_minutes=args.max_minutes,
        kickoff_offset=parse_mmss(args.kickoff), camera_mode=args.camera_mode,
        progress=lambda msg, frac: print(f"[{frac:.0%}] {msg}"),
    )
    print(json.dumps(informe, ensure_ascii=False, indent=2))
