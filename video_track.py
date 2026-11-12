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
  es cada equipo en la UI (los colores de ``match.py`` son solo el valor
  inicial). También se asignan el color de cada portero y el del árbitro:
  el portero cuenta como su equipo y su posición marca dónde está su
  portería (en seguimiento, qué portería se ataca); el árbitro no cuenta
  para posesión ni agrupación.
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
* ``YOLO.predict()`` fotograma a fotograma + asociación propia entre
  muestras (vecino más cercano por posición y altura). ByteTrack se quitó:
  con cámara que sigue al balón el pan/zoom entre muestras rompía sus
  tracks (de 18 personas a 0 en pocos fotogramas) y se tragaba el balón
  (0% de muestras con balón frente a ~20% con predict()).
* Personas = clase COCO 0. Balón = clase COCO 32, suavizado EMA e
  interpolación de huecos de hasta 1 s.
* Compensación del movimiento de cámara: se estima el desplazamiento común
  de los jugadores (mediana por muestra) y se resta al balón; las velocidades
  y distancias se expresan también en "alturas de jugador por segundo", que
  son robustas al zoom.
* Equipo por color: torso del bbox en HSV frente a los colores elegidos
  (mediana de la mitad más saturada, para rayas y contraluz). Gana el
  equipo conocido con más votos por track ID. Si no se distingue:
  "desconocido".
* Posesión = jugador más cercano al balón, suavizada.
* Candidatos desde señales del tracking (pico de velocidad, parada brusca,
  agrupación, balón fuera de plano, balón parado, saque de portero,
  transición, cambio de posesión solo, balón parado largo, entrada al
  área...), no de cambios de conteo. Cada candidato se clasifica SOLO con
  el tracking: ventana de ±3 s con velocidad, balón perdido, balón parado,
  cambio de posesión, agrupación, avance y (en fija_amplia) cruce de la
  línea de gol y entrada al área. Sin Groq, sin API key, sin límites de
  tokens. La confianza es heurística y nunca pasa de 0,80.
* Pases por trayectoria: cambio de portador (el balón sale de un jugador,
  vuela sin portador y llega a un compañero; si llega a un rival es un
  pase fallido). Base de las estadísticas de pases y secuencias, y el
  clasificador los usa para etiquetar pases que no son rápidos.
* Estadísticas e informe: posesión, eventos por equipo y por tramo, pases
  y secuencias, tiros, duelos, recuperaciones, transiciones, balón parado,
  territorio (proxy por cercanía a cada portero) y balón (distancia,
  velocidad, cobertura). Se escriben ``<video>_informe.md`` + 2 CSV junto
  a los clips, y la pantalla de Vídeo los muestra con descarga.
* Solo se corta clip si event != "uncertain" y confianza >= 0,65.

Límites honestos de clasificar solo con tracking (sin ver la jugada):
en modo seguimiento (VEO/móvil, el habitual) no hay posición en el campo,
así que el gol, el córner, el centro y la entrada al área NO se pueden
distinguir con seguridad: un gol sale como tiro, un córner como balón
parado y un centro como pase. En modo fija_amplia (todo el campo en plano)
sí se distinguen, porque la posición del balón es posición en el campo.
La posesión por tramos se calcula del tracking, como antes.
"""

from __future__ import annotations

import csv
import json
import math
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
DEFAULT_MIN_CONFIDENCE = 0.65
DEFAULT_SAMPLE_FPS = 10                     # ~10 fps (respuesta del usuario); bajar a 5 si va lento
DEFAULT_MAX_CANDIDATES = 15
DEFAULT_PRE_ROLL = 8                        # segundos antes del evento en el clip
DEFAULT_POST_ROLL = 7                       # segundos después del evento en el clip
MIN_GAP_SECONDS = 20                        # separación mínima entre candidatos
MERGE_WINDOW_SECONDS = 8                    # señales a menos de 8 s se fusionan en 1 candidato
BALL_GAP_MAX_SECONDS = 1.0                  # huecos de balón <= 1 s se interpolan
BALL_EMA_ALPHA = 0.4                        # suavizado del balón (0-1, mayor = más reactivo)
BALL_MAX_SPEED_BH = 25.0                    # salto máximo creíble del balón
# en alturas/s (coordenadas de plano: balón + movimiento de cámara). Un tiro
# durísimo son ~17 alturas/s; a conf 0,10 YOLO confunde el balón con líneas
# blancas y el balón "teletransporta" a 30-45 alturas/s. Esos saltos se
# descartan en _suavizar_balon (ver ahí).
POSSESSION_RADIUS_HEIGHTS = 2.5             # alturas de jugador balón-jugador para posesión
COLOR_DISTANCE_MAX = 55.0                   # distancia HSV máxima para asignar equipo
COLOR_MARGIN_MIN = 6.0                      # margen mínimo entre la mejor y la 2ª referencia
# Asociación propia entre fotogramas (ver _track_video): ByteTrack pierde
# los tracks con cámara que sigue al balón (pan/zoom entre muestras) y se
# traga el balón (detecciones pequeñas de baja confianza). Con predict() +
# vecino más cercano los IDs se mantienen y el balón aparece en ~20% de
# muestras en vez de ~0%.
TRACK_MATCH_RADIUS = 8.0                    # radio de asociación en unidades de plano (0-100)
TRACK_MAX_GAP_SAMPLES = 15                  # muestras sin ver un track antes de darlo por perdido
VOTE_MIN_KNOWN = 3                          # votos mínimos de un equipo conocido para ganar
VOTE_MIN_KNOWN_FRAC = 0.25                  # y al menos el 25% de todos los votos del track
SPEED_PEAK_BH = 5.0                         # alturas/s para "pico de velocidad"
SPEED_FAST_BH = 3.0                         # alturas/s para "balón rápido"
STOP_FROM_BH = 4.0                          # velocidad inicial para "parada brusca"
STOP_TO_BH = 0.8                            # velocidad final para "parada brusca"
CROWD_RADIUS_HEIGHTS = 3.5                  # radio (alturas) para "agrupación"
CROWD_MIN_PEOPLE = 5                        # personas en ese radio para "agrupación"
TRANSITION_ADVANCE_HEIGHTS = 8.0            # alturas de avance en 3 s para "transición"
SET_PIECE_STOP_SECONDS = 2.0                # balón parado para "balón parado / saque"
TRAMO_MINUTES = 15                          # tramos para la posesión por tramos
DEFAULT_GK_COLOR_LOCAL = "#FFEB3B"          # portero local (amarillo)
DEFAULT_GK_COLOR_VISITANTE = "#FF9800"      # portero visitante (naranja)
DEFAULT_REFEREE_COLOR = "#000000"           # árbitro (negro)
GK_NEAR_HEIGHTS = 4.0                       # balón cerca del portero (alturas)
GK_AREA_HEIGHTS = 6.0                       # balón en zona del portero (alturas)
# Faltas: una falta es un frenazo + agrupación + juego parado después
# (el balón queda parado o se pierde mientras se prepara el lanzamiento).
# Las rachas de balón parado NO sirven: en seguimiento el balón se pierde
# a menudo y una sola muestra sin balón reseteaba la racha a 0 (en el
# partido real ninguna falta llegaba a 2 s seguidas). Por eso se miden
# TIEMPOS TOTALES en (t, t+6 s]. El árbitro no es un requisito (solo está
# visible en el 44% de las muestras): sube la confianza a 0,65 y el tope
# de 5 faltas por partido conserva antes las que tienen árbitro cerca.
FOUL_STOP_AFTER_SECONDS = 1.0               # balón visto y parado tras la falta (total)
FOUL_LOST_AFTER_SECONDS = 2.0               # o balón perdido tras la falta (total)
REFEREE_NEAR_HEIGHTS = 6.0                  # árbitro junto al balón en ±2 s
MAX_FOUL_CUTS = 5                           # tope de faltas por partido
# Goles en seguimiento: un tiro que se pierde y se reanuda en el centro
# (saque de centro) es gol; si reaparece cerca del portero es parada o
# saque de puerta (tiro). Sin portero visible también funciona. La
# reanudación se mide también en tiempo total (no racha) y con gente
# alrededor (un saque de centro reúne a los dos equipos).
GOAL_LOST_MIN_SECONDS = 4.0                 # balón perdido al menos 4 s tras el tiro
GOAL_RESTART_WINDOW_SECONDS = 35.0          # ventana para buscar la reanudación
GOAL_RESTART_STOP_SECONDS = 1.5             # balón parado en la reanudación (total)
GOAL_RESTART_MIN_PEOPLE = 6                 # personas junto al balón en la reanudación
# Pases por trayectoria (cambio de portador del balón)
CARRIER_RADIUS_HEIGHTS = 1.8                # balón-jugador para "portador del balón"
PASS_FLIGHT_MIN_SECONDS = 0.2               # vuelo mínimo para ser pase (no un toque)
PASS_FLIGHT_MAX_SECONDS = 4.0               # tiempo máximo entre salida y llegada
PASS_SPEED_MIN_BH = 1.5                     # velocidad mínima del balón en vuelo
PASS_MIN_DISTANCE_BH = 3.0                  # distancia mínima del pase (alturas)
SEQUENCE_GAP_SECONDS = 8.0                  # hueco máximo entre pases de una secuencia
# Homografía / campo (105 x 68 m). La referencia es la librería `sports`
# de Roboflow (SoccerPitchConfiguration 12000x7000 cm + ViewTransformer):
# aquí se implementa con cv2 para no depender de ella en el PC del
# usuario. El modelo de keypoints del campo es el Soccana YOLOv11n-pose
# (29 puntos: esquinas, áreas, círculo central) de HuggingFace; se
# descarga solo la primera vez que se usa el mapa del campo.
PITCH_LENGTH_M = 105.0                      # largo del campo en metros
PITCH_WIDTH_M = 68.0                        # ancho del campo en metros
PITCH_GOAL_HALF_WIDTH_M = 3.66              # media portería (7,32 m)
PITCH_BOX_DEPTH_M = 16.5                    # área grande: profundidad
PITCH_BOX_HALF_WIDTH_M = 20.16              # área grande: media anchura
PITCH_GOAL_BOX_DEPTH_M = 5.5                # área pequeña: profundidad
PITCH_GOAL_BOX_HALF_WIDTH_M = 9.16          # área pequeña: media anchura
PITCH_CENTER_R_M = 9.15                     # radio del círculo central
PITCH_KEYPOINT_MODEL_URL = (
    "https://huggingface.co/Adit-jain/Soccana_Keypoint/resolve/main/"
    "Model/weights/best.pt")
PITCH_KEYPOINT_MODEL_FILE = "pitch_keypoints.pt"
PITCH_KEYFRAME_SECONDS = 5.0                # cada cuántos s se recalibra el campo
PITCH_KEYPOINT_CONF = 0.5                   # confianza mínima del keypoint
PITCH_KEYPOINT_IMGSZ = 320                  # 0,20 s/predicción en CPU (640: 0,50 s)
PITCH_MIN_KEYPOINTS = 4                     # mínimos para una homografía
PITCH_OBS_MAX_AGE_S = 60.0                  # los keypoints viejos caducan
PITCH_MAX_SPEED_MS = 35.0                   # ~126 km/h; más rápido es artefacto
PITCH_GOAL_LINE_MARGIN_M = 1.0              # margen al cruzar la línea de gol

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
    """Error al procesar el vídeo (archivo, ffmpeg o YOLO)."""


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
    radar_video: Path | str | None = None,
) -> Path:
    """Corta un clip con FFmpeg (H.264 + AAC). Nombre: MMSS_evento.mp4.

    Los tiempos son ABSOLUTOS del vídeo (no relativos al saque inicial).
    Si se pasa radar_video (un mp4 con el mapa del campo, misma duración
    que el clip), se incrusta abajo a la derecha con overlay.
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
    if radar_video is not None and Path(radar_video).exists():
        _run([
            _tool_path("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-i", str(video_path),
            "-i", str(radar_video),
            "-t", f"{end - start:.3f}",
            "-filter_complex",
            "[0:v][1:v]overlay=W-w-10:H-h-10:shortest=1[v]",
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart",
            str(output_path),
        ])
    else:
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


def export_radar_video(
    muestras: list[dict],
    t_video_inicio: float,
    t_video_fin: float,
    output_path: Path | str,
    colores: dict,
    fps: float = 2.0,
) -> Path | None:
    """Genera el vídeo del radar (mapa del campo con puntos de colores)
    para un intervalo de vídeo. Un fotograma cada 1/fps s, con el radar
    de la muestra más cercana. Devuelve None si no hay coordenadas de
    campo en el intervalo."""
    import bisect
    import cv2
    tiempos = [m["t_video"] for m in muestras]
    if not tiempos:
        return None
    i0 = bisect.bisect_left(tiempos, t_video_inicio)
    i1 = bisect.bisect_right(tiempos, t_video_fin)
    if i0 >= i1:
        return None
    if not any(muestras[i].get("pitch_ok") for i in range(i0, i1)):
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n_frames = max(1, int((t_video_fin - t_video_inicio) * fps))
    for k in range(n_frames + 1):
        t_video = t_video_inicio + k / fps
        if t_video > t_video_fin:
            break
        j = bisect.bisect_left(tiempos, t_video)
        idx = min(max(j, i0), i1 - 1)
        if j > i0 and (j >= i1 or abs(tiempos[j - 1] - t_video) < abs(tiempos[j] - t_video)):
            idx = j - 1
        img = draw_radar(muestras[idx], colores)
        if writer is None:
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer is not None:
        writer.release()
    return output_path if output_path.exists() else None


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
    """Color mediano del torso (centro del bbox) en HSV OpenCV. None si no se puede.

    Mediana solo de la mitad más saturada: en camisetas a rayas la mitad
    blanca diluye el color y en contraluz el blanco lavado hace lo mismo;
    los píxeles de color (los de la camiseta) son los más saturados.

    NOTA: se probó KMeans (k=2, como en el proyecto football_analysis del
    vídeo) para separar la camiseta del fondo, con dos reglas de elección
    (grupo dominante / más saturado, y grupo del centro del torso). En
    datos reales de este partido salió PEOR que la mediana: el césped
    verde y saturado domina el torso cuando el jugador está lejos y
    KMeans lo elegía como camiseta (4 falsos porteros verdes en 07:54,
    desconocido 14 en el gol frente a 12 con la mediana). El KMeans del
    vídeo funciona con cámara fija y jugadores grandes; con VEO, jugadores
    lejos/pequeños y contraluz de atardecer, la mediana es más robusta.
    """
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
    pixeles = hsv.reshape(-1, 3)
    umbral_s = np.median(pixeles[:, 1])
    vivos = pixeles[pixeles[:, 1] >= umbral_s]
    if len(vivos) == 0:
        return None
    mediana = np.median(vivos, axis=0)
    return float(mediana[0]), float(mediana[1]), float(mediana[2])


def _clasificar_color(hsv, ref_local, ref_visitante,
                      ref_gk_local=None, ref_gk_visitante=None, ref_arbitro=None) -> str:
    """'local' / 'visitante' / 'gk_local' / 'gk_visitante' / 'arbitro' /
    'desconocido' según distancia HSV a los colores de referencia.

    Los porteros y el árbitro tienen su propio color: sin ellos salían
    "desconocido" y el árbitro junto al balón falseaba la posesión.
    Si el color está demasiado lejos de todos, o entre dos referencias
    (p. ej. portero con camiseta parecida a la de su equipo), desconocido.
    """
    if hsv is None:
        return "desconocido"
    refs = {"local": ref_local, "visitante": ref_visitante}
    if ref_gk_local is not None:
        refs["gk_local"] = ref_gk_local
    if ref_gk_visitante is not None:
        refs["gk_visitante"] = ref_gk_visitante
    if ref_arbitro is not None:
        refs["arbitro"] = ref_arbitro
    distancias = {k: _distancia_hsv(*hsv, *v) for k, v in refs.items()}
    orden = sorted(distancias.items(), key=lambda kv: kv[1])
    mejor, d_mejor = orden[0]
    if d_mejor > COLOR_DISTANCE_MAX:
        return "desconocido"
    if len(orden) > 1 and orden[1][1] - d_mejor < COLOR_MARGIN_MIN:
        return "desconocido"
    return mejor


def _equipo_para_posesion(team: str) -> str | None:
    """Equipo que cuenta para la posesión: el portero cuenta como su equipo;
    el árbitro y los desconocidos no cuentan (se mantiene la última posesión)."""
    if team in ("local", "visitante"):
        return team
    if team == "gk_local":
        return "local"
    if team == "gk_visitante":
        return "visitante"
    return None


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
# Homografía: del fotograma al campo en metros (mapa / radar)
# ---------------------------------------------------------------------------
# Coordenadas en metros de los 29 keypoints del modelo Soccana (ver
# validar_homografia.py en el workspace de análisis): x=0 portería
# izquierda, x=105 derecha; y=0 banda lejana, y=68 banda cercana. Caja
# grande 16,5x40,32, caja pequeña 5,5x18,32, círculo central r=9,15,
# punto de penalti a 11 m. Validado en el Cirbonero-Milagres: error medio
# de reproyección 0,5-0,7 m; en seguimiento solo se ven 6-8 puntos por
# fotograma (casi siempre el centro), así que la H se suaviza en el
# tiempo y su calidad depende de la dispersión en x (el eje largo).
PITCH_KEYPOINTS_M: dict[int, tuple[float, float]] = {
    0: (0, 0), 16: (105, 0), 9: (0, 68), 25: (105, 68),
    1: (0, 13.84), 2: (16.5, 13.84), 3: (0, 54.16), 4: (16.5, 54.16),
    5: (0, 24.84), 6: (5.5, 24.84), 7: (0, 43.16), 8: (5.5, 43.16),
    17: (105, 13.84), 18: (88.5, 13.84), 19: (105, 54.16), 20: (88.5, 54.16),
    21: (105, 24.84), 22: (99.5, 24.84), 23: (105, 43.16), 24: (99.5, 43.16),
    10: (20.15, 34), 26: (84.85, 34),
    11: (52.5, 0), 12: (52.5, 68), 13: (52.5, 24.85), 14: (52.5, 43.15),
    15: (52.5, 34), 27: (43.35, 34), 28: (61.65, 34),
}
PITCH_H_MAX_AGE_S = 30.0                # H más vieja que esto no vale


def _cargar_modelo_campo(ruta: str | Path | None = None):
    """Carga el modelo de keypoints del campo (lo descarga de HuggingFace
    la primera vez si no está). Devuelve el modelo YOLO o None si no se
    pudo (el análisis sigue sin mapa del campo)."""
    import urllib.request
    if ruta is None:
        ruta = Path(__file__).resolve().parent / PITCH_KEYPOINT_MODEL_FILE
    ruta = Path(ruta)
    if not ruta.exists():
        try:
            print(f"Descargando modelo de keypoints del campo ({PITCH_KEYPOINT_MODEL_URL})...")
            urllib.request.urlretrieve(PITCH_KEYPOINT_MODEL_URL, str(ruta))
        except Exception as error:
            print(f"No se pudo descargar el modelo del campo: {error}")
            return None
    try:
        from ultralytics import YOLO
        return YOLO(str(ruta))
    except Exception as error:
        print(f"No se pudo cargar el modelo del campo ({ruta}): {error}")
        return None


class PitchMapper:
    """Homografía fotograma -> campo (metros) con suavizado temporal.

    En seguimiento la cámara se mueve, así que cada keyframe (cada
    PITCH_KEYFRAME_SECONDS) se vuelve a estimar H con los keypoints
    visibles en ESE fotograma (mezclar puntos de fotogramas distintos
    sería mezclar dos vistas). La H nueva se suaviza con la anterior
    (0,7/0,3) y solo se acepta si los puntos cubren bastante campo en
    el eje largo: calidad alta (dispersión en x >= 40 m), media
    (>= 20 m) o baja (solo el centro). Sin H válida reciente no hay
    coordenadas de campo: el clasificador usa las reglas de plano.
    """

    def __init__(self):
        self.H = None
        self.calidad = "ninguna"
        self.n_puntos = 0
        self.ultima_t: float | None = None

    def update(self, kpts, t: float) -> bool:
        """kpts: array (29, 3) x, y, conf en píxeles. Devuelve True si
        la H quedó actualizada (o sigue válida)."""
        import numpy as np
        import cv2
        vis = [i for i in range(29)
               if i in PITCH_KEYPOINTS_M and kpts[i, 2] >= PITCH_KEYPOINT_CONF]
        if len(vis) >= PITCH_MIN_KEYPOINTS:
            src = np.array([[kpts[i, 0], kpts[i, 1]] for i in vis],
                           dtype=np.float32)
            dst = np.array([PITCH_KEYPOINTS_M[i] for i in vis],
                           dtype=np.float32)
            H, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is not None:
                proj = cv2.perspectiveTransform(
                    src.reshape(-1, 1, 2), H).reshape(-1, 2)
                err = float(np.linalg.norm(proj - dst, axis=1).mean())
                if err <= 2.0:
                    spread_x = float(dst[:, 0].max() - dst[:, 0].min())
                    calidad = ("alta" if spread_x >= 40.0
                               else "media" if spread_x >= 20.0 else "baja")
                    if self.H is not None:
                        H = 0.7 * H + 0.3 * self.H
                        H = H / H[2, 2]
                    self.H = H
                    self.calidad = calidad
                    self.n_puntos = len(vis)
                    self.ultima_t = t
        if self.H is None:
            return False
        if self.ultima_t is not None and t - self.ultima_t > PITCH_H_MAX_AGE_S:
            return False
        return True

    def to_pitch(self, x_px: float, y_px: float) -> tuple[float, float] | None:
        if self.H is None:
            return None
        import cv2
        import numpy as np
        p = cv2.perspectiveTransform(
            np.array([[[float(x_px), float(y_px)]]], dtype=np.float32),
            self.H)[0, 0]
        x_m, y_m = float(p[0]), float(p[1])
        # La homografía con pocos keypoints extrapola mal lejos de ellos
        # (porteros en x=2273 m): fuera de un margen sensato el punto no
        # es fiable y se descarta en vez de envenenar estadísticas y radar.
        if not (-20.0 <= x_m <= 125.0 and -20.0 <= y_m <= 88.0):
            return None
        return x_m, y_m

    @property
    def valido(self) -> bool:
        return self.H is not None


def _en_area_campo(xy: tuple[float, float] | None) -> bool:
    """Balón dentro de un área grande (16,5 m x 40,32 m)."""
    if xy is None:
        return False
    x, y = xy
    return ((x <= PITCH_BOX_DEPTH_M or x >= PITCH_LENGTH_M - PITCH_BOX_DEPTH_M)
            and PITCH_WIDTH_M / 2 - PITCH_BOX_HALF_WIDTH_M <= y
            <= PITCH_WIDTH_M / 2 + PITCH_BOX_HALF_WIDTH_M)


def _en_banda_campo(xy: tuple[float, float] | None) -> bool:
    if xy is None:
        return False
    _x, y = xy
    return y <= 10.0 or y >= PITCH_WIDTH_M - 10.0


def _cruza_linea_gol(xy_antes: tuple[float, float] | None,
                     xy_despues: tuple[float, float] | None) -> bool:
    """El balón cruza una línea de gol por dentro de la portería."""
    if xy_antes is None or xy_despues is None:
        return False
    for limite in (0.0, PITCH_LENGTH_M):
        if (xy_antes[0] - limite) * (xy_despues[0] - limite) < 0:
            y_cruce = (xy_antes[1] + xy_despues[1]) / 2.0
            if abs(y_cruce - PITCH_WIDTH_M / 2) <= PITCH_GOAL_HALF_WIDTH_M + PITCH_GOAL_LINE_MARGIN_M:
                return True
    return False


def draw_radar(muestra: dict, colores: dict, ancho: int = 420,
               alto: int = 280) -> "np.ndarray":
    """Dibuja el mapa del campo (radar) de una muestra: puntos de colores
    por jugador (local/visitante, porteros con borde, árbitro), balón
    blanco. Usa las coordenadas de campo (pitch_x/pitch_y) de la muestra.
    Devuelve una imagen BGR (numpy) lista para incrustar en un clip."""
    import numpy as np
    import cv2
    img = np.full((alto, ancho, 3), (46, 125, 50), dtype=np.uint8)  # verde
    m = 14  # margen
    x0, y0 = m, m
    w = ancho - 2 * m
    h = alto - 2 * m
    ex, ey = w / PITCH_LENGTH_M, h / PITCH_WIDTH_M

    def _px(x_m: float, y_m: float) -> tuple[int, int]:
        return (int(x0 + x_m * ex), int(y0 + y_m * ey))

    def _linea(p1, p2, color=(255, 255, 255), grosor=1):
        cv2.line(img, _px(*p1), _px(*p2), color, grosor, cv2.LINE_AA)

    # Contorno, medio campo, círculo central y áreas.
    cv2.rectangle(img, _px(0, 0), _px(PITCH_LENGTH_M, PITCH_WIDTH_M),
                  (255, 255, 255), 1, cv2.LINE_AA)
    _linea((PITCH_LENGTH_M / 2, 0), (PITCH_LENGTH_M / 2, PITCH_WIDTH_M))
    cv2.circle(img, _px(PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2),
               int(PITCH_CENTER_R_M * ex), (255, 255, 255), 1, cv2.LINE_AA)
    for gx in (0.0, PITCH_LENGTH_M):
        d = PITCH_BOX_DEPTH_M if gx == 0.0 else -PITCH_BOX_DEPTH_M
        g = PITCH_GOAL_BOX_DEPTH_M if gx == 0.0 else -PITCH_GOAL_BOX_DEPTH_M
        y1 = PITCH_WIDTH_M / 2 - PITCH_BOX_HALF_WIDTH_M
        y2 = PITCH_WIDTH_M / 2 + PITCH_BOX_HALF_WIDTH_M
        _linea((gx, y1), (gx + d, y1)); _linea((gx + d, y1), (gx + d, y2))
        _linea((gx + d, y2), (gx, y2))
        y1 = PITCH_WIDTH_M / 2 - PITCH_GOAL_BOX_HALF_WIDTH_M
        y2 = PITCH_WIDTH_M / 2 + PITCH_GOAL_BOX_HALF_WIDTH_M
        _linea((gx, y1), (gx + g, y1)); _linea((gx + g, y1), (gx + g, y2))
        _linea((gx + g, y2), (gx, y2))
    # Punto de penalti y centro.
    for px_, py_ in ((11, 34), (94, 34), (52.5, 34)):
        cv2.circle(img, _px(px_, py_), 2, (255, 255, 255), -1, cv2.LINE_AA)

    def _color_bgr(hex_color: str) -> tuple[int, int, int]:
        r, g, b = _hex_a_rgb(hex_color)
        return (b, g, r)

    c_local = _color_bgr(colores.get("local", "#1E88E5"))
    c_visitante = _color_bgr(colores.get("visitante", "#E53935"))
    c_arbitro = _color_bgr(colores.get("arbitro", "#CCFF00"))
    for p in muestra.get("people", []):
        if p.get("pitch_x") is None or p.get("pitch_y") is None:
            continue
        team = p.get("team", "desconocido")
        if team in ("local", "gk_local"):
            color = c_local
        elif team in ("visitante", "gk_visitante"):
            color = c_visitante
        elif team == "arbitro":
            color = c_arbitro
        else:
            continue
        # La homografía extrapola mal en los bordes (y<0 o y>68): el
        # punto se recorta al campo para que nunca desaparezca del radar.
        cx, cy = _px(min(max(p["pitch_x"], 0.0), PITCH_LENGTH_M),
                      min(max(p["pitch_y"], 0.0), PITCH_WIDTH_M))
        if team in ("gk_local", "gk_visitante"):
            cv2.circle(img, (cx, cy), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 5, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)
    balon = muestra.get("ball")
    if balon and balon.get("pitch_x") is not None:
        cx, cy = _px(min(max(balon["pitch_x"], 0.0), PITCH_LENGTH_M),
                      min(max(balon["pitch_y"], 0.0), PITCH_WIDTH_M))
        cv2.circle(img, (cx, cy), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def _anotar_campo_desde_video(
    video_path: Path | str,
    muestras: list[dict],
    pitch_model: str | None = None,
    keyframe_seconds: float = PITCH_KEYFRAME_SECONDS,
) -> tuple[list[dict], int]:
    """Mapea unas muestras ya trackeadas (con t_video y posiciones de
    plano 0-100) a metros de campo, pasando el modelo de keypoints por
    el vídeo (un keyframe cada keyframe_seconds). Para remezclar chunks
    guardados sin volver a pasar YOLO: el plano 0-100 ES la posición en
    píxeles normalizada (pies = centro + h/2). Añade por muestra
    pitch_ok/pitch_calidad y por persona/balón pitch_x/pitch_y; después
    _suavizar_balon arrastra las coordenadas de campo del balón
    (ball_pitch_s) y _compensar_camara calcula ball_speed_ms.
    Devuelve (muestras, nº de muestras con campo)."""
    import cv2
    video_path = Path(video_path).expanduser().resolve()
    modelo = _cargar_modelo_campo(pitch_model)
    if modelo is None or not muestras:
        for m in muestras:
            m["pitch_ok"] = False
            m["pitch_calidad"] = "ninguna"
        return muestras, 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"No se pudo abrir el vídeo: {video_path}")
    ancho = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0
    alto = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0
    fps_real = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mapper = PitchMapper()
    # Keyframes: del primer al último t_video de las muestras.
    t_ini = muestras[0]["t_video"]
    t_fin = muestras[-1]["t_video"]
    serie_h: list[tuple[float, object, str]] = []  # (t_video, H, calidad)
    try:
        t_k = t_ini
        while t_k <= t_fin:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_k * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                try:
                    rk = modelo.predict(frame, imgsz=PITCH_KEYPOINT_IMGSZ,
                                        conf=0.25, verbose=False)[0]
                    if (rk.keypoints is not None
                            and len(rk.keypoints.data) > 0):
                        # update() True = hay H vigente en este keyframe
                        # (nueva o anterior si sigue fresca).
                        if mapper.update(rk.keypoints.data[0].cpu().numpy(), t_k):
                            serie_h.append((t_k, mapper.H.copy(), mapper.calidad))
                except Exception:
                    pass
            t_k += keyframe_seconds
    finally:
        cap.release()
    if not serie_h:
        for m in muestras:
            m["pitch_ok"] = False
            m["pitch_calidad"] = "ninguna"
        return muestras, 0
    import bisect
    tiempos_h = [h[0] for h in serie_h]
    con_campo = 0
    for m in muestras:
        j = bisect.bisect_right(tiempos_h, m["t_video"]) - 1
        if j < 0:
            j = 0
        t_h, H, calidad = serie_h[j]
        if m["t_video"] - t_h > PITCH_H_MAX_AGE_S or H is None:
            m["pitch_ok"] = False
            m["pitch_calidad"] = "ninguna"
            continue
        import numpy as np
        def _map(x100: float, y100: float):
            p = cv2.perspectiveTransform(
                np.array([[[x100 / 100.0 * ancho, y100 / 100.0 * alto]]],
                         dtype=np.float32), H)[0, 0]
            return float(p[0]), float(p[1])
        for p in m.get("people", []):
            pp = _map(p["x"], p["y"] + p.get("h", 0.0) / 2.0)
            p["pitch_x"], p["pitch_y"] = pp
            p["px"] = p["x"] / 100.0 * ancho
            p["py"] = (p["y"] + p.get("h", 0.0) / 2.0) / 100.0 * alto
        if m.get("ball") is not None:
            pp = _map(m["ball"]["x"], m["ball"]["y"])
            m["ball"]["pitch_x"], m["ball"]["pitch_y"] = pp
        m["pitch_ok"] = True
        m["pitch_calidad"] = calidad
        con_campo += 1
    return muestras, con_campo


def _corregir_eje_x_campo(muestras: list[dict]) -> int:
    """Corrige el eje x del campo con los porteros.

    Con keypoints solo del centro, la homografía comprime todo hacia el
    centro (el balón del pase atrás 09:49 salía en x=52 en vez de x~5,
    y el 87% del balón caía en el tercio central). Los dos porteros
    están en sus líneas de gol (x=0 y x=105): se estira el eje x para
    que caigan ahí. Se calcula en una rejilla cada 30 s (ventana ±30 s,
    medianas de la x de cada portero) y se interpola; la y no se toca
    (la dan bien los keypoints del centro). Solo se aplica si los dos
    porteros están separados >30 m en el mapa y el factor queda en
    0,5-3,0. Devuelve el nº de muestras corregidas.
    """
    import bisect
    if not muestras:
        return 0
    tiempos = [m["t"] for m in muestras]

    def _x_gk(m: dict, equipo: str) -> float | None:
        xs = sorted(p["pitch_x"] for p in m.get("people", [])
                    if p.get("team") == equipo and p.get("pitch_x") is not None)
        return xs[len(xs) // 2] if xs else None

    # Rejilla cada 30 s.
    t_ini, t_fin = tiempos[0], tiempos[-1]
    rejilla: list[tuple[float, float, float]] = []
    tg = t_ini
    while tg <= t_fin:
        i0 = bisect.bisect_left(tiempos, tg - 30.0)
        i1 = bisect.bisect_right(tiempos, tg + 30.0)
        xs_loc = [_x_gk(muestras[i], "gk_local") for i in range(i0, i1)]
        xs_vis = [_x_gk(muestras[i], "gk_visitante") for i in range(i0, i1)]
        xs_loc = sorted(v for v in xs_loc if v is not None)
        xs_vis = sorted(v for v in xs_vis if v is not None)
        if xs_loc and xs_vis:
            x1 = min(xs_loc[len(xs_loc) // 2], xs_vis[len(xs_vis) // 2])
            x2 = max(xs_loc[len(xs_loc) // 2], xs_vis[len(xs_vis) // 2])
            # Solo si los porteros están bien separados en el mapa: con
            # keypoints solo del centro la x sale comprimida y los
            # porteros caen cerca uno del otro (x2-x1 ~23 m); estirar
            # desde ahí es un factor salvaje (a>2) que empuja los puntos
            # fuera del campo y vacía el radar. Sin separación fiable,
            # mejor no corregir: las posiciones comprimidas al menos son
            # consistentes y el radar muestra la jugada.
            if 40.0 <= x2 - x1 <= 120.0:
                a = 105.0 / (x2 - x1)
                if 1.0 <= a <= 2.0:
                    rejilla.append((tg, a, -a * x1))
        tg += 30.0
    if not rejilla:
        return 0
    n = 0
    for m in muestras:
        t = m["t"]
        if t <= rejilla[0][0]:
            a, b = rejilla[0][1], rejilla[0][2]
        elif t >= rejilla[-1][0]:
            a, b = rejilla[-1][1], rejilla[-1][2]
        else:
            j = bisect.bisect_right([r[0] for r in rejilla], t) - 1
            t0, a0, b0 = rejilla[j]
            t1, a1, b1 = rejilla[j + 1]
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            a, b = a0 + f * (a1 - a0), b0 + f * (b1 - b0)
        for p in m.get("people", []):
            if p.get("pitch_x") is not None:
                p["pitch_x"] = a * p["pitch_x"] + b
        if m.get("ball") is not None and m["ball"].get("pitch_x") is not None:
            m["ball"]["pitch_x"] = a * m["ball"]["pitch_x"] + b
        if m.get("ball_pitch_s") is not None:
            m["ball_pitch_s"]["x"] = a * m["ball_pitch_s"]["x"] + b
        n += 1
    # La velocidad en campo se recalcula con las posiciones corregidas
    # (el factor `a` cambia la escala en x).
    for i, m in enumerate(muestras):
        v_ms = 0.0
        if m.get("ball_pitch_s") is not None:
            j = i
            while j > 0 and muestras[j]["t"] > m["t"] - 0.5:
                j -= 1
            anterior = muestras[j]
            if (anterior.get("ball_pitch_s") is not None
                    and anterior["t"] < m["t"]):
                dt = m["t"] - anterior["t"]
                if dt > 0:
                    v_ms = min(math.hypot(
                        m["ball_pitch_s"]["x"] - anterior["ball_pitch_s"]["x"],
                        m["ball_pitch_s"]["y"] - anterior["ball_pitch_s"]["y"]) / dt,
                        PITCH_MAX_SPEED_MS)
        m["ball_speed_ms"] = v_ms
    return n


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
    color_gk_local_hex: str = DEFAULT_GK_COLOR_LOCAL,
    color_gk_visitante_hex: str = DEFAULT_GK_COLOR_VISITANTE,
    color_arbitro_hex: str = DEFAULT_REFEREE_COLOR,
    flip_x: bool = False,
    flip_y: bool = False,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    progress: ProgressCallback | None = None,
    conf: float = 0.10,
    pitch_model: str | None = None,
    pitch_keyframe_seconds: float = PITCH_KEYFRAME_SECONDS,
) -> tuple[list[dict], float, dict[int, str]]:
    """Pasa YOLO por el vídeo fotograma a fotograma con IDs persistentes.

    Analiza solo [start_seconds, end_seconds] (el calentamiento, antes del
    saque inicial, no se procesa). Devuelve (muestras, duración_del_partido,
    equipo_por_track_id). En cada muestra ``t`` es relativo al saque inicial
    y ``t_video`` absoluto. Las coordenadas x, y son DE PLANO (0-100 en la
    imagen), no de campo: con cámara que sigue al balón no hay correspondencia
    fija con el campo.

    Si hay modelo de campo (pitch_model o el archivo pitch_keypoints.pt
    junto a este módulo), cada keyframe recalibra la homografía y cada
    persona/balón lleva además pitch_x/pitch_y en metros de campo (105x68)
    y la muestra pitch_ok. Sin modelo, esas claves son None/False y todo
    sigue con las reglas de plano.
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
    if end_seconds is None:
        end_seconds = total_duration
    end_seconds = min(float(end_seconds), total_duration)
    duracion_partido = max(0.0, end_seconds - start_seconds)
    progress = progress or (lambda _m, _v: None)

    paso = max(1, int(round(fps_real / max(1, sample_fps))))
    ref_local = _rgb_a_hsv_opencv(*_hex_a_rgb(color_local_hex))
    ref_visitante = _rgb_a_hsv_opencv(*_hex_a_rgb(color_visitante_hex))
    ref_gk_local = _rgb_a_hsv_opencv(*_hex_a_rgb(color_gk_local_hex))
    ref_gk_visitante = _rgb_a_hsv_opencv(*_hex_a_rgb(color_gk_visitante_hex))
    ref_arbitro = _rgb_a_hsv_opencv(*_hex_a_rgb(color_arbitro_hex))

    try:
        detector = YOLO(yolo_model)
    except Exception as error:
        cap.release()
        raise VideoProcessingError(f"No se pudo cargar YOLO ({yolo_model}): {error}") from error

    # Modelo de keypoints del campo (opcional): pitch_model con ruta lo
    # usa; None usa pitch_keypoints.pt junto a este módulo si existe;
    # "" lo desactiva. Sin él no hay mapa del campo.
    modelo_campo = None
    mapper = PitchMapper()
    if pitch_model:
        modelo_campo = _cargar_modelo_campo(pitch_model)
    elif pitch_model is None and (Path(__file__).resolve().parent
                                  / PITCH_KEYPOINT_MODEL_FILE).exists():
        modelo_campo = _cargar_modelo_campo()
    ultimo_keyframe = -1e9

    # Saltar hasta el saque inicial
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_seconds) * 1000.0)
    frame_idx = int(start_seconds * fps_real)
    muestras: list[dict] = []
    votos_color: dict[int, Counter] = defaultdict(Counter)
    # Asociación propia entre muestras: predict() no da IDs, así que cada
    # persona se casa con el track activo más cercano (radio + altura
    # parecida). Sin ByteTrack: con cámara que sigue al balón el pan/zoom
    # entre muestras (0,1 s) rompe sus predicciones de Kalman y sus
    # umbrales (new_track 0,25) se tragan el balón; con predict() el balón
    # aparece en ~20% de muestras y las personas en ~18 por fotograma.
    tracks_activos: dict[int, dict] = {}
    siguiente_id = 0
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
                    resultados = detector.predict(
                        frame, classes=[0, 32], imgsz=imgsz, verbose=False,
                        device="cpu", half=False, conf=conf,
                    )
                except Exception as error:
                    raise VideoProcessingError(
                        f"Falló la detección con YOLO: {error}"
                    ) from error
                personas: list[dict] = []
                balon: dict | None = None
                if resultados and len(resultados) > 0:
                    res = resultados[0]
                    if res.boxes is not None and res.boxes.cls is not None:
                        clases = res.boxes.cls.cpu().tolist()
                        cajas = res.boxes.xyxy.cpu().tolist()
                        confs = (res.boxes.conf.cpu().tolist()
                                 if res.boxes.conf is not None
                                 else [0.0] * len(clases))
                        detecciones = []
                        for clase, caja, conf_det in zip(clases, cajas, confs):
                            clase = int(clase)
                            x0, y0, x1, y1 = [float(v) for v in caja]
                            cx = ((x0 + x1) / 2.0) / ancho * 100.0
                            cy = ((y0 + y1) / 2.0) / alto * 100.0
                            if flip_x:
                                cx = 100.0 - cx
                            if flip_y:
                                cy = 100.0 - cy
                            if clase == 0:
                                detecciones.append({
                                    "cx": cx, "cy": cy,
                                    "w": (x1 - x0) / ancho * 100.0,
                                    "h": (y1 - y0) / alto * 100.0,
                                    "conf": float(conf_det),
                                    "caja": (int(x0), int(y0), int(x1), int(y1)),
                                })
                            elif clase == 32:
                                if balon is None or conf_det > balon["conf"]:
                                    balon = {"x": cx, "y": cy, "conf": float(conf_det),
                                             "px": (x0 + x1) / 2.0,
                                             "py": (y0 + y1) / 2.0,
                                             "pitch_x": None, "pitch_y": None}
                        # Asociar por confianza descendente: la más segura elige primero.
                        detecciones.sort(key=lambda d: -d["conf"])
                        usados: set[int] = set()
                        for det in detecciones:
                            mejor_id: int | None = None
                            mejor_dist = TRACK_MATCH_RADIUS
                            for tid, tr in tracks_activos.items():
                                if tid in usados:
                                    continue
                                if procesados - tr["ultimo"] > TRACK_MAX_GAP_SAMPLES:
                                    continue
                                dist = math.hypot(det["cx"] - tr["x"],
                                                  det["cy"] - tr["y"])
                                if dist > mejor_dist:
                                    continue
                                # Misma persona ≈ misma altura en plano.
                                if abs(det["h"] - tr["h"]) > 0.5 * max(det["h"], tr["h"], 1.0):
                                    continue
                                mejor_id, mejor_dist = tid, dist
                            if mejor_id is None:
                                mejor_id = siguiente_id
                                siguiente_id += 1
                            usados.add(mejor_id)
                            tracks_activos[mejor_id] = {
                                "x": det["cx"], "y": det["cy"],
                                "h": det["h"], "ultimo": procesados,
                            }
                            personas.append({
                                "track_id": mejor_id,
                                "x": det["cx"], "y": det["cy"],
                                "w": det["w"], "h": det["h"],
                                # Pies en píxeles del fotograma (para la
                                # homografía: el centro del bbox no está
                                # en el suelo y falsea la posición).
                                "px": (det["caja"][0] + det["caja"][2]) / 2.0,
                                "py": float(det["caja"][3]),
                                "pitch_x": None, "pitch_y": None,
                                "team": "desconocido",
                            })
                            x0, y0, x1, y1 = det["caja"]
                            hsv = _color_torso(frame, x0, y0, x1, y1)
                            votos_color[mejor_id][
                                _clasificar_color(hsv, ref_local, ref_visitante,
                                                  ref_gk_local, ref_gk_visitante,
                                                  ref_arbitro)
                            ] += 1
                        # Limpiar tracks que llevan demasiado sin verse.
                        for tid in [t for t, tr in tracks_activos.items()
                                    if procesados - tr["ultimo"] > TRACK_MAX_GAP_SAMPLES]:
                            del tracks_activos[tid]
                # Keyframe del campo: recalibrar la homografía con los
                # keypoints visibles en este fotograma (0,20 s a 320 px).
                if (modelo_campo is not None
                        and t_video - ultimo_keyframe >= pitch_keyframe_seconds):
                    ultimo_keyframe = t_video
                    try:
                        rk = modelo_campo.predict(
                            frame, imgsz=PITCH_KEYPOINT_IMGSZ, conf=0.25,
                            verbose=False)[0]
                        if (rk.keypoints is not None
                                and len(rk.keypoints.data) > 0):
                            mapper.update(rk.keypoints.data[0].cpu().numpy(),
                                          t_video)
                    except Exception:
                        pass
                # Mapear pies/balón a metros de campo con la H vigente.
                pitch_ok = (mapper.valido and mapper.ultima_t is not None
                            and t_video - mapper.ultima_t <= PITCH_H_MAX_AGE_S)
                if pitch_ok:
                    for persona in personas:
                        pp = mapper.to_pitch(persona["px"], persona["py"])
                        persona["pitch_x"], persona["pitch_y"] = (
                            pp if pp is not None else (None, None))
                    if balon is not None:
                        pp = mapper.to_pitch(balon["px"], balon["py"])
                        balon["pitch_x"], balon["pitch_y"] = (
                            pp if pp is not None else (None, None))
                muestras.append({"t": t, "t_video": t_video,
                                 "people": personas, "ball": balon,
                                 "pitch_ok": pitch_ok,
                                 "pitch_calidad": mapper.calidad if pitch_ok
                                 else "ninguna"})
            frame_idx += 1
    finally:
        cap.release()

    # Equipo por track ID: gana el equipo conocido con más votos (los
    # "desconocido" no votan). Antes pedía mayoría del 60% y con torsos
    # ruidosos (rayas, contraluz, césped) casi ningún track llegaba: todos
    # salían desconocido y la posesión era 72% desconocido.
    equipo_por_track: dict[int, str] = {}
    for track_id, votos in votos_color.items():
        total = sum(votos.values())
        conocidos = {k: v for k, v in votos.items() if k != "desconocido"}
        if not conocidos or total < VOTE_MIN_KNOWN:
            equipo_por_track[track_id] = "desconocido"
            continue
        equipo, n = max(conocidos.items(), key=lambda kv: kv[1])
        if n < VOTE_MIN_KNOWN or n / total < VOTE_MIN_KNOWN_FRAC:
            equipo_por_track[track_id] = "desconocido"
            continue
        equipo_por_track[track_id] = equipo
    for muestra in muestras:
        for persona in muestra["people"]:
            persona["team"] = equipo_por_track.get(persona["track_id"], "desconocido")
    return muestras, duracion_partido, equipo_por_track


def _suavizar_balon(muestras: list[dict]) -> list[dict]:
    """EMA + interpolación lineal de huecos de balón <= 1 s. Añade 'ball_s'.

    Un solo pase sobre las muestras (índices): un hueco <= 1 s son como
    mucho 10 muestras a 10 fps, así que el coste es lineal.

    Antes de interpolar se descartan los saltos imposibles: a conf 0,10
    YOLO a veces confunde el balón con una línea blanca o una media, y el
    balón "teletransporta" (30-45 alturas/s, cuando un tiro durísimo son
    ~17). Una detección que se aleja de la última aceptada más rápido que
    BALL_MAX_SPEED_BH se descarta; si la siguiente detección confirma el
    salto (está cerca de la descartada), se acepta (el balón se movió de
    verdad tras un hueco largo, p. ej. un saque de puerta).
    """
    detecciones = [(i, m["t"], m["ball"]["x"], m["ball"]["y"],
                    m["ball"].get("pitch_x"), m["ball"].get("pitch_y"))
                   for i, m in enumerate(muestras) if m["ball"] is not None]
    if not detecciones:
        for m in muestras:
            m["ball_s"] = None
            m["ball_pitch_s"] = None
        return muestras
    import numpy as np
    alturas_todas = [p["h"] for m in muestras for p in m.get("people", [])
                     if p.get("h", 0) > 0]
    altura_global = float(np.median(alturas_todas)) if alturas_todas else 2.0

    def _altura_en(idx: int) -> float:
        hs = [p["h"] for p in muestras[idx].get("people", [])
              if p.get("h", 0) > 0]
        return float(np.median(hs)) if hs else altura_global

    filtradas: list[tuple] = []
    pendiente = None
    for det in detecciones:
        if not filtradas:
            filtradas.append(det)
            continue
        idx, t, x, y = det[0], det[1], det[2], det[3]
        idx0, t0, x0, y0 = filtradas[-1][:4]
        dt = t - t0
        if dt <= 0:
            continue
        dist_bh = math.hypot(x - x0, y - y0) / max(1.0, _altura_en(idx))
        if dist_bh / dt <= BALL_MAX_SPEED_BH:
            filtradas.append(det)
            pendiente = None
        elif pendiente is not None:
            idxp, tp, xp, yp = pendiente[:4]
            dtp = t - tp
            if dtp > 0:
                dist_p = math.hypot(x - xp, y - yp) / max(1.0, _altura_en(idx))
                if dist_p / dtp <= BALL_MAX_SPEED_BH:
                    filtradas.append(pendiente)
                    filtradas.append(det)
                    pendiente = None
                    continue
            pendiente = det
        else:
            pendiente = det
    detecciones = filtradas
    if not detecciones:
        for m in muestras:
            m["ball_s"] = None
            m["ball_pitch_s"] = None
        return muestras
    serie: dict[int, tuple[float, float]] = {}
    serie_pitch: dict[int, tuple[float, float] | None] = {}
    for k, (idx, t, x, y, px, py) in enumerate(detecciones):
        serie[idx] = (x, y)
        serie_pitch[idx] = (px, py) if px is not None and py is not None else None
        if k + 1 < len(detecciones):
            idx1, t1, x1, y1, px1, py1 = detecciones[k + 1]
            if 0 < t1 - t <= BALL_GAP_MAX_SECONDS:
                for j in range(idx + 1, idx1):
                    f = (muestras[j]["t"] - t) / (t1 - t)
                    serie[j] = (x + f * (x1 - x), y + f * (y1 - y))
                    if (px is not None and py is not None
                            and px1 is not None and py1 is not None):
                        serie_pitch[j] = (px + f * (px1 - px),
                                          py + f * (py1 - py))
    ema: tuple[float, float] | None = None
    ema_pitch: tuple[float, float] | None = None
    ultimo_t: float | None = None
    for i, m in enumerate(muestras):
        t = m["t"]
        pos = serie.get(i)
        pos_pitch = serie_pitch.get(i)
        if pos is not None:
            ema = pos if ema is None else (
                BALL_EMA_ALPHA * pos[0] + (1 - BALL_EMA_ALPHA) * ema[0],
                BALL_EMA_ALPHA * pos[1] + (1 - BALL_EMA_ALPHA) * ema[1],
            )
            ultimo_t = t
            m["ball_s"] = {"x": ema[0], "y": ema[1]}
            if pos_pitch is not None:
                ema_pitch = pos_pitch if ema_pitch is None else (
                    BALL_EMA_ALPHA * pos_pitch[0] + (1 - BALL_EMA_ALPHA) * ema_pitch[0],
                    BALL_EMA_ALPHA * pos_pitch[1] + (1 - BALL_EMA_ALPHA) * ema_pitch[1],
                )
                m["ball_pitch_s"] = {"x": ema_pitch[0], "y": ema_pitch[1]}
            else:
                m["ball_pitch_s"] = None
        else:
            if ultimo_t is not None and t - ultimo_t > BALL_GAP_MAX_SECONDS:
                ema = None
                ema_pitch = None
            m["ball_s"] = {"x": ema[0], "y": ema[1]} if ema is not None else None
            m["ball_pitch_s"] = ({"x": ema_pitch[0], "y": ema_pitch[1]}
                                 if ema_pitch is not None else None)
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
        # Velocidad en el campo (m/s): la homografía ya absorbe el
        # movimiento de cámara, no necesita compensación. Se capa: más
        # de 126 km/h en 0,5 s es un salto de la H o un balón falso.
        velocidad_ms = 0.0
        if m.get("ball_pitch_s") is not None:
            j = i
            while j > 0 and muestras[j]["t"] > m["t"] - 0.5:
                j -= 1
            anterior = muestras[j]
            if (anterior.get("ball_pitch_s") is not None
                    and anterior["t"] < m["t"]):
                dt = m["t"] - anterior["t"]
                if dt > 0:
                    velocidad_ms = math.hypot(
                        m["ball_pitch_s"]["x"] - anterior["ball_pitch_s"]["x"],
                        m["ball_pitch_s"]["y"] - anterior["ball_pitch_s"]["y"]) / dt
                    velocidad_ms = min(velocidad_ms, PITCH_MAX_SPEED_MS)
        m["ball_speed"] = velocidad_u
        m["ball_speed_bh"] = velocidad_bh
        m["ball_speed_ms"] = velocidad_ms
    return muestras


def _anotar_posesion(muestras: list[dict], team_local: str, team_visitante: str) -> list[dict]:
    """Posesión por muestra: jugador más cercano al balón (en alturas).

    Solo cuentan jugadores de un equipo (el portero cuenta como su equipo);
    el árbitro y los desconocidos no cambian la posesión. Se mantiene la
    última posesión cuando el balón va solo (pase en vuelo).
    """
    posesion_actual = "desconocido"
    for m in muestras:
        if m["ball_scene"] is not None and m["people"]:
            def dist_alturas(p):
                if _equipo_para_posesion(p.get("team")) is None:
                    return float("inf")
                # Posición de la persona también en escena (alturas)
                px = (p["x"] - m["cam_off"][0]) / max(1.0, m["median_height"])
                py = (p["y"] - m["cam_off"][1]) / max(1.0, m["median_height"])
                return math.hypot(px - m["ball_scene"][0], py - m["ball_scene"][1])
            cercana = min(m["people"], key=dist_alturas)
            if dist_alturas(cercana) <= POSSESSION_RADIUS_HEIGHTS:
                posesion_actual = _equipo_para_posesion(cercana["team"])
        m["possession"] = posesion_actual
        m["possession_team_name"] = (
            team_local if posesion_actual == "local"
            else team_visitante if posesion_actual == "visitante"
            else "Desconocido"
        )
    return muestras


def _anotar_porteros(muestras: list[dict]) -> list[dict]:
    """Posición de cada portero en escena (alturas) por muestra.

    Añade 'gk_local_scene' / 'gk_visitante_scene' (tupla o None). La posición
    del portero marca dónde está su portería: en seguimiento es la única
    referencia de qué portería se ataca. Si hay varios con ese color (falso
    positivo), se usa la media.
    """
    for m in muestras:
        for clave, rol in (("gk_local_scene", "gk_local"),
                           ("gk_visitante_scene", "gk_visitante")):
            puntos = []
            for p in m["people"]:
                if p.get("team") != rol:
                    continue
                px = (p["x"] - m["cam_off"][0]) / max(1.0, m["median_height"])
                py = (p["y"] - m["cam_off"][1]) / max(1.0, m["median_height"])
                puntos.append((px, py))
            if puntos:
                m[clave] = (sum(p[0] for p in puntos) / len(puntos),
                            sum(p[1] for p in puntos) / len(puntos))
            else:
                m[clave] = None
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


def _persona_en_escena(m: dict, p: dict) -> tuple[float, float]:
    """Posición de una persona en escena (alturas de jugador)."""
    return ((p["x"] - m["cam_off"][0]) / max(1.0, m["median_height"]),
            (p["y"] - m["cam_off"][1]) / max(1.0, m["median_height"]))


def _portador(m: dict) -> tuple[int, str] | None:
    """Portador del balón en una muestra: (track_id, equipo) o None.

    El jugador de un equipo (el portero cuenta como su equipo; el árbitro
    y los desconocidos no) más cercano al balón, dentro del radio de
    portador. Sin balón en escena no hay portador.
    """
    if m.get("ball_scene") is None:
        return None
    bx, by = m["ball_scene"]
    mejor: tuple[int, str] | None = None
    mejor_d = CARRIER_RADIUS_HEIGHTS
    for p in m.get("people", []):
        equipo = _equipo_para_posesion(p.get("team"))
        if equipo is None:
            continue
        px, py = _persona_en_escena(m, p)
        d = math.hypot(px - bx, py - by)
        if d < mejor_d:
            mejor_d = d
            mejor = (int(p["track_id"]), equipo)
    return mejor


def _detectar_pases_por_trayectoria(muestras: list[dict]) -> list[dict]:
    """Pases por cambio de portador: el balón sale de un jugador y llega
    a un compañero (o a un rival: pase fallido).

    Un pase es: portador A del equipo T (estable ≥0,25 s) -> vuelo sin
    portador (0,2-4,0 s, balón rápido) -> portador B del mismo equipo y
    distinto track ID. Si B es rival, es un pase fallido (pérdida) de T.
    Sin receptor en 4 s no se cuenta (balón fuera, despeje sin destino).
    Devuelve [{"t", "t_llegada", "team", "from_track", "to_track",
    "distancia_bh", "duracion_s", "completado"}, ...] con t relativo al
    saque inicial. Es la base de las estadísticas de pases y secuencias,
    y el clasificador lo usa para etiquetar pases que no son rápidos.
    """
    pases: list[dict] = []
    n = len(muestras)
    if n < 3:
        return pases
    dt = muestras[1]["t"] - muestras[0]["t"]
    if dt <= 0:
        dt = 0.1
    control: tuple[int, str, float] | None = None  # (track, equipo, t_inicio)
    control_estable = 0.0
    vuelo_inicio: float | None = None
    vuelo_salida: tuple[int, str, tuple[float, float]] | None = None
    vuelo_vel_max = 0.0
    for i, m in enumerate(muestras):
        portador = _portador(m)
        if portador is not None:
            if control is not None and portador[0] == control[0]:
                control_estable = m["t"] - control[2]
            else:
                # Nuevo portador: ¿cierra un vuelo (pase) o es control nuevo?
                # (control es None durante el vuelo: el origen del pase
                # está en vuelo_salida, no en control.)
                if vuelo_inicio is not None and vuelo_salida is not None:
                    duracion = m["t"] - vuelo_inicio
                    if (PASS_FLIGHT_MIN_SECONDS <= duracion <= PASS_FLIGHT_MAX_SECONDS
                            and vuelo_vel_max >= PASS_SPEED_MIN_BH):
                        origen = vuelo_salida[2]
                        destino = m["ball_scene"]
                        distancia = (math.hypot(destino[0] - origen[0],
                                                destino[1] - origen[1])
                                     if destino is not None else 0.0)
                        if distancia >= PASS_MIN_DISTANCE_BH:
                            pases.append({
                                "t": round(vuelo_inicio, 2),
                                "t_llegada": round(m["t"], 2),
                                "team": vuelo_salida[1],
                                "from_track": vuelo_salida[0],
                                "to_track": int(portador[0]),
                                "distancia_bh": round(distancia, 1),
                                "duracion_s": round(duracion, 2),
                                "completado": portador[1] == vuelo_salida[1],
                            })
                vuelo_inicio = None
                vuelo_salida = None
                vuelo_vel_max = 0.0
                control = (int(portador[0]), portador[1], m["t"])
                control_estable = 0.0
        else:
            # Sin portador: si había control estable, empieza el vuelo.
            if (control is not None and control_estable >= 0.25
                    and vuelo_inicio is None):
                anterior = muestras[i - 1] if i > 0 else m
                if anterior.get("ball_scene") is not None:
                    vuelo_inicio = anterior["t"]
                    vuelo_salida = (control[0], control[1], anterior["ball_scene"])
                    vuelo_vel_max = 0.0
            if vuelo_inicio is not None:
                vuelo_vel_max = max(vuelo_vel_max, m.get("ball_speed_bh") or 0.0)
                if m["t"] - vuelo_inicio > PASS_FLIGHT_MAX_SECONDS:
                    vuelo_inicio = None
                    vuelo_salida = None
                    vuelo_vel_max = 0.0
            control = None
            control_estable = 0.0
    return pases


def _secuencias_de_pases(pases: list[dict]) -> list[dict]:
    """Cadenas de pases completados seguidos del mismo equipo.

    Dos pases seguidos del mismo equipo con menos de 8 s entre la llegada
    de uno y la salida del siguiente forman parte de la misma secuencia.
    Devuelve [{"team", "inicio_s", "fin_s", "pases", "duracion_s"}, ...].
    """
    completados = sorted((p for p in pases if p["completado"]), key=lambda p: p["t"])
    secuencias: list[dict] = []
    actual: dict | None = None
    for p in completados:
        if (actual is not None and p["team"] == actual["team"]
                and p["t"] - actual["fin_s"] <= SEQUENCE_GAP_SECONDS):
            actual["pases"] += 1
            actual["fin_s"] = p["t_llegada"]
            actual["duracion_s"] = round(p["t_llegada"] - actual["inicio_s"], 1)
        else:
            if actual is not None:
                secuencias.append(actual)
            actual = {"team": p["team"], "inicio_s": p["t"],
                      "fin_s": p["t_llegada"], "pases": 1,
                      "duracion_s": round(p["t_llegada"] - p["t"], 1)}
    if actual is not None:
        secuencias.append(actual)
    return secuencias


def _calcular_estadisticas(
    muestras: list[dict], resultados: list[dict], pases: list[dict],
    secuencias: list[dict], team_local: str, team_visitante: str,
    duracion_partido: float,
) -> dict:
    """Estadísticas del partido a partir del tracking y los cortes.

    Todo son recuentos del tracking (por equipo y por tramo), no
    estimaciones externas: posesión, eventos clasificados, pases por
    trayectoria y secuencias, tiros, duelos, recuperaciones, transiciones,
    balón parado, territorio (proxy por cercanía a cada portero) y balón
    (distancia recorrida, velocidad, % de muestras con balón).
    """
    def _pct(n: int, total: int) -> int:
        return round(100.0 * n / total) if total else 0

    # Posesión total y por tramos.
    conteo_pos = Counter(m.get("possession") for m in muestras)
    total_m = sum(conteo_pos.values()) or 1
    posesion = {
        "local_pct": _pct(conteo_pos.get("local", 0), total_m),
        "visitante_pct": _pct(conteo_pos.get("visitante", 0), total_m),
        "desconocido_pct": _pct(conteo_pos.get("desconocido", 0), total_m),
    }

    # Eventos clasificados: por tipo y por equipo, y por tramo de 15'.
    eventos_por_tipo: dict[str, dict[str, int]] = {}
    eventos_por_tramo: list[dict] = []
    for r in resultados:
        if r["event"] == "uncertain":
            continue
        equipo = "local" if r["team"] == team_local else (
            "visitante" if r["team"] == team_visitante else "desconocido")
        eventos_por_tipo.setdefault(r["event"], {"local": 0, "visitante": 0})
        if equipo in ("local", "visitante"):
            eventos_por_tipo[r["event"]][equipo] += 1
    if resultados and duracion_partido > 0:
        n_tramos = max(1, int(math.ceil(duracion_partido / (TRAMO_MINUTES * 60.0))))
        for k in range(n_tramos):
            inicio = k * TRAMO_MINUTES * 60.0
            fin = min(duracion_partido, (k + 1) * TRAMO_MINUTES * 60.0)
            fila: dict[str, int | str] = {"tramo": f"{int(inicio // 60)}-{int(fin // 60)}'"}
            for r in resultados:
                if r["event"] == "uncertain" or not (inicio <= r["seconds"] < fin):
                    continue
                equipo = "local" if r["team"] == team_local else (
                    "visitante" if r["team"] == team_visitante else None)
                if equipo is None:
                    continue
                clave = f"{r['event']}_{equipo}"
                fila[clave] = int(fila.get(clave, 0)) + 1  # type: ignore[operator]
            eventos_por_tramo.append(fila)  # type: ignore[arg-type]

    # Pases por trayectoria: por equipo, precisión y secuencias.
    pases_stats: dict[str, dict[str, int | float]] = {}
    for equipo in ("local", "visitante"):
        del_equipo = [p for p in pases if p["team"] == equipo]
        ok = [p for p in del_equipo if p["completado"]]
        sec = [s for s in secuencias if s["team"] == equipo]
        pases_stats[equipo] = {
            "total": len(del_equipo),
            "completados": len(ok),
            "precision_pct": _pct(len(ok), len(del_equipo)),
            "distancia_media_bh": (round(sum(p["distancia_bh"] for p in ok) / len(ok), 1)
                                   if ok else 0.0),
            "secuencias": len(sec),
            "secuencia_max_pases": max((s["pases"] for s in sec), default=0),
            "secuencia_media_pases": (round(sum(s["pases"] for s in sec) / len(sec), 1)
                                      if sec else 0.0),
        }

    # Tiros y goles (del clasificador).
    def _cuenta(eventos: tuple[str, ...], equipo: str) -> int:
        return sum(eventos_por_tipo.get(e, {}).get(equipo, 0) for e in eventos)

    tiros = {
        equipo: {
            "goles": _cuenta(("goal",), equipo),
            "tiros_a_puerta": _cuenta(("shot_on_target", "shot"), equipo),
            "tiros_fuera": _cuenta(("shot_off_target",), equipo),
            "tiros_total": _cuenta(("goal", "shot", "shot_on_target",
                                    "shot_off_target"), equipo),
        } for equipo in ("local", "visitante")
    }

    # Territorio (proxy en seguimiento): % de muestras con el balón más
    # cerca de la portería rival que de la propia (tercio de ataque).
    tercio = {"local": 0, "visitante": 0, "total": 0}
    for m in muestras:
        if m.get("ball_scene") is None:
            continue
        d_local = d_visitante = None
        if m.get("gk_local_scene") is not None:
            d_local = math.hypot(m["ball_scene"][0] - m["gk_local_scene"][0],
                                 m["ball_scene"][1] - m["gk_local_scene"][1])
        if m.get("gk_visitante_scene") is not None:
            d_visitante = math.hypot(m["ball_scene"][0] - m["gk_visitante_scene"][0],
                                     m["ball_scene"][1] - m["gk_visitante_scene"][1])
        if d_local is None or d_visitante is None:
            continue
        tercio["total"] += 1
        if d_visitante < d_local:
            tercio["local"] += 1
        else:
            tercio["visitante"] += 1
    territorio = {
        "local_pct": _pct(tercio["local"], tercio["total"]),
        "visitante_pct": _pct(tercio["visitante"], tercio["total"]),
        "muestras_con_porteros": tercio["total"],
    }

    # Balón: distancia recorrida, velocidad y cobertura de detección.
    distancia = 0.0
    velocidades = []
    con_balon = 0
    personas_total = 0
    # En metros de campo (homografía): distancia real, velocidad en km/h
    # y % de muestras con coordenadas de campo.
    distancia_m = 0.0
    velocidades_ms = []
    con_campo = 0
    tercios = {"izquierda": 0, "centro": 0, "derecha": 0, "total": 0}
    for i, m in enumerate(muestras):
        if m.get("ball_scene") is not None:
            con_balon += 1
            velocidades.append(m.get("ball_speed_bh") or 0.0)
            if i > 0 and muestras[i - 1].get("ball_scene") is not None:
                distancia += math.hypot(
                    m["ball_scene"][0] - muestras[i - 1]["ball_scene"][0],
                    m["ball_scene"][1] - muestras[i - 1]["ball_scene"][1])
        if m.get("ball_pitch_s") is not None:
            con_campo += 1
            velocidades_ms.append(m.get("ball_speed_ms") or 0.0)
            if i > 0 and muestras[i - 1].get("ball_pitch_s") is not None:
                paso_m = math.hypot(
                    m["ball_pitch_s"]["x"] - muestras[i - 1]["ball_pitch_s"]["x"],
                    m["ball_pitch_s"]["y"] - muestras[i - 1]["ball_pitch_s"]["y"])
                dt_paso = m["t"] - muestras[i - 1]["t"]
                # Un salto más rápido que el máximo físico es un salto de
                # la homografía, no distancia recorrida: no se acumula.
                if dt_paso > 0 and paso_m / dt_paso <= PITCH_MAX_SPEED_MS:
                    distancia_m += paso_m
            x_campo = m["ball_pitch_s"]["x"]
            tercios["total"] += 1
            if x_campo < PITCH_LENGTH_M / 3:
                tercios["izquierda"] += 1
            elif x_campo > 2 * PITCH_LENGTH_M / 3:
                tercios["derecha"] += 1
            else:
                tercios["centro"] += 1
        personas_total += len(m.get("people", []))
    # Máxima como el mayor valor SIN capar: el máximo puro suele ser el
    # tope de 126 km/h (un salto de la homografía), no un tiro real.
    _v_sin_tope = [v for v in velocidades_ms if v < PITCH_MAX_SPEED_MS]
    balon = {
        "distancia_recorrida_bh": round(distancia, 0),
        "velocidad_media_bh": (round(sum(velocidades) / len(velocidades), 2)
                               if velocidades else 0.0),
        "velocidad_max_bh": round(max(velocidades), 1) if velocidades else 0.0,
        "muestras_con_balon_pct": _pct(con_balon, len(muestras)),
        "personas_medias_por_muestra": (round(personas_total / len(muestras), 1)
                                        if muestras else 0.0),
        "distancia_recorrida_m": round(distancia_m, 0),
        "velocidad_media_kmh": (round(3.6 * sum(velocidades_ms) / len(velocidades_ms), 1)
                                if velocidades_ms else 0.0),
        "velocidad_max_kmh": (round(3.6 * max(_v_sin_tope), 1)
                              if _v_sin_tope else 0.0),
        "muestras_con_campo_pct": _pct(con_campo, len(muestras)),
    }
    # Distancia media de los pases en metros (posición del balón en campo
    # al salir y al llegar).
    tiempos = [m["t"] for m in muestras]
    for equipo in ("local", "visitante"):
        del_equipo = [p for p in pases if p["team"] == equipo and p["completado"]]
        dists_m = []
        for p in del_equipo:
            m0 = _muestra_en(muestras, tiempos, p["t"])
            m1 = _muestra_en(muestras, tiempos, p["t_llegada"])
            if (m0 is not None and m1 is not None
                    and m0.get("ball_pitch_s") is not None
                    and m1.get("ball_pitch_s") is not None):
                dists_m.append(math.hypot(
                    m1["ball_pitch_s"]["x"] - m0["ball_pitch_s"]["x"],
                    m1["ball_pitch_s"]["y"] - m0["ball_pitch_s"]["y"]))
        pases_stats[equipo]["distancia_media_m"] = (
            round(sum(dists_m) / len(dists_m), 1) if dists_m else 0.0)

    return {
        "duracion_partido_s": round(duracion_partido, 1),
        "muestras": len(muestras),
        "posesion": posesion,
        "posesion_por_tramos": posesion_por_tramos(muestras, team_local, team_visitante),
        "eventos_por_tipo": eventos_por_tipo,
        "eventos_por_tramo": eventos_por_tramo,
        "pases": pases_stats,
        "tiros": tiros,
        "territorio": territorio,
        "tercios_campo": ({
            "izquierda_pct": _pct(tercios["izquierda"], tercios["total"]),
            "centro_pct": _pct(tercios["centro"], tercios["total"]),
            "derecha_pct": _pct(tercios["derecha"], tercios["total"]),
        } if tercios["total"] else None),
        "balon": balon,
        "duelos": {e: _cuenta(("duel",), e) for e in ("local", "visitante")},
        "recuperaciones": {e: _cuenta(("recovery",), e) for e in ("local", "visitante")},
        "perdidas": {e: _cuenta(("loss",), e) for e in ("local", "visitante")},
        "transiciones": {e: _cuenta(("transition", "counterattack"), e)
                         for e in ("local", "visitante")},
        "centros": {e: _cuenta(("cross",), e) for e in ("local", "visitante")},
        "corners": {e: _cuenta(("corner",), e) for e in ("local", "visitante")},
        "balon_parado": {e: _cuenta(("set_piece", "foul"), e) for e in ("local", "visitante")},
        "distribuciones_portero": {e: _cuenta(("gk_distribution",), e)
                                   for e in ("local", "visitante")},
        "entradas_area": {e: _cuenta(("area_entry", "key_pass"), e)
                          for e in ("local", "visitante")},
    }


def _exportar_informe(informe: dict, estadisticas: dict, pases: list[dict],
                      video_path: Path, output_dir: Path) -> dict:
    """Escribe el informe profesional del partido: markdown + 2 CSV.

    ``{stem}_informe.md``: resumen con tablas (posesión, eventos, pases,
    tiros, territorio, balón) y lista de cortes. ``{stem}_eventos.csv``:
    un fila por corte. ``{stem}_estadisticas.csv``: métrica por equipo.
    Devuelve {"informe_md", "eventos_csv", "estadisticas_csv"} (rutas str).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    est = estadisticas
    team_local = informe.get("team_local", "Local")
    team_visitante = informe.get("team_visitante", "Visitante")
    resultados = informe.get("results", [])
    nombre_md = output_dir / f"{video_path.stem}_informe.md"
    nombre_ev = output_dir / f"{video_path.stem}_eventos.csv"
    nombre_est = output_dir / f"{video_path.stem}_estadisticas.csv"

    def _fila_eventos(evento: str) -> str:
        e = est["eventos_por_tipo"].get(evento, {})
        return f"| {EVENT_NAMES_ES.get(evento, evento)} | {e.get('local', 0)} | {e.get('visitante', 0)} |"

    lineas = [
        f"# Informe de vídeo: {team_local} vs {team_visitante}",
        "",
        f"- Duración analizada: {format_mmss(est['duracion_partido_s'])}",
        f"- Saque inicial en el vídeo: {format_mmss(informe.get('kickoff_offset_s', 0.0))}",
        f"- Muestras YOLO: {est['muestras']} · Muestras con balón: "
        f"{est['balon']['muestras_con_balon_pct']}%",
        "",
        "## Posesión",
        "",
        f"- **{team_local}: {est['posesion']['local_pct']}%** · "
        f"**{team_visitante}: {est['posesion']['visitante_pct']}%** · "
        f"Sin asignar: {est['posesion']['desconocido_pct']}%",
        "",
        "| Tramo | " + team_local + " | " + team_visitante + " | Sin asignar |",
        "|---|---|---|---|",
    ]
    for tramo in est["posesion_por_tramos"]:
        lineas.append(f"| {tramo['tramo']} | {tramo['local_pct']}% | "
                      f"{tramo['visitante_pct']}% | {tramo['desconocido_pct']}% |")
    lineas += [
        "",
        "## Eventos",
        "",
        f"| Evento | {team_local} | {team_visitante} |",
        "|---|---|---|",
    ]
    for evento in ("goal", "shot_on_target", "shot_off_target", "shot", "key_pass",
                   "pass", "cross", "corner", "foul", "duel", "recovery", "loss",
                   "transition", "counterattack", "area_entry", "set_piece",
                   "gk_distribution"):
        if evento in est["eventos_por_tipo"]:
            lineas.append(_fila_eventos(evento))
    p_loc, p_vis = est["pases"]["local"], est["pases"]["visitante"]
    t_loc, t_vis = est["tiros"]["local"], est["tiros"]["visitante"]
    lineas += [
        "",
        "## Pases (por trayectoria: cambio de portador)",
        "",
        f"| | {team_local} | {team_visitante} |",
        "|---|---|---|",
        f"| Pases intentados | {p_loc['total']} | {p_vis['total']} |",
        f"| Pases completados | {p_loc['completados']} | {p_vis['completados']} |",
        f"| Precisión | {p_loc['precision_pct']}% | {p_vis['precision_pct']}% |",
        f"| Distancia media (alturas) | {p_loc['distancia_media_bh']} | {p_vis['distancia_media_bh']} |",
        f"| Secuencias | {p_loc['secuencias']} | {p_vis['secuencias']} |",
        f"| Secuencia más larga (pases) | {p_loc['secuencia_max_pases']} | {p_vis['secuencia_max_pases']} |",
        f"| Secuencia media (pases) | {p_loc['secuencia_media_pases']} | {p_vis['secuencia_media_pases']} |",
        "",
        "## Tiros",
        "",
        f"| | {team_local} | {team_visitante} |",
        "|---|---|---|",
        f"| Goles | {t_loc['goles']} | {t_vis['goles']} |",
        f"| Tiros a puerta | {t_loc['tiros_a_puerta']} | {t_vis['tiros_a_puerta']} |",
        f"| Tiros fuera | {t_loc['tiros_fuera']} | {t_vis['tiros_fuera']} |",
        f"| Tiros totales | {t_loc['tiros_total']} | {t_vis['tiros_total']} |",
        "",
        "## Territorio y balón",
        "",
        f"- Tercio de ataque (balón más cerca de la portería rival): "
        f"{team_local} {est['territorio']['local_pct']}% · "
        f"{team_visitante} {est['territorio']['visitante_pct']}% "
        f"(con ambos porteros visibles en {est['territorio']['muestras_con_porteros']} muestras)",
        f"- Distancia recorrida por el balón: {est['balon']['distancia_recorrida_bh']} alturas",
        f"- Velocidad media del balón: {est['balon']['velocidad_media_bh']} alturas/s · "
        f"máxima: {est['balon']['velocidad_max_bh']} alturas/s",
        "",
        "## Cortes",
        "",
        "| Minuto | Evento | Equipo | Confianza | Resumen |",
        "|---|---|---|---|---|",
    ]
    for r in resultados:
        lineas.append(f"| {format_mmss(r['seconds'])} | "
                      f"{EVENT_NAMES_ES.get(r['event'], r['event'])} | {r['team']} | "
                      f"{int(r['confidence'] * 100)}% | {r['summary']} |")
    nombre_md.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    with open(nombre_ev, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["minuto", "segundo", "evento", "equipo", "confianza",
                    "resumen", "zona", "clip"])
        for r in resultados:
            w.writerow([r["minute"], r["seconds"], r["event"], r["team"],
                        r["confidence"], r["summary"], r["zone"],
                        r["clip_path"] or ""])
    with open(nombre_est, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", team_local, team_visitante])
        w.writerow(["posesion_pct", est["posesion"]["local_pct"],
                    est["posesion"]["visitante_pct"]])
        for evento, vals in sorted(est["eventos_por_tipo"].items()):
            w.writerow([f"eventos_{evento}", vals.get("local", 0),
                        vals.get("visitante", 0)])
        for clave in ("total", "completados", "precision_pct",
                      "distancia_media_bh", "secuencias", "secuencia_max_pases",
                      "secuencia_media_pases"):
            w.writerow([f"pases_{clave}", p_loc[clave], p_vis[clave]])
        for clave in ("goles", "tiros_a_puerta", "tiros_fuera", "tiros_total"):
            w.writerow([f"tiros_{clave}", t_loc[clave], t_vis[clave]])
        w.writerow(["territorio_ataque_pct", est["territorio"]["local_pct"],
                    est["territorio"]["visitante_pct"]])
        w.writerow(["balon_distancia_bh", est["balon"]["distancia_recorrida_bh"], ""])
        w.writerow(["balon_velocidad_media_bh", est["balon"]["velocidad_media_bh"], ""])
        w.writerow(["balon_velocidad_max_bh", est["balon"]["velocidad_max_bh"], ""])
    return {"informe_md": str(nombre_md), "eventos_csv": str(nombre_ev),
            "estadisticas_csv": str(nombre_est)}


def _detectar_candidatos(muestras: list[dict], camera_mode: str) -> list[dict]:
    """Candidatos a evento desde señales relativas del tracking.

    Señales (robustas a pan/zoom porque usan velocidad relativa y distancias
    en alturas de jugador, no posiciones absolutas en el campo):
    pico de velocidad, parada brusca, agrupación junto al balón, balón fuera
    de plano, balón parado + agrupación (balón parado), saque del portero,
    cambio de posesión + avance rápido (transición), pase largo rápido,
    cambio de posesión solo (recuperación / duelo), balón parado largo
    (≥3 s), entrada al área (solo fija_amplia) y balón cerca del portero
    (zona de portería: tiro / gol).
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
        # Agrupación junto al balón (en alturas; el árbitro no cuenta).
        if m["ball_scene"] is not None:
            n_cerca = 0
            for p in personas:
                if p.get("team") == "arbitro":
                    continue
                px = (p["x"] - m["cam_off"][0]) / max(1.0, m["median_height"])
                py = (p["y"] - m["cam_off"][1]) / max(1.0, m["median_height"])
                if math.hypot(px - m["ball_scene"][0], py - m["ball_scene"][1]) <= CROWD_RADIUS_HEIGHTS:
                    n_cerca += 1
            if n_cerca >= CROWD_MIN_PEOPLE:
                senales.append(f"agrupación de {n_cerca} jugadores junto al balón")
                score += 2.0
        # Balón cerca de un portero: la jugada está en zona de portería
        # (tiro a puerta, gol, o balón al área). La posición del portero
        # marca qué portería se ataca.
        if m["ball_scene"] is not None:
            for clave_gk in ("gk_local_scene", "gk_visitante_scene"):
                gk = m.get(clave_gk)
                if gk is not None and math.hypot(
                        m["ball_scene"][0] - gk[0],
                        m["ball_scene"][1] - gk[1]) <= GK_NEAR_HEIGHTS:
                    senales.append("balón cerca del portero")
                    score += 2.0
                    break
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
        # Cambio de posesión solo (sin avance rápido): recuperación / duelo.
        if (hace_3s is not None
                and hace_3s["possession"] in ("local", "visitante")
                and m["possession"] in ("local", "visitante")
                and hace_3s["possession"] != m["possession"]):
            senales.append("cambio de posesión")
            score += 2.0
        # Balón parado largo (≥3 s): balón parado / falta / saque.
        # Solo cuentan muestras con el balón VISTO y parado: cuando se
        # pierde el balón ball_speed_bh es 0 por defecto, y sin este
        # chequeo cualquier pérdida de balón de 3 s salía como "parado".
        parado = 0.0
        k = i
        while (k > 0 and muestras[k - 1].get("ball_scene") is not None
               and muestras[k - 1]["ball_speed_bh"] <= STOP_TO_BH):
            parado += dt
            k -= 1
            if parado >= 3.0:
                break
        if parado >= 3.0:
            senales.append("balón parado (≥3 s)")
            score += 1.5
        # Entrada al área (solo fija_amplia): del tercio medio al tercio final.
        if (camera_mode == "fija_amplia" and hace_3s is not None
                and m["ball_s"] is not None and hace_3s.get("ball_s") is not None):
            x_antes = hace_3s["ball_s"]["x"]
            x_ahora = m["ball_s"]["x"]
            if 17.0 < x_antes < 83.0 and (x_ahora >= 83.0 or x_ahora <= 17.0):
                senales.append("entrada al área")
                score += 2.0
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


def _limitar_faltas(resultados: list[dict],
                    max_fouls: int = MAX_FOUL_CUTS) -> list[dict]:
    """Tope de faltas por partido: si hay más de max_fouls, conserva las
    de mayor confianza y convierte el resto en duelo (lo que suele ser un
    falso positivo de falta: disputa con parada y cambio de posesión).
    Se aplica DESPUÉS de clasificar y ANTES de estadísticas/informe, y
    renombra el clip si ya se había exportado como falta.
    """
    faltas = [r for r in resultados if r.get("event") == "foul"]
    if len(faltas) <= max_fouls:
        return resultados
    ordenadas = sorted(faltas, key=lambda r: r.get("confidence", 0.0),
                       reverse=True)
    for r in ordenadas[max_fouls:]:
        r["event"] = "duel"
        r["confidence"] = 0.60
        r["summary"] = f"Duelo ganado por el {r.get('team', 'Desconocido')}"
        clip = r.get("clip_path")
        if clip and "_foul" in clip:
            nuevo = clip.replace("_foul", "_duel")
            try:
                import os
                os.rename(clip, nuevo)
                r["clip_path"] = nuevo
            except OSError:
                pass
    return resultados


# ---------------------------------------------------------------------------
# Clasificación por tracking (sin Groq)
# ---------------------------------------------------------------------------
def _muestra_en(muestras: list[dict], tiempos: list[float], t: float) -> dict | None:
    """Muestra más cercana al instante t (o None si no hay muestras)."""
    import bisect
    if not muestras:
        return None
    i = bisect.bisect_left(tiempos, t)
    if i >= len(muestras):
        i = len(muestras) - 1
    elif i > 0 and abs(tiempos[i - 1] - t) < abs(tiempos[i] - t):
        i -= 1
    return muestras[i]


def _posesion_mayoritaria(muestras: list[dict], tiempos: list[float],
                          t0: float, t1: float) -> str:
    """Posesión mayoritaria ('local'/'visitante'/'desconocido') entre t0 y t1."""
    import bisect
    i0 = bisect.bisect_left(tiempos, t0)
    i1 = bisect.bisect_right(tiempos, t1)
    conteo = Counter(m["possession"] for m in muestras[i0:i1])
    if not conteo:
        return "desconocido"
    return conteo.most_common(1)[0][0]


def _clasificar_por_tracking(cand: dict, muestras: list[dict], camera_mode: str,
                             team_local: str, team_visitante: str,
                             pases: list[dict] | None = None) -> dict:
    """Clasifica un candidato solo con el tracking YOLO.

    Ventana de ±3 s alrededor del candidato: velocidad del balón, si se
    pierde el balón tras la jugada, cuánto llevaba parado, si cambia la
    posesión, cuánta gente hay junto al balón, cuánto avanza, distancia a
    cada portero (su posición marca qué portería se ataca) y (solo en
    modo fija_amplia) si cruza la línea de gol o entra al área.

    Devuelve event/confidence/summary/team, como antes devolvía Groq.
    La confianza es heurística y nunca pasa de 0,80: sin ver la jugada no
    se puede asegurar un gol o un córner. Con mapa del campo (homografía,
    muestras con ball_pitch_s) el gol se detecta por cruce de la línea,
    el tiro exige que el balón vaya hacia la portería rival (un despeje
    o un saque de portero no es tiro) y el córner, el centro y la entrada
    al área usan la posición real en metros. Sin mapa, en seguimiento el
    gol se etiqueta cuando un tiro rápido se pierde ≥4 s y el juego se
    reanuda en el centro (saque de centro: balón parado con jugadores de
    los dos equipos alrededor y lejos de cualquier portero); si reaparece
    cerca del portero es parada o saque de puerta (tiro). La falta exige
    balón parado después (se prepara el lanzamiento) y árbitro cerca, y
    hay un tope de 5 faltas por partido (_limitar_faltas).
    """
    import bisect
    t = float(cand["seconds"])
    tiempos = [m["t"] for m in muestras]
    if not tiempos:
        return {"event": "uncertain", "confidence": 0.0,
                "summary": "Sin muestras de tracking.", "team": "Desconocido"}
    fija = camera_mode == "fija_amplia"

    def _res(event: str, confidence: float, summary: str, team: str) -> dict:
        return {"event": event, "confidence": confidence,
                "summary": summary, "team": team}

    m0 = _muestra_en(muestras, tiempos, t)
    m_antes = _muestra_en(muestras, tiempos, t - 2.0)
    m_despues = _muestra_en(muestras, tiempos, t + 2.0)
    posesion_antes = _posesion_mayoritaria(muestras, tiempos, t - 3.0, t - 0.5)
    posesion_despues = _posesion_mayoritaria(muestras, tiempos, t + 0.5, t + 3.0)
    cambio_posesion = (posesion_antes in ("local", "visitante")
                       and posesion_despues in ("local", "visitante")
                       and posesion_antes != posesion_despues)
    equipo = (team_local if posesion_despues == "local"
              else team_visitante if posesion_despues == "visitante"
              else cand.get("team") or "Desconocido")

    # Velocidad máxima del balón en [-1, +2] s y velocidad tras la jugada.
    i0 = bisect.bisect_left(tiempos, t - 1.0)
    i1 = bisect.bisect_right(tiempos, t + 2.0)
    ventana = muestras[i0:i1]
    vel_max = max((m.get("ball_speed_bh") or 0.0) for m in ventana) if ventana else 0.0
    m_despues_1 = _muestra_en(muestras, tiempos, t + 1.5)
    vel_despues = (m_despues_1 or {}).get("ball_speed_bh") or 0.0

    # Velocidad máxima en una ventana amplia [-8, +2] s (para el gol):
    # el candidato de un gol se detecta en la pérdida o en la
    # celebración, varios segundos DESPUÉS del tiro, así que el pico de
    # velocidad del tiro queda fuera de [-1, +2]. Se guarda también el
    # instante del pico: el equipo que tira es el que tenía el balón
    # justo antes de ese pico (el que dispara), no el que tiene el balón
    # en el saque de centro posterior (el que encaja el gol: en el gol
    # del Cirbonero ~06:07 la posesión en el saque era del Milagres).
    i0g = bisect.bisect_left(tiempos, t - 8.0)
    i1g = bisect.bisect_right(tiempos, t + 2.0)
    ventana_gol = muestras[i0g:i1g]
    vel_max_gol = (max((m.get("ball_speed_bh") or 0.0) for m in ventana_gol)
                   if ventana_gol else 0.0)
    t_vel_max_gol = t
    if vel_max_gol > 0:
        for m in ventana_gol:
            if (m.get("ball_speed_bh") or 0.0) >= vel_max_gol:
                t_vel_max_gol = m["t"]
                break
    posesion_tiro_gol = _posesion_mayoritaria(
        muestras, tiempos, t_vel_max_gol - 3.0, t_vel_max_gol - 0.5)

    # Cuánto llevaba parado el balón antes del candidato. Solo cuentan
    # muestras con el balón VISTO y parado: cuando se pierde, la velocidad
    # es 0 por defecto y sin este chequeo una pérdida larga salía como
    # "parado" (misma corrección que en _detectar_candidatos).
    parado_s = 0.0
    dt = tiempos[1] - tiempos[0] if len(tiempos) > 1 else 0.1
    j = bisect.bisect_left(tiempos, t)
    while j > 0:
        if muestras[j - 1].get("ball_scene") is None:
            break
        v_parado = muestras[j - 1].get("ball_speed_bh")
        if v_parado is None or v_parado > STOP_TO_BH:
            break
        parado_s += dt
        j -= 1

    # ¿Se pierde el balón tras el candidato? (fracción de muestras sin balón)
    k0 = bisect.bisect_left(tiempos, t + 0.5)
    k1 = bisect.bisect_right(tiempos, t + 3.0)
    tramo_despues = muestras[k0:k1]
    perdido = (sum(1 for m in tramo_despues if m.get("ball_scene") is None)
               / max(1, len(tramo_despues))) if tramo_despues else 0.0

    # Gente junto al balón en el instante del candidato (el árbitro no cuenta).
    agrupacion = 0
    if m0 is not None and m0.get("ball_scene") is not None:
        bx, by = m0["ball_scene"]
        for p in m0["people"]:
            if p.get("team") == "arbitro":
                continue
            px = (p["x"] - m0["cam_off"][0]) / max(1.0, m0["median_height"])
            py = (p["y"] - m0["cam_off"][1]) / max(1.0, m0["median_height"])
            if math.hypot(px - bx, py - by) <= CROWD_RADIUS_HEIGHTS:
                agrupacion += 1

    # Avance del balón en la ventana [-2, +2] s (en alturas de jugador).
    avance = 0.0
    if (m_antes is not None and m_despues is not None
            and m_antes.get("ball_scene") is not None
            and m_despues.get("ball_scene") is not None):
        avance = math.hypot(m_despues["ball_scene"][0] - m_antes["ball_scene"][0],
                            m_despues["ball_scene"][1] - m_antes["ball_scene"][1])

    # Posición en campo (solo fiable en fija_amplia: X = eje de ataque,
    # 0 portería local, 100 visitante; Y = ancho).
    def _xy(m):
        if m is None or m.get("ball_s") is None:
            return None
        return m["ball_s"]["x"], m["ball_s"]["y"]

    xy_antes, xy_0, xy_despues = _xy(m_antes), _xy(m0), _xy(m_despues)

    def _en_area(xy):
        return xy is not None and (xy[0] >= 83.0 or xy[0] <= 17.0)

    def _cerca_linea_gol(xy):
        return xy is not None and (xy[0] >= 95.0 or xy[0] <= 5.0)

    def _en_banda(xy):
        return xy is not None and (xy[1] <= 15.0 or xy[1] >= 85.0)

    cruza_gol = False
    if fija and xy_antes is not None and xy_despues is not None:
        for limite in (97.0, 3.0):
            if (xy_antes[0] - limite) * (xy_despues[0] - limite) < 0:
                cruza_gol = True
    entra_area = fija and not _en_area(xy_antes) and _en_area(xy_despues)

    # Posición en metros de campo (homografía): lo mismo que fija_amplia
    # pero vale también en seguimiento cuando hay mapa del campo. La
    # dirección del balón (hacia qué portería va) es lo que distingue un
    # tiro de un despeje o un pase atrás: sin campo, cualquier balón
    # rápido que se pierde es "tiro" (el error del 08:30, saque del
    # portero verde, y de los ataques por banda).
    def _xy_campo(m):
        if m is None or m.get("ball_pitch_s") is None:
            return None
        return m["ball_pitch_s"]["x"], m["ball_pitch_s"]["y"]

    xy_campo_antes, xy_campo_0, xy_campo_despues = (
        _xy_campo(m_antes), _xy_campo(m0), _xy_campo(m_despues))
    hay_campo = any(m.get("ball_pitch_s") is not None for m in ventana)
    cruza_gol_campo = _cruza_linea_gol(xy_campo_antes, xy_campo_despues)
    entra_area_campo = (not _en_area_campo(xy_campo_antes)
                        and _en_area_campo(xy_campo_despues))
    # Velocidad del balón en el campo (vector): el par de muestras
    # consecutivas con mayor velocidad en la ventana [-1, +2] s.
    vx_campo = 0.0
    vel_campo_max = 0.0
    for a, b in zip(ventana, ventana[1:]):
        if a.get("ball_pitch_s") is None or b.get("ball_pitch_s") is None:
            continue
        dt_v = b["t"] - a["t"]
        if dt_v <= 0:
            continue
        vx = (b["ball_pitch_s"]["x"] - a["ball_pitch_s"]["x"]) / dt_v
        vy = (b["ball_pitch_s"]["y"] - a["ball_pitch_s"]["y"]) / dt_v
        v = math.hypot(vx, vy)
        if v > vel_campo_max:
            vel_campo_max, vx_campo = v, vx
    # Portería atacada: la del portero RIVAL (su x en metros en la
    # ventana). Se resuelve después de calcular gk_cercano (abajo).

    # Distancia del balón a cada portero (en alturas). La posición del
    # portero marca su portería: el equipo que ataca es el contrario al
    # del portero más cercano. En seguimiento es la única referencia de
    # qué portería se ataca.
    def _dist_portero(m, clave):
        gk = m.get(clave) if m is not None else None
        if m is None or m.get("ball_scene") is None or gk is None:
            return None
        return math.hypot(m["ball_scene"][0] - gk[0], m["ball_scene"][1] - gk[1])

    dist_gk_local = _dist_portero(m0, "gk_local_scene")
    dist_gk_visitante = _dist_portero(m0, "gk_visitante_scene")
    dist_gk_antes = min(
        [d for d in (_dist_portero(m_antes, "gk_local_scene"),
                     _dist_portero(m_antes, "gk_visitante_scene"))
         if d is not None], default=None)
    dist_gk_despues = min(
        [d for d in (_dist_portero(m_despues, "gk_local_scene"),
                     _dist_portero(m_despues, "gk_visitante_scene"))
         if d is not None], default=None)
    if dist_gk_local is not None and (dist_gk_visitante is None
                                      or dist_gk_local <= dist_gk_visitante):
        gk_cercano, dist_gk = "gk_local", dist_gk_local
        equipo_atacante = team_visitante
    elif dist_gk_visitante is not None:
        gk_cercano, dist_gk = "gk_visitante", dist_gk_visitante
        equipo_atacante = team_local
    else:
        gk_cercano, dist_gk, equipo_atacante = None, None, equipo
    cerca_portero = dist_gk is not None and dist_gk <= GK_NEAR_HEIGHTS
    en_zona_portero = (dist_gk_despues is not None
                       and dist_gk_despues <= GK_AREA_HEIGHTS)
    entra_zona_portero = (dist_gk_antes is not None
                          and dist_gk_antes > GK_AREA_HEIGHTS
                          and en_zona_portero)
    # El equipo que ataca la portería del portero cercano es el contrario.
    # Un balón rápido junto al portero solo es tiro/gol si lo mueve el
    # equipo atacante: un pase atrás al propio portero no es un tiro.
    atacante_codigo = ("visitante" if gk_cercano == "gk_local"
                       else "local" if gk_cercano == "gk_visitante" else None)
    ataca = (atacante_codigo is not None
             and posesion_antes in ("desconocido", atacante_codigo))

    # Portería atacada en metros (para la dirección del tiro): la del
    # portero RIVAL, que es el del equipo contrario al que tenía el balón
    # antes (posesion_antes). No el portero más cercano: en el centro
    # del campo el más cercano es arbitrario y vetaba tiros buenos.
    def _x_portero(clave_equipo):
        xs = []
        for m in ventana:
            for p in m.get("people", []):
                if p.get("team") == clave_equipo and p.get("pitch_x") is not None:
                    xs.append(p["pitch_x"])
        if not xs:
            return None
        xs.sort()
        return xs[len(xs) // 2]

    clave_gk_rival = ({"local": "gk_visitante",
                       "visitante": "gk_local"}.get(posesion_antes)
                      or gk_cercano)
    x_gk_rival = _x_portero(clave_gk_rival) if (hay_campo and clave_gk_rival) else None
    # Dirección del tiro: None si no se sabe (sin portero rival visible
    # o sin velocidad en campo); solo veta un tiro cuando SE SABE que el
    # balón va hacia la portería propia (despeje) o de lado (centro).
    hacia_porteria = None
    if x_gk_rival is not None and vel_campo_max > 0:
        hacia_porteria = ((x_gk_rival > PITCH_LENGTH_M / 2 and vx_campo > 2.0)
                          or (x_gk_rival < PITCH_LENGTH_M / 2 and vx_campo < -2.0))
    # Equipo que tira: el que tenía el balón justo antes del tiro (el
    # que dispara), no el que ataca según el portero cercano: en el gol
    # del Cirbonero ~06:07 la posesión antes era del Cirbonero y el
    # equipo_atacante salía Milagres.
    equipo_tiro = (team_local if posesion_antes == "local"
                   else team_visitante if posesion_antes == "visitante"
                   else equipo_atacante)
    # Para el gol se usa la posesión en el pico de velocidad (el tiro),
    # que puede ser segundos antes del candidato (el candidato se
    # detecta en la pérdida/celebración posterior).
    equipo_tiro_gol = (team_local if posesion_tiro_gol == "local"
                       else team_visitante if posesion_tiro_gol == "visitante"
                       else equipo_tiro)
    senales = " ".join(cand.get("signals", []))

    # Balón parado DESPUÉS del candidato: TIEMPO TOTAL (no racha) con el
    # balón VISTO y parado en (t, t+6 s], y tiempo total con el balón
    # PERDIDO en esa ventana. Tras una falta el juego se para: el balón
    # queda en el suelo para el lanzamiento o fuera de plano mientras se
    # prepara; tras un duelo sigue el juego. Una racha no sirve porque en
    # seguimiento el balón se pierde a menudo y un solo hueco la resetea.
    parado_despues = 0.0
    perdido_despues = 0.0
    for m in muestras[bisect.bisect_left(tiempos, t):
                      bisect.bisect_right(tiempos, t + 6.0)]:
        if m.get("ball_scene") is None:
            perdido_despues += dt
        elif (m.get("ball_speed_bh") or 0.0) <= STOP_TO_BH:
            parado_despues += dt
    juego_parado = (parado_despues >= FOUL_STOP_AFTER_SECONDS
                    or perdido_despues >= FOUL_LOST_AFTER_SECONDS)
    # Frenazo en el candidato: balón rápido justo antes y parado (o
    # perdido) justo después. No usa la señal "parada brusca": el
    # candidato agrega señales de ±8 s y ese pico puede ser de otra jugada.
    frenazo = vel_max >= STOP_FROM_BH and vel_despues <= STOP_TO_BH

    # Pase atrás al propio portero: el balón acaba junto al portero del
    # equipo que tiene la posesión después (alguna muestra vista en
    # (t+1, t+4 s]). Eso es posesión, no tiro (un tiro parado acaba junto
    # al portero RIVAL, no al propio). Se mira desde t+1 s porque en t el
    # balón aún está en los pies del que pasa.
    pase_atras = False
    clave_gk_propio = ({"local": "gk_local_scene",
                        "visitante": "gk_visitante_scene"}.get(posesion_despues))
    if clave_gk_propio is not None:
        for m in muestras[bisect.bisect_left(tiempos, t + 1.0):
                          bisect.bisect_right(tiempos, t + 4.0)]:
            if m.get("ball_scene") is None:
                continue
            d_propio = _dist_portero(m, clave_gk_propio)
            if d_propio is not None and d_propio <= GK_AREA_HEIGHTS:
                pase_atras = True
                break
    # Con mapa: el balón que acaba junto al portero PROPIO en metros es
    # pase atrás aunque el portero no se vea en el plano (el guard de
    # plano necesita la escena del portero en la muestra). Se usa la
    # posición en campo del portero propio (mediana en la ventana) y el
    # balón en (t+1, t+4 s]: el pase atrás del 09:49 salía como pase
    # clave porque el balón acababa en el área en metros.
    if hay_campo and not pase_atras:
        clave_gk_propio_campo = ({"local": "gk_local",
                                  "visitante": "gk_visitante"}.get(posesion_despues))
        if clave_gk_propio_campo is not None:
            xs_gk, ys_gk = [], []
            for m in muestras[bisect.bisect_left(tiempos, t - 1.0):
                              bisect.bisect_right(tiempos, t + 4.0)]:
                for p in m.get("people", []):
                    if (p.get("team") == clave_gk_propio_campo
                            and p.get("pitch_x") is not None):
                        xs_gk.append(p["pitch_x"])
                        ys_gk.append(p["pitch_y"])
            if xs_gk:
                xs_gk.sort()
                ys_gk.sort()
                gx = xs_gk[len(xs_gk) // 2]
                gy = ys_gk[len(ys_gk) // 2]
                for m in muestras[bisect.bisect_left(tiempos, t + 1.0):
                                  bisect.bisect_right(tiempos, t + 4.0)]:
                    bp = m.get("ball_pitch_s")
                    if bp is not None and math.hypot(
                            bp["x"] - gx, bp["y"] - gy) <= 12.0:
                        pase_atras = True
                        break

    # Distancia del árbitro al balón en ±2 s (mínimo en la ventana). El
    # árbitro corre hacia la falta: si no está cerca, es un duelo.
    dist_arbitro = None
    for m in muestras[bisect.bisect_left(tiempos, t - 2.0):
                      bisect.bisect_right(tiempos, t + 2.0)]:
        if m.get("ball_scene") is None:
            continue
        bx, by = m["ball_scene"]
        for p in m.get("people", []):
            if p.get("team") != "arbitro":
                continue
            px = (p["x"] - m["cam_off"][0]) / max(1.0, m["median_height"])
            py = (p["y"] - m["cam_off"][1]) / max(1.0, m["median_height"])
            d = math.hypot(px - bx, py - by)
            dist_arbitro = d if dist_arbitro is None else min(dist_arbitro, d)
    arbitro_cerca = (dist_arbitro is not None
                     and dist_arbitro <= REFEREE_NEAR_HEIGHTS)

    # Reanudación tras perder el balón (para distinguir gol de tiro):
    # cuánto se pierde el balón tras el candidato y cómo reaparece.
    # Tras un gol el balón se pierde varios segundos y reaparece parado
    # en el centro (saque de centro: jugadores de los dos equipos
    # alrededor, lejos de cualquier portero). Tras un tiro parado el balón
    # reaparece rápido cerca del portero (parada o saque de puerta).
    # Reanudación tras perder el balón (para distinguir gol de tiro):
    # cuánto se pierde el balón tras el candidato y cómo reaparece.
    # Tras un gol el balón se pierde varios segundos y reaparece parado
    # en el centro (saque de centro: jugadores de los dos equipos
    # alrededor, lejos de cualquier portero). Tras un tiro parado el balón
    # reaparece rápido cerca del portero (parada o saque de puerta).
    # La pérdida se busca en [t-8, t+35]: el candidato de un gol se
    # detecta en la pérdida o en la celebración, DESPUÉS del tiro, así
    # que la pérdida puede haber empezado antes del candidato (el gol
    # ~06:07 con candidato en 06:13 no la veía porque buscaba solo
    # después de t). La reaparición es la primera muestra con el balón
    # VISTO y PARADO en el centro: el balón llevado en brazos a la línea
    # de medio campo se mueve y no cuenta; el saque de centro es balón
    # parado en el centro (con mapa, a menos de 12 m del punto central).
    perdido_s = 0.0
    reap_idx = None
    inicio_perdida = None
    j = bisect.bisect_left(tiempos, t - 8.0)
    while (j < len(tiempos)
           and tiempos[j] <= t + GOAL_RESTART_WINDOW_SECONDS):
        if muestras[j].get("ball_scene") is None:
            break
        j += 1
    if (j < len(tiempos)
            and tiempos[j] <= t + GOAL_RESTART_WINDOW_SECONDS):
        inicio_perdida = tiempos[j]
        k = j
        while (k < len(tiempos)
               and tiempos[k] <= t + GOAL_RESTART_WINDOW_SECONDS):
            mk = muestras[k]
            if (mk.get("ball_scene") is not None
                    and (mk.get("ball_speed_bh") or 0.0) <= STOP_TO_BH):
                if mk.get("ball_pitch_s") is not None:
                    rx, ry = (mk["ball_pitch_s"]["x"],
                              mk["ball_pitch_s"]["y"])
                    if math.hypot(rx - PITCH_LENGTH_M / 2,
                                  ry - PITCH_WIDTH_M / 2) <= 12.0:
                        reap_idx = k
                        break
                else:
                    reap_idx = k
                    break
            k += 1
        if reap_idx is not None:
            perdido_s = tiempos[reap_idx] - inicio_perdida
    reap_parado = 0.0
    reap_mixta = False
    reap_cerca_gk = False
    reap_gente = 0
    reap_en_centro = True
    if reap_idx is not None:
        # Tiempo TOTAL con el balón visto y parado en (reap, reap+5 s]:
        # en el saque de centro el balón está parado un rato antes del
        # saque, pero la cámara lo encuentra cuando ya va a sacarse.
        for m in muestras[reap_idx:bisect.bisect_right(
                tiempos, tiempos[reap_idx] + 5.0)]:
            if (m.get("ball_scene") is not None
                    and (m.get("ball_speed_bh") or 0.0) <= STOP_TO_BH):
                reap_parado += dt
        m_reap = muestras[reap_idx]
        if m_reap.get("ball_pitch_s") is not None:
            rx, ry = (m_reap["ball_pitch_s"]["x"],
                      m_reap["ball_pitch_s"]["y"])
            reap_en_centro = math.hypot(
                rx - PITCH_LENGTH_M / 2, ry - PITCH_WIDTH_M / 2) <= 12.0
        if m_reap.get("ball_scene") is not None:
            bx, by = m_reap["ball_scene"]
            equipos_cerca = set()
            for p in m_reap.get("people", []):
                tm = p.get("team")
                if tm == "arbitro":
                    continue
                px = (p["x"] - m_reap["cam_off"][0]) / max(1.0, m_reap["median_height"])
                py = (p["y"] - m_reap["cam_off"][1]) / max(1.0, m_reap["median_height"])
                if math.hypot(px - bx, py - by) <= CROWD_RADIUS_HEIGHTS:
                    reap_gente += 1
                    if tm in ("local", "visitante", "gk_local", "gk_visitante"):
                        equipos_cerca.add("local" if tm in ("local", "gk_local")
                                          else "visitante")
            reap_mixta = len(equipos_cerca) >= 2
            dists_reap = [d for d in (_dist_portero(m_reap, "gk_local_scene"),
                                      _dist_portero(m_reap, "gk_visitante_scene"))
                          if d is not None]
            reap_cerca_gk = bool(dists_reap) and min(dists_reap) <= GK_NEAR_HEIGHTS
    saque_centro = (perdido_s >= GOAL_LOST_MIN_SECONDS
                    and reap_idx is not None
                    and reap_parado >= GOAL_RESTART_STOP_SECONDS
                    and reap_mixta and not reap_cerca_gk
                    and reap_gente >= GOAL_RESTART_MIN_PEOPLE
                    and reap_en_centro)

    # Colocación para el saque de centro (gol por colocación, regla
    # 3b-bis): el balón parado en el centro un buen rato. Tras un gol el
    # balón se coloca en el centro y se queda ahí mientras los equipos
    # se colocan (~15-20 s en el gol del Cirbonero ~06:07: 15,8 s parado
    # en el centro entre 06:16 y 06:46). Esto caza los goles en los que
    # el balón NO se pierde (la cámara lo sigue en la celebración y en
    # el traslado al centro), donde la regla 3b no llega porque no hay
    # pérdida larga. Solo con mapa (la colocación en el centro exige
    # metros reales).
    colocacion_s = 0.0
    colocacion_idx = None
    colocacion_gente = 0
    colocacion_mixta = False
    if hay_campo:
        for idx in range(bisect.bisect_left(tiempos, t),
                         bisect.bisect_right(tiempos, t + 30.0)):
            mm = muestras[idx]
            bp = mm.get("ball_pitch_s")
            if (mm.get("ball_scene") is not None
                    and (mm.get("ball_speed_bh") or 0.0) <= STOP_TO_BH
                    and bp is not None
                    and math.hypot(bp["x"] - PITCH_LENGTH_M / 2,
                                   bp["y"] - PITCH_WIDTH_M / 2) <= 12.0):
                colocacion_s += dt
                if colocacion_idx is None:
                    colocacion_idx = idx
        if colocacion_idx is not None:
            m_col = muestras[colocacion_idx]
            if m_col.get("ball_scene") is not None:
                bx, by = m_col["ball_scene"]
                equipos_col = set()
                # Radio amplio (8 alturas): en el saque de centro los
                # jugadores están en el círculo central (9,15 m) y en sus
                # mitades, no todos pegados al balón; con 3,5 alturas
                # solo salía 1 persona y el gol no se detectaba.
                for p in m_col.get("people", []):
                    tm = p.get("team")
                    if tm == "arbitro":
                        continue
                    px = (p["x"] - m_col["cam_off"][0]) / max(1.0, m_col["median_height"])
                    py = (p["y"] - m_col["cam_off"][1]) / max(1.0, m_col["median_height"])
                    if math.hypot(px - bx, py - by) <= 8.0:
                        colocacion_gente += 1
                        if tm in ("local", "visitante", "gk_local", "gk_visitante"):
                            equipos_col.add("local" if tm in ("local", "gk_local")
                                            else "visitante")
                colocacion_mixta = len(equipos_col) >= 2
    # Último balón rápido antes de la colocación (el tiro): se busca
    # hacia atrás desde la colocación hasta 60 s antes; el primero que
    # se encuentra es el más reciente (la celebración y el traslado son
    # lentos, así que el tiro es el último balón rápido). Se exige que
    # esté al menos 10 s antes de la colocación: el balón que se mueve
    # al colocar (el saque mismo o un pase de la colocación) es rápido
    # pero no es el tiro (en el gol ~06:07 había un pico de 3,08 a 06:20,
    # 1 s antes de la colocación, y el tiro de 4,56 a 06:05).
    t_tiro_coloc = None
    equipo_gol_coloc = None
    if colocacion_idx is not None:
        t_col = tiempos[colocacion_idx]
        for idx in range(colocacion_idx, -1, -1):
            mm = muestras[idx]
            if mm["t"] < t_col - 60.0:
                break
            if (mm.get("ball_speed_bh") or 0.0) >= SPEED_FAST_BH:
                if t_col - mm["t"] >= 10.0:
                    t_tiro_coloc = mm["t"]
                    break
                # Balón rápido pegado a la colocación (el saque mismo):
                # se sigue buscando hacia atrás.
                continue
        if t_tiro_coloc is not None:
            # El equipo que marca es el contrario al portero que encaja:
            # el balón está junto a la portería del que encaja el gol.
            # Si no hay portero cerca, la posesión antes del tiro.
            m_tiro = _muestra_en(muestras, tiempos, t_tiro_coloc)
            if m_tiro is not None and m_tiro.get("ball_scene") is not None:
                d_loc = _dist_portero(m_tiro, "gk_local_scene")
                d_vis = _dist_portero(m_tiro, "gk_visitante_scene")
                if d_loc is not None and (d_vis is None or d_loc <= d_vis):
                    equipo_gol_coloc = team_visitante
                elif d_vis is not None:
                    equipo_gol_coloc = team_local
            if equipo_gol_coloc is None:
                pos_tiro = _posesion_mayoritaria(
                    muestras, tiempos, t_tiro_coloc - 3.0, t_tiro_coloc - 0.5)
                equipo_gol_coloc = (team_local if pos_tiro == "local"
                                    else team_visitante if pos_tiro == "visitante"
                                    else equipo_tiro_gol)

    # 1. Gol: balón que cruza la línea de gol rápido (fija_amplia por
    # plano, o cualquier modo con mapa del campo). El equipo es el que
    # tenía el balón antes del tiro (el que dispara).
    if (((fija and cruza_gol) or (hay_campo and cruza_gol_campo))
            and vel_max_gol >= SPEED_FAST_BH):
        return _res("goal", 0.80, f"Gol del {equipo_tiro_gol}", equipo_tiro_gol)
    # 2. Balón parado con reanudación: córner, saque de portero o balón parado.
    #    Si la reanudación es un tiro junto al portero (falta directa), no es
    #    balón parado: se deja pasar a las reglas de tiro/gol de portero.
    if parado_s >= 2.0 and vel_despues >= SPEED_FAST_BH:
        if fija and _cerca_linea_gol(xy_0) and _en_banda(xy_0):
            return _res("corner", 0.70, f"Córner del {equipo}", equipo)
        if agrupacion <= 2:
            return _res("gk_distribution", 0.65,
                        f"Distribución del portero del {equipo}", equipo)
        if not cerca_portero:
            return _res("set_piece", 0.65, f"Balón parado del {equipo}", equipo)
    # 3b. Gol en seguimiento: tiro rápido que se pierde y se reanuda en el
    #      centro (saque de centro). Va ANTES que la falta: un gol también
    #      tiene parada brusca + agrupación, y antes la falta lo robaba.
    #      Sin reanudación clara no es gol (tiro): un gol falso es peor
    #      que un gol perdido, el usuario revisa los tiros a mano.
    #      vel_max_gol mira 8 s atrás: el candidato se detecta en la
    #      pérdida/celebración, después del tiro. No se exige
    #      `perdido >= 0.5` (pérdida tras el candidato): la pérdida del
    #      gol ya va dentro de `saque_centro` (≥4 s en [t-8, t+35]).
    if (not fija and vel_max_gol >= SPEED_FAST_BH and saque_centro):
        return _res("goal", 0.70, f"Gol del {equipo_tiro_gol}", equipo_tiro_gol)
    # 3b-bis. Gol por colocación para el saque de centro: el balón parado
    #      en el centro un buen rato (≥8 s en 30 s) con jugadores alrededor
    #      (≥3 en 8 alturas: en el saque están en el círculo central, no
    #      pegados al balón), y un tiro rápido en los 60 s anteriores a
    #      la colocación. Caza los goles en los que el balón no se pierde
    #      (la cámara lo sigue en la celebración y el traslado).
    #      No se exige que haya de los dos equipos (mixta): en el saque
    #      tras el gol del Cirbonero los azules salían todos como
    #      "desconocido" (camiseta a rayas, lejos, con poco contraste) y
    #      solo se veían rojos; la colocación larga en el centro + el
    #      tiro previo ya es señal suficiente.
    if (not fija and hay_campo and colocacion_s >= 8.0
            and colocacion_idx is not None and t_tiro_coloc is not None
            and colocacion_gente >= 3):
        return _res("goal", 0.70, f"Gol del {equipo_gol_coloc}", equipo_gol_coloc)
    # 3c. Tiro a puerta: balón rápido junto al portero (no se pierde).
    if cerca_portero and ataca and vel_max >= SPEED_FAST_BH:
        return _res("shot", 0.70, f"Tiro a puerta del {equipo_atacante}",
                    equipo_atacante)
    # 4. Tiro: balón perdido a gran velocidad, o rápido hacia la línea de gol.
    #    Un pase atrás al propio portero NO es tiro: el balón acaba junto
    #    al portero del equipo que tiene la posesión (posesión, no tiro).
    #    Con mapa del campo, el balón tiene que ir HACIA la portería rival
    #    y ACABAR CERCA de ella: un balón rápido que se pierde en el centro
    #    del campo es un pase o un despeje, no un tiro (en 10 min sin
    #    ningún tiro real salían 7 "tiros" por esta regla). Sin mapa se
    #    mantiene la regla anterior (no hay posición real que mirar).
    tiro_en_tercio_final = True
    if hay_campo:
        tiro_en_tercio_final = False
        xy_fin = xy_campo_despues
        if xy_fin is None:
            for m in reversed(ventana):
                if m.get("ball_pitch_s") is not None:
                    xy_fin = (m["ball_pitch_s"]["x"], m["ball_pitch_s"]["y"])
                    break
        if xy_fin is not None:
            if x_gk_rival is not None:
                # La x está comprimida hacia el centro cuando la H es de
                # baja calidad, pero el balón y el portero se comprimen
                # juntos: la distancia entre ellos es más fiable que la
                # x absoluta.
                tiro_en_tercio_final = abs(xy_fin[0] - x_gk_rival) <= 30.0
            else:
                tiro_en_tercio_final = (xy_fin[0] <= 30.0
                                        or xy_fin[0] >= PITCH_LENGTH_M - 30.0)
    if (vel_max >= SPEED_FAST_BH and not pase_atras and tiro_en_tercio_final
            and (perdido >= 0.5
                 or (fija and _cerca_linea_gol(xy_despues))
                 or (hay_campo and xy_campo_despues is not None
                     and (xy_campo_despues[0] <= 5.0
                          or xy_campo_despues[0] >= PITCH_LENGTH_M - 5.0)))
            and hacia_porteria is not False):
        return _res("shot", 0.70, f"Tiro del {equipo_tiro} hacia portería", equipo_tiro)
    # 5. Centro: balón rápido y largo desde banda hacia el área.
    #    Si sale desde el centro y acaba en el área es pase clave (regla 9),
    #    no centro: por eso aquí se exige banda en el origen.
    if (vel_max >= SPEED_FAST_BH and avance >= 6.0 and not cambio_posesion
            and ((fija and _en_banda(xy_antes))
                 or (hay_campo and _en_banda_campo(xy_campo_antes)))):
        return _res("cross", 0.65, f"Centro del {equipo} al área", equipo)
    # 6. Transición / contraataque: cambio de posesión + avance rápido.
    if cambio_posesion and avance >= TRANSITION_ADVANCE_HEIGHTS:
        if avance >= 12.0 and vel_max >= SPEED_FAST_BH:
            return _res("counterattack", 0.70, f"Contraataque del {equipo}", equipo)
        return _res("transition", 0.70, f"Transición del {equipo}", equipo)
    # 3. Falta: frenazo + agrupación + juego parado después (el balón
    #    queda parado o se pierde mientras se prepara el lanzamiento).
    #    Va DESPUÉS de tiro y transición: un tiro o una contra también
    #    tienen frenazo + agrupación, y antes la falta los robaba. Ya no
    #    exige cambio de posesión (en una falta el equipo que tenía el
    #    balón lo conserva: saca él la falta) ni árbitro cerca como
    #    requisito (solo está visible en el 44% de las muestras): el
    #    árbitro sube la confianza a 0,65 y el tope de 5 faltas conserva
    #    antes las que lo tienen cerca.
    if (agrupacion >= CROWD_MIN_PEOPLE and frenazo and juego_parado):
        return _res("foul", 0.65 if arbitro_cerca else 0.60,
                    f"Falta sobre el {equipo}", equipo)
    # 7. Duelo: agrupación + cambio de posesión, sin avance rápido.
    if cambio_posesion and agrupacion >= CROWD_MIN_PEOPLE:
        return _res("duel", 0.60, f"Duelo ganado por el {equipo}", equipo)
    # 8. Recuperación / pérdida: cambio de posesión sin más.
    if cambio_posesion:
        return _res("recovery", 0.65, f"Recuperación del {equipo}", equipo)
    # 9. Pase clave: mismo equipo, balón largo que acaba en el área (fija
    #    amplia o mapa del campo) o en zona del portero (seguimiento). Un
    #    pase atrás al propio portero no es pase clave: es posesión.
    if (not cambio_posesion and vel_max >= SPEED_FAST_BH and avance >= 6.0
            and not pase_atras
            and ((fija and _en_area(xy_despues)) or en_zona_portero
                 or (hay_campo and _en_area_campo(xy_campo_despues)))):
        return _res("key_pass", 0.65, f"Pase clave del {equipo} al área", equipo)
    # 10. Entrada al área (fija amplia o mapa del campo) o a zona del
    #     portero (seguimiento). Un pase atrás al propio portero no es
    #     entrada al área.
    if (entra_area or entra_area_campo or entra_zona_portero) and not pase_atras:
        return _res("area_entry", 0.60, f"Entrada al área del {equipo}", equipo)
    # 11. Pase: balón rápido que recorre distancia, mismo equipo.
    if vel_max >= SPEED_FAST_BH and avance >= 4.0 and not cambio_posesion:
        return _res("pass", 0.65, f"Pase del {equipo}", equipo)
    # 11b. Pase por trayectoria: cambio de portador del mismo equipo con
    #      vuelo, aunque el balón no sea rápido (la mayoría de los pases
    #      de centro del campo no superan SPEED_FAST_BH, así que la regla
    #      11 no los ve). Un pase fallido no etiqueta pase.
    if pases:
        for p in pases:
            if abs(p["t"] - t) <= 1.5 and p["completado"]:
                equipo_pase = (team_local if p["team"] == "local"
                               else team_visitante)
                if (p["distancia_bh"] >= 12.0 or en_zona_portero) and not pase_atras:
                    return _res("key_pass", 0.70,
                                f"Pase clave del {equipo_pase} al área", equipo_pase)
                return _res("pass", 0.70, f"Pase del {equipo_pase}", equipo_pase)
    # 12. Balón parado con reanudación lenta (sin velocidad para la regla 2).
    #     Si el balón sigue parado después, no es una jugada: incierto.
    if parado_s >= 2.0 and vel_despues > STOP_TO_BH:
        return _res("set_piece", 0.60, f"Balón parado del {equipo}", equipo)
    return {"event": "uncertain", "confidence": 0.0,
            "summary": "Jugada sin clasificar clara.", "team": equipo}


# ---------------------------------------------------------------------------
# Función principal (interna: devuelve resultados + resumen)
# ---------------------------------------------------------------------------
def _analizar_impl(
    video_path: Path | str,
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
    imgsz: int,
    flip_x: bool,
    flip_y: bool,
    max_minutes: float | None,
    kickoff_offset: float,
    camera_mode: str,
    progress: ProgressCallback | None,
    color_gk_local: str = DEFAULT_GK_COLOR_LOCAL,
    color_gk_visitante: str = DEFAULT_GK_COLOR_VISITANTE,
    color_arbitro: str = DEFAULT_REFEREE_COLOR,
    conf: float = 0.10,
    pitch_model: str | None = None,
    pitch_keyframe_seconds: float = PITCH_KEYFRAME_SECONDS,
) -> dict:
    video_path = Path(video_path).expanduser().resolve()
    if not 1 <= sample_fps <= 10:
        raise ValueError("El muestreo debe estar entre 1 y 10 fps.")
    if not 1 <= max_candidates <= 20:
        raise ValueError("El número de candidatos debe estar entre 1 y 20.")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("La confianza mínima debe estar entre 0 y 1.")
    if not 0.0 < conf <= 1.0:
        raise ValueError("La confianza de YOLO debe estar entre 0 y 1.")
    if kickoff_offset < 0:
        raise ValueError("El saque inicial no puede ser negativo.")
    if camera_mode not in ("seguimiento", "fija_amplia"):
        raise ValueError("camera_mode debe ser 'seguimiento' o 'fija_amplia'.")
    update = progress or (lambda _message, _value: None)
    duracion_video = video_duration(video_path)
    if kickoff_offset >= duracion_video:
        raise VideoProcessingError(
            "El saque inicial está después del final del vídeo. Revísalo."
        )
    update("Cargando YOLO y siguiendo el vídeo fotograma a fotograma.", 0.02)
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
        color_local, color_visitante,
        color_gk_local, color_gk_visitante, color_arbitro,
        flip_x, flip_y,
        kickoff_offset, fin, update, conf=conf,
        pitch_model=pitch_model,
        pitch_keyframe_seconds=pitch_keyframe_seconds,
    )
    update("Compensando el movimiento de cámara y suavizando el balón.", 0.45)
    muestras = _suavizar_balon(muestras)
    muestras = _compensar_camara(muestras)
    muestras = _anotar_posesion(muestras, team_local, team_visitante)
    muestras = _anotar_porteros(muestras)
    # Corregir el eje x del campo con los porteros (la homografía con
    # keypoints solo del centro comprime todo hacia el centro).
    _corregir_eje_x_campo(muestras)
    posesion_tramos = posesion_por_tramos(muestras, team_local, team_visitante)
    update("Detectando pases por trayectoria (cambio de portador).", 0.50)
    pases = _detectar_pases_por_trayectoria(muestras)
    secuencias = _secuencias_de_pases(pases)
    update("Buscando candidatos a evento en las señales del tracking.", 0.55)
    candidatos = _detectar_candidatos(muestras, camera_mode)
    candidatos = _dedup_y_limitar(candidatos, max_candidates)
    output_dir = video_path.parent / f"{video_path.stem}_clips"
    resultados: list[dict] = []
    tiempos_muestras = [m["t"] for m in muestras]
    colores_radar = {"local": color_local, "visitante": color_visitante,
                     "arbitro": color_arbitro}
    if not candidatos:
        update("YOLO no encontró señales claras de jugada en el vídeo.", 1.0)
    for n_cand, cand in enumerate(candidatos, start=1):
        t = float(cand["seconds"])  # relativo al saque inicial
        t_video = t + kickoff_offset
        update(f"YOLO clasifica candidato {n_cand} de {len(candidatos)}.",
               0.55 + 0.35 * n_cand / max(1, len(candidatos)))
        clasificacion = _clasificar_por_tracking(
            cand, muestras, camera_mode, team_local, team_visitante, pases=pases)
        clip_path: str | None = None
        radar_path: str | None = None
        if (clasificacion["event"] != "uncertain"
                and clasificacion["confidence"] >= min_confidence):
            clip_start = max(0.0, t_video - pre_roll)
            clip_end = min(duracion_video, t_video + post_roll)
            label = f"{int(t // 60):02d}{int(t % 60):02d}_{clasificacion['event']}"
            # Radar del momento del evento (PNG) + vídeo del radar para
            # incrustar en el clip (abajo a la derecha, como en la imagen
            # de referencia: puntos de colores sobre el campo).
            m_cand = _muestra_en(muestras, tiempos_muestras, t)
            if m_cand is not None and m_cand.get("pitch_ok"):
                try:
                    import cv2
                    radar_png = output_dir / f"{int(clip_start):06d}_{label}_radar.png"
                    cv2.imwrite(str(radar_png), draw_radar(m_cand, colores_radar))
                    radar_path = str(radar_png)
                except Exception:
                    radar_path = None
            radar_video = export_radar_video(
                muestras, clip_start, clip_end,
                output_dir / f"{int(clip_start):06d}_{label}_radar.mp4",
                colores_radar)
            try:
                clip_path = str(export_clip(video_path, clip_start, clip_end,
                                            output_dir, label,
                                            radar_video=radar_video))
            finally:
                if radar_video is not None:
                    Path(radar_video).unlink(missing_ok=True)
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
            "radar_path": radar_path,
            "yolo_people": int(cand["people"]),
            "yolo_ball": 1 if cand["ball"] else 0,
            "reason": "; ".join(cand["signals"]),
            "signals": list(cand["signals"]),
        })
    update("Calculando estadísticas y escribiendo el informe.", 0.95)
    resultados = _limitar_faltas(resultados)
    estadisticas = _calcular_estadisticas(
        muestras, resultados, pases, secuencias,
        team_local, team_visitante, duracion_partido)
    informe_base = {
        "results": resultados,
        "possession_by_period": posesion_tramos,
        "kickoff_offset_s": kickoff_offset,
        "match_duration_s": duracion_partido,
        "video_duration_s": duracion_video,
        "samples_analyzed": len(muestras),
        "team_local": team_local,
        "team_visitante": team_visitante,
        "team_colors": {"local": color_local, "visitante": color_visitante,
                "gk_local": color_gk_local,
                "gk_visitante": color_gk_visitante,
                "arbitro": color_arbitro},
    }
    rutas_informe = _exportar_informe(
        informe_base, estadisticas, pases, video_path, output_dir)
    update("Análisis y generación de cortes completados.", 1.0)
    return {
        **informe_base,
        "estadisticas": estadisticas,
        "pases": pases,
        "secuencias": secuencias,
        **rutas_informe,
    }


def analyze_video(
    video_path: Path | str,
    color_local: str = "#1E88E5",
    color_visitante: str = "#E53935",
    color_gk_local: str = DEFAULT_GK_COLOR_LOCAL,
    color_gk_visitante: str = DEFAULT_GK_COLOR_VISITANTE,
    color_arbitro: str = DEFAULT_REFEREE_COLOR,
    team_local: str = "Local",
    team_visitante: str = "Visitante",
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    pre_roll: int = DEFAULT_PRE_ROLL,
    post_roll: int = DEFAULT_POST_ROLL,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    imgsz: int = 960,
    flip_x: bool = False,
    flip_y: bool = False,
    max_minutes: float | None = None,
    kickoff_offset: float = 0.0,
    camera_mode: str = "seguimiento",
    progress: ProgressCallback | None = None,
    conf: float = 0.10,
    pitch_model: str | None = None,
) -> list[dict]:
    """Analiza un vídeo y devuelve la lista de cortes clasificados.

    Cada resultado: seconds (relativo al saque inicial), minute, event,
    confidence, summary, team, ball_x, ball_y (coordenadas DE PLANO, no de
    campo), zone, clip_path (str o None), radar_path (PNG del mapa del
    campo en el momento del evento, o None sin mapa), yolo_people,
    yolo_ball, reason, signals. ``clip_path`` solo existe si event !=
    "uncertain" y confidence >= min_confidence; el resto son candidatos
    descartados. Para el resumen completo (posesión por tramos, duración,
    etc.) usar ``analyze_video_full``.

    Por defecto imgsz=960 y conf=0.10: en vídeo VEO el balón es diminuto y
    a 640/conf 0.25 solo aparece en ~2% de muestras (a 960/0.10, ~20%).
    Cuesta ~1,7x más tiempo que 640; en CPU lenta bajar sample_fps a 5.

    pitch_model: ruta al modelo de keypoints del campo (pitch_keypoints.pt).
    None = usar el archivo junto a este módulo si existe; "" = sin mapa.
    """
    return _analizar_impl(
        video_path, color_local, color_visitante, team_local,
        team_visitante, sample_fps, max_candidates, pre_roll, post_roll,
        min_confidence, yolo_model, imgsz, flip_x, flip_y,
        max_minutes, kickoff_offset, camera_mode, progress,
        color_gk_local=color_gk_local,
        color_gk_visitante=color_gk_visitante, color_arbitro=color_arbitro,
        conf=conf, pitch_model=pitch_model,
    )["results"]


def analyze_video_full(
    video_path: Path | str,
    color_local: str = "#1E88E5",
    color_visitante: str = "#E53935",
    color_gk_local: str = DEFAULT_GK_COLOR_LOCAL,
    color_gk_visitante: str = DEFAULT_GK_COLOR_VISITANTE,
    color_arbitro: str = DEFAULT_REFEREE_COLOR,
    team_local: str = "Local",
    team_visitante: str = "Visitante",
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    pre_roll: int = DEFAULT_PRE_ROLL,
    post_roll: int = DEFAULT_POST_ROLL,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    imgsz: int = 960,
    flip_x: bool = False,
    flip_y: bool = False,
    max_minutes: float | None = None,
    kickoff_offset: float = 0.0,
    camera_mode: str = "seguimiento",
    progress: ProgressCallback | None = None,
    conf: float = 0.10,
    pitch_model: str | None = None,
) -> dict:
    """Como analyze_video, pero devuelve el informe completo.

    {"results": [...], "possession_by_period": [{"tramo", "local_pct",
    "visitante_pct", "desconocido_pct"}, ...], "kickoff_offset_s": ...,
    "match_duration_s": ..., "video_duration_s": ...,
    "samples_analyzed": ..., "team_colors": {"local": ..., "visitante": ...},
    "estadisticas": {...}, "pases": [...], "secuencias": [...],
    "informe_md": ..., "eventos_csv": ..., "estadisticas_csv": ...}

    Por defecto imgsz=960 y conf=0.10 (ver analyze_video).
    """
    return _analizar_impl(
        video_path, color_local, color_visitante, team_local,
        team_visitante, sample_fps, max_candidates, pre_roll, post_roll,
        min_confidence, yolo_model, imgsz, flip_x, flip_y,
        max_minutes, kickoff_offset, camera_mode, progress,
        color_gk_local=color_gk_local,
        color_gk_visitante=color_gk_visitante, color_arbitro=color_arbitro,
        conf=conf, pitch_model=pitch_model,
    )


# ---------------------------------------------------------------------------
# Prueba rápida por terminal (sin Streamlit):
#   python video_track.py "C:\\Videos\\partido.mp4" --kickoff 04:32
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analizar un vídeo de partido con YOLO (sin Groq).")
    parser.add_argument("video", type=Path, help="Ruta local del vídeo")
    parser.add_argument("--color-local", default="#1E88E5")
    parser.add_argument("--color-visitante", default="#E53935")
    parser.add_argument("--color-gk-local", default=DEFAULT_GK_COLOR_LOCAL)
    parser.add_argument("--color-gk-visitante", default=DEFAULT_GK_COLOR_VISITANTE)
    parser.add_argument("--color-arbitro", default=DEFAULT_REFEREE_COLOR)
    parser.add_argument("--team-local", default="Local")
    parser.add_argument("--team-visitante", default="Visitante")
    parser.add_argument("--sample-fps", type=int, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Resolución de inferencia (960 por defecto; "
                             "640 es más rápido pero ve peor el balón)")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Confianza mínima de YOLO (0,10 por defecto; "
                             "a 0,25 el balón casi no aparece en vídeo VEO)")
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--kickoff", default="00:00",
                        help="Saque inicial en MM:SS (el calentamiento no se analiza)")
    parser.add_argument("--camera-mode", default="seguimiento",
                        choices=["seguimiento", "fija_amplia"])
    parser.add_argument("--pitch-model", default=None,
                        help="Modelo de keypoints del campo (pitch_keypoints.pt). "
                             "Vacío = usar el archivo junto a video_track.py si "
                             "existe; --sin-mapa lo desactiva")
    parser.add_argument("--sin-mapa", action="store_true",
                        help="No usar el mapa del campo (homografía)")
    parser.add_argument("--suggest-kickoff", action="store_true",
                        help="Solo sugerir el saque inicial y salir")
    args = parser.parse_args()
    if args.suggest_kickoff:
        sugerido = suggest_kickoff(args.video, args.yolo_model)
        print("Saque inicial sugerido:",
              format_mmss(sugerido) if sugerido is not None else "no encontrado")
        raise SystemExit(0)
    informe = analyze_video_full(
        args.video,
        color_local=args.color_local, color_visitante=args.color_visitante,
        color_gk_local=args.color_gk_local,
        color_gk_visitante=args.color_gk_visitante,
        color_arbitro=args.color_arbitro,
        team_local=args.team_local, team_visitante=args.team_visitante,
        sample_fps=args.sample_fps, max_candidates=args.max_candidates,
        min_confidence=args.min_confidence, yolo_model=args.yolo_model,
        imgsz=args.imgsz, conf=args.conf,
        max_minutes=args.max_minutes,
        kickoff_offset=parse_mmss(args.kickoff), camera_mode=args.camera_mode,
        pitch_model="" if args.sin_mapa else args.pitch_model,
        progress=lambda msg, frac: print(f"[{frac:.0%}] {msg}"),
    )
    print(json.dumps(informe, ensure_ascii=False, indent=2))
