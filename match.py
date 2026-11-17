import streamlit as st
import streamlit.components.v1 as components
import json
import io
import pandas as pd
from datetime import datetime, date
import shutil
import time
import os
import subprocess
import tempfile
import bisect
import threading
import re
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from html import escape as escape_html
from tracking_catalog import read_tracking_header, inspect_tracking, deduplicate_library
from heatmap_utils import build_heatmap_histogram

# OpenCV solo se necesita para exportar clips con overlay; si no está,
# la exportación usa ffmpeg por CLI (corte sin overlay) o avisa.
try:
    import cv2
    CV2_DISPONIBLE = True
except Exception:  # noqa: BLE001
    cv2 = None
    CV2_DISPONIBLE = False

# matplotlib/numpy vienen con Streamlit; se importan con guarda para que
# la app no caiga si faltan (los mapas de calor avisarían en ese caso).
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MPL_DISPONIBLE = True
except Exception:  # noqa: BLE001
    np = None
    plt = None
    LinearSegmentedColormap = None
    MPL_DISPONIBLE = False

# mplsoccer dibuja el campo; si no está, se dibuja a mano con matplotlib.
try:
    from mplsoccer import Pitch
    MPLSOCER_DISPONIBLE = True
except Exception:  # noqa: BLE001
    Pitch = None
    MPLSOCER_DISPONIBLE = False

# El análisis YOLO (generar tracking) es opcional en local: si el módulo
# no está disponible en este equipo, la pantalla manual sigue funcionando
# con un tracking ya generado (Colab / script). La pantalla manual en sí
# no necesita video_track: el tracking se inyecta como JSON.
try:
    from video_track import analyze_video_full, parse_mmss, VideoProcessingError
    VIDEO_DISPONIBLE = True
    VIDEO_ERROR = ""
except Exception as _err_video:  # noqa: BLE001 - sin YOLO no hay análisis local
    analyze_video_full = None
    parse_mmss = None
    VideoProcessingError = Exception
    VIDEO_DISPONIBLE = False
    VIDEO_ERROR = str(_err_video)

# El tracking JSON se genera fuera de la app (notebook de Colab
# analyze_colab.ipynb con GPU, o por script/VM) y la pantalla de Vídeo
# lo carga de la carpeta (tracking_*.json junto a match.py).

# --- Servidor local del vídeo (auto-carga en la pantalla manual) ---
# El navegador no puede abrir un vídeo local solo con su ruta, así que la
# app sirve ese archivo por HTTP (hilo en segundo plano, con soporte de
# rangos para poder buscar por el vídeo, y cabeceras CORS) y la pantalla
# manual lo carga sola con su URL: al terminar el análisis, el vídeo ya
# aparece con el tracking encima, sin volver a elegirlo abajo. El selector
# de archivos de la pantalla sigue como alternativa. El servidor vive en
# un global de módulo (cada re-ejecución de Streamlit corre en el mismo
# proceso) para no re-crearlo en cada rerun: mismo puerto, y solo cambia
# el archivo servido cuando cambia la ruta del vídeo. Si el servidor no
# se puede arrancar (p. ej. en un despliegue remoto), servir_video
# devuelve "" y la pantalla cae al selector de archivos.
_VIDEO_SERVE = {"server": None, "port": None, "path": None, "error": ""}


class _VideoFileHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silenciar el log del servidor
        pass

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers",
                         "Content-Range, Accept-Ranges, Content-Length")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.end_headers()

    def do_HEAD(self):
        self._serve(head_only=True)

    def do_GET(self):
        self._serve(head_only=False)

    def _serve(self, head_only):
        path = _VIDEO_SERVE.get("path")
        if not path or not os.path.isfile(path):
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            return
        try:
            size = os.path.getsize(path)
        except OSError:
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            return
        start, end = 0, size - 1
        rango = self.headers.get("Range")
        if rango:
            m = re.match(r"bytes=(\d*)-(\d*)", rango.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self._send_cors()
                    self.end_headers()
                    return
        length = end - start + 1
        ctype = mimetypes.guess_type(path)[0] or "video/mp4"
        self.send_response(206 if rango else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if rango:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self._send_cors()
        self.end_headers()
        if head_only:
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                restante = length
                while restante > 0:
                    trozo = f.read(min(1024 * 1024, restante))
                    if not trozo:
                        break
                    self.wfile.write(trozo)
                    restante -= len(trozo)
        except (BrokenPipeError, ConnectionResetError):
            pass


def servir_video(ruta):
    """Sirve el vídeo local por HTTP y devuelve su URL, o "" si no se puede.
    El servidor se crea una sola vez (mismo puerto); si cambia la ruta, solo
    cambia el archivo servido y la URL sigue siendo la misma."""
    ruta = str(ruta)
    if not os.path.isfile(ruta):
        return ""
    srv = _VIDEO_SERVE.get("server")
    if srv is not None and _VIDEO_SERVE.get("path") == ruta:
        return f"http://127.0.0.1:{_VIDEO_SERVE['port']}/video.mp4"
    _VIDEO_SERVE["path"] = ruta
    if srv is None:
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", 0), _VideoFileHandler)
            srv.daemon_threads = True
            _VIDEO_SERVE["server"] = srv
            _VIDEO_SERVE["port"] = srv.server_address[1]
            _VIDEO_SERVE["error"] = ""
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        except OSError as err:
            _VIDEO_SERVE["server"] = None
            _VIDEO_SERVE["error"] = str(err)
            return ""
    return f"http://127.0.0.1:{_VIDEO_SERVE['port']}/video.mp4"


st.set_page_config(page_title="Match View - Regional Preferente Navarra", layout="wide")

# --- 1. CARGA DE PLANTILLAS DESDE EL EXCEL ---
@st.cache_data
def cargar_plantillas_excel():
    excel_path = "Plantillas preferente.xlsx"
    plantillas_por_equipo = {}
    
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        # El excel tiene pares de columnas: Dorsal y Nombre por cada equipo
        for i in range(0, len(df.columns) - 1, 2):
            t_name = df.columns[i]
            if "Unnamed" in str(t_name):
                t_name = df.columns[i+1]
            
            c_dorsal = df.iloc[:, i]
            c_nombre = df.iloc[:, i+1]
            
            jugadores = []
            for d, n in zip(c_dorsal, c_nombre):
                if pd.notna(n):
                    # Limpiar espacios y formatear
                    d_clean = str(d).strip()
                    n_clean = str(n).strip()
                    try:
                        d_num = int(float(d_clean))
                    except:
                        d_num = 1
                    jugadores.append({"dorsal": d_num, "nombre": n_clean, "pos": "MED"})
            
            if jugadores:
                plantillas_por_equipo[str(t_name).strip()] = jugadores
                
    return plantillas_por_equipo

PLANTILLAS_EQUIPOS = cargar_plantillas_excel()
LISTA_EQUIPOS_NAVARRA = list(PLANTILLAS_EQUIPOS.keys()) if PLANTILLAS_EQUIPOS else [
    "C.D. San Miguel", "C.D. Aluvion", "C.A. Valtierrano", "C.D. San Adrian",
    "C.D. Mendi", "C.A. Milagres de Futbol", "C.D. Larrate", "C.D. Arenas",
    "C.D. River Ega", "C.D. Funes", "C.D. Azkarrena", "C.D. Ribaforada",
    "C.A. Cirbonero \"B\"", "C.D. Idoya", "C.D. Sesma", "C.D. Alesves"
]

POSICIONES = ["POR", "DEF", "MED", "DEL"]

# --- 2. INICIALIZACIÓN DE ESTADOS ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "pagina" not in st.session_state:
    st.session_state.pagina = "alineaciones"  # "alineaciones", "partido", "final", "video"
if "eventos" not in st.session_state:
    st.session_state.eventos = []
if "inicio_tiempo" not in st.session_state:
    st.session_state.inicio_tiempo = None
if "tiempo_acumulado" not in st.session_state:
    st.session_state.tiempo_acumulado = 0
if "zona_actual" not in st.session_state:
    st.session_state.zona_actual = "Z5 - Medio Centro"

def obtener_plantilla_inicial(eq_nombre):
    if eq_nombre in PLANTILLAS_EQUIPOS:
        completa = PLANTILLAS_EQUIPOS[eq_nombre].copy()
        # Tomar los primeros 11 como titulares y los 7 siguientes como suplentes
        xi = completa[:11] if len(completa) >= 11 else completa
        sup = completa[11:18] if len(completa) > 11 else []
        return xi, sup
    else:
        xi = [{"dorsal": i+1, "nombre": f"Jugador {i+1}", "pos": "POR" if i==0 else "MED"} for i in range(11)]
        sup = [{"dorsal": i+12, "nombre": f"Suplente {i+1}", "pos": "MED"} for i in range(7)]
        return xi, sup

eq_def_1 = LISTA_EQUIPOS_NAVARRA[0] if len(LISTA_EQUIPOS_NAVARRA) > 0 else "C.D. San Miguel"
eq_def_2 = LISTA_EQUIPOS_NAVARRA[1] if len(LISTA_EQUIPOS_NAVARRA) > 1 else "C.D. Aluvion"

xi_loc_init, sup_loc_init = obtener_plantilla_inicial(eq_def_1)
xi_vis_init, sup_vis_init = obtener_plantilla_inicial(eq_def_2)

if "datos_partido" not in st.session_state:
    st.session_state.datos_partido = {
        "temporada": "2026/2027",
        "competicion": "Regional Preferente Navarra",
        "fecha": date.today(),
        "descripcion": "Jornada 1",
        "eq_local": eq_def_1,
        "eq_visitante": eq_def_2,
        "color_local": "#1E88E5",
        "color_visitante": "#E53935",
        "color_portero_local": "#FFEB3B",
        "color_portero_visitante": "#FF9800",
        "color_arbitro": "#000000",
        "form_local": "1-4-4-2",
        "form_visitante": "1-4-3-3",
        "xi_local": xi_loc_init,
        "sup_local": sup_loc_init,
        "xi_visitante": xi_vis_init,
        "sup_visitante": sup_vis_init
    }

# --- 3. PANTALLA DE ACCESO ---
# La llave NO está en el código (antes era "1234" fija y el mensaje de
# error la revelaba). Se lee de la variable de entorno
# MATCH_VIEW_PASSWORD o de los secrets de Streamlit
# (.streamlit/secrets.toml con access_password = "..."). Si no hay
# ninguna, se usa "1234" para no romper el acceso actual: cámbiala
# poniendo MATCH_VIEW_PASSWORD en tu PC antes de arrancar Streamlit.
import os as _os

def _llave_acceso():
    try:
        return st.secrets.get("access_password")
    except Exception:
        pass
    return _os.environ.get("MATCH_VIEW_PASSWORD") or "1234"

if not st.session_state.authenticated:
    st.subheader("🔒 Acceso al Panel de Análisis")
    # Con st.form, pulsar Enter en la contraseña envía el formulario
    # (sin form, Enter solo recargaba y había que pulsar "Entrar").
    with st.form("login"):
        pwd = st.text_input("Introduce la llave de acceso:", type="password")
        entrar = st.form_submit_button("Entrar", type="primary")
    if entrar:
        if pwd == _llave_acceso():
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Llave incorrecta")
    st.stop()

# --- Diseño visual: navegación única, componentes compactos ---
st.markdown("""
<style>
:root { color-scheme: dark; }
[data-testid="stAppViewContainer"] {
    background: #0B1220;
    color: #E5EAF2;
}
[data-testid="stHeader"] { background: rgba(11,18,32,.94); }
.block-container { max-width: 1560px; padding-top: 1.1rem; padding-bottom: 2.2rem; }
[data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3 { letter-spacing: -.035em; }
[data-testid="stMainBlockContainer"] h4,
[data-testid="stMainBlockContainer"] h5 { color: #DAE4F3; }
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 13px;
}
div[data-testid="stMetric"] {
    background: #131F31; border: 1px solid #233650;
    border-radius: 12px; padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #AFC1D6; }
[data-testid="stMetricValue"] { color: #F3F7FC; font-weight: 700; }
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border-radius: 10px; overflow: hidden;
}
div[data-testid="stExpander"] {
    background: #121C2B; border: 1px solid #26354C;
    border-radius: 12px;
}
div.stButton > button { border-radius: 10px; font-weight: 600; }
div.stButton > button[kind="primary"] { background: #2869DF; border-color: #2869DF; }
div[data-testid="stAlert"] { border-radius: 10px; }
hr { border-color: #243145; }
.mv-head {
    display:flex; align-items:center; justify-content:space-between;
    flex-wrap:wrap; gap:12px; margin-bottom:10px;
}
.mv-logo { font-weight:800; letter-spacing:.11em; color:#F4F7FB; font-size:1.15rem; }
.mv-logo i { color:#4E92FF; font-style:normal; }
.mv-sub { color:#93A7C0; font-size:.81rem; margin-top:2px; }
.mv-match { color:#BCCCE0; font-size:.88rem; }
@media (max-width: 760px) {
  .block-container { padding-left:.8rem; padding-right:.8rem; }
  .mv-match { display:none; }
}
</style>
""", unsafe_allow_html=True)

_match_header = st.session_state.datos_partido
st.markdown(
    '<div class="mv-head"><div><div class="mv-logo">MATCH <i>VIEW</i></div>'
    '<div class="mv-sub">Football Intelligence · análisis local</div></div>'
    '<div class="mv-match">'
    + escape_html(str(_match_header["eq_local"])) + ' &nbsp; vs &nbsp; '
    + escape_html(str(_match_header["eq_visitante"])) + '</div></div>',
    unsafe_allow_html=True,
)
_nav_options = [
    ("alineaciones", "Configurar"),
    ("partido", "Partido"),
    ("video", "Vídeo"),
    ("final", "Resultados"),
]
_nav_columns = st.columns(len(_nav_options), gap="small")
for _col, (_nav_page, _nav_label) in zip(_nav_columns, _nav_options):
    with _col:
        if st.button(
            _nav_label, key=f"mv_navigation_{_nav_page}",
            type="primary" if st.session_state.pagina == _nav_page else "secondary",
            use_container_width=True,
        ):
            if st.session_state.pagina != _nav_page:
                st.session_state.pagina = _nav_page
                st.rerun()

FORMACIONES_COORDS = {
    "1-4-4-2": [(8,50), (22,18), (22,38), (22,62), (22,82), (36,18), (36,38), (36,62), (36,82), (46,35), (46,65)],
    "1-4-3-3": [(8,50), (22,18), (22,38), (22,62), (22,82), (34,30), (32,50), (34,70), (45,20), (46,50), (45,80)],
    "1-4-2-3-1": [(8,50), (20,18), (20,38), (20,62), (20,82), (30,35), (30,65), (40,20), (40,50), (40,80), (47,50)],
    "1-3-5-2": [(8,50), (20,25), (20,50), (20,75), (34,15), (34,32), (32,50), (34,68), (34,85), (46,35), (46,65)],
    "1-5-3-2": [(8,50), (20,15), (20,32), (20,50), (20,68), (20,85), (34,30), (34,50), (34,70), (46,35), (46,65)]
}


# --- 6b. AYUDAS GENÉRICAS: mapas de calor, biblioteca y clips ---
# Todo genérico: equipos, colores, kickoff y duraciones salen del JSON
# de tracking que se cargue; nada está hardcodeado a ningún partido.

def _hex_a_rgb(hex_color):
    """'#RRGGBB' -> (r, g, b) 0-255. Si no es válido, gris."""
    try:
        h = str(hex_color).lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:  # noqa: BLE001
        return (128, 128, 128)


def _hex_a_bgr(hex_color):
    r, g, b = _hex_a_rgb(hex_color)
    return (b, g, r)


def _cmap_equipo(hex_color):
    """Colormap blanco -> color del equipo para el mapa de calor."""
    return LinearSegmentedColormap.from_list(
        "equipo", ["#ffffff", str(hex_color or "#888888")])


def _mmss(segundos):
    segundos = max(0, int(segundos))
    return f"{segundos // 60:02d}:{segundos % 60:02d}"


@st.cache_resource(show_spinner=False, max_entries=2)
def cargar_tracking_json(ruta_str, mtime):
    """Lee un tracking JSON (cacheado; se invalida si cambia el archivo)."""
    with open(ruta_str, encoding="utf-8") as f:
        return json.load(f)


def _mostrar_figura_streamlit(fig):
    """Render PNG pequeño y no mantener copias grandes de Matplotlib."""
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=80, bbox_inches="tight")
        st.image(buffer.getvalue(), use_container_width=True)
    finally:
        buffer.close()


def tramos_partido(match_duration_s):
    """Tramos seleccionables (etiqueta, inicio_s, fin_s) en minutos de
    partido: todo, mitades y tramos de 15' según la duración real."""
    dur = float(match_duration_s or 0)
    tramos = [("Todo el partido", 0.0, dur)]
    if dur > 2700:
        tramos.append(("1ª mitad (0-45')", 0.0, 2700.0))
        tramos.append(("2ª mitad (45'-fin)", 2700.0, dur))
    elif dur > 0:
        tramos.append(("1ª mitad (0-45')", 0.0, dur))
    ini = 0.0
    while ini < dur:
        fin = min(ini + 900.0, dur)
        tramos.append((f"{int(ini // 60)}-{int(fin // 60)}'", ini, fin))
        ini = fin
    return tramos


def _puntos_mapa_calor(tracking, sujeto, t_ini_video, t_fin_video):
    """Puntos (xs, ys, usa_campo) para el mapa de calor en un tramo de
    tiempo de vídeo. sujeto: 'local' | 'visitante' | 'balon'.
    Si las muestras traen coordenadas de campo (pp/bp, metros 105x68)
    se usan; si no, las del fotograma (p/b, 0-100)."""
    muestras = tracking.get("samples") or []
    tc = tracking.get("team_code") or {}
    codigos = {"local": {tc.get("local", 0), tc.get("gk_local", 2)},
               "visitante": {tc.get("visitante", 1), tc.get("gk_visitante", 3)}}.get(sujeto)
    track_id = int(sujeto.split(":", 1)[1]) if sujeto.startswith("jugador:") else None
    usa_campo = any(("pp" in s or "bp" in s) for s in muestras[:200])
    xs, ys = [], []
    for s in muestras:
        t = float(s.get("t", 0) or 0)
        if t < t_ini_video or t >= t_fin_video:
            continue
        if usa_campo:
            if sujeto == "balon":
                bp = s.get("bp")
                if bp:
                    xs.append(float(bp[0])); ys.append(float(bp[1]))
            else:
                for pt in s.get("pp") or []:
                    if ((track_id is not None and len(pt) >= 4
                         and pt[3] == track_id)
                            or (track_id is None and codigos is not None
                                and len(pt) >= 3 and pt[2] in codigos)):
                        xs.append(float(pt[0])); ys.append(float(pt[1]))
        else:
            if sujeto == "balon":
                b = s.get("b")
                if b:
                    xs.append(float(b[0])); ys.append(float(b[1]))
            else:
                for per in s.get("p") or []:
                    if ((track_id is not None and len(per) >= 6
                         and per[5] == track_id)
                            or (track_id is None and codigos is not None
                                and len(per) >= 5 and per[4] in codigos)):
                        xs.append(float(per[0])); ys.append(float(per[1]))
    return xs, ys, usa_campo


def _dibujar_campo_mano(ax):
    """Campo 105x68 dibujado a mano con matplotlib (fallback sin
    mplsoccer): rectángulo, línea central, círculo, áreas y porterías."""
    import matplotlib.patches as mpatches
    ax.add_patch(mpatches.Rectangle((0, 0), 105, 68, fill=False, lw=1.5, ec="#333333"))
    ax.plot([52.5, 52.5], [0, 68], color="#333333", lw=1.5)
    ax.add_patch(mpatches.Circle((52.5, 34), 9.15, fill=False, lw=1.5, ec="#333333"))
    ax.plot([52.5], [34], marker="o", ms=3, color="#333333")
    for x0 in (0, 105):
        sgn = 1 if x0 == 0 else -1
        ax.add_patch(mpatches.Rectangle((min(x0, x0 + sgn * 16.5), 13.84), 16.5, 40.32,
                                        fill=False, lw=1.5, ec="#333333"))
        ax.add_patch(mpatches.Rectangle((min(x0, x0 + sgn * 5.5), 24.84), 5.5, 18.32,
                                        fill=False, lw=1.5, ec="#333333"))
        ax.plot([x0 + sgn * 11], [34], marker="o", ms=3, color="#333333")
        ax.add_patch(mpatches.Rectangle((min(x0, x0 - sgn * 2.44), 30.34), 2.44, 7.32,
                                        fill=False, lw=1.5, ec="#333333"))


def generar_figura_mapa_calor(tracking, sujeto, tramo):
    """Figura matplotlib del mapa de calor. tramo: (etiqueta, ini_s, fin_s)
    en segundos de partido. Devuelve (fig, nota) o (None, motivo)."""
    if not MPL_DISPONIBLE:
        return None, "matplotlib no está disponible en este equipo."
    kickoff = float(tracking.get("kickoff_s") or 0)
    _, ini_p, fin_p = tramo
    hist, usa_campo, puntos = build_heatmap_histogram(
        tracking, sujeto, kickoff + ini_p, kickoff + fin_p)
    if not puntos:
        return None, "Sin detecciones en ese tramo para lo seleccionado."
    teams = tracking.get("teams") or {}
    colors = tracking.get("colors") or {}
    if sujeto == "balon":
        titulo = "Balón"
        color_hex = "#FFD21F"
    elif sujeto.startswith("jugador:"):
        track_id = int(sujeto.split(":", 1)[1])
        titulo = f"Jugador (ID {track_id})"
        perfil = next((p for p in
                       (tracking.get("player_stats") or {}).get("players", [])
                       if p.get("track_id") == track_id), {})
        equipo = perfil.get("team", "local")
        color_hex = colors.get(
            "visitante" if equipo in ("visitante", "gk_visitante") else "local",
            "#1E88E5")
    elif sujeto == "local":
        titulo = teams.get("local", "Local")
        color_hex = colors.get("local", "#1E88E5")
    else:
        titulo = teams.get("visitante", "Visitante")
        color_hex = colors.get("visitante", "#E53935")
    cmap = _cmap_equipo(color_hex)
    # Crear una única figura por mapa; Pitch.draw ya crea la suya.
    fig = None
    if usa_campo:
        if MPLSOCER_DISPONIBLE:
            pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                          pitch_color="white", line_color="#333333", linewidth=1.5)
            fig, ax = pitch.draw(figsize=(8, 5.2))
            fig.set_dpi(65)
        else:
            fig, ax = plt.subplots(figsize=(8, 5.2), dpi=65)
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 68)
            ax.set_aspect("equal")
            _dibujar_campo_mano(ax)
        im = ax.imshow(hist.T, extent=(0, 105, 0, 68), origin="lower",
                       cmap=cmap, alpha=0.85, aspect="auto", interpolation="nearest")
        ax.set_title(f"Mapa de calor — {titulo} ({tramo[0]})", fontsize=13)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Nº de detecciones")
        nota = "Coordenadas de campo (105×68 m)."
    else:
        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=65)
        ax.set_xlim(0, 100)
        ax.set_ylim(100, 0)
        ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((0, 0), 100, 100, fill=False, lw=1.5, ec="#333333"))
        im = ax.imshow(hist.T, extent=(0, 100, 0, 100), origin="upper",
                       cmap=cmap, alpha=0.85, aspect="auto", interpolation="nearest")
        ax.set_title(f"Mapa de calor — {titulo} ({tramo[0]})", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label="Nº de detecciones")
        nota = ("Concentración en la imagen (este tracking no trae "
                "coordenadas de campo).")
    fig.tight_layout()
    return fig, nota


# --- Visualizaciones del Menú Final (desde lo etiquetado a mano) ---
# Z1-Z9 es una rejilla 3x3 ABSOLUTA sobre el campo (105x68 m), tal como se
# muestra en la pantalla de partido: Z1-Z3 tercio izquierdo, Z4-Z6 central,
# Z7-Z9 derecho; dentro de cada tercio, Izq/Centro/Der es la banda (arriba /
# centro / abajo en el dibujo). No se registra la portería que ataca cada
# equipo en cada parte, así que las zonas se dibujan en términos absolutos,
# sin espejar por equipo. El punto marca el centro de la zona, nunca la
# posición exacta del evento: el etiquetado solo guarda la zona.

_ZONAS_ORDEN = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9"]
# Centro de cada zona en metros (105x68): x por tercio, y por banda.
_ZONAS_CENTRO = {
    "Z1": (17.5, 56.67), "Z2": (17.5, 34.0), "Z3": (17.5, 11.33),
    "Z4": (52.5, 56.67), "Z5": (52.5, 34.0), "Z6": (52.5, 11.33),
    "Z7": (87.5, 56.67), "Z8": (87.5, 34.0), "Z9": (87.5, 11.33),
}
_ZONA_ANCHO = 35.0    # 105 / 3
_ZONA_ALTO = 68.0 / 3  # ~22.67
# Color de relleno del punto de tiro según el resultado (Detalle).
_TIRO_COLORES = {"Gol": "#2E9E44", "A puerta": "#F2A007", "Fuera": "#9E9E9E"}


def _codigo_zona(valor):
    """Código Z1-Z9 de un valor de la columna Zona ('Z1 - Def. Izq' -> 'Z1';
    'Z1' -> 'Z1'). Devuelve None si no es una zona válida."""
    if valor is None:
        return None
    txt = str(valor).strip().upper()
    if len(txt) >= 2 and txt[0] == "Z" and txt[1] in "123456789":
        return txt[:2]
    return None


def _conteo_eventos_por_zona(df_eventos, equipo=None):
    """Conteo de eventos por zona Z1-Z9 (dict con las 9 zonas, aunque sea 0).
    Si se pasa equipo, solo cuenta sus eventos. Sin datos o sin columna Zona,
    devuelve todo a cero."""
    conteo = {z: 0 for z in _ZONAS_ORDEN}
    if df_eventos is None or len(df_eventos) == 0:
        return conteo
    if "Zona" not in df_eventos.columns:
        return conteo
    df = df_eventos
    if equipo is not None and "Equipo" in df.columns:
        df = df[df["Equipo"] == equipo]
    for valor in df["Zona"]:
        cod = _codigo_zona(valor)
        if cod is not None:
            conteo[cod] += 1
    return conteo


def _campo_en_ax(ax):
    """Dibuja el campo 105x68 en un ax ya creado (mplsoccer si está, si no
    el dibujo a mano de _dibujar_campo_mano)."""
    if MPLSOCER_DISPONIBLE:
        pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                      pitch_color="white", line_color="#333333", linewidth=1.5)
        pitch.draw(ax=ax)
    else:
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_aspect("equal")
        _dibujar_campo_mano(ax)
    ax.set_xticks([])
    ax.set_yticks([])


def generar_figura_mapa_tiros(df_eventos, eq_local, eq_visitante,
                              color_local="#1E88E5", color_visitante="#E53935"):
    """Figura matplotlib del mapa de tiros, desde los eventos Tiro etiquetados
    a mano. Cada tiro es un punto en el centro de su zona (Z1-Z9) con un
    pequeño desplazamiento determinista para que no se solapen los de la
    misma zona. Relleno por resultado (Detalle: Gol / A puerta / Fuera) y
    borde con el color del equipo que tira. Devuelve (fig, nota) o
    (None, motivo) si no hay tiros que dibujar."""
    if not MPL_DISPONIBLE:
        return None, "matplotlib no está disponible en este equipo."
    if df_eventos is None or len(df_eventos) == 0 or "Evento" not in df_eventos.columns:
        return None, "Sin eventos registrados en este partido."
    tiros = df_eventos[df_eventos["Evento"] == "Tiro"]
    if len(tiros) == 0:
        return None, "Sin tiros registrados en este partido."
    color_equipo = {eq_local: color_local, eq_visitante: color_visitante}
    rng = np.random.default_rng(7)  # mismo dibujo en cada ejecución
    xs, ys, caras, bordes = [], [], [], []
    n_dibujados = 0
    for _, ev in tiros.iterrows():
        cod = _codigo_zona(ev.get("Zona"))
        if cod is None:
            continue
        cx, cy = _ZONAS_CENTRO[cod]
        x = float(np.clip(cx + rng.uniform(-7, 7),
                          cx - _ZONA_ANCHO / 2 + 2, cx + _ZONA_ANCHO / 2 - 2))
        y = float(np.clip(cy + rng.uniform(-5, 5),
                          cy - _ZONA_ALTO / 2 + 2, cy + _ZONA_ALTO / 2 - 2))
        xs.append(x)
        ys.append(y)
        caras.append(_TIRO_COLORES.get(str(ev.get("Detalle", "")).strip(), "#9E9E9E"))
        bordes.append(color_equipo.get(ev.get("Equipo"), "#666666"))
        n_dibujados += 1
    if n_dibujados == 0:
        return None, "Los tiros registrados no tienen una zona Z1-Z9 válida."
    fig, ax = plt.subplots(figsize=(10, 6.8))
    _campo_en_ax(ax)
    ax.scatter(xs, ys, s=170, c=caras, edgecolors=bordes, linewidths=2.0, zorder=5)
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="#333333",
               markersize=11, label=f"{det} ({sum(1 for f in caras if f == c)})")
        for det, c in _TIRO_COLORES.items() if c in caras
    ]
    leyenda += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=color_local, markeredgewidth=2.0, markersize=11,
               label=f"Borde: {eq_local}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=color_visitante, markeredgewidth=2.0, markersize=11,
               label=f"Borde: {eq_visitante}"),
    ]
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=10)
    ax.set_title(f"Mapa de tiros — {eq_local} vs {eq_visitante} "
                 f"({n_dibujados} tiros)", fontsize=13)
    fig.tight_layout()
    nota = ("Cada punto es un tiro etiquetado a mano, situado en el centro de "
            "su zona (Z1–Z3 tercio izquierdo, Z7–Z9 derecho): no es la posición "
            "exacta del tiro. Relleno por resultado, borde por equipo.")
    return fig, nota


def generar_figura_mapa_zonas(df_eventos, eq_local, eq_visitante,
                              color_local="#1E88E5", color_visitante="#E53935"):
    """Figura matplotlib de actividad por zonas: dos campos lado a lado
    (local / visitante) con la densidad de eventos por zona Z1-Z9. Cada celda
    se colorea con el color del equipo (más intenso = más eventos) y lleva el
    conteo. La escala de intensidad es común a ambos equipos (el máximo de
    los dos), para que se puedan comparar. Devuelve (fig, nota) o
    (None, motivo) si no hay eventos con zona válida."""
    if not MPL_DISPONIBLE:
        return None, "matplotlib no está disponible en este equipo."
    if df_eventos is None or len(df_eventos) == 0 or "Zona" not in df_eventos.columns:
        return None, "Sin eventos registrados en este partido."
    c_local = _conteo_eventos_por_zona(df_eventos, eq_local)
    c_vis = _conteo_eventos_por_zona(df_eventos, eq_visitante)
    total = sum(c_local.values()) + sum(c_vis.values())
    if total == 0:
        return None, "Los eventos registrados no tienen una zona Z1-Z9 válida."
    vmax = max(max(c_local.values()), max(c_vis.values()))
    import matplotlib.patches as mpatches
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    for ax, equipo, conteo, color in (
            (axes[0], eq_local, c_local, color_local),
            (axes[1], eq_visitante, c_vis, color_visitante)):
        _campo_en_ax(ax)
        for z in _ZONAS_ORDEN:
            cx, cy = _ZONAS_CENTRO[z]
            n = conteo[z]
            ax.add_patch(mpatches.Rectangle(
                (cx - _ZONA_ANCHO / 2, cy - _ZONA_ALTO / 2),
                _ZONA_ANCHO, _ZONA_ALTO,
                facecolor=color, alpha=(0.85 * n / vmax) if n > 0 else 0.0,
                edgecolor="#333333", lw=0.8, zorder=2))
            ax.text(cx - _ZONA_ANCHO / 2 + 1.5, cy + _ZONA_ALTO / 2 - 3.5, z,
                    fontsize=8, color="#555555", ha="left", va="top", zorder=4)
            ax.text(cx, cy, str(n), fontsize=14, fontweight="bold",
                    color="#111111" if n > 0 else "#BBBBBB",
                    ha="center", va="center", zorder=4)
        ax.set_title(f"{equipo} ({sum(conteo.values())} eventos)", fontsize=12)
    fig.suptitle("Actividad por zonas — eventos etiquetados a mano", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    nota = (f"Eventos por zona Z1–Z9 (Z1–Z3 tercio izquierdo, Z7–Z9 derecho). "
            f"Intensidad común a ambos equipos: el máximo es {vmax} eventos "
            f"en una zona. Reutiliza los conteos por zona de arriba.")
    return fig, nota


def _posicion_jugador_formacion(dp, equipo, etiqueta):
    """(x, y) en metros (campo 105x68) para un jugador de la alineación:
    la posición de su puesto en la formación del equipo (espejada para el
    visitante, como la vista previa táctica). Los suplentes no tienen
    puesto en la formación: van al centro de su línea (media de los
    titulares de su misma posición). La etiqueta es "#dorsal Nombre"."""
    local = equipo == dp["eq_local"]
    xi = dp["xi_local"] if local else dp["xi_visitante"]
    sup = dp["sup_local"] if local else dp["sup_visitante"]
    form = dp.get("form_local" if local else "form_visitante", "1-4-4-2")
    coords = FORMACIONES_COORDS.get(form) or FORMACIONES_COORDS["1-4-4-2"]
    idx = None
    pos = "MED"
    for i, j in enumerate(xi):
        if f"#{j['dorsal']} {j['nombre']}" == etiqueta:
            idx = i
            pos = j.get("pos", "MED")
            break
    if idx is None:
        for j in sup:
            if f"#{j['dorsal']} {j['nombre']}" == etiqueta:
                pos = j.get("pos", "MED")
                break
    if idx is not None and idx < len(coords):
        x_pct, y_pct = coords[idx]
    else:
        xs = [coords[i][0] for i, j in enumerate(xi)
              if j.get("pos") == pos and i < len(coords)]
        ys = [coords[i][1] for i, j in enumerate(xi)
              if j.get("pos") == pos and i < len(coords)]
        if xs:
            x_pct, y_pct = sum(xs) / len(xs), sum(ys) / len(ys)
        else:
            x_pct = {"POR": 8, "DEF": 22, "MED": 35, "DEL": 46}.get(pos, 35)
            y_pct = 50
    if not local:
        x_pct = 100 - x_pct
    return x_pct / 100 * 105, (100 - y_pct) / 100 * 68


def generar_figura_red_pases(df_eventos, dp):
    """Red de pases por equipo (dos campos lado a lado): nodos = jugadores
    en su posición de formación, líneas con grosor por nº de pases
    completados entre cada pareja. Solo usa pases registrados en vivo con
    pasador y receptor (acción "Pase", Detalle "Completado"); los fallidos
    no entran en la red. Devuelve (fig, nota) o (None, motivo) si no hay
    pases que dibujar."""
    if not MPL_DISPONIBLE:
        return None, "matplotlib no está disponible en este equipo."
    if df_eventos is None or len(df_eventos) == 0 or "Evento" not in df_eventos.columns:
        return None, "Sin eventos registrados en este partido."
    if "Jugador" not in df_eventos.columns or "Receptor" not in df_eventos.columns:
        return None, ("Los eventos de este partido no tienen pasador ni "
                      "receptor: se registraron antes de añadir la acción "
                      "\"Pase\".")
    pases = df_eventos[(df_eventos["Evento"] == "Pase")
                       & (df_eventos["Detalle"] == "Completado")]
    if len(pases) == 0:
        return None, "Sin pases completados con pasador y receptor en este partido."
    eq_local, eq_visitante = dp["eq_local"], dp["eq_visitante"]
    color_equipo = {eq_local: dp.get("color_local", "#1E88E5"),
                    eq_visitante: dp.get("color_visitante", "#E53935")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    alguna = False
    for ax, equipo in ((axes[0], eq_local), (axes[1], eq_visitante)):
        _campo_en_ax(ax)
        sub = pases[pases["Equipo"] == equipo] if "Equipo" in pases.columns else pases.iloc[0:0]
        pares = {}
        implicados = {}
        for _, ev in sub.iterrows():
            a = str(ev.get("Jugador", "") or "").strip()
            b = str(ev.get("Receptor", "") or "").strip()
            if not a or not b or a == b:
                continue
            pares[(a, b)] = pares.get((a, b), 0) + 1
            implicados[a] = implicados.get(a, 0) + 1
            implicados[b] = implicados.get(b, 0) + 1
        n_pases = sum(pares.values())
        if not pares:
            ax.text(52.5, 34, "Sin pases\ncompletados", ha="center", va="center",
                    fontsize=11, color="#888888")
            ax.set_title(f"{equipo} (0 pases)", fontsize=12)
            continue
        alguna = True
        max_n = max(pares.values())
        for (a, b), n in sorted(pares.items(), key=lambda kv: kv[1]):
            xa, ya = _posicion_jugador_formacion(dp, equipo, a)
            xb, yb = _posicion_jugador_formacion(dp, equipo, b)
            ax.plot([xa, xb], [ya, yb], color=color_equipo[equipo],
                    lw=0.8 + 3.0 * (n / max_n), alpha=0.75, zorder=3,
                    solid_capstyle="round")
            ax.text((xa + xb) / 2, (ya + yb) / 2 + 2.2, str(n), fontsize=9,
                    fontweight="bold", color="#333333", ha="center",
                    va="bottom", zorder=4)
        for etiqueta, total in implicados.items():
            x, y = _posicion_jugador_formacion(dp, equipo, etiqueta)
            ax.scatter([x], [y], s=180 + 90 * min(total, 12),
                       c=color_equipo[equipo], edgecolors="white",
                       linewidths=1.5, zorder=5)
            # Etiqueta corta (dorsal + primer nombre) para que no se
            # solapen las de jugadores cercanos en la misma línea.
            _partes = etiqueta.split(" ", 1)
            _corto = _partes[0] + (" " + _partes[1].split()[0]
                                   if len(_partes) > 1 else "")
            ax.text(x, y - 4.5, _corto, fontsize=8, ha="center",
                    va="top", zorder=6)
        ax.set_title(f"{equipo} ({n_pases} pases)", fontsize=12)
    if not alguna:
        plt.close(fig)
        return None, "Sin pases completados con pasador y receptor en este partido."
    fig.suptitle("Red de pases — solo pases completados, registrados en vivo",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    nota = ("Nodos en la posición de su formación (suplentes: centro de su "
            "línea); el tamaño es el nº de pases en los que participa el "
            "jugador y el grosor, los pases entre cada pareja. La zona del "
            "receptor no se registra: la red no indica dónde se recibió "
            "cada pase.")
    return fig, nota


def leer_metadatos_tracking(ruta):
    """Metadatos del tracking sin cargar las muestras del partido en RAM."""
    try:
        datos = read_tracking_header(ruta)
        teams = datos.get("teams") or {}
        dur = float(datos.get("match_duration_s") or 0)
        stamp = os.path.getmtime(ruta)
        return {
            "partido": f"{teams.get('local', '?')} vs {teams.get('visitante', '?')}",
            "fecha": datetime.fromtimestamp(stamp).strftime("%d/%m/%Y"),
            "duracion": _mmss(dur),
            "muestras": datos.get("samples_count", "N/D"),
            "archivo": Path(ruta).name,
            "ruta": str(ruta),
            "video_filename": datos.get("video_filename") or "",
            "mtime": stamp,
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None


def buscar_biblioteca(carpeta_app):
    """Deduplicar copias del mismo partido sin cargar los samples."""
    biblioteca = []
    for ruta in sorted(Path(carpeta_app).glob("**/tracking_*.json")):
        meta = leer_metadatos_tracking(ruta)
        if meta is not None:
            biblioteca.append(meta)
    return deduplicate_library(biblioteca)


def _muestra_cercana(muestras, tiempos, t_frame):
    """Muestra con t más cercano (y <= t_frame) por búsqueda binaria."""
    idx = bisect.bisect_right(tiempos, t_frame) - 1
    return muestras[idx] if idx >= 0 else None


def _dibujar_overlay_cv2(frame, tracking, muestras, tiempos, t_frame, ancho, alto):
    """Anillos a los pies, confianza YOLO y trayectoria del balón.
    p = [x, y, w, h, code, track_id, confianza?]. La confianza
    mide detección, no continuidad de identidad ni calibración."""
    muestra = _muestra_cercana(muestras, tiempos, t_frame)
    if muestra is None:
        return
    colors = tracking.get("colors") or {}
    tc = tracking.get("team_code") or {}
    mapa_color = {
        tc.get("local", 0): colors.get("local", "#1E88E5"),
        tc.get("visitante", 1): colors.get("visitante", "#E53935"),
        tc.get("gk_local", 2): colors.get("gk_local", "#FFEB3B"),
        tc.get("gk_visitante", 3): colors.get("gk_visitante", "#FF9800"),
        tc.get("arbitro", 4): colors.get("arbitro", "#000000"),
    }
    codigos_portero = {tc.get("gk_local", 2), tc.get("gk_visitante", 3)}
    grosor = max(2, ancho // 640)
    # Trayectoria del balón: últimos 3 s (como la pantalla manual).
    pts = []
    idx = bisect.bisect_right(tiempos, t_frame) - 1
    for i in range(idx, -1, -1):
        if t_frame - float(muestras[i].get("t", 0) or 0) > 3:
            break
        b = muestras[i].get("b")
        if b:
            pts.append((int(float(b[0]) / 100 * ancho), int(float(b[1]) / 100 * alto)))
    if len(pts) > 1:
        for j in range(len(pts) - 1):
            alfa = 0.85 * (1 - j / len(pts))
            color = (int(255 * alfa), int(139 * alfa), int(61 * alfa))
            cv2.line(frame, pts[j], pts[j + 1], color, grosor)
    # Jugadores: anillos finos bajo los pies, sin ocultar el cuerpo.
    for per in muestra.get("p") or []:
        if len(per) < 5:
            continue
        try:
            x, y, w, h, code = (float(v) for v in per[:5])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, w, h, code)):
            continue
        col = _hex_a_bgr(mapa_color.get(int(code), "#888888"))
        cx = int(x / 100 * ancho)
        cy = int((y + h / 2) / 100 * alto) - 2
        radius_x = max(7, min(24, int(w / 100 * ancho * .63)))
        radius_y = max(4, min(10, int(radius_x * .36)))
        cy = max(radius_y, min(alto - radius_y, cy))
        cv2.ellipse(frame, (cx, cy), (radius_x, radius_y),
                    0, 0, 360, col,
                    4 if int(code) in codigos_portero else 3,
                    cv2.LINE_AA)
    # Balón: círculo + triángulo amarillo (como la pantalla manual).
    b = muestra.get("b")
    if b:
        bx, by = int(float(b[0]) / 100 * ancho), int(float(b[1]) / 100 * alto)
        tri_b = np.array([[bx - 7, by - 24], [bx + 7, by - 24], [bx, by - 12]], dtype=np.int32)
        cv2.fillPoly(frame, [tri_b], (31, 210, 255))
        cv2.circle(frame, (bx, by), max(5, ancho // 160), (31, 210, 255), -1)
        cv2.circle(frame, (bx, by), max(5, ancho // 160), (17, 17, 17), 2)


def render_clip_cv2(ruta_video, tracking, inicio_s, fin_s, progreso_cb=None):
    """Renderiza el clip [inicio_s, fin_s] (segundos de vídeo) con el
    overlay del tracking y devuelve los bytes del .mp4 (codec mp4v).
    Lanza ValueError si el vídeo no se puede abrir."""
    if not CV2_DISPONIBLE:
        raise RuntimeError("OpenCV (cv2) no está disponible.")
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el vídeo: {ruta_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    muestras = tracking.get("samples") or []
    tiempos = [float(s.get("t", 0) or 0) for s in muestras]
    cap.set(cv2.CAP_PROP_POS_MSEC, float(inicio_s) * 1000.0)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        vw = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (ancho, alto))
        n_total = max(1, int((fin_s - inicio_s) * fps))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_frame = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t_frame > fin_s:
                break
            _dibujar_overlay_cv2(frame, tracking, muestras, tiempos,
                                 t_frame, ancho, alto)
            vw.write(frame)
            n += 1
            if progreso_cb is not None:
                progreso_cb(min(1.0, n / n_total), f"Fotograma {n}/{n_total}")
        vw.release()
        if n == 0:
            raise ValueError("No se leyó ningún fotograma en ese tramo.")
        return Path(tmp.name).read_bytes()
    finally:
        cap.release()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def cortar_clip_ffmpeg(ruta_video, inicio_s, fin_s, salida):
    """Corte en bruto con ffmpeg (sin overlay). Devuelve True si va bien."""
    if shutil.which("ffmpeg") is None:
        return False
    cmd = ["ffmpeg", "-y", "-ss", str(inicio_s), "-to", str(fin_s),
           "-i", str(ruta_video), "-c", "copy", str(salida)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0 and Path(salida).is_file()



# --- 4. PANTALLA 1: MENÚ PRINCIPAL (ALINEACIONES Y CONFIGURACIÓN) ---
if st.session_state.pagina == "alineaciones":
    dp = st.session_state.datos_partido
    
    top_izq, top_der = st.columns([4, 1])
    with top_izq:
        st.markdown("##### 📋 Datos del Encuentro")
        c1, c2, c3, c4 = st.columns(4)
        dp["temporada"] = c1.text_input("Temporada", dp["temporada"])
        dp["competicion"] = c2.text_input("Competición", dp["competicion"])
        dp["fecha"] = c3.date_input("Fecha", dp["fecha"])
        
        # Selector de Jornada 1 a 30
        jornadas_opciones = [f"Jornada {i}" for i in range(1, 31)]
        if dp["descripcion"] not in jornadas_opciones:
            dp["descripcion"] = jornadas_opciones[0]
        dp["descripcion"] = c4.selectbox("Jornada", jornadas_opciones, index=jornadas_opciones.index(dp["descripcion"]))
        
    with top_der:
        st.write("")
        st.write("")
        if st.button("Siguiente ➡️", type="primary", use_container_width=True):
            st.session_state.pagina = "partido"
            st.rerun()

    st.divider()

    # Configuración de Equipos
    inf_izq, inf_der = st.columns(2)
    lista_formaciones = list(FORMACIONES_COORDS.keys())
    
    with inf_izq:
        st.markdown("##### ⚙️ Configuración Equipo Local")
        b1, b2, b3 = st.columns([2, 1, 2])
        nuevo_local = b1.selectbox("Equipo Local", LISTA_EQUIPOS_NAVARRA, index=LISTA_EQUIPOS_NAVARRA.index(dp["eq_local"]) if dp["eq_local"] in LISTA_EQUIPOS_NAVARRA else 0)
        if nuevo_local != dp["eq_local"]:
            dp["eq_local"] = nuevo_local
            dp["xi_local"], dp["sup_local"] = obtener_plantilla_inicial(nuevo_local)
            st.rerun()
            
        dp["color_local"] = b2.color_picker("Color Local", dp["color_local"])
        dp["form_local"] = b3.selectbox("Táctica Local", lista_formaciones, index=lista_formaciones.index(dp["form_local"]) if dp["form_local"] in lista_formaciones else 0)

    with inf_der:
        st.markdown("##### ⚙️ Configuración Equipo Visitante")
        b4, b5, b6 = st.columns([2, 1, 2])
        nuevo_vis = b4.selectbox("Equipo Visitante", LISTA_EQUIPOS_NAVARRA, index=LISTA_EQUIPOS_NAVARRA.index(dp["eq_visitante"]) if dp["eq_visitante"] in LISTA_EQUIPOS_NAVARRA else 1)
        if nuevo_vis != dp["eq_visitante"]:
            dp["eq_visitante"] = nuevo_vis
            dp["xi_visitante"], dp["sup_visitante"] = obtener_plantilla_inicial(nuevo_vis)
            st.rerun()
            
        dp["color_visitante"] = b5.color_picker("Color Visitante", dp["color_visitante"])
        dp["form_visitante"] = b6.selectbox("Táctica Visitante", lista_formaciones, index=lista_formaciones.index(dp["form_visitante"]) if dp["form_visitante"] in lista_formaciones else 1)

    # Colores de porteros y árbitro (para el vídeo): el portero cuenta como
    # su equipo y su posición marca qué portería se ataca; el árbitro no
    # cuenta para posesión ni agrupación. Tienen que ser distintos de los
    # colores de campo; si el portero cambia de camiseta, se cambia aquí.
    st.markdown("##### 🧤 Colores de porteros y árbitro (para el vídeo)")
    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        dp["color_portero_local"] = st.color_picker(
            "Portero Local", dp.get("color_portero_local", "#FFEB3B"))
    with c_p2:
        dp["color_portero_visitante"] = st.color_picker(
            "Portero Visitante", dp.get("color_portero_visitante", "#FF9800"))
    with c_p3:
        dp["color_arbitro"] = st.color_picker(
            "Árbitro", dp.get("color_arbitro", "#000000"))

    st.divider()

    # Configuración de Jugadores (Titulares y 7 Suplentes con selector de jugador del Excel y posición)
    col_jug_local, col_campo_central, col_jug_visitante = st.columns([1.5, 3, 1.5])

    def renderizar_convocatoria(eq_nombre, xi_list, sup_list, prefix):
        st.markdown(f"**Convocatoria: {eq_nombre}**")
        
        # Obtener nombres disponibles en la plantilla de ese equipo para el selectbox
        nombres_disponibles = [j["nombre"] for j in PLANTILLAS_EQUIPOS.get(eq_nombre, [])]
        if not nombres_disponibles:
            nombres_disponibles = [f"Jugador {i}" for i in range(1, 25)]

        st.caption("XI Inicial (Titulares)")
        for idx, jugador in enumerate(xi_list):
            c_d, c_n, c_p = st.columns([1, 2.8, 1.4])
            jugador["dorsal"] = c_d.number_input("D", value=int(jugador.get("dorsal", idx+1)), key=f"{prefix}_xi_d_{idx}", label_visibility="collapsed")
            
            # Selector de nombre basado en la plantilla del CSV/Excel
            current_name = jugador.get("nombre", nombres_disponibles[0])
            if current_name not in nombres_disponibles:
                nombres_disponibles.append(current_name)
            jugador["nombre"] = c_n.selectbox("Nom", nombres_disponibles, index=nombres_disponibles.index(current_name), key=f"{prefix}_xi_n_{idx}", label_visibility="collapsed")
            
            # Selector de Posición
            current_pos = jugador.get("pos", "MED")
            jugador["pos"] = c_p.selectbox("Pos", POSICIONES, index=POSICIONES.index(current_pos) if current_pos in POSICIONES else 0, key=f"{prefix}_xi_p_{idx}", label_visibility="collapsed")
        
        st.caption("Banquillo (7 Suplentes)")
        for idx, jugador in enumerate(sup_list):
            c_d, c_n, c_p = st.columns([1, 2.8, 1.4])
            jugador["dorsal"] = c_d.number_input("D", value=int(jugador.get("dorsal", idx+12)), key=f"{prefix}_sup_d_{idx}", label_visibility="collapsed")
            
            current_name = jugador.get("nombre", nombres_disponibles[min(idx+11, len(nombres_disponibles)-1)])
            if current_name not in nombres_disponibles:
                nombres_disponibles.append(current_name)
            jugador["nombre"] = c_n.selectbox("Nom", nombres_disponibles, index=nombres_disponibles.index(current_name), key=f"{prefix}_sup_n_{idx}", label_visibility="collapsed")
            
            current_pos = jugador.get("pos", "MED")
            jugador["pos"] = c_p.selectbox("Pos", POSICIONES, index=POSICIONES.index(current_pos) if current_pos in POSICIONES else 0, key=f"{prefix}_sup_p_{idx}", label_visibility="collapsed")

    with col_jug_local:
        renderizar_convocatoria(dp["eq_local"], dp["xi_local"], dp["sup_local"], "l")

    with col_campo_central:
        st.markdown("##### 🏟️ Vista Previa Táctica del XI Inicial")
        fichas_html = ""
        for i, (x, y) in enumerate(FORMACIONES_COORDS[dp["form_local"]], 1):
            fichas_html += f'<div style="position:absolute; left:{x}%; top:{y}%; transform:translate(-50%,-50%); width:24px; height:24px; background:{dp["color_local"]}; color:white; border:2px solid white; border-radius:50%; text-align:center; line-height:20px; font-size:11px; font-weight:bold;">{i}</div>'
        for i, (x, y) in enumerate(FORMACIONES_COORDS[dp["form_visitante"]], 1):
            fichas_html += f'<div style="position:absolute; left:{100-x}%; top:{y}%; transform:translate(-50%,-50%); width:24px; height:24px; background:{dp["color_visitante"]}; color:white; border:2px solid white; border-radius:50%; text-align:center; line-height:20px; font-size:11px; font-weight:bold;">{i}</div>'

        st.markdown(f"""
        <div style="position:relative; width:100%; height:420px; background-color:#2e7d32; border:3px solid white; border-radius:8px; overflow:hidden;">
            <div style="position:absolute; left:50%; top:0; bottom:0; width:2px; background:rgba(255,255,255,0.8);"></div>
            <div style="position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:70px; height:70px; border:2px solid rgba(255,255,255,0.8); border-radius:50%;"></div>
            <div style="position:absolute; left:15px; top:8px; color:white; font-weight:bold; font-size:13px;">{dp["eq_local"]}</div>
            <div style="position:absolute; right:15px; top:8px; color:white; font-weight:bold; font-size:13px;">{dp["eq_visitante"]}</div>
            {fichas_html}
        </div>
        """, unsafe_allow_html=True)

    with col_jug_visitante:
        renderizar_convocatoria(dp["eq_visitante"], dp["xi_visitante"], dp["sup_visitante"], "v")


# --- 5. PANTALLA 2: MENÚ DE JUEGO (PARTIDO) ---
elif st.session_state.pagina == "partido":
    dp = st.session_state.datos_partido

    def segundos_transcurridos():
        transcurrido = st.session_state.tiempo_acumulado
        if st.session_state.inicio_tiempo is not None:
            transcurrido += int(time.time() - st.session_state.inicio_tiempo)
        return transcurrido

    def obtener_minuto():
        transcurrido = segundos_transcurridos()
        return f"{transcurrido // 60:02d}:{transcurrido % 60:02d}"

    @st.fragment(run_every="1s")
    def mostrar_reloj():
        transcurrido = segundos_transcurridos()
        st.metric("Tiempo", obtener_minuto())
        st.progress(min(transcurrido / (90 * 60), 1.0), text="Progreso (90 min)")

    def registrar_evento(accion, sub_accion="", jugador="", receptor=""):
        # Jugador/Receptor solo se rellenan en la acción "Pase" (pasador y
        # receptor); en el resto de acciones quedan vacíos. Los eventos
        # viejos sin estos campos cargan igual (pandas los deja en NaN).
        nuevo_evento = {
            "ID": len(st.session_state.eventos) + 1,
            "Temporada": dp["temporada"],
            "Competicion": dp["competicion"],
            "Fecha": str(dp["fecha"]),
            "Descripcion": dp["descripcion"],
            "Periodo": st.session_state.periodo,
            "Minuto": obtener_minuto(),
            "Equipo": st.session_state.equipo,
            "Zona": st.session_state.zona_actual,
            "Evento": accion,
            "Detalle": sub_accion,
            "Jugador": jugador,
            "Receptor": receptor
        }
        st.session_state.eventos.append(nuevo_evento)

    # Barra superior de control
    col_volver, col_crono, col_periodo, col_equipo, col_finalizar = st.columns([1, 2, 1.2, 2, 1.3])
    with col_volver:
        if st.button("⬅️ Alineaciones", use_container_width=True):
            st.session_state.pagina = "alineaciones"
            st.rerun()

    with col_crono:
        c1, c2, c3 = st.columns(3)
        if c1.button("▶️", use_container_width=True, help="Iniciar"):
            if st.session_state.inicio_tiempo is None:
                st.session_state.inicio_tiempo = time.time()
        if c2.button("⏸️", use_container_width=True, help="Pausar"):
            if st.session_state.inicio_tiempo is not None:
                st.session_state.tiempo_acumulado += int(time.time() - st.session_state.inicio_tiempo)
                st.session_state.inicio_tiempo = None
        if c3.button("⏹️", use_container_width=True, help="Parar"):
            st.session_state.inicio_tiempo = None
            st.session_state.tiempo_acumulado = 0
        mostrar_reloj()

    with col_periodo:
        st.session_state.periodo = st.selectbox("Periodo", ["1ª Parte", "2ª Parte"])

    with col_equipo:
        st.session_state.equipo = st.radio("Equipo:", [dp["eq_local"], dp["eq_visitante"]], horizontal=True)

    with col_finalizar:
        st.write("")
        if st.button("🏁 Finalizar Partido", type="primary", use_container_width=True):
            st.session_state.pagina = "final"
            st.rerun()
        st.write("")
        if st.button("🎥 Vídeo", use_container_width=True):
            st.session_state.pagina = "video"
            st.rerun()

    st.divider()

    # DISTRIBUCIÓN: [Suplentes Izq] | [Campo de Fútbol con Zonas] | [Acciones Der]
    col_suplentes, col_campo_juego, col_acciones_der = st.columns([1.2, 2, 1.2])

    with col_suplentes:
        st.markdown("##### 👥 Banquillo (Suplentes)")
        st.markdown(f"**{dp['eq_local']}**")
        for s in dp["sup_local"]:
            st.text(f"#{s['dorsal']} {s['nombre']} ({s['pos']})")
        st.markdown(f"**{dp['eq_visitante']}**")
        for s in dp["sup_visitante"]:
            st.text(f"#{s['dorsal']} {s['nombre']} ({s['pos']})")

    with col_campo_juego:
        st.markdown(f"##### ⚽ Zona activa: `{st.session_state.zona_actual}`")
        st.caption("Selecciona la zona del campo donde ocurre la jugada:")
        
        zonas = [
            ["Z1 - Def. Izq", "Z4 - Medio Izq", "Z7 - Ataque Izq"],
            ["Z2 - Def. Centro", "Z5 - Medio Centro", "Z8 - Ataque Centro"],
            ["Z3 - Def. Der", "Z6 - Medio Der", "Z9 - Ataque Der"]
        ]
        
        for fila in zonas:
            c_z1, c_z2, c_z3 = st.columns(3)
            for i, nombre_zona in enumerate(fila):
                es_activa = st.session_state.zona_actual == nombre_zona
                tipo_btn = "primary" if es_activa else "secondary"
                target_col = [c_z1, c_z2, c_z3][i]
                if target_col.button(nombre_zona, key=f"zona_{nombre_zona}", type=tipo_btn, use_container_width=True):
                    st.session_state.zona_actual = nombre_zona
                    st.rerun()

    with col_acciones_der:
        st.markdown("##### ⚡ Registrar Acciones")
        if st.button("⚽ Gol", use_container_width=True): registrar_evento("Tiro", "Gol")
        if st.button("🥅 Tiro a Puerta", use_container_width=True): registrar_evento("Tiro", "A puerta")
        if st.button("❌ Tiro Fuera", use_container_width=True): registrar_evento("Tiro", "Fuera")
        if st.button("↗️ Centro al Área", use_container_width=True): registrar_evento("Ataque", "Centro")
        if st.button("⚡ Ataque Posicional", use_container_width=True): registrar_evento("Ataque", "Posicional")
        if st.button("🏃 Contraataque", use_container_width=True): registrar_evento("Ataque", "Transición")
        if st.button("🛡️ Recuperación", use_container_width=True): registrar_evento("Defensa", "Recuperación")
        if st.button("⚠️ Pérdida", use_container_width=True): registrar_evento("Posesión", "Pérdida")
        if st.button("🚩 Córner", use_container_width=True): registrar_evento("ABP", "Córner")
        if st.button("🛑 Falta", use_container_width=True): registrar_evento("Falta", "Cometida")
        if st.button("🟨 Tarjeta", use_container_width=True): registrar_evento("Disciplina", "Tarjeta")

        st.divider()
        st.markdown("##### 🔄 Pase")
        st.caption("Pasador y receptor del equipo seleccionado (XI + suplentes).")
        _eq_pase = st.session_state.equipo
        if _eq_pase == dp["eq_local"]:
            _xi_pase, _sup_pase = dp["xi_local"], dp["sup_local"]
        else:
            _xi_pase, _sup_pase = dp["xi_visitante"], dp["sup_visitante"]
        _jug_pase = ([f"#{j['dorsal']} {j['nombre']}" for j in _xi_pase]
                     + [f"#{j['dorsal']} {j['nombre']}" for j in _sup_pase])
        _sup_pase_set = {f"#{j['dorsal']} {j['nombre']}" for j in _sup_pase}
        _pasador = st.selectbox(
            "Pasador", _jug_pase, key=f"pase_pasador_{_eq_pase}",
            format_func=lambda v: v + " (supl.)" if v in _sup_pase_set else v)
        _receptor = st.selectbox(
            "Receptor", _jug_pase, key=f"pase_receptor_{_eq_pase}",
            index=1 if len(_jug_pase) > 1 else 0,
            format_func=lambda v: v + " (supl.)" if v in _sup_pase_set else v)
        _res_pase = st.selectbox("Resultado", ["Completado", "Fallido"],
                                 key="pase_resultado")
        if st.button("🔄 Registrar pase", use_container_width=True):
            if _pasador == _receptor:
                st.warning("El pasador y el receptor no pueden ser el mismo jugador.")
            else:
                registrar_evento("Pase", _res_pase, jugador=_pasador, receptor=_receptor)

    st.divider()

    if st.session_state.eventos:
        df = pd.DataFrame(st.session_state.eventos)
        b_undo, _ = st.columns([1, 3])
        if b_undo.button("↩️ Deshacer último evento", use_container_width=True):
            st.session_state.eventos.pop()
            st.rerun()
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)


# --- 6. PANTALLA 3: MENÚ FINAL / POST-PARTIDO ---
elif st.session_state.pagina == "final":
    dp = st.session_state.datos_partido

    col_b1, col_b1b, col_b2 = st.columns([1, 1.2, 3])
    with col_b1:
        if st.button("⬅️ Volver al Partido", use_container_width=True):
            st.session_state.pagina = "partido"
            st.rerun()
    with col_b1b:
        if st.button("🎥 Analizar Vídeo", use_container_width=True):
            st.session_state.pagina = "video"
            st.rerun()
    with col_b2:
        st.markdown(f"### 📊 Menú Final y Post-Partido: {dp['eq_local']} vs {dp['eq_visitante']} ({dp['descripcion']})")

    st.divider()

    if not st.session_state.eventos:
        st.warning("⚠️ No hay eventos registrados en este partido.")
    else:
        df_base = pd.DataFrame(st.session_state.eventos)

        st.markdown("#### 🛠️ Desglose y Modificación del CSV")
        st.caption("Edita cualquier celda antes de generar el archivo definitivo para Power BI[cite: 1].")
        
        df_editado = st.data_editor(df_base, num_rows="dynamic", use_container_width=True, key="editor_csv_final")

        st.divider()
        st.markdown("#### 📈 Estadísticas Resumidas")
        
        if "Equipo" in df_editado.columns:
            equipos = df_editado["Equipo"].unique()
            cols_m = st.columns(len(equipos) if len(equipos) > 0 else 1)
            
            for idx, eq in enumerate(equipos):
                df_eq = df_editado[df_editado["Equipo"] == eq]
                goles = len(df_eq[(df_eq["Evento"] == "Tiro") & (df_eq["Detalle"] == "Gol")]) if "Evento" in df_editado.columns else 0
                tiros = len(df_eq[df_eq["Evento"] == "Tiro"]) if "Evento" in df_editado.columns else 0
                recup = len(df_eq[df_eq["Evento"] == "Defensa"]) if "Evento" in df_editado.columns else 0
                
                with cols_m[idx]:
                    st.metric(label=f"Club: {eq}", value=f"⚽ {goles} Goles", delta=f"{tiros} Tiros | 🛡️ {recup} Recup.")

        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.markdown("**📊 Conteo por Tipo de Evento:**")
            if "Evento" in df_editado.columns:
                res_ev = df_editado["Evento"].value_counts().reset_index()
                res_ev.columns = ["Evento", "Total"]
                st.dataframe(res_ev, use_container_width=True, hide_index=True)
        
        with col_st2:
            st.markdown("**📍 Conteo por Zona del Campo:**")
            if "Zona" in df_editado.columns:
                res_zn = df_editado["Zona"].value_counts().reset_index()
                res_zn.columns = ["Zona", "Total"]
                st.dataframe(res_zn, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### 🎯 Visualizaciones (desde lo etiquetado a mano)")
        st.caption("Solo con lo registrado en vivo en esta pantalla. La red de "
                   "pases usa los pases con pasador y receptor de la acción "
                   "\"Pase\"; sin pases registrados no se dibuja.")

        fig_tiros, nota_tiros = generar_figura_mapa_tiros(
            df_editado, dp["eq_local"], dp["eq_visitante"],
            dp.get("color_local", "#1E88E5"), dp.get("color_visitante", "#E53935"))
        if fig_tiros is None:
            st.info(f"🎯 Mapa de tiros: {nota_tiros}")
        else:
            _mostrar_figura_streamlit(fig_tiros)
            st.caption(nota_tiros)
            plt.close(fig_tiros)

        fig_zonas, nota_zonas = generar_figura_mapa_zonas(
            df_editado, dp["eq_local"], dp["eq_visitante"],
            dp.get("color_local", "#1E88E5"), dp.get("color_visitante", "#E53935"))
        if fig_zonas is None:
            st.info(f"📍 Actividad por zonas: {nota_zonas}")
        else:
            _mostrar_figura_streamlit(fig_zonas)
            st.caption(nota_zonas)
            plt.close(fig_zonas)

        fig_red, nota_red = generar_figura_red_pases(df_editado, dp)
        if fig_red is None:
            st.info(f"🔗 Red de pases: {nota_red}")
        else:
            _mostrar_figura_streamlit(fig_red)
            st.caption(nota_red)
            plt.close(fig_red)

        st.info("📈 Tendencias por jornada: no disponibles. El CSV solo se "
                "descarga, no se guarda en la app, así que no hay histórico de "
                "jornadas anteriores con el que comparar.")

        st.divider()

        col_down1, col_down2 = st.columns([2, 2])
        with col_down1:
            csv_final = df_editado.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar CSV Definitivo para Power BI",
                data=csv_final,
                file_name=f"FINAL_{dp['eq_local']}_vs_{dp['eq_visitante']}_{dp['fecha']}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        with col_down2:
            if st.button("🔄 Borrar datos y empezar nuevo partido", use_container_width=True):
                st.session_state.eventos = []
                st.session_state.pagina = "alineaciones"
                st.rerun()

# --- 7. PANTALLA 4: VÍDEO (análisis manual) ---
elif st.session_state.pagina == "video":
    dp = st.session_state.datos_partido
    carpeta_app = Path(__file__).resolve().parent

    st.markdown("### Análisis de vídeo")
    st.caption("Selecciona el vídeo, revisa el campo y analiza el partido.")

    # --- Generar tracking al proporcionar el vídeo (automático) ---
    # Al pegar la ruta del vídeo, la app lanza el análisis YOLO en ese
    # mismo momento (la subida es el disparador, no hay botón de analizar):
    # el tracking que se enseña abajo es el recién generado de ESE vídeo.
    # Si el vídeo ya tiene su tracking (mismo stem), se carga directamente
    # sin re-analizar. El botón de re-analizar es solo un fallback.
    st.markdown("##### 🎥 Vídeo del partido")
    st.caption("Los datos del seguimiento se generan y cargan automáticamente.")
    c_gen1, c_gen2, c_gen3 = st.columns([3, 1, 1.4])
    with c_gen1:
        ruta_gen = st.text_input(
            "Ruta local del vídeo",
            value=st.session_state.get("video_gen_path", ""),
            key="video_gen_path_input",
            placeholder=r"C:\Videos\partido.mp4",
        )
        # Windows pega la ruta entre comillas cuando tiene espacios.
        ruta_gen = ruta_gen.strip().strip('"').strip("'").strip()
        st.session_state.video_gen_path = ruta_gen
    with c_gen2:
        st.session_state.video_gen_kickoff = st.text_input(
            "Saque inicial (MM:SS)",
            value=st.session_state.get("video_gen_kickoff", "00:00"),
            key="video_gen_kickoff_input",
            help="El análisis empieza en este minuto: corta el vídeo "
                 "ahí (el calentamiento anterior no se analiza). El "
                 "minuto de partido 00:00 es este saque. Con 00:00 se "
                 "analiza desde el principio del vídeo.",
        )
    with c_gen3:
        alcance_gen = st.selectbox(
            "Alcance",
            ["Primeros 10 minutos (prueba)", "Partido completo"],
            key="video_gen_alcance",
        )
    with st.expander("Ajustes avanzados de procesamiento", expanded=False):
        sample_fps_gen = st.selectbox(
            "FPS del análisis", [10, 5], index=0,
            key="video_gen_sample_fps",
            help="10 FPS conserva mejor el balón; 5 FPS reduce carga de CPU.")
        bg_video = st.checkbox(
            "Procesar en segundo plano", value=True,
            key="video_background",
            help="Un trabajo a la vez. El progreso se pierde si se reinicia Streamlit.")
    if bg_video and not hasattr(st, "fragment"):
        st.info("Esta versión de Streamlit no admite actualización automática "
                "del progreso. Se usará el modo clásico.")
        bg_video = False

    # Previsualizar sin lanzar YOLO: comprobar que la máscara excluye
    # graderío y personal fuera de las líneas antes de analizar horas.
    if ruta_gen and os.path.isfile(ruta_gen):
        if st.button("Previsualizar límites del campo", key="video_preview_field"):
            try:
                from pitch_filter import render_pitch_preview
                if not CV2_DISPONIBLE:
                    raise RuntimeError("OpenCV no está disponible.")
                tiempo_inicio = parse_mmss(
                    st.session_state.video_gen_kickoff or "00:00")
                video_pre = cv2.VideoCapture(ruta_gen)
                try:
                    if not video_pre.isOpened():
                        raise RuntimeError("No se pudo abrir el vídeo.")
                    video_pre.set(cv2.CAP_PROP_POS_MSEC, tiempo_inicio * 1000)
                    ok_pre, frame_pre = video_pre.read()
                finally:
                    video_pre.release()
                if not ok_pre:
                    raise RuntimeError("No se pudo obtener el fotograma del saque inicial.")
                preview, campo_detectado, mensaje_campo = render_pitch_preview(frame_pre)
                st.image(preview, caption="Línea verde: contorno del césped detectado.",
                         use_container_width=True)
                if campo_detectado:
                    st.success(mensaje_campo)
                else:
                    st.warning(mensaje_campo)
                st.caption("Esta vista no ejecuta YOLO. Comprueba que el contorno "
                           "no incluye espectadores, entrenadores ni zona técnica.")
            except Exception as error:
                st.error(f"No se pudo previsualizar el campo: {error}")
    trabajo_actual = st.session_state.get("video_bg_job")
    ocupado = False
    if trabajo_actual:
        from video_jobs import get_video_job
        snapshot = get_video_job(trabajo_actual)
        ocupado = bool(snapshot and snapshot["state"] in ("queued", "running"))
    iniciar_analisis = st.button(
        "Analizar vídeo con YOLO", key="video_launch_analysis",
        disabled=not (ruta_gen and os.path.isfile(ruta_gen)) or ocupado,
        help="Inicia el tracking con la configuración actual. Usa primero "
             "los 10 minutos de prueba para revisar el filtrado del campo.")

    # ¿Ya existe tracking para este vídeo? (mismo stem, en la carpeta de
    # la app, junto al vídeo o en su carpeta de clips).
    tracking_gen = None
    tracking_antiguo = None
    tracking_incompatible = None
    if ruta_gen:
        stem_gen = Path(ruta_gen).stem
        for cand in (
            carpeta_app / f"{stem_gen}_tracking.json",
            Path(ruta_gen).parent / f"{stem_gen}_tracking.json",
            Path(ruta_gen).parent / f"{stem_gen}_clips" / f"{stem_gen}_tracking.json",
        ):
            if not cand.is_file():
                continue
            try:
                info = inspect_tracking(cand, Path(ruta_gen).name)
            except (OSError, ValueError, TypeError):
                continue
            if info["state"] == "actual":
                tracking_gen = cand
                break
            if info["state"] == "antiguo":
                tracking_antiguo = cand
            elif info["state"] == "otro_video":
                tracking_incompatible = cand
    if tracking_antiguo is not None and tracking_gen is None:
        st.warning(
            f"Existe un tracking antiguo ({tracking_antiguo.name}). "
            "No se reutilizará automáticamente: no contiene el filtrado "
            "de campo ni todas las métricas actuales.")
    if tracking_incompatible is not None and tracking_gen is None:
        st.error(
            f"El tracking {tracking_incompatible.name} pertenece a otro vídeo. "
            "No se cargará automáticamente.")
    if tracking_gen is None and st.session_state.get("video_analizado_path") == ruta_gen:
        st.session_state.pop("video_analizado_path", None)
        if st.session_state.get("tracking_del_video"):
            st.session_state.pop("tracking_del_video", None)
    if iniciar_analisis:
        # Reprocesar incluso si ya existe un JSON antiguo o un resultado
        # en caché. No reutilizar el ID de un trabajo ya completado.
        for key in ("video_analizado_path", "video_fallo_path",
                    "video_bg_signature", "video_bg_job", "tracking_del_video"):
            st.session_state.pop(key, None)
    forzar_gen = st.session_state.pop("video_forzar", False) or iniciar_analisis
    # Si el JSON aparece antes de finalizar informes, continuar mostrando
    # el trabajo de fondo en lugar de dar el análisis por completado.
    pendiente_bg = False
    if bg_video and st.session_state.get("video_bg_job"):
        from video_jobs import get_video_job
        estado_bg = get_video_job(st.session_state.video_bg_job)
        firma_bg = st.session_state.get("video_bg_signature")
        pendiente_bg = bool(
            firma_bg and firma_bg[0] == ruta_gen
            and estado_bg and estado_bg["state"] in ("queued", "running"))
    ya_cargado_gen = bool(ruta_gen) and st.session_state.get("video_analizado_path") == ruta_gen
    fallo_gen = bool(ruta_gen) and st.session_state.get("video_fallo_path") == ruta_gen
    if ruta_gen and not os.path.isfile(ruta_gen):
        st.warning("⚠️ Esa ruta de vídeo no existe en este equipo. Revisa la ruta.")
    elif ruta_gen and tracking_gen is not None and not forzar_gen and not ya_cargado_gen and not pendiente_bg:
        # Tracking ya generado: cargarlo sin re-analizar.
        st.session_state.video_analizado_path = ruta_gen
        st.session_state.pop("video_fallo_path", None)
        st.session_state["tracking_del_video"] = str(tracking_gen)
        st.success(f"✅ Tracking ya generado para este vídeo ({tracking_gen.name}). Cargado en la pantalla manual.")
    elif (ruta_gen and (tracking_gen is None or forzar_gen or pendiente_bg)
          and not ya_cargado_gen and not fallo_gen
          and (iniciar_analisis or forzar_gen or pendiente_bg)):
        # No hay tracking: lanzar el análisis automáticamente.
        if not VIDEO_DISPONIBLE:
            st.error(f"⚠️ El análisis no está disponible en este equipo: {VIDEO_ERROR}. "
                     "Genera el tracking con el notebook de Colab.")
            st.session_state.video_fallo_path = ruta_gen
        else:
            try:
                kickoff_gen = parse_mmss(st.session_state.video_gen_kickoff or "00:00")
            except (ValueError, TypeError):
                st.error("El saque inicial debe ser MM:SS, por ejemplo 17:06.")
                st.session_state.video_fallo_path = ruta_gen
                kickoff_gen = None
            if kickoff_gen is not None:
                if bg_video:
                    from video_jobs import submit_video_job, get_video_job
                    # No compartir estados entre sesiones ni crear jobs
                    # nuevos en cada rerun de Streamlit.
                    signature = (
                        ruta_gen, kickoff_gen, alcance_gen, sample_fps_gen,
                        dp["eq_local"], dp["eq_visitante"],
                        dp["color_local"], dp["color_visitante"],
                        dp.get("color_portero_local", "#FFEB3B"),
                        dp.get("color_portero_visitante", "#FF9800"),
                        dp.get("color_arbitro", "#000000"),
                    )
                    if st.session_state.get("video_bg_signature") != signature:
                        parametros = {
                            "video_path": Path(ruta_gen),
                            "color_local": dp["color_local"],
                            "color_visitante": dp["color_visitante"],
                            "color_gk_local": dp.get("color_portero_local", "#FFEB3B"),
                            "color_gk_visitante": dp.get("color_portero_visitante", "#FF9800"),
                            "color_arbitro": dp.get("color_arbitro", "#000000"),
                            "team_local": dp["eq_local"],
                            "team_visitante": dp["eq_visitante"],
                            "sample_fps": sample_fps_gen,
                            "min_confidence": 0.65,
                            "yolo_model": "modelo_propio.pt",
                            "max_minutes": 10.0 if alcance_gen.startswith("Primeros") else None,
                            "kickoff_offset": kickoff_gen,
                            "camera_mode": "seguimiento",
                            "formation_local": dp.get("form_local", ""),
                            "formation_visitante": dp.get("form_visitante", ""),
                            "pitch_model": None,
                            "device": "cpu",
                        }
                        st.session_state.video_bg_job = submit_video_job(
                            analyze_video_full, parametros)
                        st.session_state.video_bg_signature = signature

                    @st.fragment(run_every=3)
                    def mostrar_trabajo_video():
                        job = get_video_job(st.session_state.video_bg_job)
                        if job is None:
                            st.error("El trabajo ya no existe; reinicia el análisis.")
                            st.session_state.video_fallo_path = ruta_gen
                            st.rerun()
                            return
                        st.progress(job.get("progress") or 0.0)
                        st.caption(job.get("message") or "Esperando...")
                        if job["state"] == "queued":
                            st.info("Vídeo en cola: ya hay un análisis en curso.")
                        if job["state"] == "failed":
                            st.error("Falló el análisis: " + str(job.get("error")))
                            st.session_state.video_fallo_path = ruta_gen
                            st.rerun()
                        elif job["state"] == "completed":
                            try:
                                resultado = job.get("result") or {}
                                tracking_resultado = resultado.get("tracking_json")
                                if not tracking_resultado or not os.path.isfile(tracking_resultado):
                                    raise VideoProcessingError(
                                        "No se generó el archivo de tracking.")
                                destino = carpeta_app / f"{Path(ruta_gen).stem}_tracking.json"
                                if Path(tracking_resultado).resolve() != destino.resolve():
                                    shutil.copy(tracking_resultado, destino)
                                st.session_state["tracking_del_video"] = str(destino)
                                st.session_state.video_analizado_path = ruta_gen
                                st.session_state.pop("video_fallo_path", None)
                                st.success("Análisis terminado; tracking listo.")
                            except Exception as error:
                                st.error(f"No se pudo preparar el tracking: {error}")
                                st.session_state.video_fallo_path = ruta_gen
                            st.rerun()

                    mostrar_trabajo_video()
                else:
                    barra_gen = st.progress(0.0)
                    estado_gen = st.empty()
                    def _progreso_gen(mensaje, fraccion):
                        barra_gen.progress(min(1.0, max(0.0, float(fraccion))))
                        estado_gen.write(mensaje)
                    try:
                        informe_gen = analyze_video_full(
                            Path(ruta_gen),
                            color_local=dp["color_local"],
                            color_visitante=dp["color_visitante"],
                            color_gk_local=dp.get("color_portero_local", "#FFEB3B"),
                            color_gk_visitante=dp.get("color_portero_visitante", "#FF9800"),
                            color_arbitro=dp.get("color_arbitro", "#000000"),
                            team_local=dp["eq_local"], team_visitante=dp["eq_visitante"],
                            sample_fps=sample_fps_gen,
                            min_confidence=0.65, yolo_model="modelo_propio.pt",
                            max_minutes=10.0 if alcance_gen.startswith("Primeros") else None,
                            kickoff_offset=kickoff_gen,
                            camera_mode="seguimiento",
                            formation_local=dp.get("form_local", ""),
                            formation_visitante=dp.get("form_visitante", ""),
                            pitch_model=None,
                            progress=_progreso_gen,
                            device="cpu",
                        )
                        # Copiar el tracking JSON a la carpeta de la app para
                        # que la pantalla manual lo encuentre en próximas sesiones.
                        ruta_tracking_gen = informe_gen.get("tracking_json")
                        if ruta_tracking_gen and os.path.isfile(ruta_tracking_gen):
                            destino_gen = carpeta_app / f"{Path(ruta_gen).stem}_tracking.json"
                            if Path(ruta_tracking_gen).resolve() != destino_gen.resolve():
                                shutil.copy(ruta_tracking_gen, destino_gen)
                            tracking_gen = destino_gen
                        st.session_state.video_analizado_path = ruta_gen
                        st.session_state.pop("video_fallo_path", None)
                        if tracking_gen is not None:
                            st.session_state["tracking_del_video"] = str(tracking_gen)
                        estado_gen.write("✅ Análisis completado: tracking generado de este vídeo.")
                        st.rerun()
                    except VideoProcessingError as err:
                        st.error(str(err))
                        st.session_state.video_fallo_path = ruta_gen
                    except Exception as err:  # noqa: BLE001 - mostrar cualquier fallo
                        st.error(f"Error inesperado durante el análisis: {err}")
                        st.session_state.video_fallo_path = ruta_gen
    if (ruta_gen and tracking_gen is None and not pendiente_bg
            and not fallo_gen and not iniciar_analisis and not forzar_gen):
        st.info("Vídeo listo. Revisa los límites del campo y pulsa "
                "«Analizar vídeo con YOLO» para iniciar el procesamiento.")

    # Fallback: re-analizar aunque ya exista tracking o haya fallado antes.
    if ruta_gen and (ya_cargado_gen or fallo_gen):
        if st.button("🔄 Re-analizar vídeo", key="video_gen_reanalizar"):
            st.session_state.pop("video_analizado_path", None)
            st.session_state.pop("video_fallo_path", None)
            st.session_state.video_forzar = True
            st.session_state.pop("video_bg_signature", None)
            st.session_state.pop("video_bg_job", None)
            st.rerun()

    # --- Análisis manual del vídeo ---
    # El análisis manual se hace en esta pantalla (analizador_video.html,
    # junto a match.py). El tracking (tracking_*.json, generado con el
    # notebook de Colab o por script/VM) se lee de la carpeta y se inyecta
    # en el HTML, sin subirlo a mano. Solo falta elegir el vídeo dentro
    # de la pantalla (4,2 GB, no se puede incrustar).
    st.markdown("#### Partido y seguimiento")
    st.caption("El análisis se guarda automáticamente en tu equipo; no necesitas subir archivos JSON.")
    # Qué tracking inyectar.
    # Prioridad: el del vídeo proporcionado arriba (recién generado o ya
    # existente para ese vídeo) > el elegido en la carpeta (tracking_*.json).
    tracking_elegido = None
    ruta_del_video = st.session_state.get("tracking_del_video", "")
    if ruta_del_video and os.path.isfile(ruta_del_video):
        tracking_elegido = Path(ruta_del_video)
    if tracking_elegido is None:
        archivos_tracking = sorted(carpeta_app.glob("tracking_*.json"))
        if len(archivos_tracking) > 1:
            nombres_tracking = [a.name for a in archivos_tracking]
            if st.session_state.get("tracking_elegido") not in nombres_tracking:
                st.session_state.tracking_elegido = nombres_tracking[0]
            st.selectbox(
                "Tracking del partido",
                nombres_tracking,
                key="tracking_elegido",
                help="Hay varios tracking en la carpeta; elige el del partido que vas a analizar.",
            )
            tracking_elegido = carpeta_app / st.session_state.tracking_elegido
        elif len(archivos_tracking) == 1:
            tracking_elegido = archivos_tracking[0]
    # ¿Se puede auto-cargar el vídeo en la pantalla manual? Solo si el
    # tracking inyectado es el del vídeo proporcionado arriba: ese vídeo
    # se sirve por HTTP (ver servir_video) y la pantalla lo carga sola,
    # sin volver a elegirlo. Si el tracking es otro (elegido en la
    # carpeta), el vídeo se elige a mano dentro de la pantalla.
    url_video = ""
    if (tracking_elegido is not None and ruta_del_video
            and Path(ruta_del_video) == tracking_elegido
            and ruta_gen and os.path.isfile(ruta_gen)):
        url_video = servir_video(ruta_gen)
    if tracking_elegido is None:
        st.warning("⚠️ No hay ningún tracking (tracking_*.json) en la carpeta: "
                   "la pantalla cargará solo el vídeo, sin cajas ni estadísticas. "
                   "Proporciona el vídeo arriba para generar su tracking, o genera "
                   "el tracking con el notebook de Colab o por script.")
    elif url_video:
        st.success("✅ Tracking cargado y vídeo listo: el vídeo se carga solo "
                   "en la pantalla manual con el tracking encima.")
    else:
        # El tracking está cargado, pero el vídeo hay que elegirlo dentro
        # de la pantalla manual (el navegador no abre un archivo local
        # solo con su ruta): que quede claro el paso que falta.
        st.info("✅ Tracking cargado. Ahora selecciona el vídeo dentro de la "
                "pantalla manual (botón «Seleccionar vídeo…») para verlo con "
                "el tracking encima.")
    if tracking_elegido is not None:
        try:
            cabecera = read_tracking_header(tracking_elegido)
            calidad = cabecera.get("quality") or {}
        except (OSError, ValueError, TypeError):
            calidad = {}
        if calidad:
            filtro = calidad.get("field_filter") or {}
            with st.expander("Diagnóstico: personas dentro/fuera del campo", expanded=False):
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Personas detectadas", filtro.get("detected", 0))
                q2.metric("Aceptadas", filtro.get("accepted", 0))
                q3.metric("Descartadas", filtro.get("rejected", 0))
                q4.metric("Sin verificar", filtro.get("unverified", 0))
                st.caption(
                    f"Fotogramas con máscara de césped: "
                    f"{calidad.get('field_mask_coverage_pct', 0)} % · "
                    f"Descartes del filtro: "
                    f"{calidad.get('field_filter_rejection_pct', 0)} %. "
                    "Son indicadores de cobertura, no precisión contrastada.")
                for aviso in calidad.get("warnings", []):
                    st.warning(aviso)


    st.markdown("#### Explorar resultados")
    panel_video = st.radio(
        "Sección", ["Vídeo","Jugadores","Táctica","Informes","Biblioteca"],
        horizontal=True, key="mv_video_panel", label_visibility="collapsed",
    )
    if panel_video == "Vídeo":
        mostrar_editor = st.checkbox(
            "Abrir el analizador manual con vídeo y seguimiento",
            value=False, key="video_show_manual_editor",
            help="Carga el tracking completo en el navegador solo al abrirlo. "
                 "Desactivado para mejorar el tiempo de inicio y consumo de memoria.")
        ruta_pantalla = carpeta_app / "analizador_video.html"
        if ruta_pantalla.is_file() and mostrar_editor:
            html_pantalla = ruta_pantalla.read_text(encoding="utf-8")
            datos_tracking = None
            if tracking_elegido is not None:
                try:
                    datos_tracking = cargar_tracking_json(
                        str(tracking_elegido), tracking_elegido.stat().st_mtime)
                    # Un tracking anterior puede traer posiciones calibradas
                    # pero no la lista de momentos IA. Calcularla en memoria,
                    # sin pedir al usuario que suba ni edite ningún JSON.
                    if "suggested_events" not in datos_tracking:
                        from event_candidates import suggest_key_moments
                        datos_tracking = dict(datos_tracking)
                        datos_tracking["suggested_events"] = suggest_key_moments(
                            datos_tracking.get("samples") or [],
                            kickoff_s=float(datos_tracking.get("kickoff_s") or 0))
                    # Inyectar antes del script principal. Se escapa "<" para
                    # que ningún texto del JSON pueda cerrar el <script>. Si el
                    # vídeo se sirve por HTTP (url_video), se pasa también su
                    # URL y su nombre para que la pantalla lo cargue solo.
                    inj = ("<script>window.TRACKING_DATA = "
                           + json.dumps(datos_tracking, ensure_ascii=False,
                                        separators=(",", ":")).replace("<", "\\u003c")
                           + "; window.TRACKING_FILE_NAME = "
                           + json.dumps(tracking_elegido.name, ensure_ascii=False)
                           + ";")
                    if url_video:
                        inj += ("window.VIDEO_URL = " + json.dumps(url_video)
                                + "; window.VIDEO_NAME = "
                                + json.dumps(Path(ruta_gen).name, ensure_ascii=False)
                                + ";")
                    inj += "</script>"
                    html_pantalla = html_pantalla.replace("<!-- TRACKING_DATA -->", inj, 1)
                except (ValueError, OSError) as err:
                    st.warning(f"⚠️ No se pudo cargar {tracking_elegido.name}: {err}. "
                               "La pantalla cargará sin tracking.")
            if isinstance(datos_tracking, dict):
                calidad_tracking = datos_tracking.get("quality") or {}
                if calidad_tracking:
                    with st.expander("Calidad y cobertura del tracking", expanded=False):
                        q1, q2, q3, q4 = st.columns(4)
                        q1.metric("Muestras", calidad_tracking.get("total_samples", 0))
                        q2.metric("Balón YOLO (%)", calidad_tracking.get("ball_yolo_pct", 0))
                        q3.metric("Campo válido (%)", calidad_tracking.get("pitch_coverage_pct", 0))
                        q4.metric("Equipo desconocido (%)", calidad_tracking.get("unknown_team_pct", 0))
                        st.caption(
                            f"IDs distintos: {calidad_tracking.get('unique_track_ids', 0)} · "
                            f"Personas por muestra: {calidad_tracking.get('avg_people_visible', 0)} · "
                            f"Balón estimado (%): {calidad_tracking.get('ball_predicted_pct', 0)} · "
                            f"Homografía por calidad: {calidad_tracking.get('pitch_quality_frames', {})}"
                        )
                        for aviso in calidad_tracking.get("warnings") or []:
                            st.warning(aviso)
                        st.caption(
                            "Estos porcentajes indican disponibilidad de datos, no "
                            "precisión del detector ni continuidad garantizada de los IDs.")
            # Compacidad táctica (idea VEO): si el tracking trae las
            # estadísticas, enseñar la forma de cada equipo por periodo.
            compacidad = {}
            if isinstance(datos_tracking, dict):
                compacidad = datos_tracking.get("stats", {}).get("compacidad", {}) or {}
            tramos_comp = compacidad.get("tramos") if isinstance(compacidad, dict) else None
            if tramos_comp:
                filas_comp = []
                for fila in tramos_comp:
                    loc = fila.get("local") or {}
                    vis = fila.get("visitante") or {}
                    filas_comp.append({
                        "Tramo": fila.get("tramo", ""),
                        "Área local (m²)": loc.get("area_m2"),
                        "Área visitante (m²)": vis.get("area_m2"),
                        "Anchura local (m)": loc.get("anchura_m"),
                        "Anchura visitante (m)": vis.get("anchura_m"),
                        "Profundidad local (m)": loc.get("profundidad_m"),
                        "Profundidad visitante (m)": vis.get("profundidad_m"),
                    })
                with st.expander("📐 Compacidad táctica (forma del equipo por periodo)",
                                 expanded=False):
                    st.caption("Área del Convex Hull de las posiciones en campo: "
                               "menos área = equipo más compacto. Anchura y "
                               "profundidad en metros.")
                    st.dataframe(filas_comp, use_container_width=True, hide_index=True)
            # Geometría táctica ampliada calculada solo con jugadores visibles.
            tactico = (datos_tracking.get("tactical_analysis") or {}
                       if isinstance(datos_tracking, dict) else {})
            periodos_tacticos = tactico.get("periods") or []
            if periodos_tacticos:
                filas_tacticas = []
                nombres_equipo = datos_tracking.get("teams") or {}
                for periodo in periodos_tacticos:
                    inicio = int(periodo.get("period_start_s", 0) // 60)
                    fin = int(periodo.get("period_end_s", 0) // 60)
                    cobertura = periodo.get("coverage") or {}
                    forma = periodo.get("teams") or {}
                    cambios = periodo.get("possession_recoveries_proxy") or {}
                    for lado in ("local", "visitante"):
                        perfil = forma.get(lado) or {}
                        filas_tacticas.append({
                            "Tramo": f"{inicio}-{fin}'",
                            "Equipo": nombres_equipo.get(lado, lado),
                            "Muestras útiles": perfil.get("usable_frames", 0),
                            "Muestras ambos equipos": cobertura.get("both_teams", 0),
                            "Jugadores visibles": perfil.get("visible_players"),
                            "Anchura visible (m)": perfil.get("width_m"),
                            "Profundidad visible (m)": perfil.get("depth_m"),
                            "Área visible (m²)": perfil.get("area_m2"),
                            "Compañero más cercano (m)": perfil.get("nearest_teammate_m"),
                            "Línea defensiva aprox. (m)": perfil.get("defensive_line_m"),
                            "Defensores cerca del balón": perfil.get("defenders_near_ball_10m"),
                            "Cambios de posesión (proxy)": cambios.get(lado, 0),
                        })
                with st.expander("Análisis táctico por tramos (geometría visible)",
                                 expanded=False):
                    st.caption(
                        "La cámara puede mostrar solo una parte del campo. "
                        "Anchura, profundidad y compactación corresponden al "
                        "grupo de jugadores visibles, no necesariamente a los 11. "
                        "La línea defensiva requiere ver al portero. "
                        "Presión y recuperaciones son indicadores aproximados.")
                    st.dataframe(filas_tacticas, use_container_width=True,
                                 hide_index=True)
            components.html(html_pantalla, height=1100, scrolling=True)
        elif ruta_pantalla.is_file():
            st.info("El analizador manual está cerrado para ahorrar memoria. "
                    "Activa la casilla anterior para cargar el vídeo con overlays.")
        else:
            st.warning("No se encontró analizador_video.html junto a match.py; "
                       "la pantalla manual no se puede mostrar.")


    elif panel_video == "Jugadores":
        # --- 7b. ESTADÍSTICAS INDIVIDUALES (solo tracks, no dorsales) ---
        st.divider()
        st.markdown("#### Estadísticas individuales por jugador")
        if tracking_elegido is not None:
            try:
                datos_ind = cargar_tracking_json(
                    str(tracking_elegido), tracking_elegido.stat().st_mtime)
            except (OSError, ValueError) as error:
                datos_ind = {}
                st.warning(f"No se pudieron cargar las métricas: {error}")
            perfilado = datos_ind.get("player_stats") or {}
            jugadores_ind = perfilado.get("players") or []
            if jugadores_ind:
                st.caption(
                    "Los IDs identifican trayectorias, no dorsales verificados. "
                    "Distancias, velocidades, sprints y aceleraciones son estimaciones "
                    "calculadas solo con posiciones válidas de campo, no datos GPS. "
                    "El tiempo visible no equivale a minutos disputados.")
                opciones_equipo = ["Todos", "local", "visitante", "desconocido"]
                filtro_equipo = st.selectbox(
                    "Equipo para estadísticas", opciones_equipo, key="individual_team")
                def equipo_individual(equipo):
                    if equipo == "gk_local":
                        return "local"
                    if equipo == "gk_visitante":
                        return "visitante"
                    return equipo
                filtrados = [p for p in jugadores_ind
                             if filtro_equipo == "Todos"
                             or equipo_individual(p.get("team")) == filtro_equipo]
                filas = [{
                    "ID": p.get("track_id"),
                    "Equipo": p.get("team"),
                    "Visible (s)": p.get("visible_s"),
                    "Campo válido (%)": p.get("field_coverage_pct"),
                    "Fiabilidad": p.get("reliability", "sin evaluar"),
                    "Distancia estimada (m)": p.get("distance_m"),
                    "Vel. media (km/h)": p.get("avg_speed_kmh"),
                    "Vel. máxima (km/h)": p.get("max_speed_kmh"),
                    "Aceleraciones": p.get("accelerations"),
                    "Frenadas": p.get("decelerations"),
                    "Sprints": p.get("sprints"),
                    "Cerca del balón (s)": p.get("ball_proximity_s"),
                } for p in filtrados]
                comparativa = perfilado.get("team_comparison") or {}
                if comparativa:
                    st.caption("Comparación orientativa: tracks con cobertura suficiente, no jugadores únicos.")
                    st.dataframe([{
                        "Equipo": lado,
                        "IDs observados": info.get("track_ids"),
                        "IDs comparables": info.get("reliable_track_ids"),
                        "Distancia registrada (m)": info.get("measured_distance_m"),
                        "Mayor velocidad (km/h)": info.get("max_speed_kmh"),
                    } for lado, info in comparativa.items()],
                    use_container_width=True, hide_index=True)
                if filas:
                    df_ind = pd.DataFrame(filas)
                    st.dataframe(df_ind, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Exportar tabla individual (CSV)",
                        data=df_ind.to_csv(index=False).encode("utf-8-sig"),
                        file_name="estadisticas_individuales.csv",
                        mime="text/csv", key="individual_export_csv")
                    id_elegido = st.selectbox(
                        "ID para mapa de calor individual",
                        [p["track_id"] for p in filtrados], key="individual_track")
                    perfil = next(p for p in filtrados if p["track_id"] == id_elegido)
                    st.caption(
                        f"Posición media (m): {perfil.get('mean_position_m')} · "
                        f"Tiempo con movimiento medible (s): "
                        f"{perfil.get('field_measured_s')} · "
                        f"Calidad de campo: {perfil.get('pitch_quality', {})} · "
                        f"Fiabilidad: {perfil.get('reliability', 'no evaluada')}")
                    duracion = float(datos_ind.get("match_duration_s") or 0)
                    if duracion > 0:
                        figura, nota = generar_figura_mapa_calor(
                            datos_ind, f"jugador:{id_elegido}",
                            ("Partido completo", 0.0, duracion))
                        if figura is not None:
                            try:
                                _mostrar_figura_streamlit(figura)
                                st.caption(nota)
                            finally:
                                plt.close(figura)
                        else:
                            st.info(nota)
                else:
                    st.info("No hay trayectorias del equipo seleccionado.")
            else:
                st.info("Este tracking aún no incluye métricas individuales. "
                        "Regenera el análisis con la versión nueva.")
        else:
            st.info("Carga un tracking para mostrar estadísticas por jugador.")


    elif panel_video == "Táctica":
        # --- 7c. MAPAS DE CALOR ---
        st.divider()
        st.markdown("#### 🔥 Mapas de calor")
        if tracking_elegido is None:
            st.info("ℹ️ Carga un tracking (proporciona el vídeo arriba o pon un "
                    "tracking_*.json en la carpeta) para ver los mapas de calor.")
        else:
            try:
                datos_hm = cargar_tracking_json(
                    str(tracking_elegido), tracking_elegido.stat().st_mtime)
            except Exception as err:  # noqa: BLE001
                datos_hm = None
                st.error(f"No se pudo leer {tracking_elegido.name}: {err}")
            if datos_hm is not None:
                teams_hm = datos_hm.get("teams") or {}
                nombre_local = teams_hm.get("local", "Local")
                nombre_visitante = teams_hm.get("visitante", "Visitante")
                dur_hm = float(datos_hm.get("match_duration_s") or 0)
                c_hm1, c_hm2 = st.columns(2)
                with c_hm1:
                    sujeto_hm = st.selectbox(
                        "Qué mostrar",
                        ["local", "visitante", "balon"],
                        format_func=lambda v: {
                            "local": f"{nombre_local} (local)",
                            "visitante": f"{nombre_visitante} (visitante)",
                            "balon": "Balón"}[v],
                        key="hm_sujeto",
                    )
                with c_hm2:
                    tramos_hm = tramos_partido(dur_hm)
                    tramo_hm = st.selectbox(
                        "Tramo", tramos_hm,
                        format_func=lambda t: t[0],
                        key="hm_tramo",
                    )
                fig_hm, nota_hm = generar_figura_mapa_calor(
                    datos_hm, sujeto_hm, tramo_hm)
                if fig_hm is None:
                    st.warning(f"⚠️ {nota_hm}")
                else:
                    try:
                        _mostrar_figura_streamlit(fig_hm)
                        st.caption(nota_hm)
                    finally:
                        plt.close(fig_hm)


    elif panel_video == "Informes":
        # --- 7d. INFORMES AUTOMÁTICOS ---
        st.divider()
        st.markdown("#### Informes postpartido (PDF, Excel y CSV)")
        st.caption(
            "Se generan a partir del tracking ya analizado. No se inventan "
            "goles, pases ni tiros: solo se incluyen eventos etiquetados.")
        if tracking_elegido is None:
            st.info("Carga un tracking para generar el informe del partido.")
        else:
            if st.button("Generar o actualizar informes", key="report_generate"):
                try:
                    from match_reports import generate_match_reports
                    rutas_reportes = generate_match_reports(tracking_elegido)
                    st.success("Informes generados correctamente.")
                except Exception as error:
                    st.error(f"No se pudieron generar los informes: {error}")
            base_reportes = tracking_elegido.with_suffix("")
            descargas = [
                ("PDF del partido", "_informe.pdf", "application/pdf", "pdf"),
                ("Excel de estadísticas", "_estadisticas.xlsx",
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
                ("CSV jugadores", "_jugadores.csv", "text/csv", "csv"),
                ("CSV táctico", "_tactica.csv", "text/csv", "csv"),
            ]
            rutas_existentes = [
                (titulo, Path(str(base_reportes) + sufijo), mime, extension)
                for titulo, sufijo, mime, extension in descargas
                if Path(str(base_reportes) + sufijo).is_file()
            ]
            if rutas_existentes:
                columnas_descarga = st.columns(len(rutas_existentes))
                for columna, (titulo, ruta, mime, extension) in zip(
                        columnas_descarga, rutas_existentes):
                    with columna:
                        st.download_button(
                            titulo, data=ruta.read_bytes(),
                            file_name=ruta.name, mime=mime,
                            key=f"report_download_{extension}_{ruta.stem}")
            else:
                st.info("Todavía no hay informes generados para este tracking.")


    elif panel_video == "Biblioteca":
        # --- 7c. BIBLIOTECA DE PARTIDOS ---
        st.divider()
        st.markdown("#### 📚 Biblioteca de partidos")
        st.caption("Todos los tracking_*.json encontrados bajo la carpeta de la app.")
        biblioteca = buscar_biblioteca(carpeta_app)
        if not biblioteca:
            st.info("ℹ️ No hay ningún tracking (tracking_*.json) en la carpeta ni "
                    "en sus subcarpetas. Genera uno proporcionando el vídeo "
                    "arriba o con el notebook de Colab.")
        else:
            df_bib = pd.DataFrame([{
                "Partido": m["partido"],
                "Fecha": m["fecha"],
                "Duración": m["duracion"],
                "Muestras": m["muestras"],
                "Archivo": m["archivo"],
            } for m in biblioteca])
            st.dataframe(df_bib, use_container_width=True, hide_index=True)
            opciones_bib = [f"{m['partido']} · {m['archivo']}" for m in biblioteca]
            sel_bib = st.selectbox(
                "Elige un partido para cargarlo en la pantalla manual",
                opciones_bib, key="bib_select")
            if st.button("📥 Cargar en la pantalla manual", key="bib_cargar"):
                idx_bib = opciones_bib.index(sel_bib)
                st.session_state["tracking_del_video"] = biblioteca[idx_bib]["ruta"]
                st.success(f"✅ {biblioteca[idx_bib]['partido']} cargado en la "
                           "pantalla manual.")
                st.rerun()


    if panel_video == "Informes":
        # --- 7d. EXPORTAR CLIPS COMO VÍDEO ---
        st.divider()
        st.markdown("#### 🎬 Exportar clips como vídeo")
        st.caption("Exporta un clip con círculos bajo los jugadores y seguimiento del balón.")
        ruta_clip = st.text_input(
            "Vídeo",
            value=st.session_state.get("video_gen_path", ""),
            key="clip_video_path",
            placeholder=r"C:\Videos\partido.mp4",
        ).strip().strip('"').strip("'").strip()
        # Aviso si el tracking es de otro vídeo que el elegido.
        if tracking_elegido is not None and ruta_clip:
            try:
                datos_clip = cargar_tracking_json(
                    str(tracking_elegido), tracking_elegido.stat().st_mtime)
                vf_clip = datos_clip.get("video_filename")
                if vf_clip and Path(ruta_clip).name != vf_clip:
                    st.warning(f"⚠️ El tracking es del vídeo «{vf_clip}» y el "
                               f"vídeo elegido es «{Path(ruta_clip).name}»: "
                               "el clip saldría con el tracking de otro partido.")
            except Exception:  # noqa: BLE001 - el aviso es opcional
                pass
        origen_clip = st.radio(
            "Corte a exportar",
            ["JSON exportado desde la pantalla manual", "Inicio/fin manuales"],
            horizontal=True, key="clip_origen",
        )
        inicio_clip = fin_clip = None
        etiqueta_clip = "jugada"
        if origen_clip.startswith("JSON"):
            archivo_cortes = st.file_uploader(
                "Sube analisis_match_view.json (exportado desde la pantalla manual)",
                type=["json"], key="clip_json")
            if archivo_cortes is not None:
                try:
                    datos_cortes = json.loads(archivo_cortes.read().decode("utf-8"))
                    cortes = datos_cortes.get("cuts") or []
                    if not cortes:
                        st.warning("⚠️ Ese JSON no trae cortes (cuts).")
                    else:
                        opciones_cortes = [
                            f"{c.get('label', 'corte')} "
                            f"({_mmss(c.get('start_s', 0))}–{_mmss(c.get('end_s', 0))})"
                            for c in cortes]
                        sel_corte = st.selectbox(
                            "Elige un corte", opciones_cortes, key="clip_corte")
                        corte = cortes[opciones_cortes.index(sel_corte)]
                        inicio_clip = float(corte.get("start_s", 0))
                        fin_clip = float(corte.get("end_s", 0))
                        etiqueta_clip = str(corte.get("label", "jugada"))
                        st.caption(f"Inicio {inicio_clip:.1f} s – fin {fin_clip:.1f} s "
                                   "(segundos de vídeo).")
                except (ValueError, UnicodeDecodeError) as err:
                    st.error(f"No se pudo leer ese JSON: {err}")
        else:
            c_clip1, c_clip2, c_clip3 = st.columns(3)
            with c_clip1:
                inicio_clip = st.number_input(
                    "Inicio (s de vídeo)", min_value=0.0, value=0.0, step=1.0,
                    key="clip_inicio")
            with c_clip2:
                fin_clip = st.number_input(
                    "Fin (s de vídeo)", min_value=0.0, value=10.0, step=1.0,
                    key="clip_fin")
            with c_clip3:
                etiqueta_clip = st.text_input(
                    "Etiqueta", value="jugada", key="clip_etiqueta")
        nombre_clip = ("clip_" + "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in etiqueta_clip
        ).strip("_")[:40] + f"_{int(inicio_clip or 0)}.mp4") if inicio_clip is not None else ""
        if st.button("🎬 Exportar clip", type="primary", key="clip_exportar"):
            if not ruta_clip or not os.path.isfile(ruta_clip):
                st.error("⚠️ Indica la ruta de un vídeo que exista en este equipo.")
            elif tracking_elegido is None:
                st.error("⚠️ No hay ningún tracking cargado para dibujar el overlay.")
            elif inicio_clip is None or fin_clip is None:
                st.error("⚠️ Elige un corte o pon inicio y fin.")
            elif fin_clip <= inicio_clip:
                st.error("⚠️ El fin tiene que ser mayor que el inicio.")
            elif fin_clip - inicio_clip > 300:
                st.error("⚠️ El clip es demasiado largo (máx. 5 min).")
            else:
                try:
                    datos_clip = cargar_tracking_json(
                        str(tracking_elegido), tracking_elegido.stat().st_mtime)
                except Exception as err:  # noqa: BLE001
                    datos_clip = None
                    st.error(f"No se pudo leer {tracking_elegido.name}: {err}")
                if datos_clip is not None:
                    if CV2_DISPONIBLE:
                        barra_clip = st.progress(0.0)
                        estado_clip = st.empty()
                        try:
                            bytes_clip = render_clip_cv2(
                                ruta_clip, datos_clip, inicio_clip, fin_clip,
                                progreso_cb=lambda f, m: (
                                    barra_clip.progress(f), estado_clip.write(m)))
                            st.session_state["clip_bytes"] = bytes_clip
                            st.session_state["clip_nombre"] = nombre_clip
                            estado_clip.write("✅ Clip renderizado con overlay.")
                        except Exception as err:  # noqa: BLE001
                            st.error(f"No se pudo renderizar el clip: {err}")
                    elif shutil.which("ffmpeg") is not None:
                        tmp_clip = tempfile.NamedTemporaryFile(
                            suffix=".mp4", delete=False)
                        tmp_clip.close()
                        try:
                            if cortar_clip_ffmpeg(ruta_clip, inicio_clip, fin_clip,
                                                  tmp_clip.name):
                                st.session_state["clip_bytes"] = Path(tmp_clip.name).read_bytes()
                                st.session_state["clip_nombre"] = nombre_clip
                                st.warning("⚠️ Clip exportado sin overlay: ffmpeg "
                                           "no puede dibujar el tracking encima "
                                           "(OpenCV no está disponible).")
                            else:
                                st.error("ffmpeg falló al cortar el clip.")
                        finally:
                            try:
                                os.unlink(tmp_clip.name)
                            except OSError:
                                pass
                    else:
                        st.error("⚠️ Ni OpenCV ni ffmpeg están disponibles en este "
                                 "equipo: no se puede exportar el clip.")
        if st.session_state.get("clip_bytes"):
            st.download_button(
                label=f"📥 Descargar {st.session_state.get('clip_nombre', 'clip.mp4')}",
                data=st.session_state["clip_bytes"],
                file_name=st.session_state.get("clip_nombre", "clip.mp4"),
                mime="video/mp4",
                key="clip_descargar",
            )
