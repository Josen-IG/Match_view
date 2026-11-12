import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
import time
import os
import re
from pathlib import Path

from video_ai import VideoProcessingError, analyze_video

# ============================================================
# PROTOTIPO — NO ES LA APP BASE. Usa match.py.
# Este archivo fue la primera prueba del proyecto (análisis de vídeo
# automático con video_ai.py + Groq). Contiene datos DEMO: si un CSV no
# trae minuto o coordenadas, se rellenan con valores ALEATORIOS
# (normalizar_csv_cargado) y generar_datos_preferente_demo crea 420
# eventos inventados. No usar para analizar partidos reales.
# ============================================================

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO OSCURO PRO
# ==========================================
st.set_page_config(
    page_title="Zona del Analista Pro - Regional Preferente Navarra",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    header[data-testid="stHeader"] {
        background-color: #0b0f19;
    }
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 1rem;
        max-width: 98%;
    }
    .pro-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
    }
    .pro-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .score-badge {
        background-color: #090d16;
        border: 1px solid #334155;
        padding: 6px 18px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: 2px;
    }
    .team-title-red {
        color: #ef4444;
        font-weight: 800;
        font-size: 18px;
    }
    .team-title-blue {
        color: #3b82f6;
        font-weight: 800;
        font-size: 18px;
    }
    .stat-row {
        margin-bottom: 12px;
        font-size: 12px;
    }
    .stat-labels {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .stat-bar-bg {
        width: 100%;
        height: 6px;
        background-color: #1e293b;
        border-radius: 3px;
        display: flex;
        overflow: hidden;
    }
    .stat-bar-left {
        background-color: #ef4444;
        height: 100%;
    }
    .stat-bar-right {
        background-color: #3b82f6;
        height: 100%;
    }
    div.stButton > button {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        border-color: #f59e0b;
        color: #f59e0b;
    }
    div.stButton > button[kind="primary"] {
        background-color: #f59e0b;
        color: #090d16;
        border: none;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CARGA AUTOMÁTICA DE PLANTILLAS (CSV / EXCEL)
# ==========================================
@st.cache_data
def cargar_plantillas_preferente():
    plantillas = {}
    df = None
    if os.path.exists("Plantillas preferente.xlsx"):
        try:
            df = pd.read_excel("Plantillas preferente.xlsx")
        except Exception:
            df = None
    if df is None and os.path.exists("Plantillas preferente.csv"):
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                df = pd.read_csv("Plantillas preferente.csv", sep=";", encoding=enc)
                break
            except Exception:
                continue

    if df is not None:
        for i in range(0, len(df.columns) - 1, 2):
            t_name = str(df.columns[i]).strip()
            if "Unnamed" in t_name:
                t_name = str(df.columns[i + 1]).strip()
            t_name = t_name.replace('"', '').strip()

            c_dorsal = df.iloc[:, i]
            c_nombre = df.iloc[:, i + 1]

            jugadores = []
            for d, n in zip(c_dorsal, c_nombre):
                if pd.notna(n) and str(n).strip() != "":
                    d_str = re.sub(r"[^\d]", "", str(d))
                    d_num = int(d_str) if d_str else 1
                    n_clean = str(n).replace("\xa0", " ").replace("", "").strip()
                    pos_def = "POR" if d_num in [1, 13, 25] else ("DEF" if d_num in [2, 3, 4, 5] else ("DEL" if d_num in [7, 9, 11] else "MED"))
                    jugadores.append({
                        "dorsal": d_num,
                        "nombre": n_clean,
                        "etiqueta": f"#{d_num} {n_clean}",
                        "pos": pos_def
                    })
            if jugadores:
                plantillas[t_name] = jugadores

    # Fallback en caso de que no esté el archivo en la carpeta
    if not plantillas:
        equipos_default = [
            "C.D. San Miguel", "C.D. Aluvion", "C.A. Valtierrano", "C.D. San Adrian",
            "C.D. Mendi", "C.A. Milagres de Futbol", "C.D. Larrate", "C.D. Arenas",
            "C.D. River Ega", "C.D. Funes", "C.D. Azkarrena", "C.D. Ribaforada",
            "C.A. Cirbonero B", "C.D. Idoya", "C.D. Sesma", "C.D. Alesves"
        ]
        for eq in equipos_default:
            plantillas[eq] = [
                {"dorsal": i, "nombre": f"Jugador {i}", "etiqueta": f"#{i} Jugador {i}", "pos": "POR" if i in [1, 13] else "MED"}
                for i in range(1, 19)
            ]
    return plantillas

PLANTILLAS_EQUIPOS = cargar_plantillas_preferente()
LISTA_EQUIPOS_NAVARRA = list(PLANTILLAS_EQUIPOS.keys())
POSICIONES = ["POR", "DEF", "MED", "DEL"]

FORMACIONES_COORDS = {
    "1-4-4-2": [(8,50), (22,18), (22,38), (22,62), (22,82), (36,18), (36,38), (36,62), (36,82), (46,35), (46,65)],
    "1-4-3-3": [(8,50), (22,18), (22,38), (22,62), (22,82), (34,30), (32,50), (34,70), (45,20), (46,50), (45,80)],
    "1-4-2-3-1": [(8,50), (20,18), (20,38), (20,62), (20,82), (30,35), (30,65), (40,20), (40,50), (40,80), (47,50)],
    "1-3-5-2": [(8,50), (20,25), (20,50), (20,75), (34,15), (34,32), (32,50), (34,68), (34,85), (46,35), (46,65)],
    "1-5-3-2": [(8,50), (20,15), (20,32), (20,50), (20,68), (20,85), (34,30), (34,50), (34,70), (46,35), (46,65)]
}

# Mapeo de las 9 Zonas del proyecto de Regional Preferente a Coordenadas (X, Y)
ZONA_A_XY = {
    "Z1 - Def. Izq": (17, 82),
    "Z2 - Def. Centro": (17, 50),
    "Z3 - Def. Der": (17, 18),
    "Z4 - Medio Izq": (50, 82),
    "Z5 - Medio Centro": (50, 50),
    "Z6 - Medio Der": (50, 18),
    "Z7 - Ataque Izq": (83, 82),
    "Z8 - Ataque Centro": (83, 50),
    "Z9 - Ataque Der": (83, 18)
}

def obtener_plantilla_inicial(eq_nombre):
    completa = [j.copy() for j in PLANTILLAS_EQUIPOS.get(eq_nombre, [])]
    xi = completa[:11] if len(completa) >= 11 else completa
    sup = completa[11:18] if len(completa) > 11 else []
    while len(sup) < 7:
        idx = len(sup) + 12
        sup.append({"dorsal": idx, "nombre": f"Suplente {idx}", "etiqueta": f"#{idx} Suplente {idx}", "pos": "MED"})
    return xi, sup

# ==========================================
# 2. INICIALIZACIÓN DE ESTADOS Y DATOS DEMO
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "modo_app" not in st.session_state:
    st.session_state.modo_app = "home"  # "home", "alineaciones", "etiquetado", "final", "dashboard"
if "vista_actual" not in st.session_state:
    st.session_state.vista_actual = "Estadísticas Partido"
if "eventos_tag" not in st.session_state:
    st.session_state.eventos_tag = []
if "video_ai_results" not in st.session_state:
    st.session_state.video_ai_results = []
if "video_ai_selected_clip" not in st.session_state:
    st.session_state.video_ai_selected_clip = None
if "video_source_path" not in st.session_state:
    st.session_state.video_source_path = None
if "inicio_tiempo" not in st.session_state:
    st.session_state.inicio_tiempo = None
if "tiempo_acumulado" not in st.session_state:
    st.session_state.tiempo_acumulado = 0
eq_def_1 = LISTA_EQUIPOS_NAVARRA[0]
eq_def_2 = LISTA_EQUIPOS_NAVARRA[1] if len(LISTA_EQUIPOS_NAVARRA) > 1 else LISTA_EQUIPOS_NAVARRA[0]

if "datos_partido" not in st.session_state:
    xi_l, sup_l = obtener_plantilla_inicial(eq_def_1)
    xi_v, sup_v = obtener_plantilla_inicial(eq_def_2)
    st.session_state.datos_partido = {
        "temporada": "2026/2027",
        "competicion": "Regional Preferente Navarra",
        "fecha": date.today(),
        "descripcion": "Jornada 1",
        "eq_local": eq_def_1,
        "eq_visitante": eq_def_2,
        "color_local": "#ef4444",
        "color_visitante": "#3b82f6",
        "form_local": "1-4-4-2",
        "form_visitante": "1-4-3-3",
        "xi_local": xi_l,
        "sup_local": sup_l,
        "xi_visitante": xi_v,
        "sup_visitante": sup_v
    }

def normalizar_csv_cargado(df_in, eq_loc_default, eq_vis_default):
    """Adapta cualquier CSV de la botonera de Regional Preferente al formato completo del Dashboard.
    ATENCIÓN (demo): si el CSV no trae minuto o coordenadas, se rellenan
    con valores ALEATORIOS (semilla 42). Son datos inventados, no reales.
    """
    df = df_in.copy()
    np.random.seed(42)

    # Si viene con 'Minuto' en formato MM:SS (ej. '14:25'), convertir a entero y calcular Segundo_Video
    if "Segundo_Video" not in df.columns:
        if "Minuto" in df.columns:
            def parse_sec(val):
                s = str(val).strip()
                if ":" in s:
                    partes = s.split(":")
                    return int(partes[0]) * 60 + int(partes[1])
                try:
                    return int(float(s)) * 60
                except Exception:
                    return 900
            df["Segundo_Video"] = df["Minuto"].apply(parse_sec)
            df["Minuto"] = (df["Segundo_Video"] // 60).clip(1, 95)
        else:
            df["Minuto"] = np.random.randint(1, 92, size=len(df))
            df["Segundo_Video"] = df["Minuto"] * 60

    # Si tiene 'Zona' (Z1..Z9) pero no tiene coordenadas X, Y, generarlas a partir de la zona
    if "X" not in df.columns or "Y" not in df.columns:
        xs, ys = [], []
        for _, r in df.iterrows():
            z = str(r.get("Zona", "Z5 - Medio Centro"))
            bx, by = ZONA_A_XY.get(z, (50, 50))
            xs.append(round(float(np.clip(bx + np.random.uniform(-10, 10), 2, 98)), 1))
            ys.append(round(float(np.clip(by + np.random.uniform(-10, 10), 2, 98)), 1))
        df["X"] = xs
        df["Y"] = ys

    if "X_Fin" not in df.columns or "Y_Fin" not in df.columns:
        df["X_Fin"] = np.clip(df["X"] + np.random.uniform(-15, 25, size=len(df)), 0, 100).round(1)
        df["Y_Fin"] = np.clip(df["Y"] + np.random.uniform(-18, 18, size=len(df)), 0, 100).round(1)

    if "Periodo" in df.columns:
        df["Periodo"] = df["Periodo"].replace({"1ª Parte": "1T", "2ª Parte": "2T"})
    else:
        df["Periodo"] = np.where(df["Minuto"] <= 45, "1T", "2T")

    if "Jugador" not in df.columns:
        jug_l = [j["etiqueta"] for j in PLANTILLAS_EQUIPOS.get(eq_loc_default, [])] or ["#10"]
        jug_v = [j["etiqueta"] for j in PLANTILLAS_EQUIPOS.get(eq_vis_default, [])] or ["#9"]
        df["Jugador"] = [
            np.random.choice(jug_l) if r.get("Equipo") == eq_loc_default else np.random.choice(jug_v)
            for _, r in df.iterrows()
        ]

    if "Receptor" not in df.columns:
        df["Receptor"] = df["Jugador"]

    if "Resultado" not in df.columns:
        df["Resultado"] = np.where(df.get("Detalle", "").isin(["Pérdida", "Fuera", "Fallido"]), "Fallido", "Exitoso")

    if "xT" not in df.columns:
        df["xT"] = np.round(np.random.exponential(0.02, size=len(df)), 3)

    if "Zona" not in df.columns:
        df["Zona"] = "Z5 - Medio Centro"

    return df

@st.cache_data
def generar_datos_preferente_demo(eq_local, eq_visitante, jornada="Jornada 1"):
    np.random.seed(42)
    equipos = [eq_local, eq_visitante]
    plantilla_l = [j["etiqueta"] for j in PLANTILLAS_EQUIPOS.get(eq_local, [])] or [f"#{i}" for i in range(1, 19)]
    plantilla_v = [j["etiqueta"] for j in PLANTILLAS_EQUIPOS.get(eq_visitante, [])] or [f"#{i}" for i in range(1, 19)]

    tipos_evento = ["Pase", "Tiro", "Duelo", "Recuperación", "Pérdida", "Centro", "Córner", "Falta", "Portero"]
    lista_zonas = list(ZONA_A_XY.keys())
    registros = []

    for i in range(420):
        eq = np.random.choice(equipos, p=[0.52, 0.48])
        jug_lista = plantilla_l if eq == eq_local else plantilla_v
        jug = np.random.choice(jug_lista)
        rec = np.random.choice(jug_lista)
        periodo = np.random.choice(["1T", "2T"])
        minuto = int(np.random.randint(1, 46) if periodo == "1T" else np.random.randint(46, 95))
        seg = int(minuto * 60 + np.random.randint(0, 59))

        ev = np.random.choice(tipos_evento, p=[0.50, 0.06, 0.14, 0.08, 0.08, 0.05, 0.03, 0.03, 0.03])
        zona = np.random.choice(lista_zonas)
        bx, by = ZONA_A_XY[zona]
        x1 = float(np.clip(bx + np.random.uniform(-14, 14), 3, 97))
        y1 = float(np.clip(by + np.random.uniform(-14, 14), 3, 97))
        x2 = float(np.clip(x1 + np.random.uniform(-20, 32), 0, 100))
        y2 = float(np.clip(y1 + np.random.uniform(-22, 22), 0, 100))

        resultado = np.random.choice(["Exitoso", "Fallido"], p=[0.78, 0.22])
        detalle = "Posicional" if ev == "Pase" else "Normal"
        xt = round(float(np.random.exponential(0.015)), 3)

        if ev == "Tiro":
            x1 = float(np.random.uniform(74, 98) if eq == eq_local else np.random.uniform(2, 26))
            y1 = float(np.random.uniform(22, 78))
            detalle = np.random.choice(["Gol", "A portería", "Fuera", "Bloqueado"], p=[0.12, 0.36, 0.34, 0.18])
            resultado = "Exitoso" if detalle in ["Gol", "A portería"] else "Fallido"
            xt = round(float(np.random.uniform(0.06, 0.35)), 2)
        elif ev == "Centro":
            x1 = float(np.random.uniform(70, 98) if eq == eq_local else np.random.uniform(2, 30))
            y1 = float(np.random.choice([np.random.uniform(2, 22), np.random.uniform(78, 98)]))
            x2 = float(np.random.uniform(85, 96) if eq == eq_local else np.random.uniform(4, 15))
            y2 = float(np.random.uniform(30, 70))

        registros.append({
            "ID": i + 1,
            "Temporada": "2026/2027",
            "Competicion": "Regional Preferente Navarra",
            "Descripcion": jornada,
            "Equipo": eq,
            "Jugador": jug,
            "Receptor": rec,
            "Periodo": periodo,
            "Minuto": minuto,
            "Segundo_Video": seg,
            "Zona": zona,
            "Evento": ev,
            "Detalle": detalle,
            "Resultado": resultado,
            "X": round(x1, 1),
            "Y": round(y1, 1),
            "X_Fin": round(x2, 1),
            "Y_Fin": round(y2, 1),
            "xT": xt
        })
    return pd.DataFrame(registros)

if "df_partido" not in st.session_state:
    dp_init = st.session_state.datos_partido
    st.session_state.df_partido = generar_datos_preferente_demo(dp_init["eq_local"], dp_init["eq_visitante"], dp_init["descripcion"])

# ==========================================
# 3. CONTROL DE ACCESO POR CONTRASEÑA
# ==========================================
if not st.session_state.authenticated:
    st.write("")
    st.write("")
    _, col_login, _ = st.columns([1.2, 1.6, 1.2])
    with col_login:
        st.markdown('<div class="pro-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("## 🔒 Zona del Analista Pro")
        st.caption("REGIONAL PREFERENTE DE NAVARRA · ACCESO TÉCNICO")
        # La llave NO está en el código. Se lee de MATCH_VIEW_PASSWORD o
        # de los secrets de Streamlit; si no hay ninguna, "1234".
        try:
            _llave = st.secrets.get("access_password")
        except Exception:
            _llave = None
        _llave = _llave or os.environ.get("MATCH_VIEW_PASSWORD") or "1234"
        pwd = st.text_input("Introduce la llave de acceso:", type="password")
        if st.button("Entrar al Panel", type="primary", use_container_width=True):
            if pwd == _llave:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Llave incorrecta")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==========================================
# FUNCIONES AUXILIARES DE DIBUJO
# ==========================================
def crear_campo_plotly(titulo="", color_cesped="#2e6f40"):
    fig = go.Figure()
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="white", width=2), fillcolor=color_cesped, layer="below"),
        dict(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="white", width=2), layer="below"),
        dict(type="circle", x0=41, y0=37, x1=59, y1=63, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=0, y0=21, x1=16, y1=79, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=84, y0=21, x1=100, y1=79, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=0, y0=36, x1=5.5, y1=64, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=94.5, y0=36, x1=100, y1=64, line=dict(color="white", width=2), layer="below"),
    ]
    fig.update_layout(
        title=dict(text=titulo, x=0.5, font=dict(color="white", size=15)),
        shapes=shapes,
        xaxis=dict(range=[-2, 102], visible=False, fixedrange=True),
        yaxis=dict(range=[-2, 102], visible=False, fixedrange=True, scaleanchor="x", scaleratio=0.68),
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        margin=dict(l=10, r=10, t=40, b=10),
        height=470,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5, font=dict(color="white"))
    )
    return fig

def barra_comparativa(titulo, val_izq, val_der, texto_izq=None, texto_der=None):
    t_izq = texto_izq if texto_izq is not None else str(val_izq)
    t_der = texto_der if texto_der is not None else str(val_der)
    total = (val_izq + val_der) if (val_izq + val_der) > 0 else 1
    pct_izq = int((val_izq / total) * 100)
    pct_der = 100 - pct_izq
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-labels">
            <span style="color:#ef4444;">{t_izq}</span>
            <span style="color:#cbd5e1;">{titulo}</span>
            <span style="color:#3b82f6;">{t_der}</span>
        </div>
        <div class="stat-bar-bg">
            <div class="stat-bar-left" style="width:{pct_izq}%;"></div>
            <div class="stat-bar-right" style="width:{pct_der}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PANTALLA 1: HOME (INICIO)
# ==========================================
if st.session_state.modo_app == "home":
    dp = st.session_state.datos_partido
    _, col_centro, _ = st.columns([0.8, 2.4, 0.8])
    with col_centro:
        st.markdown(f"""
        <div style="text-align:center; padding: 25px 10px;">
            <div style="font-size:52px;">⚽</div>
            <h1 style="color:#f59e0b; font-weight:800; margin-bottom:4px;">Zona del Analista Pro</h1>
            <p style="color:#94a3b8; font-size:13px; letter-spacing:2px;">REGIONAL PREFERENTE DE NAVARRA · {len(LISTA_EQUIPOS_NAVARRA)} PLANTILLAS CARGADAS</p>
        </div>
        """, unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        if b1.button("📋 Configurar Alineaciones y Jornada", type="primary", use_container_width=True):
            st.session_state.modo_app = "alineaciones"
            st.rerun()
        if b2.button("🎥 Video-Etiquetado y Cortes", use_container_width=True):
            st.session_state.modo_app = "etiquetado"
            st.rerun()
        if b3.button("📊 Ir al Dashboard Pro", use_container_width=True):
            st.session_state.df_partido = generar_datos_preferente_demo(dp["eq_local"], dp["eq_visitante"], dp["descripcion"])
            st.session_state.modo_app = "dashboard"
            st.rerun()

        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("##### 📂 Cargar CSV de Partido (Compatible con Botonera por Zonas Z1-Z9 o Coordenadas X,Y)")
        archivo_csv = st.file_uploader("Seleccionar CSV de partido", type=["csv"], label_visibility="collapsed")
        if archivo_csv is not None:
            df_subido = pd.read_csv(archivo_csv)
            if "Equipo" in df_subido.columns:
                eqs_csv = list(df_subido["Equipo"].dropna().unique())
                if len(eqs_csv) >= 1:
                    dp["eq_local"] = str(eqs_csv[0])
                if len(eqs_csv) >= 2:
                    dp["eq_visitante"] = str(eqs_csv[1])
            st.session_state.df_partido = normalizar_csv_cargado(df_subido, dp["eq_local"], dp["eq_visitante"])
            st.session_state.modo_app = "dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PANTALLA 2: MENÚ DE ALINEACIONES (PREFERENTE NAVARRA)
# ==========================================
elif st.session_state.modo_app == "alineaciones":
    dp = st.session_state.datos_partido

    top_izq, top_der = st.columns([4, 1.5])
    with top_izq:
        st.markdown("#### 📋 Configuración del Encuentro y Convocatorias (18 Jugadores)")
        c1, c2, c3, c4 = st.columns(4)
        dp["temporada"] = c1.text_input("Temporada", dp["temporada"])
        dp["competicion"] = c2.text_input("Competición", dp["competicion"])
        dp["fecha"] = c3.date_input("Fecha", dp["fecha"])
        jornadas_opciones = [f"Jornada {i}" for i in range(1, 31)]
        if dp["descripcion"] not in jornadas_opciones:
            dp["descripcion"] = jornadas_opciones[0]
        dp["descripcion"] = c4.selectbox("Jornada (1 a 30)", jornadas_opciones, index=jornadas_opciones.index(dp["descripcion"]))

    with top_der:
        st.write("")
        n_b1, n_b2 = st.columns(2)
        if n_b1.button("🏠 Inicio", use_container_width=True):
            st.session_state.modo_app = "home"
            st.rerun()
        if n_b2.button("Ir al Partido ➡️", type="primary", use_container_width=True):
            st.session_state.df_partido = generar_datos_preferente_demo(dp["eq_local"], dp["eq_visitante"], dp["descripcion"])
            st.session_state.modo_app = "etiquetado"
            st.rerun()

    st.divider()

    inf_izq, inf_der = st.columns(2)
    lista_formaciones = list(FORMACIONES_COORDS.keys())

    with inf_izq:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("##### ⚙️ Equipo Local")
        b1, b2, b3 = st.columns([2.2, 0.8, 1.5])
        nuevo_local = b1.selectbox("Club Local", LISTA_EQUIPOS_NAVARRA, index=LISTA_EQUIPOS_NAVARRA.index(dp["eq_local"]) if dp["eq_local"] in LISTA_EQUIPOS_NAVARRA else 0)
        if nuevo_local != dp["eq_local"]:
            dp["eq_local"] = nuevo_local
            dp["xi_local"], dp["sup_local"] = obtener_plantilla_inicial(nuevo_local)
            st.rerun()
        dp["color_local"] = b2.color_picker("Color L", dp["color_local"])
        dp["form_local"] = b3.selectbox("Sistema Local", lista_formaciones, index=lista_formaciones.index(dp["form_local"]))
        st.markdown('</div>', unsafe_allow_html=True)

    with inf_der:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("##### ⚙️ Equipo Visitante")
        b4, b5, b6 = st.columns([2.2, 0.8, 1.5])
        nuevo_vis = b4.selectbox("Club Visitante", LISTA_EQUIPOS_NAVARRA, index=LISTA_EQUIPOS_NAVARRA.index(dp["eq_visitante"]) if dp["eq_visitante"] in LISTA_EQUIPOS_NAVARRA else 1)
        if nuevo_vis != dp["eq_visitante"]:
            dp["eq_visitante"] = nuevo_vis
            dp["xi_visitante"], dp["sup_visitante"] = obtener_plantilla_inicial(nuevo_vis)
            st.rerun()
        dp["color_visitante"] = b5.color_picker("Color V", dp["color_visitante"])
        dp["form_visitante"] = b6.selectbox("Sistema Visitante", lista_formaciones, index=lista_formaciones.index(dp["form_visitante"]))
        st.markdown('</div>', unsafe_allow_html=True)

    col_jug_local, col_campo_central, col_jug_visitante = st.columns([1.6, 2.8, 1.6])

    def renderizar_convocatoria(eq_nombre, xi_list, sup_list, prefix):
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown(f"**👥 {eq_nombre} (Plantilla Oficial)**")
        plantilla_club = PLANTILLAS_EQUIPOS.get(eq_nombre, [])
        nombres_disponibles = [j["nombre"] for j in plantilla_club] or [f"Jugador {i}" for i in range(1, 25)]
        mapa_dorsales = {j["nombre"]: j["dorsal"] for j in plantilla_club}

        st.caption("XI Inicial (11 Titulares)")
        for idx, jugador in enumerate(xi_list):
            c_d, c_n, c_p = st.columns([0.9, 2.8, 1.3])
            current_name = jugador.get("nombre", nombres_disponibles[0])
            if current_name not in nombres_disponibles:
                nombres_disponibles.append(current_name)
            nuevo_nom = c_n.selectbox("Jugador", nombres_disponibles, index=nombres_disponibles.index(current_name), key=f"{prefix}_xi_n_{idx}", label_visibility="collapsed")
            if nuevo_nom != current_name:
                jugador["nombre"] = nuevo_nom
                jugador["dorsal"] = mapa_dorsales.get(nuevo_nom, jugador.get("dorsal", idx + 1))
            jugador["dorsal"] = c_d.number_input("D", value=int(jugador.get("dorsal", idx + 1)), key=f"{prefix}_xi_d_{idx}", label_visibility="collapsed")
            jugador["etiqueta"] = f"#{jugador['dorsal']} {jugador['nombre']}"
            current_pos = jugador.get("pos", "MED")
            jugador["pos"] = c_p.selectbox("Pos", POSICIONES, index=POSICIONES.index(current_pos) if current_pos in POSICIONES else 2, key=f"{prefix}_xi_p_{idx}", label_visibility="collapsed")

        st.caption("Banquillo (7 Suplentes)")
        for idx, jugador in enumerate(sup_list):
            c_d, c_n, c_p = st.columns([0.9, 2.8, 1.3])
            current_name = jugador.get("nombre", nombres_disponibles[min(idx + 11, len(nombres_disponibles) - 1)])
            if current_name not in nombres_disponibles:
                nombres_disponibles.append(current_name)
            nuevo_nom = c_n.selectbox("Suplente", nombres_disponibles, index=nombres_disponibles.index(current_name), key=f"{prefix}_sup_n_{idx}", label_visibility="collapsed")
            if nuevo_nom != current_name:
                jugador["nombre"] = nuevo_nom
                jugador["dorsal"] = mapa_dorsales.get(nuevo_nom, jugador.get("dorsal", idx + 12))
            jugador["dorsal"] = c_d.number_input("D", value=int(jugador.get("dorsal", idx + 12)), key=f"{prefix}_sup_d_{idx}", label_visibility="collapsed")
            jugador["etiqueta"] = f"#{jugador['dorsal']} {jugador['nombre']}"
            current_pos = jugador.get("pos", "MED")
            jugador["pos"] = c_p.selectbox("Pos", POSICIONES, index=POSICIONES.index(current_pos) if current_pos in POSICIONES else 2, key=f"{prefix}_sup_p_{idx}", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_jug_local:
        renderizar_convocatoria(dp["eq_local"], dp["xi_local"], dp["sup_local"], "l")

    with col_campo_central:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("##### 🏟️ Pizarra de Alineaciones Enfrentadas")
        fichas_html = ""
        for idx, (x, y) in enumerate(FORMACIONES_COORDS[dp["form_local"]]):
            dorsal_l = dp["xi_local"][idx]["dorsal"] if idx < len(dp["xi_local"]) else idx + 1
            fichas_html += f'<div style="position:absolute; left:{x}%; top:{y}%; transform:translate(-50%,-50%); width:28px; height:28px; background:{dp["color_local"]}; color:white; border:2px solid white; border-radius:50%; text-align:center; line-height:24px; font-size:11px; font-weight:bold;">{dorsal_l}</div>'
        for idx, (x, y) in enumerate(FORMACIONES_COORDS[dp["form_visitante"]]):
            dorsal_v = dp["xi_visitante"][idx]["dorsal"] if idx < len(dp["xi_visitante"]) else idx + 1
            fichas_html += f'<div style="position:absolute; left:{100-x}%; top:{y}%; transform:translate(-50%,-50%); width:28px; height:28px; background:{dp["color_visitante"]}; color:white; border:2px solid white; border-radius:50%; text-align:center; line-height:24px; font-size:11px; font-weight:bold;">{dorsal_v}</div>'

        st.markdown(f"""
        <div style="position:relative; width:100%; height:440px; background-color:#2e6f40; border:3px solid white; border-radius:8px; overflow:hidden;">
            <div style="position:absolute; left:50%; top:0; bottom:0; width:2px; background:rgba(255,255,255,0.85);"></div>
            <div style="position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:80px; height:80px; border:2px solid rgba(255,255,255,0.85); border-radius:50%;"></div>
            <div style="position:absolute; left:15px; top:10px; color:white; font-weight:bold; font-size:14px; background:rgba(0,0,0,0.45); padding:3px 8px; border-radius:4px;">{dp["eq_local"]} ({dp["form_local"]})</div>
            <div style="position:absolute; right:15px; top:10px; color:white; font-weight:bold; font-size:14px; background:rgba(0,0,0,0.45); padding:3px 8px; border-radius:4px;">{dp["eq_visitante"]} ({dp["form_visitante"]})</div>
            {fichas_html}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_jug_visitante:
        renderizar_convocatoria(dp["eq_visitante"], dp["xi_visitante"], dp["sup_visitante"], "v")

# ==========================================
# PANTALLA 3: VIDEO-ETIQUETADO + BOTONERA Z1-Z9 + CORTES AUTOMÁTICOS
# ==========================================
elif st.session_state.modo_app == "etiquetado":
    dp = st.session_state.datos_partido

    def segundos_transcurridos():
        transcurrido = st.session_state.tiempo_acumulado
        if st.session_state.inicio_tiempo is not None:
            transcurrido += int(time.time() - st.session_state.inicio_tiempo)
        return transcurrido

    c_top1, c_top2, c_top3, c_top4 = st.columns([1.2, 2.8, 1.5, 1.5])
    with c_top1:
        if st.button("⬅️ Alineaciones", use_container_width=True):
            st.session_state.modo_app = "alineaciones"
            st.rerun()
    with c_top2:
        st.markdown(f"#### 🎥 ETIQUETADO Y CORTES: {dp['eq_local']} vs {dp['eq_visitante']} ({dp['descripcion']})")
    with c_top3:
        if st.button("🛠️ Post-Partido / Editar CSV", use_container_width=True):
            st.session_state.modo_app = "final"
            st.rerun()
    with c_top4:
        if st.button("📊 Ir al Dashboard Pro", type="primary", use_container_width=True):
            if st.session_state.eventos_tag:
                st.session_state.df_partido = pd.DataFrame(st.session_state.eventos_tag)
            st.session_state.modo_app = "dashboard"
            st.rerun()

    col_plantilla, col_vid, col_eventos = st.columns([1.3, 3.1, 1.5])

    # --- COLUMNA IZQUIERDA: CRONÓMETRO, PLANTILLA REAL Y VENTANA DE CORTE ---
    with col_plantilla:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("**⏱️ CRONÓMETRO / TIEMPO**")
        cr1, cr2, cr3 = st.columns(3)
        if cr1.button("▶️", use_container_width=True, help="Iniciar cronómetro"):
            if st.session_state.inicio_tiempo is None:
                st.session_state.inicio_tiempo = time.time()
        if cr2.button("⏸️", use_container_width=True, help="Pausar"):
            if st.session_state.inicio_tiempo is not None:
                st.session_state.tiempo_acumulado += int(time.time() - st.session_state.inicio_tiempo)
                st.session_state.inicio_tiempo = None
        if cr3.button("⏹️", use_container_width=True, help="Reiniciar"):
            st.session_state.inicio_tiempo = None
            st.session_state.tiempo_acumulado = 0

        seg_crono = segundos_transcurridos()
        c_min, c_seg = st.columns(2)
        min_sel = c_min.number_input("Minuto", min_value=0, max_value=120, value=int(seg_crono // 60) if seg_crono > 0 else 15)
        seg_sel = c_seg.number_input("Segundo", min_value=0, max_value=59, value=int(seg_crono % 60))

        st.divider()
        st.markdown("**👥 PLANTILLA ACTIVA**")
        eq_sel = st.radio("Equipo en acción:", [dp["eq_local"], dp["eq_visitante"]])
        convocatoria_activa = dp["xi_local"] + dp["sup_local"] if eq_sel == dp["eq_local"] else dp["xi_visitante"] + dp["sup_visitante"]
        lista_etiquetas_jug = [j["etiqueta"] for j in convocatoria_activa] or ["#10"]
        jug_sel = st.selectbox("Jugador protagonista:", lista_etiquetas_jug)
        rec_sel = st.selectbox("Receptor (si es pase):", lista_etiquetas_jug, index=min(1, len(lista_etiquetas_jug) - 1))
        per_sel = st.selectbox("Periodo:", ["1T", "2T"])

        st.divider()
        st.markdown("**✂️ VENTANA DEL CORTE**")
        pre_roll = st.slider("Segundos antes de la jugada", 3, 20, 8)
        post_roll = st.slider("Segundos después de la jugada", 3, 20, 7)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- COLUMNA CENTRAL: VÍDEO Y CORTES IMPORTANTES ---
    with col_vid:
        video_path_text = st.text_input(
            "Ruta local del vídeo",
            placeholder=r"C:\Videos\partido.mp4",
            help="El vídeo se procesa directamente desde el disco y no se sube a Streamlit.",
        )
        video_path = Path(video_path_text).expanduser() if video_path_text.strip() else None
        video_ready = video_path is not None and video_path.is_file()
        if video_ready:
            resolved_video_path = str(video_path.resolve())
            if st.session_state.video_source_path != resolved_video_path:
                st.session_state.video_source_path = resolved_video_path
                st.session_state.video_ai_results = []
                st.session_state.video_ai_selected_clip = None

        if video_path_text.strip() and not video_ready:
            st.error("No se encuentra el archivo en esa ruta. Comprueba la ruta local.")

        if video_ready:
            st.markdown('<div class="pro-card">', unsafe_allow_html=True)
            st.markdown("##### 🤖 Cortes automáticos con YOLO + Groq")
            st.caption(
                "YOLO prioriza momentos detectando personas, balón y movimiento. "
                "Groq clasifica los candidatos y FFmpeg genera los clips."
            )
            st.warning(
                "Los fotogramas candidatos se envían a Groq; el vídeo completo "
                "permanece local. Revisa las sugerencias: pueden equivocarse."
            )
            with st.form("video_ai_analysis_form"):
                sample_interval = st.number_input(
                    "Intervalo de muestreo (segundos)",
                    min_value=2,
                    max_value=60,
                    value=5,
                )
                max_candidates = st.number_input(
                    "Máximo de momentos para revisar con Groq",
                    min_value=1,
                    max_value=20,
                    value=8,
                )
                min_confidence = st.slider(
                    "Confianza mínima para crear un corte",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.65,
                    step=0.05,
                )
                analizar = st.form_submit_button(
                    "Analizar y generar cortes",
                    type="primary",
                    width="stretch",
                )

            if analizar:
                groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
                if not groq_api_key:
                    st.error(
                        "No está configurada la variable de entorno GROQ_API_KEY. "
                        "En PowerShell, define $env:GROQ_API_KEY con tu clave y "
                        "abre Streamlit desde esa misma terminal."
                    )
                else:
                    progress_bar = st.progress(0.0)
                    progress_text = st.empty()

                    def actualizar_progreso(message, fraction):
                        progress_text.text(message)
                        progress_bar.progress(fraction)

                    try:
                        st.session_state.video_ai_selected_clip = None
                        st.session_state.video_ai_results = analyze_video(
                            video_path=video_path,
                            api_key=groq_api_key,
                            sample_interval=int(sample_interval),
                            max_candidates=int(max_candidates),
                            pre_roll=int(pre_roll),
                            post_roll=int(post_roll),
                            min_confidence=float(min_confidence),
                            progress=actualizar_progreso,
                        )
                    except (FileNotFoundError, VideoProcessingError) as error:
                        st.error(f"No se pudo analizar el vídeo: {error}")
                    else:
                        st.success(
                            f"Análisis terminado. Los clips están en "
                            f"`{video_path.parent / (video_path.stem + '_clips')}`."
                        )
                        if not st.session_state.video_ai_results:
                            st.warning(
                                "YOLO no detectó personas ni balón en los fotogramas "
                                "muestreados, así que no se enviaron candidatos a Groq."
                            )

            if st.session_state.video_ai_results:
                event_names = {
                    "goal": "Gol",
                    "shot": "Tiro",
                    "corner": "Córner",
                    "foul": "Falta",
                    "pass": "Pase",
                    "duel": "Duelo",
                    "transition": "Transición",
                    "uncertain": "Incierto",
                }
                clip_options = [
                    result
                    for result in st.session_state.video_ai_results
                    if result["clip_path"]
                    and Path(result["clip_path"]).is_file()
                ]
                if clip_options:
                    clip_paths = {result["clip_path"] for result in clip_options}
                    if st.session_state.video_ai_selected_clip not in clip_paths:
                        st.session_state.video_ai_selected_clip = clip_options[0]["clip_path"]
                    selected_clip = next(
                        result
                        for result in clip_options
                        if result["clip_path"] == st.session_state.video_ai_selected_clip
                    )

                    st.video(
                        selected_clip["clip_path"],
                        autoplay=True,
                        muted=True,
                        alt="Clip sugerido por YOLO y Groq",
                    )

                    title_col, filter_col = st.columns([1.2, 1.8])
                    with title_col:
                        st.markdown("##### 🔥 Cortes más importantes del partido")
                    available_events = list(
                        dict.fromkeys(
                            event_names.get(result["event"], "Incierto")
                            for result in clip_options
                        )
                    )
                    default_events = [
                        event
                        for event in ["Gol", "Tiro", "Córner"]
                        if event in available_events
                    ] or available_events
                    with filter_col:
                        selected_events = st.multiselect(
                            "Filtrar jugadas:",
                            available_events,
                            default=default_events,
                            key="ai_clip_event_filter",
                        )

                    filtered_clips = [
                        result
                        for result in clip_options
                        if event_names.get(result["event"], "Incierto") in selected_events
                    ]
                    st.caption(
                        f"Se han generado {len(filtered_clips)} cortes clave. "
                        "Pulsa un botón para reproducir el clip:"
                    )

                    if filtered_clips:
                        clip_columns = st.columns(3)
                        for index, result in enumerate(filtered_clips):
                            minute, second = divmod(int(result["seconds"]), 60)
                            label = (
                                f"{minute:02d}:{second:02d} | "
                                f"{event_names.get(result['event'], 'Incierto')} "
                                f"({result['confidence']:.0%})"
                            )
                            if clip_columns[index % 3].button(
                                label,
                                key=f"important_clip_{index}",
                                type=(
                                    "primary"
                                    if result["clip_path"] == selected_clip["clip_path"]
                                    else "secondary"
                                ),
                                width="stretch",
                            ):
                                st.session_state.video_ai_selected_clip = result["clip_path"]
                                st.rerun()

                        st.caption(
                            f"YOLO detectó {selected_clip['yolo_people']} personas y "
                            f"{selected_clip['yolo_ball']} balones. "
                            f"{selected_clip['reason']}"
                        )
                    else:
                        st.info("No hay clips de esas categorías. Cambia el filtro para ver más.")
                else:
                    st.info(
                        "No hay clips listos para mostrar. Ajusta la confianza mínima "
                        "y vuelve a analizar."
                    )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(
                "Indica la ruta local del vídeo para crear clips con FFmpeg y activar "
                "el análisis asistido."
            )

    # --- COLUMNA DERECHA: BOTONERA DE ACCIONES ---
    zona_registro = "Sin especificar"
    x_val, y_val = 50, 50
    with col_eventos:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("**⚡ REGISTRAR ACCIÓN**")
        botones = [
            ("Tiro", "Gol", "⚽ Gol"),
            ("Tiro", "A portería", "🥅 Tiro a Puerta"),
            ("Tiro", "Fuera", "❌ Tiro Fuera"),
            ("Pase", "Exitoso", "✅ Pase Exitoso"),
            ("Pase", "Fallido", "⚠️ Pase Fallido"),
            ("Centro", "Exitoso", "↗️ Centro al Área"),
            ("Ataque", "Posicional", "⚡ Ataque Posicional"),
            ("Ataque", "Transición", "🏃 Contraataque"),
            ("Recuperación", "Exitoso", "🛡️ Recuperación"),
            ("Pérdida", "Fallido", "❗ Pérdida"),
            ("Duelo", "Exitoso", "⚔️ Duelo Ganado"),
            ("Córner", "Exitoso", "🚩 Córner"),
            ("Falta", "Cometida", "🛑 Falta"),
            ("Disciplina", "Tarjeta", "🟨 Tarjeta")
        ]
        for ev, res, label in botones:
            if st.button(label, use_container_width=True):
                seg_total = int(min_sel * 60 + seg_sel)
                st.session_state.eventos_tag.append({
                    "ID": len(st.session_state.eventos_tag) + 1,
                    "Temporada": dp["temporada"],
                    "Competicion": dp["competicion"],
                    "Fecha": str(dp["fecha"]),
                    "Descripcion": dp["descripcion"],
                    "Equipo": eq_sel,
                    "Jugador": jug_sel,
                    "Receptor": rec_sel,
                    "Periodo": per_sel,
                    "Minuto": min_sel,
                    "Segundo_Video": seg_total,
                    "Zona": zona_registro,
                    "Evento": ev,
                    "Detalle": res,
                    "Resultado": "Exitoso" if res in ["Exitoso", "Gol", "A portería", "Posicional", "Transición"] else "Fallido",
                    "X": x_val,
                    "Y": y_val,
                    "X_Fin": min(100, x_val + 15),
                    "Y_Fin": y_val,
                    "xT": 0.18 if ev == "Tiro" else 0.04
                })
                st.toast(f"Registrado: {label} - {jug_sel}")
                st.rerun()

        if st.session_state.eventos_tag:
            if st.button("↩️ Deshacer último", use_container_width=True):
                st.session_state.eventos_tag.pop()
                st.rerun()
            df_t = pd.DataFrame(st.session_state.eventos_tag)
            st.dataframe(df_t[["Minuto", "Equipo", "Jugador", "Zona", "Evento"]].iloc[::-1], height=170, hide_index=True)
            st.download_button(
                "📥 Descargar CSV Rápido",
                df_t.to_csv(index=False).encode("utf-8"),
                f"{dp['eq_local']}_vs_{dp['eq_visitante']}_{dp['descripcion']}.csv",
                "text/csv",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PANTALLA 4: MENÚ FINAL / POST-PARTIDO (EDITOR DE CSV)
# ==========================================
elif st.session_state.modo_app == "final":
    dp = st.session_state.datos_partido

    c_b1, c_b2, c_b3 = st.columns([1.2, 3.6, 1.5])
    with c_b1:
        if st.button("⬅️ Volver a Etiquetado", use_container_width=True):
            st.session_state.modo_app = "etiquetado"
            st.rerun()
    with c_b2:
        st.markdown(f"### 📊 Post-Partido y Editor CSV: {dp['eq_local']} vs {dp['eq_visitante']} ({dp['descripcion']})")
    with c_b3:
        if st.button("📈 Abrir en Dashboard Pro", type="primary", use_container_width=True):
            st.session_state.modo_app = "dashboard"
            st.rerun()

    df_actual = pd.DataFrame(st.session_state.eventos_tag) if st.session_state.eventos_tag else st.session_state.df_partido.copy()

    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown("#### 🛠️ Desglose y Modificación Interactiva del CSV")
    st.caption("Edita cualquier celda de la tabla (jugador, zona, minuto, coordenadas o resultado) antes de actualizar el Dashboard o exportar para Power BI.")
    df_editado = st.data_editor(df_actual, num_rows="dynamic", use_container_width=True, key="editor_final_preferente")
    st.session_state.df_partido = df_editado
    st.markdown('</div>', unsafe_allow_html=True)

    col_st1, col_st2, col_st3 = st.columns(3)
    with col_st1:
        st.markdown('<div class="pro-card"><strong>📊 Conteo por Tipo de Evento</strong>', unsafe_allow_html=True)
        res_ev = df_editado["Evento"].value_counts().reset_index()
        res_ev.columns = ["Evento", "Total"]
        st.dataframe(res_ev, use_container_width=True, hide_index=True, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_st2:
        st.markdown('<div class="pro-card"><strong>📍 Conteo por Zona (Z1 - Z9)</strong>', unsafe_allow_html=True)
        if "Zona" in df_editado.columns:
            res_zn = df_editado["Zona"].value_counts().reset_index()
            res_zn.columns = ["Zona", "Total"]
            st.dataframe(res_zn, use_container_width=True, hide_index=True, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_st3:
        st.markdown('<div class="pro-card"><strong>📥 Exportación Definitiva</strong>', unsafe_allow_html=True)
        csv_final = df_editado.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar CSV Definitivo (Power BI / Pro)",
            data=csv_final,
            file_name=f"FINAL_{dp['eq_local']}_vs_{dp['eq_visitante']}_{dp['descripcion']}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
        if st.button("🔄 Reiniciar eventos en vivo", use_container_width=True):
            st.session_state.eventos_tag = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PANTALLA 5: DASHBOARD PRINCIPAL ZONA DEL ANALISTA PRO
# ==========================================
elif st.session_state.modo_app == "dashboard":
    dp = st.session_state.datos_partido
    df = st.session_state.df_partido.copy()

    eq_local = dp["eq_local"]
    eq_visitante = dp["eq_visitante"]

    goles_loc = len(df[(df["Equipo"] == eq_local) & (df["Evento"] == "Tiro") & (df["Detalle"] == "Gol")])
    goles_vis = len(df[(df["Equipo"] == eq_visitante) & (df["Evento"] == "Tiro") & (df["Detalle"] == "Gol")])

    top_nav1, top_nav2, top_nav3 = st.columns([1.7, 3.2, 1.6])
    with top_nav1:
        n1, n2, n3 = st.columns(3)
        if n1.button("🏠 INICIO", use_container_width=True):
            st.session_state.modo_app = "home"
            st.rerun()
        if n2.button("📋 XI", use_container_width=True):
            st.session_state.modo_app = "alineaciones"
            st.rerun()
        if n3.button("🎥 CORTES", use_container_width=True):
            st.session_state.modo_app = "etiquetado"
            st.rerun()

    with top_nav2:
        st.markdown(f"""
        <div class="pro-header">
            <span class="team-title-red">🛡️ {eq_local}</span>
            <span class="score-badge">{goles_loc} : {goles_vis}</span>
            <span class="team-title-blue">{eq_visitante} 🛡️</span>
        </div>
        """, unsafe_allow_html=True)

    with top_nav3:
        st.caption(f"**{dp['competicion']}** · {dp['descripcion']}")
        if st.button("🛠️ Editar Tabla CSV", use_container_width=True):
            st.session_state.modo_app = "final"
            st.rerun()

    col_izq, col_centro, col_der = st.columns([1.25, 3.6, 1.55])

    # --- COLUMNA IZQUIERDA: PLANTILLAS CON ESTADÍSTICAS DEL PARTIDO ---
    with col_izq:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="team-title-red">{eq_local}</span>', unsafe_allow_html=True)
        df_loc = df[df["Equipo"] == eq_local]
        resumen_loc = df_loc.groupby("Jugador").agg(
            P=("Evento", lambda x: (x == "Pase").sum()),
            T=("Evento", lambda x: (x == "Tiro").sum()),
            D=("Evento", lambda x: (x == "Duelo").sum()),
            R=("Evento", lambda x: (x == "Recuperación").sum())
        ).reset_index()
        st.dataframe(resumen_loc, use_container_width=True, hide_index=True, height=230)

        st.markdown(f'<span class="team-title-blue">{eq_visitante}</span>', unsafe_allow_html=True)
        df_vis = df[df["Equipo"] == eq_visitante]
        resumen_vis = df_vis.groupby("Jugador").agg(
            P=("Evento", lambda x: (x == "Pase").sum()),
            T=("Evento", lambda x: (x == "Tiro").sum()),
            D=("Evento", lambda x: (x == "Duelo").sum()),
            R=("Evento", lambda x: (x == "Recuperación").sum())
        ).reset_index()
        st.dataframe(resumen_vis, use_container_width=True, hide_index=True, height=230)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- COLUMNA DERECHA: PANEL DE CONTROL ---
    with col_der:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("**🎛️ PANEL DE CONTROL**")
        f_eq = st.selectbox("Equipo", ["Ambos Equipos", eq_local, eq_visitante], label_visibility="collapsed")
        f_per = st.radio("Periodo", ["Total", "1T", "2T"], horizontal=True)

        lista_vistas = [
            "Estadísticas Partido",
            "Mapa de Tiros",
            "Zona del Pase",
            "Cronología del Partido",
            "xT Acumulado",
            "Dinámicas del Partido",
            "Matriz de Pases",
            "Pases Últ. Tercio",
            "Centros",
            "Entradas al Área",
            "Balón Parado",
            "Zona de Dominio",
            "Voronoi Territorial",
            "Distrib. Portero",
            "Informe Jugador"
        ]
        st.session_state.vista_actual = st.radio("Módulos de Análisis:", lista_vistas)
        st.markdown('</div>', unsafe_allow_html=True)

    if f_per != "Total":
        df = df[df["Periodo"] == f_per]
    df_filtrado = df[df["Equipo"] == f_eq] if f_eq != "Ambos Equipos" else df

    # --- COLUMNA CENTRAL: LOS 15 MÓDULOS TÁCTICOS ---
    with col_centro:
        vista = st.session_state.vista_actual

        if vista == "Estadísticas Partido":
            d_l = df[df["Equipo"] == eq_local]
            d_v = df[df["Equipo"] == eq_visitante]
            p_l, p_v = len(d_l[d_l["Evento"] == "Pase"]), len(d_v[d_v["Evento"] == "Pase"])
            tot_p = max(1, p_l + p_v)
            pos_l, pos_v = int(round((p_l / tot_p) * 100)), 100 - int(round((p_l / tot_p) * 100))

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="pro-card"><h6>📊 POSESIÓN Y CONTROL</h6>', unsafe_allow_html=True)
                barra_comparativa("Posesión (%)", pos_l, pos_v, f"{pos_l}%", f"{pos_v}%")
                barra_comparativa("Pases totales", p_l, p_v)
                barra_comparativa("Pases exitosos", len(d_l[(d_l["Evento"]=="Pase") & (d_l["Resultado"]=="Exitoso")]), len(d_v[(d_v["Evento"]=="Pase") & (d_v["Resultado"]=="Exitoso")]))
                barra_comparativa("Eventos en Z5 (Medio Centro)", len(d_l[d_l["Zona"].str.contains("Z5", na=False)]), len(d_v[d_v["Zona"].str.contains("Z5", na=False)]))
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="pro-card"><h6>📐 DISTRIBUCIÓN DE JUEGO</h6>', unsafe_allow_html=True)
                barra_comparativa("Pases al último tercio", len(d_l[(d_l["Evento"]=="Pase") & (d_l["X_Fin"]>66)]), len(d_v[(d_v["Evento"]=="Pase") & (d_v["X_Fin"]<34)]))
                barra_comparativa("Ataques Banda Izq (Z1/Z4/Z7)", len(d_l[d_l["Zona"].str.contains("Izq", na=False)]), len(d_v[d_v["Zona"].str.contains("Izq", na=False)]))
                barra_comparativa("Ataques Banda Der (Z3/Z6/Z9)", len(d_l[d_l["Zona"].str.contains("Der", na=False)]), len(d_v[d_v["Zona"].str.contains("Der", na=False)]))
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="pro-card"><h6>⚡ CREACIÓN DE OCASIONES</h6>', unsafe_allow_html=True)
                barra_comparativa("xT Total (Amenaza)", round(d_l["xT"].sum(), 2), round(d_v["xT"].sum(), 2))
                barra_comparativa("Centros totales", len(d_l[d_l["Evento"]=="Centro"]), len(d_v[d_v["Evento"]=="Centro"]))
                barra_comparativa("Córners (ABP)", len(d_l[d_l["Evento"]=="Córner"]), len(d_v[d_v["Evento"]=="Córner"]))
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="pro-card"><h6>🥅 TIROS</h6>', unsafe_allow_html=True)
                barra_comparativa("Goles", goles_loc, goles_vis)
                barra_comparativa("Tiros totales", len(d_l[d_l["Evento"]=="Tiro"]), len(d_v[d_v["Evento"]=="Tiro"]))
                barra_comparativa("Tiros a portería", len(d_l[(d_l["Evento"]=="Tiro") & (d_l["Resultado"]=="Exitoso")]), len(d_v[(d_v["Evento"]=="Tiro") & (d_v["Resultado"]=="Exitoso")]))
                st.markdown('</div>', unsafe_allow_html=True)

            with c3:
                st.markdown('<div class="pro-card"><h6>⚔️ DUELOS Y PRESIÓN</h6>', unsafe_allow_html=True)
                barra_comparativa("Duelos totales", len(d_l[d_l["Evento"]=="Duelo"]), len(d_v[d_v["Evento"]=="Duelo"]))
                barra_comparativa("Duelos ganados", len(d_l[(d_l["Evento"]=="Duelo") & (d_l["Resultado"]=="Exitoso")]), len(d_v[(d_v["Evento"]=="Duelo") & (d_v["Resultado"]=="Exitoso")]))
                barra_comparativa("Faltas cometidas", len(d_l[d_l["Evento"]=="Falta"]), len(d_v[d_v["Evento"]=="Falta"]))
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="pro-card"><h6>🛡️ RECUPERACIONES Y PÉRDIDAS</h6>', unsafe_allow_html=True)
                barra_comparativa("Recuperaciones totales", len(d_l[d_l["Evento"]=="Recuperación"]), len(d_v[d_v["Evento"]=="Recuperación"]))
                barra_comparativa("Pérdidas totales", len(d_l[d_l["Evento"]=="Pérdida"]), len(d_v[d_v["Evento"]=="Pérdida"]))
                st.markdown('</div>', unsafe_allow_html=True)

        elif vista == "Mapa de Tiros":
            tiros = df_filtrado[df_filtrado["Evento"] == "Tiro"]
            fig = crear_campo_plotly("MAPA DE TIROS")
            for eq, color in [(eq_local, "#ef4444"), (eq_visitante, "#3b82f6")]:
                sub = tiros[tiros["Equipo"] == eq]
                if not sub.empty:
                    fig.add_trace(go.Scatter(
                        x=sub["X"], y=sub["Y"], mode="markers+text",
                        text=sub["Jugador"], textposition="top center",
                        marker=dict(
                            size=sub["Detalle"].map({"Gol": 18, "A portería": 13, "Fuera": 11, "Bloqueado": 10}).fillna(12),
                            color=color,
                            symbol=sub["Detalle"].map({"Gol": "star", "A portería": "circle", "Fuera": "x", "Bloqueado": "square"}).fillna("circle"),
                            line=dict(color="white", width=1.5)
                        ),
                        name=f"{eq} ({len(sub)} tiros)"
                    ))
            st.plotly_chart(fig, use_container_width=True)

        elif vista == "Zona del Pase":
            pases = df_filtrado[df_filtrado["Evento"] == "Pase"]
            fig = crear_campo_plotly("ZONA DEL PASE (DISTRIBUCIÓN POR SECTORES)")
            fig.add_trace(go.Histogram2d(
                x=pases["X"], y=pases["Y"],
                xbins=dict(start=0, end=100, size=33.3),
                ybins=dict(start=0, end=100, size=33.3),
                colorscale=[[0, "rgba(30,41,59,0.3)"], [0.5, "rgba(59,130,246,0.55)"], [1, "rgba(239,68,68,0.75)"]],
                texttemplate="%{z} acciones",
                showscale=False
            ))
            st.plotly_chart(fig, use_container_width=True)

        elif vista == "Cronología del Partido":
            fig = go.Figure()
            fig.add_hline(y=0, line=dict(color="#94a3b8", width=3))
            tiros = df[df["Evento"] == "Tiro"]
            for eq, y_pos, col in [(eq_local, 0.5, "#ef4444"), (eq_visitante, -0.5, "#3b82f6")]:
                sub = tiros[tiros["Equipo"] == eq]
                fig.add_trace(go.Scatter(
                    x=sub["Minuto"], y=[y_pos]*len(sub),
                    mode="markers+text", text=sub["Jugador"] + " (" + sub["Detalle"] + ")",
                    textposition="top center" if y_pos > 0 else "bottom center",
                    marker=dict(size=14, color=col, line=dict(color="white", width=1.5)),
                    name=eq
                ))
            fig.update_layout(
                title="CRONOLOGÍA DE OCASIONES DEL PARTIDO",
                xaxis=dict(title="Minuto de Partido", range=[0, 96], gridcolor="#1e293b"),
                yaxis=dict(range=[-1.5, 1.5], visible=False),
                plot_bgcolor="#111827", paper_bgcolor="#111827", font=dict(color="white"), height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        elif vista == "xT Acumulado":
            fig = go.Figure()
            for eq, col in [(eq_local, "#ef4444"), (eq_visitante, "#3b82f6")]:
                sub = df[df["Equipo"] == eq].sort_values("Minuto")
                sub["xT_Cum"] = sub["xT"].cumsum()
                fig.add_trace(go.Scatter(
                    x=sub["Minuto"], y=sub["xT_Cum"], mode="lines",
                    line=dict(color=col, width=3, shape="hv"), name=eq
                ))
            fig.update_layout(
                title="xT ACUMULADO EN EL TIEMPO · AMENAZA GENERADA",
                xaxis=dict(title="Minuto", gridcolor="#1e293b"),
                yaxis=dict(title="xT Acumulado", gridcolor="#1e293b"),
                plot_bgcolor="#111827", paper_bgcolor="#111827", font=dict(color="white"), height=450
            )
            st.plotly_chart(fig, use_container_width=True)

        elif vista == "Dinámicas del Partido":
            tramos = ["1-15'", "16-30'", "31-45'", "46-60'", "61-75'", "76-90'"]
            g1, g2 = st.columns(2)
            with g1:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=tramos, y=[49, 44, 52, 41, 40, 56], mode="lines+markers+text", text=["49%","44%","52%","41%","40%","56%"], textposition="top center", name=eq_local, line=dict(color="#ef4444", width=3)))
                fig1.add_trace(go.Scatter(x=tramos, y=[51, 56, 48, 59, 60, 44], mode="lines+markers+text", text=["51%","56%","48%","59%","60%","44%"], textposition="bottom center", name=eq_visitante, line=dict(color="#3b82f6", width=3)))
                fig1.update_layout(title="Posesión por tramo de 15' (%)", plot_bgcolor="#111827", paper_bgcolor="#111827", font=dict(color="white"), height=320)
                st.plotly_chart(fig1, use_container_width=True)
            with g2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=tramos, y=[0.18, 0.28, 0.45, 0.14, 0.65, 0.23], mode="lines+markers", name=eq_local, line=dict(color="#ef4444", width=3)))
                fig2.add_trace(go.Scatter(x=tramos, y=[0.06, 0.15, 0.12, 0.19, 0.08, 0.12], mode="lines+markers", name=eq_visitante, line=dict(color="#3b82f6", width=3)))
                fig2.update_layout(title="xT por tramo de 15'", plot_bgcolor="#111827", paper_bgcolor="#111827", font=dict(color="white"), height=320)
                st.plotly_chart(fig2, use_container_width=True)

        elif vista == "Matriz de Pases":
            eq_mat = st.radio("Seleccionar club:", [eq_local, eq_visitante], horizontal=True)
            pases_eq = df[(df["Equipo"] == eq_mat) & (df["Evento"] == "Pase")]
            matriz = pd.crosstab(pases_eq["Jugador"], pases_eq["Receptor"])
            fig = go.Figure(data=go.Heatmap(
                z=matriz.values, x=matriz.columns, y=matriz.index,
                colorscale="Reds" if eq_mat == eq_local else "Blues",
                text=matriz.values, texttemplate="%{text}", showscale=False
            ))
            fig.update_layout(title=f"MATRIZ DE CONEXIONES DE PASE — {eq_mat}", plot_bgcolor="#111827", paper_bgcolor="#111827", font=dict(color="white"), height=460)
            st.plotly_chart(fig, use_container_width=True)

        elif vista in ["Pases Últ. Tercio", "Centros", "Entradas al Área", "Balón Parado", "Distrib. Portero"]:
            fig = crear_campo_plotly(vista.upper())
            if vista == "Pases Últ. Tercio":
                sub = df_filtrado[(df_filtrado["Evento"] == "Pase") & ((df_filtrado["X_Fin"] > 66) | (df_filtrado["X_Fin"] < 34))]
            elif vista == "Centros":
                sub = df_filtrado[df_filtrado["Evento"] == "Centro"]
            elif vista == "Entradas al Área":
                sub = df_filtrado[(df_filtrado["Evento"] == "Pase") & ((df_filtrado["X_Fin"] > 84) | (df_filtrado["X_Fin"] < 16)) & (df_filtrado["Y_Fin"].between(21, 79))]
            elif vista == "Balón Parado":
                sub = df_filtrado[df_filtrado["Evento"].isin(["Córner", "Falta"])]
            else:
                sub = df_filtrado[df_filtrado["Evento"] == "Portero"]

            for _, r in sub.head(65).iterrows():
                col = "#ef4444" if r["Equipo"] == eq_local else "#3b82f6"
                fig.add_trace(go.Scatter(
                    x=[r["X"], r["X_Fin"]], y=[r["Y"], r["Y_Fin"]],
                    mode="lines+markers", line=dict(color=col, width=2),
                    marker=dict(size=[5, 9], symbol=["circle", "arrow-bar-up"]),
                    showlegend=False
                ))
            st.plotly_chart(fig, use_container_width=True)

        elif vista == "Zona de Dominio":
            fig = crear_campo_plotly("ZONA DE DOMINIO TERRITORIAL (CUADRÍCULA TÁCTICA)")
            z_dom = np.random.randint(30, 85, size=(3, 3))
            fig.add_trace(go.Heatmap(
                z=z_dom, x=[17, 50, 83], y=[18, 50, 82],
                colorscale=[[0, "rgba(59,130,246,0.65)"], [0.5, "rgba(30,41,59,0.5)"], [1, "rgba(239,68,68,0.7)"]],
                text=z_dom, texttemplate="%{text}%", showscale=False
            ))
            st.plotly_chart(fig, use_container_width=True)

        elif vista == "Voronoi Territorial":
            fig = crear_campo_plotly(f"VORONOI TERRITORIAL — {eq_local} vs {eq_visitante}")
            pts_x_l = [c[0] for c in FORMACIONES_COORDS[dp["form_local"]]]
            pts_y_l = [c[1] for c in FORMACIONES_COORDS[dp["form_local"]]]
            pts_x_v = [100 - c[0] for c in FORMACIONES_COORDS[dp["form_visitante"]]]
            pts_y_v = [c[1] for c in FORMACIONES_COORDS[dp["form_visitante"]]]

            pts_x = np.array(pts_x_l + pts_x_v)
            pts_y = np.array(pts_y_l + pts_y_v)
            dorsales_v = [str(j["dorsal"]) for j in dp["xi_local"][:11]] + [str(j["dorsal"]) for j in dp["xi_visitante"][:11]]

            gx, gy = np.meshgrid(np.linspace(0, 100, 120), np.linspace(0, 100, 80))
            grid_pts = np.c_[gx.ravel(), gy.ravel()]
            player_pts = np.c_[pts_x, pts_y]
            dist = np.linalg.norm(grid_pts[:, None, :] - player_pts[None, :, :], axis=2)
            nearest = np.argmin(dist, axis=1)
            owner = (nearest < 11).astype(int).reshape(gx.shape)

            fig.add_trace(go.Heatmap(
                z=owner, x=np.linspace(0, 100, 120), y=np.linspace(0, 100, 80),
                colorscale=[[0, "rgba(59,130,246,0.55)"], [1, "rgba(239,68,68,0.55)"]],
                showscale=False
            ))
            for i in range(len(pts_x)):
                col = "#ef4444" if i < 11 else "#3b82f6"
                fig.add_trace(go.Scatter(
                    x=[pts_x[i]], y=[pts_y[i]], mode="markers+text",
                    text=[f"#{dorsales_v[i]}"], textfont=dict(color="white", size=10),
                    marker=dict(size=22, color=col, line=dict(color="white", width=2)),
                    showlegend=False
                ))
            st.plotly_chart(fig, use_container_width=True)

        elif vista == "Informe Jugador":
            c_j1, c_j2 = st.columns(2)
            eq_j = c_j1.selectbox("Club:", [eq_local, eq_visitante])
            jug_lista = sorted(df[df["Equipo"] == eq_j]["Jugador"].dropna().unique())
            jug_activo = c_j2.selectbox("Seleccionar Futbolista:", jug_lista)

            df_j = df[(df["Equipo"] == eq_j) & (df["Jugador"] == jug_activo)]
            p1, p2 = st.columns(2)
            with p1:
                fig_j1 = crear_campo_plotly(f"INTERVENCIONES — {jug_activo}")
                fig_j1.add_trace(go.Scatter(x=df_j["X"], y=df_j["Y"], mode="markers", marker=dict(size=10, color="#38bdf8", line=dict(color="white", width=1)), name="Acciones"))
                st.plotly_chart(fig_j1, use_container_width=True)
            with p2:
                fig_j2 = crear_campo_plotly(f"MAPA DE PASES — {jug_activo}")
                pases_j = df_j[df_j["Evento"] == "Pase"]
                for _, r in pases_j.iterrows():
                    fig_j2.add_trace(go.Scatter(x=[r["X"], r["X_Fin"]], y=[r["Y"], r["Y_Fin"]], mode="lines+markers", line=dict(color="#22c55e" if r["Resultado"]=="Exitoso" else "#ef4444", width=2), showlegend=False))
                st.plotly_chart(fig_j2, use_container_width=True)
