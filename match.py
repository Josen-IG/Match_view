import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, date
import time
import os
from pathlib import Path


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

FORMACIONES_COORDS = {
    "1-4-4-2": [(8,50), (22,18), (22,38), (22,62), (22,82), (36,18), (36,38), (36,62), (36,82), (46,35), (46,65)],
    "1-4-3-3": [(8,50), (22,18), (22,38), (22,62), (22,82), (34,30), (32,50), (34,70), (45,20), (46,50), (45,80)],
    "1-4-2-3-1": [(8,50), (20,18), (20,38), (20,62), (20,82), (30,35), (30,65), (40,20), (40,50), (40,80), (47,50)],
    "1-3-5-2": [(8,50), (20,25), (20,50), (20,75), (34,15), (34,32), (32,50), (34,68), (34,85), (46,35), (46,65)],
    "1-5-3-2": [(8,50), (20,15), (20,32), (20,50), (20,68), (20,85), (34,30), (34,50), (34,70), (46,35), (46,65)]
}


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

    def registrar_evento(accion, sub_accion=""):
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
            "Detalle": sub_accion
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

    col_v1, col_v2, col_v3 = st.columns([1, 1.2, 3])
    with col_v1:
        if st.button("⬅️ Partido", use_container_width=True):
            st.session_state.pagina = "partido"
            st.rerun()
    with col_v2:
        if st.button("📊 Final / CSV", use_container_width=True):
            st.session_state.pagina = "final"
            st.rerun()
    with col_v3:
        st.markdown(f"### 🎥 Análisis de Vídeo: {dp['eq_local']} vs {dp['eq_visitante']}")

    st.divider()

    # --- Análisis manual del vídeo ---
    # El análisis manual se hace en esta pantalla (analizador_video.html,
    # junto a match.py), que se alimenta de un tracking JSON generado por
    # video_track.py (export_tracking_json) o de uno ya generado antes.
    st.markdown("#### 🎬 Análisis manual del vídeo")
    st.caption(
        "Aquí se analiza el partido a mano: carga el vídeo y el tracking JSON "
        "y verás el tracking encima del vídeo (cajas, balón, trayectoria), "
        "las estadísticas en vivo y las herramientas para señalar jugadas "
        "(flechas, círculos, texto) y marcar cortes. Todo lo que anotes se "
        "exporta en JSON al final. La pantalla no lee los archivos sola: "
        "tienes que elegirlos con los selectores de archivo de dentro."
    )
    ruta_pantalla = Path(__file__).resolve().parent / "analizador_video.html"
    if ruta_pantalla.is_file():
        html_pantalla = ruta_pantalla.read_text(encoding="utf-8")
        components.html(html_pantalla, height=1100, scrolling=True)
    else:
        st.warning("⚠️ No se encontró analizador_video.html junto a match.py; "
                   "la pantalla de análisis manual no se puede mostrar.")
