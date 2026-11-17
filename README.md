# Match_view / Fut_base

Herramienta para entrenadores: registro de eventos de partido, análisis de
vídeo con YOLO y dashboard postpartido. Regional Preferente de Navarra.

## Ejecutar la aplicación

En Windows, instala las dependencias de Python y ejecuta la app base
(`match.py`, no `prueba.py`, que es solo un prototipo):

```powershell
py -m pip install -r requirements.txt
py -m streamlit run match.py
```

`requirements.txt` incluye `matplotlib` y `mplsoccer`, que la app usa para
renderizar el campo y el mapa de calor; si faltan, la aplicación cae a un
fallback manual sin romper la ejecución.

La app tiene 4 pantallas: pre-partido (alineaciones y colores), etiquetado
en vivo, vídeo y dashboard postpartido.

## Acceso

La pantalla de acceso pide una llave. Se lee de la variable de entorno
`MATCH_VIEW_PASSWORD` o de los secrets de Streamlit
(`.streamlit/secrets.toml` con `access_password = "..."`). Si no hay
ninguna, la llave es `1234`: cámbiala poniendo la variable de entorno
antes de arrancar Streamlit.

## Análisis de vídeo (YOLO, sin Groq)

La pantalla de Vídeo analiza el partido con YOLO11 (`video_track.py`):
detección y seguimiento propio de personas y balón, clasificación de
equipos por color de camiseta (definidos en Alineaciones), homografía
para posiciones estimadas en metros y radar con los jugadores. El sistema
también identifica **candidatos a tiro, córner y posible gol** usando
trayectorias del balón proyectadas al campo, sin confundirlos con eventos
confirmados. La detección es orientativa y requiere revisión del vídeo.

- El modelo de keypoints del campo (`pitch_keypoints.pt`, 85 MB) tiene
  que estar junto a `video_track.py`. Sin él no hay mapa ni radar.
  Enlace de descarga en `README_VIDEO.md`.
- El análisis usa por defecto el modelo personalizado
  `modelo_propio.pt`; puede indicarse otro peso YOLO.
- Requiere `ffmpeg` y `ffprobe` en `PATH`. Los clips se guardan en
  `video_clips/` junto al vídeo.
- En CPU el análisis de un partido completo puede tardar varias horas,
  según el equipo, vídeo y FPS. Empieza con 10 minutos de prueba; en la
  pantalla se puede usar una cola de análisis en segundo plano (un trabajo
  a la vez) con progreso. El trabajo se pierde si se reinicia el servidor.
  Ver `README_VIDEO.md` para el detalle del detector y los parámetros.

Groq se eliminó: la clasificación es solo con YOLO y las reglas de
`video_track.py`. `video_ai.py` (flujo antiguo con Groq) y `prueba.py`
(prototipo) se mantienen en el repo por referencia, no son la app base.

## Tracking, métricas e informes

El análisis de fútbol incorpora módulos reutilizables:

| Archivo | Responsabilidad |
| --- | --- |
| `video_track.py` | Detección YOLO, balón, homografía, posesión y exportación |
| `player_tracking.py` | Asociación de IDs, compensación pan/zoom y reidentificación conservadora |
| `player_metrics.py` | Distancia, velocidad, aceleraciones, sprints y zonas por ID |
| `tactical_metrics.py` | Geometría visible, presión aproximada, ocupación y periodos |
| `quality_metrics.py` | Cobertura del tracking, avisos de calidad e inconsistencias |
| `match_reports.py` | Informes PDF, Excel y CSV exportables desde JSON |
| `video_jobs.py` | Cola local de vídeo en segundo plano y progreso por trabajo |
| `event_candidates.py` | Sugerencias conservadoras de tiros, córneres y posibles cruces de línea de gol |
| `match.py` | Interfaz Streamlit: filtros, mapas, reportes, descargas |

Cada archivo se mantiene una sola vez. Los módulos de análisis se pueden
importar para pruebas sin abrir Streamlit.

### Jugadas sugeridas por IA

Tras analizar un partido en la pantalla de vídeo, el JSON generado
automáticamente incorpora `suggested_events` con momentos **posibles**:
tiros, córneres y cruces de línea de gol. No son eventos verificados;
los goles no se contabilizan automáticamente.

El modelo exige coordenadas de balón calibradas al campo (`bp`) y
detecciones directas de YOLO (`bf=yolo`). Si faltan, **no inventa
jugadas**. Los clips sugeridos incluyen aproximadamente ocho segundos
antes y seis después de la detección, sujetos al tramo analizado.

En el visor, la línea de tiempo muestra una pista de **Jugadas IA**
con marcadores de colores. Se puede reproducir el momento, guardar un
corte y clasificarlo como revisado o descartado. Los estados de revisión
y cortes sugeridos se guardan localmente en el navegador cuando está
disponible su almacenamiento. No es necesario subir archivos JSON:
`match.py` lee el tracking automáticamente. Los datos antiguos con
coordenadas válidas pueden generar sugerencias en memoria.

**Limitación importante:** estas reglas geométricas aún no han sido
validadas con partidos completos etiquetados. La perspectiva y los
movimientos de cámara pueden generar falsos positivos y omisiones.

### Radar táctico sincronizado y posesiones

El visor de **Vídeo** muestra un radar 2D junto al reproductor. Las posiciones
se dibujan únicamente a partir de las coordenadas métricas `pp` (jugadores)
y `bp` (balón) generadas mediante homografía. Se actualiza al avanzar,
pausar o buscar un instante. Cuando faltan datos, el radar indica cobertura
parcial o ausencia de posiciones en vez de reconstruir jugadores ocultos.

La tarjeta **Posesiones** ofrece cantidad y duración media de los tramos con
control estimado, además de una lista filtrable por equipo y por el tramo
actual de cinco minutos. Se separan posesiones por cambio de equipo,
disputa o hueco temporal; solo se visualizan segmentos observados
suficientemente largos. Al pulsar una posesión, el vídeo salta a su inicio.
Las zonas se muestran como *tercio izquierdo, central o derecho* y únicamente
cuando el balón tiene coordenadas de campo válidas. No se infiere
automáticamente qué equipo está atacando a qué portería.

El panel **Contexto táctico** aprovecha `tactical_analysis` para mostrar
anchura, profundidad, jugadores visibles y proximidad de defensores al balón
por periodos de cinco minutos. Exige varias muestras con cobertura útil;
si no existen, indica información insuficiente. No afirma formaciones ni
alturas de bloque cuando no puede verificarlas.

El visor y `match.py` cargan el tracking almacenado automáticamente, sin
que sea necesario subir archivos JSON. La línea de tiempo IA y la revisión
de jugadas sugeridas se conservan.

### Estadísticas por jugador

Se muestran por `track_id`, **no por dorsal verificado**. Los cálculos de
distancia y velocidad necesitan homografía; si no hay posiciones válidas, las
métricas se dejan sin dato. Los saltos de velocidad poco creíbles se ignoran.
El tiempo visible no equivale necesariamente a minutos jugados.

Las métricas tácticas describen los jugadores **visibles en cámara**. Con una
cámara que sigue el balón no se puede asegurar una formación completa ni
medir con exactitud la presión del equipo. Las recuperaciones de posesión
son cambios estimados, no eventos etiquetados. Para evaluar la continuidad
de identidad de los IDs hacen falta vídeos con anotaciones humanas.

### Archivos generados

El tracking se guarda como `<video>_tracking.json`, conservando el campo
`p` del formato anterior; `pp` incorpora opcionalmente un cuarto elemento,
el ID de track. Se incluyen además `player_stats`, `tactical_analysis`
y `quality`, además de `suggested_events`. Los datos antiguos se pueden seguir abriendo; para obtener las
métricas nuevas hay que regenerar el tracking.

El informe genera, junto al JSON:

- `<video>_tracking_informe.pdf`: resumen postpartido, tablas y mapas.
- `<video>_tracking_estadisticas.xlsx`: resumen, jugadores, táctica y eventos.
- `<video>_tracking_jugadores.csv` y `<video>_tracking_tactica.csv`.

También se pueden regenerar o descargar desde la pantalla de Vídeo.
El PDF y el Excel requieren `reportlab` y `XlsxWriter`, incluidos en
`requirements.txt`.

### Ejecutar las pruebas

```powershell
py -m pip install -r requirements.txt
py -m unittest discover -s tests -v
py -m streamlit run match.py
```

GitHub Actions ejecuta las pruebas en cada commit de la rama y en las
propuestas de integración hacia `main`.

### Validación recomendada antes de integrar en `main`

Probar primero 10 minutos de un vídeo VEO real con ambos equipos y
porteros visibles en distintos tramos. Comprobar continuidad de IDs en
cruces, oclusiones y desplazamientos de cámara; cobertura del campo,
métricas individuales, posesión, mapas, calidad e informes. Comparar con
la versión original del mismo vídeo. Las pruebas automatizadas incluyen
simulaciones, **no certifican precisión en un partido real**.

## Datos

- `Plantillas preferente.csv/.xlsx`: plantillas de los 16 equipos.
- `FC26_20250921.csv`: datos FIFA 26 / Sofifa (19-09-2025).
- `Futbase.ipynb`: análisis de jugadores base. Lee `Futbase.csv`, que
  no está en el repositorio (punto 5, evaluación de cantera, fuera de
  alcance por ahora).
- `futbol_base.html`: formulario de registro de jugador base.
