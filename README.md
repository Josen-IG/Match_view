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
no etiqueta tiros, córneres ni goles de forma automática: las antiguas
reglas geométricas producían falsos positivos y están desactivadas hasta
disponer de un detector temporal validado con vídeos etiquetados.

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

## Seguimiento temporal del balón y jugadores ocultos

El backend distingue ahora tres clases de coordenadas del balón:

- **YOLO (`bf=yolo`)**: detección observada que ha superado el filtro.
- **Interpolación (`bf=interpolado`)**: posición reconstruida dentro de un
  hueco corto con observaciones a ambos lados; no es una detección.
- **Kalman (`bf=kalman`)**: pronóstico desde una observación anterior,
  limitado a 1 segundo; si no reaparece, la posición queda desconocida.

El filtro ya no reintroduce como detecciones reales los puntos intermedios
interpolados. Las muestras pueden guardar `bu` (incertidumbre interna de
posición en coordenadas de imagen 0-100) y `bg` (segundos sin observación);
**no son errores medidos contra anotaciones humanas**. Se rechazan cambios
incompatibles con la trayectoria y las detecciones descartadas no vuelven
a contarse como balón observado.

El tracker de jugadores guarda en `pe` hipótesis de posición a muy corto
plazo para IDs temporalmente ocultos. **No se mezclan con `p` ni `pp`**,
que representan detecciones reales y posiciones calibradas respectivamente;
no afectan a distancias, velocidad ni radar. El backend marca la edad y
la incertidumbre creciente, y deja de predecir tras el horizonte corto.
Un portero fuera de cámara durante minutos sigue teniendo posición
**desconocida**, aunque se conserve su último ID.

`quality_metrics.py` distingue cobertura YOLO, interpolación, predicción,
incertidumbre interna y número de hipótesis de jugadores ocultos. La
interfaz no cambia; los campos nuevos aparecen tras **reanalizar el vídeo**.

### Reentrenamiento de YOLO: datos necesarios

Los pesos de `modelo_propio.pt` **no se han reentrenado** en esta fase:
las imágenes completas del dataset no están en GitHub. En `dataset/`
están las etiquetas, `data.yaml` y el proceso de preparación, mientras
que `dataset/README.md` indica dónde está el ZIP de imágenes de
entrenamiento. El ajuste posterior debe usar esos archivos y validar
la detección de balón con partidos completos no vistos durante el entreno
para evitar fugas entre fotogramas del mismo partido.

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

### Jugadas y eventos verificados

Las antiguas reglas de velocidad y dirección del balón confundían despejes,
centros o pases largos con tiros. **Se han desactivado por defecto**:
`suggest_key_moments` devuelve una lista vacía en producción y el visor
omite los eventos heredados con `source=ball_pitch_heuristic`. Se conserva el
código experimental para evaluar hipótesis en vídeos anotados
(`experimental=True`), pero sus resultados no se publican como eventos
reales.

La línea de tiempo, el reproductor, las anotaciones y los cortes manuales
siguen disponibles. Los eventos aportados por una fuente independiente
con trazabilidad no se deben mezclar con las hipótesis geométricas.
Para reactivar **tiros, goles o córneres automáticos** se necesita un banco
de clips etiquetados y un detector temporal que verifique contexto de
jugador-balón, portería y reanudación, con evaluación de precisión y recall
en vídeos reales.

### Analítica observada, sin cifras inventadas

Debajo de la lista de jugadas se muestran métricas directamente derivadas
de `samples` del tracking: número de muestras, porcentaje con balón visto
directamente por YOLO, porcentaje con posición del balón calibrada,
detecciones de personas por muestra y distribución de observaciones de
jugadores sobre el terreno. Se puede filtrar por tramos de cinco minutos.
**Cobertura no significa precisión**; tampoco se confunden detecciones con
jugadores únicos. Si no existen coordenadas calibradas, el mapa permanece
sin datos en vez de proyectar píxeles linealmente.

El analizador mantiene una sola vista de vídeo y prioriza radar, posesión y
contexto táctico en la columna derecha. Configuración, dibujos, cortes y
exportaciones están agrupados en paneles desplegables.

### Fiabilidad de coordenadas del radar (VEO)

La pantalla de vídeo conserva la detección YOLO de jugadores y balón
aunque no pueda construir un radar fiable. Las posiciones de campo
se incluyen en los nuevos JSON solo si hay una homografía de calidad
media o alta, con reproyección ≤1,5 m, inliers ≥70 % y recalibración
con una antigüedad máxima de 3 s. Esto evita dibujar jugadores usando
homografías mantenidas durante decenas de segundos tras el movimiento
de la cámara.

El visor utiliza `rk` para indicar si la calibración de esa muestra
supera el control de calidad y `ra` para la edad de la calibración.
Si falta `rk` (JSON generado antes de esta mejora), no presenta
posiciones de campo como fiables: **hay que reanalizar ese vídeo**.
También se rechazan saltos imposibles de IDs y cambios colectivos de
posición sospechosos entre muestras.

Este filtro limita errores evidentes, pero no demuestra que una
homografía aceptada coincida exactamente con el vídeo. Para validar
la precisión hacen falta fragmentos con líneas/puntos del campo
anotados manualmente y comparar reproyección, posiciones y estabilidad
en grabaciones reales de distintos partidos.

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
