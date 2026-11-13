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
equipos por color de camiseta (heredado de Alineaciones), homografía
para posiciones reales en metros y radar con los jugadores. Genera los
cortes como botones con un resumen breve; el guardado en eventos es
manual, nunca automático.

- El modelo de keypoints del campo (`pitch_keypoints.pt`, 85 MB) tiene
  que estar junto a `video_track.py`. Sin él no hay mapa ni radar.
  Enlace de descarga en `README_VIDEO.md`.
- Los pesos YOLO (`yolo11s.pt`) están en el repositorio.
- Requiere `ffmpeg` y `ffprobe` en `PATH`. Los clips se guardan en
  `video_clips/` junto al vídeo.
- El análisis completo de un partido (105 min) tarda ~3-4 h en CPU a
  5 fps; en un Ryzen 7 se puede subir a 10 fps. Ver `README_VIDEO.md`
  para el detalle de las reglas y las versiones.

Groq se eliminó: la clasificación es solo con YOLO y las reglas de
`video_track.py`. `video_ai.py` (flujo antiguo con Groq) y `prueba.py`
(prototipo) se mantienen en el repo por referencia, no son la app base.

## Datos

- `Plantillas preferente.csv/.xlsx`: plantillas de los 16 equipos.
- `FC26_20250921.csv`: datos FIFA 26 / Sofifa (19-09-2025).
- `Futbase.ipynb`: análisis de jugadores base. Lee `Futbase.csv`, que
  no está en el repositorio (punto 5, evaluación de cantera, fuera de
  alcance por ahora).
- `futbol_base.html`: formulario de registro de jugador base.
