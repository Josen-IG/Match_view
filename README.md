# Fut_base
Proporcionar a los entrenadores de inferiores una interfaz gráfica para visualizar el progreso de sus jugadores a lo largo de la temporada, pasando de la recolección manual de datos a dashboards interactivos.

## Ejecutar la aplicación

En Windows, instala las dependencias de Python y ejecuta la app:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run prueba.py
```

La creación y el análisis de vídeo requieren que `ffmpeg` y `ffprobe` estén
instalados y disponibles en `PATH`. En la pantalla de etiquetado, introduce la
ruta local del vídeo; el archivo completo no se carga en Streamlit. Los clips
generados se guardan en una carpeta `<nombre_del_video>_clips` junto al vídeo.

Para el análisis asistido, configura la variable de entorno `GROQ_API_KEY` en
PowerShell y ejecuta la aplicación desde esa misma terminal:

```powershell
$env:GROQ_API_KEY = "tu_clave_nueva"
py -m streamlit run prueba.py
```

La clave no se pide en la interfaz ni se escribe en el código fuente.

YOLO detecta y sigue personas y balón, y prioriza momentos según las detecciones
y su movimiento; Groq clasifica una selección de esos fotogramas y FFmpeg genera
clips H.264/AAC para las sugerencias con confianza suficiente. El informe muestra
lo que YOLO detectó. Los fotogramas seleccionados se envían a Groq; el vídeo
completo permanece local. Las etiquetas son sugerencias y deben revisarse: este
flujo no garantiza detectar goles u otras acciones con precisión. La primera
ejecución puede descargar los pesos del modelo YOLO.
