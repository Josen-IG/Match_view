# Vídeo — detección de jugadas para Match_view (v2)

Nuevo módulo de análisis de vídeo para `match.py`. No toca `prueba.py` ni el
repo: los archivos viven en `~/workspace/match_view_video_v2/` y se copian a
mano a la carpeta de Match_view cuando Josenrique lo decida.

- `video_track.py` — tracking continuo con YOLO + clasificación por tracking + cortes con FFmpeg. Sin Groq: no hace falta API key ni internet.
- `match.py` — copia completa de `match.py` (commit a415be4) con la pantalla **Vídeo** añadida.
- `README_VIDEO.md` — este archivo.

## Qué cambió respecto a video_ai.py

| Antes (video_ai.py) | Ahora (video_track.py) |
|---|---|
| Una foto cada 5 s, YOLO por lotes de imágenes fijas | `YOLO.track(persist=True, tracker="bytetrack.yaml")` fotograma a fotograma, IDs persistentes en todo el vídeo |
| Solo contaba personas (clase 0) y balón (32) | Personas + balón, con equipo por color de camiseta (voto por mayoría por track ID) |
| Balón sin seguir: si se perdía, desaparecía | Balón suavizado EMA + interpolación de huecos ≤ 1 s |
| Candidatos por cambios de conteo | Candidatos por señales del tracking: pico de velocidad, parada brusca, agrupación, balón fuera de plano, balón parado, saque de portero, transición, cambio de posesión solo, balón parado largo, entrada al área, pase largo |
| Clasificación con Groq (visión) | Clasificación solo con el tracking: ventana de ±3 s con velocidad, balón perdido, balón parado, cambio de posesión, agrupación, avance y (en fija amplia) cruce de la línea de gol y entrada al área. Sin API key ni límites de tokens |
| Solo gol/tiro/córner/falta | Todas las jugadas: gol, tiro a puerta/fuera, pase, pase clave, centro, córner, falta, tarjeta, duelo, recuperación, pérdida, transición/contraataque, entrada al área, balón parado, distribución del portero, posesión por tramos |
| Coordenadas de campo fijas | Tracking relativo: la cámara sigue al balón, así que la posición en imagen NO es posición en el campo (ver "Limitaciones") |
| Analizaba desde el minuto 0 | El calentamiento no se analiza: saque inicial manual + sugerencia automática |
| Colores fijos de match.py | Colores heredados de Alineaciones, sin volver a elegirlos en Vídeo |

## Respuestas del usuario (2026-11-11) — ya aplicadas

1. **Vídeo**: partido completo, pero los primeros minutos son calentamiento.
   Saque inicial manual en la UI (MM:SS, p. ej. 04:32); todos los cortes son
   relativos a ese inicio. `suggest_kickoff()` propone el primer momento con
   balón en el centro y formación 11+11, pero el manual manda siempre.
2. **Cámara**: fija pero **sigue al balón**. Principalmente VEO, a veces móvil.
   El tracking es relativo (trayectoria, proximidad, velocidad, agrupación) y
   compensa el movimiento de cámara (mediana del desplazamiento de los
   jugadores) y el zoom (distancias en alturas de jugador). La zona Z1-Z9
   solo se calcula como aproximación en modo "fija amplia"; en modo
   "seguimiento" queda "Sin especificar". No se asume vídeo panorámico de VEO.
3. **Equipos**: por colores, heredados de Alineaciones. La pantalla de
   Vídeo usa directamente el Color Local / Visitante puestos en
   Alineaciones, sin volver a elegirlos. Si hay que cambiarlos, se cambian
   en Alineaciones.
4. **Jugadas**: todas las variables, análisis en profundidad (lista arriba).
   Prioridad a gol/tiro/córner, pero se cubren todas.
5. **PC**: Windows, Ryzen 7 serie 7000, **CPU (sin GPU)**. YOLO11s por
   defecto, imgsz 640, muestreo ~10 fps, `device="cpu"`.
6. **Cortes**: solo botones con resumen breve, guardado **manual** a eventos.
   Nada se añade solo al CSV.

## Cómo ejecutar

En PowerShell, en la carpeta de Match_view:

```powershell
py -m pip install streamlit pandas ultralytics opencv-python-headless
py -m streamlit run match.py
```

1. Entrar con la llave (1234, sin cambios en esta tarea).
2. Configurar equipos y colores en Alineaciones (son los que usa el vídeo).
3. Ir a **🎥 Vídeo** desde Partido o desde Final.
4. Escribir la ruta local del vídeo del partido completo.
5. Poner el saque inicial (MM:SS) o pulsar **🔍 Sugerir inicio** y revisarlo.
6. Revisar los colores heredados de Alineaciones que muestra la pantalla.
7. Elegir parámetros (por defecto: 10 fps, 15 candidatos, confianza 0,65, YOLO11s).
8. **▶️ Analizar vídeo**. Los cortes aparecen como botones
   `MM:SS | Gol (82%) - resumen breve`. Al pulsar uno se reproduce el clip
   y se puede **➕ Añadir a eventos** (manual, uno a uno). Los descartados /
   inciertos se muestran aparte, con el motivo.
9. En **Final** se revisa y se descarga el CSV para Power BI como siempre.

### Por terminal (sin Streamlit)

```powershell
# Solo sugerir el saque inicial
python video_track.py "C:\Videos\partido.mp4" --suggest-kickoff

# Analizar los primeros 10 minutos (prueba)
python video_track.py "C:\Videos\partido.mp4" --kickoff 04:32 --max-minutes 10

# Partido completo
python video_track.py "C:\Videos\partido.mp4" --kickoff 04:32
```

## Supuestos y defaults (configurables)

| Parámetro | Default | Dónde se cambia |
|---|---|---|
| Modelo YOLO | `yolo11s.pt` (11n más rápido, 11m más preciso) | UI Vídeo / `--yolo-model` |
| Muestreo | 10 fps | UI Vídeo / `--sample-fps` |
| imgsz | 640 | parámetro `imgsz` |
| Máx. candidatos | 15 (límite 20) | UI Vídeo / `--max-candidates` |
| Confianza mínima | 0,65 | UI Vídeo / `--min-confidence` |
| Clip | 8 s antes / 7 s después, H.264/AAC | `pre_roll` / `post_roll` |
| Separación mínima entre candidatos | 20 s | `MIN_GAP_SECONDS` |
| Modo de cámara | `seguimiento` (VEO/móvil) | UI Vídeo / `--camera-mode` |
| Clips | carpeta `<video>_clips`, nombre `MMSS_evento.mp4` | automático |

## Cómo clasifica YOLO cada jugada (sin Groq)

`_clasificar_por_tracking()` mira una ventana de ±3 s alrededor de cada
candidato: velocidad del balón, si se pierde tras la jugada, cuánto llevaba
parado, si cambia la posesión, cuánta gente hay junto al balón, cuánto
avanza y (solo en fija amplia) si cruza la línea de gol o entra al área.

| Jugada | Cómo se distingue | Confianza |
|---|---|---|
| Gol | Solo fija amplia: balón que cruza la línea de gol rápido | 0,80 |
| Tiro | Balón perdido a gran velocidad, o rápido hacia la línea de gol | 0,70 |
| Córner | Balón parado ≥2 s + reanudación rápida desde la esquina (fija amplia) | 0,70 |
| Balón parado | Balón parado ≥2 s + reanudación rápida | 0,65 |
| Distribución portero | Balón parado ≥2 s + reanudación, con poca gente cerca | 0,65 |
| Falta | Parada brusca + agrupación + cambio de posesión | 0,60 |
| Centro | Balón rápido y largo desde banda hacia el área (fija amplia) | 0,65 |
| Transición / contraataque | Cambio de posesión + avance rápido | 0,70 |
| Duelo | Agrupación + cambio de posesión | 0,60 |
| Recuperación | Cambio de posesión sin más | 0,65 |
| Pase clave | Mismo equipo, balón largo que acaba en el área (fija amplia) | 0,65 |
| Entrada al área | Del tercio medio al tercio final (fija amplia) | 0,60 |
| Pase | Balón rápido que recorre distancia, mismo equipo | 0,65 |

La confianza nunca pasa de 0,80: sin ver la jugada no se puede asegurar
un gol o un córner. La posesión por tramos (15') se sigue calculando del
tracking y se muestra en la pantalla de Vídeo.

## Cómo probar

1. **Prueba rápida (10 min)**: vídeo de un partido, `--max-minutes 10` o
   "Primeros 10 minutos (prueba)" en la UI, con el saque inicial puesto a mano.
   Comprueba: ¿los cortes tienen sentido? ¿YOLO clasifica bien gol/tiro/córner?
2. **Sugerencia de inicio**: `--suggest-kickoff` y compara con el saque real.
3. **Colores**: comprueba en Vídeo que los colores heredados de Alineaciones
   son las camisetas de cada equipo. Si no, cámbialos en Alineaciones.
4. **Partido completo**: solo cuando la prueba de 10 min vaya bien. En CPU
   tarda más que en tiempo real (ver abajo).

Verificación de sintaxis realizada el 2026-11-11:

```
python -m py_compile video_track.py  -> OK
python -m py_compile match.py        -> OK
```

Y prueba de funciones puras (sin vídeo ni GPU): `parse_mmss("04:32") = 272.0`,
`zona_desde_xy(83, 18) = Z9 - Ataque Der`, deduplicación de candidatos y
posesión por tramos correctas. El análisis completo con vídeo real aún no
se ha probado: hace falta un vídeo de partido.

## Limitaciones conocidas

- **Cámara que sigue al balón**: la posición en imagen no es posición en el
  campo. La zona Z1-Z9 no se puede saber con seguridad en modo seguimiento;
  por eso queda "Sin especificar" (mejor no inventarla). En modo fija amplia
  es solo una aproximación. Esto afecta también a la distinción de jugadas
  que dependen de la zona: en seguimiento, el gol sale como tiro, el córner
  como balón parado, el centro como pase y la entrada al área no se detecta.
  En fija amplia sí se distinguen (ver la tabla de clasificación).
- **Balón pequeño y rápido**: YOLO11s puede perder el balón en tiros largos
  o con oclusiones. Se interpola hasta 1 s; si se pierde más, la señal se
  pierde y la jugada puede salir como tiro o quedar incierta.
- **VEO panorámico**: si el vídeo es panorámico (muy ancho), el tracking
  sigue funcionando, pero las distancias en alturas se calculan igual; la
  zona sigue sin ser fiable. No se ha probado con ese formato.
- **Móvil**: más temblor y otro ángulo. La compensación de cámara ayuda, pero
  un vídeo muy movido degradará el tracking.
- **Sugerencia de saque inicial**: heurística (balón en el centro + 15+
  personas + formación). En calentamiento también puede haber muchos
  jugadores en campo; puede fallar. El valor manual manda siempre.
- **Colores**: si las camisetas son parecidas (p. ej. blancas vs grises) o
  hay mucho sol/sombra, la asignación por color falla y muchos jugadores
  quedan "desconocido". La asignación manual de la paleta mitiga esto.
- **Clasificación sin ver la jugada**: las etiquetas son heurísticas sobre
  el tracking, no una confirmación visual. La confianza nunca pasa de 0,80
  y las jugadas ambiguas (falta, duelo) quedan en 0,60. Los candidatos
  descartados se muestran con el motivo para que se vea por qué.
- **CPU**: en Ryzen 7 sin GPU, un partido de 90 min a 10 fps tarda más que
  en tiempo real (estimación: 1,5-3 h según el PC). Si va lento: bajar a
  5 fps, usar `yolo11n.pt`, o analizar por mitades.
- **Posesión por tramos**: se calcula del tracking (jugador más cercano al
  balón), no de un cronómetro de posesión real; es una aproximación.

## Qué sigue pendiente (del usuario / por validar)

- Las 5 preguntas ya están respondidas y aplicadas (ver arriba).
- **Validar con un vídeo real de partido**: probar 10 min y ajustar
  umbrales (velocidad, agrupación, confianza) si los cortes no tienen sentido.
- Confirmar si el VEO da vídeo panorámico o normal (cambia la calibración
  de distancias, no el flujo).
- Confirmar la orientación del campo en cada grabación (si el ataque va a la
  derecha o a la izquierda en la 1ª parte) si se quiere usar el modo fija
  amplia con zonas.
- La contraseña "1234" sigue hardcodeada (fuera del alcance de esta tarea).
- `prueba.py` no se ha tocado (era solo una prueba de dirección).
