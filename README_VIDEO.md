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
| Solo contaba personas (clase 0) y balón (32) | Personas + balón, con equipo por color de camiseta (el más votado entre los conocidos por track ID) |
| Balón sin seguir: si se perdía, desaparecía | Balón suavizado EMA + interpolación de huecos ≤ 1 s |
| Candidatos por cambios de conteo | Candidatos por señales del tracking: pico de velocidad, parada brusca, agrupación, balón fuera de plano, balón parado, saque de portero, transición, cambio de posesión solo, balón parado largo, entrada al área, balón cerca del portero, pase largo |
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
   en Alineaciones. También se ponen allí los colores de cada portero
   (amarillo y naranja por defecto) y del árbitro (negro): el portero
   cuenta como su equipo y su posición marca qué portería se ataca; el
   árbitro no cuenta para posesión ni agrupación. Los colores tienen que
   ser distintos de los de campo; si un portero cambia de camiseta, se
   cambia su color en Alineaciones.
4. **Jugadas**: todas las variables, análisis en profundidad (lista arriba).
   Prioridad a gol/tiro/córner, pero se cubren todas.
5. **PC**: Windows, Ryzen 7 serie 7000, **CPU (sin GPU)**. YOLO11s por
   defecto, imgsz 960, conf 0,10, muestreo ~10 fps, `device="cpu"`.
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
| Muestreo | 10 fps (5 fps si la CPU va lenta) | UI Vídeo / `--sample-fps` |
| imgsz | 960 | parámetro `imgsz` / `--imgsz` |
| Confianza YOLO | 0,10 | parámetro `conf` / `--conf` |
| Máx. candidatos | 15 (límite 20) | UI Vídeo / `--max-candidates` |
| Confianza mínima | 0,65 | UI Vídeo / `--min-confidence` |
| Clip | 8 s antes / 7 s después, H.264/AAC | `pre_roll` / `post_roll` |
| Separación mínima entre candidatos | 20 s | `MIN_GAP_SECONDS` |
| Modo de cámara | `seguimiento` (VEO/móvil) | UI Vídeo / `--camera-mode` |
| Clips | carpeta `<video>_clips`, nombre `MMSS_evento.mp4` | automático |

> **Por qué imgsz 960 y conf 0,10:** en vídeo VEO el balón es diminuto. A
> 640/conf 0,25 solo aparece en ~2% de las muestras y la posesión salía
> 72% desconocido; a 960/0,10 aparece en ~20% y el 70-86% de las personas
> tienen equipo. Cuesta ~1,7x más tiempo que 640; en CPU lenta se compensa
> bajando el muestreo a 5 fps (el partido completo a 5 fps tarda ~3 h en
> una VM lenta; en un Ryzen 7, 10 fps es viable).
>
> **Tracking sin ByteTrack:** se usa `YOLO.predict()` + asociación propia
> (vecino más cercano por posición y altura). ByteTrack se quitó porque con
> cámara que sigue al balón el pan/zoom entre muestras rompía sus tracks
> (de 18 personas a 0 en pocos fotogramas) y se tragaba el balón
> (0% de muestras con balón frente a ~20% con predict()). El equipo por
> track es el más votado entre los conocidos (no mayoría del 60%), y el
> color del torso se mide con la mediana de la mitad más saturada, para
> que las rayas y el contraluz no lo diluyan.

## Cómo clasifica YOLO cada jugada (sin Groq)

`_clasificar_por_tracking()` mira una ventana de ±3 s alrededor de cada
candidato: velocidad del balón, si se pierde tras la jugada, cuánto llevaba
parado, si cambia la posesión, cuánta gente hay junto al balón, cuánto
avanza, distancia a cada portero (su posición marca qué portería se ataca)
y (solo en fija amplia) si cruza la línea de gol o entra al área.

| Jugada | Cómo se distingue | Confianza |
|---|---|---|
| Gol | Fija amplia o **con mapa**: balón que cruza la línea de gol rápido (0,80). Seguimiento: (a) tiro rápido que se pierde ≥4 s y se reanuda en el centro (saque de centro: balón parado **en total** ≥1,5 s en los 5 s tras reaparecer, **≥6 jugadores** de los dos equipos alrededor, lejos de cualquier portero, y la reaparición **en el centro** del campo con mapa); (b) **por colocación**: el balón parado en el centro ≥8 s en 30 s con gente alrededor y un tiro rápido en los 60 s anteriores a la colocación — caza los goles en los que el balón **no se pierde** (la cámara lo sigue en la celebración y el traslado al centro). Los tiempos son totales, no rachas: en seguimiento el balón se pierde a menudo y una sola muestra sin balón rompe la racha. El equipo del gol es el que tenía la posesión **antes** del tiro (el pico de velocidad se busca 8 s atrás, porque el candidato se detecta en la pérdida/celebración, después del tiro); si no hay portero cerca, el equipo es el contrario al portero junto al balón en el tiro (el que encaja) | 0,80 / 0,70 |
| Tiro a puerta | Balón rápido junto al portero | 0,70 |
| Tiro | Balón perdido a gran velocidad, o rápido hacia la línea de gol (con mapa: a menos de 5 m de la línea). **Un balón que acaba en el propio portero es pase atrás, no tiro** (tampoco pase clave ni entrada al área): el guard se mira en el plano y, con mapa, en metros (balón a menos de 12 m del portero propio). **Con mapa, el tiro exige además que el balón vaya hacia la portería rival** (velocidad en x hacia la portería > 2 m/s) **y que acabe cerca de ella** (a menos de 30 m del portero rival en metros; sin portero visible, a menos de 30 m de una línea de gol): un balón rápido que se pierde en el centro del campo es un pase o un despeje, no un tiro (en 10 min sin ningún tiro real salían 7 "tiros"). Un despeje hacia la propia portería no es tiro | 0,70 |
| Córner | Balón parado ≥2 s + reanudación rápida desde la esquina (fija amplia) | 0,70 |
| Balón parado | Balón parado ≥2 s + reanudación rápida | 0,65 |
| Distribución portero | Balón parado ≥2 s + reanudación, con poca gente cerca | 0,65 |
| Falta | Frenazo (el balón pasa de ≥4 a ≤0,8 alturas/s) + agrupación ≥5 + juego parado después (balón parado **en total** ≥1 s o perdido ≥2 s en los 6 s siguientes). El árbitro cerca (≤6 alturas) sube la confianza a 0,65, pero **no es requisito**: solo se ve en el 44% de las muestras. Sin cambio de posesión exigido: en una falta el equipo que tenía el balón lo conserva. Tope de 5 faltas por partido (el resto pasan a duelo) | 0,60 / 0,65 |
| Centro | Balón rápido y largo desde banda hacia el área (fija amplia **o con mapa**: balón en banda en metros reales) | 0,65 |
| Transición / contraataque | Cambio de posesión + avance rápido | 0,70 |
| Duelo | Agrupación + cambio de posesión | 0,60 |
| Recuperación | Cambio de posesión sin más | 0,65 |
| Pase clave | Mismo equipo, balón largo que acaba en el área o en zona del portero (con mapa: que acaba **en el área en metros**) (no si es el propio portero: eso es pase atrás) | 0,65 |
| Entrada al área | Del tercio medio al tercio final (fija amplia) o a zona del portero (con mapa: el balón **cruza la línea del área en metros**) (no si es el propio portero) | 0,60 |
| Pase | Balón rápido que recorre distancia, mismo equipo | 0,65 |
| Pase (por trayectoria) | Cambio de portador del mismo equipo con vuelo (aunque el balón no sea rápido: la mayoría de los pases de centro del campo) | 0,70 |
| Pase clave (por trayectoria) | Pase por trayectoria largo (≥12 alturas) o que acaba en zona del portero (no si es el propio portero) | 0,70 |

La confianza nunca pasa de 0,80: sin ver la jugada no se puede asegurar
un gol o un córner. La posesión por tramos (15') se sigue calculando del
tracking y se muestra en la pantalla de Vídeo.

## Mapa del campo (homografía) — posiciones en metros reales

Cada jugador y el balón se proyectan sobre un campo de **105 x 68 m**.
Eso convierte el tracking relativo en posición real: se sabe si un balón
rápido va hacia la portería rival (tiro) o hacia la propia (despeje o pase
atrás), si cruza la línea de gol (gol), si entra al área, desde qué banda
sale un centro, y las distancias de los pases y la velocidad del balón en
m y km/h.

**Cómo funciona**: un modelo YOLO de pose (Soccana Keypoint, YOLOv11n-pose,
29 puntos clave del campo: líneas, áreas, círculo central, esquinas)
detecta las líneas visibles en un fotograma cada 5 s; con esos puntos se
calcula la homografía (`cv2.findHomography`) que convierte píxeles de la
imagen en metros de campo. La homografía se suaviza entre fotogramas y
caduca a los 30 s sin líneas nuevas. **El modelo se descarga solo la
primera vez** de HuggingFace (`pitch_keypoints.pt`, ~86 MB); si el archivo
está junto a `video_track.py` se usa directamente, y si la descarga falla
el análisis sigue sin mapa.

**Calidad**: depende de qué líneas se ven. Si solo se ve el centro del
campo (círculo central, línea de medio campo) la calidad es **baja** y la
coordenada x (la longitud del campo) sale comprimida hacia el centro.
Para corregirlo se usa un ancla que no falla: **los dos porteros están en
sus líneas de gol** (x = 0 y x = 105 m). El eje x se estira para que los
porteros queden en sus líneas; la coordenada y (la anchura) ya viene bien
constrainida por las líneas del centro.

**Radar en los cortes**: cada corte lleva el mapa del campo con los
jugadores como **puntos de colores** (color de cada equipo, porteros con
borde, balón blanco) incrustado **centrado abajo** en el clip, pequeño
(480x300, como el radar de los videojuegos) y semitransparente (70%,
se ve el vídeo por debajo), y una imagen PNG del radar en el momento
de la jugada (en la pantalla de Vídeo se muestra al seleccionar el
corte). Los "desconocido" se dibujan en gris: antes no se dibujaban y
el radar salía vacío en los momentos en los que casi todos eran
desconocido. El radar es un vídeo (2 fps), no una foto fija: los
puntos se mueven con los jugadores durante el corte.

**Higiene del repo v12 (2026-11-12):** la llave de acceso "1234" estaba
fija en el código de `match.py` y `prueba.py` y el mensaje de error la
revelaba. Ahora se lee de la variable de entorno `MATCH_VIEW_PASSWORD`
o de los secrets de Streamlit (`.streamlit/secrets.toml`); si no hay
ninguna, sigue siendo "1234" para no romper el acceso. `README.md`
reescrito: describía el flujo viejo con Groq y `prueba.py`; ahora dice
que la app base es `match.py`, que el vídeo es solo YOLO (Groq
eliminado) y que `Futbase.csv` no está en el repo. `prueba.py` marcado
como prototipo con datos demo (rellena minuto/coordenadas con valores
aleatorios si el CSV no los trae).

**Cobertura del mapa v12 (2026-11-12):** el mapa se caía 5 veces en
10 min (huecos de 10-30 s) porque la H caducaba a los 30 s sin
keyframes y en seguimiento el modelo de keypoints falla cuando la
cámara se acerca. Ahora la H vale 60 s (aguanta esos huecos: la cámara
no se mueve tanto en 60 s) y `to_pitch` acepta puntos a unos metros
fuera del campo (el balón sale con frecuencia; antes se descartaban).
Medido en las muestras v10: los 5 huecos son de 10-30 s, todos < 60 s,
así que el mapa ya no se cae; la cobertura del balón en campo sube del
55% (limitada además por la detección del balón, 75%). Tests:
test_pitch 18/18 (caducidad 60 s, to_pitch x=-25 m). La verificación
completa (repasar los 10 min) queda para el remix, que usa la v12.

**Pases v12 (2026-11-12):** solo se detectaban 14 pases cuando a ojo
son 20+ por equipo. Causa: se exigía que el portador hubiera tenido el
balón 0,25 s seguidos antes del pase, y un pase al primer toque no
tiene control estable (el portador apenas toca el balón). Ahora no se
exige control estable para iniciar el vuelo, y la distancia mínima baja
de 3,0 a 2,0 alturas (pases cortos). Medido en las muestras v10: 42
pases (21 local, 21 visitante; 24 completados), antes 14 (5/9). Los
eventos no cambian. Tests: 38 checks, resto igual.

**Velocidad del balón v12 (2026-11-12):** la velocidad máxima salía
siempre 126 km/h (el tope) porque la homografía salta: el balón está
casi parado en la escena pero su posición en campo salta 60 m de golpe
(caso real t=125,1: balón a 0,1 alturas/s en escena, la H decía
126 km/h). Ahora la velocidad en campo se capa por la velocidad en
escena (2,5 m por altura de jugador: un jugador mide ~1,75 m y la
perspectiva estira la escala como mucho 2,5x) y la máxima del informe
es el percentil 99, no el máximo puro (que es casi siempre el tope).
Medido en las muestras v10: media 24,7 -> 15,3 km/h, máxima 126,0 ->
105,7 km/h, muestras en el tope 65 -> 10. Tests: test_pitch 18/18
(2 nuevos), resto igual.

**Posesión v12 (2026-11-12):** la posesión ya no se le regala al que
defiende. Causa raíz: cuando un equipo ataca, los defensas se agrupan
junto al balón y el "jugador más cercano" es un defensa el 76% del
tiempo en el tercio rival, así que la posesión se invertía (47/52
cuando a ojo es 55/45) y además parpadeaba 266 veces en 10 min. Ahora
hay dos posesiones: la DURA (m["possession"], cambia en cuanto hay un
portador cerca, la usa el clasificador de eventos para detectar duelos
y transiciones) y la SUAVE (m["possession_suave"], solo para las
estadísticas): cambia solo si el mismo equipo es el más cercano 1 s
seguido, y 3 s si es el defensor en su propio tercio. Medido en las
muestras v10: suave 52/47 (antes 47/52), 39 cambios (antes 266); los
eventos no cambian (6 duelos, 1 gol, 0 tiros). Tests: test_pases 38
checks (3 nuevos), resto igual.

**Tiros falsos v12 (2026-11-12):** los 4 "tiros" de la v10 eran falsos
(0 tiros reales en el análisis manual) y los 4 salían por la regla 3c
("Tiro a puerta": balón rápido junto al portero). Esa regla no miraba
qué pasa después con el balón. Ahora exige, además de estar junto al
portero y que ataque el equipo con posesión: que el balón se pierda
tras el candidato, acabe cerca de la línea de gol (parada o fuera) o
se PARE junto al portero (parada: el portero la caza); y que el
portero rival esté en su portería (a menos de 35 m de una línea de
gol), porque si el "portero" está en el centro del campo el balón
junto a él no es un tiro a puerta (falso tiro 07:23). Verificado en
las muestras v10: 0 tiros (antes 4), gol real 06:13 sin cambios, los
4 falsos ahora son recuperación/duelo/pase clave/falta. Tests:
test_pitch 16/16 (3 nuevos para la 3c), test_porteros 35/35,
test_faltas_goles 17/17, test_pases 35 checks.

**Posesión v10 (2026-11-12):** la clasificación de color ahora manda
el MATIZ. Causa raíz de la posesión invertida (42/58 cuando a ojo es
55/45): el azul en sombra (S~99, V~65) quedaba a distancia ~82 del ref
azul y salía "desconocido" (23,3% desconocido, la mayoría azules). La
distancia HSV pesaba S 0,5 / V 0,3 y la sombra dominaba al matiz. Ahora
S 0,25 / V 0,15, con reglas para negro (V<50 y S<80 → ref más oscuro) y
blanco/gris (S<40 → desconocido), y un ref oscuro solo vale si la V de
entrada también es baja (un rojo en sombra no es el portero de negro).
Medido en fotogramas reales: desconocido 23,3% → 9,2%, local 25,8% →
39,0%, visitante 37,6% → 35,9% (ratio 52/48, cerca del 55/45 manual).
Tests: test_porteros 35/35, test_pitch 13/13, test_faltas_goles 15/15,
test_pases 35 checks.

**Radar v9 (2026-11-12, a petición del usuario):** se ven TODOS los
jugadores traqueados, con el color de su camiseta tal como se ve en el
vídeo. Cada persona guarda su color real de torso (`color_rgb`, del
`_color_torso`): los "desconocido" ya no salen en gris, salen con su
color azul/rojo/verde de verdad. Además, el vídeo del radar ya no se
queda vacío cuando la homografía caduca un momento: cada fotograma usa
la muestra CON mapa más cercana (aguanta la última posición válida).
OJO: sin el modelo `pitch_keypoints.pt` junto a `video_track.py` no
hay mapa y ningún jugador tiene posición: el radar sale vacío. Es el
archivo de 85 MB del enlace de Drive. Test en test_pitch.py (13/13).

**Radar v7 (2026-11-12, a petición del usuario):** más pequeño
(480x300, antes 960x600), semitransparente (opacidad 0,70 en el
overlay de FFmpeg) y con los desconocido en gris. Test en
test_pitch.py (12/12).

**Gol v8 (2026-11-12):** la reaparición de la regla 3b debe quedarse
**5 s en el centro** (en los 10 s siguientes): en un saque de centro
de verdad el balón se queda ahí ~15 s mientras se colocan; en el
falso gol de 06:57 solo estaba ~3 s y luego el mapa saltaba fuera del
radio. Además, si el balón salta más de 8 m en el mapa en 3 s estando
parado en el plano, el mapa no es fiable y el centro no se fía. Con
esto, el falso gol de 06:57 ya no es gol (sale como falta) y el gol
real del Cirbonero sigue detectándose (06:13 y 06:16). Test en
test_faltas_goles.py (15/15).

**Gol v7 (2026-11-12):** el radio del centro para el saque de centro
/ colocación con calidad baja pasa de 20 m a **25 m**: en la pasada
completa el balón de la colocación sale a 22 m del centro (la x
comprimida lo desplaza) y con 20 m no se detectaba el gol. Con 25 m,
el gol del Cirbonero se detecta bien (06:13 y 06:16). Queda un gol
falso a 06:57 (Milagres) por la regla 3b: la homografía es inestable
ahí (el balón salta 16 m en 2 s) y la reaparición cae cerca del
centro. Test en test_faltas_goles.py (14/14).

**Mejoras v5 (2026-11-12, los 6 puntos de la demo):**
- **Posesión**: si el jugador más cercano al balón es "desconocido",
  la posesión NO cambia (antes se le daba al conocido más cercano,
  a menudo un rival, y la dejaba invertida). Test en
  test_pases_estadisticas.py (35 checks).
- **Gol**: el radio del centro para el saque de centro / colocación
  es 20 m (no 12 m) cuando la homografía es de calidad baja, porque
  la x comprimida desplaza el centro. Test en test_faltas_goles.py
  (14/14).
- **Radar x**: corrección del eje x anclada en el punto medio entre
  los dos porteros, factor capado a 2,5 (antes no se corregía nada
  con separación <40 m y los puntos salían apelotonados en el centro).
  Test en test_pitch.py (11/11).
- **Mapa en el PC del usuario**: `analyze_video_full` devuelve
  `mapa_pedido` / `mapa_activo`; la pantalla Vídeo muestra un aviso
  bien visible si se pidió el mapa y no se activó (modelo
  pitch_keypoints.pt ausente y descarga fallida), con el enlace para
  descargarlo.
- **Balón**: interpolación de huecos ampliada de 1 s a 2 s (el balón
  se ve solo en ~60% de las muestras; los huecos de 1-2 s cortaban
  la trayectoria). Segura por el filtro de saltos (velocidad implícita
  <= 25 alturas/s). Test en test_pases_estadisticas.py.
- **Tiros falsos (v4)**: revalidado en los 10 min con las reglas
  nuevas (ver test10_campo/resultado.json).

**SigLIP + UMAP + KMeans para asignar equipos: PROBADO Y DESCARTADO**
(2026-11-12, del notebook Roboflow football-ai). El `TeamClassifier`
(`google/siglip-base-patch16-224`) separa los recortes por **tamaño /
distancia del jugador, no por el equipo**: clusters [46, 14] (inestable
sin semilla: [24, 36] en otra ejecución), acuerdo con el color solo
54% (azar), y el cluster mayoritario mezcla azules y rojos. Los 18
"desconocido" por color solo se rescatan 5. SigLIP es un modelo general
de visión-lenguaje, no distingue camisetas; el color (mediana de la
mitad más saturada) sigue siendo mejor en este partido. El modelo
está descargado en `~/workspace/match_view_analysis/siglip-base-patch16-224/`
(775 MB) por si se quiere reintentar, pero **no se integra**.

**Rescate por mejor vista** (en su lugar): cada track guarda la
clasificación de su fotograma más cercano (bbox más alto). Si la
pluralidad de votos no llega, se usa esa vista. Así un jugador que
sale "desconocido" lejos pero se ve claro de cerca queda asignado.
`_equipo_por_track(votos, mejor_vista)`; test en test_porteros.py
(35/35).

**Límites**: con cámara que sigue al balón (VEO) la homografía cambia en
cada fotograma; si en un momento no se ven líneas, la muestra queda sin
campo (en el partido de prueba, el 96% de las muestras tiene campo, pero
casi todo en calidad baja). Con keypoints solo del centro la x sale
comprimida y los porteros caen cerca uno del otro: la corrección por
porteros solo se aplica si están bien separados (≥40 m en el mapa) y con
un factor suave (1,0-2,0); si no, no se corrige y el radar muestra la
jugada comprimida pero con los puntos dentro del campo (los puntos se
recortan a los límites del campo para que nunca desaparezcan). Los puntos
que la homografía manda fuera de un margen sensato (x en [-20, 125],
y en [-20, 88]) se descartan. El balón suele perderse justo en el momento
del gol (queda tapado por jugadores), así que **el gol se detecta por el
saque de centro** en seguimiento: por pérdida + reaparición en el centro,
o por colocación larga en el centro con un tiro previo; la regla de cruce
de línea solo vale cuando el balón es visible. La velocidad máxima del
balón es el mayor valor sin capar (el tope de 126 km/h es un salto de la
homografía, no un tiro real).

En la pantalla de Vídeo hay un checkbox **"🗺️ Mapa del campo
(homografía)"** (activado por defecto). Para desactivarlo por terminal:
`--sin-mapa`; para usar otro archivo de modelo: `--pitch-model RUTA`.

## Estadísticas e informe del partido (analista profesional)

Cada análisis, además de los cortes, calcula estadísticas del tracking y
escribe un informe junto a los clips, en la carpeta `<video>_clips`:

- **`<video>_informe.md`**: resumen con tablas (posesión por tramos,
  eventos por equipo, pases, tiros, territorio y balón) y lista de cortes.
- **`<video>_eventos.csv`**: una fila por corte (minuto, evento, equipo,
  confianza, resumen, zona, clip).
- **`<video>_estadisticas.csv`**: una fila por métrica, con la columna de
  cada equipo.

**Pases por trayectoria**: un pase es un cambio de portador —el balón sale
de un jugador y, tras un vuelo sin portador (0,2-4 s, balón rápido), llega
a un compañero; si llega a un rival es un pase fallido. Con eso se calculan
pases intentados/completados, precisión, distancia media y **secuencias**
(cadenas de pases seguidos del mismo equipo, con menos de 8 s entre ellos:
número de secuencias, la más larga y la media). El clasificador usa estos
pases para etiquetar pases que no son rápidos (regla 11b).

**Territorio**: sin mapa es un proxy (tercio de ataque según la posición
de cada portero). **Con mapa**, la posición del balón en metros da los
tercios reales del campo: % de muestras con el balón en el tercio izquierdo
(0-35 m), central (35-70 m) y derecho (70-105 m).

**Balón**: distancia recorrida (en alturas de jugador) y, **con mapa**,
en metros y velocidad en km/h, con el % de muestras con balón detectado
(cobertura del tracking) y el % con campo calculado.

En la pantalla de Vídeo de `match.py` se muestra la tabla de estadísticas
y hay botones para descargar el informe (.md) y los dos CSV.

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
posesión por tramos correctas, 33/33 checks del clasificador por tracking
con muestras sintéticas (gol, tiro, córner, falta, centro, contraataque,
duelo, pase clave, entrada al área, gol/tiro con portero, pase atrás no es
tiro, colores de porteros/árbitro, árbitro excluido de posesión y agrupación)
y 32/32 checks de pases por trayectoria, secuencias, estadísticas, informe
y regla 11b del clasificador (pase completado/fallido, vuelo largo no es
pase, secuencias por equipo, posesión 50/50, eventos y tiros contados,
precisión de pase, informe md/csv, pase lento etiquetado como pase 0,70 y
pase largo como pase clave), 10/10 checks de falta/gol con las reglas v2
(falta sin árbitro, pase atrás al portero no es tiro, gol por saque de
centro con tiempos totales) y 10/10 checks del mapa del campo (gol por cruce
de línea, tiro hacia la portería rival, despeje hacia la propia no es tiro,
entrada al área por metros, centro desde banda por metros, pase atrás con
mapa no es pase clave, gol por colocación para el saque, balón rápido
perdido en el centro no es tiro, homografía con 4 esquinas y calidad baja
con solo el centro, radar con balón en el centro).
En vídeo real (10 min del Cirbonero B - Milagres, ventana 05:24-07:54) el
gol del Cirbonero ~06:07 se detecta como gol del Cirbonero a las 06:16
(por colocación para el saque); la validación completa de 10 min con los
cortes y el radar está en curso.

**Colores de camiseta**: el color del torso es la mediana de la mitad más
saturada de los píxeles. Se probó KMeans (k=2, como en el proyecto
football_analysis del vídeo de referencia) para separar la camiseta del
fondo, con dos reglas de elección (grupo dominante / más saturado, y
grupo del centro del torso). En datos reales de este partido salió peor
que la mediana: el césped verde y saturado domina el torso cuando el
jugador está lejos y KMeans lo elegía como camiseta (4 falsos porteros
verdes en 07:54, 14 desconocidos en el gol frente a 12 con la mediana).
El KMeans del vídeo funciona con cámara fija y jugadores grandes; con
VEO, jugadores lejos/pequeños y contraluz de atardecer, la mediana es
más robusta. El color se compara con los 5 colores que pone el usuario
en Alineaciones (local, visitante, portero local, portero visitante,
árbitro) por distancia HSV, con margen mínimo entre la mejor y la segunda
referencia.

## Limitaciones conocidas

- **Cámara que sigue al balón**: la posición en imagen no es posición en el
  campo. La zona Z1-Z9 no se puede saber con seguridad en modo seguimiento;
  por eso queda "Sin especificar" (mejor no inventarla). En modo fija amplia
  es solo una aproximación. **El mapa del campo (homografía) mitiga esto en
  seguimiento**: con el campo calculado se sabe si el balón va a la portería
  rival (tiro) o a la propia (despeje), si cruza la línea (gol) y si entra
  al área; pero si solo se ve el centro del campo la coordenada x queda
  comprimida y la corrección por porteros es la que la estira. En
  seguimiento, el gol se detecta por la reanudación en el centro (tiro
  rápido + balón perdido ≥4 s + saque de centro), no solo por el portero;
  sin esa reanudación el gol sale como tiro. El córner sigue necesitando
  fija amplia.
- **Porteros**: solo se ven cuando la jugada está cerca de su área (ahí es
  donde importan). Si su camiseta se parece a la de su equipo o a la del
  árbitro, el voto por color puede confundirlos: el color del portero tiene
  que ser distinto. Un pase atrás rápido al propio portero no se etiqueta
  como tiro (se exige que el balón lo mueva el equipo atacante).
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
