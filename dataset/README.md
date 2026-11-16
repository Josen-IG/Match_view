# Dataset propio YOLO — genérico (cualquier partido)

4874 fotogramas, 3899 train / 975 val (split 80/20 estratificado por partido,
sin fuga: cada fotograma va a un solo split). Auto-etiquetados y corregidos
(ronda 2: fuera logos/gráficos estáticos y balones duplicados; 1 balón por
fotograma como máximo).

- `images/train/` (3899) y `images/val/` (975): fotogramas JPG (las imágenes NO están en
  el repo; están en el zip `dataset_yolo_propio_images.zip` en Drive, carpeta "Git Hub").
- `labels/train/` y `labels/val/`: etiquetas YOLO (una línea por caja:
  `clase x_centro y_centro ancho alto`, todo normalizado 0-1).
- `data.yaml`: configuración del dataset para entrenar con Ultralytics YOLO.
- `extract_frames_match.py`, `autolabel_match.py`, `merge_dataset_multi.py`:
  pipeline para añadir un partido nuevo (extraer fotogramas, auto-etiquetar,
  fusionar al dataset).

## Composición (por qué generaliza)

| Fuente | Imágenes | Train | Val | Jugadores | Balones | Frames con balón |
|---|---|---|---|---|---|---|
| Cirbonero B vs Milagres, todo el partido (1 cada 30 s) | 210 | 168 | 42 | 3.371 | 103 | 94 |
| Cirbonero B vs Milagres, 2ª mitad, luz oscura (1 cada 45 s) | 58 | 46 | 12 | 903 | 21 | 21 |
| Cirbonero B vs Milagres, ronda 2 del balón (huecos >10 s + rejilla 1/15 s) | 496 | 397 | 99 | 7.827 | 314 | 314 |
| Sevilla Atlético vs Betis Deportivo (1ª Fed) | 251 | 201 | 50 | 2.714 | 66 | 66 |
| CF Talavera vs Zamora CF (1ª Fed) | 232 | 186 | 46 | 2.125 | 105 | 105 |
| Antequera CF vs Real Madrid Castilla (1ª Fed, noche) | 198 | 158 | 40 | 2.402 | 68 | 68 |
| Real Madrid Castilla vs Real Avilés (1ª Fed) | 204 | 163 | 41 | 2.124 | 106 | 106 |
| RSD Alcalá vs CF Fuenlabrada (2ª Fed) | 260 | 208 | 52 | 2.298 | 105 | 104 |
| RSD Alcalá vs Real Madrid C (2ª Fed) | 276 | 221 | 55 | 2.929 | 138 | 138 |
| CD Badajoz vs CD Tenerife B (2ª Fed) | 245 | 196 | 49 | 3.600 | 156 | 156 |
| Recreativo de Huelva vs Atlético Antoniano (2ª Fed) | 210 | 168 | 42 | 3.511 | 91 | 91 |
| Betis C vs Conil CF (3ª Fed) | 240 | 192 | 48 | 2.671 | 143 | 143 |
| Betis C vs Sevilla C (3ª Fed) | 199 | 159 | 40 | 2.628 | 106 | 106 |
| Jove Español vs Villarreal C (3ª Fed, noche) | 261 | 209 | 52 | 3.307 | 52 | 52 |
| CD Galapagar vs Atlético de Madrid C (3ª Fed) | 293 | 234 | 59 | 2.452 | 104 | 104 |
| Cornellà vs Mollerussa (3ª Fed) | 282 | 226 | 56 | 2.600 | 111 | 111 |
| San José Obrero vs Atlético Tomelloso (Preferente) | 246 | 197 | 49 | 3.539 | 60 | 60 |
| Hellín CF vs Atlético Tomelloso (Preferente) | 283 | 226 | 57 | 4.332 | 108 | 108 |
| U.D. Casetas vs Ontiñena (Preferente) | 188 | 150 | 38 | 2.480 | 54 | 54 |
| CF Pozuelo vs AD Torrejón (3ª Fed, césped artificial) | 242 | 194 | 48 | 3.068 | 67 | 67 |
| **Total** | **4.874** | **3.899** | **975** | **60.881** | **2.078** | **2.068 (42%)** |

18 partidos distintos (1ª/2ª/3ª Federación y Preferente) + 496 fotogramas
extra del Cirbonero-Milagres, varios estadios, cámaras de TV y VEO, día,
atardecer y noche con focos, y camisetas de todos los colores. Eso, más la
augmentación fuerte del entrenamiento (HSV, escala, mosaic, mixup), es lo
que le da generalización.

No entraron: Palencia CF vs Atlético Bembibre (su subida a Drive falló
repetidamente) ni Selaya FC vs CD Cayón (sustituto aprobado y luego descartado
por el usuario: su subida a Drive también se atasca y con 16 vídeos ya había
de sobra). Sus fotogramas y etiquetas quedan en la VM, fuera del dataset.

## Ronda 2 del balón (2026-11-16)
El balón era el cuello de botella (mAP50 0,752, recall 0,627: se escapaba
1 de cada 3). Cambios:
- **496 fotogramas nuevos del Cirbonero-Milagres** en los momentos donde el
  modelo perdía el balón (huecos >10 s del tracking completo) + rejilla
  1/15 s. Etiquetados con 3 votantes (GemsLabs + YOLO11s + modelo_propio) y
  repaso del balón a imgsz 1280 en los fotogramas sin balón (el balón mide
  3-5 px a 640 y 6-10 px a 1280 en el plano abierto de VEO).
- **Limpieza de etiquetas falsas de balón** (ver `ronda2/limpiar_balon.py` en
  la VM): fuera 153 logos/gráficos estáticos (grupos apretados, spread
  <= 0,01), 507 balones duplicados (queda el más cercano a un jugador) y 86
  cajas absurdas. OJO: la primera versión de la limpieza borró balones
  reales — la cámara VEO en seguimiento centra el balón, así que aparece
  cerca del centro en muchos fotogramas (spread 0,02-0,04); eso NO es un
  gráfico estático. Corregido exigiendo spread <= 0,01.
- **Oversampling del balón en el entreno:** `merge_dataset_multi.py
  --dup-ball 1` duplica cada fotograma de train con balón (train 3899 ->
  5539; val intacto para métricas honestas). Las copias (`*_r2dup1`) son un
  artefacto del entreno local y NO están en el repo.

## Clases
- `0` football (balón)
- `1` player (jugador)
- `2` referee (árbitro) — pendiente: hay que etiquetarlo a mano, el
  auto-etiquetado no lo detecta.

El modelo NO clasifica equipos por color: solo detecta balón y jugador.
Los equipos se distinguen después por color de camiseta en `video_track.py`,
así que el modelo vale para cualquier partido.

## Cómo se etiquetó
- Jugadores y balones: modelo GemsLabs `custom_model.pt` (fútbol) +
  YOLO11s para reforzar el balón (imgsz 640, conf 0.25).
- El balón es la clase más difícil: revisa los fotogramas sin balón, en muchos
  sí se ve pero ningún modelo lo detectó.

## Para hacerlo aún más genérico
- **Dataset público GemsLabs-football-26 (Roboflow, CC BY 4.0):** 276 imágenes
  de varios partidos (distintos estadios, luces y cámaras). El notebook
  `train_colab.ipynb` (celda 4, opcional) lo descarga y lo fusiona si le
  pegas tu API key gratuita de Roboflow. No se pudo descargar desde aquí
  (Roboflow exige API key); en Colab se hace en esa celda.
- **Modelos GemsLabs ya entrenados (en Drive, carpeta "trained-model- GemsLabs"):**
  `train dataset-nano.zip` (YOLO26n, mAP50 0.58) y `train dataset- small.zip`
  (YOLO26s). Son salidas de entrenamiento sobre ese dataset, no el dataset
  en sí. El `best.pt` de YOLO26s sirve como modelo alternativo o como
  "profesor" para auto-etiquetar más fotogramas.
- **Más partidos propios:** extrae 1 fotograma cada 30 s de otro partido
  (`extract_frames_match.py`), auto-etiquétalo igual (`autolabel_match.py`),
  añádelo a `merge_dataset_multi.py` y vuelve a entrenar.

## Corregir
Abre las imágenes con las etiquetas en Roboflow, LabelImg o CVAT:
añade los balones que falten, corrige cajas mal puestas y etiqueta
al árbitro (amarillo fosforito) como clase 2.
