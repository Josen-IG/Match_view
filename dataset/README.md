# Dataset propio YOLO — genérico (cualquier partido)

4136 fotogramas, 3308 train / 828 val (split 80/20 estratificado por partido,
sin fuga: cada fotograma va a un solo split). Auto-etiquetados y listos para corregir a mano.

- `images/train/` (3308) y `images/val/` (828): fotogramas JPG (las imágenes NO están en
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
| Cirbonero B vs Milagres, 2ª mitad, luz oscura (1 cada 45 s) | 58 | 46 | 12 | 903 | 23 | 21 |
| Sevilla Atlético vs Betis Deportivo (1ª Fed) | 251 | 201 | 50 | 2.714 | 74 | 66 |
| CF Talavera vs Zamora CF (1ª Fed) | 232 | 186 | 46 | 2.125 | 129 | 115 |
| Antequera CF vs Real Madrid Castilla (1ª Fed, noche) | 198 | 158 | 40 | 2.402 | 87 | 68 |
| Real Madrid Castilla vs Real Avilés (1ª Fed) | 204 | 163 | 41 | 2.124 | 145 | 107 |
| RSD Alcalá vs CF Fuenlabrada (2ª Fed) | 260 | 208 | 52 | 2.298 | 159 | 128 |
| RSD Alcalá vs Real Madrid C (2ª Fed) | 276 | 221 | 55 | 2.929 | 190 | 159 |
| CD Badajoz vs CD Tenerife B (2ª Fed) | 245 | 196 | 49 | 3.600 | 253 | 159 |
| Recreativo de Huelva vs Atlético Antoniano (2ª Fed) | 210 | 168 | 42 | 3.511 | 111 | 93 |
| Betis C vs Conil CF (3ª Fed) | 240 | 192 | 48 | 2.671 | 208 | 143 |
| Betis C vs Sevilla C (3ª Fed) | 199 | 159 | 40 | 2.628 | 143 | 107 |
| Jove Español vs Villarreal C (3ª Fed, noche) | 261 | 209 | 52 | 3.307 | 62 | 53 |
| CD Galapagar vs Atlético de Madrid C (3ª Fed) | 293 | 234 | 59 | 2.452 | 145 | 119 |
| Cornellà vs Mollerussa (3ª Fed) | 282 | 226 | 56 | 2.600 | 136 | 111 |
| San José Obrero vs Atlético Tomelloso (Preferente) | 246 | 197 | 49 | 3.539 | 66 | 60 |
| Hellín CF vs Atlético Tomelloso (Preferente) | 283 | 226 | 57 | 4.332 | 168 | 109 |
| U.D. Casetas vs Ontiñena (Preferente) | 188 | 150 | 38 | 2.480 | 58 | 54 |
| **Total** | **4.136** | **3.308** | **828** | **49.986** | **2.260** | **1.766 (43%)** |

18 partidos distintos (1ª/2ª/3ª Federación y Preferente), varios estadios,
cámaras de TV y VEO, día, atardecer y noche con focos, y camisetas de todos
los colores. Eso, más la augmentación fuerte del entrenamiento (HSV, escala,
mosaic, mixup), es lo que le da generalización.

No entraron: Palencia CF vs Atlético Bembibre (su subida a Drive falló
repetidamente) ni Selaya FC vs CD Cayón (sustituto aprobado y luego descartado
por el usuario: su subida a Drive también se atasca y con 16 vídeos ya había
de sobra). Sus fotogramas y etiquetas quedan en la VM, fuera del dataset.

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
