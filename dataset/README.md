# Dataset propio YOLO — genérico (cualquier partido)

268 fotogramas, 214 train / 54 val (split 80/20 estratificado por fuente,
para que val tenga de las dos luces). Auto-etiquetados y listos para corregir a mano.

- `images/train/` (214) y `images/val/` (54): fotogramas JPG 1920x1080.
- `labels/train/` y `labels/val/`: etiquetas YOLO (una línea por caja:
  `clase x_centro y_centro ancho alto`, todo normalizado 0-1).
- `data.yaml`: configuración del dataset para entrenar con Ultralytics YOLO.

## Composición (por qué generaliza)

| Fuente | Imágenes | Train | Val | Jugadores | Balones | Frames con balón |
|---|---|---|---|---|---|---|
| Cirbonero B vs Milagres, todo el partido (1 cada 30 s) | 210 | 168 | 42 | 3.371 | 103 | 94 |
| Cirbonero B vs Milagres, 2ª mitad, luz oscura (1 cada 45 s) | 58 | 46 | 12 | 903 | 23 | 21 |
| **Total** | **268** | **214** | **54** | **4.274** | **126** | **115 (43%)** |

La 2ª mitad es de noche (tras el atardecer), así que el modelo ve el mismo
estadio con dos luces muy distintas. Eso, más la augmentación fuerte del
entrenamiento (HSV, escala, mosaic, mixup), es lo que le da generalización.

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
- **Más partidos propios:** extrae 1 fotograma cada 30-45 s de otro partido,
  auto-etiquétalo igual (GemsLabs + YOLO11s) y vuelve a entrenar.

## Corregir
Abre las imágenes con las etiquetas en Roboflow, LabelImg o CVAT:
añade los balones que falten, corrige cajas mal puestas y etiqueta
al árbitro (amarillo fosforito) como clase 2.
