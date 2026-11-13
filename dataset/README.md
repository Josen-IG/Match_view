# Dataset propio — Cirbonero B vs Milagres (VEO)

210 fotogramas del partido (1 cada 30 s, desde el saque hasta el minuto 105),
auto-etiquetados y listos para corregir a mano.

- `images/train/` (168) y `images/val/` (42): fotogramas JPG 1920x1080.
- `labels/train/` y `labels/val/`: etiquetas YOLO (una línea por caja:
  `clase x_centro y_centro ancho alto`, todo normalizado 0-1).
- `data.yaml`: configuración del dataset para entrenar con Ultralytics YOLO.

## Clases
- `0` football (balón)
- `1` player (jugador)
- `2` referee (árbitro) — pendiente: hay que etiquetarlo a mano, el
  auto-etiquetado no lo detecta.

## Cómo se etiquetó
- Jugadores y balones: modelo GemsLabs `custom_model.pt` (fútbol) +
  YOLO11s para reforzar el balón.
- Resultado: 3.371 jugadores (16,1 por fotograma), balón en 94/210
  fotogramas (45%). El balón es la clase más difícil: revisa los
  fotogramas sin balón, en muchos sí se ve pero ningún modelo lo detectó.

## Corregir
Abre las imágenes con las etiquetas en Roboflow, LabelImg o CVAT:
añade los balones que falten, corrige cajas mal puestas y etiqueta
al árbitro (amarillo fosforito) como clase 2.
