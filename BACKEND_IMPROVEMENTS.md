# Mejoras del motor de análisis de Match_view

Esta fase modifica el **backend**, no la interfaz. Ninguna métrica de
detección, identificación ni reconocimiento de eventos se considera
validada sin comparar con anotaciones humanas de partidos no vistos.

## 1. Detección YOLO y reentrenamiento

- `modelo_propio.pt` sigue siendo el peso activo. **No se han entrenado
  ni desplegado nuevos pesos**.
- `training_pipeline.py --audit` verifica que existan imágenes, etiquetas
  compatibles y divisiones de entrenamiento/validación.
- `hard_example_mining.py` crea una cola de fotogramas difíciles:
  desapariciones del balón, oclusiones de IDs y mala calibración. La cola
  **no asigna etiquetas automáticamente**.
- `training_pipeline.py --train` puede ejecutar un nuevo ajuste fino
  **únicamente cuando las imágenes originales estén disponibles**.
- Reservar partidos completos que no intervengan en entrenamiento ni
  validación. Evitar dividir fotogramas adyacentes del mismo partido entre
  conjuntos usados para afirmar generalización.

Comandos opcionales desde la raíz del repositorio:

```powershell
py training_pipeline.py --audit
py hard_example_mining.py --tracking "partido_tracking.json" --video "C:\Videos\partido.mp4" --output hard_examples
# Después de corregir las etiquetas y disponer de images/train y images/val:
py training_pipeline.py --train --dataset dataset --weights modelo_propio.pt --epochs 60 --device 0
```

El último comando requiere una instalación de Ultralytics y una GPU
compatible si se usa `--device 0`. En CPU se puede indicar
`--device cpu`, aunque resulta mucho más lento. Los pesos resultantes
son **candidatos**, no reemplazan el modelo activo automáticamente.

## 2. Asociación de jugadores / reidentificación

`video_track.py` extrae histogramas HSV del torso como descriptor de
apariencia. `player_tracking.py` utiliza esta señal con la trayectoria,
altura, color y equipo; actualiza lentamente la apariencia para limitar
contaminación entre jugadores. Es **ReID clásico por apariencia**, no un
modelo neuronal de reidentificación entrenado ni reconocimiento de dorsales.
Se mantiene un ID oculto por periodos cortos, como hipótesis separada de
la detección. No se inventa un portero fuera de cámara.

## 3. Balón y predicción

La tubería conserva diferencias explícitas entre
`yolo` (medición aceptada), `interpolado` (estimación retrospectiva)
y `kalman` (predicción temporal). Las predicciones no deben
considerarse detecciones nuevas ni alimentar métricas de precisión del
detector. Se guarda edad desde última medida e incertidumbre del filtro.
Después de un hueco superior a 1 s la posición deja de considerarse
conocida sin nueva observación.

## 4. Campo y homografía

`radar_quality.py` exige calibraciones recientes y consistentes.
`reject_collective_pitch_jumps` anula coordenadas métricas cuando varios
jugadores con IDs coincidentes sufren un salto físicamente imposible
al mismo tiempo; la detección YOLO en píxeles se conserva. Este control
no sustituye a una evaluación con marcas del campo anotadas.

## 5. Eventos y jugadas

`temporal_event_evidence.py` contiene **hipótesis experimentales**.
Exige jugador cercano antes de un desplazamiento de balón, varias
observaciones YOLO consecutivas y trayectoria hacia una portería.
Un pase, centro o despeje puede seguir dando un falso positivo.

Por eso **NO se activa la clasificación automática de tiros en producción**
ni se añaden candidatos experimentales a la línea de tiempo. Solo se
reactivará cuando un clasificador temporal evaluado supere umbrales de
precisión/recall medidos en un conjunto etiquetado.

## 6. Evaluación de precisión

`model_evaluation.py` compara un JSON de anotaciones humanas con el
tracking exportado. El formato de referencia es:

```json
{
  "frames": [
    {
      "t": 900.0,
      "players": [{"id": "local_7", "x": 43.5, "y": 51.0}],
      "ball": [48.0, 54.0],
      "pitch": {"local_7": [45.0, 26.0]}
    }
  ],
  "events": [{"type": "shot", "t": 912.4}]
}
```

Las posiciones de la imagen usan 0–100 y las de campo, metros. Se generan:
TP/FP/FN, precisión/recall, cambios de ID observados en fotogramas
etiquetados, error de posición del balón y error de posición métrica
cuando existe referencia. **No son HOTA/IDF1 oficiales**; son métricas
descriptivas de muestras etiquetadas.

```powershell
py model_evaluation.py --ground-truth "anotaciones.json" --tracking "partido_tracking.json" --output "evaluacion.json"
py -m unittest discover -s tests -v
```

Para afirmar que el sistema ha mejorado, ejecutar versión anterior y
candidata sobre **los mismos partidos inéditos y etiquetas humanas**,
comparar además latencia, mAP de YOLO e IDF1/HOTA con una implementación
de evaluación MOT estándar. Sin este conjunto de referencia la mejora
es **estructural, no una mejora porcentual demostrada**.

La pestaña Vídeo y `match.py` permanecen sin cambios.
