# Match_view: experimento del 17 de noviembre de 2026

Estado inicial: entrenamiento pendiente; no hay métricas nuevas ni despliegue.
Fuente de datos/código: commit 30061bd41b2385d0cb94f6c325ffb2cc0042dd40.

## Auditoría ejecutada en Colab T4

- ZIP de Drive: 580.967.795 bytes; 4.378 imágenes (3.502 train / 876 val).
- Etiquetas del repositorio: 4.874. Faltan las 496 imágenes cirbonero_r2
  (397 train / 99 val); se excluyen sus etiquetas de la copia del experimento.
- Todas las imágenes disponibles tienen etiquetas y son decodificables.
- Objetos disponibles: 53.054 jugadores y 1.764 balones.
- Duplicados exactos por hash de píxeles: 0.
- Pares de imágenes parecidas (dHash Hamming <=4): 101; requieren revisión,
  no se eliminan automáticamente por parecido.
- Cajas que exceden parcialmente la imagen: 186 en las imágenes disponibles.
  Se recortan sus bordes en la copia de trabajo; los originales se conservan.
  El registro guarda antes/después, hashes de imágenes/etiquetas y exclusiones.
- Los 18 partidos están presentes en ambos splits. La validación no es un
  holdout de partidos independientes.
- Las etiquetas son principalmente automáticas. El examen visual de una muestra
  no sustituye una revisión humana exhaustiva de falsos balones y omisiones.

## Experimento

`retrain_audited_colab.ipynb` ejecuta `retrain_experiment.py`. Compara el baseline
con el ajuste del YOLO11n actual y YOLO11s preentrenado. No altera Match_view.
Resolución 960, batch 8, semilla 42, AdamW, máximo 60 épocas, paciencia 12;
4.887 entradas de train tras repetir una vez las imágenes con balón.
La validación mantiene 876 imágenes, sin sobremuestreo.

Por modelo se guardan AP por clase, TP/FP/FN y precisión/recall a umbral fijo
(conf .25, matching IoU .5, NMS IoU .7), recall para objetos diminutos y latencia
CPU con muestras/hilos/resolución iguales. Se distingue del umbral de mejor F1.
La CPU de Colab no representa la CPU Windows del usuario.

Artefactos: carpeta nueva `Mi unidad/Match_view_retraining_20261117T180409Z`.
Incluye código ejecutado, auditoría, configuración, entorno, baseline y salidas
por candidato. Se sincronizan checkpoints al guardar cada época. Colab puede
interrumpir la sesión: no hay garantía de ejecución desatendida.

## Condición de integración

No desplegar por una mejora de mAP global sobre autoetiquetas. Antes de sustituir
pesos, exigir evaluación contra anotaciones humanas verificadas de partidos
inéditos; comparar recall y precisión del balón a un umbral fijo, FP, AP del balón
y jugadores y coste de CPU. Conservar el baseline y verificar regresiones y CI.
Las hipótesis de tiros/goles/córneres siguen desactivadas. La interfaz está intacta.

Pruebas locales: 111 tests unittest superados (incluyen 5 nuevos de cajas, IoU y
agrupación por partido). Los resultados de GitHub Actions deben comprobarse tras
el commit. No se presentan métricas históricas como evaluación de este experimento.
