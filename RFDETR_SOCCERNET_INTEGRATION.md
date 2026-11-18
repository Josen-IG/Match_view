# RF-DETR SoccerNet — integración 2

## Estado de los componentes

- [x] YOLOv9 Soccer Ball: integrado como respaldo del balón.
- [x] RF-DETR SoccerNet: integrado como segundo respaldo y revisión de personas.
- [ ] YOLO Football Pitch Detection: pendiente de integrar este checkpoint.
- [ ] OpenSportsLab — Ball Action Spotting: pendiente.
- [ ] SoccerNet Game State Reconstruction: pendiente.

Las marcas indican integración del código, no precisión validada en partidos.
Los pesos del detector principal no cambian. No se realiza entrenamiento.

## Uso local

```powershell
git pull origin main
python -m pip install -r requirements.txt -r requirements-rfdetr.txt
python -m streamlit run match.py
```

Activa «Revisar detecciones con RF-DETR SoccerNet». Puede usarse junto con
YOLOv9 o solo con el modelo base. La primera descarga pública es de unos
1,6 GB; no usa tokens guardados. Para evitar la descarga, coloca el checkpoint
`checkpoint_best_regular.pth` junto a `match.py` o en `weights/`.

Ejemplo CLI: sustituye la ruta por un vídeo que exista en tu equipo.

```powershell
python video_track.py "C:\Videos\mi_video.mp4" --ball-model hf --rfdetr-model hf --max-minutes 1 --sample-fps 5
```

La CLI también acepta `--rfdetr-model RUTA.pth`, `--rfdetr-conf 0.4` y
`--device 0` para la GPU. El formulario actual ejecuta el análisis en CPU.
La API `analyze_video_full` admite `rfdetr_model` y `rfdetr_conf`.

## Qué hace esta etapa

1. El detector principal se ejecuta en cada fotograma muestreado.
2. Si no devuelve candidatos de balón, se consulta YOLOv9 si está habilitado.
3. Si sigue sin haber candidatos, se consulta RF-DETR si está habilitado.
4. Los candidatos recuperados pasan por la selección y el procesamiento
   temporal que ya usa el proyecto. No se comparan puntuaciones entre modelos.

RF-DETR devuelve balón, jugador, árbitro y portero. En los fotogramas que
revisa, sus cajas de personas se emparejan con las del detector principal
mediante IoU >= 0,5, sin duplicarlas ni crear IDs. Se guardan como evidencia;
no sustituyen las asignaciones por color, ni se usan para excluir árbitros
o calcular equipos automáticamente en esta etapa.

El respaldo actúa cuando no hay candidatos; no corrige un balón incorrecto
que haya aceptado otro detector. Más detecciones no garantizan más precisión.
Todavía no genera pases, tiros o goles en «Jugadas IA».

## Procedencia y compatibilidad

- Checkpoint: `julianzu9612/RFDETR-Soccernet`.
- Revisión fija: `7e567611ea77efd3a6144b3a61eced5a8c8df8d7`.
- Archivo: `weights/checkpoint_best_regular.pth`.
- Licencia declarada por el repositorio: Apache-2.0.
- El título de la ficha dice Large, pero su `inference.py` usa RFDETRBase,
  reinicializa la cabeza a cuatro clases y carga el state dict. Seguimos ese
  cargador, con rfdetr 1.2.1 (versión contemporánea al checkpoint), sin descarga
  adicional de pesos base y con carga estricta. No cambiamos a un modelo
  genérico ni aceptamos pesos parciales si hay incompatibilidad.
- PyTorch >=2.6 para deserialización `weights_only=True`, permitiendo únicamente
  `argparse.Namespace` como global adicional del checkpoint heredado.
- Dependencias opcionales: `requirements-rfdetr.txt`. Si no se habilita,
  no se importa rfdetr ni se descargan sus pesos.

Fuentes consultadas:
https://huggingface.co/julianzu9612/RFDETR-Soccernet
https://huggingface.co/julianzu9612/RFDETR-Soccernet/blob/7e567611ea77efd3a6144b3a61eced5a8c8df8d7/inference.py
https://pypi.org/project/rfdetr/1.2.1/

## Datos exportados

- Muestra cruda: `ball_detector`, `rfdetr_reviewed` y `people[].rfdetr_review`.
- Tracking JSON: `bd` indica el detector del balón observado; `rr` indica
  que el fotograma fue revisado; `rv` contiene revisiones asociadas al track ID.
- `bf` conserva la separación actual entre observación y predicción temporal.
- Las revisiones no cuentan como jugadas confirmadas ni detecciones adicionales.

## Validación

13 pruebas nuevas con modelos simulados: carga estricta y cuatro clases,
error de incompatibilidad, carga local, descarga anónima, RGB y coordenadas,
cajas inválidas, asociación de personas, dispositivos y umbrales, desactivación,
prioridad de detectores, recuperación del balón, errores de inferencia y
propagación por las APIs públicas. También pasan 23 pruebas anteriores del
detector de balón, predicción temporal, calidad y flujo mínimo.

```powershell
python -m unittest discover -s tests -p test_rfdetr_detector.py
```

Pendiente: instalar y ejecutar la dependencia y checkpoint reales, comprobar
compatibilidad de sus pesos, RAM/VRAM y medir precisión y latencia en el mismo
fragmento con y sin RF-DETR. La red del entorno de implementación no permitió
descargar el checkpoint; las pruebas simuladas no validan esa compatibilidad.
