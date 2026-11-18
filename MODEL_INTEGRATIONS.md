# Modelos de Match_view: integración y puesta en marcha

| Componente | Código conectado | Pesos reales y precisión en vídeo |
|---|---|---|
| YOLOv9 Soccer Ball | Sí, respaldo del balón | Pendiente de evaluación comparativa |
| RF-DETR SoccerNet | Sí, segundo respaldo y revisión de personas | Pendiente de carga y evaluación |
| YOLO Football Pitch Detection | Sí, topología independiente de 32 puntos | Pendiente de carga y evaluación |
| OpenSportsLab Ball Action Spotting | Sí, proceso independiente y panel Jugadas IA | Pendiente de instalar entorno y probar inferencia |
| SoccerNet Game State Reconstruction | Sí, conector al pipeline oficial independiente | Pendiente de instalar entorno y probar inferencia |

Las marcas de integración no significan que estos checkpoints se hayan
ejecutado o validado en este entorno. Las pruebas usan salidas simuladas y
calibración sintética. No se ha medido una mejora de precisión en tus vídeos.

## 1. Actualizar la aplicación

Detén Streamlit y ejecuta desde la carpeta Match_view:

```powershell
git pull origin main
python -m pip install -r requirements.txt
python -m streamlit run match.py
```

El formulario usa modo automático, sin casillas. El detector base conserva sus
pesos y se añaden componentes según disponibilidad. Los fallos opcionales se
registran y el análisis continúa con los restantes. No es una selección por
porcentajes de acierto: todavía faltan datos etiquetados de evaluación.

- YOLOv9 recupera el balón cuando no hay candidatos del detector base.
- RF-DETR entra si los anteriores siguen sin candidatos. Requiere instalar
  `requirements-rfdetr.txt` como se explica en RFDETR_SOCCERNET_INTEGRATION.md.
- El modo automático intenta cargar YOLO Football Pitch Detection, el modelo de
  32 puntos (unos 140 MB). También admite el archivo
  `yolo-football-pitch-detection.pt` junto a `match.py`. La topología de este
  checkpoint es distinta de la del anterior modelo Soccana de 29 puntos.

El campo se representa con dimensiones nominales de 105 × 68 m. Un terreno
con otras dimensiones introduce error de escala: las velocidades y distancias
son estimaciones, no mediciones GPS. El nuevo detector se ejecuta a 1280 px;
puede resultar bastante más lento en CPU. No modifica el detector de personas.

## 2. OpenSportsLab: entorno separado Python 3.12

El checkpoint de acciones ocupa unos 152 MB. El código que usamos admite
lectura OpenCV en CPU; no requiere instalar DALI. Se fija tanto la versión del
código como la revisión de los pesos. Para crear su entorno independiente:

```powershell
python -m pip install uv
python setup_model_envs.py actions
$env:MATCHVIEW_OSL_PYTHON = "$PWD\.model-envs\actions\Scripts\python.exe"
python -m streamlit run match.py
```

El modo automático detecta este entorno y usa OpenSportsLab si está preparado. El intérprete configurado y sus
dependencias se verifican antes de comenzar el tracking. Puedes usar otro
entorno apuntando MATCHVIEW_OSL_PYTHON a su ejecutable Python 3.12 o superior.

El proceso recibe solo el intervalo analizado (desde el saque inicial y hasta
el límite solicitado), convertido a 25 fps y 398 × 224. Las marcas de tiempo
se trasladan al vídeo original, se filtran por confianza y se aplica supresión
temporal por clase (2 s). Las doce clases se muestran como propuestas:
pase, conducción, cabeceo, pase alto, balón fuera, centro, saque de banda,
tiro, bloqueo, entrada exitosa, tiro libre y gol.

«Jugadas IA» permite ver, guardar un corte, confirmar o descartar. Las
propuestas no se convierten por sí solas en estadísticas oficiales ni
distinguen un tiro a puerta de uno fuera. La confirmación es una revisión del
usuario guardada en el navegador; todavía no alimenta el CSV de estadísticas.

## 3. SoccerNet Game State: entorno separado Python 3.9

Este componente es un framework con varios modelos, no un archivo único.
Su proyecto fija Python 3.9, PyTorch 1.13.1, TrackLab 1.3.24 y MMCV 2.0.1.
Sus dependencias son incompatibles con las del entorno moderno de acciones.
El conector ejecuta el pipeline oficial en otro proceso y usa su salida como
evidencia de identidades/dorsales, sin sustituir los IDs ni equipos del modelo
base. Conviene preparar este entorno en una VM Linux; la instalación de MMCV
en Windows puede requerir compilación. No se ha probado esa instalación aquí.

El asistente de instalación reproduce la configuración oficial en un entorno
independiente y checkout fijado:

```powershell
python setup_model_envs.py gamestate
$env:MATCHVIEW_GSR_PYTHON = "$PWD\.model-envs\gamestate\Scripts\python.exe"
$env:MATCHVIEW_GSR_ROOT = "$PWD\.model-envs\sn-gamestate"
python -m streamlit run match.py
```

En Linux, los ejecutables quedan en `.model-envs/gamestate/bin/python` y
`.model-envs/actions/bin/python`. Usa `export MATCHVIEW_GSR_PYTHON=...`,
`export MATCHVIEW_GSR_ROOT=...` y `export MATCHVIEW_OSL_PYTHON=...` en la misma
terminal desde la que arranques Streamlit. Los entornos deben estar en la
misma máquina que la aplicación; este conector no envía vídeos a otra VM.

El modo automático detecta este entorno y usa SoccerNet Game State si está preparado. Se ejecuta el pipeline
oficial con `dataset=video`, sin evaluación contra etiquetas y sin generar
otro vídeo de visualización. El proyecto descargará los checkpoints que le
falten. El estado generado se convierte a JSON dentro de ese mismo entorno;
la aplicación recibe JSON y no abre estados pickle ajenos.

Los resultados se asocian a personas del mismo fotograma por IoU ≥ 0,5. El
panel «Identidades SoccerNet» muestra ID externo, IDs originales, dorsal, rol
y lado propuestos. Izquierda/derecha no significa local/visitante. El balón
sigue procediendo de nuestros detectores. No se sustituyen automáticamente
los IDs base ni se promete resolver todos los cambios de identidad.

## 4. CLI, datos y pruebas

Ejemplo (sustituye la ruta por un vídeo existente; configura primero entornos):

```powershell
python video_track.py "C:\Videos\mi_video.mp4" --pitch-model hf32 --actions --gamestate --max-minutes 1
```

La CLI conserva opciones manuales para diagnóstico; añade `--auto-models` para
usar la misma política automática que la aplicación.
No se reentrena ningún modelo. La inferencia adicional puede tardar bastante
más que el detector base, especialmente en CPU.

En `<video>_clips/model_outputs/` quedan las predicciones sin normalizar,
su offset y duración, y los registros de cada proceso. El tracking JSON añade
`suggested_events`, `integrations`, `gamestate` y `samples[].gi` para identidades.
`events` y las estadísticas de eventos confirmados conservan su flujo anterior.

```powershell
python -m unittest discover -s tests -p test_model_integrations.py
```

Pruebas de topologías, descarga sin token, mapeo de coordenadas, tiempos
absolutos, límites, doce acciones, supresión temporal, asociación de identidades,
exportación, lanzamiento sin shell y propagación de opciones. No son una
evaluación de los checkpoints ni de la instalación de sus dependencias.

Fuentes y revisiones:

- Campo: https://huggingface.co/martinjolif/yolo-football-pitch-detection,
  revisión `bdb4687`, licencia declarada AGPL-3.0. Orden de puntos contrastado
  con `roboflow/sports`, `SoccerPitchConfiguration.vertices`.
- Acciones: https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e,
  revisión `54e24a6`, licencia declarada AGPL-3.0.
- OpenSportsLib: https://github.com/OpenSportsLab/opensportslib,
  revisión `b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68`.
- Game State: https://github.com/SoccerNet/sn-gamestate,
  revisión `1c958345067218297d221e45e1a6405f975f83e0`, licencia GPL-3.0.

## Inicio del análisis

«Iniciar análisis en» indica un tiempo del archivo original: `15` o `15:00`
empieza a los 900 segundos; `15:30` a los 930. «10 minutos desde el inicio
indicado» analiza [inicio, inicio + 600 s), limitado por la duración del vídeo.
«Hasta el final del vídeo» analiza desde ese punto hasta el final. No se
modifica el vídeo original. Se verifica el salto por fotogramas y, cuando el
lector no lo aplica, se reabre el vídeo y se omiten los fotogramas anteriores.
Las marcas de tiempo exportadas conservan su posición absoluta en el original.

El resultado incluye `model_status`, `analysis_start_s` y `analysis_end_s`.
Los componentes omitidos se muestran al finalizar y quedan registrados en el
tracking JSON. Un componente instalado pero cuyos pesos fallen no se anuncia
como activo. El modelo principal sigue siendo imprescindible.



## Buscar goles y acciones sin repetir el tracking

Preparación inicial (Python 3.12 o superior; entorno aislado):

```powershell
python setup_model_envs.py actions
```

Match View detecta `.model-envs/actions` automáticamente; ya no hace falta
exportar `MATCHVIEW_OSL_PYTHON` en cada terminal. La variable explícita sigue
teniendo prioridad. La instalación requiere acceso a GitHub y descarga de
paquetes; el primer uso descarga los pesos de Hugging Face.

En un análisis ya guardado, abre **Buscar jugadas IA** y elige el tramo en
segundos del vídeo original. Para comprobar una jugada alrededor de 23:00,
puedes empezar con 1360–1400 segundos (22:40–23:20), ampliando el intervalo si
el minuto indicado era aproximado o se refería al reloj del partido.
También puede ejecutarse desde consola:

```powershell
python action_review.py --tracking video_tracking.json --video "C:\ruta\video.mp4" --start 1360 --end 1400
```

Se ejecuta el modelo sobre las imágenes del fragmento; no se inventa un gol
por perder el balón ni se añaden etiquetas por el minuto indicado. Las
propuestas aparecen en Jugadas IA, con tiempo absoluto, para revisión.
El tracking, los eventos verificados y las estadísticas permanecen intactos.
Al repetir, se sustituyen las propuestas sin revisar de ese modelo dentro del
tramo; se conservan las de otros tramos y las revisadas guardadas en el JSON.
Si falla el motor o el formato de salida, no se modifica el archivo original.

Una prueba de normalización con predicciones simuladas verifica que un GOAL
20 segundos después de 22:40 aparece en 23:00, incluso sin balón detectado.
Esa prueba no demuestra que el modelo reconozca un gol en un vídeo real:
la validación requiere ejecutar los pesos sobre goles y jugadas sin gol.
