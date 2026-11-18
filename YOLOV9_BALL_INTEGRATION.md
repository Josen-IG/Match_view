# YOLOv9 ball fallback (stage 1)

The existing detector and weights remain the primary detector. Enable the
optional specialist in the Streamlit analysis form, or use the API/CLI below.
It runs only on sampled frames with no base ball candidates. It does not
replace an existing base candidate, even when that candidate is incorrect.
This conservative policy avoids comparing confidence scores across models.

## VM quick test

Install the project's requirements in your existing analysis environment:

```bash
python -m pip install -r requirements.txt
python video_track.py partido.mp4 --yolo-model /path/to/best.pt --ball-model hf --ball-conf 0.2 --device 0 --kickoff 06:00 --max-minutes 1 --sample-fps 5
```

`hf` downloads `v9b_best.pt` from `acatorcini/yolov9-soccer-ball`, pinned to
revision `b30df5a`. Alternatively pass a local weights path. Use `--device cpu`
without a CUDA GPU. The existing CLI defaults to CPU.

```python
from video_track import analyze_video_full
result = analyze_video_full(
    'partido.mp4', yolo_model='/path/to/best.pt',
    ball_model='hf', ball_conf=0.2, device='0', max_minutes=1,
    debug_muestras_path='specialist_samples.pkl',
)
```

Omit `ball_model` / `--ball-model` to keep the original behavior. The
specialist's errors are surfaced explicitly rather than silently disabling it.
In raw samples, `ball_detector` identifies `base` or `yolov9_soccer_ball`.
Tracking JSON exports this as optional `bd`. Existing `bf` continues to
distinguish observed detections from temporal predictions; `bd` records the
detector at the original sampled frame, not the origin of an interpolated ball.

## Validation

Five offline tests cover specialist boxes/coordinates, invalid predictions,
class validation, disabled behavior, preserving base ball detections and
recovering a missing ball without changing the detected people.

```bash
python -m unittest discover -s tests -p test_ball_detector.py
```

These use simulated detector outputs: actual checkpoint loading, GPU memory,
latency and detection quality still need validation on the VM. Existing video
pipeline tests were attempted locally; two encounter a broken SciPy installation
(`scipy.optimize._highspy._core.simplex_constants` missing).

Compare the same short clip with and without the specialist. Review labeled
frames for missed balls and false detections, and measure runtime. Increased
coverage alone is not proof of increased accuracy. Keep the other planned
models disabled until this stage is evaluated.

Upstream: https://huggingface.co/acatorcini/yolov9-soccer-ball
Weights license stated by the author: AGPL-3.0.

## Public download authentication

Automatic loading first uses v9b_best.pt next to ball_detector.py if present.
Otherwise it downloads the public checkpoint with token=False, ignoring saved
or environment credentials (including expired tokens). No login is required.
Two additional offline tests cover anonymous download and local-file reuse.
