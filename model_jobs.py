"""Run incompatible optional model environments out of process."""
from pathlib import Path
import json
import math
import os
import subprocess
import sys
import tempfile

ACTION_LABELS = {
    "PASS": ("pass", "Pase"), "DRIVE": ("drive", "Conducción"),
    "HEADER": ("header", "Cabeceo"), "HIGH PASS": ("high_pass", "Pase alto"),
    "OUT": ("out", "Balón fuera"), "CROSS": ("cross", "Centro"),
    "THROW IN": ("throw_in", "Saque de banda"), "SHOT": ("shot", "Tiro"),
    "BALL PLAYER BLOCK": ("block", "Bloqueo"),
    "PLAYER SUCCESSFUL TACKLE": ("tackle", "Entrada exitosa"),
    "FREE KICK": ("free_kick", "Tiro libre"), "GOAL": ("goal", "Gol"),
}


def normalize_actions(payload, offset, duration, threshold=0.5, nms_seconds=2.0):
    """OSL v2 milliseconds -> original-video seconds, with per-class temporal NMS."""
    if (not 0 < threshold <= 1 or not math.isfinite(offset) or offset < 0
            or not math.isfinite(duration) or duration <= 0
            or not math.isfinite(nms_seconds) or nms_seconds <= 0):
        raise ValueError("Invalid action interval or confidence threshold.")
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("OpenSportsLab must return its v2 JSON data list.")
    if len(payload["data"]) != 1:
        raise ValueError("Expected predictions for exactly one video segment.")
    candidates = []
    for event in payload["data"][0].get("events", []):
        try:
            label = str(event["label"]).upper()
            confidence = float(event["confidence"])
            seconds = float(event["position_ms"]) / 1000
        except (KeyError, ValueError, TypeError):
            continue
        if (label not in ACTION_LABELS or not math.isfinite(seconds)
                or not math.isfinite(confidence) or not threshold <= confidence <= 1
                or not 0 <= seconds < duration):
            continue
        kind, name = ACTION_LABELS[label]
        t = offset + seconds
        candidates.append({
            "type": kind, "label": "Posible " + name.lower(),
            "t": round(t, 3), "seconds": round(seconds, 3),
            "start_s": round(max(offset, t-4), 3),
            "end_s": round(min(offset+duration, t+6), 3),
            "confidence": confidence, "status": "suggested",
            "source": "opensportslab_ball_action", "model_label": label,
            "reason": f"OpenSportsLab propone {name.lower()} ({confidence:.0%}). Requiere revisión.",
        })
    selected, occupied = [], {}
    for ev in sorted(candidates, key=lambda e: (-e["confidence"], e["t"])):
        bins = occupied.setdefault(ev["type"], {})
        bucket = int(ev["t"] / nms_seconds)
        if any(abs(ev["t"]-t) <= nms_seconds
               for b in (bucket-1, bucket, bucket+1) for t in bins.get(b, [])):
            continue
        bins.setdefault(bucket, []).append(ev["t"])
        ev["id"] = f"osl:{ev['type']}:{round(ev['t']*1000)}"
        selected.append(ev)
    return sorted(selected, key=lambda e: e["t"])


def worker_python(kind):
    key = "MATCHVIEW_OSL_PYTHON" if kind == "actions" else "MATCHVIEW_GSR_PYTHON"
    value = os.environ.get(key)
    if value:
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{key} no apunta a un intérprete Python existente: {path}")
        return str(path.resolve())
    if kind == "actions" and sys.version_info >= (3, 12):
        return sys.executable
    raise RuntimeError(
        f"Configura {key} con el Python del entorno independiente. "
        "Consulta MODEL_INTEGRATIONS.md."
    )


def validate_model_environment(kind):
    python = worker_python(kind)
    if kind == "gamestate":
        root = os.environ.get("MATCHVIEW_GSR_ROOT", "")
        if not root or not (Path(root) / "sn_gamestate").is_dir():
            raise RuntimeError("Configura MATCHVIEW_GSR_ROOT con el checkout de sn-gamestate.")
    modules = ["opensportslib"] if kind == "actions" else ["tracklab", "sn_gamestate"]
    version = ">=(3,12)" if kind == "actions" else "==(3,9)"
    code = ("import sys,importlib.util; "
            f"assert sys.version_info[:2]{version}, 'Versión de Python incompatible'; "
            f"assert all(importlib.util.find_spec(m) for m in {modules!r}), "
            "'Faltan dependencias del modelo; consulta MODEL_INTEGRATIONS.md'")
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"El entorno de {kind} no está preparado: {result.stderr[-1500:]}")


def run_optional_model(kind, video_path, offset, duration, output_dir, progress=None):
    """Crop one exact range; all outputs refer to the original video via offset."""
    python = worker_python(kind)
    root = os.environ.get("MATCHVIEW_GSR_ROOT", "") if kind == "gamestate" else ""
    if kind == "gamestate" and not (Path(root) / "sn_gamestate").is_dir():
        raise RuntimeError("Configura MATCHVIEW_GSR_ROOT con el checkout de sn-gamestate.")
    if not math.isfinite(offset) or not math.isfinite(duration) or offset < 0 or duration <= 0:
        raise ValueError("Intervalo de vídeo inválido.")
    from video_track import _tool_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().with_name("model_worker.py")
    # Local files only; filenames never interpolated into a shell command.
    with tempfile.TemporaryDirectory(prefix=f"mv-{kind}-", dir=output_dir) as temp:
        temp = Path(temp)
        segment = temp / "segment.mp4"
        command = [
            _tool_path("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(offset), "-i", str(Path(video_path).resolve()), "-t", str(duration),
            "-map", "0:v:0", "-an", "-vf",
            "scale=398:224,fps=25" if kind == "actions" else "fps=25",
            "-c:v", "libx264", "-preset", "veryfast", str(segment),
        ]
        crop = subprocess.run(command, capture_output=True, text=True, timeout=7200,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if crop.returncode:
            raise RuntimeError("No se pudo preparar el fragmento: " + crop.stderr[-1500:])
        destination = temp / "predictions.json"
        log_path = output_dir / f"{kind}_worker.log"
        env = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN="1", WANDB_MODE="disabled")
        env.pop("HF_TOKEN", None)
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
        command = [python, str(script), kind, "--video", str(segment), "--output", str(destination)]
        if root:
            command.extend(["--root", str(Path(root).resolve())])
        if progress:
            progress(f"Ejecutando {kind}; registro: {log_path.name}.", 0.85)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, timeout=14400,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode or not destination.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"Falló {kind}. Registro: {log_path}\n{tail}")
        payload = json.loads(destination.read_text(encoding="utf-8"))
    # Preserve raw predictions for review, independently of the temporary clip.
    raw_path = output_dir / f"{kind}_predictions.json"
    raw_path.write_text(json.dumps({"video_filename": Path(video_path).name,
        "offset_s": offset, "duration_s": duration, "predictions": payload},
        ensure_ascii=False), encoding="utf-8")
    return payload


def attach_gamestate(samples, payload, offset, flip_x=False, flip_y=False):
    """Associate external identities to observed boxes; keep base IDs and teams."""
    if not isinstance(payload, dict) or payload.get("source") != "soccernet_gamestate":
        raise ValueError("Invalid SoccerNet Game State output.")
    fps = float(payload["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid SoccerNet frame rate.")
    frames = {}
    for row in payload.get("detections", []):
        frame = int(row["frame"])
        if frame < 0:
            continue
        frames.setdefault(frame, []).append(row)
    matched = 0
    for sample in samples:
        frame = int(round((sample["t_video"]-offset) * fps))
        rows = frames.get(frame, [])
        used = set()
        for person in sample.get("people", []):
            best, best_iou = None, .5
            ax, ay, aw, ah = (person[k] for k in ("x", "y", "w", "h"))
            for i, row in enumerate(rows):
                if i in used:
                    continue
                bx, by, bw, bh = row["box"]
                if not all(math.isfinite(float(v)) for v in (bx, by, bw, bh)) or bw <= 0 or bh <= 0:
                    continue
                bx, by = 100-bx if flip_x else bx, 100-by if flip_y else by
                intersection = max(0, min(ax+aw/2,bx+bw/2)-max(ax-aw/2,bx-bw/2)) * max(
                    0, min(ay+ah/2,by+bh/2)-max(ay-ah/2,by-bh/2))
                union = aw*ah + bw*bh - intersection
                iou = intersection/union if union > 0 else 0
                if iou >= best_iou:
                    best, best_iou = i, iou
            if best is not None:
                row = rows[best]
                used.add(best)
                person["gamestate"] = {k: row.get(k) for k in (
                    "external_track_id", "role", "jersey_number", "team")}
                person["gamestate"]["iou"] = round(best_iou, 3)
                matched += 1
    return {"source": "soccernet_gamestate", "matched_observations": matched,
            "external_detections": sum(map(len, frames.values()))}
