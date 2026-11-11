import base64
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


ProgressCallback = Callable[[str, float], None]


class VideoProcessingError(RuntimeError):
    pass


def _tool_path(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise VideoProcessingError(
            f"No se encuentra {name}. Instala FFmpeg y asegúrate de que esté en PATH."
        )
    return executable


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip()[-2000:] or "Error desconocido."
        tool_name = Path(command[0]).name
        raise VideoProcessingError(f"Falló {tool_name}: {detail}")
    return result.stdout.strip()


def video_duration(video_path: Path) -> float:
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"No existe el vídeo: {video_path}")

    output = _run(
        [
            _tool_path("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    try:
        duration = float(output)
    except ValueError as error:
        raise VideoProcessingError("FFprobe no pudo determinar la duración del vídeo.") from error
    if duration <= 0:
        raise VideoProcessingError("La duración del vídeo no es válida.")
    return duration


def export_clip(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_dir: Path | None = None,
    label: str = "accion",
) -> Path:
    video_path = video_path.expanduser().resolve()
    duration = video_duration(video_path)
    start = max(0.0, min(float(start_seconds), duration))
    end = max(start, min(float(end_seconds), duration))
    if end <= start:
        raise VideoProcessingError("El intervalo del corte no contiene vídeo.")

    output_dir = output_dir or video_path.parent / f"{video_path.stem}_clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_") or "accion"
    output_path = output_dir / f"{int(start):06d}_{safe_label}.mp4"

    _run(
        [
            _tool_path("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{end - start:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return output_path


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _classify_candidate(client, model: str, frames: list[tuple[float, Path]]) -> dict:
    from groq import APIError

    prompt = (
        "Clasifica con prudencia esta secuencia de fotogramas de un partido de fútbol. "
        "Los tiempos aparecen junto a cada imagen. No afirmes que hubo gol si no hay "
        "evidencia visual clara. Si no se distingue una acción concreta, usa uncertain. "
        "Devuelve solo JSON con las claves event, confidence y reason. "
        "event debe ser uno de goal, shot, corner, foul, pass, duel, transition, uncertain. "
        "confidence debe ser un número entre 0 y 1 y reason una frase breve en español."
    )
    content = [{"type": "text", "text": prompt}]
    for timestamp, image_path in frames:
        content.append(
            {"type": "text", "text": f"Fotograma en {timestamp:.1f} segundos."}
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_encode_image(image_path)}"
                },
            }
        )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_completion_tokens=256,
            response_format={"type": "json_object"},
        )
    except APIError as error:
        raise VideoProcessingError(f"Error de la API de Groq: {error}") from error

    response_text = completion.choices[0].message.content
    try:
        result = json.loads(response_text)
        event = str(result["event"]).lower()
        confidence = float(result["confidence"])
        reason = str(result["reason"])
    except (TypeError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise VideoProcessingError("Groq devolvió una clasificación con formato inválido.") from error

    valid_events = {
        "goal",
        "shot",
        "corner",
        "foul",
        "pass",
        "duel",
        "transition",
        "uncertain",
    }
    if event not in valid_events or not 0 <= confidence <= 1:
        raise VideoProcessingError("Groq devolvió un tipo de jugada o confianza inválidos.")
    return {"event": event, "confidence": confidence, "reason": reason}


def analyze_video(
    video_path: Path,
    api_key: str,
    sample_interval: int = 5,
    max_candidates: int = 12,
    pre_roll: int = 8,
    post_roll: int = 7,
    min_confidence: float = 0.65,
    model_name: str = "yolo11n.pt",
    groq_model: str = "qwen/qwen3.8-27b",
    progress: ProgressCallback | None = None,
) -> list[dict]:
    video_path = video_path.expanduser().resolve()
    if not 2 <= sample_interval <= 60:
        raise ValueError("El intervalo de muestreo debe estar entre 2 y 60 segundos.")
    if not 1 <= max_candidates <= 20:
        raise ValueError("El número de candidatos debe estar entre 1 y 20.")
    if not 0 <= min_confidence <= 1:
        raise ValueError("La confianza mínima debe estar entre 0 y 1.")
    duration = video_duration(video_path)
    try:
        from groq import Groq
        from ultralytics import YOLO
    except ImportError as error:
        missing = error.name or "una dependencia"
        raise VideoProcessingError(
            f"Falta {missing}. Instala las dependencias con: py -m pip install -r requirements.txt"
        ) from error

    update = progress or (lambda _message, _value: None)
    client = Groq(api_key=api_key)
    output_dir = video_path.parent / f"{video_path.stem}_clips"

    with tempfile.TemporaryDirectory(prefix="futbase_frames_") as temporary_dir:
        frame_dir = Path(temporary_dir)
        update("Extrayendo fotogramas del vídeo con FFmpeg.", 0.05)
        _run(
            [
                _tool_path("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps=1/{sample_interval},scale=960:-2",
                "-q:v",
                "4",
                str(frame_dir / "frame_%06d.jpg"),
            ]
        )
        frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
        if not frame_paths:
            raise VideoProcessingError("FFmpeg no pudo extraer fotogramas del vídeo.")

        try:
            detector = YOLO(model_name)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            raise VideoProcessingError(f"No se pudo cargar YOLO: {error}") from error

        detections = []
        previous_tracks = {}
        batch_size = 8
        for batch_start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[batch_start : batch_start + batch_size]
            try:
                results = detector.track(
                    source=[str(path) for path in batch_paths],
                    classes=[0, 32],
                    imgsz=640,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise VideoProcessingError(f"Falló el seguimiento de fotogramas con YOLO: {error}") from error

            for offset, result in enumerate(results):
                counts = {0: 0, 32: 0}
                movement = 0.0
                if result.boxes is not None and result.boxes.cls is not None:
                    classes = result.boxes.cls.cpu().tolist()
                    track_ids = (
                        result.boxes.id.cpu().tolist()
                        if result.boxes.id is not None
                        else [None] * len(classes)
                    )
                    centers = result.boxes.xywh[:, :2].cpu().tolist()
                    for class_id, track_id, center in zip(
                        classes, track_ids, centers, strict=True
                    ):
                        class_id = int(class_id)
                        if class_id in counts:
                            counts[class_id] += 1
                            if track_id is not None:
                                track_id = int(track_id)
                                if track_id in previous_tracks:
                                    old_x, old_y = previous_tracks[track_id]
                                    movement += (
                                        (float(center[0]) - old_x) ** 2
                                        + (float(center[1]) - old_y) ** 2
                                    ) ** 0.5
                                previous_tracks[track_id] = (
                                    float(center[0]),
                                    float(center[1]),
                                )
                detections.append(
                    {
                        "index": batch_start + offset,
                        "people": counts[0],
                        "ball": counts[32],
                        "movement": movement,
                    }
                )
            update(
                f"YOLO analizó {min(batch_start + batch_size, len(frame_paths))} "
                f"de {len(frame_paths)} fotogramas.",
                0.1 + 0.45 * min(batch_start + batch_size, len(frame_paths)) / len(frame_paths),
            )

        ranked = []
        for index, detection in enumerate(detections):
            previous = detections[index - 1] if index else detection
            score = (
                abs(detection["people"] - previous["people"])
                + 2 * abs(detection["ball"] - previous["ball"])
                + 3 * detection["ball"]
                + 0.2 * detection["people"]
                + detection["movement"] / 100
            )
            if detection["people"] or detection["ball"]:
                ranked.append((score, index))
        if not ranked:
            update(
                "YOLO no detectó personas ni balón en los fotogramas analizados.",
                1.0,
            )
            return []
        ranked.sort(reverse=True)

        selected = []
        min_gap = max(1, round(20 / sample_interval))
        for score, index in ranked:
            if all(abs(index - existing_index) >= min_gap for existing_index in selected):
                selected.append(index)
            if len(selected) >= max_candidates:
                break
        selected.sort()

        results = []
        for candidate_number, index in enumerate(selected, start=1):
            neighbor_indices = sorted(
                {max(0, index - 1), index, min(len(frame_paths) - 1, index + 1)}
            )
            frames = [
                (frame_index * sample_interval, frame_paths[frame_index])
                for frame_index in neighbor_indices
            ]
            timestamp = min(index * sample_interval, duration)
            update(
                f"Groq clasifica candidato {candidate_number} de {len(selected)}.",
                0.55 + 0.35 * candidate_number / max(1, len(selected)),
            )
            classification = _classify_candidate(client, groq_model, frames)
            result = {
                "seconds": timestamp,
                "yolo_people": detections[index]["people"],
                "yolo_ball": detections[index]["ball"],
                "yolo_movement": round(detections[index]["movement"], 1),
                **classification,
                "clip_path": None,
            }

            if (
                result["event"] != "uncertain"
                and result["confidence"] >= min_confidence
            ):
                clip_start = max(0, timestamp - pre_roll)
                clip_end = min(duration, timestamp + post_roll)
                label = f"{candidate_number:02d}_{result['event']}"
                result["clip_path"] = str(
                    export_clip(video_path, clip_start, clip_end, output_dir, label)
                )
            results.append(result)

    update("Análisis y generación de cortes completados.", 1.0)
    return results
