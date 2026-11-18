"""Create an accurate, cached playback segment without changing tracking."""
from pathlib import Path
import hashlib
import json
import math
import threading

_CLIP_LOCK = threading.Lock()


def prepare_playback_clip(video_path, start_s, end_s):
    from video_track import _run, _tool_path, video_duration
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    start, end = float(start_s), float(end_s)
    if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
        raise ValueError("El intervalo de reproducción no es válido.")
    duration = video_duration(source)
    end = min(end, duration)
    if end <= start:
        raise ValueError("El inicio está después del final del vídeo.")
    if start == 0 and end >= duration:
        return source, 0.0
    stat = source.stat()
    identity = [str(source), stat.st_size, stat.st_mtime_ns, start, end, 1]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    folder = source.parent / f"{source.stem}_clips" / "playback"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"segment_{key}.mp4"
    with _CLIP_LOCK:
        if not destination.is_file() or destination.stat().st_size == 0:
            temporary = destination.with_suffix(".partial.mp4")
            try:
                # Re-encoding is intentional: stream-copy cuts can start at an
                # earlier keyframe and shift the ball overlay by several seconds.
                _run([_tool_path("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                      "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{end-start:.6f}",
                      "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                      "-preset", "veryfast", "-crf", "20", "-c:a", "aac",
                      "-movflags", "+faststart", str(temporary)])
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError("FFmpeg no generó el fragmento de reproducción.")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination, start


def media_version(video_path):
    source = Path(video_path).resolve()
    stat = source.stat()
    return hashlib.sha256(f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:20]
