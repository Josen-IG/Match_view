"""Cola local de análisis de vídeo: un trabajo de CPU a la vez.

Mantiene progreso entre reruns de Streamlit, sin crear procesos de YOLO
duplicados. El estado es temporal y se pierde si se reinicia el servidor.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import uuid


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="match-view-video")
_LOCK = threading.RLock()
_JOBS = {}


def submit_video_job(analyzer, kwargs):
    """Lanza un análisis mediante una función Python, sin tocar Streamlit.

    El analyzer recibe los argumentos de kwargs más progress=callback.
    Devuelve un identificador aleatorio de trabajo.
    """
    job_id = uuid.uuid4().hex
    with _LOCK:
        for previous_id in list(_JOBS):
            job = _JOBS[previous_id]
            if job["state"] in ("completed", "failed") and len(_JOBS) >= 48:
                del _JOBS[previous_id]
        _JOBS[job_id] = {
            "state": "queued", "message": "En cola para procesar.",
            "progress": 0.0, "result": None, "error": None,
        }

    def _update(message, fraction):
        with _LOCK:
            job = _JOBS[job_id]
            job["message"] = str(message)
            job["progress"] = max(0.0, min(1.0, float(fraction)))

    def _work():
        with _LOCK:
            _JOBS[job_id]["state"] = "running"
            _JOBS[job_id]["message"] = "Iniciando análisis."
        try:
            result = analyzer(**kwargs, progress=_update)
        except Exception as error:
            with _LOCK:
                _JOBS[job_id].update({
                    "state": "failed", "error": str(error),
                    "message": "El análisis ha fallado.",
                })
        else:
            with _LOCK:
                _JOBS[job_id].update({
                    "state": "completed", "result": result,
                    "progress": 1.0, "message": "Análisis finalizado.",
                })

    _EXECUTOR.submit(_work)
    return job_id


def get_video_job(job_id):
    """Devuelve una instantánea del estado o None si no existe."""
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None
