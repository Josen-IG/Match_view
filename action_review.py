"""Run action spotting again without recomputing or altering tracking."""
from pathlib import Path
import argparse
import json
import math
import os
import tempfile

from model_jobs import normalize_actions, run_optional_model, validate_model_environment


def analyze_saved_actions(tracking_path, video_path, start_s=None, end_s=None,
                          threshold=.5, progress=None):
    path = Path(tracking_path).resolve()
    original = path.read_bytes()
    data = json.loads(original)
    video = Path(video_path).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if data.get("video_filename") != video.name:
        raise ValueError("El seguimiento no corresponde al vídeo seleccionado.")
    low = float(data.get("analysis_start_s", data.get("kickoff_s", 0)))
    high = data.get("analysis_end_s")
    high = float(high if high is not None else low + float(data["match_duration_s"]))
    start = low if start_s is None else float(start_s)
    end = high if end_s is None else float(end_s)
    if (not all(math.isfinite(v) for v in (low, high, start, end, threshold))
            or not low <= start < end <= high or not 0 < threshold <= 1):
        raise ValueError("El tramo debe estar dentro del intervalo analizado.")
    validate_model_environment("actions")
    payload = run_optional_model("actions", video, start, end-start,
                                 path.parent / f"{video.stem}_clips" / "model_outputs",
                                 progress)
    events = normalize_actions(payload, start, end-start, threshold)
    # Refresh only this model's unreviewed proposals in the selected interval.
    # Human reviews and predictions outside that interval remain intact.
    preserved = [ev for ev in data.get("suggested_events", [])
                 if ev.get("source") != "opensportslab_ball_action"
                 or ev.get("status") in ("confirmed", "rejected")
                 or not start <= float(ev["t"]) < end]
    reviewed_ids = {ev.get("id") for ev in preserved}
    data["suggested_events"] = sorted(preserved + [ev for ev in events
        if ev["id"] not in reviewed_ids], key=lambda ev: ev["t"])
    data.setdefault("model_status", {})["actions"] = {
        "state": "active", "detail": "Búsqueda de acciones completada."}
    data.setdefault("integrations", {})["actions"] = True
    data["integrations"]["actions_review"] = {
        "start_s": start, "end_s": end, "threshold": threshold,
        "proposals": len(events), "source": "opensportslab_ball_action"}
    if path.read_bytes() != original:
        raise RuntimeError("El análisis cambió durante la búsqueda. Vuelve a intentarlo.")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, help="Seconds in the original video")
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    events = analyze_saved_actions(args.tracking, args.video, args.start, args.end)
    print(json.dumps(events, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
