"""Mine difficult match frames for HUMAN correction and YOLO fine-tuning.

Select ball detection gaps, unstable field calibration and identity
occlusions. No auto-generated boxes are treated as ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def mine_hard_moments(samples, max_frames=200, min_spacing_s=3.0,
                      min_ball_gap_s=.65):
    samples = sorted(
        (s for s in samples if isinstance(s.get("t"), (int, float))),
        key=lambda s: s["t"],
    )
    candidates = []
    observed = [(i, float(s["t"])) for i, s in enumerate(samples)
                if s.get("bf") == "yolo" and s.get("b") is not None]
    for (i, before), (j, after) in zip(observed, observed[1:]):
        if after - before >= min_ball_gap_s and j > i + 1:
            midpoint = (before + after) / 2
            s = min(samples[i+1:j], key=lambda item: abs(item["t"] - midpoint))
            candidates.append({"t": float(s["t"]), "reason": "ball_missing",
                               "severity": round(after - before, 2)})
    for s in samples:
        people = s.get("p") or []
        t = float(s["t"])
        if s.get("rk") is False and len(people) >= 6:
            candidates.append({"t": t, "reason": "pitch_uncalibrated",
                               "severity": 1.0})
        if len(s.get("pe") or []) >= 3:
            candidates.append({"t": t, "reason": "player_occlusion",
                               "severity": len(s["pe"])})
        if len(people) >= 28:
            candidates.append({"t": t, "reason": "many_people_detected",
                               "severity": len(people) - 22})
    # Keep diverse difficult moments, not dozens of adjacent frames from
    # the same occlusion. Priority: missed ball, occlusion, field, crowd.
    order = {"ball_missing": 4, "player_occlusion": 3,
             "pitch_uncalibrated": 2, "many_people_detected": 1}
    chosen = []
    for item in sorted(candidates, key=lambda v:
                       (-order[v["reason"]], -v["severity"], v["t"])):
        if all(abs(item["t"] - prior["t"]) >= min_spacing_s for prior in chosen):
            chosen.append(item)
        if len(chosen) >= max_frames:
            break
    return sorted(chosen, key=lambda x: x["t"])


def extract_review_frames(video_path, moments, output_dir):
    """Extract only if the ORIGINAL video is actually accessible locally."""
    import cv2
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    paths = []
    try:
        for idx, item in enumerate(moments):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, item["t"])*1000)
            ok, frame = cap.read()
            if not ok:
                continue
            path = output_dir / f"review_{idx:04d}_{int(item['t']*1000):09d}.jpg"
            if cv2.imwrite(str(path), frame):
                paths.append(str(path))
    finally:
        cap.release()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", required=True, help="Existing Match_view tracking JSON")
    parser.add_argument("--video", help="Original video for frame extraction")
    parser.add_argument("--output", default="hard_examples")
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()
    tracking = json.loads(Path(args.tracking).read_text(encoding="utf-8"))
    samples = tracking.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Tracking JSON requires samples")
    moments = mine_hard_moments(samples, max_frames=args.max_frames)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if args.video:
        extracted = extract_review_frames(args.video, moments, folder / "images")
    else:
        extracted = []
    manifest = {
        "source_video": tracking.get("video_filename"),
        "candidate_count": len(moments),
        "image_paths": extracted,
        "moments": moments,
        "note": "Candidates need manual image labels; no auto-labels are trusted.",
    }
    target = folder / "review_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Review queue: {len(moments)} moments, {len(extracted)} frames -> {target}")


if __name__ == "__main__":
    main()
