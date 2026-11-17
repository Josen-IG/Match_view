"""Versioned YOLO experiments. Never overwrites production weights.

Ground truth in this dataset is predominantly automatic. Its validation
scores measure agreement with those labels, not verified real-world accuracy.
Run prepare() before run(); artifacts belong in a separate experiment folder.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def match_id(stem):
    if stem.startswith(("f_", "f2_", "cirbonero_")):
        return "cirbonero_milagres"
    return stem.rsplit("_", 1)[0]


def clipped_label(line):
    """Validate a box and clip its edges, without guessing object semantics."""
    parts = line.split()
    if len(parts) != 5 or parts[0] not in ("0", "1"):
        raise ValueError("Invalid class or field count")
    x, y, w, h = map(float, parts[1:])
    if not all(math.isfinite(v) for v in (x, y, w, h)) or not (
        0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
    ):
        raise ValueError("Nonfinite or out-of-range box")
    left, top = max(0., x-w/2), max(0., y-h/2)
    right, bottom = min(1., x+w/2), min(1., y+h/2)
    changed = left != x-w/2 or right != x+w/2 or top != y-h/2 or bottom != y+h/2
    if changed:
        return f"{parts[0]} {(left+right)/2:.8f} {(top+bottom)/2:.8f} {right-left:.8f} {bottom-top:.8f}", True
    return line.strip(), False


def prepare(repo, image_root, workspace, output):
    """Build a matched copy; log every exclusion/correction and content hash."""
    import cv2
    import numpy as np
    import yaml
    repo, image_root, workspace, output = map(Path, (repo, image_root, workspace, output))
    output.mkdir(parents=True, exist_ok=True)
    data = workspace / "audited_dataset"
    assert not data.exists(), "Use a fresh workspace for reproducibility"
    report = {"source_commit": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "missing_images": [], "missing_labels": [], "corrections": [],
        "exact_duplicates": [], "near_duplicate_pairs": [], "manifest": [], "splits": {},
        "semantic_review": "Automatic labels; false/missing balls NOT exhaustively human verified.",
        "independent_match_holdout": False}
    seen, dhashes, bands = {}, [], defaultdict(list)
    for split in ("val", "train"):
        images = {p.stem: p for p in (image_root/split).rglob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")}
        labels = {p.stem: p for p in (repo/"dataset"/"labels"/split).glob("*.txt")}
        report["missing_images"] += [f"{split}/{s}" for s in sorted(labels.keys()-images.keys())]
        report["missing_labels"] += [f"{split}/{s}" for s in sorted(images.keys()-labels.keys())]
        for kind in ("images", "labels"):
            (data/kind/split).mkdir(parents=True)
        counts, matches, without_ball = Counter(), Counter(), []
        for stem in sorted(images.keys() & labels.keys()):
            source, label = images[stem], labels[stem]
            raw = source.read_bytes()
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable image: {source}")
            digest = hashlib.sha256(image.tobytes()+str(image.shape).encode()).hexdigest()
            if digest in seen:
                report["exact_duplicates"].append({"excluded": f"{split}/{stem}", "kept": seen[digest]})
                continue
            seen[digest] = f"{split}/{stem}"
            small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8))
            dhash = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
            candidates = set()
            for band in range(8):
                candidates.update(bands[(band, (dhash >> (8*band)) & 255)])
            for idx in candidates:
                old_hash, old_id = dhashes[idx]
                distance = (dhash ^ old_hash).bit_count()
                if distance <= 4:
                    report["near_duplicate_pairs"].append({"a": old_id, "b": f"{split}/{stem}", "hamming": distance})
            idx = len(dhashes)
            dhashes.append((dhash, f"{split}/{stem}"))
            for band in range(8):
                bands[(band, (dhash >> (8*band)) & 255)].append(idx)
            lines, ball = [], False
            for number, line in enumerate(label.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                fixed, changed = clipped_label(line)
                lines.append(fixed)
                category = int(fixed.split()[0])
                counts[category] += 1
                ball |= category == 0
                if changed:
                    report["corrections"].append({"file": f"{split}/{stem}.txt", "line": number,
                                                  "before": line, "after": fixed, "reason": "clip_to_image"})
            if not ball:
                without_ball.append(stem)
            dest = data/"images"/split/source.name
            shutil.copy2(source, dest)
            (data/"labels"/split/(stem+".txt")).write_text("\n".join(lines)+("\n" if lines else ""))
            matches[match_id(stem)] += 1
            report["manifest"].append({"split": split, "stem": stem, "match": match_id(stem),
                                       "sha256_pixels": digest, "sha256_file": hashlib.sha256(raw).hexdigest(),
                                       "sha256_source_labels": hashlib.sha256(label.read_bytes()).hexdigest(),
                                       "shape": list(image.shape), "has_ball": ball})
        report["splits"][split] = {"source_images": len(images), "source_labels": len(labels),
                                   "kept_images": sum(matches.values()), "objects": dict(counts),
                                   "matches": dict(matches), "frames_without_ball_label": without_ball}
    report["shared_matches"] = sorted(set(report["splits"]["train"]["matches"]) &
                                       set(report["splits"]["val"]["matches"]))
    config = {"path": str(data.resolve()), "train": "images/train", "val": "images/val",
              "nc": 2, "names": {0: "football", 1: "player"}}
    (data/"data.yaml").write_text(yaml.safe_dump(config))
    train_paths = []
    for row in report["manifest"]:
        if row["split"] == "train":
            path = next((data/"images"/"train").glob(row["stem"]+".*"))
            train_paths += [str(path.resolve())] * (2 if row["has_ball"] else 1)
    (data/"train_oversampled.txt").write_text("\n".join(train_paths)+"\n")
    config["train"] = str((data/"train_oversampled.txt").resolve())
    (data/"train.yaml").write_text(yaml.safe_dump(config))
    report["oversampled_train_entries"] = len(train_paths)
    write_json(output/"audit.json", report)
    shutil.copy2(data/"data.yaml", output/"evaluation_data.yaml")
    shutil.copy2(data/"train.yaml", output/"training_data.yaml")
    return data, report


def validate(weights, data, output, name):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    result = model.val(data=str(Path(data)/"data.yaml"), imgsz=960, batch=8,
                       device=0, conf=.001, iou=.7, seed=42, workers=2,
                       project=str(output), name=name+"_validation", plots=True)
    rows = []
    for index, category in enumerate(result.box.ap_class_index):
        rows.append({"class_id": int(category), "class_name": model.names[int(category)],
                     "precision_best_f1": float(result.box.p[index]),
                     "recall_best_f1": float(result.box.r[index]),
                     "mAP50": float(result.box.ap50[index]),
                     "mAP50_95": float(result.box.ap[index])})
    write_json(Path(output)/(name+"_metrics.json"), rows)
    return rows


def box_iou(a, b):
    area = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area/union if union > 0 else 0.


def fixed_threshold_counts(weights, data, output, name):
    """IoU .5, conf .25: actual FP counts, including frames with no ball label."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    paths = sorted((Path(data)/"images"/"val").glob("*"))
    counts = {k: Counter(tp=0, fp=0, fn=0, tiny_tp=0, tiny_total=0) for k in (0, 1)}
    results = model.predict([str(p) for p in paths], stream=True, device=0,
                            imgsz=960, conf=.25, iou=.7, verbose=False, batch=8)
    for path, result in zip(paths, results):
        height, width = result.orig_shape
        truth = defaultdict(list)
        for line in (Path(data)/"labels"/"val"/(path.stem+".txt")).read_text().splitlines():
            cat, x, y, w, h = map(float, line.split())
            truth[int(cat)].append(((width*(x-w/2), height*(y-h/2), width*(x+w/2), height*(y+h/2)),
                                    max(width*w, height*h)*960/max(width, height) <= 10))
        boxes = result.boxes
        predictions = sorted(zip(boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.xyxy.cpu().tolist()), reverse=True)
        used = {0: set(), 1: set()}
        for _, cat, box in predictions:
            cat = int(cat)
            available = [(box_iou(box, gt), idx) for idx, (gt, _) in enumerate(truth[cat]) if idx not in used[cat]]
            overlap, idx = max(available, default=(0., -1))
            if overlap >= .5:
                used[cat].add(idx)
                counts[cat]["tp"] += 1
                counts[cat]["tiny_tp"] += int(truth[cat][idx][1])
            else:
                counts[cat]["fp"] += 1
        for cat in (0, 1):
            counts[cat]["fn"] += len(truth[cat])-len(used[cat])
            counts[cat]["tiny_total"] += sum(tiny for _, tiny in truth[cat])
    report = {"conf": .25, "match_iou": .5, "nms_iou": .7, "frames": len(paths),
              "tiny_definition": "maximum box side <=10 pixels after longest-side resize to 960", "classes": {}}
    for cat, c in counts.items():
        report["classes"][str(cat)] = dict(c, precision=c["tp"]/max(1,c["tp"]+c["fp"]),
                                          recall=c["tp"]/max(1,c["tp"]+c["fn"]),
                                          tiny_recall=c["tiny_tp"]/c["tiny_total"] if c["tiny_total"] else None)
    write_json(Path(output)/(name+"_fixed_threshold.json"), report)
    return report


def cpu_benchmark(weights, data, output, name, samples=40):
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    model = YOLO(str(weights))
    all_paths = sorted((Path(data)/"images"/"val").glob("*"))
    samples = min(samples, len(all_paths))
    paths = [all_paths[i*(len(all_paths)-1)//max(1,samples-1)] for i in range(samples)]
    images = [cv2.imread(str(p)) for p in paths]
    for _ in range(3):
        model.predict(images[0], device="cpu", imgsz=960, conf=.25, iou=.7, verbose=False)
    elapsed = []
    for image in images:
        start = time.perf_counter()
        model.predict(image, device="cpu", imgsz=960, conf=.25, iou=.7, verbose=False)
        elapsed.append(1000*(time.perf_counter()-start))
    report = {"samples": len(images), "image_files": [p.name for p in paths],
              "median_ms_end_to_end": statistics.median(elapsed), "mean_ms_end_to_end": statistics.mean(elapsed),
              "threads": 2, "imgsz": 960, "hardware": platform.platform(),
              "note": "Colab CPU; not the user's Windows CPU. Decode excluded, preprocessing and NMS included."}
    write_json(Path(output)/(name+"_cpu.json"), report)
    return report


def run(repo, data, output, epochs=60):
    """Sequential n/s experiment with epoch checkpoints copied to Drive."""
    import torch
    import ultralytics
    from ultralytics import YOLO
    repo, data, output = map(Path, (repo, data, output))
    assert torch.cuda.is_available(), "A GPU is required for this experiment"
    write_json(output/"environment.json", {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
               "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(), "seed": 42})
    subprocess.run(["python", "-m", "pip", "freeze"], stdout=(output/"requirements_frozen.txt").open("w"), check=True)
    shutil.copy2(repo/"modelo_propio.pt", output/"baseline_original.pt")
    validate(repo/"modelo_propio.pt", data, output, "baseline")
    fixed_threshold_counts(repo/"modelo_propio.pt", data, output, "baseline")
    cpu_benchmark(repo/"modelo_propio.pt", data, output, "baseline")
    for name, weights in (("yolo11n_finetune", repo/"modelo_propio.pt"),
                          ("yolo11s_pretrained", repo/"yolo11s.pt")):
        model = YOLO(str(weights))
        def sync(trainer):
            shutil.copytree(trainer.save_dir, output/name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("epoch*.pt"))
            if Path(trainer.last).exists():
                shutil.copy2(trainer.last, output/name/"weights"/"resumable.pt")
        model.add_callback("on_model_save", sync)
        write_json(output/"status.json", {"phase": "training", "candidate": name, "completed": False})
        model.train(data=str(data/"train.yaml"), epochs=epochs, patience=12, imgsz=960,
                    batch=8, device=0, workers=2, seed=42, deterministic=True,
                    project=str(data.parent/"runs"), name=name, exist_ok=False,
                    optimizer="AdamW", lr0=.001, lrf=.01, weight_decay=.0005,
                    warmup_epochs=3, close_mosaic=10, mosaic=.5, mixup=0.,
                    degrees=2., translate=.05, scale=.3, shear=0., perspective=0.,
                    hsv_h=.015, hsv_s=.5, hsv_v=.3, flipud=0., fliplr=.5,
                    amp=True, save=True, save_period=-1, plots=True)
        shutil.copytree(model.trainer.save_dir, output/name, dirs_exist_ok=True)
        best = output/name/"weights"/"best.pt"
        validate(best, data, output, name)
        fixed_threshold_counts(best, data, output, name)
        cpu_benchmark(best, data, output, name)
    write_json(output/"status.json", {"phase": "experiments_complete", "completed": True,
               "deployed": False, "requires": "independent human-labelled holdout before deployment"})
