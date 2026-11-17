"""Audit and optional fine-tuning of Match_view's YOLO model.

Running this file WITHOUT --train audits the local dataset. This module
does not download or invent missing training images and never deploys
new model weights automatically.

Example:
  python training_pipeline.py --dataset dataset --audit
  python training_pipeline.py --dataset dataset --train --weights modelo_propio.pt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def audit_dataset(directory):
    """Return factual dataset availability/label consistency diagnostics."""
    root = Path(directory).resolve()
    result = {"dataset": str(root), "ready": False, "issues": [], "splits": {}}
    config = root / "data.yaml"
    if not config.is_file():
        result["issues"].append("data.yaml missing")
    all_ids = {}
    for split in ("train", "val"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images = {
            p.stem: p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        } if images_dir.is_dir() else {}
        labels = {
            p.stem: p for p in labels_dir.rglob("*.txt")
        } if labels_dir.is_dir() else {}
        classes = Counter()
        malformed = []
        for stem, path in labels.items():
            try:
                for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    parts = line.split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        malformed.append(f"{path.name}:{n}")
                        continue
                    category = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if category not in (0, 1) or not all(
                        0 <= value <= 1 for value in coords
                    ) or coords[2] <= 0 or coords[3] <= 0:
                        malformed.append(f"{path.name}:{n}")
                    else:
                        classes[category] += 1
            except (OSError, ValueError):
                malformed.append(path.name)
        missing_images = sorted(set(labels) - set(images))
        missing_labels = sorted(set(images) - set(labels))
        result["splits"][split] = {
            "images": len(images), "labels": len(labels),
            "objects_per_class": {str(k): v for k, v in classes.items()},
            "labels_without_images": len(missing_images),
            "images_without_labels": len(missing_labels),
            "invalid_label_lines": len(malformed),
            "examples": {
                "missing_image": missing_images[:5],
                "missing_label": missing_labels[:5],
                "invalid": malformed[:5],
            },
        }
        all_ids[split] = set(images)
        if not images:
            result["issues"].append(f"images/{split}: no images found")
        if missing_images or missing_labels or malformed:
            result["issues"].append(f"{split}: mismatched or invalid labels")
    overlapping = sorted(all_ids["train"] & all_ids["val"])
    result["duplicate_stems_across_splits"] = len(overlapping)
    if overlapping:
        result["issues"].append("same image stem appears in train and val")
    # A frame-level split can leak temporal context even without duplicate
    # image names. Use a separate match-level holdout for final evaluation.
    result["warnings"] = [
        "Validation frames from the same match are not independent; "
        "measure final mAP/recall on unseen MATCHES, not nearby frames.",
        "Auto-labels require manual review of missing/false balls and people.",
    ]
    result["ready"] = not result["issues"]
    return result


def train_model(dataset, weights, epochs=60, imgsz=960, batch=8, device="cpu"):
    audit = audit_dataset(dataset)
    if not audit["ready"]:
        raise ValueError("Dataset is not ready: " + "; ".join(audit["issues"]))
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights missing: {weights}")
    # This import happens ONLY when explicitly running training.
    from ultralytics import YOLO
    root = Path(dataset).resolve()
    model = YOLO(str(weights))
    return model.train(
        data=str(root / "data.yaml"), epochs=epochs, imgsz=imgsz,
        batch=batch, device=device, project=str(root / "training_runs"),
        name="candidate_no_autodeploy", exist_ok=False, seed=42,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--weights", default="modelo_propio.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    audit = audit_dataset(args.dataset)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.train:
        train_model(args.dataset, args.weights, args.epochs,
                    args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
