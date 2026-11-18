"""SoccerNet checkpoint adapter. Optional dependencies are imported on enable."""
from argparse import Namespace
from pathlib import Path
import math

RFDETR_REPO = "julianzu9612/RFDETR-Soccernet"
RFDETR_REVISION = "7e567611ea77efd3a6144b3a61eced5a8c8df8d7"
RFDETR_FILENAME = "checkpoint_best_regular.pth"
CLASS_NAMES = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}


def normalize_device(device):
    value = str(device).strip().lower()
    if value.isdigit():
        return f"cuda:{value}"
    if value in ("cpu", "mps", "cuda") or (
            value.startswith("cuda:") and value[5:].isdigit()):
        return value
    raise ValueError("RF-DETR device must be cpu, mps, cuda or a single GPU index.")


class SoccerNetDetector:
    def __init__(self, weights="hf", confidence=0.4, device="cpu"):
        if not 0 < confidence <= 1:
            raise ValueError("rfdetr_conf must be in (0, 1].")
        device = normalize_device(device)
        try:
            import torch
            from rfdetr import RFDETRBase
        except ImportError as error:
            raise RuntimeError(
                "Instala RF-DETR con: python -m pip install -r requirements-rfdetr.txt"
            ) from error
        if weights == "hf":
            root = Path(__file__).resolve().parent
            local = root / "weights" / RFDETR_FILENAME
            if not local.is_file():
                local = root / RFDETR_FILENAME
            if local.is_file():
                weights = str(local)
            else:
                from huggingface_hub import hf_hub_download
                weights = hf_hub_download(
                    RFDETR_REPO, f"weights/{RFDETR_FILENAME}",
                    revision=RFDETR_REVISION, token=False,
                )
        weights = Path(weights).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"No existe el checkpoint RF-DETR: {weights}")
        # The author's inference.py uses Base + a four-class head, despite
        # the model-card title. Never load mismatched weights with strict=False.
        # Namespace is the only extra global in this legacy training checkpoint.
        with torch.serialization.safe_globals([Namespace]):
            checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
        if not isinstance(state, dict) or not state:
            raise ValueError("RF-DETR checkpoint has no model state dictionary.")
        self.model = RFDETRBase(pretrain_weights=None, device=device)
        network = self.model.model.model
        network.reinitialize_detection_head(4)
        try:
            network.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "Checkpoint RF-DETR incompatible con el cargador SoccerNet. "
                "Usa el checkpoint publicado y requirements-rfdetr.txt. "
                "No se han aceptado pesos parciales."
            ) from error
        network.to(device)
        network.eval()
        self.confidence = confidence

    def predict(self, frame, flip_x=False, flip_y=False):
        import numpy as np
        # rfdetr.predict accepts RGB arrays; the pipeline supplies OpenCV BGR.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        detections = self.model.predict(rgb, threshold=self.confidence)
        if detections is None:
            return {"ball": [], "people": []}
        if detections.class_id is None or detections.confidence is None:
            raise ValueError("RF-DETR returned boxes without classes or confidence.")
        height, width = frame.shape[:2]
        result = {"ball": [], "people": []}
        for box, label, confidence in zip(
                detections.xyxy, detections.class_id, detections.confidence):
            if not all(math.isfinite(float(v)) for v in [*box, label, confidence]):
                continue
            name = CLASS_NAMES.get(int(label))
            if name is None or not self.confidence <= float(confidence) <= 1:
                continue
            x0, y0, x1, y1 = map(float, box)
            x0, x1 = max(0., min(width, x0)), max(0., min(width, x1))
            y0, y1 = max(0., min(height, y0)), max(0., min(height, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            cx, cy = (x0+x1)/2/width*100, (y0+y1)/2/height*100
            candidate = {
                "x": 100-cx if flip_x else cx, "y": 100-cy if flip_y else cy,
                "px": (x0+x1)/2, "py": y1, "w": (x1-x0)/width*100,
                "h": (y1-y0)/height*100, "conf": float(confidence),
                "pitch_x": None, "pitch_y": None,
                "detector": "rfdetr_soccernet", "class_name": name,
                "caja": (x0, y0, x1, y1),
            }
            result["ball" if name == "ball" else "people"].append(candidate)
        return result


def review_people(base_people, reviewed_people, min_iou=0.5):
    """Annotate matching primary people; no extra IDs or team overrides."""
    used = set()
    for person in base_people:
        a = person["caja"]
        best = None
        best_iou = min_iou
        for index, other in enumerate(reviewed_people):
            if index in used:
                continue
            b = other["caja"]
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(
                0, min(a[3], b[3])-max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
            iou = intersection / union if union > 0 else 0
            if iou >= best_iou:
                best, best_iou = index, iou
        if best is not None:
            used.add(best)
            person["rfdetr_review"] = {
                "class_name": reviewed_people[best]["class_name"],
                "confidence": reviewed_people[best]["conf"],
                "iou": best_iou,
            }
