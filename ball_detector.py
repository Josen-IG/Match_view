"""Optional specialist ball detector; no downloads unless explicitly enabled."""
from pathlib import Path
import math

BALL_REPO = "acatorcini/yolov9-soccer-ball"
BALL_REVISION = "b30df5a"  # pinned upstream revision


class SoccerBallDetector:
    def __init__(self, weights, confidence=0.2, device="cpu", imgsz=1280):
        if not 0 < confidence <= 1:
            raise ValueError("ball_conf must be in (0, 1].")
        from ultralytics import YOLO
        if weights == "hf":
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(BALL_REPO, "v9b_best.pt", revision=BALL_REVISION)
        elif not Path(weights).is_file():
            raise FileNotFoundError(f"Ball detector weights not found: {weights}")
        self.model = YOLO(str(weights))
        names = self.model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        self.class_ids = [int(i) for i, name in items
                          if str(name).lower() in ("ball", "sports ball", "soccer ball", "football")]
        if not self.class_ids:
            raise ValueError("Specialist checkpoint has no recognized ball class.")
        self.confidence, self.device, self.imgsz = confidence, device, imgsz

    def predict(self, frame, flip_x=False, flip_y=False):
        height, width = frame.shape[:2]
        results = self.model.predict(frame, classes=self.class_ids,
                                     conf=self.confidence, imgsz=self.imgsz,
                                     device=self.device, half=False, verbose=False)
        candidates = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box, confidence, label in zip(boxes.xyxy.cpu().tolist(),
                                               boxes.conf.cpu().tolist(),
                                               boxes.cls.cpu().tolist()):
                if int(label) not in self.class_ids:
                    continue
                if not all(math.isfinite(float(v)) for v in [*box, confidence]):
                    continue
                if not self.confidence <= confidence <= 1:
                    continue
                x0, y0, x1, y1 = box
                x0, x1 = max(0., min(width, x0)), max(0., min(width, x1))
                y0, y1 = max(0., min(height, y0)), max(0., min(height, y1))
                if x1 <= x0 or y1 <= y0:
                    continue
                x, y = (x0 + x1) / 2 / width * 100, (y0 + y1) / 2 / height * 100
                candidates.append({
                    "x": 100 - x if flip_x else x,
                    "y": 100 - y if flip_y else y,
                    "px": (x0 + x1) / 2, "py": y1,
                    "w": (x1 - x0) / width * 100,
                    "h": (y1 - y0) / height * 100,
                    "conf": float(confidence), "pitch_x": None, "pitch_y": None,
                    "detector": "yolov9_soccer_ball",
                })
        return candidates
